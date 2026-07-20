#!/usr/bin/env python3
"""A/B benchmark full vs row-chunked Krea2 FP8 LoRA merge on one GPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import torch
from safetensors import safe_open


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "runtime"
for entry in (REPO, RUNTIME):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from training.families.krea2.loader import load_krea2_model  # noqa: E402
from training.families.krea2.lora_fp8_merge import (  # noqa: E402
    _apply_lora_delta,
    _group_lora_layers,
    _requantize_scaled,
    merge_loras_into_fp8_model,
    stochastic_round_to_fp8,
    string_to_seed,
)


MIB = 1024 ** 2


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", action="append", required=True)
    parser.add_argument("--scale", action="append", type=float)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--mode", choices=("compare", "full", "chunked", "stream-compare"),
        default="compare",
    )
    return parser.parse_args()


def _load_lora(path: str) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def _tensor_digest(tensor: torch.Tensor) -> bytes:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).digest()


def _layer_digests(model: torch.nn.Module, layers: list[str]) -> dict[str, str]:
    modules = dict(model.named_modules())
    result: dict[str, str] = {}
    for name in layers:
        module = modules[name]
        digest = hashlib.sha256()
        digest.update(_tensor_digest(module.weight))
        scale = getattr(module, "weight_scale", None)
        if scale is not None:
            digest.update(_tensor_digest(scale))
        result[name] = digest.hexdigest()
    return result


class _NvmlPeak:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self.start = 0
        self.peak = 0
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None

    def __enter__(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            self.start = int(pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)
            self.peak = self.start

            def poll() -> None:
                while not self._stop.wait(0.01):
                    used = int(pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)
                    self.peak = max(self.peak, used)

            self._thread = threading.Thread(target=poll, daemon=True)
            self._thread.start()
        except Exception:
            self._nvml = None
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._nvml is not None:
            self.peak = max(
                self.peak,
                int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used),
            )
            self._nvml.nvmlShutdown()


def _run(model, sources, compute_dtype, chunk_rows):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with _NvmlPeak() as nvml:
        started = time.perf_counter()
        adapter = merge_loras_into_fp8_model(
            model,
            sources,
            compute_dtype=compute_dtype,
            chunk_rows=chunk_rows,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    metrics = {
        "seconds": elapsed,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
        "device_start_mib": nvml.start / MIB if nvml.start else None,
        "device_peak_mib": nvml.peak / MIB if nvml.peak else None,
        "device_net_peak_mib": (nvml.peak - nvml.start) / MIB if nvml.start else None,
    }
    return adapter, metrics


def _finish_layer(weight, scale, w16, name):
    seed = string_to_seed(f"diffusion_model.{name}.weight")
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if scale is not None:
            return _requantize_scaled(w16, weight.dtype, seed)
        return stochastic_round_to_fp8(w16, weight.dtype, seed), None
    return w16.to(weight.dtype), None


def _initial_w16(weight, scale):
    value = weight.to(torch.float16)
    return value if scale is None else value * scale.to(torch.float16)


def _stream_compare(args, sources, compute_dtype) -> None:
    """Compare every merged layer while keeping only one base weight on GPU."""
    grouped_sources = [
        (_group_lora_layers(sd), strength, source)
        for sd, strength, source in sources
    ]
    per_layer = {}
    for grouped, strength, source in grouped_sources:
        for layer_key, tensors in grouped.items():
            per_layer.setdefault(layer_key, []).append((tensors, strength, source))

    full_seconds = 0.0
    chunk_seconds = 0.0
    mismatched = []
    changed_elements = 0
    total_elements = 0
    max_abs_dequant = 0.0
    sum_abs_dequant = 0.0
    full_peak = 0
    chunk_peak = 0

    with safe_open(args.model, framework="pt", device="cpu") as base:
        weight_names = [key[:-7] for key in base.keys() if key.endswith(".weight")]
        name_by_lora_key = {name.replace(".", "_"): name for name in weight_names}
        for layer_key, layer_sources in per_layer.items():
            name = name_by_lora_key[layer_key]
            weight = base.get_tensor(f"{name}.weight").to("cuda")
            scale_key = f"{name}.weight_scale"
            scale = base.get_tensor(scale_key).to("cuda") if scale_key in base.keys() else None

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            full_w16 = _initial_w16(weight, scale)
            for tensors, strength, source in layer_sources:
                full_w16 = _apply_lora_delta(
                    full_w16, tensors, strength, name, source, compute_dtype, None,
                )
            full_value, full_scale = _finish_layer(weight, scale, full_w16, name)
            torch.cuda.synchronize()
            full_seconds += time.perf_counter() - started
            full_peak = max(full_peak, torch.cuda.max_memory_allocated())
            full_cpu = full_value.cpu()
            full_scale_cpu = None if full_scale is None else full_scale.cpu()
            del full_value, full_scale, full_w16

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            chunk_w16 = _initial_w16(weight, scale)
            for tensors, strength, source in layer_sources:
                chunk_w16 = _apply_lora_delta(
                    chunk_w16, tensors, strength, name, source,
                    compute_dtype, args.chunk_rows,
                )
            chunk_value, chunk_scale = _finish_layer(weight, scale, chunk_w16, name)
            torch.cuda.synchronize()
            chunk_seconds += time.perf_counter() - started
            chunk_peak = max(chunk_peak, torch.cuda.max_memory_allocated())

            full_bits = full_cpu.contiguous().view(torch.uint8)
            chunk_bits = chunk_value.cpu().contiguous().view(torch.uint8)
            layer_changed = int(torch.count_nonzero(full_bits != chunk_bits))
            scale_equal = (
                full_scale_cpu is None and chunk_scale is None
            ) or (
                full_scale_cpu is not None
                and chunk_scale is not None
                and torch.equal(full_scale_cpu, chunk_scale.cpu())
            )
            total_elements += int(full_bits.numel())
            changed_elements += layer_changed
            if layer_changed or not scale_equal:
                mismatched.append(name)
                full_dequant = full_cpu.float()
                chunk_dequant = chunk_value.cpu().float()
                if full_scale_cpu is not None:
                    full_dequant *= full_scale_cpu.float()
                    chunk_dequant *= chunk_scale.cpu().float()
                error = (full_dequant - chunk_dequant).abs()
                max_abs_dequant = max(max_abs_dequant, float(error.max()))
                sum_abs_dequant += float(error.sum())

            del weight, scale, full_cpu, full_scale_cpu
            del chunk_value, chunk_scale, chunk_w16
            torch.cuda.empty_cache()

    print(json.dumps({
        "precision": args.precision,
        "mode": "stream-compare",
        "chunk_rows": args.chunk_rows,
        "lora_count": len(sources),
        "merged_layers": len(per_layer),
        "bit_identical_layers": len(per_layer) - len(mismatched),
        "mismatched_layers": mismatched,
        "changed_storage_bytes": changed_elements,
        "total_storage_bytes": total_elements,
        "changed_storage_ratio": changed_elements / max(total_elements, 1),
        "max_abs_dequant_error": max_abs_dequant,
        "mean_abs_dequant_error": sum_abs_dequant / max(total_elements, 1),
        "full_seconds": full_seconds,
        "chunked_seconds": chunk_seconds,
        "full_peak_allocated_mib": full_peak / MIB,
        "chunked_peak_allocated_mib": chunk_peak / MIB,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    scales = args.scale or [1.0] * len(args.lora)
    if len(scales) != len(args.lora):
        raise ValueError("--scale count must match --lora count")
    compute_dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16

    sources = [
        (_load_lora(path), scale, Path(path).name)
        for path, scale in zip(args.lora, scales)
    ]
    if args.mode == "stream-compare":
        _stream_compare(args, sources, compute_dtype)
        return

    model = load_krea2_model(
        args.model, "cuda", torch.bfloat16, purpose="generate",
    )
    gc.collect()
    torch.cuda.empty_cache()

    if args.mode == "full":
        full, full_metrics = _run(model, sources, compute_dtype, None)
        print(json.dumps({
            "precision": args.precision,
            "mode": "full",
            "lora_count": len(sources),
            "merged_layers": len(full._backup),
            "full": full_metrics,
        }, ensure_ascii=False, indent=2))
        return

    if args.mode == "chunked":
        chunked, chunked_metrics = _run(
            model, sources, compute_dtype, args.chunk_rows,
        )
        print(json.dumps({
            "precision": args.precision,
            "mode": "chunked",
            "chunk_rows": args.chunk_rows,
            "lora_count": len(sources),
            "merged_layers": len(chunked._backup),
            "chunked": chunked_metrics,
        }, ensure_ascii=False, indent=2))
        return

    full, full_metrics = _run(model, sources, compute_dtype, None)
    layers = list(full._backup)
    full_digests = _layer_digests(model, layers)
    full.detach()
    del full
    torch.cuda.empty_cache()

    chunked, chunked_metrics = _run(
        model, sources, compute_dtype, args.chunk_rows,
    )
    chunked_digests = _layer_digests(model, layers)
    mismatched = [
        layer for layer in layers if full_digests[layer] != chunked_digests[layer]
    ]

    print(json.dumps({
        "precision": args.precision,
        "chunk_rows": args.chunk_rows,
        "lora_count": len(sources),
        "merged_layers": len(layers),
        "bit_identical_layers": len(layers) - len(mismatched),
        "mismatched_layers": mismatched,
        "full": full_metrics,
        "chunked": chunked_metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
