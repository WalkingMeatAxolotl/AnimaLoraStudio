#!/usr/bin/env python
"""Validate a Windows ROCm environment and optional Anima training assets."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.accelerator import detect_accelerator  # noqa: E402


CORE_MODULES = {
    "PIL": "Pillow",
    "einops": "einops",
    "lycoris": "lycoris-lora",
    "omegaconf": "omegaconf",
    "safetensors": "safetensors",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "yaml": "PyYAML",
}
STUDIO_MODULES = {
    "fastapi": "fastapi",
    "multipart": "python-multipart",
    "uvicorn": "uvicorn",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def _fail(message: str, errors: list[str]) -> None:
    print(f"[FAIL] {message}")
    errors.append(message)


def _check_modules(modules: dict[str, str], errors: list[str]) -> None:
    missing = [pip_name for module, pip_name in modules.items() if importlib.util.find_spec(module) is None]
    if missing:
        _fail("missing Python packages: " + ", ".join(sorted(missing)), errors)
    else:
        _ok(f"Python packages ({len(modules)})")


def _check_safetensors(path: str, kind: str, errors: list[str]) -> None:
    if not path:
        return
    p = Path(path).expanduser()
    if not p.is_file():
        _fail(f"{kind} not found: {p}", errors)
        return
    try:
        from safetensors import safe_open

        with safe_open(p, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        expected = {
            "transformer": lambda key: key.endswith("x_embedder.proj.1.weight"),
            "vae": lambda key: key.endswith("conv1.weight"),
            "text_encoder": lambda key: key.endswith("model.embed_tokens.weight"),
        }[kind]
        if not any(expected(key) for key in keys):
            _fail(f"{kind} checkpoint structure is not recognized: {p}", errors)
            return
        _ok(f"{kind}: {p} ({len(keys)} tensors, {p.stat().st_size / 1024**3:.2f} GiB)")
    except Exception as exc:  # noqa: BLE001
        _fail(f"cannot inspect {kind} {p}: {type(exc).__name__}: {exc}", errors)


def _check_dataset(path: str, errors: list[str]) -> None:
    if not path:
        return
    root = Path(path).expanduser()
    if not root.is_dir():
        _fail(f"dataset not found: {root}", errors)
        return
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    missing = [p for p in images if not p.with_suffix(".txt").is_file()]
    if not images:
        _fail(f"dataset has no supported images: {root}", errors)
    elif missing:
        _fail(f"dataset has {len(missing)}/{len(images)} images without .txt captions", errors)
    else:
        _ok(f"dataset: {root} ({len(images)} image/caption pairs)")


def _quick_gpu_test(errors: list[str]) -> None:
    try:
        import torch
        import torch.nn.functional as F

        q = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out = F.scaled_dot_product_attention(q, k, v)
        out.float().square().mean().backward()
        torch.cuda.synchronize()
        if q.grad is None or not torch.isfinite(out).all():
            raise RuntimeError("SDPA output/gradient check failed")
        del q, k, v, out
        torch.cuda.empty_cache()
        _ok("bf16 SDPA forward/backward on ROCm")
    except Exception as exc:  # noqa: BLE001
        _fail(f"ROCm bf16 SDPA test failed: {type(exc).__name__}: {exc}", errors)


def _load_config(path: str, errors: list[str]) -> dict:
    if not path:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        _fail(f"config not found: {config_path}", errors)
        return {}
    try:
        import yaml
        from studio.schema import TrainingConfig

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config = TrainingConfig.model_validate(raw).model_dump(mode="json")
        if config["attention_backend"] != "none":
            _fail("ROCm config must set attention_backend: none", errors)
        else:
            _ok(f"training config: {config_path}")
        return config
    except Exception as exc:  # noqa: BLE001
        _fail(f"invalid training config: {type(exc).__name__}: {exc}", errors)
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="")
    parser.add_argument("--transformer", default="")
    parser.add_argument("--vae", default="")
    parser.add_argument("--text-encoder", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--studio", action="store_true", help="also require Studio backend packages")
    parser.add_argument("--skip-gpu-test", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    print(f"Python: {sys.executable}")
    _check_modules(CORE_MODULES, errors)
    if args.studio:
        _check_modules(STUDIO_MODULES, errors)

    try:
        import torch
        info = detect_accelerator(torch)
        print(f"PyTorch: {torch.__version__}")
        if info.backend != "rocm" or not info.available:
            _fail(
                f"expected an available ROCm device, got backend={info.backend}, available={info.available}",
                errors,
            )
        else:
            _ok(f"{info.build}: {info.device_name} ({info.device_count} torch device(s))")
            if not args.skip_gpu_test:
                _quick_gpu_test(errors)
    except ImportError:
        _fail("torch is not installed", errors)

    config = _load_config(args.config, errors)
    transformer = args.transformer or config.get("transformer_path", "")
    vae = args.vae or config.get("vae_path", "")
    text_encoder = args.text_encoder or config.get("text_encoder_path", "")
    dataset = args.dataset or config.get("data_dir", "")
    _check_safetensors(transformer, "transformer", errors)
    _check_safetensors(vae, "vae", errors)
    _check_safetensors(text_encoder, "text_encoder", errors)
    _check_dataset(dataset, errors)

    if errors:
        print(f"\nROCm preflight failed with {len(errors)} error(s).")
        return 1
    print("\nROCm preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

