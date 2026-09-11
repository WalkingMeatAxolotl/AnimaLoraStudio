#!/usr/bin/env python
"""Reproducible LyCORIS 4.0.0 torch-eager layer/replay/Anima training CLI.

Every measurement repeat is a fresh process. No assets are read by layer/replay.
Public JSON is allowlisted; raw child logs and path bindings remain private.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "runtime"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Make the module identity stable for helper imports when invoked as a script.
if __name__ == "__main__":
    sys.modules["tools.benchmark_lycoris"] = sys.modules[__name__]

CONFIG_KEYS = {"schema_version", "profile", "dtype", "algorithms", "rank", "alpha", "factor",
               "seed", "repeats", "warmup", "measured", "resolution", "text_tokens", "train"}
TRAIN_KEYS = {"resolution", "batch_size", "grad_accum", "mixed_precision", "attention_backend",
              "grad_checkpoint", "cache_latents", "images", "epochs", "learning_rate",
              "optimizer_type", "lr_scheduler", "blocks_to_swap", "sample_steps", "sample_every",
              "save_every_epochs", "save_every_steps", "save_state_every_epochs", "save_state_every_steps"}


def read_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("nonfinite JSON number")

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def validate_config(config: dict) -> dict:
    if set(config) != CONFIG_KEYS or type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("unsupported experiment config")
    if config["profile"] not in {"cpu-smoke", "anima-2048", "anima-5120"}:
        raise ValueError("unknown profile")
    if config["dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("unsupported dtype")
    algorithms = config["algorithms"]
    if not isinstance(algorithms, list) or not algorithms or any(a not in {"lora", "lokr", "loha"} for a in algorithms):
        raise ValueError("unknown or empty algorithms")
    if len(set(algorithms)) != len(algorithms):
        raise ValueError("duplicate algorithms")
    for key, low, high in [("rank", 1, 64), ("factor", 2, 16), ("seed", 0, 2**32 - 1),
                           ("repeats", 1, 10), ("warmup", 1, 20), ("measured", 1, 100),
                           ("resolution", 256, 1024), ("text_tokens", 1, 512)]:
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError("invalid bounded experiment value")
    if config["resolution"] % 64 or type(config["alpha"]) not in {int, float} or not 0 < config["alpha"] <= 64:
        raise ValueError("invalid alpha or resolution")
    train = config["train"]
    if not isinstance(train, dict) or set(train) != TRAIN_KEYS:
        raise ValueError("invalid training profile")
    fixed = {"batch_size": 1, "grad_accum": 1, "mixed_precision": "bf16", "attention_backend": "none",
             "grad_checkpoint": True, "cache_latents": True, "optimizer_type": "adamw", "lr_scheduler": "none",
             "blocks_to_swap": 0, "sample_steps": 0, "sample_every": 0, "save_every_epochs": 0,
             "save_every_steps": 0, "save_state_every_epochs": 0, "save_state_every_steps": 0}
    if any(type(train[k]) is not type(v) or train[k] != v for k, v in fixed.items()):
        raise ValueError("R1 training safety/policy values are fixed")
    for key, low, high in [("images", 2, 32), ("epochs", 1, 20), ("resolution", 256, 1024)]:
        if type(train[key]) is not int or not low <= train[key] <= high:
            raise ValueError("invalid training bound")
    if train["resolution"] % 64 or train["images"] * train["epochs"] < config["warmup"] + config["measured"]:
        raise ValueError("training budget cannot cover successful update target")
    if type(train["learning_rate"]) not in {float, int} or not 0 < train["learning_rate"] <= 0.001:
        raise ValueError("invalid learning rate")
    return config


def eager_backend() -> dict:
    # Reject overrides before importing any LyCORIS module. This is not R2's
    # probe/fallback: even a successful optional backend is out of scope.
    if os.environ.get("LYCORIS_KERNEL_BACKEND", "torch").strip().lower() != "torch":
        raise ValueError("R1 requires the torch backend")
    os.environ["LYCORIS_KERNEL_BACKEND"] = "torch"
    from utils.lycoris_backend import get_lycoris_runtime_info

    info = get_lycoris_runtime_info()
    if info["version"] != "4.0.0" or info["resolved"] != "torch":
        raise ValueError("R1 requires LyCORIS 4.0.0 resolved torch")
    return {"version": info["version"], "requested": "torch", "resolved": "torch"}


def metric(value, unit: str, method: str, scope: str, reason: str | None = None) -> dict:
    if (value is None) != (reason is not None):
        raise ValueError("missing metrics need a reason; measured metrics must not have one")
    if value is not None and not math.isfinite(value):
        raise ValueError("metric must be finite")
    return {"value": value, "unit": unit, "method": method, "scope": scope, "unavailable_reason": reason}


def summarize(values: list[float], *, scope: str = "repeats") -> dict:
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("nonempty finite samples required")
    return {"n": len(values), "raw": values, "median": statistics.median(values),
            "min": min(values), "max": max(values),
            "sample_stdev": metric(statistics.stdev(values) if len(values) > 1 else None,
                                   "same_as_samples", "sample standard deviation", scope,
                                   None if len(values) > 1 else "insufficient_repeats")}


def metadata(device: str) -> dict:
    import torch

    def git(*args):
        result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=normal")
    # A boolean only: never disclose local untracked names or remotes.
    result = {"commit": commit, "dirty": None if dirty is None else bool(dirty),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "safetensors": version("safetensors"),
              "python": platform.python_version(), "os": platform.system(), "torch": torch.__version__,
              "cuda_build": torch.version.cuda, "backend": eager_backend(), "device_type": torch.device(device).type,
              "cpu_threads": torch.get_num_threads(), "gpu": None,
              "driver": metric(None, "version", "not queried", "device", "not_collected"),
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
              "matmul_precision": torch.get_float32_matmul_precision(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if torch.device(device).type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory,
                         "capability": [props.major, props.minor]}
    return result


def synchronize(device: str) -> None:
    import torch
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def memory_metrics(device: str, scope: str) -> dict:
    import torch
    cuda = torch.device(device).type == "cuda"
    return {name: metric(function(device) if cuda else None, "bytes", "torch allocator", scope,
                         None if cuda else "cpu_no_vram")
            for name, function in [("allocated", torch.cuda.memory_allocated),
                                   ("reserved", torch.cuda.memory_reserved),
                                   ("peak_allocated", torch.cuda.max_memory_allocated),
                                   ("peak_reserved", torch.cuda.max_memory_reserved)]}


def kernel_measurement(operation, device: str) -> dict:
    import torch
    scope = "separate diagnostic F/B pass; GPU leaf duration sum, not critical path"
    if torch.device(device).type != "cuda":
        return metric(None, "ms", "torch.profiler CUDA activities", scope, "cpu_no_cuda")
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        return metric(None, "ms", "torch.profiler CUDA activities", scope, "cuda_activity_unavailable")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        operation()
        synchronize(device)
    return kernel_events_metric(prof.events())


def kernel_events_metric(events) -> dict:
    import torch
    kernels = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA
               and not e.cpu_children and not any(word in e.name.lower() for word in ("memcpy", "memset"))]
    return metric(sum(e.time_range.elapsed_us() for e in kernels) / 1000 if kernels else None,
                  "ms", "torch.profiler CUDA leaf activities; excludes memcpy/memset",
                  "separate diagnostic F/B; duration sum is not critical path",
                  None if kernels else "kernel_events_unavailable")


def layer_worker(config: dict, case_index: int, algorithm: str, device: str, output: Path) -> dict:
    import torch
    from dataclasses import asdict
    from tools.lycoris_benchmark_cases import adapter_metadata, build_fixture, discover_cases, save_reference, vector_jacobian

    case = discover_cases(config["profile"], config["resolution"], config["text_tokens"])[case_index]
    experiment = {**config, "algorithm": algorithm}
    fixture = build_fixture(case, experiment, device)
    _, layer, adapter, x, cotangent = fixture
    operation = lambda: vector_jacobian(layer, adapter, x, cotangent, validate=False)
    synchronize(device)
    first = time.perf_counter()
    operation()
    synchronize(device)
    first_seconds = time.perf_counter() - first
    for _ in range(config["warmup"] - 1):
        operation()
    synchronize(device)
    steady = memory_metrics(device, "warmup complete")
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(config["measured"]):
        synchronize(device)
        start = time.perf_counter()
        operation()
        synchronize(device)
        durations.append(time.perf_counter() - start)
    memory = memory_metrics(device, "synchronized layer F/B measurement window")
    kernel = kernel_measurement(operation, device)
    reference = save_reference(case, experiment, fixture, output / "reference")
    return {"schema_version": 1, "status": "complete", "kind": "synthetic_layer",
            "environment": metadata(device), "case": asdict(case), "requested": experiment,
            "actual_adapter": adapter_metadata(adapter), "input_stride": list(x.stride()),
            "first_fb": metric(first_seconds, "s", "synchronized perf_counter", "first layer F/B"),
            "fb_seconds": summarize(durations, scope="synchronized iterations within one repeat"), "kernel_device_sum": kernel,
            "training_it_s": metric(None, "updates/s", "none", "layer", "not_training_case"),
            "compile_tuning": metric(None, "s", "eager", "run", "not_applicable_eager"),
            "steady_memory": steady, "memory": memory,
            "reference": str(reference.relative_to(output)).replace("\\", "/")}


def failed_result(reason: str) -> dict:
    return {"schema_version": 1, "status": "failed", "error_code": reason,
            "training_it_s": metric(None, "updates/s", "none", "failed run", reason),
            "fb_wall": metric(None, "s", "none", "failed run", reason),
            "kernel_device_sum": metric(None, "ms", "none", "failed run", reason)}


def run_children(args, config: dict, workspace: Path) -> dict:
    from tools.lycoris_benchmark_training import isolated_environment, validate_assets
    public, private = workspace / "public", workspace / "private"
    public.mkdir()
    private.mkdir()
    write_json(public / "experiment.json", config)
    rows = []
    if args.command == "layer":
        from tools.lycoris_benchmark_cases import discover_cases
        jobs = [(algo, index, "layer") for algo in config["algorithms"]
                for index, _ in enumerate(discover_cases(config["profile"], config["resolution"], config["text_tokens"]))]
    else:
        validate_assets(vars(args), workspace)
        jobs = [(args.scenario, 0, mode) for mode in ("throughput", "fb", "kernel")]
    for algorithm, index, mode in jobs:
        for repeat in range(config["repeats"]):
            run_id = f"{algorithm}-{index}-{mode}-{repeat}"
            work = private / run_id
            work.mkdir()
            out = public / run_id
            out.mkdir()
            cmd = [sys.executable, str(Path(__file__).resolve()), "_worker", "--config", str(public / "experiment.json"),
                   "--mode", mode, "--algorithm", algorithm, "--case-index", str(index),
                   "--device", args.device, "--output", str(out), "--work", str(work)]
            if args.command == "train":
                for key in ("transformer", "vae", "text_encoder", "t5_tokenizer"):
                    cmd.extend(["--" + key.replace("_", "-"), str(Path(getattr(args, key)).resolve())])
            started = time.perf_counter()
            with (work / "run.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(cmd, cwd=work, env=isolated_environment(work), stdout=log, stderr=subprocess.STDOUT)
            row = read_json(out / "result.json") if (out / "result.json").exists() else failed_result("worker_no_result")
            row["process_wall_seconds"] = time.perf_counter() - started
            row["run_id"] = run_id
            rows.append(row)
            if process.returncode or row["status"] != "complete":
                result = {"schema_version": 1, "status": "incomplete", "runs": rows}
                write_json(public / "result.json", result)
                return result
    summary = {}
    for algorithm, index, mode in jobs:
        group = [r for r in rows if r["run_id"].startswith(f"{algorithm}-{index}-{mode}-")]
        if mode == "layer":
            values = [r["fb_seconds"]["median"] for r in group]
        elif mode == "throughput":
            values = [r["training_it_s"]["value"] for r in group]
        else:
            continue
        summary[f"{algorithm}-{index}-{mode}"] = summarize(values)
    result = {"schema_version": 1, "status": "complete", "runs": rows, "repeat_summary": summary,
              "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
    write_json(public / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("layer", "train", "_worker"):
        p = sub.add_parser(command)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        p.add_argument("--output", type=Path, required=True, help="new outside-repository workspace")
        if command == "layer":
            p.add_argument("--profile", choices=["cpu-smoke", "anima-2048", "anima-5120"])
        if command != "layer":
            for key in ("transformer", "vae", "text-encoder", "t5-tokenizer"):
                p.add_argument("--" + key, required=command == "train")
        if command == "train":
            p.add_argument("--scenario", choices=["lora", "lokr"], required=True)
        if command == "_worker":
            p.add_argument("--mode", choices=["layer", "throughput", "fb", "kernel"], required=True)
            p.add_argument("--algorithm", choices=["lora", "lokr", "loha"], required=True)
            p.add_argument("--case-index", type=int, required=True)
            p.add_argument("--work", type=Path, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    child_validated = False
    try:
        eager_backend()
        if args.command == "replay":
            from tools.lycoris_benchmark_cases import replay_reference
            print(json.dumps(replay_reference(args.manifest, args.device), allow_nan=False))
            return 0
        config = read_json(args.config)
        if args.command == "layer" and args.profile:
            config["profile"] = args.profile
        validate_config(config)
        if args.command == "_worker":
            from tools.lycoris_benchmark_training import validate_worker_paths
            validate_worker_paths(args.work, args.output)
            child_validated = True
            if args.mode != "layer":
                from tools.lycoris_benchmark_training import isolated_environment
                environment = isolated_environment(args.work)
                os.environ.clear()
                os.environ.update(environment)
            if args.mode == "layer":
                result = layer_worker(config, args.case_index, args.algorithm, args.device, args.output)
            else:
                from tools.lycoris_benchmark_training import training_worker
                result = training_worker(config, args)
            write_json(args.output / "result.json", result)
            return 0 if result["status"] == "complete" else 1
        from tools.lycoris_benchmark_training import create_workspace, validate_assets
        if args.command == "train":
            if args.device != "cuda" or config["profile"] == "cpu-smoke":
                raise ValueError("R1 training requires CUDA and an architecture profile")
            validate_assets(vars(args), args.output)
        workspace = create_workspace(args.output)
        result = run_children(args, config, workspace)
        print(json.dumps({"status": result["status"], "schema_version": 1}))
        return 0 if result["status"] == "complete" else 1
    except Exception:
        # Exception strings can contain private asset paths. Full traceback only
        # goes to the private child's redirected log, never public result JSON.
        if args.command == "_worker":
            import traceback
            traceback.print_exc()
            # Only a validated child directory is writable, including errors.
            if child_validated:
                write_json(args.output / "result.json", failed_result("worker_failed"))
        else:
            print("Benchmark rejected or failed; inspect private run logs if created.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
