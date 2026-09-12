#!/usr/bin/env python
"""Reproducible LyCORIS eager and restricted Triton benchmark CLI.

Every measurement repeat is a fresh process. No assets are read by layer/replay.
Public JSON is allowlisted; raw child logs and path bindings remain private.
Schema v1 remains the immutable R1 torch-eager contract. Schema v2 adds the
R5 LoRA/LoHa torch-vs-Triton experiment without changing production defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
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

CONFIG_KEYS_V1 = {"schema_version", "profile", "dtype", "algorithms", "rank", "alpha", "factor",
                  "seed", "repeats", "warmup", "measured", "resolution", "text_tokens", "train"}
CONFIG_KEYS_V2 = CONFIG_KEYS_V1 | {"backend"}
TRAIN_KEYS = {"resolution", "batch_size", "grad_accum", "mixed_precision", "attention_backend",
              "grad_checkpoint", "cache_latents", "images", "epochs", "learning_rate",
              "optimizer_type", "lr_scheduler", "blocks_to_swap", "sample_steps", "sample_every",
              "save_every_epochs", "save_every_steps", "save_state_every_epochs", "save_state_every_steps"}


def config_backend(config: dict) -> str:
    return "torch" if config["schema_version"] == 1 else config["backend"]


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
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int:
        raise ValueError("unsupported experiment config")
    schema_version = config["schema_version"]
    expected_keys = {1: CONFIG_KEYS_V1, 2: CONFIG_KEYS_V2}.get(schema_version)
    if expected_keys is None or set(config) != expected_keys:
        raise ValueError("unsupported experiment config")
    if schema_version == 2 and config["backend"] not in {"torch", "triton"}:
        raise ValueError("R5 backend must be torch or triton")
    if config["profile"] not in {"cpu-smoke", "anima-2048", "anima-5120"}:
        raise ValueError("unknown profile")
    if config["dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("unsupported dtype")
    algorithms = config["algorithms"]
    allowed_algorithms = {"lora", "lokr", "loha"} if schema_version == 1 else {"lora", "loha"}
    if not isinstance(algorithms, list) or not algorithms or any(a not in allowed_algorithms for a in algorithms):
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
    # Preserve the R1 contract: optional overrides are never accepted here.
    if os.environ.get("LYCORIS_KERNEL_BACKEND", "torch").strip().lower() != "torch":
        raise ValueError("R1 requires the torch backend")
    os.environ["LYCORIS_KERNEL_BACKEND"] = "torch"
    info = runtime_backend("torch")
    return {"version": info["version"], "requested": "torch", "resolved": "torch"}


def runtime_backend(requested: str) -> dict:
    from utils.lycoris_backend import get_lycoris_runtime_info

    info = get_lycoris_runtime_info()
    if info["version"] != "4.0.0" or info["resolved"] != requested:
        raise ValueError(f"benchmark requires LyCORIS 4.0.0 resolved {requested}")
    return {
        "version": info["version"],
        "requested": requested,
        "resolved": info["resolved"],
        "available": list(info.get("available", ())),
        "fused": list(info.get("fused", ())),
    }


def activate_backend(config: dict, algorithm: str, device: str, *, override: str | None = None) -> dict:
    requested = override or config_backend(config)
    if requested == "torch":
        os.environ["LYCORIS_KERNEL_BACKEND"] = "torch"
        return runtime_backend("torch")
    if requested != "triton" or algorithm not in {"lora", "loha"}:
        raise ValueError("R5 Triton only supports LoRA/LoHa")
    if str(device).split(":", 1)[0].lower() != "cuda":
        raise ValueError("R5 Triton requires CUDA")
    from utils.lycoris_backend import prepare_lycoris_backend

    decision = prepare_lycoris_backend(
        algorithm=algorithm,
        device=device,
        dtype=config["dtype"].removeprefix("torch."),
        rank=config["rank"],
        alpha=config["alpha"],
        factor=config["factor"],
        requested_backend="triton",
    )
    if decision.status != "passed" or decision.configured != "triton" or decision.resolved != "triton":
        raise ValueError("R5 Triton preflight did not resolve Triton")
    return {"version": "4.0.0", **decision.as_dict()}


def metric(value, unit: str, method: str, scope: str, reason: str | None = None) -> dict:
    if (value is None) != (reason is not None):
        raise ValueError("missing metrics need a reason; measured metrics must not have one")
    if value is not None and not math.isfinite(value):
        raise ValueError("metric must be finite")
    return {"value": value, "unit": unit, "method": method, "scope": scope, "unavailable_reason": reason}


def summarize(values: list[float], *, scope: str = "repeats") -> dict:
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("nonempty finite samples required")
    median = statistics.median(values)
    return {"n": len(values), "raw": values, "median": median,
            "min": min(values), "max": max(values),
            "median_absolute_deviation": metric(statistics.median(abs(value - median) for value in values),
                                                "same_as_samples", "median absolute deviation", scope),
            "sample_stdev": metric(statistics.stdev(values) if len(values) > 1 else None,
                                   "same_as_samples", "sample standard deviation", scope,
                                   None if len(values) > 1 else "insufficient_repeats")}


def cold_start_amortization(
    torch_throughput: float, triton_throughput: float,
    torch_first_update: float, triton_first_update: float,
) -> dict:
    extra_seconds = max(0.0, triton_first_update - torch_first_update)
    savings_per_update = (1.0 / torch_throughput) - (1.0 / triton_throughput)
    scope = "diagnostic point estimate from repeat medians; not part of the acceptance gate"
    method = "first-update median delta divided by steady-state seconds/update median delta"
    if extra_seconds == 0.0:
        return metric(0.0, "updates", method, scope)
    if savings_per_update <= 0.0:
        return metric(None, "updates", method, scope, "no_steady_state_savings")
    return metric(extra_seconds / savings_per_update, "updates", method, scope)


def _percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_median_ratio(
    baseline: list[float], candidate: list[float], *, seed: int, samples: int = 20_000,
) -> dict:
    if len(baseline) < 2 or len(candidate) < 2 or samples < 1 or any(v <= 0 for v in baseline + candidate):
        raise ValueError("bootstrap requires positive repeated samples")
    generator = random.Random(seed)
    ratios = []
    for _ in range(samples):
        base = statistics.median(generator.choices(baseline, k=len(baseline)))
        trial = statistics.median(generator.choices(candidate, k=len(candidate)))
        ratios.append(trial / base)
    ratios.sort()
    point = statistics.median(candidate) / statistics.median(baseline)
    return {
        "point": point,
        "ci95": [_percentile(ratios, 0.025), _percentile(ratios, 0.975)],
        "method": "independent percentile bootstrap of median ratio",
        "samples": samples,
        "seed": seed,
    }


def compare_training_results(torch_path: Path, triton_path: Path) -> dict:
    paths = {"torch": torch_path.resolve(strict=True), "triton": triton_path.resolve(strict=True)}
    configs = {name: validate_config(read_json(path.parent / "experiment.json")) for name, path in paths.items()}
    results = {name: read_json(path) for name, path in paths.items()}
    if any(config["schema_version"] != 2 or config["backend"] != name for name, config in configs.items()):
        raise ValueError("comparison inputs must be matching schema-v2 torch and Triton runs")
    normalized = [{k: v for k, v in config.items() if k != "backend"} for config in configs.values()]
    if normalized[0] != normalized[1] or any(result.get("status") != "complete" for result in results.values()):
        raise ValueError("comparison inputs are incomplete or use different experiments")

    observed = {
        backend: {row.get("requested", {}).get("algorithm") for row in result["runs"]}
        for backend, result in results.items()
    }
    algorithms = observed["torch"]
    if (not algorithms or None in algorithms or observed["triton"] != algorithms
            or not algorithms.issubset(set(configs["torch"]["algorithms"]))):
        raise ValueError("comparison inputs contain different or unregistered algorithms")

    comparisons = {}
    for algorithm in sorted(algorithms):
        key = f"{algorithm}-0-r5"
        groups = {}
        for backend, result in results.items():
            rows = [row for row in result["runs"] if row["run_id"].startswith(key + "-")]
            if len(rows) != configs[backend]["repeats"] or any(row["status"] != "complete" for row in rows):
                raise ValueError("comparison repeat set is incomplete")
            if any(set(row["dispatch_choices"]) != {backend} for row in rows):
                raise ValueError("comparison contains an unexpected dispatcher choice")
            groups[backend] = rows
        throughput = {name: [row["training_it_s"]["value"] for row in rows] for name, rows in groups.items()}
        peak_reserved = {name: [row["window_memory"]["peak_reserved"]["value"] for row in rows]
                         for name, rows in groups.items()}
        first_update = {name: [row["first_update_seconds"]["value"] for row in rows]
                        for name, rows in groups.items()}
        ratio = bootstrap_median_ratio(
            throughput["torch"], throughput["triton"], seed=configs["torch"]["seed"],
        )
        torch_throughput = statistics.median(throughput["torch"])
        triton_throughput = statistics.median(throughput["triton"])
        torch_first_update = statistics.median(first_update["torch"])
        triton_first_update = statistics.median(first_update["triton"])
        first_update_extra = triton_first_update - torch_first_update
        amortization = cold_start_amortization(
            torch_throughput, triton_throughput, torch_first_update, triton_first_update,
        )
        memory_ratio = statistics.median(peak_reserved["triton"]) / statistics.median(peak_reserved["torch"])
        accepted = ratio["point"] >= 1.05 and ratio["ci95"][0] > 1.0 and memory_ratio <= 1.05
        comparisons[algorithm] = {
            "status": "candidate" if accepted else "no_stable_gain",
            "accepted": accepted,
            "throughput": {name: summarize(values) for name, values in throughput.items()},
            "triton_over_torch": ratio,
            "peak_reserved_bytes": {name: summarize(values) for name, values in peak_reserved.items()},
            "peak_reserved_ratio": memory_ratio,
            "first_update_seconds": {name: summarize(values) for name, values in first_update.items()},
            "triton_first_update_extra_seconds": metric(
                first_update_extra, "s", "difference of repeat medians", "first successful update diagnostic",
            ),
            "cold_start_amortization_updates": amortization,
        }
    return {
        "schema_version": 1,
        "kind": "lycoris_backend_comparison",
        "status": "complete",
        "thresholds": {"minimum_throughput_ratio": 1.05, "ci95_lower_must_exceed": 1.0,
                       "maximum_peak_reserved_ratio": 1.05},
        "comparisons": comparisons,
        "decision": "candidate" if any(value["accepted"] for value in comparisons.values()) else "keep_torch_default",
    }


def metadata(device: str, backend: dict | None = None) -> dict:
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
              "cuda_build": torch.version.cuda, "backend": backend or eager_backend(),
              "device_type": torch.device(device).type,
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
    backend = eager_backend()
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
    return {"schema_version": 1, "status": "complete", "kind": "synthetic_layer",
            "environment": metadata(device, backend), "case": asdict(case), "requested": experiment,
            "actual_adapter": adapter_metadata(adapter), "input_stride": list(x.stride()),
            "first_fb": metric(first_seconds, "s", "synchronized perf_counter", "first layer F/B"),
            "fb_seconds": summarize(durations, scope="synchronized iterations within one repeat"), "kernel_device_sum": kernel,
            "training_it_s": metric(None, "updates/s", "none", "layer", "not_training_case"),
            "compile_tuning": metric(None, "s", "eager", "run", "not_applicable_eager"),
            "steady_memory": steady, "memory": memory,
            "reference": str(save_reference(case, experiment, fixture, output / "reference", backend=backend)
                             .relative_to(output)).replace("\\", "/")}


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
        if config_backend(config) != "torch":
            raise ValueError("layer references must be generated with the torch backend")
        from tools.lycoris_benchmark_cases import discover_cases
        jobs = [(algo, index, "layer") for algo in config["algorithms"]
                for index, _ in enumerate(discover_cases(config["profile"], config["resolution"], config["text_tokens"]))]
    else:
        validate_assets(vars(args), workspace)
        modes = ("throughput", "fb", "kernel") if config["schema_version"] == 1 else ("r5",)
        jobs = [(args.scenario, 0, mode) for mode in modes]
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
                process = subprocess.run(
                    cmd,
                    cwd=work,
                    env=isolated_environment(work, backend=config_backend(config)),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
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
        elif mode in {"throughput", "r5"}:
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
            p.add_argument("--scenario", choices=["lora", "lokr", "loha"], required=True)
        if command == "_worker":
            p.add_argument("--mode", choices=["layer", "throughput", "fb", "kernel", "r5"], required=True)
            p.add_argument("--algorithm", choices=["lora", "lokr", "loha"], required=True)
            p.add_argument("--case-index", type=int, required=True)
            p.add_argument("--work", type=Path, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    replay.add_argument("--backend", choices=["torch", "triton"], default="torch")
    compare = sub.add_parser("compare")
    compare.add_argument("--torch-result", type=Path, required=True)
    compare.add_argument("--triton-result", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    child_validated = False
    try:
        if args.command == "replay":
            from tools.lycoris_benchmark_cases import replay_reference
            print(json.dumps(replay_reference(args.manifest, args.device, backend=args.backend), allow_nan=False))
            return 0
        if args.command == "compare":
            output = args.output.absolute()
            if output.exists() or not output.parent.is_dir() or output == REPO or REPO in output.parents:
                raise ValueError("comparison output must be a new file outside the repository")
            result = compare_training_results(args.torch_result, args.triton_result)
            write_json(output, result)
            print(json.dumps({"status": result["status"], "decision": result["decision"]}))
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
                environment = isolated_environment(args.work, backend=config_backend(config))
                os.environ.clear()
                os.environ.update(environment)
            if args.mode == "layer":
                if config_backend(config) != "torch":
                    raise ValueError("layer references must be generated with the torch backend")
                result = layer_worker(config, args.case_index, args.algorithm, args.device, args.output)
            else:
                from tools.lycoris_benchmark_training import training_worker
                result = training_worker(config, args)
            write_json(args.output / "result.json", result)
            return 0 if result["status"] == "complete" else 1
        from tools.lycoris_benchmark_training import create_workspace, validate_assets
        if args.command == "train":
            if args.device != "cuda" or config["profile"] == "cpu-smoke":
                raise ValueError("pipeline training requires CUDA and an architecture profile")
            allowed = {"lora", "lokr"} if config["schema_version"] == 1 else {"lora", "loha"}
            if args.scenario not in allowed:
                raise ValueError("training scenario is outside this config schema")
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
