"""Anima-only synthetic-data runner and tool-owned observation/measurement.

Uses anima_train.run_training, never a second training loop. Workspaces are
fresh, local, outside the repository/assets. This is not an OS security sandbox.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import stat
import sys
import sysconfig
import time
from collections import Counter
from pathlib import Path

from tools.benchmark_lycoris import (
    REPO, config_backend, kernel_events_metric, memory_metrics, metadata,
    metric, runtime_backend, synchronize, validate_config, write_json,
)

ASSET_KEYS = ("transformer", "vae", "text_encoder", "t5_tokenizer")


def reject_links(path: Path) -> None:
    for node in (path, *path.parents):
        if node.is_symlink() or (node.exists() and getattr(node.lstat(), "st_file_attributes", 0)
                                & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError("linked/reparse paths are not allowed")


def overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def validate_workspace(path: Path) -> Path:
    path = path.absolute()
    reject_links(path)
    resolved = path.resolve()
    if overlaps(resolved, REPO):
        raise ValueError("workspace must not overlap repository")
    if resolved.exists():
        raise ValueError("workspace must be newly created, not reused")
    if not resolved.parent.is_dir():
        raise ValueError("workspace parent must already exist")
    return resolved


def create_workspace(path: Path) -> Path:
    resolved = validate_workspace(path)
    resolved.mkdir(exist_ok=False)
    return resolved


def validate_assets(bindings: dict, workspace: Path) -> dict[str, Path]:
    result = {}
    for key in ASSET_KEYS:
        raw = bindings.get(key)
        if not raw:
            raise ValueError("all four local assets must be explicitly bound")
        path = Path(raw).absolute()
        reject_links(path)
        path = path.resolve(strict=True)
        if overlaps(path, workspace.resolve()):
            raise ValueError("workspace and assets must not overlap")
        if key in {"transformer", "vae"}:
            if not path.is_file() or path.suffix != ".safetensors" or path.stat().st_nlink != 1:
                raise ValueError("asset must be an unlinked local safetensors file")
        else:
            if not path.is_dir() or not any(path.iterdir()):
                raise ValueError("tokenizer/text encoder directory is missing or empty")
            for entry in path.rglob("*"):
                reject_links(entry)
                if entry.is_file() and entry.stat().st_nlink != 1:
                    raise ValueError("hardlinked asset tree is not supported")
        result[key] = path
    if result["transformer"] == result["vae"] or overlaps(result["text_encoder"], result["t5_tokenizer"]):
        raise ValueError("asset aliases or nested tokenizer bindings are not supported")
    return result


def validate_worker_paths(work: Path, output: Path) -> None:
    for path in (work, output):
        reject_links(path.absolute())
        if not path.is_dir() or overlaps(path.resolve(), REPO):
            raise ValueError("invalid worker directory")
    if (work.parent.name != "private" or output.parent.name != "public"
            or work.parent.parent.resolve() != output.parent.parent.resolve()
            or work.name != output.name or any(output.iterdir())):
        raise ValueError("worker directories must be fresh paired private/public children")


def isolated_environment(work: Path, *, backend: str = "torch") -> dict[str, str]:
    if backend not in {"torch", "triton"}:
        raise ValueError("benchmark backend must be torch or triton")
    # No inherited tokens, proxies, PYTHONPATH, Studio task IDs, WandB settings,
    # debug launchers or arbitrary library cache locations.
    allowed = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "NUMBER_OF_PROCESSORS",
               "PROCESSOR_ARCHITECTURE", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"}
    env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
    for folder in ("tmp", "home", "cache"):
        (work / folder).mkdir(exist_ok=True)
    for key in ("TMP", "TEMP", "TMPDIR"):
        env[key] = str(work / "tmp")
    for key in ("HOME", "USERPROFILE"):
        env[key] = str(work / "home")
    for key in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "TORCH_HOME",
                "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        env[key] = str(work / "cache" / key.lower())
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "WANDB_ENABLED": "0",
                "WANDB_MODE": "disabled", "WANDB_DISABLED": "true", "LYCORIS_KERNEL_BACKEND": backend,
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "TOKENIZERS_PARALLELISM": "false"})
    return env


def _audit_command(args) -> list[str]:
    command = args[1] if len(args) > 1 else ()
    if isinstance(command, (list, tuple)):
        return [os.fspath(value) for value in command]
    if isinstance(command, str):
        return [value.strip('"') for value in shlex.split(command, posix=os.name != "nt")]
    return []


def _trusted_ptxas_path() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return (Path(sysconfig.get_paths()["purelib"]) / "triton" / "backends" / "nvidia" / "bin"
            / f"ptxas{suffix}").resolve()


def _is_trusted_benchmark_process(command: list[str]) -> bool:
    if command == ["rocm-sdk", "path", "--root"]:
        # Triton Windows imports its AMD plugin even on NVIDIA. Permit only the
        # expected failed lookup; never execute an actual PATH-provided binary.
        return shutil.which("rocm-sdk") is None
    if len(command) == 2:
        executable, worker = map(lambda value: Path(value).resolve(), command)
        if executable == Path(sys.executable).resolve() and worker == REPO / "utils" / "_lycoris_probe_worker.py":
            return True
    if len(command) == 2 and command[1] == "--version":
        return Path(command[0]).resolve() == _trusted_ptxas_path() and _trusted_ptxas_path().is_file()
    if len(command) != 8:
        return False
    executable = Path(command[0]).resolve()
    if executable != _trusted_ptxas_path() or not executable.is_file():
        return False
    if (command[1:3] != ["-lineinfo", "-v"]
            or not re.fullmatch(r"--regAllocOptLevel=[0-4]", command[3])
            or not re.fullmatch(r"--gpu-name=sm_[0-9]+[a-z]?", command[4])
            or command[6] != "-o"):
        return False
    temporary_value = os.environ.get("TMPDIR")
    if not temporary_value:
        return False
    temporary = Path(temporary_value).resolve()
    source, target = Path(command[5]).resolve(), Path(command[7]).resolve()
    return (temporary.is_dir() and source.is_file() and source.is_relative_to(temporary)
            and target.is_relative_to(temporary) and source.suffix == ".ptx"
            and str(target) == str(source) + ".o")


def deny_network_and_processes(event, args) -> None:
    """Worker-local audit hook: no downloads, external logging or installers.

    Schema-v2 Triton runs admit only the exact production probe, Triton's pinned
    package-local ptxas with private temp files, and a guaranteed-missing
    ``rocm-sdk`` discovery attempt. Python audit hooks remain defense in depth,
    not a boundary against native code.
    """
    if event == "subprocess.Popen":
        if _is_trusted_benchmark_process(_audit_command(args)):
            return
        raise PermissionError("benchmark forbids untrusted subprocesses")
    if event in {"socket.connect", "socket.getaddrinfo", "os.system", "os.posix_spawn"}:
        raise PermissionError("benchmark forbids network and external processes")


def generate_dataset(directory: Path, *, seed: int, resolution: int, count: int) -> dict:
    import numpy as np
    from PIL import Image

    directory.mkdir(exist_ok=False)
    generator = np.random.Generator(np.random.PCG64(seed))
    # Public synthetic captions, never imported from user data.
    captions = ["A synthetic red circle on a blue background.",
                "A synthetic green square on a yellow background."]
    for index in range(count):
        pixels = generator.integers(0, 256, size=(resolution, resolution, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(directory / f"synthetic_{index:03d}.png")
        (directory / f"synthetic_{index:03d}.txt").write_text(captions[index % len(captions)], encoding="utf-8")
    return {"generator_version": 1, "rng": "numpy.PCG64", "seed": seed,
            "images": count, "resolution": resolution, "mode": "RGB", "captions": captions,
            "content": "uniform random pixels; captions are public fixtures, not image semantics"}


def make_training_args(config: dict, bindings: dict, work: Path, algorithm: str):
    from training.bootstrap import apply_yaml_config

    validate_config(config)
    schema_version = config["schema_version"]
    allowed = {"lora", "lokr"} if schema_version == 1 else {"lora", "loha"}
    if algorithm not in allowed:
        raise ValueError("training algorithm is outside this benchmark schema")
    assets = validate_assets(bindings, work)
    values = dict(config["train"])
    values.pop("images")
    values.update({"config": None, "model_family": "anima", "data_dir": str(work / "dataset"),
                   "output_dir": str(work / "output"), "output_name": "synthetic_adapter",
                   "transformer_path": str(assets["transformer"]), "vae_path": str(assets["vae"]),
                   "text_encoder_path": str(assets["text_encoder"]), "t5_tokenizer_path": str(assets["t5_tokenizer"]),
                   "lora_type": algorithm, "lora_rank": config["rank"], "lora_alpha": config["alpha"],
                   "lokr_factor": config["factor"], "lycoris_backend": config_backend(config),
                   "lora_dora": False, "lora_dropout": 0.0, "lora_rank_dropout": 0.0,
                   "lora_module_dropout": 0.0, "seed": config["seed"],
                   "max_steps": config["warmup"] + config["measured"], "num_workers": 0,
                   "auto_install": False, "interactive": False, "no_live_curve": True,
                   "monitor_state_file": str(work / "monitor" / "state.json"),
                   "resume_state": None, "resume_lora": None, "reg_data_dir": None,
                   "repeats": 1, "reg_repeats": 1, "log_every": 1})
    args = apply_yaml_config(argparse.Namespace(**values), {})
    assert_safe_args(args, work)
    return args


def assert_safe_args(args, work: Path) -> None:
    for key, relative in (("data_dir", "dataset"), ("output_dir", "output"), ("monitor_state_file", "monitor/state.json")):
        path = Path(getattr(args, key)).absolute()
        reject_links(path)
        if path.resolve() != work / relative:
            raise ValueError("training write path escaped private workspace")
    for key in ("config", "resume_state", "resume_lora", "reg_data_dir", "auto_install", "interactive",
                "sample_steps", "sample_every", "save_every_epochs", "save_every_steps",
                "save_state_every_epochs", "save_state_every_steps", "navit_packing", "leap_enabled", "sra_enabled"):
        if getattr(args, key, None):
            raise ValueError("unsafe or unsupported training setting")
    if args.model_family != "anima" or args.attention_backend != "none" or args.num_workers != 0:
        raise ValueError("training policy changed during normalization")
    if (args.lycoris_backend not in {"torch", "triton"} or args.lora_dora
            or any(float(getattr(args, key, 0.0) or 0.0) != 0.0 for key in (
                "lora_dropout", "lora_rank_dropout", "lora_module_dropout",
            ))):
        raise ValueError("unsafe LyCORIS backend or adapter policy")
    if args.lycoris_backend == "triton" and args.lora_type not in {"lora", "loha"}:
        raise ValueError("Triton benchmark only supports LoRA/LoHa")


class BenchmarkObserver:
    """Continuous update throughput OR synchronized F/B OR kernel diagnostics.

    Attempted microbatches, skipped groups, and actual updates are distinct. A
    warmup boundary is a successful update, not a batch index or loss log line.
    """

    def __init__(self, config: dict, mode: str, device: str, work: Path | None = None,
                 *, clock=time.perf_counter, sync=synchronize):
        self.config, self.mode, self.device, self.work = config, mode, device, work
        self.clock, self.sync = clock, sync
        self.successful = self.microbatches = self.images = self.loss_skipped = 0
        self.skipped = {"no_gradients": 0, "nonfinite_gradients": 0}
        self.phases, self.phase_starts, self.fb_seconds, self.kernel_metrics = {}, {}, [], []
        self.start = self.end = self.loop_start = self.loop_end = self.first_update = None
        self.window_counts = self.window_end_counts = None
        self.steady_memory = self.window_memory = None
        self.fb_start = None
        self.profiler = None
        self.kernel_reason = None
        self.hooks, self.shapes, self.initial = [], {}, []
        self.learning_changed = None
        self.gradients_finite = True
        self.gradient_batches = 0
        self.adapter = []
        self.effective = {}
        self.backend = None
        self.dispatch_choices = Counter()
        self.dispatch_originals = []
        self.kernel_profiled = False
        self.closed = False

    def _install_dispatch_observer(self) -> None:
        if self.config["schema_version"] != 2 or self.mode != "r5":
            return
        from lycoris.functional import locon, loha

        for module in (locon, loha):
            original = module.choose

            def record(*args, _original=original, **kwargs):
                choice = _original(*args, **kwargs)
                self.dispatch_choices[str(choice)] += 1
                return choice

            self.dispatch_originals.append((module, original))
            module.choose = record

    def _restore_dispatch_observer(self) -> None:
        for module, original in self.dispatch_originals:
            module.choose = original
        self.dispatch_originals.clear()

    def phase_started(self, name, ctx):
        if self.work is not None:
            assert_safe_args(ctx.args, self.work)
        self.sync(self.device)
        self.phase_starts[name] = self.clock()

    def phase_finished(self, name, ctx):
        if self.work is not None:
            assert_safe_args(ctx.args, self.work)
        self.sync(self.device)
        self.phases[name] = self.clock() - self.phase_starts[name]

    def _counts(self):
        return {"microbatches": self.microbatches, "images": self.images,
                "loss_skipped": self.loss_skipped, "skipped_groups": sum(self.skipped.values())}

    def loop_started(self, ctx):
        import torch
        from tools.lycoris_benchmark_cases import adapter_metadata

        # CPU snapshots only, outside all measurement windows. Never export real
        # training tensors; permit healthy zero first-step gradients.
        self.initial = [p.detach().cpu().clone() for p in ctx.trainable_params]
        if hasattr(ctx.injector, "network"):
            self.adapter = adapter_metadata(ctx.injector)
        if self.config["schema_version"] == 2:
            self.backend = runtime_backend(config_backend(self.config))
            self._install_dispatch_observer()
        for key in ("batch_size", "grad_accum", "resolution", "mixed_precision", "attention_backend",
                    "grad_checkpoint", "cache_latents", "blocks_to_swap", "lora_type", "lycoris_backend",
                    "lora_rank", "lora_alpha", "lokr_factor", "lora_dora", "lora_dropout",
                    "lora_rank_dropout", "lora_module_dropout", "optimizer_type", "lr_scheduler", "max_steps"):
            self.effective[key] = getattr(ctx.args, key, None)
        self.effective["dtype"] = str(ctx.dtype)
        batch_size = getattr(ctx.args, "batch_size", None)
        self.effective["effective_batch_size"] = (batch_size * ctx.args.grad_accum if batch_size is not None else None)
        self.effective["base_parameter_dtypes"] = sorted({str(p.dtype) for p in ctx.model.parameters()})
        self.effective["base_parameter_devices"] = sorted({p.device.type for p in ctx.model.parameters()})
        self.effective["attention_dispatch"] = "PyTorch SDPA; specific kernel not observed"
        if self.mode in {"fb", "r5"}:
            for name, module in ctx.model.named_modules():
                if name in {"blocks.0.self_attn.q_proj", "blocks.0.cross_attn.k_proj", "blocks.0.mlp.layer1", "blocks.0.mlp.layer2"}:
                    self.hooks.append(module.register_forward_hook(self._shape_hook(name)))
        if self.mode in {"kernel", "r5"} and (torch.device(self.device).type != "cuda" or
                torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities()):
            self.kernel_reason = "cpu_no_cuda" if torch.device(self.device).type != "cuda" else "cuda_activity_unavailable"
        self.sync(self.device)
        self.loop_start = self.clock()

    def _shape_hook(self, name):
        def record(module, inputs, output):
            if name not in self.shapes:
                self.shapes[name] = {"input_shape": list(inputs[0].shape), "input_stride": list(inputs[0].stride()),
                                     "input_dtype": str(inputs[0].dtype), "output_shape": list(output.shape),
                                     "output_dtype": str(output.dtype), "provenance": "observed_training_metadata"}
        return record

    def forward_started(self, ctx, *, batch_size):
        self.microbatches += 1
        self.images += batch_size
        active = self.start is not None and self.end is None
        r5_warmup = self.mode == "r5" and self.successful < self.config["warmup"]
        if (self.mode == "fb" and active) or r5_warmup:
            self.sync(self.device)
            self.fb_start = self.clock()
        if ((self.mode == "kernel" and active)
                or (r5_warmup and not self.kernel_profiled)) and self.kernel_reason is None:
            import torch
            self.profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                              torch.profiler.ProfilerActivity.CUDA])
            self.profiler.__enter__()
            self.kernel_profiled = True

    def backward_finished(self, ctx, *, loss_is_finite):
        self.loss_skipped += not loss_is_finite
        if self.fb_start is not None:
            self.sync(self.device)
            self.fb_seconds.append({"seconds": self.clock() - self.fb_start, "backward_performed": loss_is_finite})
            self.fb_start = None
            import torch
            gradients = [p.grad for p in ctx.trainable_params if p.grad is not None]
            self.gradients_finite &= all(bool(torch.isfinite(g).all()) for g in gradients)
            self.gradient_batches += bool(gradients)
        if self.profiler is not None:
            self.sync(self.device)
            self.profiler.__exit__(None, None, None)
            self.kernel_metrics.append(kernel_events_metric(self.profiler.events()))
            self.profiler = None

    def optimizer_step_finished(self, ctx, *, reason):
        if reason != "updated":
            self.skipped[reason] += 1
            return
        self.successful += 1
        # Only first/warmup/end boundaries synchronize. Never per-step sync in
        # headline throughput, and never mean reciprocal durations or EMA.
        if self.first_update is None:
            self.sync(self.device)
            self.first_update = self.clock()
        if self.successful == self.config["warmup"]:
            self._restore_dispatch_observer()
            self.sync(self.device)
            self.steady_memory = memory_metrics(self.device, "warmup complete")
            import torch
            if torch.device(self.device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            self.start = self.clock()
            self.window_counts = self._counts()
        if self.successful == self.config["warmup"] + self.config["measured"]:
            self.sync(self.device)
            self.end = self.clock()
            self.window_end_counts = self._counts()
            self.window_memory = memory_metrics(self.device, "measured successful update window")

    def loop_finished(self, ctx):
        import torch
        self.sync(self.device)
        self.loop_end = self.clock()
        self.learning_changed = any(not torch.equal(before, after.detach().cpu())
                                    for before, after in zip(self.initial, ctx.trainable_params, strict=True))
        self.gradients_finite &= all(bool(torch.isfinite(p).all()) for p in ctx.trainable_params)
        self.initial.clear()
        self.closed = True
        self.close()

    def close(self):
        self._restore_dispatch_observer()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None

    def result(self) -> dict:
        complete = self.closed and self.end is not None and self.learning_changed and self.gradients_finite
        wall = self.end - self.start if self.end is not None else None
        throughput_mode = self.mode in {"throughput", "r5"}
        throughput = self.config["measured"] / wall if complete and wall and throughput_mode else None
        reason = None if throughput is not None else ("diagnostic_pass" if not throughput_mode else "incomplete_or_invalid_training")
        counts = ({k: self.window_end_counts[k] - v for k, v in self.window_counts.items()}
                  if self.window_end_counts is not None else None)
        result = {"schema_version": 1, "status": "complete" if complete else "incomplete", "kind": "synthetic_data_training",
                "mode": self.mode, "training_it_s": metric(throughput, "successful optimizer updates/s",
                    "successful updates / continuous synchronized wall", "post-warmup update boundaries; includes fetch/encode/IO between boundaries", reason),
                "window_wall": metric(wall, "s", "synchronized perf_counter", "measurement boundaries",
                                      None if wall is not None else "incomplete_window"),
                "successful_updates": self.successful, "measured_updates": self.config["measured"] if wall is not None else None,
                "attempted_microbatches": self.microbatches, "skipped_groups": self.skipped,
                "skipped_loss_microbatches": self.loss_skipped, "window_counts": counts,
                "phase_seconds": self.phases, "fb_diagnostics": self.fb_seconds, "kernel_diagnostics": self.kernel_metrics,
                "kernel_unavailable": metric(None, "ms", "torch.profiler", "F/B diagnostic",
                    self.kernel_reason or ("separate_pass" if self.mode not in {"kernel", "r5"} else "see_per_microbatch_metrics")),
                "first_update_seconds": metric(self.first_update - self.loop_start if self.first_update is not None else None,
                    "s", "synchronized perf_counter", "loop entry including first fetch through first update",
                    None if self.first_update is not None else "no_successful_updates"),
                "loop_tail_seconds": metric(self.loop_end - self.end if self.closed and self.end is not None else None,
                    "s", "synchronized perf_counter", "last measured update through loop return, excludes finalize",
                    None if self.closed and self.end is not None else "incomplete_window"),
                "warmup_seconds": metric(self.start - self.loop_start if self.start is not None else None,
                    "s", "synchronized perf_counter", "loop entry through last warmup successful update",
                    None if self.start is not None else "warmup_incomplete"),
                "steady_memory": self.steady_memory, "window_memory": self.window_memory,
                "effective": self.effective, "actual_adapter": self.adapter, "observed_shapes": self.shapes,
                "learning": {"parameters_changed": self.learning_changed, "finite": self.gradients_finite,
                             "diagnostic_batches_with_gradients": self.gradient_batches},
                "compile_tuning": metric(
                    None,
                    "s",
                    "not isolated",
                    "run",
                    "included_in_warmup" if config_backend(self.config) == "triton" else "not_applicable_eager",
                ),
                "cache_policy": {"process": "fresh child per pass/repeat", "dataset": "fresh synthetic directory; VAE cache built during setup",
                    "text": "production Anima online encoding/cache policy", "os_page_cache": "unknown_not_flushed",
                    "cache_hits": metric(None, "hits", "none", "run", "no_counter_exposed")},
                "policies": {"sampling": False, "periodic_save": False, "final_and_auto_epoch_output": "private workspace",
                             "external_logging": False, "network": False, "auto_install": False}}
        if self.config["schema_version"] == 2:
            result.update({"backend": self.backend, "dispatch_choices": dict(self.dispatch_choices)})
        return result


def training_worker(config: dict, options) -> dict:
    import torch

    if options.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("real pipeline training requires CUDA; no CPU fallback")
    requested_backend = config_backend(config)
    if os.environ.get("LYCORIS_KERNEL_BACKEND", "torch") != requested_backend:
        raise ValueError("worker backend environment does not match the experiment")
    work = options.work.resolve()
    args = make_training_args(config, vars(options), work, options.algorithm)
    dataset = generate_dataset(work / "dataset", seed=config["seed"], resolution=config["train"]["resolution"],
                               count=config["train"]["images"])
    write_json(work / "training-args.private.json", vars(args))
    if config["schema_version"] == 1:
        env_info = metadata(options.device)
    else:
        env_info = metadata(options.device, {
            "version": "4.0.0", "requested": requested_backend,
            "resolved": None, "status": "preflight_pending",
        })
    sys.addaudithook(deny_network_and_processes)
    # Import after offline environment has been established by the parent child launcher.
    from anima_train import run_training

    observer = BenchmarkObserver(config, options.mode, options.device, work)
    try:
        run_training(args, observer=observer)
    finally:
        observer.close()
    result = observer.result()
    if config["schema_version"] == 2:
        backend_info = observer.backend or runtime_backend(requested_backend)
        env_info["backend"] = backend_info
        choices = set(result["dispatch_choices"])
        if not choices or choices != {requested_backend}:
            raise ValueError("observed LyCORIS dispatch does not match requested backend")
    result.update({"environment": env_info, "dataset": dataset,
                   "requested": {"algorithm": options.algorithm, "backend": requested_backend,
                                 "rank": config["rank"], "alpha": config["alpha"],
                                 "factor": config["factor"], "seed": config["seed"]}})
    return result
