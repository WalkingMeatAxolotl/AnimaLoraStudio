"""Synthetic workspace, offline policy and independent measurement passes."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.test_loop_nonfinite_loss import _batch, _make_ctx, loop_mod
from tools import benchmark_lycoris as bench
from tools import lycoris_benchmark_training as training


@pytest.fixture
def config():
    return bench.validate_config(bench.read_json(bench.REPO / "tools/benchmark_configs/lycoris_eager_v1.json"))


@pytest.fixture
def assets(tmp_path):
    result = {}
    for key in training.ASSET_KEYS:
        path = tmp_path / (key + (".safetensors" if key in {"transformer", "vae"} else ""))
        if key in {"transformer", "vae"}:
            path.write_bytes(b"synthetic test asset, never loaded")
        else:
            path.mkdir()
            (path / "config.json").write_text("{}")
        result[key] = str(path)
    return result


def test_workspace_and_asset_safety(tmp_path, assets):
    work = training.create_workspace(tmp_path / "new")
    assert work.is_dir()
    assert set(training.validate_assets(assets, work)) == set(training.ASSET_KEYS)
    with pytest.raises(ValueError):
        training.create_workspace(work)
    with pytest.raises(ValueError):
        training.validate_workspace(bench.REPO / "models" / "no-write")
    with pytest.raises(ValueError):
        training.validate_workspace(bench.REPO.parent)
    with pytest.raises(ValueError):
        training.validate_assets(assets, Path(assets["text_encoder"]) / "nested")
    with pytest.raises(ValueError):
        training.validate_assets({**assets, "t5_tokenizer": assets["text_encoder"]}, work)
    with pytest.raises(ValueError):
        training.validate_assets({**assets, "t5_tokenizer": None}, work)


def test_link_and_reparse_rejected_without_platform_privileges(tmp_path, monkeypatch):
    path = tmp_path / "unsafe"
    path.mkdir()
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == path or original(self))
    with pytest.raises(ValueError):
        training.validate_workspace(path / "child")
    monkeypatch.setattr(Path, "is_symlink", original)
    original_stat = Path.lstat
    def fake_stat(self):
        if self == path:
            return SimpleNamespace(st_file_attributes=training.stat.FILE_ATTRIBUTE_REPARSE_POINT,
                                   st_mode=original_stat(self).st_mode)
        return original_stat(self)
    monkeypatch.setattr(Path, "lstat", fake_stat)
    with pytest.raises(ValueError):
        training.reject_links(path)


def test_dataset_reproducible_generated_only(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    info = training.generate_dataset(first, seed=567, resolution=32, count=2)
    assert info == training.generate_dataset(second, seed=567, resolution=32, count=2)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}
    with pytest.raises(FileExistsError):
        training.generate_dataset(first, seed=567, resolution=32, count=2)


def test_args_use_authoritative_normalization_and_isolated_paths(tmp_path, assets, config):
    work = training.create_workspace(tmp_path / "work")
    args = training.make_training_args(config, assets, work, "lokr")
    assert args.resolution == [512]
    assert args.lokr_factor == 8 and args.lora_rank == 8 and args.lora_alpha == 4
    assert args.max_steps == 7 and args.grad_accum == 1
    assert not args.auto_install and not args.interactive
    assert args.sample_steps == args.sample_every == args.save_every_epochs == 0
    assert args.mixed_precision == "bf16" and args.attention_backend == "none"
    training.assert_safe_args(args, work)
    args.data_dir = assets["text_encoder"]
    with pytest.raises(ValueError):
        training.assert_safe_args(args, work)


@pytest.mark.parametrize("key,value", [("sample_every", 1), ("attention_backend", "flash_attn"),
                                       ("blocks_to_swap", 1), ("mixed_precision", "fp16"), ("images", 0)])
def test_training_profile_is_bounded(config, key, value):
    config["train"][key] = value
    with pytest.raises(ValueError):
        bench.validate_config(config)


def test_offline_no_external_logging_or_inherited_private_environment(tmp_path, monkeypatch):
    for key in ("WANDB_API_KEY", "LORA_TASK_ID", "PYTHONPATH", "HF_TOKEN", "HTTPS_PROXY", "ANIMA_TRACE_ID"):
        monkeypatch.setenv(key, "PRIVATE_CANARY")
    env = training.isolated_environment(tmp_path)
    assert "PRIVATE_CANARY" not in json.dumps(env)
    assert env["WANDB_MODE"] == "disabled" and env["WANDB_ENABLED"] == "0"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    for event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system", "os.posix_spawn"):
        with pytest.raises(PermissionError):
            training.deny_network_and_processes(event, ())
    training.deny_network_and_processes("open", ())


class Clock:
    def __init__(self):
        self.now = 0.0
        self.syncs = 0

    def __call__(self):
        return self.now

    def sync(self, device):
        self.syncs += 1


def test_throughput_counts_successful_updates_not_attempts_or_ema(config):
    config.update(warmup=2, measured=5)
    clock = Clock()
    observer = training.BenchmarkObserver(config, "throughput", "cpu", clock=clock, sync=clock.sync)
    observer.loop_start = 0
    for step in range(2):
        clock.now += 10
        observer.forward_started(None, batch_size=1)
        observer.backward_finished(None, loss_is_finite=True)
        observer.optimizer_step_finished(None, reason="updated")
    syncs = clock.syncs
    # Accumulated groups plus skipped loss/gradient attempts inside the window.
    for _ in range(2):
        observer.forward_started(None, batch_size=3)
        observer.backward_finished(None, loss_is_finite=False)
    observer.optimizer_step_finished(None, reason="no_gradients")
    observer.optimizer_step_finished(None, reason="nonfinite_gradients")
    for step in range(5):
        clock.now += 2
        observer.forward_started(None, batch_size=1)
        observer.backward_finished(None, loss_is_finite=True)
        observer.optimizer_step_finished(None, reason="updated")
    assert clock.syncs - syncs == 1  # only the measurement end, no per-step sync
    observer.closed = observer.learning_changed = True
    observer.loop_end = clock.now + 4
    result = observer.result()
    assert result["training_it_s"]["value"] == 0.5
    assert result["window_wall"]["value"] == 10
    assert result["successful_updates"] == 7
    assert result["window_counts"] == {"microbatches": 7, "images": 11, "loss_skipped": 2, "skipped_groups": 2}
    assert result["loop_tail_seconds"]["value"] == 4
    assert result["fb_diagnostics"] == []


@pytest.mark.parametrize("mode", ["throughput", "fb", "kernel"])
def test_observer_real_cpu_loop_with_fake_family(tmp_path, monkeypatch, config, mode):
    config.update(warmup=1, measured=2)
    ctx = _make_ctx(tmp_path, [_batch() for _ in range(6)], monkeypatch)
    observer = training.BenchmarkObserver(config, mode, "cpu")
    loop_mod.run(ctx, observer=observer)
    result = observer.result()
    assert result["status"] == "complete"
    assert result["successful_updates"] == 3
    assert result["learning"]["parameters_changed"] is True
    if mode == "throughput":
        assert result["training_it_s"]["value"] > 0
    else:
        assert result["training_it_s"]["value"] is None
    if mode == "fb":
        assert len(result["fb_diagnostics"]) == 4  # two accumulated microbatches per update
        assert result["learning"]["diagnostic_batches_with_gradients"] == 4
    if mode == "kernel":
        assert result["kernel_unavailable"]["unavailable_reason"] == "cpu_no_cuda"


def test_incomplete_or_all_skipped_is_not_a_baseline(tmp_path, monkeypatch, config):
    config.update(warmup=1, measured=2)
    ctx = _make_ctx(tmp_path, [_batch(nan=True), _batch(nan=True)], monkeypatch)
    observer = training.BenchmarkObserver(config, "throughput", "cpu")
    loop_mod.run(ctx, observer=observer)
    result = observer.result()
    assert result["status"] == "incomplete" and result["training_it_s"]["value"] is None
    assert result["learning"]["parameters_changed"] is False


def test_public_result_does_not_dump_args_paths_or_captions(tmp_path, monkeypatch, config):
    ctx = _make_ctx(tmp_path, [_batch() for _ in range(6)], monkeypatch)
    ctx.args.private_path = ctx.args.sample_prompt = "PRIVATE_CAPTION_PATH_CANARY"
    ctx.args.output_dir = str(tmp_path / "PRIVATE_CAPTION_PATH_CANARY")
    config.update(warmup=1, measured=2)
    observer = training.BenchmarkObserver(config, "fb", "cpu")
    loop_mod.run(ctx, observer=observer)
    encoded = json.dumps(observer.result(), allow_nan=False)
    assert "PRIVATE_CAPTION_PATH_CANARY" not in encoded and str(tmp_path) not in encoded


def test_cpu_real_training_rejected_before_pipeline(tmp_path, config):
    with pytest.raises(ValueError, match="requires CUDA"):
        training.training_worker(config, SimpleNamespace(device="cpu"))


def test_training_worker_uses_single_pipeline_and_cleans_hooks(tmp_path, assets, config, monkeypatch):
    import anima_train
    work = training.create_workspace(tmp_path / "work")
    options = SimpleNamespace(device="cuda", work=work, algorithm="lora", mode="throughput", **assets)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(training, "metadata", lambda device: {"backend": "torch"})
    monkeypatch.setattr(training.sys, "addaudithook", lambda hook: None)  # never install process-wide hook in pytest
    calls = []
    def pipeline(args, *, observer):
        calls.append(args)
        assert not args.auto_install and args.config is None
        assert Path(args.data_dir).parent == work
    monkeypatch.setattr(anima_train, "run_training", pipeline)
    result = training.training_worker(config, options)
    assert len(calls) == 1
    assert result["status"] == "incomplete"  # no fabricated observer notifications
    assert (work / "training-args.private.json").exists()
    assert str(work) not in json.dumps(result)


def test_internal_worker_paths_and_private_failure(tmp_path, config, monkeypatch):
    root = tmp_path / "workspace"
    work, output = root / "private" / "run", root / "public" / "run"
    work.mkdir(parents=True)
    output.mkdir(parents=True)
    training.validate_worker_paths(work, output)
    with pytest.raises(ValueError):
        training.validate_worker_paths(bench.REPO, output)
    with pytest.raises(ValueError):
        training.validate_worker_paths(output, work)
    config_path = root / "experiment.json"
    bench.write_json(config_path, config)
    def fail(*args):
        raise ValueError("PRIVATE_EXCEPTION_PATH_CANARY")
    monkeypatch.setattr(bench, "layer_worker", fail)
    monkeypatch.setattr(bench.sys, "argv", ["benchmark_lycoris", "_worker", "--config", str(config_path),
        "--mode", "layer", "--algorithm", "lora", "--case-index", "0", "--device", "cpu",
        "--output", str(output), "--work", str(work)])
    assert bench.main() == 1
    text = (output / "result.json").read_text()
    assert "PRIVATE_EXCEPTION_PATH_CANARY" not in text
    assert json.loads(text)["training_it_s"]["unavailable_reason"] == "worker_failed"
    with pytest.raises(ValueError):
        training.validate_worker_paths(work, output)  # no reuse/overwrite
