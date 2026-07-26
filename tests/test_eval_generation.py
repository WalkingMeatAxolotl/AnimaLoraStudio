"""评估出图走 daemon（`eval_generation`）。

核心不变量：
- 一次评估**一个** daemon 实例（底模只加载一次），不是每候选一个
- 一个候选一个 daemon task，`prompts` = 全部验证图 caption，`count=1`
- 图按 `item["filename"]` 落到 run 的 images/，不进测试页的 generate_cache
- baseline（lora_scale=0）不给 lora_configs，走纯底模
- 逐图 started/done/error 事件驱动 item 状态；daemon 报 error 整个候选失败
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Callable

import pytest

from studio import db, secrets
from studio.infrastructure import paths as infra_paths
from studio.services import eval_generation, eval_samples
from studio.services.projects import projects, versions

_PNG = b"\x89PNG\r\n\x1a\nfake"


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(infra_paths, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    return {"db": dbfile}


def _setup(isolated, *, validation: int = 2):
    with db.connection_for(isolated["db"]) as conn:
        project = projects.create_project(conn, title="Gen")
        version = versions.create_version(conn, project_id=project["id"], label="v1")
    vdir = versions.version_dir(project["id"], project["slug"], version["label"])
    val = vdir / "validation" / "1_data"
    val.mkdir(parents=True, exist_ok=True)
    for i in range(validation):
        (val / f"v{i}.png").write_bytes(b"png")
        (val / f"v{i}.txt").write_text(f"caption {i}", encoding="utf-8")
    out = vdir / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_epoch1.safetensors").write_bytes(b"lora")
    _write_version_config(vdir)
    return project, version, vdir


def _write_version_config(vdir: Path) -> None:
    """version config 是 daemon config 的模型路径 / 采样参数来源。"""
    import json

    cfg = {
        "model_family": "anima",
        "transformer_path": "models/diffusion_models/anima.safetensors",
        "vae_path": "models/vae/qwen.safetensors",
        "text_encoder_path": "models/text_encoders",
        "t5_tokenizer_path": "models/t5_tokenizer",
        "resolution": 1024,
        "sample_infer_steps": 20,
        "sample_cfg_scale": 5.0,
        "sample_sampler_name": "er_sde",
        "sample_scheduler": "simple",
        "attention_backend": "flash_attn",
        "mixed_precision": "bf16",
    }
    path = vdir / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")


@pytest.fixture(autouse=True)
def _stub_version_config(monkeypatch: pytest.MonkeyPatch):
    """read_version_config 走真实实现要一整套 schema；这里只需要它回一份 dict。"""
    import json

    from studio.services import version_config

    def _read(project, version):
        vdir = versions.version_dir(
            int(project["id"]), str(project.get("slug") or ""), str(version["label"]),
        )
        return json.loads((vdir / "config.json").read_text(encoding="utf-8"))

    monkeypatch.setattr(version_config, "read_version_config", _read)


class FakeDaemon:
    """按 daemon 协议回放事件的假 daemon。记录每次 submit 的 config。"""

    def __init__(self, *, fail_at: int | None = None, task_error: str | None = None):
        self.started = 0
        self.stopped = 0
        self.configs: list[dict[str, Any]] = []
        self._fail_at = fail_at
        self._task_error = task_error

    def start(self) -> None:
        self.started += 1

    def stop(self, timeout: float = 10.0) -> None:
        self.stopped += 1

    def submit_task(
        self, *, task_id: int, config: dict[str, Any], output_dir: str,
        on_event: Callable[[dict[str, Any]], None],
    ) -> str:
        self.configs.append(config)
        n = len(config["prompts"])
        on_event({"kind": "started", "task_id": task_id, "total": n})
        for i in range(n):
            on_event({"kind": "image_started", "batch_idx": i, "batch_total": n})
            if self._fail_at == i:
                on_event({"kind": "image_error", "step": i + 1, "message": "cuda oom"})
                continue
            on_event({
                "kind": "image_done", "step": i + 1, "total": n,
                "filename": f"gen_{i:04d}_p{i}_c0_s7.png",
                "image_b64": base64.b64encode(_PNG).decode("ascii"),
                "byte_size": len(_PNG),
            })
        if self._task_error:
            on_event({"kind": "error", "message": self._task_error})
        else:
            on_event({"kind": "done", "task_id": task_id})
        return "req-1"


def _run_for(isolated, *, baseline: bool = False):
    project, version, vdir = _setup(isolated)
    run = eval_samples.create_run(
        project, version, vdir,
        checkpoint_path=str(vdir / "output" / "model_epoch1.safetensors"),
        baseline=baseline, now=1000.0,
    )
    return project, version, vdir, run


# ---------------------------------------------------------------------------
# 出图 + 落盘
# ---------------------------------------------------------------------------

def test_generates_every_validation_image_into_the_run(isolated) -> None:
    _, _, vdir, run = _run_for(isolated)
    fake = FakeDaemon()

    with eval_generation.DaemonSampleGenerator(
        lambda _l: None, task_id=5, daemon=fake,
    ) as generate:
        generate(run, vdir, lambda _l: None)

    saved = eval_samples.load_run(vdir, run["run_id"])
    assert [i["status"] for i in saved["items"]] == ["done", "done"]
    for item in saved["items"]:
        # 图落在 item 计划好的文件名上（daemon 自己的 gen_*.png 命名不外泄 ——
        # 指标 runner 和样图矩阵认的是 item["filename"]）
        assert (vdir / item["path"]).read_bytes() == _PNG


def test_one_task_per_candidate_with_all_prompts(isolated) -> None:
    """一个候选 = 一个 task，prompts 是全部验证图 —— 这是底模只加载一次的前提。"""
    _, _, vdir, run = _run_for(isolated)
    fake = FakeDaemon()

    with eval_generation.DaemonSampleGenerator(
        lambda _l: None, daemon=fake,
    ) as generate:
        generate(run, vdir, lambda _l: None)

    assert len(fake.configs) == 1
    cfg = fake.configs[0]
    assert cfg["prompts"] == ["caption 0", "caption 1"]
    assert cfg["count"] == 1  # 多张会打乱 prompt ↔ item 的一一对应


def test_daemon_started_once_across_candidates(isolated) -> None:
    """跨候选复用同一个 daemon —— 这一刀的全部收益所在。"""
    _, _, vdir, run = _run_for(isolated)
    fake = FakeDaemon()

    with eval_generation.DaemonSampleGenerator(
        lambda _l: None, daemon=fake,
    ) as generate:
        for _ in range(3):
            generate(run, vdir, lambda _l: None)

    assert fake.started == 1
    assert len(fake.configs) == 3


def test_owns_daemon_lifecycle_only_when_it_created_it(isolated) -> None:
    """外部注入的 daemon 不由本类关闭（测试 / 未来共享实例场景）。"""
    _, _, vdir, _run = _run_for(isolated)
    fake = FakeDaemon()
    with eval_generation.DaemonSampleGenerator(lambda _l: None, daemon=fake):
        pass
    assert fake.stopped == 0


# ---------------------------------------------------------------------------
# config 组装
# ---------------------------------------------------------------------------

def test_generation_params_come_from_the_frozen_plan(isolated) -> None:
    _, _, vdir, run = _run_for(isolated)
    cfg = eval_generation.build_daemon_config(
        run, vdir, output_dir=vdir / "out",
    )
    assert cfg["steps"] == 20
    assert cfg["cfg_scale"] == 5.0
    assert cfg["sampler_name"] == "er_sde"
    assert cfg["width"] == 1024 and cfg["height"] == 1024
    assert cfg["model_family"] == "anima"
    # 评估不需要中间预览，关掉省 b64 编码和管道带宽
    assert cfg["preview_every_n_steps"] == 0


def test_checkpoint_becomes_a_single_lora_config(isolated) -> None:
    _, _, vdir, run = _run_for(isolated)
    cfg = eval_generation.build_daemon_config(run, vdir, output_dir=vdir / "out")
    assert len(cfg["lora_configs"]) == 1
    assert cfg["lora_configs"][0]["path"].endswith("model_epoch1.safetensors")
    assert cfg["lora_configs"][0]["scale"] == 1.0


def test_baseline_runs_with_no_lora_at_all(isolated) -> None:
    """baseline 是纯底模对照。旧路径靠 lora_scale=0 绕，现在干脆不挂 LoRA。"""
    _, _, vdir, run = _run_for(isolated, baseline=True)
    cfg = eval_generation.build_daemon_config(run, vdir, output_dir=vdir / "out")
    assert cfg["lora_configs"] == []


def test_empty_run_is_rejected(isolated) -> None:
    _, _, vdir, run = _run_for(isolated)
    run["items"] = []
    with pytest.raises(eval_generation.EvalGenerationError):
        eval_generation.build_daemon_config(run, vdir, output_dir=vdir / "out")


# ---------------------------------------------------------------------------
# 失败路径
# ---------------------------------------------------------------------------

def test_single_image_failure_marks_only_that_item(isolated) -> None:
    """一张图崩了不该拖垮整个候选 —— 其余图仍值得留着算指标。"""
    _, _, vdir, run = _run_for(isolated)
    fake = FakeDaemon(fail_at=0)

    with eval_generation.DaemonSampleGenerator(
        lambda _l: None, daemon=fake,
    ) as generate:
        generate(run, vdir, lambda _l: None)

    saved = eval_samples.load_run(vdir, run["run_id"])
    assert saved["items"][0]["status"] == "failed"
    assert "cuda oom" in str(saved["items"][0]["error"])
    assert saved["items"][1]["status"] == "done"


def test_task_level_error_raises(isolated) -> None:
    """daemon 报 task 级 error（模型加载失败等）→ 抛出去，让候选标 failed。"""
    _, _, vdir, run = _run_for(isolated)
    fake = FakeDaemon(task_error="model load failed")

    with eval_generation.DaemonSampleGenerator(
        lambda _l: None, daemon=fake,
    ) as generate:
        with pytest.raises(eval_generation.EvalGenerationError, match="model load failed"):
            generate(run, vdir, lambda _l: None)


def test_missing_image_bytes_is_an_error_not_a_silent_skip(isolated) -> None:
    """image_b64 被剥掉说明 daemon 的 cache_images 忘了关 —— 必须炸，不能静默丢图。"""
    _, _, vdir, run = _run_for(isolated)

    class _Stripped(FakeDaemon):
        def submit_task(self, *, task_id, config, output_dir, on_event):
            self.configs.append(config)
            on_event({"kind": "image_started", "batch_idx": 0, "batch_total": 1})
            on_event({"kind": "image_done", "step": 1, "filename": "gen_0000.png"})
            on_event({"kind": "done", "task_id": task_id})
            return "req-1"

    with eval_generation.DaemonSampleGenerator(
        lambda _l: None, daemon=_Stripped(),
    ) as generate:
        with pytest.raises(eval_generation.EvalGenerationError, match="cache_images"):
            generate(run, vdir, lambda _l: None)


# ---------------------------------------------------------------------------
# 族条件旋钮不能跨族污染
# ---------------------------------------------------------------------------

def _set_generate_settings(monkeypatch: pytest.MonkeyPatch, **values: Any) -> None:
    class _Gen:
        pass

    gen = _Gen()
    for k, v in values.items():
        setattr(gen, k, v)

    class _Secrets:
        generate = gen

    monkeypatch.setattr(secrets, "load", lambda: _Secrets())


def test_block_swap_is_dropped_for_families_without_the_capability(
    isolated, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """回归：全局出图设置里的 blocks_to_swap 是为 krea2 调的，anima 不支持。

    测试出图跑的是用户在那儿选的模型，评估跑的是这个 version 训练时那个底模 ——
    两者的族可以不同。原样透传会让 daemon fail-fast（`model_family='anima' 不支持
    block swap`），整个出图阶段每个候选都崩，最后 8 个候选全 failed。
    """
    _, _, vdir, run = _run_for(isolated)
    _set_generate_settings(monkeypatch, blocks_to_swap=14, vram_policy="save_vram")

    cfg = eval_generation.build_daemon_config(run, vdir, output_dir=vdir / "out")

    assert cfg["model_family"] == "anima"
    assert cfg["blocks_to_swap"] == 0
    # 非族条件的旋钮照常继承
    assert cfg["vram_policy"] == "save_vram"


def test_block_swap_survives_for_a_family_that_supports_it(
    isolated, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """门控只砍不支持的族 —— krea2 项目仍该拿到用户调好的层数。"""
    import json

    _, _, vdir, run = _run_for(isolated)
    cfg_path = vdir / "config.json"
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    raw["model_family"] = "krea2"
    cfg_path.write_text(json.dumps(raw), encoding="utf-8")
    _set_generate_settings(monkeypatch, blocks_to_swap=14)

    cfg = eval_generation.build_daemon_config(run, vdir, output_dir=vdir / "out")
    assert cfg["blocks_to_swap"] == 14


def test_unreadable_settings_fall_back_to_safe_defaults(
    isolated, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """设置文件坏了不该让评估跑不起来 —— 给保守默认（含 block swap 关闭）。"""
    def _boom():
        raise RuntimeError("secrets.json 损坏")

    monkeypatch.setattr(secrets, "load", _boom)
    _, _, vdir, run = _run_for(isolated)

    cfg = eval_generation.build_daemon_config(run, vdir, output_dir=vdir / "out")
    assert cfg["blocks_to_swap"] == 0
    assert cfg["vram_policy"] == "auto"
