"""CCIP-I eval metric runner（anime 角色身份；ONNX 数值靠真实模型，这里测生命周期）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from studio import db, secrets, server
from studio.services import eval_ccip, eval_metrics, eval_samples
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.supervisor import GPU_BOUND_JOB_KINDS


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    return {"db": dbfile}


def _new_project(isolated) -> tuple[dict[str, Any], dict[str, Any], Path]:
    with db.connection_for(isolated["db"]) as conn:
        project = projects.create_project(conn, title="Eval CCIP")
        version = versions.create_version(conn, project_id=project["id"], label="baseline")
    vdir = versions.version_dir(project["id"], project["slug"], version["label"])
    return project, version, vdir


def _seed(vdir: Path) -> None:
    val = vdir / "validation" / "1_data"
    val.mkdir(parents=True, exist_ok=True)
    (val / "a.png").write_bytes(b"png-a")
    (val / "a.txt").write_text("1girl, solo", encoding="utf-8")
    output = vdir / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_step100.safetensors").write_bytes(b"fake-lora")


def _fake_generator(run: dict[str, Any], version_dir: Path, progress) -> None:
    for idx, item in enumerate(run["items"]):
        run = eval_samples.mark_item_running(version_dir, run, idx)
        path = eval_samples.sample_image_path(version_dir, run["run_id"], item["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PNG")
        run = eval_samples.mark_item_done(version_dir, run, idx)


def _sample_run(project, version, vdir) -> dict[str, Any]:
    _seed(vdir)
    run = eval_samples.create_run(
        project, version, vdir, checkpoint_path="model_step100.safetensors", now=2000.0,
    )
    return eval_samples.run_sample_job(
        project, version, vdir, run["run_id"], generator=_fake_generator,
    )


# ---------------------------------------------------------------------------


def test_start_eval_ccip_job_marks_ccip_pending(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)
    with db.connection_for(isolated["db"]) as conn:
        job, result = eval_ccip.start_job(
            conn, project, version, vdir, run["run_id"], model_name="ccip-x",
        )
    assert job["kind"] == "eval_ccip"
    assert job["params_decoded"]["model_name"] == "ccip-x"
    assert result["metric_states"]["ccip_i"]["status"] == "pending"
    assert result["metric_states"]["dino_i"]["status"] == "not_run"


def test_run_ccip_job_with_injected_scorer(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)
    eval_metrics.save_result(vdir, run["run_id"], {"metrics": {"clip_t": 0.5}}, now=3000.0)

    result = eval_ccip.run_ccip_job(
        project, version, vdir, run["run_id"],
        scorer=lambda _r, _v, _m, _p: {"ccip_i": 0.75, "ccip_i_count": 4},
        model_name="ccip-x",
    )
    assert result["metrics"]["clip_t"] == 0.5  # 保留其它指标
    assert result["metrics"]["ccip_i"] == 0.75
    assert result["metric_states"]["ccip_i"]["status"] == "done"
    assert result["metric_states"]["ccip_i"]["count"] == 4


def test_run_ccip_job_unavailable_when_no_pairs(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)
    result = eval_ccip.run_ccip_job(
        project, version, vdir, run["run_id"],
        scorer=lambda _r, _v, _m, _p: {"ccip_i": None},
    )
    assert result["metric_states"]["ccip_i"]["status"] == "unavailable"
    assert "ccip_i" not in result["metrics"]


def test_ccip_preprocess_shape_and_norm() -> None:
    import numpy as np
    from PIL import Image

    p = Path(__file__).parent / "_ccip_tmp.png"
    Image.new("RGB", (10, 10), (255, 0, 0)).save(p)
    try:
        arr = eval_ccip._ccip_preprocess(p)
        assert arr.shape == (3, 384, 384)
        assert arr.dtype == np.float32
        # 红通道归一后 = (1-mean)/std，绿/蓝 = (0-mean)/std —— 通道值各自常数
        assert np.allclose(arr[0], (1.0 - 0.48145466) / 0.26862954, atol=1e-4)
        assert np.allclose(arr[1], (0.0 - 0.4578275) / 0.26130258, atol=1e-4)
    finally:
        p.unlink(missing_ok=True)


def test_eval_ccip_job_kind_gpu_bound() -> None:
    assert "eval_ccip" in GPU_BOUND_JOB_KINDS
