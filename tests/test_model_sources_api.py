"""统一模型来源候选端点 + catalog 统一 shape（docs/design/model-source-unification.md）。

POST/DELETE /api/model-sources/{domain}：校验从简（D3）、去重 append、移除不
动磁盘、移除当前选中回退默认；catalog["model_sources"] 行含能力位
（removable / deletable），内置 preset 不可移除（D2）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import secrets, server
from studio.services import models as model_downloader


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """secrets.json 与 models_root 都隔离到 tmp_path。"""
    sf = tmp_path / "secrets.json"
    monkeypatch.setattr(secrets, "SECRETS_FILE", sf)
    sf.write_text(
        json.dumps({"models": {"root": str(tmp_path / "models")}}),
        encoding="utf-8",
    )
    return TestClient(server.app)


def _rows(catalog: dict, domain: str) -> list[dict]:
    return catalog["model_sources"][domain]


# ---------------------------------------------------------------------------
# POST — 添加候选
# ---------------------------------------------------------------------------


def test_add_download_candidate_appears_in_catalog(client: TestClient) -> None:
    res = client.post(
        "/api/model-sources/wd14",
        json={"kind": "download", "repo": "Custom/tagger-x"},
    )
    assert res.status_code == 200
    rows = _rows(res.json(), "wd14")
    # 内置 preset 在前（不可移除），用户候选在后
    presets = [r for r in rows if r["kind"] == "preset"]
    assert [r["value"] for r in presets] == list(secrets.DEFAULT_WD14_MODELS)
    assert all(not r["removable"] and r["deletable"] for r in presets)
    user = [r for r in rows if r["kind"] == "download"]
    assert [r["value"] for r in user] == ["Custom/tagger-x"]
    assert user[0]["removable"] and user[0]["deletable"]
    assert user[0]["download_id"] == "wd14"
    assert user[0]["status_key"] == "wd14:Custom/tagger-x"


def test_add_download_candidate_is_idempotent(client: TestClient) -> None:
    for _ in range(2):
        client.post(
            "/api/model-sources/eval_clip",
            json={"kind": "download", "repo": "laion/CLIP-ViT-H-14"},
        )
    rows = [
        r for r in _rows(
            client.get("/api/models/catalog").json(), "eval_clip")
        if r["kind"] == "download"
    ]
    assert len(rows) == 1


def test_add_rejects_bad_repo_id(client: TestClient) -> None:
    res = client.post(
        "/api/model-sources/wd14",
        json={"kind": "download", "repo": "not-a-repo-id"},
    )
    assert res.status_code == 400


def test_add_rejects_unknown_domain(client: TestClient) -> None:
    res = client.post(
        "/api/model-sources/nonsense",
        json={"kind": "download", "repo": "a/b"},
    )
    assert res.status_code == 400


def test_add_rejects_filename_for_directory_domain(client: TestClient) -> None:
    res = client.post(
        "/api/model-sources/wd14",
        json={"kind": "download", "repo": "a/b", "filename": "x.onnx"},
    )
    assert res.status_code == 400


def test_add_upscaler_download_requires_filename(client: TestClient) -> None:
    res = client.post(
        "/api/model-sources/upscaler",
        json={"kind": "download", "repo": "Kim2091/UltraSharp"},
    )
    assert res.status_code == 400
    res = client.post(
        "/api/model-sources/upscaler",
        json={
            "kind": "download", "repo": "Kim2091/UltraSharp",
            "filename": "4x-UltraSharp.pth",
        },
    )
    assert res.status_code == 200


def test_add_local_wd14_requires_both_files(
    client: TestClient, tmp_path: Path
) -> None:
    d = tmp_path / "wd14-local"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"onnx")
    res = client.post(
        "/api/model-sources/wd14", json={"kind": "local", "path": str(d)},
    )
    assert res.status_code == 400  # 缺 selected_tags.csv

    (d / "selected_tags.csv").write_text("tag", encoding="utf-8")
    res = client.post(
        "/api/model-sources/wd14", json={"kind": "local", "path": str(d)},
    )
    assert res.status_code == 200
    local = [r for r in _rows(res.json(), "wd14") if r["kind"] == "local"]
    assert len(local) == 1
    assert local[0]["exists"] is True
    assert local[0]["removable"] is True
    assert local[0]["deletable"] is False   # 本地文件永不从 UI 删除
    assert local[0]["download_id"] is None


def test_add_local_rejects_relative_path(client: TestClient) -> None:
    res = client.post(
        "/api/model-sources/wd14",
        json={"kind": "local", "path": "relative/dir"},
    )
    assert res.status_code == 400


def test_add_local_family_model_requires_safetensors(
    client: TestClient, tmp_path: Path
) -> None:
    f = tmp_path / "ft.safetensors"
    f.write_bytes(b"w")
    res = client.post(
        "/api/model-sources/anima", json={"kind": "local", "path": str(f)},
    )
    assert res.status_code == 200
    # 同步进兼容面（旧端点数据模型）：models.custom
    assert str(f) in secrets.load().models.custom.get("anima", [])


# ---------------------------------------------------------------------------
# DELETE — 移除候选
# ---------------------------------------------------------------------------


def test_remove_candidate_keeps_file_on_disk(
    client: TestClient, tmp_path: Path
) -> None:
    d = tmp_path / "wd14-local"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"onnx")
    (d / "selected_tags.csv").write_text("tag", encoding="utf-8")
    client.post("/api/model-sources/wd14", json={"kind": "local", "path": str(d)})

    res = client.request(
        "DELETE", "/api/model-sources/wd14",
        json={"kind": "local", "path": str(d)},
    )
    assert res.status_code == 200
    assert not [r for r in _rows(res.json(), "wd14") if r["kind"] == "local"]
    # 移除 ≠ 删除：磁盘文件原封不动
    assert (d / "model.onnx").exists()


def test_remove_current_selection_falls_back_to_default(
    client: TestClient,
) -> None:
    client.post(
        "/api/model-sources/wd14",
        json={"kind": "download", "repo": "Custom/tagger-x"},
    )
    secrets.update({"wd14": {"model_id": "Custom/tagger-x"}})
    client.request(
        "DELETE", "/api/model-sources/wd14",
        json={"kind": "download", "repo": "Custom/tagger-x"},
    )
    s = secrets.load()
    assert s.wd14.model_id == secrets.DEFAULT_WD14_MODELS[0]
    assert all(
        c.repo != "Custom/tagger-x" for c in s.model_sources.get("wd14", [])
    )


def test_remove_eval_current_falls_back_to_schema_default(
    client: TestClient,
) -> None:
    client.post(
        "/api/model-sources/eval_dino",
        json={"kind": "download", "repo": "facebook/dinov2-base"},
    )
    secrets.update({"eval_metrics": {"dino_model_name": "facebook/dinov2-base"}})
    client.request(
        "DELETE", "/api/model-sources/eval_dino",
        json={"kind": "download", "repo": "facebook/dinov2-base"},
    )
    assert secrets.load().eval_metrics.dino_model_name == "facebook/dinov2-small"


def test_remove_missing_candidate_is_noop(client: TestClient) -> None:
    res = client.request(
        "DELETE", "/api/model-sources/wd14",
        json={"kind": "download", "repo": "Never/added"},
    )
    assert res.status_code == 200


# ---------------------------------------------------------------------------
# catalog 统一 shape — eval 三域
# ---------------------------------------------------------------------------


def test_eval_domains_have_preset_row(client: TestClient) -> None:
    catalog = client.get("/api/models/catalog").json()
    for domain, default in (
        ("eval_clip", "openai/clip-vit-base-patch32"),
        ("eval_dino", "facebook/dinov2-small"),
        ("eval_ccip", "ccip-caformer-24-randaug-pruned"),
    ):
        rows = _rows(catalog, domain)
        presets = [r for r in rows if r["kind"] == "preset"]
        assert [r["value"] for r in presets] == [default]
        assert presets[0]["is_current"] is True
        assert presets[0]["removable"] is False


# ---------------------------------------------------------------------------
# 加载器绝对路径支持（local 候选选中值 = 绝对路径）
# ---------------------------------------------------------------------------


def test_wd14_target_dir_accepts_absolute_path(tmp_path: Path) -> None:
    d = tmp_path / "anywhere" / "wd14"
    assert model_downloader.wd14_target_dir(tmp_path / "root", str(d)) == d


def test_eval_target_dir_accepts_absolute_path(tmp_path: Path) -> None:
    d = tmp_path / "dino-local"
    assert (
        model_downloader.eval_model_target_dir(tmp_path / "root", "dino", str(d))
        == d
    )


def test_upscaler_target_accepts_absolute_path(tmp_path: Path) -> None:
    f = tmp_path / "up" / "4x-custom.pth"
    assert model_downloader.upscaler_target(str(f), tmp_path / "root") == f


def test_upscaler_target_absolute_path_requires_known_ext(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        model_downloader.upscaler_target(str(tmp_path / "x.exe"), tmp_path)


# ---------------------------------------------------------------------------
# catalog 统一 shape — upscaler / 主模型族
# ---------------------------------------------------------------------------


def test_upscaler_rows_presets_and_download_candidate(
    client: TestClient,
) -> None:
    client.post(
        "/api/model-sources/upscaler",
        json={
            "kind": "download", "repo": "Kim2091/UltraSharp",
            "filename": "4x-UltraSharp.pth",
        },
    )
    rows = _rows(client.get("/api/models/catalog").json(), "upscaler")
    presets = [r for r in rows if r["kind"] == "preset"]
    assert presets and all(not r["removable"] for r in presets)
    dl = [r for r in rows if r["kind"] == "download"]
    assert len(dl) == 1
    # value = 文件名（selected_upscaler 语义）；status_key 与扫盘行同格式
    assert dl[0]["value"] == "4x-UltraSharp.pth"
    assert dl[0]["download_id"] == "upscaler_custom"
    assert dl[0]["status_key"] == "upscaler:custom:4x-UltraSharp.pth"
    assert dl[0]["description"] == "Kim2091/UltraSharp"
    assert dl[0]["candidate"]["repo"] == "Kim2091/UltraSharp"


def test_upscaler_scanned_rows_exclude_registered_candidates(
    client: TestClient, tmp_path: Path,
) -> None:
    up_dir = tmp_path / "models" / "upscalers"
    up_dir.mkdir(parents=True)
    (up_dir / "manual-drop.pth").write_bytes(b"w")
    (up_dir / "4x-UltraSharp.pth").write_bytes(b"w")
    client.post(
        "/api/model-sources/upscaler",
        json={
            "kind": "download", "repo": "Kim2091/UltraSharp",
            "filename": "4x-UltraSharp.pth",
        },
    )
    rows = _rows(client.get("/api/models/catalog").json(), "upscaler")
    scanned = [r["value"] for r in rows if r["kind"] == "scanned"]
    # 手放的文件被扫出；已登记为 download 候选的文件不重复出现
    assert scanned == ["manual-drop.pth"]
    dl = [r for r in rows if r["kind"] == "download"]
    assert dl[0]["exists"] is True


def test_family_rows_download_candidate_value_is_target_path(
    client: TestClient, tmp_path: Path,
) -> None:
    client.post(
        "/api/model-sources/anima",
        json={
            "kind": "download", "repo": "author/finetune",
            "filename": "my-finetune.safetensors",
        },
    )
    rows = _rows(client.get("/api/models/catalog").json(), "anima")
    presets = [r for r in rows if r["kind"] == "preset"]
    assert presets and presets[0]["download_id"] == "anima_main"
    dl = [r for r in rows if r["kind"] == "download"]
    assert len(dl) == 1
    # value = 落盘绝对路径（selected 已支持路径语义），下载 variant = repo 内路径
    expected = str(tmp_path / "models" / "diffusion_models" / "my-finetune.safetensors")
    assert dl[0]["value"] == expected
    assert dl[0]["download_id"] == "anima_custom"
    assert dl[0]["download_variant"] == "my-finetune.safetensors"


def test_trigger_family_custom_requires_registered_candidate() -> None:
    with pytest.raises(ValueError):
        model_downloader.trigger("anima_custom", "not-registered.safetensors")
