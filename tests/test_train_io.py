"""PP7 — 训练集导出 / 导入 round-trip + 边界。"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from studio import db
from studio.domain.errors import NotFoundError
from studio.services.projects import projects, versions
from studio.services.data_io import train_io


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    pdir = tmp_path / "projects"
    monkeypatch.setattr(projects, "PROJECTS_DIR", pdir)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    return {"db": dbfile, "tmp": tmp_path}


def _make_project_with_train(
    isolated, *, title: str = "Cosmic Kaguya", label: str = "v1"
) -> tuple[dict, dict, Path]:
    with db.connection_for(isolated["db"]) as conn:
        p = projects.create_project(conn, title=title)
        v = versions.create_version(conn, project_id=p["id"], label=label)
    train = versions.version_dir(p["id"], p["slug"], v["label"]) / "train"
    return p, v, train


def _png(content: bytes = b"fake-png") -> bytes:
    # 只是占位，IMAGE_EXTS 按后缀判定不读 magic
    return content


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_export_round_trip(isolated, tmp_path: Path) -> None:
    p, v, train = _make_project_with_train(isolated)
    # 默认已有 1_data；放两张 + 1 个 caption；再加一个 2_concept 子文件夹
    (train / "1_data").mkdir(parents=True, exist_ok=True)
    (train / "1_data" / "a.png").write_bytes(_png())
    (train / "1_data" / "a.txt").write_text("1girl, solo", encoding="utf-8")
    (train / "1_data" / "b.png").write_bytes(_png(b"b"))
    (train / "2_concept").mkdir()
    (train / "2_concept" / "c.png").write_bytes(_png(b"c"))
    (train / "2_concept" / "c.txt").write_text("2girls", encoding="utf-8")

    dest = tmp_path / "out.zip"
    with db.connection_for(isolated["db"]) as conn:
        result = train_io.export_train(conn, v["id"], dest)

    assert dest.exists()
    assert result["manifest"]["stats"]["image_count"] == 3
    assert result["manifest"]["stats"]["tagged_count"] == 2
    assert {c["folder"] for c in result["manifest"]["stats"]["concepts"]} == {
        "1_data",
        "2_concept",
    }

    with zipfile.ZipFile(dest) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "train/1_data/a.png" in names
        assert "train/1_data/a.txt" in names
        assert "train/1_data/b.png" in names
        assert "train/2_concept/c.png" in names

        # round-trip: 重新导入应该得到一个全新项目，stage=tagging
        new_zip = tmp_path / "round.zip"
    # 重新打开（上面 with 已关）
    import shutil

    shutil.copy(dest, new_zip)
    with db.connection_for(isolated["db"]) as conn:
        imported = train_io.import_train(conn, new_zip)

    assert imported["project"]["id"] != p["id"]
    assert imported["version"]["label"] == "v1"
    # ADR-0007 PR-5: import 不再推 stage
    assert imported["stats"]["image_count"] == 3
    assert imported["stats"]["tagged_count"] == 2

    new_train = versions.version_dir(
        imported["project"]["id"],
        imported["project"]["slug"],
        imported["version"]["label"],
    ) / "train"
    assert (new_train / "1_data" / "a.png").exists()
    assert (new_train / "1_data" / "a.txt").read_text(encoding="utf-8") == "1girl, solo"
    assert (new_train / "2_concept" / "c.png").exists()


def test_export_empty_train_raises(isolated, tmp_path: Path) -> None:
    p, v, train = _make_project_with_train(isolated)
    # 默认 1_data 是空的；不放任何图
    dest = tmp_path / "empty.zip"
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.export_train(conn, v["id"], dest)


def test_export_missing_version(isolated, tmp_path: Path) -> None:
    with db.connection_for(isolated["db"]) as conn, pytest.raises(
        NotFoundError, match="Version not found"
    ):
        train_io.export_train(conn, 9999, tmp_path / "x.zip")


# ---------------------------------------------------------------------------
# import — 边界
# ---------------------------------------------------------------------------


def _make_zip(tmp: Path, files: dict[str, bytes], manifest: dict | None = None) -> Path:
    p = tmp / f"in-{len(list(tmp.iterdir()))}.zip"
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_STORED) as zf:
        if manifest is not None:
            zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False))
        for name, data in files.items():
            zf.writestr(name, data)
    return p


def _basic_manifest(title: str = "Imported Project") -> dict:
    return {
        "schema_version": 1,
        "exported_at": 0,
        "source": {"title": title, "version_label": "v1", "slug": "imported-project"},
        "stats": {},
    }


def test_import_rejects_zip_slip(isolated, tmp_path: Path) -> None:
    zp = _make_zip(
        tmp_path,
        {"train/../escape.png": b"x"},
        manifest=_basic_manifest(),
    )
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.import_train(conn, zp)


def test_import_rejects_absolute_path(isolated, tmp_path: Path) -> None:
    zp = _make_zip(
        tmp_path,
        {"/etc/evil.png": b"x"},
        manifest=_basic_manifest(),
    )
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.import_train(conn, zp)


def test_import_rejects_non_train_prefix(isolated, tmp_path: Path) -> None:
    zp = _make_zip(
        tmp_path,
        {"download/x.png": b"x"},
        manifest=_basic_manifest(),
    )
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.import_train(conn, zp)


def test_import_rejects_deep_nesting(isolated, tmp_path: Path) -> None:
    zp = _make_zip(
        tmp_path,
        {"train/1_data/sub/x.png": b"x"},
        manifest=_basic_manifest(),
    )
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.import_train(conn, zp)


def test_import_rejects_missing_manifest(isolated, tmp_path: Path) -> None:
    zp = _make_zip(tmp_path, {"train/1_data/x.png": b"x"})
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.import_train(conn, zp)


def test_import_rejects_empty(isolated, tmp_path: Path) -> None:
    zp = _make_zip(tmp_path, {}, manifest=_basic_manifest())
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.import_train(conn, zp)


def test_import_slug_conflict_appends_suffix(isolated, tmp_path: Path) -> None:
    """同 slug 已存在 → 自动加 -imported-{ts} 后缀。"""
    # 先建一个 slug=cosmic-kaguya 的项目占位
    with db.connection_for(isolated["db"]) as conn:
        projects.create_project(conn, title="Cosmic Kaguya")  # slug=cosmic-kaguya

    zp = _make_zip(
        tmp_path,
        {"train/1_data/a.png": b"x"},
        manifest={
            "schema_version": 1,
            "exported_at": 0,
            "source": {"title": "Cosmic Kaguya", "version_label": "v1", "slug": "cosmic-kaguya"},
            "stats": {},
        },
    )
    with db.connection_for(isolated["db"]) as conn:
        result = train_io.import_train(conn, zp)
    assert result["project"]["slug"].startswith("cosmic-kaguya-imported-")


def test_import_keeps_concept_folder_names(isolated, tmp_path: Path) -> None:
    zp = _make_zip(
        tmp_path,
        {
            "train/3_chara/a.png": b"x",
            "train/3_chara/a.txt": b"tag1",
            "train/5_style/b.png": b"y",
        },
        manifest=_basic_manifest("Multi Concept"),
    )
    with db.connection_for(isolated["db"]) as conn:
        result = train_io.import_train(conn, zp)
    new_train = versions.version_dir(
        result["project"]["id"],
        result["project"]["slug"],
        result["version"]["label"],
    ) / "train"
    assert (new_train / "3_chara" / "a.png").exists()
    assert (new_train / "5_style" / "b.png").exists()
    assert result["stats"]["tagged_count"] == 1
    assert result["stats"]["image_count"] == 2


def test_import_bad_zip(isolated, tmp_path: Path) -> None:
    bad = tmp_path / "not.zip"
    bad.write_bytes(b"not a zip file at all")
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.import_train(conn, bad)


# ---------------------------------------------------------------------------
# bundle import: 4 全局模型路径字段跨机器处理
# ---------------------------------------------------------------------------


def test_export_bundle_records_version_and_preset_names(isolated, tmp_path: Path) -> None:
    p, v, train = _make_project_with_train(isolated, label="anime-v2")
    (train / "1_data").mkdir(parents=True, exist_ok=True)
    (train / "1_data" / "a.png").write_bytes(_png())

    dest = tmp_path / "out.bundle.zip"
    with db.connection_for(isolated["db"]) as conn:
        versions.update_version(conn, v["id"], config_name="style_preset")
        result = train_io.export_bundle(
            conn,
            v["id"],
            dest,
            train_io.BundleOptions(train=True, train_captions=True),
        )

    source = result["manifest"]["source"]
    assert source["version_label"] == "anime-v2"
    assert source["preset_name"] == "style_preset"

    with zipfile.ZipFile(dest) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["source"]["version_label"] == "anime-v2"
    assert manifest["source"]["preset_name"] == "style_preset"


def test_export_bundle_training_caches_roundtrip(isolated, tmp_path: Path) -> None:
    """cache=True：train/reg 的 latent + text cache 一并打包。

    导入后 npz mtime ≥ 图片 mtime（_is_cache_valid 的不变量，否则缓存白搬）；
    text cache 只按文件内指纹失效，无 mtime 要求。
    """
    p, v, train = _make_project_with_train(isolated)
    (train / "1_data").mkdir(parents=True, exist_ok=True)
    (train / "1_data" / "a.png").write_bytes(_png())
    (train / "1_data" / "a.npz").write_bytes(b"fake-latent")
    (train / "1_data" / "a.r512.npz").write_bytes(b"fake-latent-512")
    (train / "1_data" / "a.png.text.safetensors").write_bytes(b"fake-text")
    reg = train.parent / "reg" / "girl"
    reg.mkdir(parents=True)
    (reg / "r.png").write_bytes(_png(b"r"))
    (reg / "r.npz").write_bytes(b"fake-reg-latent")
    (reg / "r.png.text.safetensors").write_bytes(b"fake-reg-text")

    dest = tmp_path / "cache.bundle.zip"
    with db.connection_for(isolated["db"]) as conn:
        result = train_io.export_bundle(
            conn, v["id"], dest,
            train_io.BundleOptions(
                train=True, reg=True,
                train_latent_cache=True, reg_latent_cache=True,
            ),
        )

    assert result["manifest"]["includes"]["train_latent_cache"] is True
    assert result["manifest"]["includes"]["reg_latent_cache"] is True
    assert result["manifest"]["stats"]["latent_cache_count"] == 3
    assert result["manifest"]["stats"]["text_cache_count"] == 2
    with zipfile.ZipFile(dest) as zf:
        names = set(zf.namelist())
    assert "train/1_data/a.npz" in names
    assert "train/1_data/a.r512.npz" in names
    assert "reg/girl/r.npz" in names
    assert "train/1_data/a.png.text.safetensors" in names
    assert "reg/girl/r.png.text.safetensors" in names

    with db.connection_for(isolated["db"]) as conn:
        imported = train_io.import_bundle(conn, dest, presets_base=tmp_path / "presets")

    new_dir = versions.version_dir(
        imported["project"]["id"],
        imported["project"]["slug"],
        imported["version"]["label"],
    )
    npz = new_dir / "train" / "1_data" / "a.npz"
    img = new_dir / "train" / "1_data" / "a.png"
    reg_npz = new_dir / "reg" / "girl" / "r.npz"
    reg_img = new_dir / "reg" / "girl" / "r.png"
    text_cache = new_dir / "train" / "1_data" / "a.png.text.safetensors"
    reg_text_cache = new_dir / "reg" / "girl" / "r.png.text.safetensors"
    assert npz.exists() and (new_dir / "train" / "1_data" / "a.r512.npz").exists()
    assert reg_npz.exists()
    assert text_cache.exists() and reg_text_cache.exists()
    # zip 内 npz 按字典序先于图片落盘，没有 touch 修正会比图片旧
    assert npz.stat().st_mtime >= img.stat().st_mtime
    assert reg_npz.stat().st_mtime >= reg_img.stat().st_mtime


def test_export_bundle_excludes_latent_cache_by_default(isolated, tmp_path: Path) -> None:
    p, v, train = _make_project_with_train(isolated)
    (train / "1_data").mkdir(parents=True, exist_ok=True)
    (train / "1_data" / "a.png").write_bytes(_png())
    (train / "1_data" / "a.npz").write_bytes(b"fake-latent")
    (train / "1_data" / "a.png.text.safetensors").write_bytes(b"fake-text")

    dest = tmp_path / "nocache.bundle.zip"
    with db.connection_for(isolated["db"]) as conn:
        result = train_io.export_bundle(
            conn, v["id"], dest, train_io.BundleOptions(train=True),
        )

    assert result["manifest"]["includes"]["train_latent_cache"] is False
    assert result["manifest"]["includes"]["reg_latent_cache"] is False
    assert result["manifest"]["stats"]["latent_cache_count"] == 0
    assert result["manifest"]["stats"]["text_cache_count"] == 0
    with zipfile.ZipFile(dest) as zf:
        assert not [n for n in zf.namelist() if n.endswith(".npz")]
        assert not [n for n in zf.namelist() if n.endswith(".text.safetensors")]


def _named_bundle(tmp_path: Path, *, with_preset: bool) -> Path:
    bundle = tmp_path / "named.bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({
                "schema_version": 2,
                "source": {
                    "title": "Named Bundle",
                    "slug": "named-bundle",
                    "version_label": "anime-v2",
                    "preset_name": "style_preset",
                },
                "includes": {"train": True, "presets": with_preset},
            }),
        )
        zf.writestr("train/1_data/a.png", b"fake")
        if with_preset:
            import yaml
            from studio.schema import TrainingConfig

            zf.writestr(
                "presets/style_preset.yaml",
                yaml.safe_dump(TrainingConfig().model_dump(mode="python"),
                               allow_unicode=True, sort_keys=False),
            )
    return bundle


def test_import_bundle_restores_version_and_preset_names(isolated, tmp_path: Path) -> None:
    bundle = _named_bundle(tmp_path, with_preset=True)

    with db.connection_for(isolated["db"]) as conn:
        result = train_io.import_bundle(conn, bundle, presets_base=tmp_path / "presets")

    assert result["version"]["label"] == "anime-v2"
    assert result["version"]["config_name"] == "style_preset"
    train_dir = versions.version_dir(
        result["project"]["id"],
        result["project"]["slug"],
        "anime-v2",
    ) / "train"
    assert (train_dir / "1_data" / "a.png").exists()


def test_import_bundle_skips_preset_name_when_preset_missing(isolated, tmp_path: Path) -> None:
    """bundle 没带预设、本机也没有 → config_name 不回填，避免悬空引用。"""
    bundle = _named_bundle(tmp_path, with_preset=False)

    with db.connection_for(isolated["db"]) as conn:
        result = train_io.import_bundle(conn, bundle, presets_base=tmp_path / "presets")

    assert result["version"]["label"] == "anime-v2"
    assert result["version"]["config_name"] is None


def test_import_bundle_rejects_dot_version_label(isolated, tmp_path: Path) -> None:
    """manifest 不可信：纯点 label（".." == project 根）必须回退 v1，
    否则 delete_version 时 rmtree 会带走整个项目目录。"""
    bundle = tmp_path / "dots.bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({
                "schema_version": 2,
                "source": {"title": "Evil", "slug": "evil", "version_label": ".."},
                "includes": {"train": True},
            }),
        )
        zf.writestr("train/1_data/a.png", b"fake")

    with db.connection_for(isolated["db"]) as conn:
        result = train_io.import_bundle(conn, bundle, presets_base=tmp_path / "presets")

    assert result["version"]["label"] == "v1"
    vdir = versions.version_dir(
        result["project"]["id"], result["project"]["slug"], "v1"
    )
    assert (vdir / "train" / "1_data" / "a.png").exists()


def _build_bundle_with_config(
    tmp_path: Path,
    *,
    transformer_path: str,
) -> Path:
    """构造一个最小 v2 bundle.zip：1 张训练图 + presets/config.yaml。

    transformer_path 用来塞 "源机器" 的绝对路径，模拟跨机器导入场景。
    """
    import yaml
    from studio.schema import TrainingConfig

    cfg = TrainingConfig().model_dump(mode="python")
    cfg["transformer_path"] = transformer_path

    bundle = tmp_path / "in.bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({
                "schema_version": 2,
                "source": {"title": "Imported", "slug": "imported", "version_label": "v1"},
                "includes": {"train": True, "config": True},
            }),
        )
        zf.writestr("train/1_data/a.png", b"fake")
        zf.writestr(
            "presets/config.yaml",
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        )
    return bundle


def test_import_bundle_config_with_auto_sync_on_overrides_model_paths(
    isolated, tmp_path: Path, monkeypatch
) -> None:
    """auto_sync_paths=ON（默认）：bundle 内 4 全局模型字段被本机 globals 覆盖。

    源机器（Windows）导出的 `G:/models/foo.safetensors` 在异机器不可解析；
    fork_preset_for_version 走的就是这个语义，bundle import 必须一致。
    """
    from studio.services import presets as preset_flow
    from studio.services import models as model_downloader

    monkeypatch.setattr(preset_flow, "_auto_sync_paths", lambda: True)
    src_path = "G:/source-machine/anima.safetensors"
    bundle = _build_bundle_with_config(tmp_path, transformer_path=src_path)

    with db.connection_for(isolated["db"]) as conn:
        result = train_io.import_bundle(conn, bundle, presets_base=tmp_path / "presets")

    assert result["stats"]["config_imported"] is True
    p = result["project"]
    v = result["version"]
    from studio.services import version_config
    cfg = version_config.read_version_config(p, v)
    expected = model_downloader.default_paths_for_new_version()["transformer_path"]
    assert cfg["transformer_path"] == Path(expected).as_posix()
    # 源路径已被覆盖，不残留
    assert cfg["transformer_path"] != src_path


def test_import_bundle_config_with_auto_sync_off_preserves_windows_path(
    isolated, tmp_path: Path, monkeypatch
) -> None:
    """auto_sync_paths=OFF：尊重 bundle 内的绝对路径；POSIX 上 Windows 盘符
    不被 REPO_ROOT 误拼成 `<repo>/G:/...`（盘符识别在 _absolutize_model_paths 里）。"""
    from studio.services import presets as preset_flow

    monkeypatch.setattr(preset_flow, "_auto_sync_paths", lambda: False)
    src_path = "G:/source-machine/anima.safetensors"
    bundle = _build_bundle_with_config(tmp_path, transformer_path=src_path)

    with db.connection_for(isolated["db"]) as conn:
        result = train_io.import_bundle(conn, bundle, presets_base=tmp_path / "presets")

    assert result["stats"]["config_imported"] is True
    p = result["project"]
    v = result["version"]
    from studio.services import version_config
    cfg = version_config.read_version_config(p, v)
    # 路径原样保留（POSIX 形式）；关键点是不会出现 REPO_ROOT 前缀
    assert cfg["transformer_path"] == src_path
    assert "G:/" in cfg["transformer_path"]


def test_import_bundle_normalizes_legacy_presets(isolated, tmp_path: Path) -> None:
    import yaml
    from studio.schema import TrainingConfig
    from studio.services.presets import io as presets_io

    legacy = TrainingConfig().model_dump(mode="python")
    legacy.pop("attention_backend", None)
    legacy["xformers"] = True
    legacy["optimizer_type"] = "not-real"

    bundle = tmp_path / "legacy.bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({
                "schema_version": 2,
                "source": {"title": "Imported", "slug": "imported", "version_label": "v1"},
                "includes": {"presets": True},
            }),
        )
        zf.writestr("presets/legacy.yaml", yaml.safe_dump(legacy, allow_unicode=True, sort_keys=False))

    presets_base = tmp_path / "presets"
    with db.connection_for(isolated["db"]) as conn:
        result = train_io.import_bundle(conn, bundle, presets_base=presets_base)

    assert result["stats"]["preset_count"] == 1
    assert [item["name"] for item in presets_io.list_presets(presets_base)] == ["legacy"]
    cfg = presets_io.read_preset("legacy", presets_base)
    assert cfg["attention_backend"] == "xformers"
    assert cfg["optimizer_type"] == TrainingConfig().optimizer_type
