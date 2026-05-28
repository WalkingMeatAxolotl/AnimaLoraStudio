"""PP1 — versions.py: label 唯一、目录树、fork、active reassign。"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import db
from studio.services.projects import projects, versions


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    pdir = tmp_path / "projects"
    monkeypatch.setattr(projects, "PROJECTS_DIR", pdir)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    return {"db": dbfile}


def _new_project(isolated, title: str = "P1") -> dict:
    with db.connection_for(isolated["db"]) as conn:
        return projects.create_project(conn, title=title)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_version_builds_tree_and_activates(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v = versions.create_version(conn, project_id=p["id"], label="baseline")
    vdir = versions.version_dir(p["id"], p["slug"], "baseline")
    assert vdir.exists()
    for sub in ("train", "reg", "output", "samples"):
        assert (vdir / sub).is_dir()
    assert (vdir / "version.json").exists()
    # 项目里第一个版本自动设为 active
    with db.connection_for(isolated["db"]) as conn:
        p2 = projects.get_project(conn, p["id"])
    assert p2 and p2["active_version_id"] == v["id"]


def test_create_version_rejects_invalid_label(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        for bad in ("has space", "../escape", "name/sub", "中文"):
            with pytest.raises(versions.VersionError, match="label"):
                versions.create_version(conn, project_id=p["id"], label=bad)


def test_create_version_rejects_duplicate_label(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        versions.create_version(conn, project_id=p["id"], label="baseline")
        with pytest.raises(versions.VersionError, match="已存在"):
            versions.create_version(conn, project_id=p["id"], label="baseline")


# ---------------------------------------------------------------------------
# fork
# ---------------------------------------------------------------------------


def test_fork_copies_train_tree(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        src = versions.create_version(conn, project_id=p["id"], label="baseline")
    src_train = versions.version_dir(p["id"], p["slug"], "baseline") / "train"
    folder = src_train / "5_concept"
    folder.mkdir()
    (folder / "001.png").write_bytes(b"fakepng")
    (folder / "001.txt").write_text("tag1, tag2", encoding="utf-8")

    with db.connection_for(isolated["db"]) as conn:
        v2 = versions.create_version(
            conn,
            project_id=p["id"],
            label="forked",
            fork_from_version_id=src["id"],
        )
    new_folder = (
        versions.version_dir(p["id"], p["slug"], "forked")
        / "train" / "5_concept"
    )
    assert (new_folder / "001.png").read_bytes() == b"fakepng"
    assert (new_folder / "001.txt").read_text(encoding="utf-8") == "tag1, tag2"
    assert v2["config_name"] == src["config_name"]


def test_fork_full_copy_includes_reg_config_unlocked(isolated, monkeypatch) -> None:
    """PP10.1：fork 时 train/、reg/、config.yaml、.unlocked.json 全量复制；
    config.yaml 里 data_dir / reg_data_dir / output_dir / output_name 强制刷成新 version 路径。"""
    from studio.services import version_config
    from studio.schema import TrainingConfig

    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        src = versions.create_version(conn, project_id=p["id"], label="baseline")
    src_vdir = versions.version_dir(p["id"], p["slug"], "baseline")

    # train 内放图
    (src_vdir / "train" / "5_concept").mkdir()
    (src_vdir / "train" / "5_concept" / "001.png").write_bytes(b"trainpng")

    # reg/ 含 meta.json 和图
    (src_vdir / "reg" / "1_data").mkdir(parents=True)
    (src_vdir / "reg" / "meta.json").write_text(
        '{"target": 100}', encoding="utf-8"
    )
    (src_vdir / "reg" / "1_data" / "r.png").write_bytes(b"regpng")

    # 写一份 source config.yaml（路径故意指向 src 自己；fork 后应被刷掉）
    src_cfg = TrainingConfig().model_dump()
    version_config.write_version_config(p, src, src_cfg)

    # .unlocked.json 旁路文件（PP10.4 预留）
    (src_vdir / ".unlocked.json").write_text(
        '{"fields": ["resume_lora"]}', encoding="utf-8"
    )

    with db.connection_for(isolated["db"]) as conn:
        v2 = versions.create_version(
            conn,
            project_id=p["id"],
            label="forked",
            fork_from_version_id=src["id"],
        )
    new_vdir = versions.version_dir(p["id"], p["slug"], "forked")

    # train 复制
    assert (new_vdir / "train" / "5_concept" / "001.png").read_bytes() == b"trainpng"
    # reg 复制
    assert (new_vdir / "reg" / "meta.json").exists()
    assert (new_vdir / "reg" / "1_data" / "r.png").read_bytes() == b"regpng"
    # config.yaml 复制
    assert (new_vdir / "config.yaml").exists()
    # .unlocked.json 复制
    assert (new_vdir / ".unlocked.json").read_text(encoding="utf-8") == (
        '{"fields": ["resume_lora"]}'
    )

    # config.yaml 里项目特定字段已重写到新 version 路径
    new_cfg = version_config.read_version_config(p, v2)
    assert new_cfg["data_dir"] == str(new_vdir / "train")
    assert new_cfg["reg_data_dir"] == str(new_vdir / "reg")
    assert new_cfg["output_dir"] == str(new_vdir / "output")
    assert new_cfg["output_name"] == f"{p['slug']}_forked"


def test_fork_stage_done_resets_to_ready(isolated) -> None:
    """源 stage=done → 新 version 落 ready（重新进入待训练态）。"""
    # ADR-0007 PR-5: fork 不再继承 stage / phase；新 version 始终 preparing/curating。
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        src = versions.create_version(conn, project_id=p["id"], label="baseline")
        versions.update_version(conn, src["id"], status="completed")
        v2 = versions.create_version(
            conn,
            project_id=p["id"],
            label="forked",
            fork_from_version_id=src["id"],
        )
    assert v2["status"] == "preparing"
    assert v2["phase"] == "curating"


def test_fork_rejects_alien_source(isolated) -> None:
    a = _new_project(isolated, title="A")
    b = _new_project(isolated, title="B")
    with db.connection_for(isolated["db"]) as conn:
        src = versions.create_version(conn, project_id=a["id"], label="baseline")
        with pytest.raises(versions.VersionError, match="fork"):
            versions.create_version(
                conn,
                project_id=b["id"],
                label="x",
                fork_from_version_id=src["id"],
            )


# ---------------------------------------------------------------------------
# delete + active reassign
# ---------------------------------------------------------------------------


def test_delete_active_version_reassigns(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v1 = versions.create_version(conn, project_id=p["id"], label="v1")
        v2 = versions.create_version(conn, project_id=p["id"], label="v2")
        # active = v1（首次）；切到 v2
        versions.activate_version(conn, v2["id"])
        # 删 v2 → active 应回到 v1（剩下创建最新的）
        versions.delete_version(conn, v2["id"])
        p2 = projects.get_project(conn, p["id"])
    assert p2 and p2["active_version_id"] == v1["id"]


def test_delete_last_version_clears_active(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v1 = versions.create_version(conn, project_id=p["id"], label="only")
        versions.delete_version(conn, v1["id"])
        p2 = projects.get_project(conn, p["id"])
    assert p2 and p2["active_version_id"] is None


def test_delete_removes_dir(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v = versions.create_version(conn, project_id=p["id"], label="baseline")
        src = versions.version_dir(p["id"], p["slug"], "baseline")
        assert src.exists()
        versions.delete_version(conn, v["id"])
    assert not src.exists()


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_for_version_counts_train_and_reg(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v = versions.create_version(conn, project_id=p["id"], label="v1")
    vdir = versions.version_dir(p["id"], p["slug"], "v1")
    # 默认 1_data 已存在；这里加一个 5_concept 验证多 folder 计数
    (vdir / "train" / "5_concept").mkdir(parents=True)
    (vdir / "train" / "5_concept" / "a.png").write_bytes(b"x")
    (vdir / "train" / "5_concept" / "b.png").write_bytes(b"x")
    (vdir / "reg" / "1_data").mkdir(parents=True)
    (vdir / "reg" / "1_data" / "r.png").write_bytes(b"x")
    stats = versions.stats_for_version(p, v)
    assert stats["train_image_count"] == 2
    assert stats["reg_image_count"] == 1
    folder_names = {f["name"] for f in stats["train_folders"]}
    assert folder_names == {"1_data", "5_concept"}
    assert {f["name"]: f["image_count"] for f in stats["train_folders"]} == {
        "1_data": 0,
        "5_concept": 2,
    }
    assert stats["has_output"] is False


def test_create_version_provisions_default_train_folder(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        versions.create_version(conn, project_id=p["id"], label="v1")
    vdir = versions.version_dir(p["id"], p["slug"], "v1")
    assert (vdir / "train" / versions.DEFAULT_TRAIN_FOLDER).is_dir()
