"""commit 15 P0-2：/api/queue 默认过滤 generate task；可加 ?include_generate=true 兜底。

测 db.filter_out_task_types 纯函数 + 通过 db.list_tasks 验证默认 task_type
回退（fallback="train"）。fastapi endpoint 集成测留 fastapi env 装好的环境
跑（当前 dev env 没装）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import db


@pytest.fixture
def env(tmp_path: Path):
    db_path = tmp_path / "studio.db"
    db.init_db(db_path)
    return db_path


def _seed_tasks(db_path: Path) -> tuple[int, int, int]:
    with db.connection_for(db_path) as conn:
        train_tid = db.create_task(conn, name="t", config_name="t")
        reg_tid = db.create_task(conn, name="r", config_name="r")
        db.update_task(conn, reg_tid, task_type="reg_ai")
        gen_tid = db.create_task(conn, name="g", config_name="g")
        db.update_task(conn, gen_tid, task_type="generate")
    return train_tid, reg_tid, gen_tid


def test_filter_out_excludes_generate(env):
    train_tid, reg_tid, gen_tid = _seed_tasks(env)
    with db.connection_for(env) as conn:
        items = db.list_tasks(conn)
    filtered = db.filter_out_task_types(items, ("generate",))
    ids = {t["id"] for t in filtered}
    assert train_tid in ids
    assert reg_tid in ids
    assert gen_tid not in ids


def test_filter_out_default_train_fallback() -> None:
    """task_type 字段缺省时应当作 'train'（旧 task 兼容）。"""
    items = [
        {"id": 1, "task_type": None, "name": "old"},
        {"id": 2, "task_type": "generate", "name": "new"},
        {"id": 3, "name": "no-key"},
    ]
    filtered = db.filter_out_task_types(items, ("generate",))
    ids = [t["id"] for t in filtered]
    assert ids == [1, 3]


def test_filter_out_multiple_types() -> None:
    items = [
        {"id": 1, "task_type": "train"},
        {"id": 2, "task_type": "reg_ai"},
        {"id": 3, "task_type": "generate"},
    ]
    filtered = db.filter_out_task_types(items, ("generate", "reg_ai"))
    ids = [t["id"] for t in filtered]
    assert ids == [1]
