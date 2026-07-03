"""R-2 台账合并（schema 层）：tasks 承接数据作业字段（写路径切换在 R-3）。

docs/design/queue-resource-model-0.17.md §5 R-2。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import db


@pytest.fixture
def dbfile(tmp_path: Path) -> Path:
    path = tmp_path / "studio.db"
    db.init_db(path)
    return path


def test_valid_task_types_covers_all_resource_kinds() -> None:
    """db.VALID_TASK_TYPES 与 supervisor/resources.py 档位映射保持同步。

    档位归属权威在 resources.py；db 层平铺列出（infrastructure 不反向依赖
    supervisor）。本断言防两边漂移。
    """
    from studio.supervisor.resources import (
        JOB_KIND_RESOURCE_CLASS,
        TASK_TYPE_RESOURCE_CLASS,
    )
    expected = set(TASK_TYPE_RESOURCE_CLASS) | set(JOB_KIND_RESOURCE_CLASS)
    assert set(db.VALID_TASK_TYPES) == expected


def test_create_task_accepts_job_kind_with_params(dbfile: Path) -> None:
    """数据作业类 task：task_type + params + project/version 全字段落库回读。"""
    with db.connection_for(dbfile) as conn:
        tid = db.create_task(
            conn, name="tag v1", config_name="tag",
            task_type="tag",
            params={"tagger": "wd14", "threshold": 0.35},
            project_id=1, version_id=2,
        )
        task = db.get_task(conn, tid)
    assert task["task_type"] == "tag"
    assert task["params_decoded"] == {"tagger": "wd14", "threshold": 0.35}
    assert task["project_id"] == 1 and task["version_id"] == 2
    assert task["status"] == "pending"


def test_create_task_defaults_unchanged(dbfile: Path) -> None:
    """老调用方（不带新参数）行为不变：task_type=train、params NULL。"""
    with db.connection_for(dbfile) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        task = db.get_task(conn, tid)
    assert task["task_type"] == "train"
    assert task["params"] is None
    assert "params_decoded" not in task


def test_create_task_rejects_unknown_type(dbfile: Path) -> None:
    with db.connection_for(dbfile) as conn:
        with pytest.raises(ValueError):
            db.create_task(conn, name="x", config_name="c", task_type="banana")


def test_params_decoded_in_list_paths(dbfile: Path) -> None:
    """list_tasks / list_tasks_page 与 get_task 同样带 params_decoded。"""
    with db.connection_for(dbfile) as conn:
        db.create_task(
            conn, name="dl", config_name="dl",
            task_type="download", params={"tag": "usa", "count": 5},
        )
        via_list = db.list_tasks(conn, status="pending")[0]
        via_page = db.list_tasks_page(conn, statuses=("pending",))[0]
    assert via_list["params_decoded"] == {"tag": "usa", "count": 5}
    assert via_page["params_decoded"] == {"tag": "usa", "count": 5}
