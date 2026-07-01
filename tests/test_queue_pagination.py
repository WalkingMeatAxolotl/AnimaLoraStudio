"""0.17 P-A/P-C/P-E — 队列 DB 层分页 / 计数 / 搜索 / 类型排除。

覆盖 db.count_tasks / db.list_tasks_page / _build_task_filter 语义：
  - LIVE_STATUSES / HISTORY_STATUSES 分组
  - exclude_types 下沉 SQL（保证 total 与分页一致）
  - q 搜 name/config_name，LIKE 元字符转义
  - limit/offset 分页 + id DESC 排序
端点级分页测试在 test_studio_queue_endpoints.py。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import db


@pytest.fixture
def env(tmp_path: Path) -> Path:
    db_path = tmp_path / "studio.db"
    db.init_db(db_path)
    return db_path


def _mk(
    db_path: Path, name: str, *,
    status: str = "pending", task_type: str = "train", config_name: str | None = None,
) -> int:
    with db.connection_for(db_path) as conn:
        tid = db.create_task(conn, name=name, config_name=config_name or name)
        db.update_task(conn, tid, status=status, task_type=task_type)
    return tid


def test_count_history_vs_live(env: Path) -> None:
    _mk(env, "run", status="running")
    _mk(env, "pend", status="pending")
    _mk(env, "pause", status="paused")
    _mk(env, "d1", status="done")
    _mk(env, "d2", status="failed")
    _mk(env, "d3", status="canceled")
    with db.connection_for(env) as conn:
        assert db.count_tasks(conn, statuses=db.LIVE_STATUSES) == 3
        assert db.count_tasks(conn, statuses=db.HISTORY_STATUSES) == 3
        live = db.list_tasks_page(conn, statuses=db.LIVE_STATUSES)
        assert {t["status"] for t in live} == {"running", "pending", "paused"}


def test_exclude_types_lowers_count(env: Path) -> None:
    """generate/reg_ai 排除必须在 SQL 内，否则分页 total 会偏。"""
    _mk(env, "t", status="done", task_type="train")
    _mk(env, "r", status="done", task_type="reg_ai")
    _mk(env, "g", status="done", task_type="generate")
    with db.connection_for(env) as conn:
        assert db.count_tasks(conn, statuses=db.HISTORY_STATUSES) == 3
        assert db.count_tasks(
            conn, statuses=db.HISTORY_STATUSES, exclude_types=("generate", "reg_ai"),
        ) == 1
        rows = db.list_tasks_page(
            conn, statuses=db.HISTORY_STATUSES, exclude_types=("generate", "reg_ai"),
        )
        assert [r["name"] for r in rows] == ["t"]


def test_pagination_slices_and_orders_desc(env: Path) -> None:
    ids = [_mk(env, f"d{i}", status="done") for i in range(5)]  # id 递增
    with db.connection_for(env) as conn:
        assert db.count_tasks(conn, statuses=db.HISTORY_STATUSES) == 5
        p1 = db.list_tasks_page(conn, statuses=db.HISTORY_STATUSES, limit=2, offset=0)
        p2 = db.list_tasks_page(conn, statuses=db.HISTORY_STATUSES, limit=2, offset=2)
        p3 = db.list_tasks_page(conn, statuses=db.HISTORY_STATUSES, limit=2, offset=4)
    # id DESC：最新的先出
    assert [r["id"] for r in p1] == [ids[4], ids[3]]
    assert [r["id"] for r in p2] == [ids[2], ids[1]]
    assert [r["id"] for r in p3] == [ids[0]]


def test_search_matches_name_and_config(env: Path) -> None:
    _mk(env, "alpha_lora", status="done", config_name="cfg_alpha")
    _mk(env, "beta_run", status="done", config_name="cfg_beta")
    with db.connection_for(env) as conn:
        by_name = db.list_tasks_page(conn, statuses=db.HISTORY_STATUSES, q="alpha")
        assert [r["name"] for r in by_name] == ["alpha_lora"]
        by_cfg = db.list_tasks_page(conn, statuses=db.HISTORY_STATUSES, q="cfg_beta")
        assert [r["name"] for r in by_cfg] == ["beta_run"]
        assert db.count_tasks(conn, statuses=db.HISTORY_STATUSES, q="alpha") == 1


def test_search_escapes_like_metachars(env: Path) -> None:
    """含 % 的搜索按字面匹配，不当通配符。"""
    _mk(env, "fifty%off", status="done")
    _mk(env, "plain", status="done")
    with db.connection_for(env) as conn:
        hit = db.list_tasks_page(conn, statuses=db.HISTORY_STATUSES, q="%off")
        assert [r["name"] for r in hit] == ["fifty%off"]
        # 若未转义，'%' 会通配全部 → 2 条；转义后只命中字面 '%'
        assert db.count_tasks(conn, statuses=db.HISTORY_STATUSES, q="%") == 1
