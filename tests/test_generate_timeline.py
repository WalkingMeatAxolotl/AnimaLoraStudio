"""出图时间线 DB 单源:GET /api/generate/timeline 派生 + _v20 列回填。"""
from __future__ import annotations

import json
import sqlite3
import time
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from studio import db
from studio.api.routers import generate as gen
from studio.services import generate_storage as storage


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    path.write_bytes(buf.getvalue())


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    db.init_db()
    test_dir = tmp_path / "test"
    monkeypatch.setattr(gen, "TEST_IMAGES_DIR", test_dir)
    monkeypatch.setattr(storage, "TEST_IMAGES_DIR", test_dir)
    return test_dir


def _mk_task(
    *, status: str = "done",
    params: dict | None = None,
    images: list[dict] | None = None,
) -> int:
    with db.connection_for() as conn:
        task_id = db.create_task(conn, name="generate", config_name="generate", priority=0)
        fields: dict = {"task_type": "generate", "status": status}
        if params is not None:
            fields["generate_params"] = json.dumps(params, ensure_ascii=False)
        if images is not None:
            fields["generate_images"] = json.dumps(images, ensure_ascii=False)
        db.update_task(conn, task_id, **fields)
    return task_id


def _entries(**kw) -> list[dict]:
    return gen.generate_timeline(**kw)["entries"]


def test_empty_db_returns_no_entries(env) -> None:
    assert _entries() == []


def test_single_disk_row_urls_and_available(env) -> None:
    rel = "2026-08-10/single/single image 5.png"
    _png(env / rel)
    tid = _mk_task(params={"mode": "single"}, images=[{"file": rel, "src": "a.png"}])
    (e,) = _entries()
    assert e["task_id"] == tid
    assert e["mode"] == "single"
    assert e["storage"] == "disk"
    assert e["available"] is True
    img = e["images"][0]
    assert img["url"] == (
        "/api/generate/disk/image/2026-08-10/single/single%20image%205.png"
    )
    assert "w=128" in img["thumb_url"]


def test_missing_file_marks_released(env) -> None:
    _mk_task(
        params={"mode": "single"},
        images=[{"file": "2026-08-10/single/single image 9.png"}],
    )
    (e,) = _entries()
    assert e["available"] is False


def test_temp_row_without_cache_session_is_released(env) -> None:
    tid = _mk_task(params={"mode": "single"}, images=[{"cache": "img.png"}])
    (e,) = _entries()
    assert e["storage"] == "temp"
    assert e["images"][0]["url"] == f"/api/generate/{tid}/sample/img.png"
    assert e["available"] is False  # cache 未 init / session 已结束


def test_pending_row_included_failed_without_images_excluded(env) -> None:
    _mk_task(status="pending", params={"mode": "single"})
    _mk_task(status="failed", params={"mode": "single"})
    entries = _entries()
    assert [e["status"] for e in entries] == ["pending"]
    # pending 行 images 空但不 released 判定负担(available False 无图)
    assert entries[0]["images"] == []


def test_canceled_with_partial_images_included(env) -> None:
    rel = "2026-08-10/xy/xy plot 1/cell x0 y0.png"
    _png(env / rel)
    _mk_task(status="canceled", params={"mode": "xy"},
             images=[{"file": rel, "xi": 0, "yi": 0}])
    (e,) = _entries()
    assert e["status"] == "canceled"
    assert e["available"] is True


def test_xy_row_folder_and_composite(env) -> None:
    rel0 = "2026-08-10/xy/xy plot 3/cell x0 y0.png"
    rel1 = "2026-08-10/xy/xy plot 3/cell x1 y0.png"
    _png(env / rel0)
    _png(env / rel1)
    _png(env / "2026-08-10/xy/xy plot 3/xy plot.png")
    _mk_task(params={"mode": "xy"}, images=[
        {"file": rel0, "xi": 0, "yi": 0},
        {"file": rel1, "xi": 1, "yi": 0},
    ])
    (e,) = _entries()
    assert e["mode"] == "xy"
    assert e["xy_folder"] == "xy plot 3"
    assert e["composite_url"] == (
        "/api/generate/disk/image/2026-08-10/xy/xy%20plot%203/xy%20plot.png"
    )
    assert [(i["xi"], i["yi"]) for i in e["images"]] == [(0, 0), (1, 0)]


def test_pagination_and_order(env) -> None:
    ids = [
        _mk_task(params={"mode": "single"}, images=[{"cache": f"i{n}.png"}])
        for n in range(3)
    ]
    page = gen.generate_timeline(limit=2, offset=0)
    assert page["total"] == 3
    assert [e["task_id"] for e in page["entries"]] == [ids[2], ids[1]]
    page2 = gen.generate_timeline(limit=2, offset=2)
    assert [e["task_id"] for e in page2["entries"]] == [ids[0]]


# ---------------------------------------------------------------------------
# _v20 最小列回填(拍板决策 3:只 UPDATE 绝不 INSERT)
# ---------------------------------------------------------------------------


def test_v20_backfill_from_cover(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from studio.infrastructure.migrations import _v20_generate_images as v20
    import studio.paths as paths

    monkeypatch.setattr(paths, "STUDIO_DATA", tmp_path)
    test_dir = tmp_path / "test"
    # xy 文件夹:2 cells + composite
    _png(test_dir / "2026-07-01/xy/xy plot 2/cell x0 y0.png")
    _png(test_dir / "2026-07-01/xy/xy plot 2/cell x1 y0.png")
    _png(test_dir / "2026-07-01/xy/xy plot 2/xy plot.png")

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, task_type TEXT, "
        "generate_cover TEXT, created_at REAL)"
    )
    now = time.time()
    rows = [
        ("generate", r"2026-07-01\single\single image 3.png", now),  # 反斜杠老格式
        ("generate", "2026-07-01/xy/xy plot 2/xy plot.png", now),
        ("generate", None, now),           # temp 行:不回填
        ("train", None, now),              # 非 generate:不动
    ]
    conn.executemany(
        "INSERT INTO tasks (task_type, generate_cover, created_at) VALUES (?,?,?)",
        rows,
    )
    n_before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    v20.migrate(conn)
    # 绝不 INSERT
    assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == n_before

    got = {
        r[0]: (json.loads(r[1]) if r[1] else None)
        for r in conn.execute("SELECT id, generate_images FROM tasks")
    }
    assert got[1] == [{"file": "2026-07-01/single/single image 3.png"}]
    assert got[2] == [
        {"file": "2026-07-01/xy/xy plot 2/cell x0 y0.png", "xi": 0, "yi": 0},
        {"file": "2026-07-01/xy/xy plot 2/cell x1 y0.png", "xi": 1, "yi": 0},
    ]
    assert got[3] is None
    assert got[4] is None

    # 幂等:再跑一遍不重复、不覆盖
    v20.migrate(conn)
    assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == n_before


def test_v20_backfill_missing_xy_folder_empty_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studio.paths as paths
    from studio.infrastructure.migrations import _v20_generate_images as v20

    monkeypatch.setattr(paths, "STUDIO_DATA", tmp_path)
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, task_type TEXT, "
        "generate_cover TEXT, created_at REAL)"
    )
    conn.execute(
        "INSERT INTO tasks (task_type, generate_cover, created_at) VALUES "
        "('generate', '2026-07-01/xy/gone folder/xy plot.png', 0)"
    )
    v20.migrate(conn)
    row = conn.execute("SELECT generate_images FROM tasks").fetchone()
    assert json.loads(row[0]) == []  # 文件夹被手删 → 已释放行
