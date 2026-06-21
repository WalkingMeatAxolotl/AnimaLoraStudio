"""PP2 — /api/projects/{pid}/download + /api/jobs/* HTTP。"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, secrets, server
from studio.services.projects import jobs as project_jobs, projects


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    # 上传 endpoint 默认会按 download.convert_to_png 走 PIL 重编码；这里关掉，让
    # 现有用 raw bytes 的上传测试继续验证原始拷贝分支。convert 路径用专门测试。
    secrets.update(
        {
            "gelbooru": {"user_id": "u", "api_key": "k"},
            "download": {
                "convert_to_png": False,
                "remove_alpha_channel": False,
            },
        }
    )
    return {"db": dbfile}


class _StubSupervisor:
    def __init__(self) -> None:
        self.canceled: list[int] = []

    def cancel_job(self, jid: int) -> bool:
        with db.connection_for() as conn:
            j = project_jobs.get_job(conn, jid)
            if not j or j["status"] in project_jobs.TERMINAL_STATUSES:
                return False
            project_jobs.mark_canceled(conn, jid)
        self.canceled.append(jid)
        return True


@pytest.fixture
def client(isolated) -> TestClient:
    server.app.state.supervisor = _StubSupervisor()
    return TestClient(server.app)


def _make_project(client: TestClient) -> dict:
    return client.post(
        "/api/projects", json={"title": "P", "initial_version_label": None}
    ).json()


# ---------------------------------------------------------------------------
# start_download
# ---------------------------------------------------------------------------


def test_start_download_creates_job_and_advances_stage(
    client: TestClient,
) -> None:
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "char_x", "count": 5},
    )
    assert resp.status_code == 200, resp.text
    job = resp.json()
    assert job["status"] == "pending"
    assert job["kind"] == "download"
    # ADR-0007 PR-5: project 无 stage 字段；download 状态由 job + UI 实时扫派生


def test_start_download_rejects_empty_tag(client: TestClient) -> None:
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/download", json={"tag": "  ", "count": 1}
    )
    assert resp.status_code == 400


def test_start_download_requires_credentials(
    client: TestClient, isolated, monkeypatch
) -> None:
    # gelbooru 缺凭据：拒绝
    monkeypatch.setattr(
        secrets, "has_credentials_for",
        lambda src: False if src == "gelbooru" else True,
    )
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "x", "count": 1, "api_source": "gelbooru"},
    )
    assert resp.status_code == 400
    assert "gelbooru" in resp.json()["error"]["message"]


def test_start_download_danbooru_does_not_require_credentials(
    client: TestClient, isolated, monkeypatch
) -> None:
    """Danbooru 匿名也能跑，端点不应在缺凭据时阻挡。"""
    monkeypatch.setattr(
        secrets, "has_credentials_for",
        lambda src: True if src == "danbooru" else False,
    )
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "x", "count": 1, "api_source": "danbooru"},
    )
    assert resp.status_code == 200, resp.text


def test_estimate_endpoint_returns_count(
    client: TestClient, isolated, monkeypatch
) -> None:
    """estimate 端点：通过 mock downloader.estimate 返回固定数量。"""
    from studio.services.booru import downloader as dl
    monkeypatch.setattr(dl, "estimate", lambda opts: 42)
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/download/estimate",
        json={"tag": "x", "api_source": "gelbooru"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 42
    assert body["effective_query"] == "x"


def test_estimate_includes_exclude_tags(
    client: TestClient, isolated, monkeypatch
) -> None:
    secrets.update({"download": {"exclude_tags": ["comic", "monochrome"]}})
    from studio.services.booru import downloader as dl
    monkeypatch.setattr(dl, "estimate", lambda opts: 7)
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/download/estimate",
        json={"tag": "x", "api_source": "gelbooru"},
    )
    body = resp.json()
    assert body["exclude_tags"] == ["comic", "monochrome"]
    assert "-comic" in body["effective_query"]
    assert "-monochrome" in body["effective_query"]


def test_start_download_rejects_bad_source(client: TestClient) -> None:
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "x", "count": 1, "api_source": "wat"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# status / log
# ---------------------------------------------------------------------------


def test_download_status_returns_latest(client: TestClient) -> None:
    p = _make_project(client)
    j1 = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "a", "count": 1},
    ).json()
    j2 = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "b", "count": 1},
    ).json()
    r = client.get(f"/api/projects/{p['id']}/download/status").json()
    assert r["job"]["id"] == j2["id"]


def test_download_status_no_jobs(client: TestClient) -> None:
    p = _make_project(client)
    r = client.get(f"/api/projects/{p['id']}/download/status").json()
    assert r["job"] is None


def test_get_job_log_returns_tail(client: TestClient, isolated) -> None:
    p = _make_project(client)
    job = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "a", "count": 1},
    ).json()
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    r = client.get(f"/api/jobs/{job['id']}/log?tail=2").json()
    assert r["content"].splitlines() == ["c", "d"]


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------


def test_list_files_empty(client: TestClient) -> None:
    p = _make_project(client)
    r = client.get(f"/api/projects/{p['id']}/files").json()
    assert r == {"items": [], "count": 0}


def test_list_files_returns_images(client: TestClient) -> None:
    p = _make_project(client)
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    (pdir / "1.png").write_bytes(b"x")
    (pdir / "2.jpg").write_bytes(b"x")
    (pdir / "ignored.txt").write_bytes(b"x")
    r = client.get(f"/api/projects/{p['id']}/files").json()
    names = sorted(i["name"] for i in r["items"])
    assert names == ["1.png", "2.jpg"]


def test_thumb_serves_image(client: TestClient) -> None:
    p = _make_project(client)
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    (pdir / "1.png").write_bytes(b"\x89PNG fake")
    r = client.get(
        f"/api/projects/{p['id']}/thumb?bucket=download&name=1.png"
    )
    assert r.status_code == 200
    assert r.content == b"\x89PNG fake"


def test_thumb_rejects_path_traversal(client: TestClient) -> None:
    p = _make_project(client)
    r = client.get(
        f"/api/projects/{p['id']}/thumb?bucket=download&name=../etc/passwd"
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# delete files
# ---------------------------------------------------------------------------


def test_delete_files_removes_image_and_metadata(client: TestClient) -> None:
    p = _make_project(client)
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "a.png").write_bytes(b"img")
    (pdir / "a.booru.txt").write_text("tags", encoding="utf-8")
    (pdir / "a.txt").write_text("more tags", encoding="utf-8")
    (pdir / "a.json").write_text("{}", encoding="utf-8")
    (pdir / "b.png").write_bytes(b"img2")  # 不在删除列表

    r = client.post(
        f"/api/projects/{p['id']}/files/delete",
        json={"names": ["a.png"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["deleted"] == ["a.png"]
    assert body["missing"] == []
    # a.png 及其全部 metadata 都不在了
    assert not (pdir / "a.png").exists()
    assert not (pdir / "a.booru.txt").exists()
    assert not (pdir / "a.txt").exists()
    assert not (pdir / "a.json").exists()
    # b.png 不动
    assert (pdir / "b.png").exists()


def test_delete_files_reports_missing(client: TestClient) -> None:
    p = _make_project(client)
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "exists.png").write_bytes(b"x")
    r = client.post(
        f"/api/projects/{p['id']}/files/delete",
        json={"names": ["exists.png", "ghost.png"]},
    )
    body = r.json()
    assert body["deleted"] == ["exists.png"]
    assert body["missing"] == ["ghost.png"]


def test_delete_files_blocks_traversal(client: TestClient) -> None:
    p = _make_project(client)
    for bad in ("../escape.png", "..\\escape.png", "sub/x.png", ""):
        r = client.post(
            f"/api/projects/{p['id']}/files/delete",
            json={"names": [bad]},
        )
        assert r.status_code == 400


def test_delete_files_empty_request(client: TestClient) -> None:
    p = _make_project(client)
    r = client.post(
        f"/api/projects/{p['id']}/files/delete",
        json={"names": []},
    )
    assert r.status_code == 200
    assert r.json() == {"deleted": [], "missing": []}


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n, d in entries.items():
            zf.writestr(n, d)
    return buf.getvalue()


def test_upload_single_image(client: TestClient) -> None:
    p = _make_project(client)
    r = client.post(
        f"/api/projects/{p['id']}/upload",
        files=[("files", ("a.jpg", b"\xff\xd8jpgdata", "image/jpeg"))],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added"] == ["a.jpg"]
    assert body["skipped"] == []
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    assert (pdir / "a.jpg").read_bytes() == b"\xff\xd8jpgdata"
    # ADR-0007 PR-5: upload 不再推 stage；download_image_count 派生即可


def test_upload_runs_accept_many_off_event_loop(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression：accept_many 必须跑在 threadpool worker（不持有 asyncio loop），
    否则 PIL / zipfile 同步 CPU 工作会卡死 event loop —— 用户表现是上传"卡 100%"
    后 F5 刷新页面完全卡死（所有 HTTP 请求排队等被卡的 loop），并且 SIGINT 也
    无效（PIL/zipfile C 扩展持 GIL 期间 Python 收不到信号）。

    判定：把 accept_many 包一层，在调用瞬间 try asyncio.get_running_loop()
    —— 跑在 event loop 线程会返回 loop（BUG），跑在 worker 线程会 RuntimeError
    （正确）。
    """
    import asyncio
    from studio.services.dataset import uploads
    in_loop: list[bool] = []
    original = uploads.accept_many

    def tracking(*args, **kwargs):
        try:
            asyncio.get_running_loop()
            in_loop.append(True)
        except RuntimeError:
            in_loop.append(False)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "studio.api.routers.projects.ingestion.uploads_svc.accept_many",
        tracking,
    )
    p = _make_project(client)
    r = client.post(
        f"/api/projects/{p['id']}/upload",
        files=[("files", ("a.jpg", b"\xff\xd8jpgdata", "image/jpeg"))],
    )
    assert r.status_code == 200, r.text
    assert in_loop == [False], (
        "accept_many ran on the asyncio event loop thread — F5 卡死 bug 复发。"
        " 路由必须用 run_in_threadpool 把同步 CPU 工作挪进 worker 线程。"
    )


def test_upload_publishes_state_and_log_events(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """上传期间 server 端 emit SSE 事件给前端 TaskLogDrawer 显示：
    - `project_upload_state` running → done（包 try/finally：异常→failed）
    - `project_upload_log` 每条 on_log 一次（accept_many 至少推 starting / all done）
    用 zip 触发 zip 路径（含 entries / 进度行）。
    """
    captured: list[dict] = []
    from studio.infrastructure.event_bus import bus
    original_publish = bus.publish

    def capture(msg):
        captured.append(msg)
        original_publish(msg)

    monkeypatch.setattr(bus, "publish", capture)

    p = _make_project(client)
    blob = _zip_bytes({"a.jpg": b"AA", "b.png": b"BB"})
    r = client.post(
        f"/api/projects/{p['id']}/upload",
        files=[("files", ("pack.zip", blob, "application/zip"))],
    )
    assert r.status_code == 200, r.text

    states = [
        m for m in captured
        if m.get("type") == "project_upload_state" and m.get("project_id") == p["id"]
    ]
    logs = [
        m for m in captured
        if m.get("type") == "project_upload_log" and m.get("project_id") == p["id"]
    ]
    assert [s["status"] for s in states] == ["running", "done"], states
    assert any("starting" in (m.get("line") or "") for m in logs), [m["line"] for m in logs]
    assert any("all done" in (m.get("line") or "") for m in logs), [m["line"] for m in logs]


def test_upload_zip_extracts_images(client: TestClient) -> None:
    p = _make_project(client)
    blob = _zip_bytes({"a.jpg": b"AA", "sub/b.png": b"BB", "skip.txt": b"X"})
    r = client.post(
        f"/api/projects/{p['id']}/upload",
        files=[("files", ("pack.zip", blob, "application/zip"))],
    )
    body = r.json()
    assert sorted(body["added"]) == ["a.jpg", "b.png"]
    assert any("skip.txt" in s["name"] for s in body["skipped"])


def test_upload_rejects_unsupported_format(client: TestClient) -> None:
    p = _make_project(client)
    r = client.post(
        f"/api/projects/{p['id']}/upload",
        files=[("files", ("note.txt", b"x", "text/plain"))],
    )
    body = r.json()
    assert body["added"] == []
    assert len(body["skipped"]) == 1
    # ADR-0007 PR-5: project 已无 stage；没添加文件即不动 download_image_count


def test_upload_skip_existing(client: TestClient) -> None:
    p = _make_project(client)
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "x.png").write_bytes(b"old")
    r = client.post(
        f"/api/projects/{p['id']}/upload",
        files=[("files", ("x.png", b"new", "image/png"))],
    )
    body = r.json()
    assert body["added"] == []
    assert body["skipped"][0]["reason"] == "已存在，跳过"
    assert (pdir / "x.png").read_bytes() == b"old"


def test_upload_convert_to_png_renames_and_dedups(
    client: TestClient,
) -> None:
    """download.convert_to_png=True：上传 1.png + 1.jpg（不同图）→ 1.png + 1_1.png。

    解决 caption `1.txt` 被 jpg/png 两张不同图共用的问题；同 stem 加 `_N` 后缀
    保住第二张而不是跳过。
    """
    import io as _io

    from PIL import Image
    from studio import secrets

    secrets.update({"download": {"convert_to_png": True}})

    def _png(color):
        buf = _io.BytesIO()
        Image.new("RGB", (4, 4), color).save(buf, "PNG")
        return buf.getvalue()

    def _jpg(color):
        buf = _io.BytesIO()
        Image.new("RGB", (4, 4), color).save(buf, "JPEG", quality=90)
        return buf.getvalue()

    p = _make_project(client)
    r = client.post(
        f"/api/projects/{p['id']}/upload",
        files=[
            ("files", ("1.png", _png((0, 0, 0)), "image/png")),
            ("files", ("1.jpg", _jpg((255, 255, 255)), "image/jpeg")),
        ],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert sorted(body["added"]) == ["1.png", "1_1.png"]
    assert body["skipped"] == []
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    assert (pdir / "1.png").exists()
    assert (pdir / "1_1.png").exists()


def test_upload_404_for_unknown_project(client: TestClient) -> None:
    r = client.post(
        "/api/projects/999/upload",
        files=[("files", ("a.jpg", b"a", "image/jpeg"))],
    )
    assert r.status_code == 404


def test_upload_400_for_no_files(client: TestClient) -> None:
    p = _make_project(client)
    # FastAPI 在没有任何 files= 表单时返回 422（schema validation），
    # 此测验证 422/400 两者皆可接受作为「客户端错」。
    r = client.post(f"/api/projects/{p['id']}/upload")
    assert r.status_code in (400, 422)


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


def test_cancel_pending_job_endpoint(client: TestClient) -> None:
    p = _make_project(client)
    job = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "x", "count": 1},
    ).json()
    r = client.post(f"/api/jobs/{job['id']}/cancel")
    assert r.status_code == 200
    again = client.get(f"/api/jobs/{job['id']}").json()
    assert again["status"] == "canceled"


def test_cancel_terminal_job_400(client: TestClient) -> None:
    p = _make_project(client)
    job = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "x", "count": 1},
    ).json()
    with db.connection_for() as conn:
        project_jobs.mark_done(conn, job["id"])
    r = client.post(f"/api/jobs/{job['id']}/cancel")
    assert r.status_code == 400
