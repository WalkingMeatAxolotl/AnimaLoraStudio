from __future__ import annotations

from pathlib import Path

from studio import db, project_jobs, projects
from studio.workers import download_worker


def test_download_worker_passes_signal_cancel_event(
    tmp_path: Path, monkeypatch
) -> None:
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")

    with db.connection_for(dbfile) as conn:
        project = projects.create_project(conn, title="P")
        job = project_jobs.create_job(
            conn,
            project_id=project["id"],
            kind="download",
            params={"tag": "x", "count": 1},
        )

    captured = {}

    class FakeGelbooru:
        user_id = "uid"
        api_key = "key"
        save_tags = False
        convert_to_png = True
        remove_alpha_channel = False

    class FakeDanbooru:
        username = ""
        api_key = ""

    class FakeDownload:
        exclude_tags: list[str] = []

    class FakeSecrets:
        gelbooru = FakeGelbooru()
        danbooru = FakeDanbooru()
        download = FakeDownload()

    def fake_download(_opts, _dest, *, cancel_event, **_kwargs):  # noqa: ANN001
        captured["cancel_event"] = cancel_event
        download_worker._on_signal(None, None)
        assert cancel_event.is_set()
        return 0

    monkeypatch.setattr(download_worker.secrets, "load", lambda: FakeSecrets())
    monkeypatch.setattr(download_worker.downloader, "download", fake_download)

    assert download_worker.run(job["id"]) == 0
    assert captured["cancel_event"] is download_worker._cancel_event
