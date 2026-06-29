"""公告栏数据源解析测试（announcement-center Phase 1）。

覆盖 studio.services.announcements.list_posts 的解析 / zh-en 配对 / en fallback /
字段校验 / 排序，以及 GET /api/announcements 端点形状。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio.services import announcements as svc


def _write(d: Path, name: str, *, frontmatter: str, body: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8")


def test_parses_zh_en_pair(tmp_path: Path) -> None:
    _write(tmp_path, "2026-06-28-url-root.md",
           frontmatter='date: 2026-06-28\ntag: migration\ntitle: 访问地址改为根路径\npin: true\nversion: "0.16.0"',
           body="正文中文")
    _write(tmp_path, "2026-06-28-url-root.en.md",
           frontmatter='date: 2026-06-28\ntag: migration\ntitle: URL moved to root',
           body="english body")

    posts = svc.list_posts(tmp_path)
    assert len(posts) == 1
    p = posts[0]
    assert p.id == "2026-06-28-url-root"
    assert p.tag == "migration"
    assert p.pin is True
    assert p.version == "0.16.0"
    assert p.title == {"zh": "访问地址改为根路径", "en": "URL moved to root"}
    assert p.body["zh"] == "正文中文"
    assert p.body["en"] == "english body"


def test_en_fallback_to_zh_when_missing(tmp_path: Path) -> None:
    _write(tmp_path, "2026-06-28-welcome.md",
           frontmatter="date: 2026-06-28\ntag: notice\ntitle: 公告栏上线",
           body="只有中文")
    posts = svc.list_posts(tmp_path)
    assert len(posts) == 1
    p = posts[0]
    # en 文件缺失 → title/body 都 fallback 中文
    assert p.title["en"] == p.title["zh"] == "公告栏上线"
    assert p.body["en"] == p.body["zh"] == "只有中文"


def test_skips_invalid_tag_and_missing_fields(tmp_path: Path) -> None:
    _write(tmp_path, "bad-tag.md",
           frontmatter="date: 2026-06-28\ntag: bogus\ntitle: x", body="b")
    _write(tmp_path, "no-date.md",
           frontmatter="tag: notice\ntitle: x", body="b")
    _write(tmp_path, "ok.md",
           frontmatter="date: 2026-06-28\ntag: notice\ntitle: ok", body="b")
    posts = svc.list_posts(tmp_path)
    assert [p.id for p in posts] == ["ok"]


def test_sort_pin_then_date_desc(tmp_path: Path) -> None:
    _write(tmp_path, "a-old-pinned.md",
           frontmatter="date: 2026-01-01\ntag: migration\ntitle: pinned\npin: true", body="b")
    _write(tmp_path, "b-new.md",
           frontmatter="date: 2026-06-28\ntag: notice\ntitle: new", body="b")
    _write(tmp_path, "c-mid.md",
           frontmatter="date: 2026-03-15\ntag: notice\ntitle: mid", body="b")
    posts = svc.list_posts(tmp_path)
    # pin 优先（即使日期最旧），其余按 date 降序
    assert [p.id for p in posts] == ["a-old-pinned", "b-new", "c-mid"]


def test_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert svc.list_posts(tmp_path / "nope") == []


def test_ignores_readme(tmp_path: Path) -> None:
    # 目录说明文件不该被当 post（也不该刷 warning）
    (tmp_path / "README.md").write_text("# 公告编写指南\n不是 post", encoding="utf-8")
    _write(tmp_path, "ok.md",
           frontmatter="date: 2026-06-28\ntag: notice\ntitle: ok", body="b")
    assert [p.id for p in svc.list_posts(tmp_path)] == ["ok"]


def test_endpoint_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "2026-06-28-welcome.md",
           frontmatter="date: 2026-06-28\ntag: notice\ntitle: hi", body="b")
    monkeypatch.setattr(svc, "ANNOUNCEMENTS_DIR", tmp_path)
    from studio.server import app

    resp = TestClient(app).get("/api/announcements")
    assert resp.status_code == 200
    body = resp.json()
    assert "posts" in body and len(body["posts"]) == 1
    post = body["posts"][0]
    assert set(post.keys()) == {"id", "date", "tag", "title", "body", "pin", "version"}
    assert post["title"] == {"zh": "hi", "en": "hi"}
