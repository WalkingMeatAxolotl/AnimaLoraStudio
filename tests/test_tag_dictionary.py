"""Tag 翻译词典：解析 / 上传 / 下载 / API 端点。"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studio.infrastructure import tag_dictionary as td
from studio import server


@pytest.fixture
def tag_dict_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """所有读写都落到 tmp_path/tag_dictionary/，每个 test 隔离。"""
    d = tmp_path / "tag_dictionary"
    monkeypatch.setattr(td, "TAG_DICT_DIR", d)
    monkeypatch.setattr(td, "ACTIVE_JSON", d / "active.json")
    monkeypatch.setattr(td, "SOURCE_FILE", d / "source.csv")
    monkeypatch.setattr(td, "SOURCE_SQLITE", d / "source.sqlite")
    return d


def _sqlite_bytes(tmp_path: Path, rows: list[tuple[str, int, str, int]]) -> bytes:
    """构造默认源 schema 的 sqlite 文件，返回字节（喂给 fake requests）。"""
    p = tmp_path / "_make_tag.sqlite"
    p.unlink(missing_ok=True)
    conn = sqlite3.connect(p)
    conn.execute(
        "CREATE TABLE tags (name TEXT PRIMARY KEY, category INTEGER, "
        "cn_name TEXT, post_count INTEGER)"
    )
    conn.executemany("INSERT INTO tags VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return p.read_bytes()


@pytest.fixture
def client(tag_dict_dir: Path) -> TestClient:  # noqa: ARG001 (fixture chains the patch)
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# parse_csv
# ---------------------------------------------------------------------------


def test_parse_two_columns_basic() -> None:
    text = "1girl,1女孩\nsolo,单人\n"
    entries = td.parse_csv(text)
    assert entries == {"1girl": ["1女孩"], "solo": ["单人"]}


def test_parse_multiple_aliases_split_by_whitespace() -> None:
    text = "breasts,胸部 乳房 oppai\n"
    entries = td.parse_csv(text)
    assert entries["breasts"] == ["胸部", "乳房", "oppai"]


def test_parse_single_column_no_translation() -> None:
    text = "rare_tag\n"
    entries = td.parse_csv(text)
    # 仍参与英文 prefix 补全
    assert entries == {"rare tag": []}


def test_parse_underscore_to_space_in_english() -> None:
    text = "long_hair,长发\nblack_hair,黑发\n"
    entries = td.parse_csv(text)
    assert "long hair" in entries
    assert "black hair" in entries
    assert "long_hair" not in entries


def test_parse_skips_blank_and_comments() -> None:
    text = "# comment\n\n  \n1girl,1女孩\n# another\n"
    entries = td.parse_csv(text)
    assert entries == {"1girl": ["1女孩"]}


def test_parse_truncates_at_max_entries(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(td, "MAX_ENTRIES", 3)
    text = "\n".join(f"tag{i},翻译{i}" for i in range(10))
    with caplog.at_level("WARNING"):
        entries = td.parse_csv(text)
    assert len(entries) == 3
    assert any("超过" in r.message or "上限" in r.message for r in caplog.records)


def test_parse_first_comma_only() -> None:
    # tail 里有多个 token (空白分隔)；逗号只在 head/tail 分割时切第一个
    text = "1girl,1女孩 一女 a girl\n"
    entries = td.parse_csv(text)
    assert entries["1girl"] == ["1女孩", "一女", "a", "girl"]


# ---------------------------------------------------------------------------
# parse_sqlite（默认源格式）
# ---------------------------------------------------------------------------


def test_parse_sqlite_orders_by_post_count_desc(tmp_path: Path) -> None:
    _sqlite_bytes(tmp_path, [
        ("solo", 0, "单人", 100),
        ("1girl", 0, "1个女孩", 300),
        ("smile", 0, "微笑", 200),
    ])
    entries = td.parse_sqlite(tmp_path / "_make_tag.sqlite")
    # dict 插入序 = post_count 降序 = 前端 autocomplete 热度序
    assert list(entries.keys()) == ["1girl", "smile", "solo"]
    assert entries["1girl"] == ["1个女孩"]


def test_parse_sqlite_underscore_to_space(tmp_path: Path) -> None:
    _sqlite_bytes(tmp_path, [("long_hair", 0, "长发", 10)])
    entries = td.parse_sqlite(tmp_path / "_make_tag.sqlite")
    assert entries == {"long hair": ["长发"]}


def test_parse_sqlite_keeps_cn_name_with_spaces_whole(tmp_path: Path) -> None:
    # cn_name 是单个译名，含空格也不能按空白切成多个别名
    _sqlite_bytes(tmp_path, [("pokemon_(creature)", 0, "宝可梦 (生物)", 10)])
    entries = td.parse_sqlite(tmp_path / "_make_tag.sqlite")
    assert entries["pokemon (creature)"] == ["宝可梦 (生物)"]


def test_parse_sqlite_empty_cn_name_keeps_tag(tmp_path: Path) -> None:
    _sqlite_bytes(tmp_path, [("rare_tag", 0, "", 10), ("none_tag", 0, None, 5)])
    entries = td.parse_sqlite(tmp_path / "_make_tag.sqlite")
    assert entries == {"rare tag": [], "none tag": []}


def test_parse_sqlite_truncates_to_max_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(td, "MAX_ENTRIES", 2)
    _sqlite_bytes(tmp_path, [(f"tag{i}", 0, f"翻译{i}", i) for i in range(10)])
    entries = td.parse_sqlite(tmp_path / "_make_tag.sqlite")
    # 取 post_count 最高的前 2 条
    assert list(entries.keys()) == ["tag9", "tag8"]


def test_parse_sqlite_garbage_raises(tmp_path: Path) -> None:
    p = tmp_path / "garbage.sqlite"
    p.write_bytes(b"this is not a sqlite file at all, padding padding padding")
    with pytest.raises(sqlite3.Error):
        td.parse_sqlite(p)


# ---------------------------------------------------------------------------
# apply_uploaded
# ---------------------------------------------------------------------------


def test_apply_uploaded_writes_active_and_returns_meta(tag_dict_dir: Path) -> None:
    content = b"1girl,1\xe5\xa5\xb3\xe5\xad\xa9\nsolo,\xe5\x8d\x95\xe4\xba\xba\n"
    meta = td.apply_uploaded(content, "my.csv")
    assert meta["kind"] == "user"
    assert meta["source_name"] == "my.csv"
    assert meta["entry_count"] == 2
    assert td.ACTIVE_JSON.exists()
    loaded = td.load_active()
    assert loaded is not None
    entries, m = loaded
    assert entries["1girl"] == ["1女孩"]
    assert m["kind"] == "user"


def test_apply_uploaded_rejects_oversize(
    tag_dict_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(td, "MAX_BYTES", 10)
    with pytest.raises(ValueError, match="过大"):
        td.apply_uploaded(("1girl,1女孩\nsolo,单人\nfoo,bar\n" * 10).encode("utf-8"), "x.csv")


def test_apply_uploaded_rejects_zero_entries(tag_dict_dir: Path) -> None:
    with pytest.raises(ValueError, match="0 条"):
        td.apply_uploaded(b"# only comments\n\n", "empty.csv")


def test_apply_uploaded_rejects_non_utf8(tag_dict_dir: Path) -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        td.apply_uploaded(b"\xff\xfe invalid", "bad.csv")


# ---------------------------------------------------------------------------
# download_default
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, content: bytes, status: int = 200) -> None:
        self.content = content
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_download_default_writes_active(
    tag_dict_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _sqlite_bytes(tmp_path, [
        ("1girl", 0, "1个女孩", 300),
        ("solo", 0, "单人", 100),
    ])
    monkeypatch.setattr(td.requests, "get", lambda url, timeout: _FakeResp(content))
    meta = td.download_default()
    assert meta["kind"] == "default"
    assert meta["source_name"] == td.DEFAULT_SOURCE_NAME
    assert meta["entry_count"] == 2
    loaded = td.load_active()
    assert loaded is not None
    entries, _ = loaded
    assert entries["1girl"] == ["1个女孩"]
    # 留底是 sqlite，且没有半截 tmp 文件
    assert td.SOURCE_SQLITE.exists()
    assert not td.SOURCE_SQLITE.with_suffix(".sqlite.tmp").exists()


def test_download_default_removes_stale_csv_source(
    tag_dict_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """先有用户上传的 csv 留底，恢复默认后留底只剩 sqlite（互斥语义）。"""
    td.apply_uploaded("mytag,我的标签\n".encode("utf-8"), "my.csv")
    assert td.SOURCE_FILE.exists()
    content = _sqlite_bytes(tmp_path, [("1girl", 0, "1个女孩", 300)])
    monkeypatch.setattr(td.requests, "get", lambda url, timeout: _FakeResp(content))
    td.download_default()
    assert td.SOURCE_SQLITE.exists()
    assert not td.SOURCE_FILE.exists()


def test_apply_uploaded_removes_stale_sqlite_source(
    tag_dict_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _sqlite_bytes(tmp_path, [("1girl", 0, "1个女孩", 300)])
    monkeypatch.setattr(td.requests, "get", lambda url, timeout: _FakeResp(content))
    td.download_default()
    assert td.SOURCE_SQLITE.exists()
    td.apply_uploaded("mytag,我的标签\n".encode("utf-8"), "my.csv")
    assert td.SOURCE_FILE.exists()
    assert not td.SOURCE_SQLITE.exists()


def test_download_default_propagates_http_error(
    tag_dict_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(td.requests, "get", lambda url, timeout: _FakeResp(b"", 404))
    with pytest.raises(RuntimeError, match="download failed"):
        td.download_default()


def test_download_default_rejects_oversize(
    tag_dict_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(td, "MAX_DOWNLOAD_BYTES", 10)
    monkeypatch.setattr(td.requests, "get", lambda url, timeout: _FakeResp(b"x" * 11))
    with pytest.raises(RuntimeError, match="too large"):
        td.download_default()


def test_download_default_rejects_non_sqlite(
    tag_dict_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        td.requests,
        "get",
        lambda url, timeout: _FakeResp(b"english_tag,zh -- old csv format, not sqlite"),
    )
    with pytest.raises(RuntimeError, match="invalid sqlite"):
        td.download_default()
    # 解析失败不留半截留底
    assert not td.SOURCE_SQLITE.exists()
    assert not td.SOURCE_SQLITE.with_suffix(".sqlite.tmp").exists()


def test_download_default_rejects_empty_parse(
    tag_dict_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _sqlite_bytes(tmp_path, [])
    monkeypatch.setattr(td.requests, "get", lambda url, timeout: _FakeResp(content))
    with pytest.raises(RuntimeError, match="zero entries"):
        td.download_default()


# ---------------------------------------------------------------------------
# load_active
# ---------------------------------------------------------------------------


def test_load_active_returns_none_when_missing(tag_dict_dir: Path) -> None:
    assert td.load_active() is None
    assert td.get_meta() is None


def test_load_active_returns_none_when_corrupt(tag_dict_dir: Path) -> None:
    td.TAG_DICT_DIR.mkdir(parents=True, exist_ok=True)
    td.ACTIVE_JSON.write_text("not json", encoding="utf-8")
    assert td.load_active() is None


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


def test_get_meta_endpoint_uninitialized(client: TestClient) -> None:
    r = client.get("/api/tag-dictionary/meta")
    assert r.status_code == 200
    assert r.json() == {"loaded": False, "meta": None}


def test_get_data_endpoint_404_when_uninitialized(client: TestClient) -> None:
    r = client.get("/api/tag-dictionary/data")
    assert r.status_code == 404


def test_upload_endpoint_persists(client: TestClient, tag_dict_dir: Path) -> None:
    files = {"file": ("custom.csv", "mytag,我的标签\n".encode("utf-8"), "text/csv")}
    r = client.post("/api/tag-dictionary/upload", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["loaded"] is True
    assert body["meta"]["source_name"] == "custom.csv"
    assert body["meta"]["kind"] == "user"
    # 后续 GET 能拿到
    r = client.get("/api/tag-dictionary/data")
    assert r.status_code == 200
    data = r.json()
    assert data["entries"]["mytag"] == ["我的标签"]


def test_get_data_endpoint_keys_preserve_file_order(
    client: TestClient, tag_dict_dir: Path
) -> None:
    # 纯数字 tag（"69"）会被 JS 对象重排到最前；keys 数组是前端唯一可靠的
    # 行序来源，必须严格等于文件行序
    csv = "1girl,1女孩\n69,六九\nsolo,单人\n"
    files = {"file": ("o.csv", csv.encode("utf-8"), "text/csv")}
    assert client.post("/api/tag-dictionary/upload", files=files).status_code == 200
    r = client.get("/api/tag-dictionary/data")
    assert r.status_code == 200
    assert r.json()["keys"] == ["1girl", "69", "solo"]


def test_upload_endpoint_400_on_bad_file(client: TestClient) -> None:
    files = {"file": ("empty.csv", b"# comments only\n", "text/csv")}
    r = client.post("/api/tag-dictionary/upload", files=files)
    assert r.status_code == 400


def test_reset_endpoint_502_on_network_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(url: str, timeout: int) -> Any:
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(td.requests, "get", _boom)
    r = client.post("/api/tag-dictionary/reset")
    assert r.status_code == 502
    err = r.json()["error"]
    assert "network unreachable" in (err.get("message", "") + str(err.get("details", "")))


def test_reset_endpoint_success(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
    tag_dict_dir: Path, tmp_path: Path,
) -> None:
    content = _sqlite_bytes(tmp_path, [("1girl", 0, "1个女孩", 300)])
    monkeypatch.setattr(td.requests, "get", lambda url, timeout: _FakeResp(content))
    r = client.post("/api/tag-dictionary/reset")
    assert r.status_code == 200
    assert r.json()["meta"]["kind"] == "default"
