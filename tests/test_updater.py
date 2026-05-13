"""ADR 0002 self-update — updater 模块单测。

覆盖：
- current_version() smoke：跑真 git 拿 HEAD 状态（CI 上跑也行，工作区干净就 dirty=False）
- check_update() 缓存路径：写 cache + 读 cache + TTL 过期
- request_update / has_pending / apply_pending 干净退出：flag 文件读写
- apply_pending dirty tree 中止：working tree 脏时正确 abort + 写 log

不覆盖（需要网络 / 真 git fetch / pip / npm）：
- check_update fetch 失败时的 error 路径
- apply_pending 真正 pull 路径

那部分由手测覆盖（用户在本地 webui 点更新按钮验证端到端）。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from studio.services import updater


@pytest.fixture(autouse=True)
def _isolate_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 updater 模块里指向真 studio_data / tmp 的路径全部重定向到 tmp_path，
    避免污染开发机的标志文件。"""
    monkeypatch.setattr(updater, "RESTART_FLAG", tmp_path / "tmp" / "restart")
    monkeypatch.setattr(updater, "UPDATE_PENDING", tmp_path / ".update_pending")
    monkeypatch.setattr(updater, "UPDATE_CACHE", tmp_path / ".update_cache")
    monkeypatch.setattr(updater, "LAST_VERSION", tmp_path / ".last_version")
    monkeypatch.setattr(updater, "UPDATE_LOG", tmp_path / ".update_log")
    monkeypatch.setattr(updater, "UPDATE_STATUS", tmp_path / ".update_status")
    return tmp_path


# ---------------------------------------------------------------------------
# current_version
# ---------------------------------------------------------------------------

def test_current_version_smoke() -> None:
    """跑真 git，返回字段都是字符串 / bool。仓库目录里这个一定能跑。"""
    v = updater.current_version()
    assert isinstance(v.version, str) and v.version
    # commit 在 git 仓里至少是 sha 或 'unknown'
    assert isinstance(v.commit, str) and v.commit
    assert isinstance(v.commit_short, str)
    assert isinstance(v.branch, str)
    assert isinstance(v.is_dirty, bool)
    # tag 可能为 None
    assert v.tag is None or isinstance(v.tag, str)


# ---------------------------------------------------------------------------
# request_update / has_pending
# ---------------------------------------------------------------------------

def test_request_update_writes_flags(_isolate_flags: Path) -> None:
    assert not updater.has_pending()
    assert not updater.RESTART_FLAG.exists()

    updater.request_update("origin/master")

    assert updater.has_pending()
    assert updater.UPDATE_PENDING.read_text(encoding="utf-8") == "origin/master"
    assert updater.RESTART_FLAG.exists()


def test_request_update_custom_target(_isolate_flags: Path) -> None:
    updater.request_update("abc1234567")
    assert updater.UPDATE_PENDING.read_text(encoding="utf-8") == "abc1234567"


# ---------------------------------------------------------------------------
# apply_pending — 各种 abort 路径（不真 pull）
# ---------------------------------------------------------------------------

def test_apply_pending_no_pending_returns_false(_isolate_flags: Path) -> None:
    assert updater.apply_pending(emit=lambda _: None) is False


def test_apply_pending_dirty_tree_aborts(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """working tree dirty 时 apply_pending 应该 abort + 写 log + 清 .update_pending，
    不调 git fetch / reset。"""
    fake_version = updater.VersionInfo(
        version="0.0.0", commit="abc", commit_short="abc",
        commit_time_iso="", branch="master", tag=None, is_dirty=True,
    )
    monkeypatch.setattr(updater, "current_version", lambda: fake_version)

    git_calls: list[tuple] = []
    def _fake_git(*args, **kwargs):
        git_calls.append(args)
        return 0, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)

    updater.request_update("origin/master")
    result = updater.apply_pending(emit=lambda _: None)

    assert result is True  # 走过 apply 路径（即便 abort 也返 True）
    assert not updater.UPDATE_PENDING.exists()  # 清了 pending
    assert updater.UPDATE_LOG.exists()
    log = updater.UPDATE_LOG.read_text(encoding="utf-8")
    assert "[abort] working tree dirty" in log
    # 不应该调任何 git 命令（fetch / reset / pull 都不该跑）
    assert git_calls == []


def test_apply_pending_records_last_version(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """apply_pending 必须把当前 commit 写到 .last_version（rollback 用）。
    用 dirty tree 路径触发，因为它在 abort 之前已经写了。"""
    fake_version = updater.VersionInfo(
        version="0.0.0", commit="deadbeef" * 5, commit_short="deadbeef",
        commit_time_iso="", branch="master", tag=None, is_dirty=True,
    )
    monkeypatch.setattr(updater, "current_version", lambda: fake_version)
    monkeypatch.setattr(updater, "_git", lambda *a, **k: (0, "", ""))

    updater.request_update("origin/master")
    updater.apply_pending(emit=lambda _: None)

    assert updater.LAST_VERSION.exists()
    assert updater.LAST_VERSION.read_text(encoding="utf-8") == "deadbeef" * 5


# ---------------------------------------------------------------------------
# check_update — 缓存路径
# ---------------------------------------------------------------------------

def test_check_update_cache_hit(_isolate_flags: Path) -> None:
    """master 通道缓存命中：缓存还没过期就直接返回 cached 值，不调 git。"""
    cached = updater.UpdateCheckResult(
        channel="master", current_commit="abc", latest_commit="def",
        commits_ahead=3, has_update=True, latest_tag="v0.7.0",
        checked_at=time.time(),  # 刚刚
    )
    updater.UPDATE_CACHE.write_text(json.dumps(asdict(cached)), encoding="utf-8")

    result = updater.check_update(channel="master", use_cache=True)
    assert result.commits_ahead == 3
    assert result.latest_tag == "v0.7.0"
    assert result.has_update is True


def test_check_update_cache_expired(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cache TTL 超过 24h 就忽略 cache，回退到 git fetch（这里 mock）。"""
    stale = updater.UpdateCheckResult(
        channel="master", current_commit="abc", latest_commit="def",
        commits_ahead=1, has_update=True, latest_tag=None,
        checked_at=time.time() - updater.UPDATE_CACHE_TTL_SECONDS - 100,
    )
    updater.UPDATE_CACHE.write_text(json.dumps(asdict(stale)), encoding="utf-8")

    git_calls: list[tuple] = []
    def _fake_git(*args, **kwargs):
        git_calls.append(args)
        if args[:2] == ("fetch", "origin"):
            return 0, "", ""
        if args[:2] == ("rev-parse", "origin/master"):
            return 0, "newsha", ""
        if args[:3] == ("rev-list", "--count", "HEAD..origin/master"):
            return 0, "0", ""
        if args[:2] == ("rev-parse", "HEAD"):
            return 0, "newsha", ""
        return 0, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)

    result = updater.check_update(channel="master", use_cache=True)
    # 走了真 fetch 路径，cache 被覆盖
    assert any(c[:2] == ("fetch", "origin") for c in git_calls), "应当调 git fetch"
    # commits_ahead 来自新结果（0），不是旧 cache 的 1
    assert result.commits_ahead == 0


def test_check_update_force_skips_cache(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force=true (use_cache=False) 即便 cache 还新也要重 fetch。"""
    cached = updater.UpdateCheckResult(
        channel="master", current_commit="abc", latest_commit="def",
        commits_ahead=5, has_update=True, latest_tag="v0.7.0",
        checked_at=time.time(),
    )
    updater.UPDATE_CACHE.write_text(json.dumps(asdict(cached)), encoding="utf-8")

    git_calls: list[tuple] = []
    def _fake_git(*args, **kwargs):
        git_calls.append(args)
        if args[:2] == ("fetch", "origin"):
            return 0, "", ""
        if args[:2] == ("rev-parse", "origin/master"):
            return 0, "remote_sha", ""
        if args[:3] == ("rev-list", "--count", "HEAD..origin/master"):
            return 0, "2", ""
        if args[:2] == ("rev-parse", "HEAD"):
            return 0, "local_sha", ""
        if args[0] == "describe":
            return 0, "v0.8.0", ""
        return 0, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)
    monkeypatch.setattr(
        updater, "current_version",
        lambda: updater.VersionInfo(
            version="0.7.0", commit="local_sha", commit_short="local_sh",
            commit_time_iso="", branch="master", tag=None, is_dirty=False,
        ),
    )

    result = updater.check_update(channel="master", use_cache=False)
    assert any(c[:2] == ("fetch", "origin") for c in git_calls), "force 应当跳过 cache 直接 fetch"
    assert result.commits_ahead == 2


def test_check_update_fetch_failure_returns_error(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """git fetch 失败时返回 error 字段非 None，不抛异常。"""
    def _fake_git(*args, **kwargs):
        if args[:2] == ("fetch", "origin"):
            return 128, "", "fatal: unable to access 'github.com'"
        return 0, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)

    result = updater.check_update(channel="master", use_cache=False)
    assert result.error is not None
    assert "fetch failed" in result.error.lower() or "fatal" in result.error.lower()
    assert result.has_update is False


def test_check_update_invalid_channel_raises() -> None:
    with pytest.raises(ValueError, match="invalid channel"):
        updater.check_update(channel="weird-branch")


# ---------------------------------------------------------------------------
# requirements / package.json stale 检测
# ---------------------------------------------------------------------------

def test_requirements_marker_stale_no_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """没 sha256 marker 文件（全新装的 venv）应该返回 True。"""
    fake_req = tmp_path / "requirements.txt"
    fake_req.write_text("torch>=2.0\n", encoding="utf-8")
    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)
    assert updater._requirements_marker_stale() is True


def test_requirements_marker_stale_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """marker 内容与 requirements.txt sha256 一致时返回 False。"""
    import hashlib
    fake_req = tmp_path / "requirements.txt"
    content = b"torch>=2.0\nfastapi>=0.100\n"
    fake_req.write_bytes(content)
    marker = tmp_path / "venv" / ".studio-requirements.sha256"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(hashlib.sha256(content).hexdigest(), encoding="utf-8")
    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)
    assert updater._requirements_marker_stale() is False


# ---------------------------------------------------------------------------
# PR-C：update status / rollback
# ---------------------------------------------------------------------------

def test_apply_pending_writes_aborted_status(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dirty tree abort 时 .update_status 应当被写入 status='aborted'。"""
    fake = updater.VersionInfo(
        version="0.0.0", commit="abc", commit_short="abc",
        commit_time_iso="", branch="master", tag=None, is_dirty=True,
    )
    monkeypatch.setattr(updater, "current_version", lambda: fake)
    monkeypatch.setattr(updater, "_git", lambda *a, **k: (0, "", ""))

    updater.request_update("origin/master")
    updater.apply_pending(emit=lambda _: None)

    st = updater.last_status()
    assert st is not None
    assert st.status == "aborted"
    assert "dirty" in st.reason.lower()
    assert st.from_commit == "abc"


def test_apply_pending_writes_failed_status_on_fetch_error(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """git fetch 失败应当写 status='failed' + reason 含 stderr 摘要。"""
    fake = updater.VersionInfo(
        version="0.0.0", commit="abc", commit_short="abc",
        commit_time_iso="", branch="master", tag=None, is_dirty=False,
    )
    monkeypatch.setattr(updater, "current_version", lambda: fake)
    def _fake_git(*args, **kwargs):
        if args[:2] == ("fetch", "origin"):
            return 128, "", "fatal: Could not resolve host"
        return 0, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)

    updater.request_update("origin/master")
    updater.apply_pending(emit=lambda _: None)

    st = updater.last_status()
    assert st is not None
    assert st.status == "failed"
    assert "Could not resolve host" in st.reason or "git fetch" in st.reason


def test_last_status_returns_none_when_missing(_isolate_flags: Path) -> None:
    assert updater.last_status() is None


def test_rollback_target_no_file(_isolate_flags: Path) -> None:
    """没 .last_version 时返回 None。"""
    assert updater.rollback_target() is None


def test_rollback_target_validates_commit_exists(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """.last_version 存在但 commit 已 GC 时返回 None。"""
    updater.LAST_VERSION.write_text("deadbeef" * 5, encoding="utf-8")
    def _fake_git(*args, **kwargs):
        if args[:2] == ("cat-file", "-e"):
            return 1, "", "fatal: not a valid object name"
        return 0, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)
    assert updater.rollback_target() is None


def test_rollback_target_valid(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """.last_version + commit 存在 → 返回 sha。"""
    updater.LAST_VERSION.write_text("cafebabe" * 5, encoding="utf-8")
    monkeypatch.setattr(updater, "_git", lambda *a, **k: (0, "", ""))
    assert updater.rollback_target() == "cafebabe" * 5


def test_request_rollback_writes_pending(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rollback 应当走 request_update 路径，target 是 .last_version 的 sha。"""
    updater.LAST_VERSION.write_text("c0ffeec0ffee" + "0" * 28, encoding="utf-8")
    monkeypatch.setattr(updater, "_git", lambda *a, **k: (0, "", ""))

    target = updater.request_rollback()
    assert target is not None
    assert updater.has_pending()
    assert updater.UPDATE_PENDING.read_text(encoding="utf-8") == target


def test_request_rollback_returns_none_without_target(_isolate_flags: Path) -> None:
    """没 .last_version 时 request_rollback 返回 None 且不写 pending。"""
    assert updater.request_rollback() is None
    assert not updater.has_pending()


def test_read_update_log(_isolate_flags: Path) -> None:
    """read_update_log 应当返回完整文件内容。"""
    updater.UPDATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    updater.UPDATE_LOG.write_text("line 1\nline 2\n", encoding="utf-8")
    assert updater.read_update_log() == "line 1\nline 2\n"


def test_read_update_log_missing(_isolate_flags: Path) -> None:
    assert updater.read_update_log() == ""


def test_last_status_tolerates_utf8_bom(_isolate_flags: Path) -> None:
    """Windows PowerShell 5.1 写文件默认带 UTF-8 BOM；read_text(utf-8) 不剥 BOM
    导致 json.loads 抛 JSONDecodeError，UI 看到 status=null 什么都不显示。
    用 utf-8-sig 读应当透明剥 BOM 并正常 parse。"""
    BOM = "﻿"
    json_str = '{"status": "failed", "reason": "test", "target": "origin/master", ' \
               '"from_commit": "abc", "to_commit": "abc", "started_at": 1.0, ' \
               '"finished_at": 2.0, "deps_changed": false, "log_excerpt": ""}'
    updater.UPDATE_STATUS.parent.mkdir(parents=True, exist_ok=True)
    updater.UPDATE_STATUS.write_text(BOM + json_str, encoding="utf-8")

    st = updater.last_status()
    assert st is not None, "带 BOM 的 .update_status 应当被正常 parse"
    assert st.status == "failed"
    assert st.reason == "test"


def test_rollback_target_tolerates_utf8_bom(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同上，针对 .last_version 文本。"""
    BOM = "﻿"
    sha = "deadbeef" * 5
    updater.LAST_VERSION.parent.mkdir(parents=True, exist_ok=True)
    updater.LAST_VERSION.write_text(BOM + sha, encoding="utf-8")
    monkeypatch.setattr(updater, "_git", lambda *a, **k: (0, "", ""))
    assert updater.rollback_target() == sha


# ---------------------------------------------------------------------------
# target_has_self_update (chunk 4 safety net)
# ---------------------------------------------------------------------------


def test_target_has_self_update_true_when_marker_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """git cat-file -e <ref>:studio/services/updater.py 返 0 → True。"""
    monkeypatch.setattr(updater, "_git",
                       lambda *args, **_k: (0, "", "") if args[0] == "cat-file" else (1, "", ""))
    assert updater.target_has_self_update("any-ref") is True


def test_target_has_self_update_false_when_marker_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """marker 文件不存在（pre-self-update commit）→ False。"""
    monkeypatch.setattr(updater, "_git", lambda *args, **_k: (1, "", "does not exist"))
    assert updater.target_has_self_update("ancient-commit") is False


def test_target_has_self_update_false_on_git_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """git 失败（ref 无效 / 仓库损坏）→ 保守返 False，让 preflight 阻断。"""
    monkeypatch.setattr(updater, "_git", lambda *args, **_k: (128, "", "fatal: invalid object"))
    assert updater.target_has_self_update("garbage") is False


# ---------------------------------------------------------------------------
# dev_commits (chunk 3)
# ---------------------------------------------------------------------------


def _fake_git_factory(plans: dict[tuple[str, ...], tuple[int, str, str]]):
    """构造 _git 假实现：根据传入的 args tuple 匹配 plans 返回（rc, out, err）。
    未命中 → (1, '', 'no plan')。
    """
    def fake(*args: str, **_kw):
        return plans.get(args, (1, "", "no plan for: " + " ".join(args)))
    return fake


def test_dev_commits_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch + log 都成功 → 解析出 commits 列表，fetched=True 无 error。"""
    log_out = "\x00".join(["a" * 40, "aaaaaaaa", "first msg", "2026-05-13T11:00:00+00:00", "alice"]) + "\n" \
            + "\x00".join(["b" * 40, "bbbbbbbb", "second msg", "2026-05-12T22:00:00+00:00", "bob"])
    plans = {
        ("fetch", "origin", "dev"): (0, "", ""),
        ("log", "-10", "--format=%H%x00%h%x00%s%x00%cI%x00%an", "origin/dev"): (0, log_out, ""),
    }
    monkeypatch.setattr(updater, "_git", _fake_git_factory(plans))
    r = updater.dev_commits(limit=10)
    assert r.fetched is True
    assert r.error is None
    assert len(r.commits) == 2
    assert r.commits[0].sha == "a" * 40
    assert r.commits[0].short_sha == "aaaaaaaa"
    assert r.commits[0].msg == "first msg"
    assert r.commits[0].author == "alice"
    assert r.commits[1].sha == "b" * 40


def test_dev_commits_limit_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    """limit < 1 or > 50 → clamp 到 [1, 50]。"""
    captured: list[tuple[str, ...]] = []
    def fake(*args: str, **_kw):
        captured.append(args)
        if args[0] == "fetch":
            return (0, "", "")
        return (0, "", "")
    monkeypatch.setattr(updater, "_git", fake)
    updater.dev_commits(limit=999)
    log_call = next(a for a in captured if a[0] == "log")
    assert "-50" in log_call
    captured.clear()
    updater.dev_commits(limit=0)
    log_call = next(a for a in captured if a[0] == "log")
    assert "-1" in log_call


def test_dev_commits_fetch_fails_but_log_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """git fetch 失败（离线）但本地 origin/dev 缓存还有 → commits 仍返回，
    fetched=False + error 文案给 UI 提示陈旧。"""
    log_out = "\x00".join(["c" * 40, "cccccccc", "cached msg", "2026-05-01T00:00:00+00:00", "you"])
    plans = {
        ("fetch", "origin", "dev"): (1, "", "Could not resolve host: github.com"),
        ("log", "-10", "--format=%H%x00%h%x00%s%x00%cI%x00%an", "origin/dev"): (0, log_out, ""),
    }
    monkeypatch.setattr(updater, "_git", _fake_git_factory(plans))
    r = updater.dev_commits(limit=10)
    assert r.fetched is False
    assert r.error is not None and "Could not resolve" in r.error
    assert len(r.commits) == 1
    assert r.commits[0].short_sha == "cccccccc"


def test_dev_commits_no_origin_dev_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    """`origin/dev` 不存在（首次 clone 没跟，或远端删了）→ commits=[]，
    带 error 给 UI 显示。"""
    plans = {
        ("fetch", "origin", "dev"): (1, "", "fatal: couldn't find remote ref dev"),
        ("log", "-10", "--format=%H%x00%h%x00%s%x00%cI%x00%an", "origin/dev"): (128, "", "fatal: ambiguous argument 'origin/dev'"),
    }
    monkeypatch.setattr(updater, "_git", _fake_git_factory(plans))
    r = updater.dev_commits(limit=10)
    assert r.fetched is False
    assert r.commits == []
    assert r.error is not None


def test_dev_commits_malformed_log_lines_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """字段不够 5 个的行（比如 commit msg 含 NUL 字符这种异常情况）跳过，不抛。"""
    log_out = "broken_line_without_nul\n" + \
              "\x00".join(["d" * 40, "dddddddd", "ok msg", "2026-05-13T00:00:00+00:00", "alice"])
    plans = {
        ("fetch", "origin", "dev"): (0, "", ""),
        ("log", "-10", "--format=%H%x00%h%x00%s%x00%cI%x00%an", "origin/dev"): (0, log_out, ""),
    }
    monkeypatch.setattr(updater, "_git", _fake_git_factory(plans))
    r = updater.dev_commits(limit=10)
    assert len(r.commits) == 1
    assert r.commits[0].short_sha == "dddddddd"
