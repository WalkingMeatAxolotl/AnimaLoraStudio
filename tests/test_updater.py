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
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from studio.services.runtime import updater


@pytest.fixture(autouse=True)
def _isolate_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 updater 模块里指向真 studio_data / tmp 的路径全部重定向到 tmp_path，
    避免污染开发机的标志文件。"""
    monkeypatch.setattr(updater, "RESTART_FLAG", tmp_path / "tmp" / "restart")
    monkeypatch.setattr(updater, "UPDATE_PENDING", tmp_path / ".update_pending")
    monkeypatch.setattr(updater, "UPDATE_FORCE", tmp_path / ".update_force")
    monkeypatch.setattr(updater, "UPDATE_CACHE", tmp_path / ".update_cache")
    monkeypatch.setattr(updater, "LAST_VERSION", tmp_path / ".last_version")
    monkeypatch.setattr(updater, "UPDATE_LOG", tmp_path / ".update_log")
    monkeypatch.setattr(updater, "UPDATE_STATUS", tmp_path / ".update_status")
    monkeypatch.setattr(updater, "PRESERVE_HOLDING", tmp_path / ".update_preserve")
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
# _porcelain_dirty_paths — allowlist 剔除（自动 churn 文件不算 dirty）
# ---------------------------------------------------------------------------

def test_porcelain_dirty_paths_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """git status --porcelain 输出空 → 工作树干净。"""
    monkeypatch.setattr(updater, "_git", lambda *a, **k: (0, "", ""))
    assert updater._porcelain_dirty_paths() == []


def test_porcelain_dirty_paths_excludes_lockfile(monkeypatch: pytest.MonkeyPatch) -> None:
    """唯一改动是 package-lock.json（npm 按本机版本自动 churn）→ 视为干净。

    用户报的 bug：新 npm 启动期改写 lockfile 补 "peer": true → working tree
    凭空 dirty → 自更新 precondition 闸死。allowlist 剔除后不再触发。
    """
    monkeypatch.setattr(
        updater, "_git",
        lambda *a, **k: (0, " M studio/web/package-lock.json", ""),
    )
    assert updater._porcelain_dirty_paths() == []


def test_porcelain_dirty_paths_keeps_real_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """lockfile churn + 真实手改文件混在一起 → 只剩真实改动，仍算 dirty。"""
    out = " M studio/web/package-lock.json\n M runtime/anima_train.py"
    monkeypatch.setattr(updater, "_git", lambda *a, **k: (0, out, ""))
    assert updater._porcelain_dirty_paths() == ["runtime/anima_train.py"]


def test_porcelain_dirty_paths_rename_uses_new_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """重命名行 `R  old -> new` 取箭头右侧的新路径。"""
    monkeypatch.setattr(updater, "_git", lambda *a, **k: (0, "R  a/old.py -> a/new.py", ""))
    assert updater._porcelain_dirty_paths() == ["a/new.py"]


def test_current_version_clean_when_only_lockfile_churn(monkeypatch: pytest.MonkeyPatch) -> None:
    """current_version().is_dirty 跟随 _porcelain_dirty_paths：仅 lockfile churn →
    is_dirty=False（这条把"自更新被 lockfile 卡死"的回归守住）。"""
    monkeypatch.setattr(updater, "_porcelain_dirty_paths", lambda: [])
    monkeypatch.setattr(updater, "git_repo_status",
                        lambda: updater.GitRepoStatus(True, True, True))
    monkeypatch.setattr(updater, "_git", lambda *a, **k: (0, "abc123def456", ""))
    assert updater.current_version().is_dirty is False
    monkeypatch.setattr(updater, "_porcelain_dirty_paths", lambda: ["runtime/x.py"])
    assert updater.current_version().is_dirty is True


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


def test_request_update_no_force_marker_by_default(_isolate_flags: Path) -> None:
    updater.request_update("origin/master")
    assert not updater.UPDATE_FORCE.exists()


def test_request_update_force_writes_marker(_isolate_flags: Path) -> None:
    updater.request_update("origin/master", force=True)
    assert updater.UPDATE_FORCE.exists()


def test_request_update_clears_stale_force_marker(_isolate_flags: Path) -> None:
    """上次写过 force marker 但没消费；本次普通（非 force）请求必须清掉它，
    避免普通更新被误判成 force 覆盖 dirty 工作树。"""
    updater.UPDATE_FORCE.parent.mkdir(parents=True, exist_ok=True)
    updater.UPDATE_FORCE.touch()
    updater.request_update("origin/master")  # force 默认 False
    assert not updater.UPDATE_FORCE.exists()


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
        installed_kind="custom", installed_label="自定义（master @ abc）",
        stable_version=None,
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


def test_apply_pending_force_overwrites_dirty(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force marker 存在时，dirty 工作树**不**中止：照常 git fetch + reset --hard
    （覆盖本地改动），收尾清掉 force marker，status=ok。"""
    fake = updater.VersionInfo(
        version="0.0.0", commit="abc", commit_short="abc",
        commit_time_iso="", branch="master", tag=None, is_dirty=True,
        installed_kind="custom", installed_label="自定义（master @ abc）",
        stable_version=None,
    )
    monkeypatch.setattr(updater, "current_version", lambda: fake)
    monkeypatch.setattr(updater, "_requirements_marker_stale", lambda: False)
    monkeypatch.setattr(updater, "_package_json_changed", lambda: False)

    git_calls: list[tuple] = []
    def _fake_git(*args, **kwargs):
        git_calls.append(args)
        return 0, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)

    updater.request_update("origin/master", force=True)
    result = updater.apply_pending(emit=lambda _: None)

    assert result is True
    # 没 abort：fetch + reset --hard 都跑了
    assert any(c[:2] == ("fetch", "origin") for c in git_calls), "force 路径应当 git fetch"
    assert any(c[0] == "reset" and "--hard" in c for c in git_calls), "force 路径应当 reset --hard"
    # force marker + pending 都清了
    assert not updater.UPDATE_FORCE.exists()
    assert not updater.UPDATE_PENDING.exists()
    st = updater.last_status()
    assert st is not None and st.status == "ok"
    log = updater.UPDATE_LOG.read_text(encoding="utf-8")
    assert "[force]" in log and "[abort]" not in log


def test_apply_pending_records_last_version(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """apply_pending 必须把当前 commit 写到 .last_version（rollback 用）。
    用 dirty tree 路径触发，因为它在 abort 之前已经写了。"""
    fake_version = updater.VersionInfo(
        version="0.0.0", commit="deadbeef" * 5, commit_short="deadbeef",
        commit_time_iso="", branch="master", tag=None, is_dirty=True,
        installed_kind="custom", installed_label="自定义（master @ deadbeef）",
        stable_version=None,
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
            installed_kind="stable", installed_label="v0.7.0",
            stable_version="v0.7.0",
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


def test_check_update_master_fetch_uses_tags(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """git fetch 必须带 --tags：显式 refspec 关闭 auto-follow-tags，缺 --tags
    会让新 release tag 进不来，下游 describe fallback 错把旧 tag 当 latest。
    （0.10.2 用户报"已是最新"实测撞到的根因）"""
    git_calls: list[tuple] = []
    def _fake_git(*args, **kwargs):
        git_calls.append(args)
        if args[:2] == ("fetch", "origin"):
            return 0, "", ""
        if args[:2] == ("rev-parse", "origin/master"):
            return 0, "remote_sha", ""
        if args[0] == "rev-list":
            return 0, "0", ""
        if args[:2] == ("rev-parse", "HEAD"):
            return 0, "remote_sha", ""
        if args[0] == "describe":
            return 1, "", ""
        return 0, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)
    monkeypatch.setattr(
        updater, "current_version",
        lambda: updater.VersionInfo(
            version="0.10.2", commit="remote_sha", commit_short="remote_s",
            commit_time_iso="", branch="master", tag=None, is_dirty=False,
            installed_kind="stable", installed_label="v0.10.2",
            stable_version="v0.10.2",
        ),
    )

    updater.check_update(channel="master", use_cache=False)
    fetch_calls = [c for c in git_calls if c[:2] == ("fetch", "origin")]
    assert fetch_calls, "应当调 git fetch"
    assert any("--tags" in c for c in fetch_calls), \
        "git fetch 必须带 --tags 才能拉新 release tag"


def test_check_update_returns_update_available_when_new_release_tag(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0.10.2 用户，远端发了 v0.11.0：fetch 带 --tags 后 describe --exact-match
    成功，latest_version=v0.11.0，状态机正确报 update_available。"""
    def _fake_git(*args, **kwargs):
        if args[:2] == ("fetch", "origin"):
            return 0, "", ""
        if args[:2] == ("rev-parse", "origin/master"):
            return 0, "v0_11_0_sha", ""
        if args[:3] == ("rev-list", "--count", "HEAD..origin/master"):
            return 0, "37", ""
        if args[:3] == ("rev-list", "--count", "origin/master..HEAD"):
            return 0, "0", ""
        if args[:2] == ("rev-parse", "HEAD"):
            return 0, "v0_10_2_sha", ""
        if args[:3] == ("describe", "--tags", "--exact-match"):
            return 0, "v0.11.0", ""
        return 0, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)
    monkeypatch.setattr(
        updater, "current_version",
        lambda: updater.VersionInfo(
            version="0.10.2", commit="v0_10_2_sha", commit_short="v0_10_2_",
            commit_time_iso="", branch="master", tag="v0.10.2", is_dirty=False,
            installed_kind="stable", installed_label="v0.10.2",
            stable_version="v0.10.2",
        ),
    )

    result = updater.check_update(channel="master", use_cache=False)
    assert result.state == "update_available"
    assert result.installed_version == "v0.10.2"
    assert result.latest_version == "v0.11.0"
    assert result.behind_count == 37


def test_check_update_does_not_use_abbrev0_tag_as_latest_version(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """defense in depth：远端 master 处于「上次 release 之后、下次 release 之前」
    时（HEAD 没打 tag，--exact-match 失败），--abbrev=0 fallback 返回上次 release
    tag。这个 tag 只能用于 UI 显示（latest_tag），**不能**赋给 latest_version —
    否则状态机会把 installed_version == latest_version 误判成 up_to_date，
    丢掉 release 后 dev 合到 master 的中间更新。"""
    def _fake_git(*args, **kwargs):
        if args[:2] == ("fetch", "origin"):
            return 0, "", ""
        if args[:2] == ("rev-parse", "origin/master"):
            return 0, "mid_release_sha", ""
        if args[:3] == ("rev-list", "--count", "HEAD..origin/master"):
            return 0, "5", ""
        if args[:3] == ("rev-list", "--count", "origin/master..HEAD"):
            return 0, "0", ""
        if args[:2] == ("rev-parse", "HEAD"):
            return 0, "v0_10_2_sha", ""
        if args[:3] == ("describe", "--tags", "--exact-match"):
            return 1, "", "fatal: no exact match"
        if args[:3] == ("describe", "--tags", "--abbrev=0"):
            return 0, "v0.10.2", ""
        return 0, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)
    monkeypatch.setattr(
        updater, "current_version",
        lambda: updater.VersionInfo(
            version="0.10.2", commit="v0_10_2_sha", commit_short="v0_10_2_",
            commit_time_iso="", branch="master", tag="v0.10.2", is_dirty=False,
            installed_kind="stable", installed_label="v0.10.2",
            stable_version="v0.10.2",
        ),
    )

    result = updater.check_update(channel="master", use_cache=False)
    assert result.latest_tag == "v0.10.2"
    assert result.latest_version is None, \
        "--abbrev=0 fallback 不应赋给 latest_version，避免误报 up_to_date"
    assert result.state == "update_available"
    assert result.behind_count == 5


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


def test_package_json_changed_when_lockfile_newer_than_node_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    web_dir = root / "studio" / "web"
    node_modules = web_dir / "node_modules"
    node_modules.mkdir(parents=True)
    marker = node_modules / ".package-lock.json"
    marker.write_text("{}", encoding="utf-8")
    pkg = web_dir / "package.json"
    pkg.write_text("{}", encoding="utf-8")
    lock = web_dir / "package-lock.json"
    lock.write_text("{}", encoding="utf-8")
    os.utime(marker, (100, 100))
    os.utime(pkg, (100, 100))
    os.utime(lock, (200, 200))

    monkeypatch.setattr(updater, "REPO_ROOT", root)

    assert updater._package_json_changed() is True


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
        installed_kind="custom", installed_label="自定义（master @ abc）",
        stable_version=None,
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
        installed_kind="custom", installed_label="自定义（master @ abc）",
        stable_version=None,
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
# 0.8.1 hotfix：zip 解压用户的 git 仓库自动 init
# ---------------------------------------------------------------------------

def test_git_repo_status_git_binary_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """git 不在 PATH 时 git_repo_status 三 False —— 后续 .git/origin 检测都跳过。"""
    monkeypatch.setattr(updater, "_git",
                        lambda *a, **k: (1, "", "git not found on PATH"))
    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)
    s = updater.git_repo_status()
    assert s.git_available is False
    assert s.has_dot_git is False
    assert s.has_origin is False
    assert s.is_repo is False


def test_git_repo_status_no_dot_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """zip 解压典型场景：git 在 PATH 但目录里没 .git/。"""
    monkeypatch.setattr(updater, "_git",
                        lambda *a, **k: (0, "git version 2.40.0", ""))
    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)  # 空 tmp_path 没 .git/
    s = updater.git_repo_status()
    assert s.git_available is True
    assert s.has_dot_git is False
    assert s.has_origin is False
    assert s.is_repo is False


def test_git_repo_status_no_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """.git/ 存在但 remote get-url origin 失败（用户自己 git init 过但没加 remote）。"""
    (tmp_path / ".git").mkdir()
    def _fake_git(*args, **_):
        if args == ("--version",):
            return 0, "git version 2.40.0", ""
        if args[:2] == ("remote", "get-url"):
            return 2, "", "error: No such remote 'origin'"
        return 0, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)
    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)
    s = updater.git_repo_status()
    assert s.git_available is True
    assert s.has_dot_git is True
    assert s.has_origin is False
    assert s.is_repo is False


def test_git_repo_status_full_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """git clone 用户的正常路径：三个都 True。"""
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(updater, "_git", lambda *a, **k: (0, "ok", ""))
    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)
    s = updater.git_repo_status()
    assert s.is_repo is True


def test_current_version_zip_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """zip 模式（is_repo=False）→ installed_kind='zip' + is_git_repo=False，
    不会去调真 git 命令（不会 rev-parse / log / describe）。"""
    git_calls: list[tuple] = []
    def _fake_git(*args, **_):
        git_calls.append(args)
        # --version 仍要回 0（git binary 在 PATH，只是没 .git/）
        if args == ("--version",):
            return 0, "git version 2.40.0", ""
        # 其他都不应被调用 —— 早 return 走 zip 分支
        return 0, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)
    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)

    v = updater.current_version()
    assert v.installed_kind == "zip"
    assert v.is_git_repo is False
    assert v.git_available is True
    assert v.commit == "unknown"
    assert v.branch == "detached"
    assert "zip" in v.installed_label.lower() or "zip 安装" in v.installed_label
    # 早 return：除了 --version 不该调任何 git
    other_calls = [c for c in git_calls if c != ("--version",)]
    assert other_calls == [], f"zip 模式不该调 git 子命令，实际调了 {other_calls}"


def test_current_version_zip_mode_no_git_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """zip + 没装 git → git_available=False（前端用这个区分文案）。"""
    monkeypatch.setattr(updater, "_git", lambda *a, **k: (1, "", "git not found"))
    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)
    v = updater.current_version()
    assert v.is_git_repo is False
    assert v.git_available is False
    assert v.installed_kind == "zip"


def test_bootstrap_aborts_without_git_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """bootstrap 第一关：git binary 不在 PATH 直接返 ok=False，error 文案
    指向 git-scm 下载页（前端给用户提示）。"""
    monkeypatch.setattr(updater, "_git", lambda *a, **k: (1, "", "git not found"))
    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)
    r = updater.bootstrap_git_repo()
    assert r.ok is False
    assert r.error is not None
    assert "git" in r.error.lower()


def test_bootstrap_full_sequence_uses_version_tag_anchor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """zip 用户 happy path：__version__ 对应的 release tag 在远端存在 →
    bootstrap 把 HEAD anchor 在 vX.Y.Z tag 上（让 working tree 看起来"干净"）。

    验证调用序列：init → symbolic-ref master → remote add origin → fetch
    --tags → rev-parse tag（成功）→ reset --hard tag → rev-parse anchor。"""
    calls: list[tuple] = []
    def _fake_git(*args, **_):
        calls.append(args)
        if args == ("--version",):
            return 0, "git version 2.40.0", ""
        if args[:2] == ("remote", "get-url"):
            # init 后还没加 origin → 失败一次（触发 remote add）
            return 2, "", "no remote"
        if args[0] == "rev-parse" and args[1] == "--verify":
            # v{__version__} tag 存在
            return 0, "abc123", ""
        if args[0] == "rev-parse":
            # 末尾的 rev-parse anchor → 返回 sha
            return 0, "abc123def456" + "0" * 20, ""
        return 0, "", ""

    monkeypatch.setattr(updater, "_git", _fake_git)
    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)  # 空 → has_dot_git=False
    monkeypatch.setattr(updater, "__version__", "0.8.0")

    r = updater.bootstrap_git_repo()
    assert r.ok is True, f"bootstrap should succeed, error={r.error}"
    assert r.anchor_kind == "version_tag"
    assert r.anchor and len(r.anchor) > 0

    # 序列验证：必须包含 init / symbolic-ref / remote add / fetch / reset
    first_args = [c[0] for c in calls]
    assert "init" in first_args
    assert "symbolic-ref" in first_args
    # remote add 应当被调（因为前面 get-url 失败）
    assert any(c[:2] == ("remote", "add") and c[2] == "origin" for c in calls), \
        "应当调 git remote add origin"
    # fetch origin master --tags
    assert any(c[:3] == ("fetch", "origin", "master") for c in calls)
    # reset --hard 到 anchor（v{__version__}）—— 强制对齐 working tree，
    # 避免 npm install 等启动期修改导致 init 完就 dirty
    assert any(c[0] == "reset" and "--hard" in c for c in calls)
    # 反向断言：绝不能用 --mixed（会保留 working tree，撞 v0.8.1 实测 bug）
    assert not any(c[0] == "reset" and "--mixed" in c for c in calls), \
        "bootstrap 必须用 --hard 强制对齐 working tree（v0.8.1 用 --mixed 撞过 dirty bug）"


def test_bootstrap_fallbacks_to_master_head_when_no_version_tag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """__version__ 对应的 tag 在远端不存在（开发版 / 非 release commit zip）→
    anchor fallback 到 FETCH_HEAD。"""
    calls: list[tuple] = []
    def _fake_git(*args, **_):
        calls.append(args)
        if args == ("--version",):
            return 0, "ok", ""
        if args[:2] == ("remote", "get-url"):
            return 2, "", ""
        if args[0] == "rev-parse" and args[1] == "--verify":
            # v{__version__} tag 不存在
            return 1, "", "fatal: ambiguous argument"
        if args[0] == "rev-parse":
            return 0, "fetched_head_sha", ""
        return 0, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)
    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)

    r = updater.bootstrap_git_repo()
    assert r.ok is True
    assert r.anchor_kind == "master_head"
    # reset 的 target 应当是 FETCH_HEAD（不是 vX.Y.Z）
    reset_calls = [c for c in calls if c[0] == "reset"]
    assert len(reset_calls) == 1
    assert "FETCH_HEAD" in reset_calls[0]


def test_bootstrap_fetch_failure_propagates_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fetch 失败（网络 / DNS）→ ok=False + 错误文案含 stderr 摘要。
    不应继续到 reset 步骤。"""
    calls: list[tuple] = []
    def _fake_git(*args, **_):
        calls.append(args)
        if args == ("--version",):
            return 0, "ok", ""
        if args[:2] == ("remote", "get-url"):
            return 2, "", ""
        if args[0] == "fetch":
            return 128, "", "fatal: unable to access 'https://github.com/'"
        return 0, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)
    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)

    r = updater.bootstrap_git_repo()
    assert r.ok is False
    assert r.error is not None
    assert "fetch" in r.error.lower() or "github" in r.error.lower()
    # reset 不应被调（fetch 失败就中止）
    assert not any(c[0] == "reset" for c in calls)


def test_bootstrap_happy_path_leaves_clean_working_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """端到端回归：真起 file:// 本地 origin + 模拟 zip 解压目录（含 npm
    自动改过的文件），bootstrap 完后 `git status --porcelain` 应当为空。

    v0.8.1 撞过的 bug：bootstrap 用 `--mixed`，npm install 改过 package-lock.json
    后 init 完就 dirty → pre-flight 卡更新。换 `--hard` 后 working tree 强制
    对齐到 anchor，npm 改动被覆盖，状态干净。

    没装 git 直接 skip（CI 上一般有 git；个别隔离环境可能没）。"""
    import shutil
    import subprocess

    if not shutil.which("git"):
        pytest.skip("git binary 不在 PATH，跳过端到端 bootstrap 测试")

    def _run(*args: str, cwd: Path) -> None:
        """在 cwd 下跑 git，失败抛 AssertionError 带 stderr。"""
        proc = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0, f"git {' '.join(args)} 失败:\n{proc.stderr}"

    # ---- 1. 起一个本地 bare repo 作为 origin ----
    origin_dir = tmp_path / "origin.git"
    origin_dir.mkdir()
    _run("init", "--bare", "--initial-branch=master", ".", cwd=origin_dir)

    # ---- 2. 起一个 working repo，commit + tag v9.9.9，push 到 origin ----
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    _run("init", "--initial-branch=master", ".", cwd=src_dir)
    _run("config", "user.email", "test@example.com", cwd=src_dir)
    _run("config", "user.name", "Test", cwd=src_dir)
    # commit 一些"上游"文件
    (src_dir / "README.md").write_text("upstream readme\n", encoding="utf-8")
    (src_dir / "package-lock.json").write_text('{"upstream": true}\n', encoding="utf-8")
    _run("add", "-A", cwd=src_dir)
    _run("commit", "-m", "release v9.9.9", cwd=src_dir)
    _run("tag", "v9.9.9", cwd=src_dir)
    _run("remote", "add", "origin", str(origin_dir), cwd=src_dir)
    _run("push", "origin", "master", "--tags", cwd=src_dir)

    # ---- 3. 模拟 zip 解压：把 src/ 内容（不含 .git/）拷到 zip_dir ----
    zip_dir = tmp_path / "zip"
    zip_dir.mkdir()
    for f in src_dir.iterdir():
        if f.name == ".git":
            continue
        shutil.copy(f, zip_dir / f.name)

    # ---- 4. 模拟 npm install 改过 package-lock.json（v0.8.1 bug 触发点）----
    (zip_dir / "package-lock.json").write_text(
        '{"upstream": true, "npm_local_modification": "different"}\n',
        encoding="utf-8",
    )

    # ---- 5. monkeypatch updater 指向这个模拟环境，跑 bootstrap ----
    monkeypatch.setattr(updater, "REPO_ROOT", zip_dir)
    monkeypatch.setattr(updater, "ORIGIN_URL", str(origin_dir))
    monkeypatch.setattr(updater, "__version__", "9.9.9")  # 匹配 tag

    result = updater.bootstrap_git_repo()
    assert result.ok is True, f"bootstrap 失败: {result.error}"
    assert result.anchor_kind == "version_tag", "应当 anchor 在 v9.9.9 tag"

    # ---- 6. 关键断言：working tree 完全干净（不含 untracked）----
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(zip_dir), capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "", \
        f"bootstrap 完应当 clean，实际 status:\n{proc.stdout}"

    # ---- 7. package-lock.json 被强制覆盖回上游版本（npm 改动丢了）----
    assert (zip_dir / "package-lock.json").read_text(encoding="utf-8") == '{"upstream": true}\n'


def test_bootstrap_honors_env_origin_url_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ANIMA_STUDIO_ORIGIN_URL env var 覆盖默认 origin URL —— fork 维护者可配。

    注意：updater 在 import 时读 env，这里 monkeypatch 模块属性等价。"""
    custom_url = "https://example.com/my-fork/AnimaLoraStudio.git"
    monkeypatch.setattr(updater, "ORIGIN_URL", custom_url)
    captured_url: list[str] = []
    def _fake_git(*args, **_):
        if args == ("--version",):
            return 0, "ok", ""
        if args[:2] == ("remote", "get-url"):
            return 2, "", ""
        if args[:3] == ("remote", "add", "origin"):
            captured_url.append(args[3])
            return 0, "", ""
        if args[0] == "rev-parse" and args[1] == "--verify":
            return 1, "", ""
        if args[0] == "rev-parse":
            return 0, "sha", ""
        return 0, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)
    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)

    r = updater.bootstrap_git_repo()
    assert r.ok is True
    assert captured_url == [custom_url], \
        f"应当用 ORIGIN_URL 覆盖值，实际调用 = {captured_url}"


# ---------------------------------------------------------------------------
# target_has_self_update (chunk 4 safety net)
# ---------------------------------------------------------------------------


def test_target_has_self_update_true_when_marker_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """任一候选路径 git cat-file -e <ref>:<path> 返 0 → True。"""
    monkeypatch.setattr(updater, "_git",
                       lambda *args, **_k: (0, "", "") if args[0] == "cat-file" else (1, "", ""))
    assert updater.target_has_self_update("any-ref") is True


def test_target_has_self_update_false_when_marker_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """所有候选路径都不存在（pre-self-update commit）→ False。"""
    monkeypatch.setattr(updater, "_git", lambda *args, **_k: (1, "", "does not exist"))
    assert updater.target_has_self_update("ancient-commit") is False


def test_target_has_self_update_false_on_git_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """git 失败（ref 无效 / 仓库损坏）→ 保守返 False，让 preflight 阻断。"""
    monkeypatch.setattr(updater, "_git", lambda *args, **_k: (128, "", "fatal: invalid object"))
    assert updater.target_has_self_update("garbage") is False


def test_self_update_markers_track_real_file() -> None:
    """marker 路径必须跟着 updater.py 的真实位置走。

    回归守卫：0.11.0 (PR #143) services/ 重构把 updater.py 从
    studio/services/ 搬到 studio/services/runtime/，但 _SELF_UPDATE_MARKER
    没跟着改，导致所有重构后的 commit 都没有旧路径文件 → target_has_self_update
    恒返 False → preflight 把每个新版本误判为"早于自更新 feature"并阻断切换。
    断言至少一个候选路径在当前工作树真实存在，搬文件忘改 marker 时直接挂。
    """
    from studio.paths import REPO_ROOT
    assert any((REPO_ROOT / p).exists() for p in updater._SELF_UPDATE_MARKERS), (
        f"_SELF_UPDATE_MARKERS 没有一个指向真实文件：{updater._SELF_UPDATE_MARKERS}"
        "——updater.py 搬家后忘了更新 marker 路径？"
    )


def test_legacy_updater_shim_present_for_0_10_2_clients() -> None:
    """studio/services/updater.py 兼容 shim 必须保留。

    0.10.2 及更早的 client 在 preflight 里只检查这一条单路径 marker
    （`_SELF_UPDATE_MARKER = "studio/services/updater.py"`）。v0.11.0 (PR #143)
    services/ 重构把文件搬到 `runtime/` 之后，旧路径如果**真的**被删，
    v0.10.2 client 升到 origin/master 会被 preflight 误判"目标版本早于
    webui 自更新 feature"并阻断按钮。shim 文件本身存在性就够（git cat-file
    只检查 blob 存在），但必须可正常 import 防止外部脚本误用。

    删除条件：grep 历史所有发布版的 `_SELF_UPDATE_MARKER*` 值，确认每个有人
    可能装的旧版本都已识别新路径 `studio/services/runtime/updater.py`。
    """
    from studio.paths import REPO_ROOT
    shim = REPO_ROOT / "studio" / "services" / "updater.py"
    assert shim.exists(), (
        f"backward-compat shim {shim} 缺失！"
        "0.10.2- client 的 target_has_self_update 检查这条单路径，"
        "删了会让旧用户没法通过 webui 升级。"
    )
    # 能 import 且暴露关键符号（让外部脚本 from studio.services.updater import 不挂）
    from studio.services import updater as legacy
    assert hasattr(legacy, "check_update")
    assert hasattr(legacy, "current_version")
    assert hasattr(legacy, "target_has_self_update")


# ---------------------------------------------------------------------------
# exact_tag_for (rollback UI 显示 tag 而非 sha 用)
# ---------------------------------------------------------------------------


def test_exact_tag_for_returns_tag_when_commit_tagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """git describe --tags --exact-match 命中 → 返回 tag 字符串。"""
    monkeypatch.setattr(updater, "_git",
                       lambda *args, **_k: (0, "v0.6.0", "") if args[0] == "describe" else (1, "", ""))
    assert updater.exact_tag_for("deadbeef" * 5) == "v0.6.0"


def test_exact_tag_for_none_when_no_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """commit 上没打精确 tag → describe 返非 0 → None（caller fallback 到 sha[:8]）。"""
    monkeypatch.setattr(updater, "_git",
                       lambda *args, **_k: (128, "", "fatal: no exact match"))
    assert updater.exact_tag_for("deadbeef" * 5) is None


def test_exact_tag_for_empty_sha_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """空 sha 早退；不调 git。"""
    called = []
    monkeypatch.setattr(updater, "_git", lambda *a, **_k: (called.append(a), (0, "v", ""))[1])
    assert updater.exact_tag_for("") is None
    assert called == []


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


# ---------------------------------------------------------------------------
# ADR 0005：installed_kind / installed_label 分类
# ---------------------------------------------------------------------------


def test_classify_install_stable_via_exact_tag() -> None:
    """HEAD 命中 vX.Y.Z release tag → stable + label = tag。"""
    kind, label, ver = updater._classify_install(
        commit="a" * 40, exact_tag="v0.8.0", branch="master",
        short="aaaaaaaa", commit_time_iso="2026-05-16T00:00:00Z",
        is_dirty=False,
    )
    assert kind == "stable"
    assert label == "v0.8.0"
    assert ver == "v0.8.0"


def test_classify_install_stable_with_dirty_suffix() -> None:
    kind, label, ver = updater._classify_install(
        commit="a" * 40, exact_tag="v0.8.0", branch="master",
        short="aaaaaaaa", commit_time_iso="", is_dirty=True,
    )
    assert kind == "stable"
    assert "未提交修改" in label
    assert ver == "v0.8.0"


def test_classify_install_stable_via_version_tree_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HEAD 没 tag，但 __version__ 匹 vX.Y.Z 且 tree 与 tag commit 一致 → stable。

    覆盖 release 直后场景：本地 commit 是 release commit（在 dev 分支上），
    tag 打在 master merge commit 上 —— exact_tag 不命中但 tree 一致。
    """
    monkeypatch.setattr(updater, "__version__", "0.8.0")
    def _fake_git(*args, **_kw):
        # rev-parse v0.8.0^{commit} → 返回 tag commit
        if args[:2] == ("rev-parse", "v0.8.0^{commit}"):
            return 0, "merge_commit_sha", ""
        # diff --quiet release_commit merge_commit → 0 表示 tree 一致
        if args[:2] == ("diff", "--quiet"):
            return 0, "", ""
        return 1, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)

    kind, label, ver = updater._classify_install(
        commit="release_commit_sha", exact_tag=None, branch="master",
        short="release_", commit_time_iso="", is_dirty=False,
    )
    assert kind == "stable"
    assert ver == "v0.8.0"


def test_classify_install_dev_when_commit_equals_origin_dev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """commit == origin/dev HEAD → dev + label 含 sha + 时间。"""
    monkeypatch.setattr(updater, "__version__", "0.0.0-noversion")  # 强制不匹 stable
    def _fake_git(*args, **_kw):
        if args[:2] == ("rev-parse", "v0.0.0-noversion^{commit}"):
            return 1, "", ""
        if args[:2] == ("rev-parse", "origin/dev"):
            return 0, "dev_head_sha", ""
        return 1, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)

    kind, label, ver = updater._classify_install(
        commit="dev_head_sha", exact_tag=None, branch="anything",
        short="dev_head", commit_time_iso="2026-05-16T19:50:00-05:00",
        is_dirty=False,
    )
    assert kind == "dev"
    assert "dev @ dev_head" in label
    assert "2026-05-16 19:50" in label
    assert ver is None


def test_classify_install_custom_on_feature_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """既不命中 release tag 也不在 dev HEAD → custom + label 含 branch + sha。"""
    monkeypatch.setattr(updater, "__version__", "0.0.0-noversion")
    monkeypatch.setattr(updater, "_git",
                        lambda *a, **_k: (1, "", "no plan"))

    kind, label, ver = updater._classify_install(
        commit="feature_commit", exact_tag=None, branch="feat/something",
        short="feature_", commit_time_iso="", is_dirty=False,
    )
    assert kind == "custom"
    assert "feat/something" in label
    assert "feature_" in label
    assert ver is None


def test_classify_install_custom_detached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(updater, "__version__", "0.0.0-noversion")
    monkeypatch.setattr(updater, "_git",
                        lambda *a, **_k: (1, "", "no plan"))

    kind, label, ver = updater._classify_install(
        commit="random_commit", exact_tag=None, branch="detached",
        short="random__", commit_time_iso="", is_dirty=False,
    )
    assert kind == "custom"
    assert "random__" in label
    assert "detached" not in label  # detached 时不暴露字面 "detached"，只显示 "（@ sha）"
    assert ver is None


# ---------------------------------------------------------------------------
# ADR 0005：check_update state 状态机
# ---------------------------------------------------------------------------


def test_check_update_master_up_to_date_same_version(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """master 通道：installed_version == latest_version → state=up_to_date。

    即使 commit 不同（release 后本地是 release commit, 远端是 merge commit），
    版本号相同就视为已是最新 —— 不再用 commit 词汇糊弄用户。
    """
    monkeypatch.setattr(
        updater, "current_version",
        lambda: updater.VersionInfo(
            version="0.8.0", commit="release_commit", commit_short="release_",
            commit_time_iso="", branch="master", tag=None, is_dirty=False,
            installed_kind="stable", installed_label="v0.8.0",
            stable_version="v0.8.0",
        ),
    )
    def _fake_git(*args, **_kw):
        if args[:2] == ("fetch", "origin"):
            return 0, "", ""
        if args[:2] == ("rev-parse", "origin/master"):
            return 0, "merge_commit", ""
        if args[:3] == ("rev-list", "--count", "HEAD..origin/master"):
            return 0, "2", ""
        if args[:3] == ("rev-list", "--count", "origin/master..HEAD"):
            return 0, "0", ""
        if args[0] == "describe":
            return 0, "v0.8.0", ""
        return 1, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)

    r = updater.check_update("master", use_cache=False)
    assert r.state == "up_to_date"
    assert r.installed_version == "v0.8.0"
    assert r.latest_version == "v0.8.0"
    assert r.has_update is False


def test_check_update_master_update_available_diff_version(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """master 通道：installed=v0.7.0, latest=v0.8.0 → state=update_available。"""
    monkeypatch.setattr(
        updater, "current_version",
        lambda: updater.VersionInfo(
            version="0.7.0", commit="v070_commit", commit_short="v070_com",
            commit_time_iso="", branch="master", tag="v0.7.0", is_dirty=False,
            installed_kind="stable", installed_label="v0.7.0",
            stable_version="v0.7.0",
        ),
    )
    def _fake_git(*args, **_kw):
        if args[:2] == ("fetch", "origin"):
            return 0, "", ""
        if args[:2] == ("rev-parse", "origin/master"):
            return 0, "v080_commit", ""
        if args[:3] == ("rev-list", "--count", "HEAD..origin/master"):
            return 0, "5", ""
        if args[:3] == ("rev-list", "--count", "origin/master..HEAD"):
            return 0, "0", ""
        if args[0] == "describe":
            return 0, "v0.8.0", ""
        return 1, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)

    r = updater.check_update("master", use_cache=False)
    assert r.state == "update_available"
    assert r.installed_version == "v0.7.0"
    assert r.latest_version == "v0.8.0"
    assert r.has_update is True
    assert r.behind_count == 5


def test_check_update_dev_up_to_date_when_commit_matches(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dev 通道：当前 commit == origin/dev HEAD → state=up_to_date。"""
    monkeypatch.setattr(
        updater, "current_version",
        lambda: updater.VersionInfo(
            version="0.8.0", commit="dev_tip", commit_short="dev_tip_",
            commit_time_iso="", branch="dev", tag=None, is_dirty=False,
            installed_kind="dev", installed_label="dev @ dev_tip_",
            stable_version=None,
        ),
    )
    def _fake_git(*args, **_kw):
        if args[:2] == ("fetch", "origin"):
            return 0, "", ""
        if args[:2] == ("rev-parse", "origin/dev"):
            return 0, "dev_tip", ""
        if args[:3] == ("rev-list", "--count", "HEAD..origin/dev"):
            return 0, "0", ""
        if args[:3] == ("rev-list", "--count", "origin/dev..HEAD"):
            return 0, "0", ""
        return 1, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)

    r = updater.check_update("dev", use_cache=False)
    assert r.state == "up_to_date"
    assert r.behind_count == 0


def test_check_update_dev_update_available_when_behind(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dev 通道：本地落后 origin/dev N commit → state=update_available + behind_count=N。"""
    monkeypatch.setattr(
        updater, "current_version",
        lambda: updater.VersionInfo(
            version="0.8.0", commit="old_dev_commit", commit_short="old_dev_",
            commit_time_iso="", branch="dev", tag=None, is_dirty=False,
            installed_kind="custom", installed_label="自定义（dev @ old_dev_）",
            stable_version=None,
        ),
    )
    def _fake_git(*args, **_kw):
        if args[:2] == ("fetch", "origin"):
            return 0, "", ""
        if args[:2] == ("rev-parse", "origin/dev"):
            return 0, "new_dev_tip", ""
        if args[:3] == ("rev-list", "--count", "HEAD..origin/dev"):
            return 0, "3", ""
        if args[:3] == ("rev-list", "--count", "origin/dev..HEAD"):
            return 0, "0", ""
        return 1, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)

    r = updater.check_update("dev", use_cache=False)
    assert r.state == "update_available"
    assert r.behind_count == 3
    assert r.has_update is True


def test_check_update_dev_ahead_when_local_leads(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dev 通道：本地 commit 领先 origin/dev（罕见，回滚或抢跑） → state=ahead。"""
    monkeypatch.setattr(
        updater, "current_version",
        lambda: updater.VersionInfo(
            version="0.8.0", commit="ahead_commit", commit_short="ahead_co",
            commit_time_iso="", branch="dev", tag=None, is_dirty=False,
            installed_kind="custom", installed_label="自定义",
            stable_version=None,
        ),
    )
    def _fake_git(*args, **_kw):
        if args[:2] == ("fetch", "origin"):
            return 0, "", ""
        if args[:2] == ("rev-parse", "origin/dev"):
            return 0, "old_remote_tip", ""
        if args[:3] == ("rev-list", "--count", "HEAD..origin/dev"):
            return 0, "0", ""
        if args[:3] == ("rev-list", "--count", "origin/dev..HEAD"):
            return 0, "2", ""
        return 1, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)

    r = updater.check_update("dev", use_cache=False)
    assert r.state == "ahead"


def test_check_update_fetch_error_returns_detached(
    _isolate_flags: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fetch 失败 → state=detached + error 字段填，前端不该认为"有更新"。"""
    def _fake_git(*args, **_kw):
        if args[:2] == ("fetch", "origin"):
            return 128, "", "fatal: network unreachable"
        return 0, "", ""
    monkeypatch.setattr(updater, "_git", _fake_git)

    r = updater.check_update("master", use_cache=False)
    assert r.state == "detached"
    assert r.error is not None


# ---------------------------------------------------------------------------
# 模型数据文件保护（_PRESERVE_ON_RESET）：reset --hard 跨过"删模型数据"的 commit
# 时，老用户磁盘上的已下载文件必须保住，不被删→重下
# ---------------------------------------------------------------------------

def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    def g(*args: str) -> None:
        subprocess.run(["git", *args], cwd=str(repo),
                       capture_output=True, text=True, check=True)
    g("init", "-q")
    g("config", "user.email", "t@example.com")
    g("config", "user.name", "tester")
    g("config", "commit.gpgsign", "false")


def _git_in(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(repo),
                          capture_output=True, text=True, check=True).stdout.strip()


def test_preserve_model_data_survives_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真 git 仓库验证护栏：reset --hard 跨过"删模型数据文件"的 commit 时，
    hardcode 清单里的文件被备份 → reset 删掉 → 还原，磁盘副本保住、内容不变、
    且变为 untracked（更新零感知，不触发重下）。"""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    monkeypatch.setattr(updater, "REPO_ROOT", repo)

    rel = "models/t5_tokenizer/spiece.model"  # _PRESERVE_ON_RESET 里的一项
    f = repo / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"SENTINEL-WEIGHTS-9f3a")
    _git_in(repo, "add", "-A")
    _git_in(repo, "commit", "-qm", "A: model data tracked (old version)")

    # 新版本：git rm 掉它 + 整体 ignore models/
    _git_in(repo, "rm", "-q", rel)
    (repo / ".gitignore").write_text("models/\n", encoding="utf-8")
    _git_in(repo, "add", "-A")
    _git_in(repo, "commit", "-qm", "B: untrack + ignore models/")
    target = _git_in(repo, "rev-parse", "HEAD")

    # 老用户：回到 A（文件被追踪 + 在磁盘）
    _git_in(repo, "checkout", "-q", "HEAD~1")
    assert f.exists()

    log: list[str] = []
    saved = updater._backup_preserved_files(log)
    assert rel in saved, "应备份这个 hardcode 的模型数据文件"

    rc, _, err = updater._git("reset", "--hard", target)
    assert rc == 0, err
    assert not f.exists(), "前置确认：reset --hard 确实会删掉这个已追踪文件"

    updater._restore_preserved_files(saved, emit=lambda _: None, log_lines=log)
    assert f.exists(), "护栏应把文件还原回磁盘（更新不重下）"
    assert f.read_bytes() == b"SENTINEL-WEIGHTS-9f3a", "内容必须原样"
    assert _git_in(repo, "ls-files", "--", rel) == "", "还原后应为 untracked"
    assert not updater.PRESERVE_HOLDING.exists(), "备份目录应收尾清掉"


def test_preserve_only_touches_hardcoded_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """护栏只认 hardcode 清单：不在清单里的文件（含代码 / 未来新增）即便被删也
    照常删，不会被误留 —— 守"不要对其他文件或未来代码文件生效"。"""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    monkeypatch.setattr(updater, "REPO_ROOT", repo)

    rel = "models/some_future_module.py"  # 不在 _PRESERVE_ON_RESET 里
    f = repo / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("# code\n", encoding="utf-8")
    _git_in(repo, "add", "-A")
    _git_in(repo, "commit", "-qm", "A")
    _git_in(repo, "rm", "-q", rel)
    _git_in(repo, "commit", "-qm", "B: delete it")
    target = _git_in(repo, "rev-parse", "HEAD")
    _git_in(repo, "checkout", "-q", "HEAD~1")
    assert f.exists()

    log: list[str] = []
    saved = updater._backup_preserved_files(log)
    assert saved == [], "非清单文件不应被备份"

    rc, _, err = updater._git("reset", "--hard", target)
    assert rc == 0, err
    updater._restore_preserved_files(saved, emit=lambda _: None, log_lines=log)
    assert not f.exists(), "非清单文件应被 reset 正常删除，不被护栏挽留"


def test_preserve_skips_file_user_moved_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """用户把模型迁走/删掉(文件已不在磁盘)时,护栏绝不"复活"它:备份只认
    "既被追踪又在磁盘上"的文件,缺一不可 → 不备份 → reset 后不还原。
    守"用户迁移 models 后不会被我们还原回来"。"""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    monkeypatch.setattr(updater, "REPO_ROOT", repo)

    rel = "models/t5_tokenizer/spiece.model"  # _PRESERVE_ON_RESET 里的一项
    f = repo / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"OLD")
    _git_in(repo, "add", "-A")
    _git_in(repo, "commit", "-qm", "A: tracked")
    _git_in(repo, "rm", "-q", rel)
    (repo / ".gitignore").write_text("models/\n", encoding="utf-8")
    _git_in(repo, "add", "-A")
    _git_in(repo, "commit", "-qm", "B: untrack")
    target = _git_in(repo, "rev-parse", "HEAD")

    _git_in(repo, "checkout", "-q", "HEAD~1")  # 回到 A：文件被追踪
    f.unlink()                                  # 用户把它迁走/删了：磁盘上没了
    assert not f.exists()

    log: list[str] = []
    saved = updater._backup_preserved_files(log)
    assert saved == [], "磁盘上不存在的文件不该被备份"

    rc, _, err = updater._git("reset", "--hard", target)
    assert rc == 0, err
    updater._restore_preserved_files(saved, emit=lambda _: None, log_lines=log)
    assert not f.exists(), "用户迁走/删掉的文件不该被护栏还原回来"
