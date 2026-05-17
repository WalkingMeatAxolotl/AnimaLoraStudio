"""Webui 内自更新机制（ADR 0002）— git pull + 重启 + apply pending deps。

详见 [`docs/adr/0002-webui-self-update.md`](../../docs/adr/0002-webui-self-update.md)。

模块职责：
- 查询当前 git 状态（HEAD / branch / tag / dirty）
- 检查远端是否有新版本（git fetch + rev-list 比对，TTL 24h 缓存）
- 写 `studio_data/.update_pending` + `tmp/restart` 让 cli.py 启动期接管
- cli.py 启动期 `apply_pending()` 执行 git pull + 增量 pip install / npm install

关键 flag / 文件协议：

| 路径 | 含义 | 作者 → 读者 |
| --- | --- | --- |
| `tmp/restart` | 需要重启 | server → cli.py / wrapper |
| `studio_data/.update_pending` | 启动期要 git pull，内容是 target ref | server → cli.py |
| `studio_data/.update_cache` | 自动检查结果缓存（TTL 24h） | check_update() 自管 |
| `studio_data/.last_version` | 上一版 commit（rollback 用，PR-C 启用） | apply_pending |
| `studio_data/.update_log` | 最近一次 update 的日志（PR-C 展示） | apply_pending |
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .. import __version__
from ..paths import REPO_ROOT, STUDIO_DATA

logger = logging.getLogger(__name__)

# ----- Flag / 缓存文件路径 ------------------------------------------------
RESTART_FLAG = REPO_ROOT / "tmp" / "restart"
UPDATE_PENDING = STUDIO_DATA / ".update_pending"
UPDATE_CACHE = STUDIO_DATA / ".update_cache"
LAST_VERSION = STUDIO_DATA / ".last_version"
UPDATE_LOG = STUDIO_DATA / ".update_log"
UPDATE_STATUS = STUDIO_DATA / ".update_status"   # PR-C：结构化最近一次 update 结果

UPDATE_CACHE_TTL_SECONDS = 24 * 3600
GIT_FETCH_TIMEOUT = 30.0
GIT_PULL_TIMEOUT = 120.0

# zip 解压用户没有 .git/，自更新功能完全失效。bootstrap_git_repo() 一次性
# 在本地 init + remote add origin + fetch master，之后走正常 self-update 路径。
# fork 维护者通过 env var ANIMA_STUDIO_ORIGIN_URL 覆盖默认上游 URL。
DEFAULT_ORIGIN_URL = "https://github.com/WalkingMeatAxolotl/AnimaLoraStudio.git"
ORIGIN_URL = os.environ.get("ANIMA_STUDIO_ORIGIN_URL", "").strip() or DEFAULT_ORIGIN_URL


# ----- 数据类型 -----------------------------------------------------------
@dataclass
class VersionInfo:
    """当前仓库 git 状态 + 产品视角的"装了什么"分类。

    产品 UI 应使用 `installed_kind` / `installed_label`，不要依赖 `branch`。
    切换通道走 `git reset --hard`，不改 branch 名 —— branch 字段只做 debug。
    """
    version: str               # studio.__version__ (0.6.0)
    commit: str                # 完整 sha
    commit_short: str          # 前 8 位
    commit_time_iso: str       # ISO8601
    branch: str                # master / dev / detached / feature-name（debug 用）
    tag: Optional[str]         # HEAD 上的 tag（仅 exact match），无则 None
    is_dirty: bool             # working tree 有未提交改动
    # ---- 产品 UI 用的「装了什么」分类（前端唯一应该看的字段）----
    # 真实生产路径里这三个永远由 _classify_install 派生；给默认值只是让单测
    # 构造 fake VersionInfo 时不用每次都写满（ADR 0005 漏改的 hotfix 同步）。
    installed_kind: str = "custom"  # "stable" / "dev" / "custom" / "zip"
    installed_label: str = ""       # 用户可读："v0.8.0" / "dev @ f6f202b · 2026-05-16" / "自定义（feat/foo @ a1b2c3d）" / "v0.8.0（zip 安装）"
    stable_version: Optional[str] = None  # "vX.Y.Z" 形式，仅 installed_kind=stable 时填；用于版本号比对
    # ---- zip 模式探测（0.8.1 hotfix）----
    is_git_repo: bool = True   # False = REPO_ROOT/.git 缺失 / 没有 origin remote（zip 解压用户）
    git_available: bool = True # False = git binary 不在 PATH


@dataclass
class UpdateCheckResult:
    """git fetch + 比对结果。

    前端应使用 `state` + `installed_version` / `latest_version` / `behind_count`，
    不要依赖 `commits_ahead`（git 词汇）/ `has_update`（兼容字段）。
    """
    channel: str               # master / dev
    current_commit: str
    latest_commit: str
    commits_ahead: int         # 内部 debug：local 落后 remote 多少 commit
    has_update: bool           # 兼容字段 = (state == "update_available")
    latest_tag: Optional[str]  # remote 最新 tag（仅 master 通道有）
    checked_at: float          # epoch
    # ---- 产品 UI 用的状态机 ----
    state: str = "up_to_date"  # "up_to_date" / "update_available" / "ahead" / "detached"
    installed_version: Optional[str] = None  # 当前装的稳定版 tag (vX.Y.Z)，仅 stable 时填
    latest_version: Optional[str] = None     # 远端最新稳定版 tag（master 通道）
    behind_count: int = 0      # 前端文案"N 项更新"用（= commits_ahead，但语义更清楚）
    error: Optional[str] = None  # fetch 失败时填


@dataclass
class DevCommit:
    """dev 通道一条 commit 摘要（chunk 3 — VersionSection dev 卡时间线用）。"""
    sha: str            # 完整 sha，作为 performSystemUpdate(target=...) 的 ref
    short_sha: str      # 前 8 位用于展示
    msg: str            # commit subject line
    time_iso: str       # author commit time, ISO8601
    author: str         # author name


@dataclass
class DevCommitsResult:
    """`git log origin/dev` 结果。fetched=False 时 commits 用本地缓存（如有）。"""
    commits: list[DevCommit] = field(default_factory=list)
    fetched: bool = False
    error: Optional[str] = None


@dataclass
class UpdateStatus:
    """最近一次 update 的结构化结果（PR-C）。apply_pending 完成时写到磁盘，
    UI 用来判断"上次更新成功 / 失败 / 中止"，失败时展示原因。"""
    status: str                # ok / aborted / failed / partial
    reason: str                # 失败 / 中止时的简短原因；成功时空串
    target: str                # 用户请求的 ref (origin/master / commit hash)
    from_commit: str           # 走 git reset 之前的 commit
    to_commit: str             # 走 git reset 之后的 commit (失败时 = from_commit)
    started_at: float
    finished_at: float
    deps_changed: bool         # 走了 pip install 或 npm install
    log_excerpt: str           # 末尾几行 .update_log 内容


# ----- Git 调用 helper ----------------------------------------------------
def _git(*args: str, timeout: float = 15.0) -> tuple[int, str, str]:
    """跑 git 命令，返回 (rc, stdout, stderr)。"""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 1, "", "git not found on PATH"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)


# 稳定版 tag 形如 v0.8.0。HEAD 命中这种 tag 就归类 stable。
# 第四段（如 v0.8.0-rc1）暂时不在版本面板的"稳定版"语义里，先归 custom。
_RELEASE_TAG_RE = re.compile(r"^v\d+\.\d+\.\d+$")


# ----- zip 用户检测 + 一键 init（0.8.1 hotfix）---------------------------
@dataclass
class GitRepoStatus:
    """REPO_ROOT 的 git 可用性快照。版本面板用这个决定显示哪类 banner。"""
    git_available: bool   # git binary 在 PATH（不依赖 .git 存在）
    has_dot_git: bool     # REPO_ROOT/.git 目录存在
    has_origin: bool      # origin remote 已配置（蕴含 has_dot_git）
    @property
    def is_repo(self) -> bool:
        """版本面板视角的"可以走自更新"= 三者全 True。"""
        return self.git_available and self.has_dot_git and self.has_origin


def git_repo_status() -> GitRepoStatus:
    """三态检测：① git binary？② .git/？③ origin remote？

    顺序固定：git binary 不在 PATH → 后两步无意义直接 False（避免误判）。
    """
    rc, _, _ = _git("--version")
    git_available = rc == 0
    if not git_available:
        return GitRepoStatus(False, False, False)
    has_dot_git = (REPO_ROOT / ".git").exists()
    if not has_dot_git:
        return GitRepoStatus(True, False, False)
    rc, _, _ = _git("remote", "get-url", "origin")
    return GitRepoStatus(True, True, rc == 0)


@dataclass
class BootstrapResult:
    """bootstrap_git_repo() 结果。anchor = HEAD 最终指向的 ref（vX.Y.Z tag 或 FETCH_HEAD sha）。"""
    ok: bool
    anchor: Optional[str] = None
    anchor_kind: str = ""      # "version_tag" / "master_head"；ok=False 时空
    error: Optional[str] = None


def bootstrap_git_repo() -> BootstrapResult:
    """zip 解压用户首次启用自更新功能：在 REPO_ROOT 上 init git 仓库。

    流程：
    1. precondition：git binary 在 PATH（否则返 error，提示用户先装 git）
    2. .git/ 不存在 → `git init` + `git symbolic-ref HEAD refs/heads/master`
       （强制 master 默认 branch，避免不同 git 版本 main/master 默认差异）
    3. origin remote 不存在 → `git remote add origin {ORIGIN_URL}`
       （ORIGIN_URL 走 env var 覆盖，fork 维护者可配）
    4. `git fetch origin master --tags`（拉完整 master 历史 + 全部 tag；
       不带 --depth 保证 dev 通道时间线 / 回滚到任意 commit 可用）
    5. 找 `v{__version__}` tag：存在则用作 anchor（让 HEAD 指向用户当前装
       的版本对应的 tag commit，working tree 完全对齐）；不存在则 fallback
       到 FETCH_HEAD（master HEAD）
    6. `git reset --hard {anchor}` —— 同时更新 HEAD / index / working tree，
       强制对齐到 anchor 的 tree。注意**会覆盖用户在 zip 目录里的所有修改**
       （npm install 自动改的 package-lock.json、用户手动 tweak 的配置等）。

    为什么 `--hard` 不是 `--mixed`：
    - 早期版本用 `--mixed` 想保留用户文件，但 Windows 上 `studio.bat run`
      启动期会跑 npm install 改 package-lock.json，导致 init 完就 dirty →
      pre-flight 卡更新 → 用户无法触发自更新（v0.8.1 实测撞到）
    - zip 用户场景下，"启用自动更新"的潜台词就是"对齐到上游稳定版"，强制
      覆盖比保留本地随机改动更符合期望
    - banner 文案对此显式提示

    bootstrap 后 _classify_install 回 "stable"（HEAD == v{__version__} tag
    commit），版本面板正常显示「v0.8.0」+「检查更新」可用 + working tree clean。

    不在范围：fetch dev / 拉 dev_commits。用户切到 dev 通道时由 check_update
    / dev_commits 自己触发首次 fetch dev（多等几秒，可接受）。
    """
    status = git_repo_status()
    if not status.git_available:
        return BootstrapResult(
            ok=False,
            error="git binary 不在 PATH。请先安装 git（https://git-scm.com/downloads）后重启 Studio。",
        )

    # 1. git init（仅当 .git/ 不存在）
    if not status.has_dot_git:
        rc, _, err = _git("init", str(REPO_ROOT))
        if rc != 0:
            return BootstrapResult(ok=False, error=f"git init 失败: {err[:200]}")
        # 强制 default branch = master（git 2.28+ 起 init.defaultBranch 默认可能是 main）
        rc, _, err = _git("symbolic-ref", "HEAD", "refs/heads/master")
        if rc != 0:
            return BootstrapResult(ok=False, error=f"git symbolic-ref 失败: {err[:200]}")

    # 2. origin remote
    rc, _, _ = _git("remote", "get-url", "origin")
    if rc != 0:
        rc, _, err = _git("remote", "add", "origin", ORIGIN_URL)
        if rc != 0:
            return BootstrapResult(ok=False, error=f"git remote add 失败: {err[:200]}")

    # 3. fetch master + tags（这一步是大头，30-60 MB）
    rc, _, err = _git("fetch", "origin", "master", "--tags", timeout=GIT_FETCH_TIMEOUT * 4)
    if rc != 0:
        return BootstrapResult(ok=False, error=f"git fetch 失败: {err[:200]}")

    # 4. 选 anchor：优先匹 __version__ 的 release tag
    anchor_ref: str
    anchor_kind: str
    version_tag = f"v{__version__}"
    if _RELEASE_TAG_RE.match(version_tag):
        rc, _, _ = _git("rev-parse", "--verify", f"refs/tags/{version_tag}^{{commit}}")
        if rc == 0:
            anchor_ref = version_tag
            anchor_kind = "version_tag"
        else:
            anchor_ref = "FETCH_HEAD"
            anchor_kind = "master_head"
    else:
        anchor_ref = "FETCH_HEAD"
        anchor_kind = "master_head"

    # 5. reset --hard：HEAD + index + working tree 全部对齐到 anchor。
    # 会覆盖 zip 目录里 npm install 改过的 lockfile / 用户手动的本地 tweak；
    # 这是 zip 用户"启用自动更新"的预期行为（banner 文案显式提示）。
    rc, _, err = _git("reset", "--hard", anchor_ref, timeout=GIT_PULL_TIMEOUT)
    if rc != 0:
        return BootstrapResult(ok=False, error=f"git reset 失败: {err[:200]}")

    # 解析 anchor sha 给前端 / 日志用
    rc, anchor_sha, _ = _git("rev-parse", anchor_ref)
    return BootstrapResult(
        ok=True,
        anchor=anchor_sha if rc == 0 else anchor_ref,
        anchor_kind=anchor_kind,
    )


# ----- 公开 API -----------------------------------------------------------
def current_version() -> VersionInfo:
    """读当前仓库状态。git 不可用 / zip 解压模式时返回占位值（不抛）。"""
    repo = git_repo_status()
    if not repo.is_repo:
        # zip 模式：没有 .git/ 或 origin remote。版本面板看 is_git_repo
        # 决定显示 init banner，不再依赖 commit/branch 字段。
        return VersionInfo(
            version=__version__,
            commit="unknown",
            commit_short="?",
            commit_time_iso="",
            branch="detached",
            tag=None,
            is_dirty=False,
            installed_kind="zip",
            installed_label=f"v{__version__}（zip 安装）",
            stable_version=None,
            is_git_repo=False,
            git_available=repo.git_available,
        )

    rc, head, _ = _git("rev-parse", "HEAD")
    commit = head if rc == 0 else "unknown"
    short = commit[:8] if commit != "unknown" else "?"

    rc, branch, _ = _git("rev-parse", "--abbrev-ref", "HEAD")
    if rc != 0 or branch == "HEAD":
        branch = "detached"

    rc, ctime, _ = _git("log", "-1", "--format=%cI", "HEAD")
    ctime_iso = ctime if rc == 0 else ""

    rc, tag, _ = _git("describe", "--tags", "--exact-match", "HEAD")
    exact_tag = tag if rc == 0 else None

    # --untracked-files=no：untracked 文件（pr18_review.md / 临时笔记 / 没进 gitignore
    # 的草稿等）不影响 git reset --hard，它们会原地保留。dirty 仅指"修改了 tracked 文件"。
    rc, status, _ = _git("status", "--porcelain", "--untracked-files=no")
    is_dirty = rc == 0 and bool(status)

    installed_kind, installed_label, stable_version = _classify_install(
        commit=commit,
        exact_tag=exact_tag,
        branch=branch,
        short=short,
        commit_time_iso=ctime_iso,
        is_dirty=is_dirty,
    )

    return VersionInfo(
        version=__version__,
        commit=commit,
        commit_short=short,
        commit_time_iso=ctime_iso,
        branch=branch,
        tag=exact_tag,
        is_dirty=is_dirty,
        installed_kind=installed_kind,
        installed_label=installed_label,
        stable_version=stable_version,
        is_git_repo=True,
        git_available=True,
    )


def _classify_install(
    *,
    commit: str,
    exact_tag: Optional[str],
    branch: str,
    short: str,
    commit_time_iso: str,
    is_dirty: bool,
) -> tuple[str, str, Optional[str]]:
    """推断 (installed_kind, installed_label, stable_version)。

    优先级：
    1. HEAD 命中 vX.Y.Z release tag → stable（最常见的稳定版情形）
    2. `__version__` 匹 vX.Y.Z tag 且当前 commit 与 tag commit 的 tree 一致 → stable
       覆盖 "release commit 在 dev 上，tag 打在 master 的 merge commit 上" 这种
       release 直后场景（两个 commit 内容相同，用户语义上装的就是稳定版）
    3. commit == origin/dev HEAD → dev
    4. else → custom（feature branch / detached / 任意 commit）

    返回三元组：
    - installed_kind / installed_label：给前端做 UI 文案的
    - stable_version：仅 stable 时为 "vX.Y.Z"，否则 None；后端做版本号比对用

    label 是给用户看的字符串；dirty 时追加"· 未提交修改"。
    """
    dirty_suffix = " · 未提交修改" if is_dirty else ""

    # 1. HEAD 命中 release tag
    if exact_tag and _RELEASE_TAG_RE.match(exact_tag):
        return "stable", f"{exact_tag}{dirty_suffix}", exact_tag

    if commit and commit != "unknown":
        # 2. __version__ 字符串匹 release tag 且 tree 一致
        version_tag = f"v{__version__}"
        if _RELEASE_TAG_RE.match(version_tag):
            rc, tag_commit, _ = _git("rev-parse", f"{version_tag}^{{commit}}")
            if rc == 0 and tag_commit:
                if tag_commit == commit:
                    # 极少触发（exact_tag 应已捕获），保险起见
                    return "stable", f"{version_tag}{dirty_suffix}", version_tag
                # tree 一致 = 文件内容完全相同（merge / cherry-pick 常见）
                rc_diff, _, _ = _git("diff", "--quiet", commit, tag_commit)
                if rc_diff == 0:
                    return "stable", f"{version_tag}{dirty_suffix}", version_tag

        # 3. commit == origin/dev HEAD
        rc, dev_head, _ = _git("rev-parse", "origin/dev")
        if rc == 0 and dev_head and commit == dev_head:
            date_part = ""
            if commit_time_iso:
                date_part = f" · {commit_time_iso[:16].replace('T', ' ')}"
            return "dev", f"dev @ {short}{date_part}{dirty_suffix}", None

    # 4. custom
    if not branch or branch == "detached":
        return "custom", f"自定义（@ {short}）{dirty_suffix}", None
    return "custom", f"自定义（{branch} @ {short}）{dirty_suffix}", None


def check_update(channel: str = "master", use_cache: bool = True) -> UpdateCheckResult:
    """`git fetch origin {channel}` + 比对本地 HEAD 与 `origin/{channel}`。

    channel 仅接受 'master' / 'dev'。Master 走 24h 缓存（cache 写到磁盘）；
    dev 不写缓存（开发者主动检查，避免污染 master 的"有更新"信号）。

    输出走 state 状态机（up_to_date / update_available / ahead / detached）。
    state 推断：
    - master 通道：优先比较 installed_version vs latest_version（版本号语义），
      没有版本号（installed_kind != stable）时回落到 commit 比较
    - dev 通道：直接比较 commit hash；commits_ahead>0 → update_available；
      =0 但 sha 不一致 → ahead（你超前）或 detached
    """
    if channel not in ("master", "dev"):
        raise ValueError(f"invalid channel: {channel}")

    if channel == "master" and use_cache:
        cached = _read_cache()
        if cached is not None:
            return cached

    cur = current_version()
    checked_at = time.time()

    rc, _, stderr = _git("fetch", "origin", channel, timeout=GIT_FETCH_TIMEOUT)
    if rc != 0:
        return UpdateCheckResult(
            channel=channel, current_commit=cur.commit, latest_commit="",
            commits_ahead=0, has_update=False, latest_tag=None,
            checked_at=checked_at, state="detached", behind_count=0,
            installed_version=None, latest_version=None,
            error=f"git fetch failed: {stderr[:200]}",
        )

    rc, latest, _ = _git("rev-parse", f"origin/{channel}")
    if rc != 0:
        return UpdateCheckResult(
            channel=channel, current_commit=cur.commit, latest_commit="",
            commits_ahead=0, has_update=False, latest_tag=None,
            checked_at=checked_at, state="detached", behind_count=0,
            installed_version=None, latest_version=None,
            error=f"git rev-parse origin/{channel} failed",
        )

    # commit 计数 —— behind = origin 比本地多多少；ahead = 本地比 origin 多多少
    rc, behind_str, _ = _git("rev-list", "--count", f"HEAD..origin/{channel}")
    behind = int(behind_str) if rc == 0 and behind_str.isdigit() else 0
    rc, ahead_str, _ = _git("rev-list", "--count", f"origin/{channel}..HEAD")
    ahead = int(ahead_str) if rc == 0 and ahead_str.isdigit() else 0

    latest_tag: Optional[str] = None
    latest_version: Optional[str] = None
    rc, tag_at_remote, _ = _git("describe", "--tags", "--exact-match", latest)
    if rc == 0:
        latest_tag = tag_at_remote
        if _RELEASE_TAG_RE.match(tag_at_remote):
            latest_version = tag_at_remote
    if not latest_tag:
        # describe 精确匹配失败 → fallback：取最近 reachable tag（保留兼容）
        rc, tag_near, _ = _git("describe", "--tags", "--abbrev=0", latest)
        if rc == 0:
            latest_tag = tag_near
            if channel == "master" and _RELEASE_TAG_RE.match(tag_near):
                latest_version = tag_near

    installed_version = cur.stable_version

    # ---- 状态机推断 ----
    if channel == "master":
        # 版本号优先：版本号相同（已是最新稳定版）→ up_to_date
        if installed_version and latest_version and installed_version == latest_version:
            state = "up_to_date"
        elif installed_version and latest_version and installed_version != latest_version:
            # 装了稳定版且远端有更新稳定版 → 提示更新
            state = "update_available"
        elif cur.commit == latest:
            # 没装 stable（custom / dev）但 commit 与 origin/master 完全一致
            state = "up_to_date"
        elif behind > 0:
            state = "update_available"
        elif ahead > 0:
            state = "ahead"
        else:
            state = "detached"
    else:  # dev
        if cur.commit == latest:
            state = "up_to_date"
        elif behind > 0:
            state = "update_available"
        elif ahead > 0:
            state = "ahead"
        else:
            state = "detached"

    result = UpdateCheckResult(
        channel=channel,
        current_commit=cur.commit,
        latest_commit=latest,
        commits_ahead=behind,
        has_update=(state == "update_available"),
        latest_tag=latest_tag,
        checked_at=checked_at,
        state=state,
        installed_version=installed_version,
        latest_version=latest_version,
        behind_count=behind,
    )

    if channel == "master":
        _write_cache(result)

    return result


def resolve_ref(ref: str) -> Optional[str]:
    """`git rev-parse <ref>` → 完整 sha；ref 不存在 → None。"""
    rc, out, _ = _git("rev-parse", ref)
    return out if rc == 0 else None


def exact_tag_for(sha: str) -> Optional[str]:
    """`git describe --tags --exact-match <sha>` → tag 字符串；commit 上没打
    tag → None。给 UI 用：rollback 按钮文案优先显示 tag（v0.6.0）而非
    裸 sha；没 tag 时 caller fallback 到 sha[:8]。
    """
    if not sha:
        return None
    rc, out, _ = _git("describe", "--tags", "--exact-match", sha)
    return out if rc == 0 and out else None


# Self-update feature 引入的 marker 文件。target 上不存在 → 切过去就丢失
# webui 升级能力（只能 CLI git pull 救援）。preflight() err 级别阻断。
_SELF_UPDATE_MARKER = "studio/services/updater.py"


def target_has_self_update(target_ref: str) -> bool:
    """目标 ref 上是否带 webui 自更新 feature。

    用 `git cat-file -e <ref>:<path>` 测文件存在性（不读内容，效率比 git
    show 高且无 stdout 输出污染）。失败 / ref 无效 → False（保守）。
    """
    rc, _, _ = _git("cat-file", "-e", f"{target_ref}:{_SELF_UPDATE_MARKER}")
    return rc == 0


_REQ_NAME_RE = re.compile(r"^([A-Za-z0-9_\-\.\[\]]+)")


def _parse_requirements(text: str) -> dict[str, str]:
    """`requirements.txt` 内容 → {pkg_name_lowercased: full_spec_line}。

    跳过注释 / 空行 / `-r ...` / `-e ...` 引用。识别包名 = 行首字母数字下划线
    点连字符 + 可选 extras `[...]`；不解析 marker / hash —— 这里只用来粗略
    diff 提示用户"有变化"，准确版本控制走 pip 自己的解析。
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = _REQ_NAME_RE.match(line)
        if not m:
            continue
        out[m.group(1).lower()] = line
    return out


@dataclass
class RequirementsDiff:
    added: list[str] = field(default_factory=list)    # 新增的包名（target 有，current 没）
    removed: list[str] = field(default_factory=list)  # 移除的包名（current 有，target 没）
    changed: list[dict[str, str]] = field(default_factory=list)
    # changed item: {"name": "...", "from": "pkg==1.0", "to": "pkg==2.0"}


def requirements_diff(target_ref: str) -> RequirementsDiff:
    """Diff `requirements.txt` 在 HEAD 与 target_ref 之间。

    git show 失败（target ref 不解析 / 文件在 target 上不存在）→ 空 diff，
    UI 当作"无变化"处理。current requirements.txt 缺失同样空 diff。
    """
    target_resolved = resolve_ref(target_ref)
    if target_resolved is None:
        return RequirementsDiff()
    rc, target_text, _ = _git("show", f"{target_resolved}:requirements.txt")
    if rc != 0:
        return RequirementsDiff()

    cur_path = REPO_ROOT / "requirements.txt"
    if not cur_path.exists():
        return RequirementsDiff()
    try:
        cur_text = cur_path.read_text(encoding="utf-8-sig")
    except OSError:
        return RequirementsDiff()

    cur = _parse_requirements(cur_text)
    tgt = _parse_requirements(target_text)
    added = sorted(set(tgt.keys()) - set(cur.keys()))
    removed = sorted(set(cur.keys()) - set(tgt.keys()))
    changed: list[dict[str, str]] = []
    for name in sorted(set(tgt.keys()) & set(cur.keys())):
        if cur[name] != tgt[name]:
            changed.append({"name": name, "from": cur[name], "to": tgt[name]})
    return RequirementsDiff(added=added, removed=removed, changed=changed)


def dev_commits(limit: int = 10) -> DevCommitsResult:
    """`git fetch origin dev` + `git log origin/dev -<limit>`，返回最近 commits。

    Chunk 3 — VersionSection dev 卡时间线 + 任意 commit 切换用。

    - fetch 失败仍尝试读本地 origin/dev 缓存（用户离线或网络问题时，至少
      能看到上次 fetch 的状态而不是白屏）
    - 解析失败 / 仓库没 origin/dev → commits=[] + error 文案
    - limit clamp 到 1-50 之间
    """
    limit = max(1, min(50, int(limit)))

    rc_fetch, _, fetch_err = _git("fetch", "origin", "dev", timeout=GIT_FETCH_TIMEOUT)
    fetched = rc_fetch == 0
    fetch_error_msg: Optional[str] = None if fetched else f"git fetch dev: {fetch_err[:200]}"

    # NUL-separated 字段格式：%H sha · %h short · %s subject · %cI iso time · %an author
    fmt = "%H%x00%h%x00%s%x00%cI%x00%an"
    rc, out, log_err = _git("log", f"-{limit}", f"--format={fmt}", "origin/dev")
    if rc != 0:
        # 没 origin/dev ref（首次 clone 未跟 dev / 远端被删等）
        return DevCommitsResult(
            commits=[],
            fetched=fetched,
            error=fetch_error_msg or f"git log origin/dev: {log_err[:200]}",
        )

    commits: list[DevCommit] = []
    for line in out.splitlines():
        parts = line.split("\x00")
        if len(parts) < 5:
            continue
        commits.append(DevCommit(
            sha=parts[0],
            short_sha=parts[1],
            msg=parts[2],
            time_iso=parts[3],
            author=parts[4],
        ))

    return DevCommitsResult(commits=commits, fetched=fetched, error=fetch_error_msg)


def request_update(target: str = "origin/master") -> None:
    """server 端调：写 .update_pending + tmp/restart 让 cli.py 启动期接管。"""
    UPDATE_PENDING.parent.mkdir(parents=True, exist_ok=True)
    UPDATE_PENDING.write_text(target, encoding="utf-8")
    RESTART_FLAG.parent.mkdir(parents=True, exist_ok=True)
    RESTART_FLAG.touch()


def has_pending() -> bool:
    return UPDATE_PENDING.exists()


def apply_pending(emit: Callable[[str], None] = print) -> bool:
    """cli.py 启动期调。返回 True = 走过 pull 路径；False = 无 pending 跳过。

    流程：
    1. 读 .update_pending 拿 target ref
    2. 写 .last_version（rollback 用）
    3. precondition：working tree 必须干净（理论上 server 已查过，这里再保一层）
    4. `git fetch origin` + `git reset --hard {target}`（避免 merge 冲突）
    5. requirements.txt sha256 marker 比对 → 改了就 `pip install -r`
    6. studio/web/package.json mtime > node_modules/.package-lock.json → `npm install`
    7. 清 cache（让下次 check_update 重 fetch）+ 清 .update_pending
    8. 写结构化 .update_status（PR-C，UI 展示"上次更新结果"用）

    失败的每一步都写 .update_log 和 .update_status，但不抛异常 — 让 cli.py
    继续走后面的 bootstrap，server 至少能起来（UI 端会看到失败 banner）。

    状态枚举：
    - ok：git 切换成功，无 deps 失败
    - aborted：precondition 失败（dirty tree）
    - failed：git fetch / reset 失败
    - partial：git 切换成功但 pip / npm 失败（功能可能不完整）
    """
    if not has_pending():
        return False

    target = UPDATE_PENDING.read_text(encoding="utf-8-sig").strip() or "origin/master"
    emit(f"[updater] applying pending update → {target}")

    started_at = time.time()
    cur = current_version()
    log_lines: list[str] = [
        f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} update {cur.commit_short} → {target} ===",
        f"branch={cur.branch} tag={cur.tag or '-'} dirty={cur.is_dirty}",
    ]

    # 保存上一版本（rollback 用）
    try:
        LAST_VERSION.parent.mkdir(parents=True, exist_ok=True)
        LAST_VERSION.write_text(cur.commit, encoding="utf-8")
    except OSError as e:
        log_lines.append(f"[warn] failed to write .last_version: {e}")

    def _done(status: str, reason: str, to_commit: str, deps_changed: bool) -> bool:
        """收尾：写 .update_status + .update_log + 清 .update_pending + 清 cache。"""
        finished_at = time.time()
        _write_status(UpdateStatus(
            status=status,
            reason=reason,
            target=target,
            from_commit=cur.commit,
            to_commit=to_commit,
            started_at=started_at,
            finished_at=finished_at,
            deps_changed=deps_changed,
            log_excerpt="\n".join(log_lines[-20:]),
        ))
        _finalize(log_lines)
        try:
            if UPDATE_CACHE.exists():
                UPDATE_CACHE.unlink()
        except OSError:
            pass
        return True

    # 1. precondition：working tree 干净
    if cur.is_dirty:
        log_lines.append("[abort] working tree dirty")
        emit("[updater] working tree dirty, aborting update")
        return _done("aborted", "working tree dirty", cur.commit, False)

    # 2. git fetch
    log_lines.append("[git] fetch origin")
    rc, _, stderr = _git("fetch", "origin", timeout=GIT_FETCH_TIMEOUT)
    if rc != 0:
        log_lines.append(f"[git fetch] FAILED rc={rc} stderr={stderr}")
        emit(f"[updater] git fetch failed: {stderr[:200]}")
        return _done("failed", f"git fetch: {stderr[:120]}", cur.commit, False)

    # 3. git reset --hard target（避免 merge conflict；working tree 干净已验过）
    log_lines.append(f"[git] reset --hard {target}")
    rc, _, stderr = _git("reset", "--hard", target, timeout=GIT_PULL_TIMEOUT)
    if rc != 0:
        log_lines.append(f"[git reset] FAILED rc={rc} stderr={stderr}")
        emit(f"[updater] git reset failed: {stderr[:200]}")
        return _done("failed", f"git reset: {stderr[:120]}", cur.commit, False)

    new = current_version()
    log_lines.append(f"[ok] now at {new.commit_short} ({new.tag or new.branch})")
    emit(f"[updater] git updated → {new.commit_short}")

    deps_changed = False
    deps_failed_reason = ""

    # 4. requirements.txt 改了 → 增量 pip install（不 --upgrade，仅补缺）
    if _requirements_marker_stale():
        deps_changed = True
        log_lines.append("[pip] requirements.txt changed; pip install -r")
        emit("[updater] requirements.txt changed, pip install (may take a few minutes)...")
        rc = subprocess.call(
            [sys.executable, "-m", "pip", "install", "-r", str(REPO_ROOT / "requirements.txt")]
        )
        log_lines.append(f"[pip] exit code {rc}")
        if rc == 0:
            marker = REPO_ROOT / "venv" / ".studio-requirements.sha256"
            tool = REPO_ROOT / "tools" / "check_requirements_changed.py"
            if tool.exists():
                subprocess.call([
                    sys.executable, str(tool),
                    "--marker", str(marker), "--update-marker",
                ])
        else:
            deps_failed_reason = f"pip exit {rc}"

    # 5. package.json 改了 → npm install
    if _package_json_changed():
        deps_changed = True
        log_lines.append("[npm] package.json changed; npm install")
        emit("[updater] studio/web/package.json changed, npm install...")
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if npm:
            rc = subprocess.call([npm, "install"], cwd=str(REPO_ROOT / "studio" / "web"))
            log_lines.append(f"[npm] exit code {rc}")
            if rc != 0:
                deps_failed_reason = (
                    f"{deps_failed_reason + '; ' if deps_failed_reason else ''}npm exit {rc}"
                )
        else:
            log_lines.append("[npm] not found on PATH, skipping (cli.py bootstrap will retry)")

    if deps_failed_reason:
        log_lines.append(f"[partial] git ok 但 deps 失败: {deps_failed_reason}")
        return _done("partial", deps_failed_reason, new.commit, deps_changed)

    log_lines.append("[done]")
    return _done("ok", "", new.commit, deps_changed)


def last_status() -> Optional[UpdateStatus]:
    """读 .update_status；不存在 / 损坏 → None。"""
    if not UPDATE_STATUS.exists():
        return None
    try:
        data = json.loads(UPDATE_STATUS.read_text(encoding="utf-8-sig"))
        return UpdateStatus(**data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def read_update_log() -> str:
    """完整 .update_log 内容；不存在返回空串。"""
    if not UPDATE_LOG.exists():
        return ""
    try:
        return UPDATE_LOG.read_text(encoding="utf-8-sig")
    except OSError:
        return ""


def rollback_target() -> Optional[str]:
    """读 .last_version。返回 commit sha 或 None（首次未更新过 / 文件缺失）。

    校验 commit 在仓库里存在 — 防止仓库被强制 GC 掉 .last_version 指向的孤儿
    commit。验不过返 None，UI 隐藏回滚按钮。
    """
    if not LAST_VERSION.exists():
        return None
    try:
        sha = LAST_VERSION.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return None
    if not sha:
        return None
    rc, _, _ = _git("cat-file", "-e", sha)
    return sha if rc == 0 else None


def request_rollback() -> Optional[str]:
    """读 .last_version 内容，调 request_update(target=<sha>)。

    没有 .last_version 或 commit 不存在 → 返回 None（调用方应当返 409 / 422）。
    成功调度 → 返回 target sha。

    回滚流程与正向 update 完全一样（同一个 apply_pending 处理），所以下次
    UI 上 .last_version 会自动被更新成"现在的版本"，支持来回切。
    """
    sha = rollback_target()
    if sha is None:
        return None
    request_update(sha)
    return sha


# ----- 内部 helpers ------------------------------------------------------
def _read_cache() -> Optional[UpdateCheckResult]:
    if not UPDATE_CACHE.exists():
        return None
    try:
        data = json.loads(UPDATE_CACHE.read_text(encoding="utf-8-sig"))
        age = time.time() - float(data.get("checked_at", 0))
        if age > UPDATE_CACHE_TTL_SECONDS or age < 0:
            return None
        return UpdateCheckResult(**data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _write_cache(result: UpdateCheckResult) -> None:
    """原子写：先写 .tmp 再 rename，避免并发读 corrupt。"""
    try:
        UPDATE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = UPDATE_CACHE.with_suffix(UPDATE_CACHE.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        tmp.replace(UPDATE_CACHE)
    except OSError as e:
        logger.warning("failed to write update cache: %s", e)


def _finalize(log_lines: list[str]) -> None:
    """写 update.log + 清 .update_pending 标志。"""
    try:
        UPDATE_LOG.parent.mkdir(parents=True, exist_ok=True)
        UPDATE_LOG.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    except OSError:
        pass
    try:
        if UPDATE_PENDING.exists():
            UPDATE_PENDING.unlink()
    except OSError:
        pass


def _write_status(status: UpdateStatus) -> None:
    """原子写 .update_status（PR-C）。"""
    try:
        UPDATE_STATUS.parent.mkdir(parents=True, exist_ok=True)
        tmp = UPDATE_STATUS.with_suffix(UPDATE_STATUS.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(status), indent=2), encoding="utf-8")
        tmp.replace(UPDATE_STATUS)
    except OSError as e:
        logger.warning("failed to write .update_status: %s", e)


def _requirements_marker_stale() -> bool:
    """requirements.txt sha256 vs venv/.studio-requirements.sha256 marker。

    复用 studio.sh / studio.bat 已用的 marker（兼容 cold-start bootstrap）。
    """
    req = REPO_ROOT / "requirements.txt"
    marker = REPO_ROOT / "venv" / ".studio-requirements.sha256"
    if not req.exists():
        return False
    digest = hashlib.sha256(req.read_bytes()).hexdigest()
    if not marker.exists():
        return True  # 没 marker：可能从未装过，安全起见按 stale
    try:
        return marker.read_text(encoding="utf-8-sig").strip() != digest
    except OSError:
        return True


def _package_json_changed() -> bool:
    """前端依赖声明比 node_modules 安装标记新时才视为需要 npm install。"""
    web_dir = REPO_ROOT / "studio" / "web"
    marker = web_dir / "node_modules" / ".package-lock.json"
    if not marker.exists():
        return False
    try:
        marker_mtime = marker.stat().st_mtime
        for f in (web_dir / "package.json", web_dir / "package-lock.json"):
            if f.exists() and f.stat().st_mtime > marker_mtime:
                return True
    except OSError:
        return False
    return False
