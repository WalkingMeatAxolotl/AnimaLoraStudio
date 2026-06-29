"""release post 校验 + 版本号同步 + CHANGELOG.md 派生（ADR 0013）。

changelog 来源 = `docs/announcements/` 下 `tag: release` 的 markdown post，
一版一文件（双文件双语，工具只用中文 `<id>.md`）。编写指南见
`docs/announcements/README.md`。本工具不创建 post —— 那是维护者发版时写 md 的事。

Subcommands:
    validate         校验全部 release post 的 frontmatter（version/date/title/文件名自洽/唯一）
    bump             同步「最高版本」release post 的 version 到 __init__.py / package.json /
                     package-lock.json + 重写 CHANGELOG.md
    render-changelog 仅从 release post 重写 CHANGELOG.md，不动版本号文件
    verify-versions  跨文件 drift 检查：__init__.py / package.json / package-lock.json 必须一致

Examples:
    python tools/bump_version.py validate
    python tools/bump_version.py bump                 # 取最高版本的 release post
    python tools/bump_version.py bump --version 0.16.0
    python tools/bump_version.py render-changelog
    python tools/bump_version.py verify-versions
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# Windows Python stdout 默认 cp936，打中文直接 UnicodeEncodeError。工具自带正确编码。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

REPO_ROOT = Path(__file__).resolve().parent.parent
ANNOUNCEMENTS_DIR = REPO_ROOT / "docs" / "announcements"
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"
STUDIO_INIT_PATH = REPO_ROOT / "studio" / "__init__.py"
PACKAGE_JSON_PATH = REPO_ROOT / "studio" / "web" / "package.json"
PACKAGE_LOCK_PATH = REPO_ROOT / "studio" / "web" / "package-lock.json"

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][\w.\-]+)?$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ─── release post 加载 ──────────────────────────────────────────────────────
@dataclass
class ReleasePost:
    version: str
    date: str
    title: str
    body: str
    filename: str


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """`---\\n<yaml>\\n---\\n<body>` → (meta, body)。无 frontmatter → ({}, 全文)。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}, text
    try:
        meta = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, "\n".join(lines[end + 1:]).strip("\n")


def _semver_tuple(v: str) -> tuple[int, ...]:
    core = v.split("-", 1)[0].split("+", 1)[0]
    try:
        return tuple(int(x) for x in core.split("."))
    except ValueError:
        return (0,)


def load_release_posts() -> list[ReleasePost]:
    """读 docs/announcements/ 下 `tag: release` 的中文 post，按 version 降序。"""
    posts: list[ReleasePost] = []
    if not ANNOUNCEMENTS_DIR.is_dir():
        return posts
    for p in sorted(ANNOUNCEMENTS_DIR.glob("*.md")):
        if p.name.endswith(".en.md") or p.name.lower() == "readme.md":
            continue
        meta, body = _split_frontmatter(p.read_text(encoding="utf-8"))
        if meta.get("tag") != "release":
            continue
        posts.append(ReleasePost(
            version=str(meta.get("version", "")).strip(),
            date=str(meta.get("date", "")).strip(),
            title=str(meta.get("title", "")).strip(),
            body=body,
            filename=p.name,
        ))
    posts.sort(key=lambda r: _semver_tuple(r.version), reverse=True)
    return posts


# ─── 校验 ───────────────────────────────────────────────────────────────────
@dataclass
class ValidateIssue:
    level: str   # "error" | "warn"
    location: str
    message: str


@dataclass
class ValidateResult:
    issues: list[ValidateIssue] = field(default_factory=list)

    def add_error(self, location: str, message: str) -> None:
        self.issues.append(ValidateIssue("error", location, message))

    def add_warn(self, location: str, message: str) -> None:
        self.issues.append(ValidateIssue("warn", location, message))

    @property
    def has_errors(self) -> bool:
        return any(i.level == "error" for i in self.issues)


def validate(posts: list[ReleasePost]) -> ValidateResult:
    """校验 release post frontmatter。详见 docs/announcements/README.md。"""
    r = ValidateResult()
    if not posts:
        r.add_error("/", "docs/announcements/ 下没有 tag: release 的 post")
        return r

    seen: set[str] = set()
    for post in posts:
        loc = post.filename
        if not SEMVER_RE.match(post.version):
            r.add_error(f"{loc}.version", f"无效 semver：{post.version!r}")
        elif post.version in seen:
            r.add_error(f"{loc}.version", f"重复版本 {post.version}")
        else:
            seen.add(post.version)

        if not DATE_RE.match(post.date):
            r.add_error(f"{loc}.date", f"date 必须是 ISO YYYY-MM-DD：{post.date!r}")
        if not post.title:
            r.add_error(f"{loc}.title", "title 必填")
        if not post.body.strip():
            r.add_warn(f"{loc}", "正文为空")

        # 文件名约定：<date>-v<version>.md
        if DATE_RE.match(post.date) and SEMVER_RE.match(post.version):
            expect = f"{post.date}-v{post.version}.md"
            if post.filename != expect:
                r.add_warn(f"{loc}", f"文件名建议 {expect}（与 frontmatter 自洽）")
    return r


def print_validate_result(r: ValidateResult) -> None:
    errors = [i for i in r.issues if i.level == "error"]
    warns = [i for i in r.issues if i.level == "warn"]
    for i in r.issues:
        print(f"  {'✗' if i.level == 'error' else '!'} {i.location}: {i.message}")
    if not errors and not warns:
        print("validate ok — 没有问题")
    elif not errors:
        print(f"validate ok — {len(warns)} 个 warning（不阻塞）")
    else:
        print(f"validate FAILED — {len(errors)} 个 error / {len(warns)} 个 warning")


# ─── CHANGELOG 派生 ─────────────────────────────────────────────────────────
def render_changelog(posts: list[ReleasePost]) -> str:
    """从 release post（version 降序）拼出 CHANGELOG.md（ADR 0013）。"""
    lines: list[str] = [
        "# Changelog",
        "",
        "> **本文件由 [`tools/bump_version.py render-changelog`](tools/bump_version.py)"
        " 从 [`docs/announcements/`](docs/announcements/) 的 `tag: release` post 自动派生**",
        "> —— 请改那些 markdown，不要改本文件。编写指南见"
        " [`docs/announcements/README.md`](docs/announcements/README.md)。",
        "",
        "---",
        "",
    ]
    for post in posts:
        lines.append(f"## v{post.version} — {post.date}")
        lines.append("")
        if post.body.strip():
            lines.append(post.body.rstrip())
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


# ─── 版本号文件读写 ─────────────────────────────────────────────────────────
def _read_studio_version() -> Optional[str]:
    if not STUDIO_INIT_PATH.exists():
        return None
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', STUDIO_INIT_PATH.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def _write_studio_version(new_version: str) -> bool:
    if not STUDIO_INIT_PATH.exists():
        return False
    txt = STUDIO_INIT_PATH.read_text(encoding="utf-8")
    new_txt, n = re.subn(
        r'(__version__\s*=\s*["\'])([^"\']+)(["\'])', rf'\g<1>{new_version}\g<3>', txt, count=1)
    if n == 0:
        return False
    write_atomic(STUDIO_INIT_PATH, new_txt)
    return True


def _read_package_json_version() -> Optional[str]:
    if not PACKAGE_JSON_PATH.exists():
        return None
    return json.loads(PACKAGE_JSON_PATH.read_text(encoding="utf-8")).get("version")


def _write_package_json_version(new_version: str) -> bool:
    if not PACKAGE_JSON_PATH.exists():
        return False
    txt = PACKAGE_JSON_PATH.read_text(encoding="utf-8")
    new_txt, n = re.subn(
        r'(\"version\"\s*:\s*\")([^\"]+)(\")', rf'\g<1>{new_version}\g<3>', txt, count=1)
    if n == 0:
        return False
    write_atomic(PACKAGE_JSON_PATH, new_txt)
    return True


def _read_package_lock_version() -> Optional[str]:
    if not PACKAGE_LOCK_PATH.exists():
        return None
    return json.loads(PACKAGE_LOCK_PATH.read_text(encoding="utf-8")).get("version")


def _write_package_lock_version(new_version: str) -> bool:
    """更新 lockfile 顶层 + packages[""] 两处 version（count=2，绝不动 deps 的 version）。"""
    if not PACKAGE_LOCK_PATH.exists():
        return False
    txt = PACKAGE_LOCK_PATH.read_text(encoding="utf-8")
    new_txt, n = re.subn(
        r'(\"version\"\s*:\s*\")([^\"]+)(\")', rf'\g<1>{new_version}\g<3>', txt, count=2)
    if n < 2:
        return False
    write_atomic(PACKAGE_LOCK_PATH, new_txt)
    return True


def _read_package_lock_packages_version() -> Optional[str]:
    if not PACKAGE_LOCK_PATH.exists():
        return None
    pkgs = (json.loads(PACKAGE_LOCK_PATH.read_text(encoding="utf-8")).get("packages") or {})
    root = pkgs.get("") or {}
    return root.get("version") if isinstance(root, dict) else None


# ─── subcommands ────────────────────────────────────────────────────────────
def cmd_validate(_args: argparse.Namespace) -> int:
    r = validate(load_release_posts())
    print_validate_result(r)
    return 1 if r.has_errors else 0


def cmd_render_changelog(_args: argparse.Namespace) -> int:
    posts = load_release_posts()
    r = validate(posts)
    if r.has_errors:
        print("validate FAILED — render-changelog 拒绝执行：")
        print_validate_result(r)
        return 1
    write_atomic(CHANGELOG_PATH, render_changelog(posts))
    print(f"[render] {CHANGELOG_PATH.relative_to(REPO_ROOT)} 重写完成（{len(posts)} 个版本）")
    return 0


def cmd_bump(args: argparse.Namespace) -> int:
    posts = load_release_posts()
    r = validate(posts)
    if r.has_errors:
        print("validate FAILED — bump 拒绝执行：")
        print_validate_result(r)
        return 1
    print_validate_result(r)
    if not posts:
        print("没有 release post，bump 无事可做", file=sys.stderr)
        return 2

    top_version = posts[0].version  # 已按 semver 降序
    if args.version and args.version != top_version:
        print(
            f"--version={args.version} 与最高 release post 版本={top_version} 不符。\n"
            "release post 是 source of truth；要 bump 到该版本，请先写"
            f" docs/announcements/<date>-v{args.version}.md。",
            file=sys.stderr,
        )
        return 2

    target = args.version or top_version
    print(f"\n[bump] target version: {target}")
    print(f"[bump] studio/__init__.py: {_read_studio_version()} → {target}")
    print(f"[bump] studio/web/package.json: {_read_package_json_version()} → {target}")
    print(f"[bump] studio/web/package-lock.json: {_read_package_lock_version()} → {target}")

    if not _write_studio_version(target):
        print("[bump] WARN: studio/__init__.py 没找到 __version__，跳过", file=sys.stderr)
    if not _write_package_json_version(target):
        print("[bump] WARN: package.json 没找到 version，跳过", file=sys.stderr)
    if not _write_package_lock_version(target):
        print("[bump] WARN: package-lock.json 没找到两处 version，跳过", file=sys.stderr)

    write_atomic(CHANGELOG_PATH, render_changelog(posts))
    print(f"[bump] {CHANGELOG_PATH.relative_to(REPO_ROOT)} 重写完成")

    drift_rc = cmd_verify_versions(argparse.Namespace())
    if drift_rc != 0:
        print("[bump] FAIL: 自检发现 drift，请人工核对", file=sys.stderr)
        return drift_rc

    print(f"\nnext: git add -A && git commit -m 'chore(release): {target}' && git tag v{target} && git push --tags")
    return 0


def cmd_verify_versions(_args: argparse.Namespace) -> int:
    """跨文件 version drift 校验。四处必须一致；任一不等返 1。"""
    rows = [
        ("studio/__init__.py (__version__)", _read_studio_version()),
        ("studio/web/package.json (version)", _read_package_json_version()),
        ("studio/web/package-lock.json (top version)", _read_package_lock_version()),
        ('studio/web/package-lock.json (packages[""].version)', _read_package_lock_packages_version()),
    ]
    distinct = {v for _, v in rows}
    if len(distinct) == 1 and None not in distinct:
        print(f"[verify-versions] OK · 四处一致 = {distinct.pop()}")
        return 0
    print("[verify-versions] FAIL · 跨文件 version drift：", file=sys.stderr)
    for label, value in rows:
        print(f"  {label:<55} = {value!r}", file=sys.stderr)
    print("\n修法：跑 `python tools/bump_version.py bump` 重新同步，或手动校准。", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bump_version", description=__doc__.splitlines()[0] if __doc__ else "")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("validate", help="校验 release post frontmatter").set_defaults(func=cmd_validate)
    sub.add_parser("render-changelog", help="从 release post 重写 CHANGELOG.md").set_defaults(func=cmd_render_changelog)
    p_b = sub.add_parser("bump", help="同步版本号 + 重写 CHANGELOG.md")
    p_b.add_argument("--version", help="期望目标版本（与最高 release post 不符则报错）", default=None)
    p_b.set_defaults(func=cmd_bump)
    sub.add_parser("verify-versions", help="跨文件 version drift 检查（CI gate）").set_defaults(func=cmd_verify_versions)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
