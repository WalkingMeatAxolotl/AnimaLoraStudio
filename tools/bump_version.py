"""release_notes.yaml 校验 + 版本号同步 + CHANGELOG.md 派生。

详见 docs/release-notes-spec.md。本工具不创建 entries —— 那是 agent 改 yaml 的事。

Subcommands:
    validate         schema 校验整个 yaml
    bump             同步 yaml top version 到 __init__.py / package.json + 重写 CHANGELOG.md
    render-changelog 仅重写 CHANGELOG.md，不动版本号文件

Examples:
    python tools/bump_version.py validate
    python tools/bump_version.py bump           # 读 yaml top version
    python tools/bump_version.py bump --version 0.6.1
    python tools/bump_version.py render-changelog
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

REPO_ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = REPO_ROOT / "release_notes.yaml"
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"
STUDIO_INIT_PATH = REPO_ROOT / "studio" / "__init__.py"
PACKAGE_JSON_PATH = REPO_ROOT / "studio" / "web" / "package.json"

# ─── 校验规则（与 docs/release-notes-spec.md §8 一致） ──────────────────────

KIND_WHITELIST = ("added", "changed", "improved", "fixed", "removed", "deprecated", "security")
KIND_DISPLAY = {
    "added": "新增",
    "changed": "变更",
    "improved": "改进",
    "fixed": "修复",
    "removed": "删除",
    "deprecated": "弃用",
    "security": "安全",
}
KIND_ORDER = ("security", "added", "changed", "improved", "fixed", "deprecated", "removed")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][\w.\-]+)?$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SUMMARY_MAX = 80
MARKDOWN_FORBIDDEN_IN_SUMMARY = ("**",)  # 只检 `**bold**`；`__init__.py` 之类 `_` 太常见


@dataclass
class ValidateIssue:
    """单条校验问题。level: error 让 bump 退出非零；warn 仅打印。"""
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


def _semver_tuple(v: str) -> tuple[int, ...]:
    """`0.6.1` → (0, 6, 1)；带 suffix 也吃，比较时只看 numeric 主版本。"""
    core = v.split("-", 1)[0].split("+", 1)[0]
    return tuple(int(x) for x in core.split("."))


def load_yaml() -> list[dict[str, Any]]:
    """读 yaml 返回 versions list。空 / 不存在 → []。yaml 解析错抛 ValueError。"""
    if not YAML_PATH.exists():
        return []
    try:
        data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"release_notes.yaml YAML 解析失败：{e}") from e
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError("release_notes.yaml 顶层必须是 list[version block]")
    return data


def validate(versions: list[dict[str, Any]]) -> ValidateResult:
    """schema 校验整个 yaml。详见 docs/release-notes-spec.md §8。"""
    r = ValidateResult()
    if not versions:
        r.add_error("/", "release_notes.yaml 至少要有一个 version block")
        return r

    seen_versions: set[str] = set()
    prev_tuple: Optional[tuple[int, ...]] = None
    pr_count: dict[int, int] = {}

    for idx, block in enumerate(versions):
        loc = f"[{idx}]"
        if not isinstance(block, dict):
            r.add_error(loc, "version block 必须是 dict")
            continue

        # version
        version = block.get("version")
        if not isinstance(version, str) or not SEMVER_RE.match(version):
            r.add_error(f"{loc}.version", f"无效 semver：{version!r}")
        else:
            if version in seen_versions:
                r.add_error(f"{loc}.version", f"重复版本 {version}")
            seen_versions.add(version)
            cur_tuple = _semver_tuple(version)
            if prev_tuple is not None and cur_tuple >= prev_tuple:
                r.add_error(
                    f"{loc}.version",
                    f"版本顺序错误（应当 latest 在 top）：{version} 不该排在前一个之后",
                )
            prev_tuple = cur_tuple

        # date
        date = block.get("date")
        if not isinstance(date, str) or not DATE_RE.match(date):
            r.add_error(f"{loc}.date", f"date 必须是 ISO YYYY-MM-DD：{date!r}")

        # summary (block-level, optional)
        block_summary = block.get("summary")
        if block_summary is not None and not isinstance(block_summary, str):
            r.add_error(f"{loc}.summary", "block summary 必须是 str（或省略）")

        # entries
        entries = block.get("entries")
        if not isinstance(entries, list) or not entries:
            r.add_error(f"{loc}.entries", "entries 必须是非空 list")
            continue
        if len(entries) >= 12:
            r.add_warn(f"{loc}.entries", f"{len(entries)} 条 entry 偏多，考虑合并同主题")

        for ei, entry in enumerate(entries):
            eloc = f"{loc}.entries[{ei}]"
            if not isinstance(entry, dict):
                r.add_error(eloc, "entry 必须是 dict")
                continue

            kind = entry.get("kind")
            if kind not in KIND_WHITELIST:
                r.add_error(f"{eloc}.kind", f"未知 kind：{kind!r}（允许：{', '.join(KIND_WHITELIST)}）")

            summary = entry.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                r.add_error(f"{eloc}.summary", "summary 必填且非空")
            else:
                if len(summary) > SUMMARY_MAX:
                    r.add_error(
                        f"{eloc}.summary",
                        f"summary 超长（{len(summary)} > {SUMMARY_MAX} 字符）：{summary[:40]}…",
                    )
                for tok in MARKDOWN_FORBIDDEN_IN_SUMMARY:
                    if tok in summary:
                        r.add_error(
                            f"{eloc}.summary",
                            f"summary 不允许 markdown 标记 {tok!r}（放进 detail）",
                        )
                if len(summary.strip()) < 8:
                    r.add_warn(f"{eloc}.summary", "summary 太短可能信息量不足")

            pr_refs = entry.get("pr_refs")
            if pr_refs is not None:
                if not isinstance(pr_refs, list):
                    r.add_error(f"{eloc}.pr_refs", "pr_refs 必须是 list[int] 或省略")
                else:
                    for pi, p in enumerate(pr_refs):
                        if not isinstance(p, int) or p <= 0 or p > 9999:
                            r.add_error(f"{eloc}.pr_refs[{pi}]", f"PR 号必须是正 int ≤ 9999：{p!r}")
                        elif isinstance(p, int):
                            pr_count[p] = pr_count.get(p, 0) + 1

            detail = entry.get("detail")
            if detail is not None and not isinstance(detail, str):
                r.add_error(f"{eloc}.detail", "detail 必须是 str（或省略）")

    # PR 重复出现警告
    for p, count in pr_count.items():
        if count > 3:
            r.add_warn("/", f"PR #{p} 出现在 {count} 条 entry 里（可能拆得太细）")

    return r


def print_validate_result(r: ValidateResult) -> None:
    """打印校验结果。"""
    errors = [i for i in r.issues if i.level == "error"]
    warns = [i for i in r.issues if i.level == "warn"]
    for i in r.issues:
        marker = "✗" if i.level == "error" else "!"
        print(f"  {marker} {i.location}: {i.message}")
    if not errors and not warns:
        print("validate ok — 没有问题")
    elif not errors:
        print(f"validate ok — {len(warns)} 个 warning（不阻塞）")
    else:
        print(f"validate FAILED — {len(errors)} 个 error / {len(warns)} 个 warning")


def render_changelog(versions: list[dict[str, Any]]) -> str:
    """从 yaml 派生 CHANGELOG.md markdown 内容。"""
    lines: list[str] = []
    lines.append("# Changelog")
    lines.append("")
    lines.append("> **本文件由 [`tools/bump_version.py render-changelog`](tools/bump_version.py)")
    lines.append("> 从 [`release_notes.yaml`](release_notes.yaml) 自动派生 —— 请改 yaml，不要改本文件。")
    lines.append("> 编写规范见 [`docs/release-notes-spec.md`](docs/release-notes-spec.md)。")
    lines.append("")
    lines.append("---")
    lines.append("")

    for block in versions:
        version = block.get("version", "?")
        date = block.get("date", "?")
        block_summary = block.get("summary")
        lines.append(f"## [{version}] — {date}")
        lines.append("")
        if block_summary:
            lines.append(block_summary)
            lines.append("")

        # entries 按 KIND_ORDER 分组（同 kind 内保留 yaml 出现顺序）
        by_kind: dict[str, list[dict[str, Any]]] = {}
        for entry in block.get("entries", []):
            by_kind.setdefault(entry.get("kind", "other"), []).append(entry)

        for kind in KIND_ORDER:
            if kind not in by_kind:
                continue
            lines.append(f"### {KIND_DISPLAY.get(kind, kind)}")
            lines.append("")
            for entry in by_kind[kind]:
                summary = entry.get("summary", "").strip()
                lines.append(f"- **{summary}**")
                detail = entry.get("detail")
                if detail:
                    # detail 整段以两空格缩进作为 bullet 的子内容
                    for line in detail.rstrip().splitlines():
                        if line.strip():
                            lines.append(f"  {line}")
                        else:
                            lines.append("")
                lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_atomic(path: Path, content: str) -> None:
    """写文件：先写 .tmp 再 rename，避免崩溃中途破坏现有文件。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def cmd_validate(_args: argparse.Namespace) -> int:
    try:
        versions = load_yaml()
    except (OSError, ValueError) as e:
        print(f"load failed: {e}", file=sys.stderr)
        return 2
    r = validate(versions)
    print_validate_result(r)
    return 1 if r.has_errors else 0


def cmd_render_changelog(_args: argparse.Namespace) -> int:
    try:
        versions = load_yaml()
    except (OSError, ValueError) as e:
        print(f"load failed: {e}", file=sys.stderr)
        return 2
    r = validate(versions)
    if r.has_errors:
        print("validate FAILED — render-changelog 拒绝执行：")
        print_validate_result(r)
        return 1
    content = render_changelog(versions)
    write_atomic(CHANGELOG_PATH, content)
    print(f"[render] {CHANGELOG_PATH.relative_to(REPO_ROOT)} 重写完成（{len(versions)} 个版本）")
    return 0


def _read_studio_version() -> Optional[str]:
    if not STUDIO_INIT_PATH.exists():
        return None
    txt = STUDIO_INIT_PATH.read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', txt)
    return m.group(1) if m else None


def _write_studio_version(new_version: str) -> bool:
    if not STUDIO_INIT_PATH.exists():
        return False
    txt = STUDIO_INIT_PATH.read_text(encoding="utf-8")
    new_txt, n = re.subn(
        r'(__version__\s*=\s*["\'])([^"\']+)(["\'])',
        rf'\g<1>{new_version}\g<3>',
        txt,
        count=1,
    )
    if n == 0:
        return False
    write_atomic(STUDIO_INIT_PATH, new_txt)
    return True


def _read_package_json_version() -> Optional[str]:
    if not PACKAGE_JSON_PATH.exists():
        return None
    data = json.loads(PACKAGE_JSON_PATH.read_text(encoding="utf-8"))
    return data.get("version")


def _write_package_json_version(new_version: str) -> bool:
    if not PACKAGE_JSON_PATH.exists():
        return False
    # 用正则替换 "version": "..." 一行，保留原始 indent / trailing newline。
    txt = PACKAGE_JSON_PATH.read_text(encoding="utf-8")
    new_txt, n = re.subn(
        r'(\"version\"\s*:\s*\")([^\"]+)(\")',
        rf'\g<1>{new_version}\g<3>',
        txt,
        count=1,
    )
    if n == 0:
        return False
    write_atomic(PACKAGE_JSON_PATH, new_txt)
    return True


def cmd_bump(args: argparse.Namespace) -> int:
    try:
        versions = load_yaml()
    except (OSError, ValueError) as e:
        print(f"load failed: {e}", file=sys.stderr)
        return 2

    r = validate(versions)
    if r.has_errors:
        print("validate FAILED — bump 拒绝执行：")
        print_validate_result(r)
        return 1
    print_validate_result(r)

    if not versions:
        print("release_notes.yaml 没有版本 block，bump 无事可做", file=sys.stderr)
        return 2

    top_version = versions[0]["version"]
    if args.version and args.version != top_version:
        print(
            f"--version={args.version} 与 yaml top version={top_version} 不符。\n"
            "yaml 是 source of truth；要 bump 到 {args.version}，请先在 yaml 顶部加该 block。",
            file=sys.stderr,
        )
        return 2

    target_version = args.version or top_version
    cur_studio = _read_studio_version()
    cur_pkg = _read_package_json_version()

    print(f"\n[bump] target version: {target_version}")
    print(f"[bump] studio/__init__.py: {cur_studio} → {target_version}")
    print(f"[bump] studio/web/package.json: {cur_pkg} → {target_version}")

    if not _write_studio_version(target_version):
        print("[bump] WARN: studio/__init__.py 没找到 __version__ 字段，跳过", file=sys.stderr)
    if not _write_package_json_version(target_version):
        print("[bump] WARN: package.json 没找到 version 字段，跳过", file=sys.stderr)

    content = render_changelog(versions)
    write_atomic(CHANGELOG_PATH, content)
    print(f"[bump] {CHANGELOG_PATH.relative_to(REPO_ROOT)} 重写完成")

    print()
    print("next: 检查 git diff 后")
    print(f"  git add -A && git commit -m 'chore(release): {target_version}'")
    print(f"  git tag v{target_version}")
    print(f"  git push --tags")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bump_version", description=__doc__.splitlines()[0] if __doc__ else "")
    sub = p.add_subparsers(dest="cmd")

    p_v = sub.add_parser("validate", help="schema 校验 release_notes.yaml")
    p_v.set_defaults(func=cmd_validate)

    p_r = sub.add_parser("render-changelog", help="从 yaml 重写 CHANGELOG.md")
    p_r.set_defaults(func=cmd_render_changelog)

    p_b = sub.add_parser("bump", help="同步版本号到 __init__.py + package.json + CHANGELOG.md")
    p_b.add_argument("--version", help="期望的目标版本（与 yaml top 不符时报错）", default=None)
    p_b.set_defaults(func=cmd_bump)

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
