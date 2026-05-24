"""把 2026-05-21 ~ 2026-05-24 之间前端 `downloadCurrentPreset` bug 导出的
"假 yaml 真 toml"预设文件转成真 yaml。

Bug 详情：那 3 天的 Studio 前端在导出预设时调 `generateToml(config)` 生成
手写 TOML（key = value 平铺 + 多行 `{...}` 假 inline table + null 写成
`key = `），但 blob MIME / 文件后缀都打了 yaml 标签。server
`parse_preset_bytes` 仅认 yaml/json，所以这些文件再上传会被拒：
    "预设格式错误（顶层不是 mapping）"

新版前端已改走 server `GET /api/presets/{name}/download` 端点直发磁盘上的
原始 yaml，不再产生此类文件。本工具仅用于救活历史下载文件，不进 server
依赖链路（avoid 给生产 import 加 TOML parser）。

Usage:
    python tools/preset_toml_to_yaml.py broken1.yaml [broken2.yaml ...]
        每个文件输出到同目录的 `<stem>.fixed.yaml`，原文件保留。
    python tools/preset_toml_to_yaml.py --in-place broken.yaml
        就地覆盖原文件，覆盖前备份为 `<stem>.yaml.toml-bak`。
    python tools/preset_toml_to_yaml.py --output out.yaml broken.yaml
        指定输出路径（仅单文件适用）。
    python tools/preset_toml_to_yaml.py --no-validate broken.yaml
        跳过 TrainingConfig 校验（schema 漂移时应急用，默认开启）。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    import tomllib  # py 3.11+
except ImportError:  # pragma: no cover - py 3.10 及以下走 tomli 兜底
    try:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]
    except ImportError:
        sys.stderr.write(
            "缺 TOML 解析库：py < 3.11 需先 `pip install tomli`。\n"
        )
        sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _preprocess(text: str) -> tuple[str, list[str]]:
    """修复前端 `generateToml` 产出的非标准 TOML，返回 (cleaned_text, warnings)。

    两类已知不合规：
    - **多行 `{...}` 块**：TOML inline table 必须单行；这里把多行块的每行
      `k = v` 拼成单行 `{ k = v, k2 = v2 }`。
    - **空值 `key = `（null/undefined）**：TOML 不允许空 rhs；直接 drop 这些
      key，由 pydantic 走 schema 默认值 / Optional[None] 兜底。
    """
    warnings: list[str] = []
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # 多行块开头：`key = {`
        m = re.match(r"^(\S.*?=\s*)\{\s*$", line)
        if m:
            prefix = m.group(1).rstrip()
            inner: list[str] = []
            j = i + 1
            while j < len(lines) and lines[j].strip() != "}":
                s = lines[j].strip()
                if s:
                    inner.append(s)
                j += 1
            if j >= len(lines):
                # 没找到闭合 } —— 留给 tomllib 报原始错
                out.append(line)
                i += 1
                continue
            out.append(f"{prefix} {{ {', '.join(inner)} }}")
            i = j + 1
            continue
        # 空值：`key = ` 或 `key =`（rhs 全空白）
        if "=" in stripped and not stripped.startswith("#"):
            k, _, v = stripped.partition("=")
            if v.strip() == "":
                warnings.append(f"drop 空值 key: {k.strip()}")
                i += 1
                continue
        out.append(line)
        i += 1
    return "\n".join(out) + "\n", warnings


def _validate(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """走 TrainingConfig.model_validate 规范化；返回 (规范化后的 dict, warnings)。"""
    # 延迟 import：tool 默认应该能在不装训练栈的环境跑（仅 yaml + tomllib）
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from studio.schema import TrainingConfig  # type: ignore[import-not-found]
    except ImportError as exc:
        return data, [f"跳过 schema 校验（import 失败：{exc}）"]
    try:
        cfg = TrainingConfig.model_validate(data)
    except Exception as exc:
        # 校验失败不致命：写一份原始数据 + 警告，让用户人工修
        return data, [f"TrainingConfig 校验失败（已写原始数据，请手工修）：{exc}"]
    return cfg.model_dump(mode="python"), []


def convert(text: str, validate: bool = True) -> tuple[str, list[str]]:
    """主转换：返回 (yaml_text, warnings)。"""
    warnings: list[str] = []
    cleaned, pre_warns = _preprocess(text)
    warnings.extend(pre_warns)
    try:
        data = tomllib.loads(cleaned)
    except Exception as exc:
        raise SystemExit(f"TOML 解析失败：{exc}\n（预处理后内容前 200 字符）\n{cleaned[:200]}")
    if not isinstance(data, dict):
        raise SystemExit(f"TOML 顶层不是 mapping，得到 {type(data).__name__}")
    if validate:
        data, val_warns = _validate(data)
        warnings.extend(val_warns)
    yaml_text = yaml.safe_dump(
        data, allow_unicode=True, sort_keys=False, default_flow_style=False
    )
    return yaml_text, warnings


def _convert_file(src: Path, dst: Path, validate: bool, in_place: bool) -> None:
    if not src.exists():
        raise SystemExit(f"文件不存在：{src}")
    text = src.read_text(encoding="utf-8")
    yaml_text, warnings = convert(text, validate=validate)
    if in_place:
        bak = src.with_suffix(src.suffix + ".toml-bak")
        if bak.exists():
            raise SystemExit(f"备份目标已存在，先删它再重试：{bak}")
        src.rename(bak)
        sys.stderr.write(f"  备份原文件 → {bak}\n")
    dst.write_text(yaml_text, encoding="utf-8")
    sys.stderr.write(f"  写入 → {dst}\n")
    for w in warnings:
        sys.stderr.write(f"  ⚠ {w}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("files", nargs="+", type=Path, help="待转换的 .yaml（实为 TOML）文件")
    ap.add_argument(
        "--in-place", action="store_true",
        help="就地覆盖原文件，原文件备份为 .yaml.toml-bak",
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help="指定输出路径，仅在传单个文件时生效",
    )
    ap.add_argument(
        "--no-validate", action="store_true",
        help="跳过 TrainingConfig schema 校验（schema 漂移时应急用）",
    )
    args = ap.parse_args(argv)

    if args.output and len(args.files) > 1:
        ap.error("--output 仅支持单文件输入")
    if args.output and args.in_place:
        ap.error("--output 与 --in-place 互斥")

    validate = not args.no_validate
    for src in args.files:
        sys.stderr.write(f"转换 {src}\n")
        if args.in_place:
            dst = src
        elif args.output:
            dst = args.output
        else:
            dst = src.with_name(f"{src.stem}.fixed{src.suffix}")
        _convert_file(src, dst, validate=validate, in_place=args.in_place)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
