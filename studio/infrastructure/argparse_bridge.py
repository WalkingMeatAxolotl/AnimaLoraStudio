"""把 pydantic v2 模型反向编译成 `argparse.ArgumentParser`。

设计目标：
    - 让 schema.py 的 TrainingConfig 成为 CLI / YAML / Web 表单的唯一权威源
    - 现有 anima_train.py 的 CLI 习惯：少量字段使用别名（如 --lr ↔ learning_rate）
      —— 通过 json_schema_extra={"cli_alias": "--lr"} 显式声明
    - 不试图替换 argparse 的语义，只是自动把字段类型/约束/默认值翻译成参数声明

支持的字段类型：
    bool                  → BooleanOptionalAction，--foo / --no-foo
    Literal["a", "b"]     → choices=["a","b"]
    int / float / str     → type=...
    list[T]               → nargs="*", type=T
    Optional[T]           → 同 T，但默认 None；空字符串 / None 都能落到默认值
"""
from __future__ import annotations

import argparse
import types
from typing import Any, Literal, Optional, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo


# ---------------------------------------------------------------------------
# 类型分析
# ---------------------------------------------------------------------------


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Optional[X] / X | None → (X, True)。否则 (annotation, False)。"""
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], True
    return annotation, False


def _is_list(annotation: Any) -> bool:
    return get_origin(annotation) in (list,)


def _list_item_type(annotation: Any) -> Any:
    args = get_args(annotation)
    return args[0] if args else str


def _is_literal(annotation: Any) -> bool:
    return get_origin(annotation) is Literal


# ---------------------------------------------------------------------------
# 字段 → argparse
# ---------------------------------------------------------------------------


def _default_value(field: FieldInfo) -> Any:
    """提取 Field default / default_factory（pydantic v2 用 PydanticUndefined 表示无默认）。"""
    from pydantic_core import PydanticUndefined

    if field.default is not PydanticUndefined and field.default is not None:
        return field.default
    if field.default_factory is not None:
        try:
            return field.default_factory()  # type: ignore[call-arg]
        except TypeError:  # validator-based factory
            return None
    return None if field.default is None else field.default


def _flag_for(name: str, field: FieldInfo) -> str:
    extra = field.json_schema_extra or {}
    alias = extra.get("cli_alias") if isinstance(extra, dict) else None
    return alias or "--" + name.replace("_", "-")


def add_argument_for(parser: argparse.ArgumentParser, name: str, field: FieldInfo) -> None:
    """把单个字段加到 parser。dest 始终等于字段名（即下划线形式）。"""
    flag = _flag_for(name, field)
    annotation, is_optional = _unwrap_optional(field.annotation)
    default = _default_value(field)
    help_text = (field.description or "").strip()

    # bool ----------------------------------------------------------------
    if annotation is bool:
        parser.add_argument(
            flag,
            dest=name,
            action=argparse.BooleanOptionalAction,
            default=bool(default) if default is not None else False,
            help=help_text or None,
        )
        return

    # Literal -------------------------------------------------------------
    if _is_literal(annotation):
        choices = list(get_args(annotation))
        # Literal 元素类型一致；若全是 str，type=str
        item_t = type(choices[0]) if choices else str
        parser.add_argument(
            flag,
            dest=name,
            choices=choices,
            type=item_t,
            default=default,
            help=help_text or None,
        )
        return

    # list[T] -------------------------------------------------------------
    if _is_list(annotation):
        item_t = _list_item_type(annotation)
        parser.add_argument(
            flag,
            dest=name,
            nargs="*",
            type=item_t,
            default=default if default is not None else [],
            help=help_text or None,
        )
        return

    # int / float / str ---------------------------------------------------
    if annotation is int:
        parser.add_argument(flag, dest=name, type=int, default=default, help=help_text or None)
        return
    if annotation is float:
        parser.add_argument(flag, dest=name, type=float, default=default, help=help_text or None)
        return

    # 默认按字符串处理（包括 str、Optional[str]、未知类型）
    parser.add_argument(
        flag,
        dest=name,
        type=str,
        default=default if default is not None else ("" if not is_optional else None),
        help=help_text or None,
    )


def build_parser(
    model_cls: type[BaseModel],
    *,
    prog: str | None = None,
    description: str | None = None,
    add_config_arg: bool = True,
) -> argparse.ArgumentParser:
    """从 pydantic 模型生成完整 parser。

    add_config_arg=True 时自动加上 `--config PATH`（指向 YAML 配置）。
    """
    parser = argparse.ArgumentParser(prog=prog, description=description)
    if add_config_arg:
        parser.add_argument("--config", default="", help="YAML 配置文件路径")
    for name, field in model_cls.model_fields.items():
        add_argument_for(parser, name, field)
    return parser


# ---------------------------------------------------------------------------
# YAML → Namespace 合并（CLI 优先于 YAML）
# ---------------------------------------------------------------------------


def merge_yaml_into_namespace(
    args: argparse.Namespace,
    yaml_data: dict[str, Any],
    model_cls: type[BaseModel],
) -> argparse.Namespace:
    """把 YAML 的字段写进 args，但仅当 args 当前值仍等于该字段的默认值时。

    这与 anima_train.py 现有的「CLI 显式设置则保留」语义一致；CLI 的实际显式
    设置无法直接探测，因此用「值 == 默认值」做近似。
    """
    fields = model_cls.model_fields
    for key, value in yaml_data.items():
        if key not in fields:
            continue
        default = _default_value(fields[key])
        current = getattr(args, key, default)
        # 只在 args 还是默认值时让 YAML 覆盖
        if current == default:
            setattr(args, key, value)
    return args
