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


def add_argument_for(
    parser: argparse.ArgumentParser,
    name: str,
    field: FieldInfo,
    *,
    suppress_default: bool = False,
) -> None:
    """把单个字段加到 parser。dest 始终等于字段名（即下划线形式）。

    suppress_default=True 时所有参数 default=argparse.SUPPRESS —— parse 产物
    Namespace 只含用户显式传入的键（sparse），CLI 显式性由键的存在性精确判定，
    供 namespace_from_config 做 CLI > YAML 合并。
    """
    flag = _flag_for(name, field)
    annotation, is_optional = _unwrap_optional(field.annotation)
    default = _default_value(field)
    # argparse format_help 把 description 当 printf 模板做 `% params` 展开，
    # description 里裸 `%`（如 "占 90%"）会让 --help 直接 ValueError。
    # 项目里 schema description 同时给 Web UI / i18n 用，不应被 argparse 语义污染，
    # 因此在 bridge 一层把所有裸 `%` 转义 —— 全项目无人使用 %(default)s 这类
    # argparse named substitution，escape 不会破坏既有用法。
    help_text = (field.description or "").strip().replace("%", "%%")

    # bool ----------------------------------------------------------------
    if annotation is bool:
        # Optional[bool] 的默认值保留 None —— 表示「未指定」；
        # 非 Optional 的 bool 则 fallback 到 False。
        if is_optional:
            actual_default = default  # keep None
        else:
            actual_default = bool(default) if default is not None else False
        if suppress_default:
            actual_default = argparse.SUPPRESS
        # Python 3.13+ 的 argparse 拒绝把 --no-X 形式的 flag 传给
        # BooleanOptionalAction（会自动衍生 --no-no-X 与字段重名）。
        # 字段名以 no_ 开头时退化为一对互斥 store_true/store_false：
        #   --no-X    → store_true  (no_X = True)
        #   --X       → store_false (no_X = False)
        if name.startswith("no_") and len(name) > 3:
            positive = "--" + name[3:].replace("_", "-")
            parser.add_argument(
                flag,
                dest=name,
                action="store_true",
                default=actual_default,
                help=help_text or None,
            )
            # 第二个同 dest action 的 default 必须与第一个一致：argparse 在
            # parse 开始时按注册顺序 setattr default，非 SUPPRESS 的 None 会
            # 把 sparse namespace 撑出一个假显式键。
            parser.add_argument(
                positive, dest=name, action="store_false",
                default=actual_default, help=None,
            )
        else:
            parser.add_argument(
                flag,
                dest=name,
                action=argparse.BooleanOptionalAction,
                default=actual_default,
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
            default=argparse.SUPPRESS if suppress_default else default,
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
            default=argparse.SUPPRESS if suppress_default
            else (default if default is not None else []),
            help=help_text or None,
        )
        return

    # int / float / str ---------------------------------------------------
    if annotation is int:
        parser.add_argument(
            flag, dest=name, type=int,
            default=argparse.SUPPRESS if suppress_default else default,
            help=help_text or None,
        )
        return
    if annotation is float:
        parser.add_argument(
            flag, dest=name, type=float,
            default=argparse.SUPPRESS if suppress_default else default,
            help=help_text or None,
        )
        return

    # 默认按字符串处理（包括 str、Optional[str]、未知类型）
    parser.add_argument(
        flag,
        dest=name,
        type=str,
        default=argparse.SUPPRESS if suppress_default
        else (default if default is not None else ("" if not is_optional else None)),
        help=help_text or None,
    )


def build_parser(
    model_cls: type[BaseModel],
    *,
    prog: str | None = None,
    description: str | None = None,
    add_config_arg: bool = True,
    suppress_defaults: bool = False,
) -> argparse.ArgumentParser:
    """从 pydantic 模型生成完整 parser。

    add_config_arg=True 时自动加上 `--config PATH`（指向 YAML 配置；不参与
    suppress —— 调用方在合并前就要读它）。
    suppress_defaults=True 时 schema 字段全部 default=argparse.SUPPRESS，
    parse 产物只含用户显式传入的键，配合 namespace_from_config 使用。
    """
    parser = argparse.ArgumentParser(prog=prog, description=description)
    if add_config_arg:
        parser.add_argument("--config", default="", help="YAML 配置文件路径")
    for name, field in model_cls.model_fields.items():
        add_argument_for(parser, name, field, suppress_default=suppress_defaults)
    return parser


# ---------------------------------------------------------------------------
# YAML + CLI 显式值 → pydantic 校验 → 完整 Namespace
# ---------------------------------------------------------------------------


def namespace_from_config(
    args: argparse.Namespace,
    yaml_data: dict[str, Any],
    model_cls: type[BaseModel],
) -> argparse.Namespace:
    """合并 YAML 与 CLI 显式值，经 model_cls 完整构造后展开成 Namespace。

    取代旧 merge_yaml_into_namespace（「值==默认值」近似 CLI 显式性且绕过全部
    validator——字段迁移 / FAMILY_CONFIG_DEFAULTS 族默认 overlay / 互斥与能力
    校验在 trainer 路径全部失效，如 krea2 缺键 shuffle_caption 落回 anima 语义
    默认 True 而拒训）。调用方的 parser 必须以 suppress_defaults=True 构建，
    `args` 只含显式键，CLI > YAML 的优先级是精确判定。

    合并语义：
    - yaml_data 原样交给 pydantic —— 旧键由 before-validator 迁移，未知键按
      模型的 extra 策略处理（TrainingConfig 为 ignore）
    - args 中属于 schema 的键覆盖 YAML（CLI 显式优先）；合并结果整体过一次
      模型构造，任何来源组合出的非法配置都在此 fail-fast
    - args 中不属于 schema 的键（--interactive 等 CLI-only 开关）原样带回

    Raises:
        pydantic.ValidationError: 合并结果非法（互斥冲突 / 族能力越界等）。
    """
    fields = model_cls.model_fields
    explicit = vars(args)
    merged = dict(yaml_data)
    merged.update({k: v for k, v in explicit.items() if k in fields})
    cfg = model_cls(**merged)
    out = argparse.Namespace(
        **{k: v for k, v in explicit.items() if k not in fields}
    )
    for key, value in cfg.model_dump(mode="python").items():
        setattr(out, key, value)
    return out
