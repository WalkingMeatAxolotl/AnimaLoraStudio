"""按 show_when 元数据裁剪「当前配置下 UI 不可见」的字段。

前端 SchemaForm 用 Field 元数据里的 show_when 决定字段显隐（求值器在
studio/web/src/lib/schema.ts 的 evalShowWhen）。这里提供同一语义的 Python
镜像，在 yaml 落盘前把 show_when 为假的字段从 dump 里删掉，让 config.yaml /
预设文件 / 任务快照只包含真正生效的参数。改这里的求值语义必须同步改前端。

runtime 侧的安全性：argparse parser 与默认值同样派生自 TrainingConfig
（studio/infrastructure/argparse_bridge.py），yaml 缺失的键在
merge_yaml_into_namespace 后落回同一份 schema 默认值，与显式写默认值等价。

只裁 show_when —— disable_when 命中的字段被前端 reset 到 disable_value，
而 disable_value 可能不等于字段默认值（如 Prodigy 时 lr_scheduler 被钉在
"none"），裁掉会让 runtime 读回默认值改变行为，所以保留。
"""
from typing import Any, Mapping

from pydantic import BaseModel

from .training import TrainingConfig

_MISSING = object()


def _js_str(value: Any) -> str:
    """JS `String(v)` 的等价物 —— show_when 比较按前端的字符串化语义。

    差异点：JS 的 true/false 小写；整数值的 float 不带小数点
    （String(1.0) === "1"，而 Python str(1.0) == "1.0"，training.py 里
    `timestep_schedule_shift!=1` 依赖这一点）。
    """
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if value is _MISSING:
        return "undefined"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def eval_show_when(expr: str | None, values: Mapping[str, Any]) -> bool:
    """schema.ts evalShowWhen 的逐字镜像：`||` 分支 any、`&&` 合取 all、
    `==`/`!=` 字符串比较；空表达式与解析不出的表达式都返回 True（failsafe）。"""
    if not expr:
        return True
    branches = [p.strip() for p in expr.split("||") if p.strip()]
    if len(branches) > 1:
        return any(eval_show_when(b, values) for b in branches)
    ands = [p.strip() for p in expr.split("&&") if p.strip()]
    if len(ands) > 1:
        return all(eval_show_when(c, values) for c in ands)
    eq = expr.split("==")
    if len(eq) == 2:
        return _js_str(values.get(eq[0].strip(), _MISSING)) == eq[1].strip()
    ne = expr.split("!=")
    if len(ne) == 2:
        return _js_str(values.get(ne[0].strip(), _MISSING)) != ne[1].strip()
    return True


def prune_inactive_fields(
    dumped: dict[str, Any], model_cls: type[BaseModel] = TrainingConfig
) -> dict[str, Any]:
    """从 model_dump 结果里删掉 show_when 求值为假的字段。

    所有表达式都对完整的 `dumped` 求值（与前端对完整表单 state 求值一致），
    先删的字段不影响后续判断。没有 show_when 的字段（含 hidden / disable_when
    字段）原样保留。
    """
    out = dict(dumped)
    for name, field in model_cls.model_fields.items():
        extra = field.json_schema_extra
        if not isinstance(extra, dict):
            continue
        expr = extra.get("show_when")
        if isinstance(expr, str) and expr and not eval_show_when(expr, dumped):
            out.pop(name, None)
    return out
