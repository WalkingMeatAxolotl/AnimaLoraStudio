"""按 UI 可见性元数据裁剪 yaml 落盘不需要的字段。

两条裁剪规则（studio/web/src/lib/schema.ts 的 pruneInactiveConfig 是同一
语义的前端镜像，YAML 预览抽屉靠它和落盘内容一致，改语义必须两边同步）：

1. show_when 求值为假 —— 当前配置下 UI 不可见、值不生效的字段
   （求值器 eval_show_when 逐字镜像前端 evalShowWhen）。
2. hidden=True 且值等于 schema 默认值 —— UI 永不渲染的字段（终端体验旋钮、
   trigger_word 等），默认值落盘纯属噪音；非默认值（裸 CLI 用户手写覆盖 /
   Tagging 页写入的 trigger_word）照常保留。

runtime 侧的安全性（config 管线刀 1 / R1 起）：trainer 加载 yaml 走与 Studio
同一条 TrainingConfig 构造路径（argparse_bridge.namespace_from_config），缺失
键经 pydantic 默认值 + FAMILY_CONFIG_DEFAULTS 族 overlay 补齐 —— 与保存端裁剪
前的读回值逐字段一致。历史教训：旧 merge_yaml_into_namespace 用 argparse 裸
默认补缺键、不走族 overlay，krea2 上被裁掉的 shuffle_caption 落回 anima 语义
默认 True 而拒训；裁剪的安全性永远以「trainer 读回 == 保存端读回」为准。

不裁 disable_when —— 命中的字段被前端 reset 到 disable_value，而
disable_value 可能不等于字段默认值（如 Prodigy 时 lr_scheduler 被钉在
"none"），裁掉会让 runtime 读回默认值改变行为，所以保留。
"""
from typing import Any

from pydantic import BaseModel

# 求值器已下沉 config_rules(零依赖叶子,training 的 validator 也要用,放这边
# 会与 `config_prune → training` 成环);此处 re-export 保持既有消费方不变。
from .config_rules import _MISSING, _js_str, eval_show_when  # noqa: F401
from .training import TrainingConfig


def prune_inactive_fields(
    dumped: dict[str, Any], model_cls: type[BaseModel] = TrainingConfig
) -> dict[str, Any]:
    """从 model_dump 结果里删掉 show_when 求值为假的字段，以及 hidden=True
    且仍是 schema 默认值的字段。

    所有表达式都对完整的 `dumped` 求值（与前端对完整表单 state 求值一致），
    先删的字段不影响后续判断。disable_when 字段与无元数据字段原样保留。
    """
    out = dict(dumped)
    for name, field in model_cls.model_fields.items():
        extra = field.json_schema_extra
        if not isinstance(extra, dict):
            continue
        expr = extra.get("show_when")
        if isinstance(expr, str) and expr and not eval_show_when(expr, dumped):
            out.pop(name, None)
            continue
        if extra.get("hidden") and name in out:
            if out[name] == field.get_default(call_default_factory=True):
                out.pop(name)
    return out
