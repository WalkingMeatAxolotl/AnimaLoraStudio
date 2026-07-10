"""domain 共享原语：Field 元信息 helper、attention backend 类型、分组顺序。

被 training.py / generate.py / reg.py 等 model 文件共用。

注意：不使用 `from __future__ import annotations`——Pydantic v2 + Python 3.12+
在延迟求值模式下会将 typing._SpecialForm 当成 schema key，触发 AttributeError。
"""
from typing import Any, Literal


def _meta(group: str, control: str = "auto", **extra: Any) -> dict[str, Any]:
    """Field 的 json_schema_extra payload —— 前端按 group 分区、按 show_when 条件显示。"""
    return {"group": group, "control": control, **extra}


# attention backend 三选一（替代历史的 xformers / flash_attn 双 bool）
AttentionBackend = Literal["none", "xformers", "flash_attn"]


# 前端 SchemaForm 按这个顺序渲染区块。
# 每组：(key, label, default_collapsed)。default_collapsed=True 让前端初始默认折叠。
# 模型路径 readonly 显示「自动 · 全局设置」徽章，不折叠。
GROUP_ORDER: list[tuple[str, str, bool]] = [
    ("model", "模型路径", False),
    ("dataset", "数据集", False),
    ("caption", "Caption 处理", False),
    ("lora", "网络设置", False),
    ("training", "训练参数", False),
    ("noise_augmentation", "噪声增强", False),
    ("timestep_sampling", "时间步采样", False),
    ("loss", "损失", False),
    ("system", "系统与性能", False),
    ("output", "输出与保存", False),
    ("sample", "采样", False),
    ("eval_validation", "训练后指标评估", True),
    ("monitor", "监控与进度", False),
]
