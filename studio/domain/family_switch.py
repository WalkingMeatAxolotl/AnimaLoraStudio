"""模型族切换的纯计算（多模型 P4-3，C5）。

「切换族」不是裸字段编辑：4 个权重路径要按目标族重算、族风味字段
（sampler / scheduler / timestep 等）要重置为目标族默认、目标族不支持的
能力字段要关闭——否则落盘的就是 `model_family: krea2` + anima 路径的
坏 config（#419 债 C2），或 t5 自定义路径被 show_when 裁剪静默丢失（C3）。

本模块只做 dict → dict 的纯计算与变更清单；路径重算依赖 services 层
（依赖方向 domain ← services 不可反向），由调用方注入 `path_defaults`。
不落盘——前端确认后走正常保存链路。
"""
from typing import Any

from .common import FAMILY_CONFIG_DEFAULTS, capability_violations
from .training import TrainingConfig


def _flavor_keys() -> set:
    """族风味字段 = 全部族 config_defaults 键的并集。

    这些字段的语义按族漂移（krea2 的 sampler=euler / timestep=krea2_shift 等），
    切换时统一重置：目标族有 overlay 用 overlay 值，否则回 schema 默认。
    """
    keys: set = set()
    for defaults in FAMILY_CONFIG_DEFAULTS.values():
        keys |= set(defaults)
    return keys


def switch_family(
    config: dict[str, Any],
    target: str,
    path_defaults: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """把一份 TrainingConfig dict 显式切换到 `target` 族。

    Args:
        config: 当前 config（dict，允许部分字段缺失）
        target: 目标族 id（调用方已校验合法）
        path_defaults: 目标族的 4 个权重路径（services 层
            `default_paths_for_new_version(family=target)` 的产物）

    Returns:
        (切换后的完整 config dict, 变更清单)。变更清单按字段列
        `{"field", "from", "to"}`，仅含值真正变化的字段；`from` 缺失时为
        None（config 里原本没有该字段）。
    """
    schema_defaults = TrainingConfig().model_dump(mode="python")
    new: dict[str, Any] = dict(config)
    new["model_family"] = target
    # 路径统一正斜杠（yaml 落盘同款归一化）；与旧值仅斜杠风格不同 = 同一
    # 文件，保留原值不产生假变更行（str(Path) 在 Windows 上吐反斜杠）。
    for key, value in path_defaults.items():
        normalized = str(value).replace("\\", "/") if value else value
        old = config.get(key)
        if old is not None and str(old).replace("\\", "/") == normalized:
            new[key] = old
        else:
            new[key] = normalized

    overlay = FAMILY_CONFIG_DEFAULTS.get(target, {})
    for key in sorted(_flavor_keys()):
        new[key] = overlay.get(key, schema_defaults.get(key))

    # 目标族不支持的能力字段（navit / sra / leap / tag 语义等）显式关回
    # schema 默认——否则 capability validator 会拒绝整份 config。
    for field in capability_violations(target, new):
        new[field] = schema_defaults.get(field)

    changes = [
        {"field": key, "from": config.get(key), "to": new[key]}
        for key in sorted(new)
        if config.get(key) != new[key]
    ]
    return new, changes
