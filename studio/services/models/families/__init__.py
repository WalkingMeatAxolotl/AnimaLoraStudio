"""模型族资产 registry（studio 侧三居所之一，多模型 PR-4）。

与 runtime `training/families` 的 SPECS 共用族名 join key（"anima" / "krea2"），
同步由 tests/test_model_family_gating.py 系列锁死。判定标准（01 §8.1）：出现在
TrainingConfig 权重路径字段里的是「模型族资产」进本包；打标 / 放大 / 评估 /
预览解码等「工具模型」永远留在 ..paths。

每个族模块暴露一个 ASSETS 对象（duck-typed）：
- family_id / display_name
- default_paths_for_new_version(base_model) — 新建 version 的权重绝对路径
- transformer_path_for(sel) — 显式底模选择 → transformer 绝对路径
- selected_variant() — Settings 当前选中 variant
- catalog_sections(root, models_cfg) — /api/models/catalog 的本族区块
"""
from __future__ import annotations

from . import anima as _anima

FAMILY_ASSETS = {
    "anima": _anima.ASSETS,
}


def get_assets(family_id: str):
    try:
        return FAMILY_ASSETS[str(family_id)]
    except KeyError:
        raise ValueError(
            f"未知模型族 '{family_id}'，已注册: {sorted(FAMILY_ASSETS)}"
        ) from None
