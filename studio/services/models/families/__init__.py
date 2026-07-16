"""模型族资产 registry（studio 侧三居所之一，多模型 PR-4）。

与 runtime `training/families` 的 SPECS 共用族名 join key（"anima" / "krea2"）。
下载资产允许先于训练实现落地，测试保证所有 runtime 族都有对应资产清单。
判定标准（01 §8.1）：出现在
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

from typing import Optional

from . import anima as _anima
from . import krea2 as _krea2

FAMILY_ASSETS = {
    "anima": _anima.ASSETS,
    "krea2": _krea2.ASSETS,
}


def get_assets(family_id: str):
    try:
        return FAMILY_ASSETS[str(family_id)]
    except KeyError:
        raise ValueError(
            f"未知模型族 '{family_id}'，已注册: {sorted(FAMILY_ASSETS)}"
        ) from None


def default_paths_for_new_version(
    base_model: Optional[str] = None, *, family: str = "anima"
) -> dict[str, str]:
    """按族解析新建 version 的 4 个权重路径字段（registry 派发，多模型 P4-1）。

    历史上这个名字直接绑定 Anima 实现，6 个调用面（preset fork / 保存为预设 /
    version config hint / bundle 导入 / path-defaults 端点 / 先验生成）都拿到
    anima 路径——krea2 版本一「换预设」config 即被 anima 路径覆写。调用方必须
    把 config 里的 `model_family` 传进来；未知族抛 ValueError（列已注册项）。
    """
    return get_assets(family).default_paths_for_new_version(base_model)
