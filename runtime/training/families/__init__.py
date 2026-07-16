"""模型族 registry —— 第 8 套 plugin registry，首个架构级（多模型支持 Phase 1）。

PR-1 只承载 ModelSpec（声明式常量单一来源 + latent 缓存指纹协议）；
ModelFamily 行为接口与 get_family() 派发随 PR-2b 落地，派发经 anima_train
公共命名空间收口全部 6 个调用面（docs/design/multi-model/04-synthesis.md D8'/§5）。
"""

from __future__ import annotations

from training.families.spec import (  # noqa: F401  (re-export)
    ConstantShift,
    KNOWN_CAPABILITIES,
    LatentSpec,
    LoraOutputSpec,
    ModelSpec,
    ResolutionAwareShift,
    SamplingDefaults,
    TextSpec,
    validate_spec,
)
from training.families.anima import ANIMA_SPEC
from training.families.krea2 import KREA2_SPEC

SPECS: dict[str, ModelSpec] = {}


def _register(spec: ModelSpec) -> None:
    validate_spec(spec)
    if spec.family_id in SPECS:
        raise ValueError(f"模型族重复注册: {spec.family_id}")
    SPECS[spec.family_id] = spec


_register(ANIMA_SPEC)
_register(KREA2_SPEC)


_FAMILIES: dict[str, object] = {}


def get_family(family_id: str):
    """按族 id 取 ModelFamily 实例（惰性构造）。未知 id → ValueError。"""
    get_spec(family_id)  # 未知 id 在此报错并列出已注册项
    fam = _FAMILIES.get(family_id)
    if fam is None:
        if family_id == "anima":
            from training.families.anima.family import AnimaFamily

            fam = AnimaFamily()
        elif family_id == "krea2":
            from training.families.krea2 import Krea2Family

            fam = Krea2Family()
        else:  # pragma: no cover - registry 与 SPECS 同步维护
            raise ValueError(f"模型族 '{family_id}' 缺少 ModelFamily 实现")
        _FAMILIES[family_id] = fam
    return fam


def resolve_family(args):
    """从 args/cfg 解析 model_family（缺省 anima，D7 零迁移）。支持
    argparse.Namespace 与 dict 两种载体（旁路调用方的 cfg 是 dict）。"""
    if isinstance(args, dict):
        fid = args.get("model_family") or "anima"
    else:
        fid = getattr(args, "model_family", "anima") or "anima"
    return get_family(str(fid))


def get_spec(family_id: str) -> ModelSpec:
    """按族 id 取 ModelSpec。未知 id → ValueError 并列出已注册项。"""
    try:
        return SPECS[str(family_id)]
    except KeyError:
        raise ValueError(
            f"未知模型族 '{family_id}'，已注册: {sorted(SPECS)}"
        ) from None
