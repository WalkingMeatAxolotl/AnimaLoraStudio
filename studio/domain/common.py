"""domain 共享原语：Field 元信息 helper、attention backend 类型、分组顺序。

被 training.py / generate.py / reg.py 等 model 文件共用。

注意：不使用 `from __future__ import annotations`——Pydantic v2 + Python 3.12+
在延迟求值模式下会将 typing._SpecialForm 当成 schema key，触发 AttributeError。
"""
from typing import Any, Literal


def _meta(group: str, control: str = "auto", **extra: Any) -> dict[str, Any]:
    """Field 的 json_schema_extra payload —— 前端按 group 分区、按 show_when 条件显示。"""
    return {"group": group, "control": control, **extra}


# ─── 多模型能力矩阵（多模型 PR-3，04-synthesis D5/D11） ───────────────────
# 单一权威源（config 管线刀 1 / R3）：runtime training/families 的 SPECS 直接
# 引用本表（依赖方向 runtime → studio，本模块是零依赖纯数据叶子）。studio/domain
# 仍不 import runtime（server 启动不依赖 runtime sys.path）。
# tests/test_model_family_gating.py 锁 SPECS 与本表的同一性（防再复制成镜像）。
FAMILY_CAPABILITIES: dict[str, frozenset] = {
    "anima": frozenset({
        "navit", "sra", "leap", "compile_blocks",
        "caption_tag_ops", "online_text", "masked_loss",
    }),
    "krea2": frozenset({"masked_loss", "text_cache"}),
}

# 族默认 overlay 的单一权威源（ModelSpec.config_defaults 直接引用本表，R3）。
# 只在字段缺失时 overlay，显式配置不改写。消费方:pydantic before-validator
# `_apply_family_config_defaults`（studio 与 trainer 现共用这一条加载路径，R1）。
FAMILY_CONFIG_DEFAULTS: dict[str, dict[str, Any]] = {
    "anima": {},
    "krea2": {
        "shuffle_caption": False,
        "keep_tokens": 0,
        "tag_dropout": 0.0,
        "text_encoder_cache": True,
        "attention_backend": "none",
        "timestep_sampling": "krea2_shift",
        "timestep_shift_resolution_aware": False,
        "sample_sampler_name": "euler",
        "sample_scheduler": "krea2_shift",
        "sample_infer_steps": 28,
        "sample_cfg_scale": 4.5,
    },
}

#: schema model_family Literal 的值域（顺序即 UI 下拉顺序）
MODEL_FAMILIES: tuple[str, ...] = tuple(FAMILY_CAPABILITIES)

# ─── 采样端白名单（多模型 P4-2，A13） ─────────────────────────────────────
# 单一权威源（首项 = 族默认值）：runtime SPECS[fam].sampling.samplers/schedulers
# 直接引用本表（R3，同能力矩阵的依赖方向说明）。
# 这两个字段是硬白名单——两族的 sample_image 都在入口对名单外的值 raise
# （anima: families/anima/sampling.py Comfy parity 校验；krea2:
# families/krea2/sampling.py 同款），所以跨族值在配置层就报错（fail-fast）。
FAMILY_SAMPLING: dict[str, dict[str, tuple[str, ...]]] = {
    "anima": {
        "samplers": ("er_sde", "dpmpp_3m_sde"),
        "schedulers": ("simple", "sgm_uniform"),
    },
    "krea2": {
        "samplers": ("euler",),
        "schedulers": ("krea2_shift",),
    },
}

#: Literal 收紧（#256）前就存在的族：sampler/scheduler 白名单外的存量值按
#: #256 迁移契约静默归并到族默认（grandfather，与 D12 npz 无键 / D13 LoRA
#: 无标记同款）。Literal 时代出生的族没有 legacy 语料——白名单外的 union 值
#: 一律报错（fail-fast + 可操作文案），不静默改写。
LEGACY_SAMPLING_FAMILIES: frozenset = frozenset({"anima"})

#: timestep_sampling 里按族门控的选项。其余选项是共享循环通用实现，全族可用。
#: krea2_shift 机制上任何族都能跑（loop 按 requires_token_counts 供
#: token_counts），但 mu 插值按 K2 校准——只做 UI 引导性隐藏，后端不设闸
#: （A1：同代码不限制）。第 3 个 resolution-aware 族复用该策略时加进元组即可。
TIMESTEP_SAMPLING_OPTION_FAMILIES: dict[str, tuple[str, ...]] = {
    "krea2_shift": ("krea2",),
}


def option_gates(option_families: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """把「选项 → 支持它的族」编译成 option 级 show_when 表达式映射。

    cap_gate 的 option 级版本：作者写时展开，三个 show_when 求值器零新文法。
    产物进 Field 元信息 `option_show_when`，前端按当前 model_family 过滤下拉
    选项。未出现在映射里的选项永远可见。
    """
    return {
        opt: "||".join(f"model_family=={f}" for f in sorted(fams))
        for opt, fams in option_families.items()
    }


def sampling_option_gates(kind: str) -> dict[str, str]:
    """从 FAMILY_SAMPLING 推导 samplers / schedulers 的 option 门控映射。

    全族都支持的值不设门（永远可见）；其余值按支持它的族展开表达式。
    """
    by_option: dict[str, list[str]] = {}
    for fam, spec in FAMILY_SAMPLING.items():
        for value in spec[kind]:
            by_option.setdefault(value, []).append(fam)
    all_families = set(FAMILY_SAMPLING)
    return option_gates({
        value: tuple(fams)
        for value, fams in by_option.items()
        if set(fams) != all_families
    })

#: 字段 → 所需能力位。驱动三层防线：show_when（作者写时经 cap_gate 展开）、
#: TrainingConfig validator、trainer bootstrap 校验。判定口径：字段值为
#: 真值/非零才算"启用"该能力（默认关闭的字段对任何族都合法）。
FIELD_CAPABILITY_REQUIREMENTS: dict[str, str] = {
    "navit_packing": "navit",
    "sra_enabled": "sra",
    "leap_enabled": "leap",
    "masked_loss": "masked_loss",
    "shuffle_caption": "caption_tag_ops",
    "keep_tokens": "caption_tag_ops",
    "tag_dropout": "caption_tag_ops",
}


def cap_gate(capability: str) -> str:
    """把能力位编译成 show_when 字段比较表达式（作者写时展开，运行时零新文法）。

    cap_gate("navit") → "model_family==anima"；未来第 3 族支持该能力时自动
    变为 "model_family==anima||model_family==foo"（只改 FAMILY_CAPABILITIES）。
    三个 show_when 求值器（前端 schema.ts / config_prune / YAML 预览）零改动。
    """
    fams = sorted(f for f, caps in FAMILY_CAPABILITIES.items() if capability in caps)
    if not fams:
        raise ValueError(f"没有任何模型族支持能力位 '{capability}'")
    return "||".join(f"model_family=={f}" for f in fams)


def capability_violations(model_family: str, values: dict) -> list[str]:
    """返回「已启用但当前族不支持」的字段名列表（validator / bootstrap 共用）。"""
    caps = FAMILY_CAPABILITIES.get(str(model_family))
    if caps is None:
        return []  # 未知族由 Literal / resolve_family 各自报错，这里不重复
    bad = []
    for field, cap in FIELD_CAPABILITY_REQUIREMENTS.items():
        if cap in caps:
            continue
        v = values.get(field)
        if isinstance(v, bool):
            active = v
        else:
            active = bool(v)  # int/float 非零即启用
        if active:
            bad.append(field)
    return bad


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
