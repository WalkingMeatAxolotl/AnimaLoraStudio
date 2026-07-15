"""Timestep 采样器 plugin registry（参考 ADR 0003 PR-C adapter registry 模式）。

`build_timestep_sampler(args, total_steps)` 按 args 派发到具体采样器：
- 默认走 baseline（包装 sample_t 的 4 种 mode）
- args.infonoise_enabled == True 时走 InfoNoise 自适应采样
- args.timestep_sampling == "krea2_shift" 时走 Krea2 resolution-aware 采样

加新采样器：
1. 写 timestep_samplers/{name}.py 含 `build(args, total_steps) -> TimestepSamplerProtocol`
2. 本文件 BUILDERS 字典 / build_timestep_sampler 派发逻辑加一行
3. studio/schema.py 加对应启用字段
4. 完。phases/optimizer.py / loop.py / context.py 0 改动。
"""

from __future__ import annotations

from training.timestep_samplers import baseline, infonoise, krea2_shift
from training.timestep_samplers.protocol import TimestepSamplerProtocol

__all__ = [
    "TimestepSamplerProtocol",
    "BUILDERS",
    "build_timestep_sampler",
]


# 单一 truth source：所有采样器 build 工厂的注册表
# baseline 是兜底（任何 args 都能构造），其他都是 adaptive（按 args 字段判定启用）
BUILDERS: dict[str, callable] = {
    "baseline": baseline.build,
    "infonoise": infonoise.build,
    "krea2_shift": krea2_shift.build,
}


def build_timestep_sampler(args, total_steps) -> TimestepSamplerProtocol:
    """按 args 派发到对应采样器；总是返回非 None 实例（baseline 兜底）。

    Krea2 dynamic shift 与 InfoNoise 都定义 timestep 分布，不能同时启用。
    """
    mode = str(getattr(args, "timestep_sampling", "logit_normal") or "logit_normal").lower()
    if mode == "krea2_shift":
        if getattr(args, "infonoise_enabled", False):
            raise ValueError("krea2_shift 与 infonoise_enabled 不能同时启用")
        return BUILDERS["krea2_shift"](args, total_steps)
    if getattr(args, "infonoise_enabled", False):
        return BUILDERS["infonoise"](args, total_steps)
    return BUILDERS["baseline"](args, total_steps)
