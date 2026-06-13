"""Inference sampler plugin registry（ADR 0003 PR-C）。

加新 sampler（Euler / Heun / DPM++2M / DDIM / unipc）的步骤：
1. 写 training/inference_samplers/{variant}.py 含 `sample(denoise_fn, x, sigmas, **kw)`
2. 本文件 BUILDERS 字典加一行
3. （可选）改 studio/schema.py 的 sample_sampler_name 加 Literal 校验

详见 ADR 0003 "Case 8: Euler / DPM++2M"。

注：sample_sampler_name 在 schema 里是 str 不是 Literal，未注册名会回退到
sampling.py 里 inline 的简化 Euler ODE 路径（保持原 main() 行为）。
"""

from __future__ import annotations

from typing import Callable

from training.inference_samplers import dpmpp_3m_sde, er_sde

__all__ = ["BUILDERS", "build_inference_sampler"]


BUILDERS: dict[str, Callable] = {
    "er_sde": er_sde.sample,
    "dpmpp_3m_sde": dpmpp_3m_sde.sample,
}


def build_inference_sampler(name: str) -> Callable:
    """按 name 取 sampler fn；未注册返回 None（caller 应走 fallback）。"""
    return BUILDERS.get(str(name).lower().strip())
