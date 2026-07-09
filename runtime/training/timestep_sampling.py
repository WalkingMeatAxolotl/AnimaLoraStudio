"""Flow Matching timestep 采样：logit_normal / uniform / mode 等模式。

抽自原 runtime/anima_train.py L1817-1846（ADR 0003 PR-A）。

ADR 0003 把不同 mode 放同一文件（一文件多 fn）—— 每个 mode 是一行 sigmoid /
shift / clamp 数学，不引入 schema 字段（用 schema.timestep_sampling: str），
不需要 plugin subfolder。
"""

from __future__ import annotations

import torch


def sample_t(
    bs,
    device,
    mode: str = "logit_normal",
    shift: float = 3.0,
    mix_low_prob: float = 0.0,
    timestep_schedule_shift: float = 1.0,
) -> torch.Tensor:
    """采样 Flow Matching 时间步 t ∈ (0, 1)。

    mode:
      logit_normal       — SD3/Anima 默认，偏向中间 t；shift>1 推向高噪声端
      uniform            — 均匀采样，对细节端和结构端覆盖更均衡
      logit_normal_low   — logit-normal 反向 shift，偏向低噪声/细节端
      mode               — SD3 mode-distribution，集中在某个 sigma 附近
      mixed_uniform_low  — 每样本独立按 mix_low_prob 概率走 logit_normal_low，
                           其余走 uniform；细节端强化但保留 uniform 覆盖度
      mixed_uniform_logit— 同上但偏置端走 logit_normal（标准），适合不想偏低 t 的场景

    mix_low_prob — mixed_* mode 下走偏置端的样本比例（默认 0 = 全 uniform）
    timestep_schedule_shift — 采样完成后对 t 做的额外 σ schedule 偏移（默认 1.0 = 无 shift）；
                     公式 t' = (t * s) / (1 + (s - 1) * t)（SD3/FLUX shifted schedule，
                     Möbius 变换）；与 timestep_shift 不同：后者作用于 logit-normal 内部
                     的 sigmoid 后 u 值，前者作用于最终 t
    """
    mode = (mode or "logit_normal").lower()

    if mode == "uniform":
        t = torch.rand(bs, device=device).clamp(1e-4, 1 - 1e-4)
        return _apply_timestep_schedule_shift(t, timestep_schedule_shift)

    if mode in ("mixed_uniform_low", "mixed_uniform_logit"):
        p = max(0.0, min(1.0, float(mix_low_prob)))
        use_biased = torch.rand(bs, device=device) < p
        t_uniform = torch.rand(bs, device=device).clamp(1e-4, 1 - 1e-4)
        biased_mode = "logit_normal_low" if mode == "mixed_uniform_low" else "logit_normal"
        # 递归调用基础 mode 采样偏置端（不再传 mix_low_prob 避免循环；不在内部再做 schedule shift，
        # 留到最后统一应用）
        t_biased = sample_t(bs, device, mode=biased_mode, shift=shift, mix_low_prob=0.0, timestep_schedule_shift=1.0)
        t = torch.where(use_biased, t_biased, t_uniform)
        return _apply_timestep_schedule_shift(t, timestep_schedule_shift)

    u = torch.sigmoid(torch.randn(bs, device=device))

    if mode == "logit_normal_low":
        s = max(float(shift), 1e-4)
        u = (u * (1.0 / s)) / (1 + (1.0 / s - 1) * u)
        return _apply_timestep_schedule_shift(u.clamp(1e-4, 1 - 1e-4), timestep_schedule_shift)

    if mode == "mode":
        s = float(shift)
        u = 1 - u - s * (torch.cos(torch.pi * 0.5 * u) ** 2 - 1 + u)
        return _apply_timestep_schedule_shift(u.clamp(1e-4, 1 - 1e-4), timestep_schedule_shift)

    # logit_normal（默认）+ shift
    s = float(shift)
    u = (u * s) / (1 + (s - 1) * u)
    return _apply_timestep_schedule_shift(u.clamp(1e-4, 1 - 1e-4), timestep_schedule_shift)


def _apply_timestep_schedule_shift(t: torch.Tensor, timestep_schedule_shift: float) -> torch.Tensor:
    """对采样后的 t 做额外 σ schedule 偏移：t' = (t * s) / (1 + (s - 1) * t)。

    SD3/FLUX shifted schedule 的 Möbius 变换；s == 1.0 时恒等映射（默认行为不变）。
    s > 1 推向高噪声端；s < 1 偏低噪声端。
    """
    s = float(timestep_schedule_shift)
    if s == 1.0:
        return t
    return ((t * s) / (1 + (s - 1) * t)).clamp(1e-4, 1 - 1e-4)


def latent_token_counts(latents, patch_spatial: int = 2) -> list[int]:
    """每样本的 patch-token 数（``timestep_shift_resolution_aware`` 用）。

    - list/tuple（NaViT 逐图 latent，各 ``[.,C,T,h_i,w_i]``，形状可异构）→ 逐图计数；
    - 批量网格 tensor ``[B,C,T,H,W]``（ARB / 多分辨率，batch 内同尺寸）→ B 个等值计数。

    ``patch_spatial=2`` 与 ``CachedLatentDataset._fill_bucket_for_index`` 的
    token 口径一致（1 token = 16×16px）。
    """
    if isinstance(latents, (list, tuple)):
        return [
            int(l.shape[-2] // patch_spatial) * int(l.shape[-1] // patch_spatial)
            for l in latents
        ]
    n = int(latents.shape[-2] // patch_spatial) * int(latents.shape[-1] // patch_spatial)
    return [n] * int(latents.shape[0])


def apply_resolution_shift(t: torch.Tensor, token_counts, base_tokens: int) -> torch.Tensor:
    """SD3（arXiv 2403.03206 §5.3.2）分辨率相关 timestep shift 修正。

    对每样本施加 Möbius 偏移 t' = (t·s)/(1+(s−1)·t)，s_i = sqrt(n_i / n_base)。
    同一 t 下分辨率越高的图相邻像素相关性越强、等效 SNR 越高，需要更高的 t 才达到
    与基准档等效的破坏程度——所以大图（n_i > n_base）推向高噪声端、小图反之，
    n_i == n_base 恒等。

    Möbius 偏移按 s 乘法复合（shift(s1) ∘ shift(s2) == shift(s1·s2)），因此本修正
    与全局 timestep_shift / timestep_schedule_shift 正交：全局值仍是"基准档的校准值"，
    本函数只补分辨率相对基准档的差。
    """
    s = (
        torch.as_tensor(token_counts, dtype=t.dtype, device=t.device)
        .clamp_min(1)
        .div(float(max(1, int(base_tokens))))
        .sqrt()
    )
    return ((t * s) / (1 + (s - 1) * t)).clamp(1e-4, 1 - 1e-4)
