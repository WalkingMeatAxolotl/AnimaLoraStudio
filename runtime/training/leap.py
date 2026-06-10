"""LeapAlign 两步跳跃自蒸馏训练步（去奖励模型版）。

源自 LeapAlign 论文 (arXiv 2604.15311v2) 的 two-step leap trajectory，但去掉了
奖励模型，把"最大化奖励"换成"两步跳跃预测的 x0 逼近数据集真实 x0"的自蒸馏目标。

与原版 LeapAlign_Code/fastvideo/train_leapalign_flux.py 的关键区别：
- 原版必须先 online rollout 跑完整采样轨迹拿 x0（最吃显存）；这里数据集本就有真实
  x0，直接加噪到任意时刻，**无需 rollout**，天然适配 LoRA。
- 原版 loss = max(0, λ - reward(x0))，唯一信号来自奖励模型；这里 loss = MSE(x̂0, x0)，
  信号来自真实数据。
- 保留：two-step leap、latent connector、gradient discounting、traj-sim weighting。

约定 rectified flow（与 training/loop.py 一致）：
- t=0 为数据端，t=1 为噪声端
- x_t = (1-t)·x0 + t·x1，velocity v = x1 - x0
- 一步跳跃（从时刻 a 跳到时刻 b，a>b）：x̂_b = x_a - (a-b)·v_θ(x_a, a)
"""

from __future__ import annotations

import torch

from training.model_loading import forward_with_optional_checkpoint


def sample_two_timesteps(
    bs: int,
    device,
    min_gap: float = 0.1,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """per-sample 采样两个时刻 (k, j)，保证 k > j 且间隔 ≥ min_gap，均 ∈ (0,1)。

    k 偏噪声端（大 t），j 偏数据端（小 t）。先在 (0,1) 采两点排序成 (hi, lo)，
    间隔不足时把 hi 往噪声端推、lo 往数据端拉，再 clamp 回开区间。
    """
    a = torch.rand(bs, device=device, dtype=dtype)
    b = torch.rand(bs, device=device, dtype=dtype)
    k = torch.maximum(a, b)
    j = torch.minimum(a, b)

    # 间隔不足 min_gap 时撑开：各取一半缺口往两端推
    deficit = (min_gap - (k - j)).clamp(min=0.0) * 0.5
    k = k + deficit
    j = j - deficit

    eps = 1e-3
    k = k.clamp(min=eps + min_gap, max=1.0 - eps)
    # j 上界是 per-sample 的 (k - eps)，clamp 不接受张量上界，用 minimum + 标量下界
    j = torch.minimum(j, k - eps).clamp(min=eps)
    return k, j


def leap_training_step(
    model,
    x0: torch.Tensor,
    noise: torch.Tensor,
    cross: torch.Tensor,
    pad_mask: torch.Tensor,
    t_k: torch.Tensor,
    t_j: torch.Tensor,
    *,
    nested_grad_coe: float = 0.3,
    traj_sim_weighting: bool = False,
    traj_sim_min: float = 0.1,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """两步跳跃自蒸馏，返回 per-sample loss (B,)（未 reduction，未乘外部样本权重）。

    Args:
        model        — Anima transformer（接受 (B,) per-sample timestep）
        x0           — 真实 latent，shape (B,C,T,H,W)
        noise        — 噪声 x1，与 x0 同 shape（由 make_noise 生成）
        cross        — 文本条件 embedding
        pad_mask     — padding mask
        t_k, t_j     — per-sample 时刻 (B,)，t_k > t_j
        nested_grad_coe   — 梯度折扣 α（论文 Eq 9）：缩放嵌套梯度，0=砍掉/1=不折扣
        traj_sim_weighting — 是否启用轨迹相似度加权（论文 Eq 12）
        traj_sim_min       — 相似度加权下限 τ（防近乎相同的对被过度放大）
        use_checkpoint     — 模型前向是否走梯度检查点
    """
    # 广播到 latent 维度 (B,1,1,1,1)
    k = t_k.view(-1, *([1] * (x0.ndim - 1)))
    j = t_j.view(-1, *([1] * (x0.ndim - 1)))

    # 真实带噪 latent（无需 rollout）
    x_k = (1.0 - k) * x0 + k * noise
    x_j_real = (1.0 - j) * x0 + j * noise

    # ── 第一跳（带梯度）：x_k --v_k--> x̂_{j|k} ──
    v_k = forward_with_optional_checkpoint(
        model, x_k, t_k.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    x_hat_j = x_k - (k - j) * v_k

    # ── latent connector（论文 Eq 6）：前向数值=真值，反向梯度流回 v_k ──
    x_j = x_hat_j + (x_j_real - x_hat_j).detach()

    # ── 梯度折扣（论文 Eq 9）：缩放第二跳对 x_j 的嵌套梯度为 α 倍 ──
    if nested_grad_coe <= 0.0:
        x_j_in = x_j.detach()
    elif nested_grad_coe >= 1.0:
        x_j_in = x_j
    else:
        x_j_in = nested_grad_coe * x_j + (1.0 - nested_grad_coe) * x_j.detach()

    # ── 第二跳（带梯度）：x_j --v_j--> x̂_{0|j} ──
    v_j = forward_with_optional_checkpoint(
        model, x_j_in, t_j.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    x_hat_0 = x_j - j * v_j

    # ── 自蒸馏 loss：两步跳跃预测的 x̂0 逼近真实 x0（取代奖励）──
    loss_per_sample = (x_hat_0.float() - x0.float()).pow(2).mean(
        dim=tuple(range(1, x0.ndim))
    )

    # ── 轨迹相似度加权（论文 Eq 12）：跳跃越贴近真实路径，权重越高 ──
    if traj_sim_weighting:
        with torch.no_grad():
            d_j = (x_j_real.float() - x_hat_j.float()).abs().mean(
                dim=tuple(range(1, x0.ndim))
            ).clamp(min=traj_sim_min)
            d_0 = (x0.float() - x_hat_0.float()).abs().mean(
                dim=tuple(range(1, x0.ndim))
            ).clamp(min=traj_sim_min)
            w_sim = 1.0 / (d_j + d_0)
        loss_per_sample = loss_per_sample * w_sim

    return loss_per_sample
