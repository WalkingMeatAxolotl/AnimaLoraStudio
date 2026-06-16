"""LeapAlign / FlowBP 轨迹自蒸馏训练步（去奖励模型版）。

源自 LeapAlign 论文 (arXiv 2604.15311v2) 的 two-step leap trajectory，以及 FlowBP
论文 (arXiv 2606.11075) 的代理轨迹设计空间。两者原版都靠奖励模型反传，这里统一去掉
奖励模型，把"最大化奖励"换成"沿代理轨迹积分出的 x̂0 逼近数据集真实 x0"的自蒸馏目标。

与原版 LeapAlign_Code/fastvideo/train_leapalign_flux.py 的关键区别：
- 原版必须先 online rollout 跑完整采样轨迹拿 x0（最吃显存）；这里数据集本就有真实
  x0，直接加噪到任意时刻，**无需 rollout**，天然适配 LoRA。
- 原版 loss = max(0, λ - reward(x0))，唯一信号来自奖励模型；这里 loss = MSE(x̂0, x0)，
  信号来自真实数据。
- 保留 LeapAlign 力学：two-step leap、latent connector、gradient discounting、
  traj-sim weighting；并把 FlowBP 的"代理轨迹设计空间"以四个 variant 暴露出来。

四个 variant（统一形式：解析构造轨迹点 + 沿轨迹积分 x̂0 + MSE(x̂0, x0) + traj-sim 尾）：
- original  : 两步跳 + straight-through connector + α 折扣（= LeapAlign 现状，K=2，1 雅可比）
- sparse    : K 点 Euler 重放，纯直接项求和（FlowBP-Sparse，零 connector / 零雅可比）
- bridge    : 两步跳 + Euler 重构 connector + α 折扣（FlowBP-Bridge，结构精确无 ST 偏差）
- lagrange  : 两段跳，每段三点 Lagrange/Simpson 积分（FlowBP-Lagrange，降单段积分误差）

自蒸馏特性注记：真值是解析直线插值点、无 rollout 噪声，connector 残差被釜底抽薪，
故 bridge/lagrange 相比 original 增益收窄；sparse 是唯一结构性差异（去 connector +
K 点稠密监督），代价是 K× 前向 + K× activation 显存。

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

    # 自蒸馏 loss + 轨迹相似度加权（共享尾）
    return _finalize_loss(
        x_hat_0, x0,
        x_hat_inter=x_hat_j, x_inter_real=x_j_real,
        traj_sim_weighting=traj_sim_weighting, traj_sim_min=traj_sim_min,
    )


def _finalize_loss(
    x_hat_0: torch.Tensor,
    x0: torch.Tensor,
    *,
    x_hat_inter: torch.Tensor | None = None,
    x_inter_real: torch.Tensor | None = None,
    traj_sim_weighting: bool = False,
    traj_sim_min: float = 0.1,
) -> torch.Tensor:
    """四变体共享的 loss 尾：自蒸馏 MSE(x̂0, x0) + 可选轨迹相似度加权（论文 Eq 12）。

    Args:
        x_hat_0      — 沿代理轨迹积分出的 x0 估计，shape (B,C,...)
        x0           — 真实 latent
        x_hat_inter  — 中间端点预测（用于 traj-sim 的 d_inter 项）；None 则只用 d_0
        x_inter_real — 中间端点真值（与 x_hat_inter 配对）
        traj_sim_weighting — 是否启用轨迹相似度加权
        traj_sim_min       — 相似度加权下限 τ（防近乎相同的对被过度放大）

    跳跃越贴近真实路径（残差越小）权重越高，抑制大跨度跳跃的离谱预测主导 loss。
    sparse 无中间端点（x_hat_inter=None），退化为仅按终点残差加权。

    量纲注记：original/bridge/lagrange 的 w_sim = 1/(d_inter + d_0) ≈ 1/(2d)，
    sparse 的 w_sim = 1/d_0 ≈ 1/d。同残差下 sparse 被放大约 2×。自适应优化器能吸收，
    但对照实验（original vs sparse 都开 traj_sim）的有效 loss magnitude 不可比，对比
    曲线时需注意。
    """
    loss_per_sample = (x_hat_0.float() - x0.float()).pow(2).mean(
        dim=tuple(range(1, x0.ndim))
    )

    if traj_sim_weighting:
        with torch.no_grad():
            d_0 = (x0.float() - x_hat_0.float()).abs().mean(
                dim=tuple(range(1, x0.ndim))
            ).clamp(min=traj_sim_min)
            if x_hat_inter is not None and x_inter_real is not None:
                d_inter = (x_inter_real.float() - x_hat_inter.float()).abs().mean(
                    dim=tuple(range(1, x0.ndim))
                ).clamp(min=traj_sim_min)
                w_sim = 1.0 / (d_inter + d_0)
            else:
                w_sim = 1.0 / d_0
        loss_per_sample = loss_per_sample * w_sim

    return loss_per_sample


def sample_activation_timesteps(
    bs: int,
    device,
    k: int,
    min_gap: float = 0.1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """per-sample 采样 K 个降序时刻 t_1 > t_2 > ... > t_K，均 ∈ (0,1)。

    用于 FlowBP-Sparse 的激活集：从噪声端 t_1 出发，沿 K 个支撑点 Euler 重放到数据端。
    参考论文（FlowBP §B）用 Dirichlet 划分保证支撑点沿 (0,1) 铺开、不塌缩；这里取等价
    的"分层抖动（stratified jitter）"实现：把 (0,1) 等分 K 格，每格内采一个 uniform
    点，天然保证（i）每点独立落在 (i/k, (i+1)/k)，覆盖整条轨迹；（ii）严格降序；
    （iii）相邻间隔 ≥ ~0（实际下界 ≈ 0，远好于旧贪心下压的塌缩风险）。

    min_gap 在 sparse 下语义为"总跨度提示"：若需保证最大跨度 t_1-t_K ≥ min_gap，
    调用方可在传入前约束（这里不强制，因为分层抖动已天然铺满 (0,1)）。

    返回 shape (B, K) 的张量，每行降序。
    """
    if k < 2:
        raise ValueError(f"sample_activation_timesteps 需要 k>=2，收到 k={k}")

    # 分层抖动：第 i 格 (i/k, (i+1)/k) 内采 uniform 点
    # idx = 0..K-1 对应数据端→噪声端，采完再翻成降序（噪声端在前）
    idx = torch.arange(k, device=device, dtype=dtype)
    jitter = torch.rand(bs, k, device=device, dtype=dtype)
    pts = (idx + jitter) / k  # 每格内 uniform，shape (B, K)，升序
    pts, _ = torch.sort(pts, dim=1, descending=True)  # 降序：t_1 > ... > t_K

    # 边界裁剪到开区间 (0,1)：分层抖动天然落在 (0,1) 内，仅兜底数值边界
    eps = 1e-3
    pts = pts.clamp(min=eps, max=1.0 - eps)
    return pts


def sparse_training_step(
    model,
    x0: torch.Tensor,
    noise: torch.Tensor,
    cross: torch.Tensor,
    pad_mask: torch.Tensor,
    t_steps: torch.Tensor,
    *,
    traj_sim_weighting: bool = False,
    traj_sim_min: float = 0.1,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """FlowBP-Sparse 自蒸馏：K 点 Euler 重放，纯直接项求和（零 connector / 零雅可比）。

    所有轨迹点用解析直线插值 x_{t_i}=(1-t_i)x0+t_i·noise（天然 detach），梯度只经速度
    v_θ(x_{t_i}, t_i) 流回 θ。Euler 望远镜求和：

        x̂0 = x_{t_1} - Σ_{i=1..K} (t_i - t_{i+1}) · v_θ(x_{t_i}, t_i)，    t_{K+1}:=0

    当 v_θ ≡ 真实速度 (noise-x0) 时，每段 (t_i-t_{i+1})·v = x_{t_i}-x_{t_{i+1}}，
    级数完美望远镜回到 x_{t_1}-(x_{t_1}-x0)=x0。监督是 K 个时刻速度的加权积分一致性，
    是原始 flow matching 单点 MSE 的多点稠密推广。

    Args:
        t_steps — per-sample 降序时刻 (B, K)，由 sample_activation_timesteps 生成
    """
    bs, k = t_steps.shape
    view = (-1, *([1] * (x0.ndim - 1)))

    # 起点 x_{t_1}（解析、detach）
    t1 = t_steps[:, 0].reshape(*view)
    x_hat_0 = (1.0 - t1) * x0 + t1 * noise

    # 逐支撑点：解析构造带噪点（detach），前向求速度（带梯度），按 Euler 步长累减
    for i in range(k):
        t_i = t_steps[:, i]
        t_i_b = t_i.reshape(*view)
        x_i = (1.0 - t_i_b) * x0 + t_i_b * noise  # 解析点，对 θ 无梯度
        v_i = forward_with_optional_checkpoint(
            model, x_i, t_i.reshape(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
        )
        # 步长 h_i = t_i - t_{i+1}，末段 t_{K+1}:=0（h_i = t_i）
        if i + 1 < k:
            h_i = (t_i - t_steps[:, i + 1]).reshape(*view)
        else:
            h_i = t_i_b
        x_hat_0 = x_hat_0 - h_i * v_i

    # sparse 无中间端点，traj-sim 仅按终点残差加权
    return _finalize_loss(
        x_hat_0, x0,
        traj_sim_weighting=traj_sim_weighting, traj_sim_min=traj_sim_min,
    )


def bridge_training_step(
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
    """FlowBP-Bridge 自蒸馏：两步跳 + Euler 重构 connector（无 straight-through 偏差）。

    与 original 唯一区别在 connector：original 把 x_j 的前向值换成真值 x_j_real
    （straight-through，前向数值=真值、反向梯度流回 v_k）；bridge 直接用 Euler 重构值
    x̂_j 当第二跳输入，结构上精确无偏（前向数值=自身轨迹点），梯度通过 α 缩放的单雅可比
    ∂x̂_j/∂v_k 跨段传播。α=0 退化为只训第二跳，α=1 完整单雅可比。
    """
    k = t_k.view(-1, *([1] * (x0.ndim - 1)))
    j = t_j.view(-1, *([1] * (x0.ndim - 1)))

    x_k = (1.0 - k) * x0 + k * noise
    x_j_real = (1.0 - j) * x0 + j * noise

    # 第一跳（带梯度）
    v_k = forward_with_optional_checkpoint(
        model, x_k, t_k.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    x_hat_j = x_k - (k - j) * v_k

    # Euler 重构 connector：直接用 x̂_j（不替换为真值），梯度按 α 折扣
    if nested_grad_coe <= 0.0:
        x_j_in = x_hat_j.detach()
    elif nested_grad_coe >= 1.0:
        x_j_in = x_hat_j
    else:
        x_j_in = nested_grad_coe * x_hat_j + (1.0 - nested_grad_coe) * x_hat_j.detach()

    # 第二跳（带梯度）：从重构端点出发
    v_j = forward_with_optional_checkpoint(
        model, x_j_in, t_j.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    x_hat_0 = x_j_in - j * v_j

    return _finalize_loss(
        x_hat_0, x0,
        x_hat_inter=x_hat_j, x_inter_real=x_j_real,
        traj_sim_weighting=traj_sim_weighting, traj_sim_min=traj_sim_min,
    )


def lagrange_training_step(
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
    """FlowBP-Lagrange 自蒸馏：保留两段跳拓扑，每段用三点 Lagrange 积分（论文 §A.2）。

    三支撑点 Lagrange 插值积分，等距时退化为 Simpson 规则，权重 1/6·[1, 4, 1]：
    论文 §A.2 称 "Simpson-like positive weights"。比 original/bridge 的单点 Euler
    （误差 O(Δt²)）高两阶（Simpson 误差 O(Δt⁵)）。

    每段积分形式（段 [b,a]，a>b，从 a 跳到 b，三点 a > m > b，m=(a+b)/2）：

        ∫_b^a v dt ≈ (a-b)/6 · [v(x_a) + 4·v(x_m) + v(x_b)]

    三点都需速度：端点 x_a、中点 x_m、另端点 x_b。**梯度约束**（论文 Eq 4/24，摘要
    "at most one Jacobian factor"）：非 bridge-anchor 的 active support 一律对 latent
    输入取 stop-gradient——v_θ(sg(x_i), σ_i)，梯度只经 θ 不回传到 x_i。本自蒸馏实现
    没有 rollout，每个时刻的 cached latent 就是解析直线插值真值，故三点输入都用解析真值
    （天然 detach、不依赖 θ）：
      - 端点 x_a、中点 x_m=(1-m)x0+m·noise；
      - 另端点 x_b 的真值在自蒸馏下**已知**：第一段段终点 b=j 即 x_j_real，第二段段终点
        b=0 即 x0 本身（在 t=0 评估）。
    这与论文取 cached-latent 速度一致，避免旧版用 Euler 预测点 x̂_b=x_a-(a-b)·v(x_a)
    带入的未折扣嵌套雅可比（∂v(x̂_b)/∂v(x_a)），后者会绕过 α 违反单雅可比约束。
    唯一保留 latent 雅可比的是第二段起点 x_j（bridge anchor，α 缩放的 connector）。

    代价：每段 3 次前向（端点 + 中点 + 另端点），两段共 6 次前向 ≈ 6× 算力 + 显存，
    是四 variant 中最重者。两段串联：第一段 k→j，connector（straight-through，
    同 original）+ α 折扣后第二段 j→0。
    """
    k = t_k.view(-1, *([1] * (x0.ndim - 1)))
    j = t_j.view(-1, *([1] * (x0.ndim - 1)))
    m1 = (k + j) * 0.5  # 第一段中点
    t_m1 = (t_k + t_j) * 0.5

    x_k = (1.0 - k) * x0 + k * noise
    x_j_real = (1.0 - j) * x0 + j * noise
    x_m1_real = (1.0 - m1) * x0 + m1 * noise  # 中点解析真值（detach）

    # ── 第一段三点 Lagrange / Simpson 积分：x_k, x_m1, x_j_real ──
    v_k = forward_with_optional_checkpoint(
        model, x_k, t_k.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    v_m1 = forward_with_optional_checkpoint(
        model, x_m1_real, t_m1.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    # 段终点（j）速度：自蒸馏下真值已知 = x_j_real（解析、detach），非 Euler 预测点，
    # 避免未折扣嵌套雅可比（论文 Eq 4：非 anchor support 取 v_θ(sg(x_i))）
    v_j_seg1 = forward_with_optional_checkpoint(
        model, x_j_real, t_j.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    v_seg1 = (v_k + 4.0 * v_m1 + v_j_seg1) / 6.0  # Simpson 权重 1:4:1
    x_hat_j = x_k - (k - j) * v_seg1

    # connector（straight-through，同 original）+ α 折扣
    x_j = x_hat_j + (x_j_real - x_hat_j).detach()
    if nested_grad_coe <= 0.0:
        x_j_in = x_j.detach()
    elif nested_grad_coe >= 1.0:
        x_j_in = x_j
    else:
        x_j_in = nested_grad_coe * x_j + (1.0 - nested_grad_coe) * x_j.detach()

    # ── 第二段三点 Lagrange / Simpson 积分：x_j, x_m2, x0 ──
    m2 = j * 0.5  # 第二段中点（j→0 的中点是 j/2）
    t_m2 = t_j * 0.5
    x_m2_real = (1.0 - m2) * x0 + m2 * noise
    v_j = forward_with_optional_checkpoint(
        model, x_j_in, t_j.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    v_m2 = forward_with_optional_checkpoint(
        model, x_m2_real, t_m2.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    # 段终点（0）速度：自蒸馏下真值已知 = x0（在 t=0 评估），非 Euler 预测点
    t_0 = torch.zeros_like(t_j)
    v_0_seg2 = forward_with_optional_checkpoint(
        model, x0, t_0.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    v_seg2 = (v_j + 4.0 * v_m2 + v_0_seg2) / 6.0  # Simpson 权重 1:4:1
    x_hat_0 = x_j_in - j * v_seg2

    return _finalize_loss(
        x_hat_0, x0,
        x_hat_inter=x_hat_j, x_inter_real=x_j_real,
        traj_sim_weighting=traj_sim_weighting, traj_sim_min=traj_sim_min,
    )
