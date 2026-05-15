"""InfoNoise 自适应时间步采样器。

基于 I-MMSE 恒等式 d/dσ H[x0|xσ] = mmse(σ)/σ³，动态估计各噪声区间的信息量，
把采样概率集中在"信息窗口"内，跳过极高/极低噪声的低效区间。

在 Flow Matching 的 t ∈ (0,1) 空间工作，内部把 t 映射到 σ = t/(1-t)
后在 log-σ 空间均匀分 bin，以保持与原始论文的一致性。
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class InfoNoiseScheduler:
    """InfoNoise 自适应时间步采样器。"""

    def __init__(
        self,
        K: int = 64,
        t_min: float = 0.001,
        t_max: float = 0.999,
        N_warm: int = 5000,
        M: int = 100,
        B: int = 256,
        beta: float = 0.9,
        n_gate: int = 3,
        p_onset: float = 0.002,
        N_min: int = 50,
        baseline_shift: float = 3.0,
    ):
        import numpy as np
        from collections import deque

        self.K = K
        self.N_warm = N_warm
        self.M = M
        self.B = B
        self.beta = beta
        self.n_gate = n_gate
        self.p_onset = p_onset
        self.N_min = N_min
        self.baseline_shift = baseline_shift
        self._internal_step = 0

        sigma_min = t_min / (1.0 - t_min)
        sigma_max = t_max / (1.0 - t_max)
        log_edges = np.linspace(np.log(sigma_min), np.log(sigma_max), K + 1)
        self._log_sigma_edges = log_edges
        self._delta_log_sigma = float(log_edges[1] - log_edges[0])
        self._sigma_centers = np.exp(0.5 * (log_edges[:-1] + log_edges[1:]))

        self._fifo = [deque(maxlen=B) for _ in range(K)]
        self._mse_ema = np.zeros(K, dtype=np.float64)
        self._n_count = np.zeros(K, dtype=np.int32)
        self._cdf_values: Optional["np.ndarray"] = None

    def sample(self, bs: int, device) -> torch.Tensor:
        """采样 t ∈ (0,1)。热身期用 logit-normal baseline，之后用自适应 CDF。"""
        if self._cdf_values is None:
            return self._sample_baseline(bs, device)
        import numpy as np
        u = torch.rand(bs).numpy()
        log_sigma = np.interp(u, self._cdf_values, self._log_sigma_edges)
        sigma = np.exp(log_sigma)
        t = sigma / (1.0 + sigma)
        return torch.tensor(t, device=device, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)

    def _sample_baseline(self, bs: int, device) -> torch.Tensor:
        u = torch.sigmoid(torch.randn(bs, device=device))
        s = self.baseline_shift
        t = (u * s) / (1 + (s - 1) * u)
        return t.clamp(1e-4, 1 - 1e-4)

    def record(self, t: torch.Tensor, raw_mse: torch.Tensor):
        """记录 per-sample 原始 MSE（不含任何 loss weight）到对应 bin。"""
        import numpy as np
        t_np = t.detach().cpu().float().numpy()
        mse_np = raw_mse.detach().cpu().float().numpy()
        sigma_np = t_np / np.clip(1.0 - t_np, 1e-8, None)
        log_sigma_np = np.log(np.clip(sigma_np, 1e-8, None))
        edges_inner = self._log_sigma_edges[1:-1]
        for i in range(len(t_np)):
            k = int(np.searchsorted(edges_inner, log_sigma_np[i]))
            self._fifo[k].append(float(mse_np[i]))
            self._n_count[k] = min(self._n_count[k] + 1, self.B)
        self._internal_step += 1

    def maybe_refresh(self, global_step: int):
        """条件满足时刷新 schedule（每 M 步、热身结束后、每 bin 有足够样本）。"""
        if self._internal_step < self.N_warm:
            return
        if global_step % self.M != 0:
            return
        import numpy as np
        if int(np.min(self._n_count)) < self.N_min:
            return
        self._refresh()

    def _refresh(self):
        import numpy as np

        # Step A+B: 平均 loss + EMA 平滑
        l_bar = np.array([
            float(np.mean(list(buf))) if buf else 0.0
            for buf in self._fifo
        ])
        self._mse_ema = (1.0 - self.beta) * self._mse_ema + self.beta * l_bar

        # Step C: entropy rate r̂_k = mse_k / σ_k³
        r_hat = self._mse_ema / (self._sigma_centers ** 3 + 1e-30)

        # Step D: 找 gate pivot c（从低 σ 向高 σ 扫，取第一个超过 p_onset 的前一个 bin）
        r_max = float(r_hat.max())
        if r_max < 1e-30:
            return
        r_norm = r_hat / r_max
        above = r_norm >= self.p_onset
        if not any(above):
            return
        first_above = int(above.argmax())
        c = float(self._sigma_centers[max(0, first_above - 1)])

        # Step E: gate g(σ) = σⁿ / (σⁿ + cⁿ)
        sn = self._sigma_centers ** self.n_gate
        cn = c ** self.n_gate
        r_tilde = r_hat * sn / (sn + cn + 1e-30)

        # Step F+G: 归一化 + 构建 CDF（log-σ 空间梯形积分，bins 等宽所以直接求和）
        q = r_tilde.clip(0.0)
        Z = float(q.sum() * self._delta_log_sigma)
        if Z < 1e-30:
            return
        q_norm = q / Z
        cdf = np.concatenate([[0.0], np.cumsum(q_norm * self._delta_log_sigma)])
        cdf[-1] = 1.0
        self._cdf_values = cdf.clip(0.0, 1.0)


def build_info_noise(args, total_steps: Optional[int]) -> Optional[InfoNoiseScheduler]:
    """按 args 构建 InfoNoiseScheduler；未启用时返回 None。"""
    if not getattr(args, "infonoise_enabled", False):
        return None

    n_warm_cfg = int(getattr(args, "infonoise_N_warm", 0) or 0)
    if n_warm_cfg <= 0:
        n_warm_cfg = max(200, int((total_steps or 5000) * 0.2))
        logger.info(f"InfoNoise N_warm 自动设置为 {n_warm_cfg} 步（总步数 {total_steps} × 20%）")

    scheduler = InfoNoiseScheduler(
        K=int(getattr(args, "infonoise_K", 64) or 64),
        N_warm=n_warm_cfg,
        M=int(getattr(args, "infonoise_M", 100) or 100),
        B=int(getattr(args, "infonoise_B", 256) or 256),
        beta=float(getattr(args, "infonoise_beta", 0.9) or 0.9),
        N_min=int(getattr(args, "infonoise_N_min", 50) or 50),
        baseline_shift=float(getattr(args, "timestep_shift", 3.0) or 3.0),
    )
    logger.info(
        f"InfoNoise 已启用：K={scheduler.K}, N_warm={scheduler.N_warm}, "
        f"M={scheduler.M}, B={scheduler.B}, beta={scheduler.beta}"
    )
    return scheduler
