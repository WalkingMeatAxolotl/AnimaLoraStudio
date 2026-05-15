"""InfoNoise 自适应时间步采样器。

基于 I-MMSE 恒等式 d/dσ H[x0|xσ] = mmse(σ)/σ³，动态估计各噪声区间的信息量，
把采样概率集中在"信息窗口"内，跳过极高/极低噪声的低效区间。

在 Flow Matching 的 t ∈ (0,1) 空间工作，内部把 t 映射到 σ = t/(1-t)
后在 log-σ 空间均匀分 bin，以保持与原始论文的一致性。

参考论文：arxiv 2602.18647 "Information-Guided Noise Allocation for Efficient
Diffusion Training"，Algorithm 1 第 11 行：
    mse^k ← (1-β)·mse^k + β·ℓ̄_k
**β 乘的是新值**（responsiveness 强、平滑弱）；FIFO buffer 已做了一轮平均，
EMA 是第二层平滑，β=0.9 即"新值占 90% 权重"是论文设计意图，不是 bug。
**任何"按主观 EMA 直觉"翻转公式的 PR 都会改错算法，请先 verify 论文。**
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
        baseline_mode: str = "logit_normal",
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
        self.baseline_mode = baseline_mode
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
        # 可观测性（P1-1 反方/正方都标了"InfoNoise 静默退化是 trust killer"）：
        # last_refresh_status 暴露 _refresh 上一次的退出原因；refresh_attempts 计退化次数；
        # warned_cold_start 防 logger 刷屏。
        self._last_refresh_status: str = "not_refreshed_yet"
        self._refresh_attempts: int = 0
        self._refresh_degraded_count: int = 0
        self._warned_cold_start: bool = False

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
        # P1-3：warmup / CDF 未就绪时沿用用户 schema 选的 timestep_sampling，
        # 而不是写死 logit_normal_shift。复用 training.timestep_sampling.sample_t
        # 避免分叉两份分布逻辑。
        from training.timestep_sampling import sample_t
        return sample_t(bs, device, mode=self.baseline_mode, shift=self.baseline_shift)

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
            self._last_refresh_status = "skipped_bins_not_full"
            return
        self._refresh()
        # 冷启动退化 trip wire（P1-1）：跑完一次完整 _refresh 但 CDF 仍未就绪
        # → InfoNoise 静默走 baseline，必须告知用户避免"花算力没效果"。
        if self._cdf_values is None and not self._warned_cold_start:
            logger.warning(
                "InfoNoise: warmup 已过且各 bin 样本充足，但首次 schedule "
                "刷新仍未产生有效 CDF（原因：%s）。当前继续使用 logit-normal "
                "baseline 采样；若该状态持续到训练后期，说明你的 loss 分布在 "
                "log-σ 空间过于均匀（如已收敛模型），InfoNoise 无加速效果，"
                "建议关闭 infonoise_enabled。",
                self._last_refresh_status,
            )
            self._warned_cold_start = True

    def status(self) -> dict:
        """暴露当前 scheduler 状态（给 wandb 监控 / debug 用）。"""
        return {
            "kind": "infonoise",
            "cdf_ready": self._cdf_values is not None,
            "last_refresh_status": self._last_refresh_status,
            "refresh_attempts": self._refresh_attempts,
            "refresh_degraded_count": self._refresh_degraded_count,
            "internal_step": self._internal_step,
        }

    def _refresh(self):
        import numpy as np

        self._refresh_attempts += 1

        # Step A+B: 平均 loss + EMA 平滑
        # 论文 Algorithm 1 第 11 行：mse^k ← (1-β)·mse^k + β·ℓ̄_k
        # β 乘新值，beta=0.9 即"新值占 90% 权重"（responsiveness 强、第二层轻平滑）。
        # 顶部 docstring 有更完整说明，不要按主观 EMA 直觉翻转。
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
            self._last_refresh_status = "mse_collapsed"
            self._refresh_degraded_count += 1
            return
        r_norm = r_hat / r_max
        above = r_norm >= self.p_onset
        if not any(above):
            self._last_refresh_status = "gate_empty"
            self._refresh_degraded_count += 1
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
            self._last_refresh_status = "normalizer_too_small"
            self._refresh_degraded_count += 1
            return
        q_norm = q / Z
        cdf = np.concatenate([[0.0], np.cumsum(q_norm * self._delta_log_sigma)])
        cdf[-1] = 1.0
        self._cdf_values = cdf.clip(0.0, 1.0)
        self._last_refresh_status = "ok"


def build(args, total_steps: Optional[int]) -> InfoNoiseScheduler:
    """按 args 构建 InfoNoiseScheduler。

    调用方应已经判定 args.infonoise_enabled == True；这里不再重复判（让
    timestep_samplers.__init__.build_timestep_sampler 统一做派发判定）。
    """
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
        baseline_mode=str(getattr(args, "timestep_sampling", "logit_normal") or "logit_normal"),
    )
    logger.info(
        f"InfoNoise 已启用：K={scheduler.K}, N_warm={scheduler.N_warm}, "
        f"M={scheduler.M}, B={scheduler.B}, beta={scheduler.beta}, "
        f"baseline={scheduler.baseline_mode}(shift={scheduler.baseline_shift})"
    )
    return scheduler
