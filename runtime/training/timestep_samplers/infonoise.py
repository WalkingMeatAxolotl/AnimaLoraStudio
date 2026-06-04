"""InfoNoise 自适应时间步采样器。

基于 I-MMSE 恒等式动态估计各噪声区间的信息量，把采样概率集中到信息窗口
（"既不太简单也不太难"的 σ 区段）。在 Flow Matching t ∈ (0,1) 空间工作，
内部映射到 σ = t/(1-t) 后在 log-σ 空间均匀分 K bin。

参考论文：arxiv 2602.18647 "Information-Guided Noise Allocation for Efficient
Diffusion Training"（§3.1 + Algorithm 1）。

关键实现选择（与论文对齐 + 偏离说明）：

- **EMA 平滑方向**：m̂_k ← (1-β)·m̂_k + β·ℓ̄_k，β 乘新值。β=0.9 即新值占
  90% 权重；FIFO B=256 已做底层平滑，EMA 是二次轻平滑。论文 §3.1 描述
  "smoothed binwise estimate" 但未给字面公式；该方向由
  test_ema_responsiveness_codifies_design_choice codify。

- **Entropy rate r̂_k = mse_k / σ_k²**（log-σ 空间，论文附录 B.2 Eq 61）。
  与 Δlog σ 求和自洽。注：论文 §3 VE 通道给的是 mse/σ³ (Eq 59)，FM-OT 路径
  字面给的是 (1-u)/u³ (Eq 64)；σ² log-σ 形式在两种路径下统一，且消解
  低 σ 端 1/σ³ universal tail 主导归一化的问题（§B.6）。

- **Gate pivot c**：默认 c=0.15（论文 §5 CIFAR 报告值）。设 gate_pivot_c=0
  走 dynamic Eq 87 实现（从高 σ 向低 σ 扫，找最后一个 r_norm ≥ p_onset
  的 bin）。dynamic 在 mmse 单调降形状下会退化，c=0.15 跨形状鲁棒；
  详见 tools/infonoise_e2e_verify.py 报告。

- **Warmup 单位**：N_warm 按 optimizer step 计（在 maybe_refresh 内 gate），
  与 build 里 total_steps × 20% 默认值同维度。grad_accum>1 不影响 warmup
  时长。_internal_step 是 record 累计（micro-batch 粒度），仅诊断用。
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np
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
        gate_pivot_c: float = 0.15,
        baseline_shift: float = 3.0,
        baseline_mode: str = "logit_normal",
        baseline_mix_low_prob: float = 0.0,
        baseline_timestep_schedule_shift: float = 1.0,
    ):
        if not 0.0 < p_onset < 1.0:
            raise ValueError(f"p_onset 必须 ∈ (0,1)，得到 {p_onset}")
        self.K = K
        self.N_warm = N_warm
        self.M = M
        self.B = B
        self.beta = beta
        self.n_gate = n_gate
        self.p_onset = p_onset
        self.N_min = N_min
        self.gate_pivot_c = gate_pivot_c
        self.baseline_shift = baseline_shift
        self.baseline_mode = baseline_mode
        self.baseline_mix_low_prob = baseline_mix_low_prob
        self.baseline_timestep_schedule_shift = baseline_timestep_schedule_shift
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
        self._cdf_values: Optional[np.ndarray] = None
        # 可观测性：last_refresh_status 暴露 _refresh 上一次的退出原因；
        # refresh_attempts 计退化次数；warned_cold_start 防 logger 刷屏。
        self._last_refresh_status: str = "not_refreshed_yet"
        self._refresh_attempts: int = 0
        self._refresh_degraded_count: int = 0
        self._warned_cold_start: bool = False

    def sample(self, bs: int, device) -> torch.Tensor:
        """采样 t ∈ (0,1)。热身期用 logit-normal baseline，之后用自适应 CDF。"""
        if self._cdf_values is None:
            return self._sample_baseline(bs, device)
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
        return sample_t(
            bs,
            device,
            mode=self.baseline_mode,
            shift=self.baseline_shift,
            mix_low_prob=self.baseline_mix_low_prob,
            timestep_schedule_shift=self.baseline_timestep_schedule_shift,
        )

    def record(self, t: torch.Tensor, raw_mse: torch.Tensor):
        """记录 per-sample 原始 MSE（不含任何 loss weight）到对应 bin。"""
        t_np = t.detach().cpu().float().numpy()
        mse_np = raw_mse.detach().cpu().float().numpy()
        sigma_np = t_np / np.clip(1.0 - t_np, 1e-8, None)
        log_sigma_np = np.log(np.clip(sigma_np, 1e-8, None))
        edges_inner = self._log_sigma_edges[1:-1]
        for i in range(len(t_np)):
            k = int(np.searchsorted(edges_inner, log_sigma_np[i]))
            self._fifo[k].append(float(mse_np[i]))
            self._n_count[k] = min(self._n_count[k] + 1, self.B)
        self._internal_step += 1   # 仅诊断用：micro-batch 累计计数，不参与 warmup 判定

    def maybe_refresh(self, global_step: int):
        """条件满足时刷新 schedule（每 M 步、热身结束后、每 bin 有足够样本）。

        warmup gate 用 global_step (optimizer step)，与 N_warm 在 build 里按
        total_steps × 20% 计算的语义一致。grad_accum>1 项目下用 _internal_step
        会让 warmup 提前 grad_accum× 结束（PR #TODO 修复）。
        """
        if global_step < self.N_warm:
            return
        if global_step % self.M != 0:
            return
        if int(np.min(self._n_count)) < self.N_min:
            self._last_refresh_status = "skipped_bins_not_full"
            return
        self._refresh()
        # 冷启动 trip wire：跑完一次完整 _refresh 但 CDF 仍未就绪 → InfoNoise
        # 静默走 baseline，logger.warning 一次性提醒用户避免"花算力没效果"。
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

    # ─── pause/resume 支持（ADR 0006 Addendum 1）：保存自适应 schedule 防止 resume 丢 CDF ───
    # 不保存 hyperparameter（K/B/N_warm/M/beta/...）—— 这些由 args 重建；只保存 *学到的* 状态。
    # 但记录 K/B 让 load_state_dict 做形状校验，避免不同配置 ckpt 间错乱。

    # v1 = σ³ entropy rate + buggy pivot；v2 = σ² log-σ entropy rate + paper-aligned pivot
    _STATE_VERSION = 2

    def state_dict(self) -> dict:
        return {
            "__version__": self._STATE_VERSION,
            "K": self.K,
            "B": self.B,
            "fifo": [list(buf) for buf in self._fifo],
            "mse_ema": self._mse_ema.copy(),
            "n_count": self._n_count.copy(),
            "cdf_values": None if self._cdf_values is None else self._cdf_values.copy(),
            "internal_step": int(self._internal_step),
            "last_refresh_status": self._last_refresh_status,
            "refresh_attempts": int(self._refresh_attempts),
            "refresh_degraded_count": int(self._refresh_degraded_count),
            "warned_cold_start": bool(self._warned_cold_start),
        }

    def load_state_dict(self, state: dict) -> None:
        saved_version = int(state.get("__version__", 1))
        if saved_version != self._STATE_VERSION:
            # v1 用 σ³ 公式 + buggy pivot，跟 v2 数学语义不兼容；丢 mse_ema/cdf 走冷启动
            logger.warning(
                "InfoNoise resume: state_dict version %d (current=%d) — 算法已变更"
                "（σ³→σ² entropy rate + paper-aligned gate pivot），丢弃 mse_ema/cdf "
                "走冷启动 warmup。",
                saved_version, self._STATE_VERSION,
            )
            return
        saved_K = int(state.get("K", self.K))
        saved_B = int(state.get("B", self.B))
        if saved_K != self.K or saved_B != self.B:
            # 已经跑了几小时，配置改了不要崩 —— 退回冷启动让 warmup 重走。
            logger.warning(
                "InfoNoise resume: shape mismatch (saved K=%d B=%d, current K=%d B=%d) "
                "—— 跳过 sampler state 加载，从冷启动重 warmup。",
                saved_K, saved_B, self.K, self.B,
            )
            return
        self._fifo = [deque(buf, maxlen=self.B) for buf in state["fifo"]]
        self._mse_ema = np.asarray(state["mse_ema"], dtype=np.float64).copy()
        self._n_count = np.asarray(state["n_count"], dtype=np.int32).copy()
        cdf = state.get("cdf_values")
        self._cdf_values = None if cdf is None else np.asarray(cdf, dtype=np.float64).copy()
        self._internal_step = int(state.get("internal_step", 0))
        self._last_refresh_status = str(state.get("last_refresh_status", "not_refreshed_yet"))
        self._refresh_attempts = int(state.get("refresh_attempts", 0))
        self._refresh_degraded_count = int(state.get("refresh_degraded_count", 0))
        self._warned_cold_start = bool(state.get("warned_cold_start", False))

    def _refresh(self):
        self._refresh_attempts += 1

        # Step A+B: 平均 loss + EMA 平滑（论文 §3.1 binwise smoothing）
        # 实现选择：m̂_k ← (1-β)·m̂_k + β·ℓ̄_k，β 控制新值权重（β=0.9 → 新值占 90%）。
        # 论文 §3.1 描述 "smoothed binwise estimate" 但未给字面公式；测试
        # test_ema_responsiveness_codifies_design_choice 锁定此选择。
        l_bar = np.array([
            float(np.mean(list(buf))) if buf else 0.0
            for buf in self._fifo
        ])
        self._mse_ema = (1.0 - self.beta) * self._mse_ema + self.beta * l_bar

        # Step C: entropy rate r̂_k = mse_k / σ_k² (log-σ 空间，论文附录 B.2 Eq 61)
        # 注：早期实现用 mse/σ³（VE 通道 Eq 59）+ Δlog σ 求和缺一档 σ Jacobian；
        # σ² 让 log-σ 空间 entropy rate 与归一化自洽（M1/M3 修复）。
        r_hat = self._mse_ema / (self._sigma_centers ** 2 + 1e-30)

        # Step D: gate pivot c
        # 默认走 gate_pivot_c=0.15（paper §5 CIFAR 报告值）；gate_pivot_c=0
        # 走 dynamic Eq 87 字面实现（从高 σ 向低 σ 扫，找最后一个 r_norm ≥ p_onset 的 bin）。
        # 早期实现 above.argmax() 选低 σ 端第一个 above → c=σ_min 让 gate 退化为恒等
        # 映射（E2E 实测 mass 99% 集中到 σ_min quarter）。
        r_max = float(r_hat.max())
        if r_max < 1e-30:
            self._last_refresh_status = "mse_collapsed"
            self._refresh_degraded_count += 1
            return
        if self.gate_pivot_c > 0:
            c = float(self.gate_pivot_c)
        else:
            r_norm = r_hat / r_max
            above = r_norm >= self.p_onset
            # any(above) 严格恒为 True：r_norm.max() ≡ 1.0 ≥ p_onset
            # （__init__ 已 assert p_onset ∈ (0,1)），故 dynamic 路径不需早退分支
            last_above = int(np.where(above)[0][-1])
            c = float(self._sigma_centers[last_above])

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
        gate_pivot_c=float(getattr(args, "infonoise_gate_pivot_c", 0.15) or 0.0),
        baseline_shift=float(getattr(args, "timestep_shift", 3.0) or 3.0),
        baseline_mode=str(getattr(args, "timestep_sampling", "logit_normal") or "logit_normal"),
        baseline_mix_low_prob=float(getattr(args, "timestep_mix_low_prob", 0.0) or 0.0),
        baseline_timestep_schedule_shift=float(getattr(args, "timestep_schedule_shift", 1.0) or 1.0),
    )
    logger.info(
        f"InfoNoise 已启用：K={scheduler.K}, N_warm={scheduler.N_warm}, "
        f"M={scheduler.M}, B={scheduler.B}, beta={scheduler.beta}, "
        f"gate_pivot_c={scheduler.gate_pivot_c}, "
        f"baseline={scheduler.baseline_mode}(shift={scheduler.baseline_shift}, "
        f"mix_low_prob={scheduler.baseline_mix_low_prob}, "
        f"timestep_schedule_shift={scheduler.baseline_timestep_schedule_shift})"
    )
    return scheduler
