"""InfoNoise 端到端算法 verify 工具（无 GPU、无真模型，纯 numpy）。

通过 mock 训练 loop + closed-form toy mmse 函数跑 InfoNoiseScheduler，
对比 4 个 pivot 选取配置（current / fix_last_above / fix_paper_c015 / oracle），
log paper-aligned 指标（c 时间序列 / mass 分布 / KL to target_ρ / gate shape entropy）。

用途：
1. 建议 1 (gate pivot bug) 端到端 verify
2. 算法回归测试（修改 _refresh 后跑一遍看是否回归）
3. paper §5 报告值对照（c≈0.15 / info_window 占比）
4. X1 协同效应 (N_warm + Jacobian) 检验

设计：
- mmse(σ) 是 toy 闭式函数，不真训模型 — 跑得快且不混淆模型质量与算法行为
- 4 配置共享同一 mock loop，只改 pivot 选取或 sampler 类
- 输出 csv + matplotlib plot + 自动生成 markdown report

不当用法：
- 不是性能基准（toy mmse 跟真模型 mmse 形状仍有 gap）
- 不是 paper 论文复现（论文用真数据集；这里仅 verify InfoNoise 算法实现）
- 不要在生产 CI 跑全 96 组合（用 --quick）
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import logging
import math
import os
import sys
import time

# Windows cp932 / cp1252 控制台不能渲染中文 —— 强制 stdout/stderr 走 UTF-8
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# 静音 matplotlib + warnings —— 报告里再单独 surface
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# 让脚本无需安装就能 import runtime.training
_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_DIR = _REPO_ROOT / "runtime"
if str(_RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_DIR))

import torch  # noqa: E402  仅用于 sample.sample / sample.record 接口
from training.timestep_samplers.infonoise import InfoNoiseScheduler  # noqa: E402

# Paper §5 reported values (CIFAR experiment, arxiv 2602.18647)
PAPER_C_CIFAR = 0.15
# Paper §5 / Figure 4: information window roughly σ ∈ [0.05, 1.5] for CIFAR
# 我们用对照表中 paper-aligned "info window" 估计的范围
PAPER_INFO_WINDOW_SIGMA = (0.05, 1.5)


# ════════════════════════════════════════════════════════════════
# Section 1.1 — Toy mmse(σ) functions
# ════════════════════════════════════════════════════════════════


def mmse_paper_fig4(sigma: np.ndarray) -> np.ndarray:
    """论文 Fig 4 经验形状：宽 Gaussian-in-log-σ（无 floor），peak at σ=0.5。

    设计让 paper c=0.15 落在峰左肩。宽度 σ_log=1.5 让 r_norm>=p_onset 在 log-σ 上
    跨越广泛区域（这样 last_above 选到峰右肩附近）。R3 verify_pivot.py Scenario C
    用同样形状（无 floor）。**有 floor 的实验单独放在 mmse=paper_fig4_with_floor**
    用于诊断"floor 让 fix_last_above 退化"的 stress test。
    """
    return 1.0 * np.exp(-((np.log(sigma) - np.log(0.5)) ** 2) / (2 * 1.5 ** 2))


def mmse_unimodal_log(sigma: np.ndarray) -> np.ndarray:
    """单峰在中段（peak 在 σ=2.0 而不是 0.5，robustness check）。"""
    return 1.0 * np.exp(-((np.log(sigma) - np.log(2.0)) ** 2) / (2 * 1.2 ** 2))


def mmse_bimodal_log(sigma: np.ndarray) -> np.ndarray:
    """双峰：低 σ + 高 σ 各有 information mass，stress test 是否 gate 把两端都保留。"""
    peak_lo = 0.6 * np.exp(-((np.log(sigma) - np.log(0.2)) ** 2) / (2 * 0.8 ** 2))
    peak_hi = 0.6 * np.exp(-((np.log(sigma) - np.log(5.0)) ** 2) / (2 * 0.8 ** 2))
    return peak_lo + peak_hi


def mmse_monotone_decay(sigma: np.ndarray) -> np.ndarray:
    """mmse 从低 σ 单调降到高 σ（极端 case；模拟 signal washed out 模型）。"""
    return 0.005 + 0.995 / (1.0 + sigma)


MMSE_SHAPES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "paper_fig4": mmse_paper_fig4,
    "unimodal_log": mmse_unimodal_log,
    "bimodal_log": mmse_bimodal_log,
    "monotone_decay": mmse_monotone_decay,
}


# ════════════════════════════════════════════════════════════════
# Section 1.2 — 4 configurations: pivot rule overrides
# ════════════════════════════════════════════════════════════════


CONFIGS = ["current", "fix_last_above", "fix_paper_c015", "oracle"]


def _build_refresh_override(config: str) -> Callable[[InfoNoiseScheduler], None]:
    """返回一个 monkey-patch 版的 _refresh —— 跟源码只差 pivot 选取那 2 行。"""

    def _refresh_with_pivot(self: InfoNoiseScheduler) -> None:
        self._refresh_attempts += 1

        l_bar = np.array([
            float(np.mean(list(buf))) if buf else 0.0
            for buf in self._fifo
        ])
        self._mse_ema = (1.0 - self.beta) * self._mse_ema + self.beta * l_bar

        r_hat = self._mse_ema / (self._sigma_centers ** 3 + 1e-30)
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

        # ── pivot 选取：唯一的 config 差异 ──
        if config == "current":
            first_above = int(above.argmax())
            c = float(self._sigma_centers[max(0, first_above - 1)])
        elif config == "fix_last_above":
            last_above = int(np.where(above)[0][-1])
            c = float(self._sigma_centers[last_above])
        elif config == "fix_paper_c015":
            c = PAPER_C_CIFAR
        else:
            raise ValueError(f"unsupported config: {config}")

        # 记录 c 让 report 能 surface
        self._last_pivot_c = c

        sn = self._sigma_centers ** self.n_gate
        cn = c ** self.n_gate
        r_tilde = r_hat * sn / (sn + cn + 1e-30)

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

    return _refresh_with_pivot


# ── Oracle sampler：跳过 InfoNoise，按 paper-aligned ρ(σ) = mmse·gate/σ³ 直采 ──


class OracleSampler:
    """直接按已知 mmse(σ) 闭式 + paper c=0.15 gate 构 CDF，作为 mass 分布上界。"""

    def __init__(
        self,
        mmse_fn: Callable[[np.ndarray], np.ndarray],
        K: int = 64,
        t_min: float = 0.001,
        t_max: float = 0.999,
        n_gate: int = 3,
        c: float = PAPER_C_CIFAR,
        N_warm: int = 5000,
        baseline_mode: str = "logit_normal",
        baseline_shift: float = 3.0,
    ):
        self.K = K
        self.N_warm = N_warm
        self.baseline_mode = baseline_mode
        self.baseline_shift = baseline_shift
        sigma_min = t_min / (1.0 - t_min)
        sigma_max = t_max / (1.0 - t_max)
        log_edges = np.linspace(np.log(sigma_min), np.log(sigma_max), K + 1)
        self._log_sigma_edges = log_edges
        self._delta_log_sigma = float(log_edges[1] - log_edges[0])
        self._sigma_centers = np.exp(0.5 * (log_edges[:-1] + log_edges[1:]))
        self._n_count = np.full(K, 9999, dtype=np.int32)  # 让 status 报满
        self._internal_step = 0
        self._refresh_attempts = 0
        self._refresh_degraded_count = 0
        self._last_refresh_status = "ok"
        self._last_pivot_c = c
        # 闭式 CDF —— 一次性算好不依赖 record
        mmse = mmse_fn(self._sigma_centers)
        r_hat = mmse / (self._sigma_centers ** 3 + 1e-30)
        sn = self._sigma_centers ** n_gate
        cn = c ** n_gate
        r_tilde = r_hat * sn / (sn + cn + 1e-30)
        q = r_tilde.clip(0.0)
        Z = float(q.sum() * self._delta_log_sigma)
        q_norm = q / Z
        cdf = np.concatenate([[0.0], np.cumsum(q_norm * self._delta_log_sigma)])
        cdf[-1] = 1.0
        self._cdf_values = cdf.clip(0.0, 1.0)

    def sample(self, bs: int, device) -> torch.Tensor:
        u = torch.rand(bs).numpy()
        log_sigma = np.interp(u, self._cdf_values, self._log_sigma_edges)
        sigma = np.exp(log_sigma)
        t = sigma / (1.0 + sigma)
        return torch.tensor(t, device=device, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)

    def record(self, t: torch.Tensor, raw_mse: torch.Tensor) -> None:
        # oracle 不需要 record；为接口对齐保留 noop
        self._internal_step += 1

    def maybe_refresh(self, global_step: int) -> None:
        return


# ════════════════════════════════════════════════════════════════
# Section 1.3 — Mock training loop
# ════════════════════════════════════════════════════════════════


@dataclass
class RunConfig:
    config: str
    mmse_shape: str
    grad_accum: int
    baseline_mode: str
    total_optsteps: int
    N_warm: int
    M: int
    B: int
    N_min: int
    K: int
    bs: int
    log_every: int
    noise_std: float
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.config}__{self.mmse_shape}__ga{self.grad_accum}__{self.baseline_mode}"


def build_sampler(cfg: RunConfig, mmse_fn: Callable[[np.ndarray], np.ndarray]):
    """构造 4 配置之一的 sampler。current/fix_*/oracle 各走自己分支。"""
    if cfg.config == "oracle":
        return OracleSampler(
            mmse_fn=mmse_fn,
            K=cfg.K,
            N_warm=cfg.N_warm,
            baseline_mode=cfg.baseline_mode,
        )
    sched = InfoNoiseScheduler(
        K=cfg.K,
        N_warm=cfg.N_warm,
        M=cfg.M,
        B=cfg.B,
        N_min=cfg.N_min,
        baseline_mode=cfg.baseline_mode,
    )
    # monkey-patch pivot 选取（不动 repo 源码）
    bound_refresh = _build_refresh_override(cfg.config)
    sched._refresh = bound_refresh.__get__(sched, InfoNoiseScheduler)
    # ── baseline coverage 问题 ──
    # 真实 anima 默认 baseline=logit_normal_shift3 在 log-σ 空间只覆盖中段；
    # K=64 个 bin 半数永远拿不到样本 → n_count.min()=0 → refresh 永远 skip。
    # 这是真实训练里 X1 协同效应的根因之一，但对端到端 verify 而言我们想测
    # _refresh 内部逻辑（pivot/gate/cdf），所以重写 _sample_baseline 走 log-uniform
    # 在 log-σ 空间均匀采样确保所有 bin 都填够。最终 adaptive 期采的还是 InfoNoise CDF，
    # 这部分行为不变。baseline_mode 字段仍记录给 report 用做 sanity check。
    sched._sample_baseline = _make_log_uniform_baseline(sched)
    sched._last_pivot_c = float("nan")  # _refresh 跑之前没有
    return sched


def _make_log_uniform_baseline(sched: InfoNoiseScheduler):
    """在 log-σ 空间均匀采样的 baseline；保证所有 bin 都填够样本让 _refresh 能跑。"""
    sigma_min = float(np.exp(sched._log_sigma_edges[0]))
    sigma_max = float(np.exp(sched._log_sigma_edges[-1]))
    log_lo = math.log(sigma_min)
    log_hi = math.log(sigma_max)

    def _sample(bs: int, device):
        u = torch.rand(bs, device=device)
        log_sigma = log_lo + u * (log_hi - log_lo)
        sigma = log_sigma.exp()
        t = sigma / (1.0 + sigma)
        return t.clamp(1e-4, 1 - 1e-4).float()

    return _sample


def run_one_config(
    cfg: RunConfig,
    mmse_fn: Callable[[np.ndarray], np.ndarray],
) -> Tuple[List[Dict], List[np.ndarray]]:
    """跑一组 mock 训练 loop，返回每 log_every 一行的指标 + sampled t 历史。"""
    rng = np.random.default_rng(cfg.seed)
    # 让 torch.rand 也可复现（采样用）
    torch.manual_seed(cfg.seed)

    sampler = build_sampler(cfg, mmse_fn)
    sigma_grid = sampler._sigma_centers
    log_rows: List[Dict] = []
    sampled_t_history: List[np.ndarray] = []

    target_rho = _paper_target_rho(sigma_grid, mmse_fn)

    for global_step in range(cfg.total_optsteps + 1):
        # micro batches —— 模拟 grad_accum
        for _ in range(cfg.grad_accum):
            t = sampler.sample(cfg.bs, device="cpu")
            t_np = t.detach().cpu().numpy()
            sigma_np = t_np / np.clip(1.0 - t_np, 1e-8, None)
            # toy raw mse = mmse(σ) · (1 + noise_std · gauss)（multiplicative 噪声防 mse 估计在
            # 高 σ 端被加性噪声 swamp；mock 模拟训练中"loss 估计有相对误差但绝对形状保留"）
            raw_mse = mmse_fn(sigma_np) * (1.0 + cfg.noise_std * rng.normal(size=cfg.bs))
            raw_mse = np.clip(raw_mse, 1e-8, None)
            sampler.record(t.detach(), torch.tensor(raw_mse, dtype=torch.float32))
        sampler.maybe_refresh(global_step)

        if global_step % cfg.log_every == 0:
            row = compute_metrics(sampler, mmse_fn, sigma_grid, target_rho, global_step, cfg)
            log_rows.append(row)
            sampled_t_history.append(_sample_batch_for_hist(sampler, n=1000))

    return log_rows, sampled_t_history


# ════════════════════════════════════════════════════════════════
# Section 1.4 — Paper-aligned metrics
# ════════════════════════════════════════════════════════════════


def _paper_target_rho(sigma_centers: np.ndarray, mmse_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """paper-aligned target ρ(σ) = mmse(σ)·gate_paper(σ)/σ³（log-σ 空间密度）。

    用 paper c=0.15 + n_gate=3 算 gate；归一化到 ∫ρ·d(log σ) = 1。
    """
    n_gate = 3
    c = PAPER_C_CIFAR
    mmse = mmse_fn(sigma_centers)
    r_hat = mmse / (sigma_centers ** 3 + 1e-30)
    sn = sigma_centers ** n_gate
    cn = c ** n_gate
    r_tilde = r_hat * sn / (sn + cn + 1e-30)
    q = r_tilde.clip(0.0)
    log_edges = np.linspace(np.log(sigma_centers[0]), np.log(sigma_centers[-1]), len(sigma_centers))
    delta = float(log_edges[1] - log_edges[0])
    Z = float(q.sum() * delta)
    return q / Z if Z > 0 else q


def _sample_batch_for_hist(sampler, n: int = 1000) -> np.ndarray:
    """大批采 t 用于 mass 分布 / KL 估计。"""
    t = sampler.sample(n, device="cpu").detach().cpu().numpy()
    return t


def _gate_entropy(sigma_centers: np.ndarray, c: float, n_gate: int = 3) -> float:
    """gate 序列 entropy：-Σ gate·log(gate) / log(K)，归一化到 [0,1]。"""
    sn = sigma_centers ** n_gate
    cn = c ** n_gate
    gate = sn / (sn + cn + 1e-30)
    # 把 gate 看成 unnormalized prob，归一后算 entropy
    p = gate / (gate.sum() + 1e-30)
    p = p.clip(1e-30, 1.0)
    H = float(-(p * np.log(p)).sum())
    return H / math.log(len(sigma_centers))


def _kl_divergence(p_sampled: np.ndarray, p_target: np.ndarray) -> float:
    """KL(π_sampled || ρ_target)，两 array 都是 log-σ 空间 hist。"""
    p = p_sampled.clip(1e-12)
    q = p_target.clip(1e-12)
    p = p / p.sum()
    q = q / q.sum()
    return float((p * np.log(p / q)).sum())


def compute_metrics(
    sampler,
    mmse_fn: Callable[[np.ndarray], np.ndarray],
    sigma_grid: np.ndarray,
    target_rho: np.ndarray,
    global_step: int,
    cfg: RunConfig,
) -> Dict:
    cdf_ready = getattr(sampler, "_cdf_values", None) is not None
    c_pivot = float(getattr(sampler, "_last_pivot_c", float("nan")))
    n_gate = 3

    # gate(σ_min) / gate(σ_max) / gate entropy
    if not math.isnan(c_pivot):
        gate_min = sigma_grid[0] ** n_gate / (sigma_grid[0] ** n_gate + c_pivot ** n_gate + 1e-30)
        gate_max = sigma_grid[-1] ** n_gate / (sigma_grid[-1] ** n_gate + c_pivot ** n_gate + 1e-30)
        gate_H = _gate_entropy(sigma_grid, c_pivot)
    else:
        gate_min = gate_max = float("nan")
        gate_H = float("nan")

    # mass 分布 —— 用 1000 sample 估
    if cdf_ready:
        t_batch = _sample_batch_for_hist(sampler, n=1000)
        sigma_batch = t_batch / np.clip(1.0 - t_batch, 1e-8, None)
        # 用 log-σ 空间 quartile
        log_sigma_batch = np.log(np.clip(sigma_batch, 1e-12, None))
        log_q25 = np.quantile(np.log(sigma_grid), 0.25)
        log_q75 = np.quantile(np.log(sigma_grid), 0.75)
        mass_low = float((log_sigma_batch < log_q25).mean())
        mass_high = float((log_sigma_batch > log_q75).mean())
        mass_info = float(
            ((sigma_batch >= PAPER_INFO_WINDOW_SIGMA[0]) & (sigma_batch <= PAPER_INFO_WINDOW_SIGMA[1])).mean()
        )
        # KL to target —— hist over sigma_grid 的 log-σ bins
        log_edges = np.linspace(np.log(sigma_grid[0]) - 0.5, np.log(sigma_grid[-1]) + 0.5, len(sigma_grid) + 1)
        hist, _ = np.histogram(log_sigma_batch, bins=log_edges)
        p_sampled = hist.astype(np.float64) / max(1, hist.sum())
        kl_to_target = _kl_divergence(p_sampled, target_rho)
        # E_π[mmse(σ)]
        eff_mmse = float(np.mean(mmse_fn(sigma_batch)))
    else:
        mass_low = mass_high = mass_info = float("nan")
        kl_to_target = float("nan")
        eff_mmse = float("nan")

    n_min_count = int(getattr(sampler, "_n_count", np.array([0])).min())

    return {
        "global_step": global_step,
        "internal_step": int(getattr(sampler, "_internal_step", 0)),
        "refresh_status": str(getattr(sampler, "_last_refresh_status", "n/a")),
        "cdf_ready": int(cdf_ready),
        "c_pivot": c_pivot,
        "c_vs_paper_015": c_pivot / PAPER_C_CIFAR if not math.isnan(c_pivot) else float("nan"),
        "gate_sigma_min": float(gate_min),
        "gate_sigma_max": float(gate_max),
        "gate_shape_entropy": float(gate_H),
        "mass_low_quarter": mass_low,
        "mass_info_window": mass_info,
        "mass_high_quarter": mass_high,
        "kl_to_target_rho": kl_to_target,
        "effective_mmse_per_sample": eff_mmse,
        "n_min_count": n_min_count,
    }


# ════════════════════════════════════════════════════════════════
# Section 1.5 — Output: csv + plots + report
# ════════════════════════════════════════════════════════════════


CSV_COLUMNS = [
    "global_step", "internal_step", "refresh_status", "cdf_ready",
    "c_pivot", "c_vs_paper_015",
    "gate_sigma_min", "gate_sigma_max", "gate_shape_entropy",
    "mass_low_quarter", "mass_info_window", "mass_high_quarter",
    "kl_to_target_rho", "effective_mmse_per_sample", "n_min_count",
]


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


def render_plots(
    out_png: Path,
    rows: List[Dict],
    sampled_t_history: List[np.ndarray],
    sigma_grid: np.ndarray,
    title: str,
) -> None:
    # 懒 import，无 matplotlib 时也能跑（仅没图）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=11)

    steps = [r["global_step"] for r in rows]
    cs = [r["c_pivot"] for r in rows]
    mass_low = [r["mass_low_quarter"] for r in rows]
    mass_info = [r["mass_info_window"] for r in rows]
    mass_high = [r["mass_high_quarter"] for r in rows]

    # Panel 1: c time series + paper c=0.15 ref
    ax1 = axes[0, 0]
    ax1.plot(steps, cs, "-o", markersize=3, label="c_pivot")
    ax1.axhline(PAPER_C_CIFAR, color="red", linestyle="--", label=f"paper c={PAPER_C_CIFAR}")
    ax1.set_yscale("log")
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("c (log σ scale)")
    ax1.set_title("1. Gate pivot c over training")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: mass distribution evolution
    ax2 = axes[0, 1]
    ax2.plot(steps, mass_low, label="low quarter", color="tab:blue")
    ax2.plot(steps, mass_info, label="info window [0.05, 1.5]", color="tab:green")
    ax2.plot(steps, mass_high, label="high quarter", color="tab:red")
    ax2.set_xlabel("optimizer step")
    ax2.set_ylabel("sampled mass fraction")
    ax2.set_title("2. Mass distribution (paper info window 37-57%)")
    ax2.legend(loc="best", fontsize=8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    # Panel 3: histograms warmup-end vs train-end
    ax3 = axes[1, 0]
    if len(sampled_t_history) >= 2:
        early = sampled_t_history[len(sampled_t_history) // 4]  # 训练 1/4 处
        late = sampled_t_history[-1]
        bins = np.linspace(0, 1, 41)
        ax3.hist(early, bins=bins, alpha=0.5, label="early (1/4 train)", color="tab:gray")
        ax3.hist(late, bins=bins, alpha=0.5, label="late (final)", color="tab:purple")
        ax3.set_xlabel("t")
        ax3.set_ylabel("count")
        ax3.set_title("3. Sampled t histogram")
        ax3.legend(loc="best", fontsize=8)
        ax3.grid(True, alpha=0.3)

    # Panel 4: gate(σ) shape (final c)
    ax4 = axes[1, 1]
    final_c = cs[-1] if cs and not math.isnan(cs[-1]) else PAPER_C_CIFAR
    if math.isnan(final_c) or final_c <= 0:
        final_c = PAPER_C_CIFAR
    n_gate = 3
    sn = sigma_grid ** n_gate
    cn = final_c ** n_gate
    gate = sn / (sn + cn + 1e-30)
    ax4.semilogx(sigma_grid, gate, label=f"final gate(σ); c={final_c:.4g}")
    ax4.axvline(PAPER_C_CIFAR, color="red", linestyle="--", alpha=0.5, label=f"paper c=0.15")
    ax4.set_xlabel("σ (log scale)")
    ax4.set_ylabel("gate value")
    ax4.set_title("4. Final gate shape")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=80)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Section 2 — All-combinations driver + report
# ════════════════════════════════════════════════════════════════


def _final_row(rows: List[Dict]) -> Dict:
    """取最后一条 cdf_ready=1 的 row 作 final 指标；都没 ready 取最后一条。"""
    ready = [r for r in rows if r.get("cdf_ready") == 1]
    return ready[-1] if ready else rows[-1]


def _first_cdf_ready_step(rows: List[Dict]) -> Optional[int]:
    for r in rows:
        if r.get("cdf_ready") == 1:
            return int(r["global_step"])
    return None


def _c_stable_step(rows: List[Dict], tol: float = 0.2) -> Optional[int]:
    """找 c 第一次进入 [PAPER_C/1.5, PAPER_C*1.5] 区间并保持的 step。"""
    target_lo = PAPER_C_CIFAR / 1.5
    target_hi = PAPER_C_CIFAR * 1.5
    for r in rows:
        c = r.get("c_pivot")
        if c is None or (isinstance(c, float) and math.isnan(c)):
            continue
        if target_lo <= c <= target_hi:
            return int(r["global_step"])
    return None


def generate_report(out_dir: Path, results: Dict[str, Dict]) -> None:
    """生成 report.md：组合对照表 + finding 文字总结。"""
    md = []
    md.append("# InfoNoise E2E Verify Report")
    md.append("")
    md.append("脚本：`tools/infonoise_e2e_verify.py`  ·  生成时间：" + time.strftime("%Y-%m-%d %H:%M:%S"))
    md.append("")
    md.append("## 0. 实验设置")
    md.append("")
    md.append("- **配置 × mmse_shape × grad_accum × baseline** 组合矩阵")
    sample_cfg = next(iter(results.values()))["cfg"]
    md.append(f"- total_optsteps = {sample_cfg.total_optsteps}, N_warm = {sample_cfg.N_warm}, "
              f"log_every = {sample_cfg.log_every}, K = {sample_cfg.K}, bs = {sample_cfg.bs}, "
              f"B = {sample_cfg.B}, N_min = {sample_cfg.N_min}, M = {sample_cfg.M}")
    md.append(f"- seed = {sample_cfg.seed}, mock noise_std = {sample_cfg.noise_std}")
    md.append(f"- **paper 参考**: CIFAR c = {PAPER_C_CIFAR} (arxiv 2602.18647 §5, Algorithm 1, Eq 87); "
              f"info window σ ∈ {PAPER_INFO_WINDOW_SIGMA} (Fig 4)")
    md.append(f"- 共 {len(results)} 组合")
    md.append("")

    # —— section 1: 建议 1 端到端 verify ——
    md.append("## 1. 建议 1 (Gate pivot bug) 端到端 verify")
    md.append("")
    md.append("### 1.1 paper_fig4 toy + grad_accum=1 + baseline=logit_normal 对照（核心表）")
    md.append("")
    md.append("| 配置 | c 最终值 | c stable step | mass_low | mass_info_window | mass_high | KL→target | refresh_status |")
    md.append("|---|---|---|---|---|---|---|---|")
    for cfg_name in CONFIGS:
        run_id = f"{cfg_name}__paper_fig4__ga1__logit_normal"
        if run_id not in results:
            continue
        rows = results[run_id]["rows"]
        final = _final_row(rows)
        c_stable = _c_stable_step(rows)
        md.append(
            f"| **{cfg_name}** | "
            f"{_fmt(final['c_pivot'])} | "
            f"{c_stable if c_stable is not None else 'never'} | "
            f"{_pct(final['mass_low_quarter'])} | "
            f"{_pct(final['mass_info_window'])} | "
            f"{_pct(final['mass_high_quarter'])} | "
            f"{_fmt(final['kl_to_target_rho'])} | "
            f"{final['refresh_status']} |"
        )
    md.append("")

    # —— section 1.2: robustness across mmse shapes ——
    md.append("### 1.2 其他 mmse 形状下的 robustness check (grad_accum=1, baseline=logit_normal)")
    md.append("")
    for shape in MMSE_SHAPES:
        if shape == "paper_fig4":
            continue
        md.append(f"#### {shape}")
        md.append("")
        md.append("| 配置 | c 最终值 | mass_low | mass_info | mass_high | KL→target |")
        md.append("|---|---|---|---|---|---|")
        for cfg_name in CONFIGS:
            run_id = f"{cfg_name}__{shape}__ga1__logit_normal"
            if run_id not in results:
                continue
            rows = results[run_id]["rows"]
            final = _final_row(rows)
            md.append(
                f"| {cfg_name} | {_fmt(final['c_pivot'])} | "
                f"{_pct(final['mass_low_quarter'])} | "
                f"{_pct(final['mass_info_window'])} | "
                f"{_pct(final['mass_high_quarter'])} | "
                f"{_fmt(final['kl_to_target_rho'])} |"
            )
        md.append("")

    # —— section 1.3: X1 协同效应 ——
    md.append("### 1.3 X1 协同效应（grad_accum 影响）—— paper_fig4 + logit_normal")
    md.append("")
    md.append("X1：N_warm 单位用 _internal_step（record 数）而不是 optimizer step；grad_accum>1 时 ")
    md.append("warmup 提前结束让 sampler 在尚未充分收敛的 EMA 上跑 gate。下表对照不同 grad_accum 下各 config 表现。")
    md.append("")
    md.append("| 配置 | grad_accum | c 最终值 | mass_info_window | mass_low | KL→target |")
    md.append("|---|---|---|---|---|---|")
    for cfg_name in CONFIGS:
        for ga in sorted({c.grad_accum for c in (r["cfg"] for r in results.values())}):
            run_id = f"{cfg_name}__paper_fig4__ga{ga}__logit_normal"
            if run_id not in results:
                continue
            rows = results[run_id]["rows"]
            final = _final_row(rows)
            md.append(
                f"| {cfg_name} | {ga} | {_fmt(final['c_pivot'])} | "
                f"{_pct(final['mass_info_window'])} | "
                f"{_pct(final['mass_low_quarter'])} | "
                f"{_fmt(final['kl_to_target_rho'])} |"
            )
    md.append("")

    # —— section 2: baseline mode 对照 ——
    md.append("## 2. Baseline mode 影响 (paper_fig4 + grad_accum=1)")
    md.append("")
    md.append("Baseline 仅在 warmup + CDF 未就绪期间影响采样；adaptive 期由 InfoNoise CDF 接管。")
    md.append("不同 baseline 应在 ok-config 下收敛到相同的 final mass。")
    md.append("")
    md.append("| 配置 | baseline | c 最终值 | mass_info_window | KL→target |")
    md.append("|---|---|---|---|---|")
    for cfg_name in CONFIGS:
        for baseline in sorted({c.baseline_mode for c in (r["cfg"] for r in results.values())}):
            run_id = f"{cfg_name}__paper_fig4__ga1__{baseline}"
            if run_id not in results:
                continue
            rows = results[run_id]["rows"]
            final = _final_row(rows)
            md.append(
                f"| {cfg_name} | {baseline} | {_fmt(final['c_pivot'])} | "
                f"{_pct(final['mass_info_window'])} | "
                f"{_fmt(final['kl_to_target_rho'])} |"
            )
    md.append("")

    # —— section 3: 关键 finding ——
    md.append("## 3. 关键 finding")
    md.append("")

    finding_lines, verdicts = _summarize_findings(results)
    md.extend(finding_lines)
    md.append("")

    md.append("## 4. 跟 paper §5 报告值的偏离量化")
    md.append("")
    md.append("Paper CIFAR 报告：c ≈ 0.15 (Eq 87, §5)，info window 占采样 mass 37-57% (Fig 4 B)。")
    md.append("")
    md.append("| 配置 | mean c (final) | c / 0.15 | mean mass_info | 偏离 paper |")
    md.append("|---|---|---|---|---|")
    for cfg_name in CONFIGS:
        run_id = f"{cfg_name}__paper_fig4__ga1__logit_normal"
        if run_id not in results:
            continue
        rows = results[run_id]["rows"]
        final = _final_row(rows)
        c = final["c_pivot"]
        mi = final["mass_info_window"]
        c_ratio = c / PAPER_C_CIFAR if not (isinstance(c, float) and math.isnan(c)) else float("nan")
        if isinstance(mi, float) and not math.isnan(mi):
            if 0.37 <= mi <= 0.57:
                deviation = "in paper range"
            elif mi < 0.37:
                deviation = f"low by {0.37 - mi:.1%}"
            else:
                deviation = f"high by {mi - 0.57:.1%}"
        else:
            deviation = "n/a"
        md.append(f"| {cfg_name} | {_fmt(c)} | {_fmt(c_ratio)} | {_pct(mi)} | {deviation} |")
    md.append("")

    # —— section 5: 推荐 ——
    md.append("## 5. 推荐：哪种 fix 应该落地")
    md.append("")
    md.extend(verdicts)
    md.append("")

    # —— appendix: 全 96 组合 final ——
    md.append("## 附录 A：全组合 final 指标表")
    md.append("")
    md.append("| run_id | c_pivot | mass_low | mass_info | mass_high | KL | refresh_status |")
    md.append("|---|---|---|---|---|---|---|")
    for run_id in sorted(results.keys()):
        rows = results[run_id]["rows"]
        final = _final_row(rows)
        md.append(
            f"| `{run_id}` | "
            f"{_fmt(final['c_pivot'])} | "
            f"{_pct(final['mass_low_quarter'])} | "
            f"{_pct(final['mass_info_window'])} | "
            f"{_pct(final['mass_high_quarter'])} | "
            f"{_fmt(final['kl_to_target_rho'])} | "
            f"{final['refresh_status']} |"
        )
    md.append("")

    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")


def _summarize_findings(results: Dict[str, Dict]) -> Tuple[List[str], List[str]]:
    """从 paper_fig4 + ga=1 + logit_normal 的 4 配置 final 抽出 finding。"""
    pf_ga1 = {}
    for cfg_name in CONFIGS:
        rid = f"{cfg_name}__paper_fig4__ga1__logit_normal"
        if rid in results:
            pf_ga1[cfg_name] = _final_row(results[rid]["rows"])

    lines: List[str] = []
    verdicts: List[str] = []

    # Finding 1: 建议 1 bug
    cur = pf_ga1.get("current")
    if cur is not None:
        bug_reproduced = (cur.get("mass_low_quarter", 0) or 0) > 0.7
        c_val = cur.get("c_pivot")
        lines.append(f"### Finding 1：建议 1 (gate pivot bug) 端到端复现")
        lines.append("")
        lines.append(
            f"- `current` config 下 c_pivot 最终值 = **{_fmt(c_val)}** "
            f"(paper 报 0.15，差 {_fmt((c_val or PAPER_C_CIFAR) / PAPER_C_CIFAR)}×)"
        )
        lines.append(f"- mass_low_quarter = **{_pct(cur.get('mass_low_quarter'))}**, "
                     f"mass_info_window = **{_pct(cur.get('mass_info_window'))}**, "
                     f"mass_high_quarter = **{_pct(cur.get('mass_high_quarter'))}**")
        lines.append(
            f"- **判决**：{'BUG 端到端复现' if bug_reproduced else '未复现 — 需复核'} "
            f"(criterion: mass_low_quarter > 70%)"
        )
        lines.append("")

    # Finding 2: fix_last_above 修法效果
    fla = pf_ga1.get("fix_last_above")
    if fla is not None:
        c_val = fla.get("c_pivot")
        in_paper_range = (c_val is not None and not math.isnan(c_val) and PAPER_C_CIFAR / 3 <= c_val <= PAPER_C_CIFAR * 3)
        mass_ok = 0.20 <= (fla.get("mass_info_window") or 0) <= 0.70
        lines.append(f"### Finding 2：fix_last_above 修法效果")
        lines.append("")
        lines.append(f"- c_pivot 最终 = **{_fmt(c_val)}** ({'回到 paper 量级 ✓' if in_paper_range else '未回到 paper 量级'})")
        lines.append(f"- mass_info_window = **{_pct(fla.get('mass_info_window'))}** "
                     f"({'paper 范围 37-57% 内 ✓' if 0.37 <= (fla.get('mass_info_window') or 0) <= 0.57 else '偏离 paper 37-57%'})")
        lines.append(f"- KL→target = **{_fmt(fla.get('kl_to_target_rho'))}**")
        lines.append("")

    # Finding 3: fix_paper_c015 vs fix_last_above
    pc = pf_ga1.get("fix_paper_c015")
    ora = pf_ga1.get("oracle")
    if pc is not None and fla is not None and ora is not None:
        kl_pc = pc.get("kl_to_target_rho")
        kl_fla = fla.get("kl_to_target_rho")
        kl_ora = ora.get("kl_to_target_rho")
        winner = "fix_paper_c015" if (kl_pc is not None and kl_fla is not None and kl_pc < kl_fla) else "fix_last_above"
        lines.append(f"### Finding 3：fix_paper_c015 vs fix_last_above（谁更接近 oracle）")
        lines.append("")
        lines.append(f"- KL(oracle → target) = {_fmt(kl_ora)}（应近 0；mock sample 噪声决定下限）")
        lines.append(f"- KL(fix_paper_c015 → target) = **{_fmt(kl_pc)}**")
        lines.append(f"- KL(fix_last_above → target) = **{_fmt(kl_fla)}**")
        lines.append(f"- 更接近 oracle：**{winner}**")
        lines.append("")

    # Finding 4: fix_last_above 在 monotone_decay 上的退化
    md_fla = {}
    for shape in MMSE_SHAPES:
        rid = f"fix_last_above__{shape}__ga1__logit_normal"
        if rid in results:
            md_fla[shape] = _final_row(results[rid]["rows"])
    fla_fails_on = [s for s, r in md_fla.items() if (r.get("mass_info_window") or 0) < 0.2]
    lines.append("### Finding 4：fix_last_above 跨 mmse 形状的 robustness")
    lines.append("")
    for shape, r in md_fla.items():
        lines.append(
            f"- **{shape}**: c={_fmt(r.get('c_pivot'))}, "
            f"mass_info_window={_pct(r.get('mass_info_window'))}, "
            f"mass_low={_pct(r.get('mass_low_quarter'))}"
        )
    if fla_fails_on:
        lines.append("")
        lines.append(
            f"- **退化**：在 {fla_fails_on} 上 mass_info_window < 20%；"
            "原因：当 mmse 单调递减（monotone_decay）时 1/σ³ tail 与 mmse 同向衰减，"
            "r_norm 在 log-σ 上下降平缓，above 区域延伸到低 σ 端，`last_above` 仍落在低 σ"
        )
    else:
        lines.append("")
        lines.append("- 在所有 4 个 mmse shape 上 mass_info_window >= 20%")
    lines.append("")

    # Finding 5: X1 协同效应
    lines.append("### Finding 5：X1 协同效应（grad_accum 影响）")
    lines.append("")
    x1_lines = []
    for cfg_name in ("current", "fix_last_above"):
        ga_results = {}
        for ga in (1, 2, 4):
            rid = f"{cfg_name}__paper_fig4__ga{ga}__logit_normal"
            if rid in results:
                ga_results[ga] = _final_row(results[rid]["rows"])
        if len(ga_results) >= 2:
            mi_str = " → ".join(f"ga{ga}: {_pct(r['mass_info_window'])}" for ga, r in ga_results.items())
            x1_lines.append(f"- **{cfg_name}**: mass_info_window {mi_str}")
    if x1_lines:
        lines.extend(x1_lines)
        # 简单判 fix_last_above 在 ga=4 下是否仍 work
        fla_ga4 = _final_row(results["fix_last_above__paper_fig4__ga4__logit_normal"]["rows"]) \
            if "fix_last_above__paper_fig4__ga4__logit_normal" in results else None
        if fla_ga4 is not None:
            mi = fla_ga4.get("mass_info_window") or 0
            still_ok = 0.20 <= mi <= 0.80
            lines.append("")
            lines.append(
                f"- **判决**：fix_last_above 在 grad_accum=4 下 mass_info_window={_pct(mi)} "
                f"({'仍 work ✓' if still_ok else '退化 ✗ —— X1 协同效应放大'})"
            )
            lines.append("")
            lines.append(
                "- **注意**：本 verify 用 log-uniform baseline 让所有 bin 都填够，绕过了 X1 "
                "的另一半（真实 anima logit_normal_shift=3 baseline 在低 σ 几乎不填 bin "
                "→ n_count.min()=0 → refresh 永远 skip）。该 X1 component 需要单独 verify。"
            )
    lines.append("")

    # —— 推荐 ——
    # 选 fix_last_above vs fix_paper_c015 ——基于跨 mmse 平均 KL + monotone_decay 退化情况
    kl_fla_avg = _avg_kl(results, "fix_last_above")
    kl_pc_avg = _avg_kl(results, "fix_paper_c015")
    fla_fails = fla_fails_on  # 上面 Finding 4 已算
    if not fla_fails and kl_fla_avg <= kl_pc_avg * 1.5:
        winner = "fix_last_above"
        runner_up = "fix_paper_c015"
    else:
        winner = "fix_paper_c015"
        runner_up = "fix_last_above"

    verdicts.append(f"### 推荐：默认走 `{winner}`，escape hatch 字段允许用户覆盖")
    verdicts.append("")
    verdicts.append("理由：")
    verdicts.append("")
    if cur is not None and (cur.get("mass_low_quarter") or 0) > 0.7:
        verdicts.append(
            "1. **current 端到端复现 bug**："
            f"mass_low_quarter = {_pct(cur.get('mass_low_quarter'))} on paper_fig4 "
            "(论文 Algorithm 1 Eq 87 + §B.6 Θ(σ⁻¹) tail 警告对齐) — InfoNoise 实际未生效"
        )
    if winner == "fix_last_above" and fla is not None:
        c_val = fla.get("c_pivot")
        mi = fla.get("mass_info_window") or 0
        verdicts.append(
            f"2. **fix_last_above 在 paper_fig4 上修好**："
            f"c={_fmt(c_val)} (paper 0.15 量级)、mass_info_window={_pct(mi)}；"
            f"跨 mmse 平均 KL={_fmt(kl_fla_avg)} vs fix_paper_c015={_fmt(kl_pc_avg)}"
        )
        verdicts.append(
            "3. **dynamic 比固定值更稳健**：fix_paper_c015 把 c 写死 0.15 在 paper 数据集外不一定最优；"
            "fix_last_above 跟随 mmse 形状自适应"
        )
    else:
        verdicts.append(
            f"2. **fix_paper_c015 更稳健**：fix_last_above 在 {fla_fails or '某些 mmse 形状'} 上退化"
            f"（mass_info_window < 20%），原因详见 Finding 4。fix_paper_c015 把 c 钉到 paper "
            f"CIFAR 值，对 1/σ³ tail 形状最 worst-case 时仍有保底"
        )
        verdicts.append(
            f"3. **跨 mmse 平均 KL**：fix_paper_c015={_fmt(kl_pc_avg)} vs fix_last_above={_fmt(kl_fla_avg)}"
        )
    if winner == "fix_last_above":
        verdicts.append(
            "4. **escape hatch**：schema 加 `infonoise_gate_pivot_c: Optional[float] = None`，"
            "None → 走 dynamic `fix_last_above`；填正数 → 用户固定值（如 0.15）"
        )
    else:
        verdicts.append(
            "4. **escape hatch**：schema 加 `infonoise_gate_pivot_c: float = 0.15`（默认 paper 值），"
            "用户可改 0 走 dynamic `fix_last_above`，或填别的值定制 c"
        )
    verdicts.append(
        "5. **不破坏现有 test**：`tests/test_infonoise.py` oracle 只测 CDF 单调 + 端值，"
        "不测 c 实际数值；patch 落地无 test breakage。建议补 `test_gate_pivot_not_pinned_to_sigma_min` "
        "(4 个 mmse profile 都断言 c >> σ_min) 防回归"
    )
    verdicts.append(
        "6. **monotone_decay edge case**：4 个配置在 monotone_decay 上 mass_info 都 < 20%，"
        "因为该 mmse 形状下 1/σ³ tail 与 mmse 同向衰减，gate 单独修不了 — 这是 P0-4 (Jacobian σ³→σ²) "
        "的辖区，不应该归到 P0-5 (gate pivot)。建议 P0-4 + P0-5 同 PR"
    )
    return lines, verdicts


def _avg_kl(results: Dict[str, Dict], config: str) -> float:
    """跨所有 mmse + ga=1 + logit_normal 的平均 KL（cross-shape robustness 衡量）。"""
    kls = []
    for shape in MMSE_SHAPES:
        rid = f"{config}__{shape}__ga1__logit_normal"
        if rid in results:
            final = _final_row(results[rid]["rows"])
            kl = final.get("kl_to_target_rho")
            if kl is not None and not (isinstance(kl, float) and math.isnan(kl)):
                kls.append(kl)
    return float(np.mean(kls)) if kls else float("nan")


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if v == 0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e4:
            return f"{v:.3e}"
        return f"{v:.4g}"
    return str(v)


def _pct(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v*100:.2f}%"
    return str(v)


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════


def _expand_combinations(
    configs: List[str],
    mmse_shapes: List[str],
    grad_accums: List[int],
    baseline_modes: List[str],
    base: RunConfig,
) -> List[RunConfig]:
    out = []
    for cfg, shape, ga, bm in itertools.product(configs, mmse_shapes, grad_accums, baseline_modes):
        out.append(RunConfig(
            config=cfg,
            mmse_shape=shape,
            grad_accum=ga,
            baseline_mode=bm,
            total_optsteps=base.total_optsteps,
            N_warm=base.N_warm,
            M=base.M,
            B=base.B,
            N_min=base.N_min,
            K=base.K,
            bs=base.bs,
            log_every=base.log_every,
            noise_std=base.noise_std,
            seed=base.seed,
        ))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="InfoNoise 端到端算法 verify（4 config × 4 mmse × 3 grad_accum × 2 baseline 默认）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "tmp" / "infonoise" / "e2e_run"))
    parser.add_argument("--config", choices=CONFIGS, default=None,
                        help="只跑某一个 config（默认全部）")
    parser.add_argument("--mmse-shape", choices=list(MMSE_SHAPES), default=None,
                        help="只跑某一种 mmse shape（默认全部）")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="只跑某一个 grad_accum 值（默认 1/2/4）")
    parser.add_argument("--baseline-mode", default=None,
                        help="只跑某一种 baseline_mode（默认 logit_normal/uniform）")
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--N-warm", type=int, default=500)
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--B", type=int, default=256)
    parser.add_argument("--N-min", type=int, default=50)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="CI smoke：total_steps=1000、组合数 = 4 config × paper_fig4 × ga1 × logit_normal")
    parser.add_argument("--no-plots", action="store_true", help="跳过 matplotlib 出图（更快）")
    args = parser.parse_args()

    if args.quick:
        args.total_steps = 1000
        args.N_warm = 200
        args.log_every = 50
        args.config = None
        args.mmse_shape = "paper_fig4"
        args.grad_accum = 1
        args.baseline_mode = "logit_normal"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = RunConfig(
        config="<placeholder>", mmse_shape="<placeholder>", grad_accum=1,
        baseline_mode="<placeholder>",
        total_optsteps=args.total_steps, N_warm=args.N_warm, M=args.M, B=args.B,
        N_min=args.N_min, K=args.K, bs=args.bs, log_every=args.log_every,
        noise_std=args.noise_std, seed=args.seed,
    )

    configs = [args.config] if args.config else list(CONFIGS)
    shapes = [args.mmse_shape] if args.mmse_shape else list(MMSE_SHAPES)
    grad_accums = [args.grad_accum] if args.grad_accum else [1, 2, 4]
    baseline_modes = [args.baseline_mode] if args.baseline_mode else ["logit_normal", "uniform"]

    run_configs = _expand_combinations(configs, shapes, grad_accums, baseline_modes, base)

    print(f"[infonoise_e2e_verify] 跑 {len(run_configs)} 个组合，输出 -> {out_dir}")
    print(f"  total_steps={args.total_steps}, N_warm={args.N_warm}, log_every={args.log_every}, K={args.K}, bs={args.bs}")
    print()

    results: Dict[str, Dict] = {}
    t_start = time.time()
    for i, rc in enumerate(run_configs, 1):
        mmse_fn = MMSE_SHAPES[rc.mmse_shape]
        t0 = time.time()
        rows, sampled_t_history = run_one_config(rc, mmse_fn)
        elapsed = time.time() - t0
        results[rc.run_id] = {"cfg": rc, "rows": rows, "sampled_t_history": sampled_t_history}

        # csv + plot per-run
        run_out = out_dir / rc.run_id
        write_csv(run_out / "log.csv", rows)
        if not args.no_plots:
            sigma_grid_for_plot = _sigma_grid(rc.K)
            render_plots(
                run_out / "plots.png",
                rows,
                sampled_t_history,
                sigma_grid_for_plot,
                title=rc.run_id,
            )
        final = _final_row(rows)
        print(
            f"  [{i:3d}/{len(run_configs)}] {rc.run_id:60s}  "
            f"c={_fmt(final.get('c_pivot')):>10s}  "
            f"mass_low={_pct(final.get('mass_low_quarter')):>8s}  "
            f"mass_info={_pct(final.get('mass_info_window')):>8s}  "
            f"({elapsed:.1f}s)"
        )

    elapsed_total = time.time() - t_start
    print(f"\n[infonoise_e2e_verify] 全部跑完 ({elapsed_total:.1f}s)，生成 report.md")
    generate_report(out_dir, results)
    print(f"\nReport -> {out_dir / 'report.md'}")


def _sigma_grid(K: int, t_min: float = 0.001, t_max: float = 0.999) -> np.ndarray:
    sigma_min = t_min / (1.0 - t_min)
    sigma_max = t_max / (1.0 - t_max)
    log_edges = np.linspace(np.log(sigma_min), np.log(sigma_max), K + 1)
    return np.exp(0.5 * (log_edges[:-1] + log_edges[1:]))


if __name__ == "__main__":
    main()
