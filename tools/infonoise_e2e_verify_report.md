# InfoNoise E2E Verify Report

脚本：`tools/infonoise_e2e_verify.py`  ·  生成时间：2026-06-04 16:58:58

## 0. 实验设置

- **配置 × mmse_shape × grad_accum × baseline** 组合矩阵
- total_optsteps = 5000, N_warm = 500, log_every = 100, K = 64, bs = 16, B = 256, N_min = 50, M = 100
- seed = 42, mock noise_std = 0.1
- **paper 参考**: CIFAR c = 0.15 (arxiv 2602.18647 §5, Algorithm 1, Eq 87); info window σ ∈ (0.05, 1.5) (Fig 4)
- 共 96 组合

## 1. 建议 1 (Gate pivot bug) 端到端 verify

### 1.1 paper_fig4 toy + grad_accum=1 + baseline=logit_normal 对照（核心表）

| 配置 | c 最终值 | c stable step | mass_low | mass_info_window | mass_high | KL→target | refresh_status |
|---|---|---|---|---|---|---|---|
| **current** | 0.001115 | never | 98.70% | 0.80% | 0.00% | 3.968 | ok |
| **fix_last_above** | 0.1037 | 500 | 19.90% | 63.30% | 0.00% | 0.0341 | ok |
| **fix_paper_c015** | 0.15 | 500 | 15.00% | 73.00% | 0.00% | 0.02865 | ok |
| **oracle** | 0.15 | 0 | 15.70% | 71.40% | 0.00% | 0.02488 | ok |

### 1.2 其他 mmse 形状下的 robustness check (grad_accum=1, baseline=logit_normal)

#### unimodal_log

| 配置 | c 最终值 | mass_low | mass_info | mass_high | KL→target |
|---|---|---|---|---|---|
| current | 0.001115 | 56.00% | 29.00% | 0.00% | 2.474 |
| fix_last_above | 1.715 | 0.10% | 80.10% | 0.00% | 1.956 |
| fix_paper_c015 | 0.15 | 1.30% | 96.30% | 0.00% | 0.02438 |
| oracle | 0.15 | 1.00% | 96.30% | 0.00% | 0.01986 |

#### bimodal_log

| 配置 | c 最终值 | mass_low | mass_info | mass_high | KL→target |
|---|---|---|---|---|---|
| current | 0.001115 | 55.60% | 23.90% | 0.00% | 1.499 |
| fix_last_above | 0.4698 | 1.30% | 95.70% | 0.00% | 0.4453 |
| fix_paper_c015 | 0.15 | 2.70% | 90.30% | 0.00% | 0.03396 |
| oracle | 0.15 | 2.90% | 88.80% | 0.00% | 0.02973 |

#### monotone_decay

| 配置 | c 最终值 | mass_low | mass_info | mass_high | KL→target |
|---|---|---|---|---|---|
| current | 0.001115 | 100.00% | 0.00% | 0.00% | 1.391 |
| fix_last_above | 0.007779 | 99.80% | 0.10% | 0.00% | 0.6339 |
| fix_paper_c015 | 0.15 | 72.80% | 19.10% | 0.00% | 0.07673 |
| oracle | 0.15 | 74.10% | 17.70% | 0.00% | 0.08269 |

### 1.3 X1 协同效应（grad_accum 影响）—— paper_fig4 + logit_normal

X1：N_warm 单位用 _internal_step（record 数）而不是 optimizer step；grad_accum>1 时 
warmup 提前结束让 sampler 在尚未充分收敛的 EMA 上跑 gate。下表对照不同 grad_accum 下各 config 表现。

| 配置 | grad_accum | c 最终值 | mass_info_window | mass_low | KL→target |
|---|---|---|---|---|---|
| current | 1 | 0.001115 | 0.80% | 98.70% | 3.968 |
| current | 2 | 0.001115 | 1.10% | 98.50% | 4.021 |
| current | 4 | 0.001115 | 0.60% | 98.50% | 4.032 |
| fix_last_above | 1 | 0.1037 | 63.30% | 19.90% | 0.0341 |
| fix_last_above | 2 | 0.1037 | 61.30% | 22.90% | 0.04011 |
| fix_last_above | 4 | 0.1037 | 63.00% | 20.80% | 0.04351 |
| fix_paper_c015 | 1 | 0.15 | 73.00% | 15.00% | 0.02865 |
| fix_paper_c015 | 2 | 0.15 | 71.20% | 17.70% | 0.02338 |
| fix_paper_c015 | 4 | 0.15 | 73.70% | 14.80% | 0.02691 |
| oracle | 1 | 0.15 | 71.40% | 15.70% | 0.02488 |
| oracle | 2 | 0.15 | 73.10% | 15.60% | 0.02607 |
| oracle | 4 | 0.15 | 71.30% | 15.10% | 0.01885 |

## 2. Baseline mode 影响 (paper_fig4 + grad_accum=1)

Baseline 仅在 warmup + CDF 未就绪期间影响采样；adaptive 期由 InfoNoise CDF 接管。
不同 baseline 应在 ok-config 下收敛到相同的 final mass。

| 配置 | baseline | c 最终值 | mass_info_window | KL→target |
|---|---|---|---|---|
| current | logit_normal | 0.001115 | 0.80% | 3.968 |
| current | uniform | 0.001115 | 0.80% | 3.968 |
| fix_last_above | logit_normal | 0.1037 | 63.30% | 0.0341 |
| fix_last_above | uniform | 0.1037 | 63.30% | 0.0341 |
| fix_paper_c015 | logit_normal | 0.15 | 73.00% | 0.02865 |
| fix_paper_c015 | uniform | 0.15 | 73.00% | 0.02865 |
| oracle | logit_normal | 0.15 | 71.40% | 0.02488 |
| oracle | uniform | 0.15 | 71.40% | 0.02488 |

## 3. 关键 finding

### Finding 1：建议 1 (gate pivot bug) 端到端复现

- `current` config 下 c_pivot 最终值 = **0.001115** (paper 报 0.15，差 0.007434×)
- mass_low_quarter = **98.70%**, mass_info_window = **0.80%**, mass_high_quarter = **0.00%**
- **判决**：BUG 端到端复现 (criterion: mass_low_quarter > 70%)

### Finding 2：fix_last_above 修法效果

- c_pivot 最终 = **0.1037** (回到 paper 量级 ✓)
- mass_info_window = **63.30%** (偏离 paper 37-57%)
- KL→target = **0.0341**

### Finding 3：fix_paper_c015 vs fix_last_above（谁更接近 oracle）

- KL(oracle → target) = 0.02488（应近 0；mock sample 噪声决定下限）
- KL(fix_paper_c015 → target) = **0.02865**
- KL(fix_last_above → target) = **0.0341**
- 更接近 oracle：**fix_paper_c015**

### Finding 4：fix_last_above 跨 mmse 形状的 robustness

- **paper_fig4**: c=0.1037, mass_info_window=63.30%, mass_low=19.90%
- **unimodal_log**: c=1.715, mass_info_window=80.10%, mass_low=0.10%
- **bimodal_log**: c=0.4698, mass_info_window=95.70%, mass_low=1.30%
- **monotone_decay**: c=0.007779, mass_info_window=0.10%, mass_low=99.80%

- **退化**：在 ['monotone_decay'] 上 mass_info_window < 20%；原因：当 mmse 单调递减（monotone_decay）时 1/σ³ tail 与 mmse 同向衰减，r_norm 在 log-σ 上下降平缓，above 区域延伸到低 σ 端，`last_above` 仍落在低 σ

### Finding 5：X1 协同效应（grad_accum 影响）

- **current**: mass_info_window ga1: 0.80% → ga2: 1.10% → ga4: 0.60%
- **fix_last_above**: mass_info_window ga1: 63.30% → ga2: 61.30% → ga4: 63.00%

- **判决**：fix_last_above 在 grad_accum=4 下 mass_info_window=63.00% (仍 work ✓)

- **注意**：本 verify 用 log-uniform baseline 让所有 bin 都填够，绕过了 X1 的另一半（真实 anima logit_normal_shift=3 baseline 在低 σ 几乎不填 bin → n_count.min()=0 → refresh 永远 skip）。该 X1 component 需要单独 verify。


## 4. 跟 paper §5 报告值的偏离量化

Paper CIFAR 报告：c ≈ 0.15 (Eq 87, §5)，info window 占采样 mass 37-57% (Fig 4 B)。

| 配置 | mean c (final) | c / 0.15 | mean mass_info | 偏离 paper |
|---|---|---|---|---|
| current | 0.001115 | 0.007434 | 0.80% | low by 36.2% |
| fix_last_above | 0.1037 | 0.6913 | 63.30% | high by 6.3% |
| fix_paper_c015 | 0.15 | 1 | 73.00% | high by 16.0% |
| oracle | 0.15 | 1 | 71.40% | high by 14.4% |

## 5. 推荐：哪种 fix 应该落地

### 推荐：默认走 `fix_paper_c015`，escape hatch 字段允许用户覆盖

理由：

1. **current 端到端复现 bug**：mass_low_quarter = 98.70% on paper_fig4 (论文 Algorithm 1 Eq 87 + §B.6 Θ(σ⁻¹) tail 警告对齐) — InfoNoise 实际未生效
2. **fix_paper_c015 更稳健**：fix_last_above 在 ['monotone_decay'] 上退化（mass_info_window < 20%），原因详见 Finding 4。fix_paper_c015 把 c 钉到 paper CIFAR 值，对 1/σ³ tail 形状最 worst-case 时仍有保底
3. **跨 mmse 平均 KL**：fix_paper_c015=0.04093 vs fix_last_above=0.7673
4. **escape hatch**：schema 加 `infonoise_gate_pivot_c: float = 0.15`（默认 paper 值），用户可改 0 走 dynamic `fix_last_above`，或填别的值定制 c
5. **不破坏现有 test**：`tests/test_infonoise.py` oracle 只测 CDF 单调 + 端值，不测 c 实际数值；patch 落地无 test breakage。建议补 `test_gate_pivot_not_pinned_to_sigma_min` (4 个 mmse profile 都断言 c >> σ_min) 防回归
6. **monotone_decay edge case**：4 个配置在 monotone_decay 上 mass_info 都 < 20%，因为该 mmse 形状下 1/σ³ tail 与 mmse 同向衰减，gate 单独修不了 — 这是 P0-4 (Jacobian σ³→σ²) 的辖区，不应该归到 P0-5 (gate pivot)。建议 P0-4 + P0-5 同 PR

## 附录 A：全组合 final 指标表

| run_id | c_pivot | mass_low | mass_info | mass_high | KL | refresh_status |
|---|---|---|---|---|---|---|
| `current__bimodal_log__ga1__logit_normal` | 0.001115 | 55.60% | 23.90% | 0.00% | 1.499 | ok |
| `current__bimodal_log__ga1__uniform` | 0.001115 | 55.60% | 23.90% | 0.00% | 1.499 | ok |
| `current__bimodal_log__ga2__logit_normal` | 0.001115 | 57.90% | 22.40% | 0.00% | 1.618 | ok |
| `current__bimodal_log__ga2__uniform` | 0.001115 | 57.90% | 22.40% | 0.00% | 1.618 | ok |
| `current__bimodal_log__ga4__logit_normal` | 0.001115 | 59.00% | 23.10% | 0.00% | 1.575 | ok |
| `current__bimodal_log__ga4__uniform` | 0.001115 | 59.00% | 23.10% | 0.00% | 1.575 | ok |
| `current__monotone_decay__ga1__logit_normal` | 0.001115 | 100.00% | 0.00% | 0.00% | 1.391 | ok |
| `current__monotone_decay__ga1__uniform` | 0.001115 | 100.00% | 0.00% | 0.00% | 1.391 | ok |
| `current__monotone_decay__ga2__logit_normal` | 0.001115 | 100.00% | 0.00% | 0.00% | 1.381 | ok |
| `current__monotone_decay__ga2__uniform` | 0.001115 | 100.00% | 0.00% | 0.00% | 1.381 | ok |
| `current__monotone_decay__ga4__logit_normal` | 0.001115 | 100.00% | 0.00% | 0.00% | 1.392 | ok |
| `current__monotone_decay__ga4__uniform` | 0.001115 | 100.00% | 0.00% | 0.00% | 1.392 | ok |
| `current__paper_fig4__ga1__logit_normal` | 0.001115 | 98.70% | 0.80% | 0.00% | 3.968 | ok |
| `current__paper_fig4__ga1__uniform` | 0.001115 | 98.70% | 0.80% | 0.00% | 3.968 | ok |
| `current__paper_fig4__ga2__logit_normal` | 0.001115 | 98.50% | 1.10% | 0.00% | 4.021 | ok |
| `current__paper_fig4__ga2__uniform` | 0.001115 | 98.50% | 1.10% | 0.00% | 4.021 | ok |
| `current__paper_fig4__ga4__logit_normal` | 0.001115 | 98.50% | 0.60% | 0.00% | 4.032 | ok |
| `current__paper_fig4__ga4__uniform` | 0.001115 | 98.50% | 0.60% | 0.00% | 4.032 | ok |
| `current__unimodal_log__ga1__logit_normal` | 0.001115 | 56.00% | 29.00% | 0.00% | 2.474 | ok |
| `current__unimodal_log__ga1__uniform` | 0.001115 | 56.00% | 29.00% | 0.00% | 2.474 | ok |
| `current__unimodal_log__ga2__logit_normal` | 0.001115 | 58.80% | 28.20% | 0.00% | 2.662 | ok |
| `current__unimodal_log__ga2__uniform` | 0.001115 | 58.80% | 28.20% | 0.00% | 2.662 | ok |
| `current__unimodal_log__ga4__logit_normal` | 0.001115 | 59.10% | 28.70% | 0.00% | 2.591 | ok |
| `current__unimodal_log__ga4__uniform` | 0.001115 | 59.10% | 28.70% | 0.00% | 2.591 | ok |
| `fix_last_above__bimodal_log__ga1__logit_normal` | 0.4698 | 1.30% | 95.70% | 0.00% | 0.4453 | ok |
| `fix_last_above__bimodal_log__ga1__uniform` | 0.4698 | 1.30% | 95.70% | 0.00% | 0.4453 | ok |
| `fix_last_above__bimodal_log__ga2__logit_normal` | 0.4698 | 2.20% | 92.90% | 0.00% | 0.4072 | ok |
| `fix_last_above__bimodal_log__ga2__uniform` | 0.4698 | 2.20% | 92.90% | 0.00% | 0.4072 | ok |
| `fix_last_above__bimodal_log__ga4__logit_normal` | 0.4698 | 1.70% | 94.30% | 0.00% | 0.4087 | ok |
| `fix_last_above__bimodal_log__ga4__uniform` | 0.4698 | 1.70% | 94.30% | 0.00% | 0.4087 | ok |
| `fix_last_above__monotone_decay__ga1__logit_normal` | 0.007779 | 99.80% | 0.10% | 0.00% | 0.6339 | ok |
| `fix_last_above__monotone_decay__ga1__uniform` | 0.007779 | 99.80% | 0.10% | 0.00% | 0.6339 | ok |
| `fix_last_above__monotone_decay__ga2__logit_normal` | 0.007779 | 99.50% | 0.20% | 0.00% | 0.6182 | ok |
| `fix_last_above__monotone_decay__ga2__uniform` | 0.007779 | 99.50% | 0.20% | 0.00% | 0.6182 | ok |
| `fix_last_above__monotone_decay__ga4__logit_normal` | 0.007779 | 99.70% | 0.10% | 0.00% | 0.6409 | ok |
| `fix_last_above__monotone_decay__ga4__uniform` | 0.007779 | 99.70% | 0.10% | 0.00% | 0.6409 | ok |
| `fix_last_above__paper_fig4__ga1__logit_normal` | 0.1037 | 19.90% | 63.30% | 0.00% | 0.0341 | ok |
| `fix_last_above__paper_fig4__ga1__uniform` | 0.1037 | 19.90% | 63.30% | 0.00% | 0.0341 | ok |
| `fix_last_above__paper_fig4__ga2__logit_normal` | 0.1037 | 22.90% | 61.30% | 0.00% | 0.04011 | ok |
| `fix_last_above__paper_fig4__ga2__uniform` | 0.1037 | 22.90% | 61.30% | 0.00% | 0.04011 | ok |
| `fix_last_above__paper_fig4__ga4__logit_normal` | 0.1037 | 20.80% | 63.00% | 0.00% | 0.04351 | ok |
| `fix_last_above__paper_fig4__ga4__uniform` | 0.1037 | 20.80% | 63.00% | 0.00% | 0.04351 | ok |
| `fix_last_above__unimodal_log__ga1__logit_normal` | 1.715 | 0.10% | 80.10% | 0.00% | 1.956 | ok |
| `fix_last_above__unimodal_log__ga1__uniform` | 1.715 | 0.10% | 80.10% | 0.00% | 1.956 | ok |
| `fix_last_above__unimodal_log__ga2__logit_normal` | 1.715 | 0.10% | 80.00% | 0.00% | 1.84 | ok |
| `fix_last_above__unimodal_log__ga2__uniform` | 1.715 | 0.10% | 80.00% | 0.00% | 1.84 | ok |
| `fix_last_above__unimodal_log__ga4__logit_normal` | 1.715 | 0.20% | 80.80% | 0.00% | 1.892 | ok |
| `fix_last_above__unimodal_log__ga4__uniform` | 1.715 | 0.20% | 80.80% | 0.00% | 1.892 | ok |
| `fix_paper_c015__bimodal_log__ga1__logit_normal` | 0.15 | 2.70% | 90.30% | 0.00% | 0.03396 | ok |
| `fix_paper_c015__bimodal_log__ga1__uniform` | 0.15 | 2.70% | 90.30% | 0.00% | 0.03396 | ok |
| `fix_paper_c015__bimodal_log__ga2__logit_normal` | 0.15 | 3.70% | 88.30% | 0.00% | 0.02415 | ok |
| `fix_paper_c015__bimodal_log__ga2__uniform` | 0.15 | 3.70% | 88.30% | 0.00% | 0.02415 | ok |
| `fix_paper_c015__bimodal_log__ga4__logit_normal` | 0.15 | 3.50% | 88.60% | 0.00% | 0.02432 | ok |
| `fix_paper_c015__bimodal_log__ga4__uniform` | 0.15 | 3.50% | 88.60% | 0.00% | 0.02432 | ok |
| `fix_paper_c015__monotone_decay__ga1__logit_normal` | 0.15 | 72.80% | 19.10% | 0.00% | 0.07673 | ok |
| `fix_paper_c015__monotone_decay__ga1__uniform` | 0.15 | 72.80% | 19.10% | 0.00% | 0.07673 | ok |
| `fix_paper_c015__monotone_decay__ga2__logit_normal` | 0.15 | 73.20% | 18.80% | 0.00% | 0.07675 | ok |
| `fix_paper_c015__monotone_decay__ga2__uniform` | 0.15 | 73.20% | 18.80% | 0.00% | 0.07675 | ok |
| `fix_paper_c015__monotone_decay__ga4__logit_normal` | 0.15 | 73.90% | 18.40% | 0.00% | 0.07874 | ok |
| `fix_paper_c015__monotone_decay__ga4__uniform` | 0.15 | 73.90% | 18.40% | 0.00% | 0.07874 | ok |
| `fix_paper_c015__paper_fig4__ga1__logit_normal` | 0.15 | 15.00% | 73.00% | 0.00% | 0.02865 | ok |
| `fix_paper_c015__paper_fig4__ga1__uniform` | 0.15 | 15.00% | 73.00% | 0.00% | 0.02865 | ok |
| `fix_paper_c015__paper_fig4__ga2__logit_normal` | 0.15 | 17.70% | 71.20% | 0.00% | 0.02338 | ok |
| `fix_paper_c015__paper_fig4__ga2__uniform` | 0.15 | 17.70% | 71.20% | 0.00% | 0.02338 | ok |
| `fix_paper_c015__paper_fig4__ga4__logit_normal` | 0.15 | 14.80% | 73.70% | 0.00% | 0.02691 | ok |
| `fix_paper_c015__paper_fig4__ga4__uniform` | 0.15 | 14.80% | 73.70% | 0.00% | 0.02691 | ok |
| `fix_paper_c015__unimodal_log__ga1__logit_normal` | 0.15 | 1.30% | 96.30% | 0.00% | 0.02438 | ok |
| `fix_paper_c015__unimodal_log__ga1__uniform` | 0.15 | 1.30% | 96.30% | 0.00% | 0.02438 | ok |
| `fix_paper_c015__unimodal_log__ga2__logit_normal` | 0.15 | 1.80% | 94.60% | 0.00% | 0.02193 | ok |
| `fix_paper_c015__unimodal_log__ga2__uniform` | 0.15 | 1.80% | 94.60% | 0.00% | 0.02193 | ok |
| `fix_paper_c015__unimodal_log__ga4__logit_normal` | 0.15 | 1.20% | 95.90% | 0.00% | 0.02352 | ok |
| `fix_paper_c015__unimodal_log__ga4__uniform` | 0.15 | 1.20% | 95.90% | 0.00% | 0.02352 | ok |
| `oracle__bimodal_log__ga1__logit_normal` | 0.15 | 2.90% | 88.80% | 0.00% | 0.02973 | ok |
| `oracle__bimodal_log__ga1__uniform` | 0.15 | 2.90% | 88.80% | 0.00% | 0.02973 | ok |
| `oracle__bimodal_log__ga2__logit_normal` | 0.15 | 2.60% | 90.50% | 0.00% | 0.03433 | ok |
| `oracle__bimodal_log__ga2__uniform` | 0.15 | 2.60% | 90.50% | 0.00% | 0.03433 | ok |
| `oracle__bimodal_log__ga4__logit_normal` | 0.15 | 2.50% | 90.20% | 0.00% | 0.02371 | ok |
| `oracle__bimodal_log__ga4__uniform` | 0.15 | 2.50% | 90.20% | 0.00% | 0.02371 | ok |
| `oracle__monotone_decay__ga1__logit_normal` | 0.15 | 74.10% | 17.70% | 0.00% | 0.08269 | ok |
| `oracle__monotone_decay__ga1__uniform` | 0.15 | 74.10% | 17.70% | 0.00% | 0.08269 | ok |
| `oracle__monotone_decay__ga2__logit_normal` | 0.15 | 71.30% | 20.10% | 0.00% | 0.07809 | ok |
| `oracle__monotone_decay__ga2__uniform` | 0.15 | 71.30% | 20.10% | 0.00% | 0.07809 | ok |
| `oracle__monotone_decay__ga4__logit_normal` | 0.15 | 73.30% | 18.90% | 0.00% | 0.07919 | ok |
| `oracle__monotone_decay__ga4__uniform` | 0.15 | 73.30% | 18.90% | 0.00% | 0.07919 | ok |
| `oracle__paper_fig4__ga1__logit_normal` | 0.15 | 15.70% | 71.40% | 0.00% | 0.02488 | ok |
| `oracle__paper_fig4__ga1__uniform` | 0.15 | 15.70% | 71.40% | 0.00% | 0.02488 | ok |
| `oracle__paper_fig4__ga2__logit_normal` | 0.15 | 15.60% | 73.10% | 0.00% | 0.02607 | ok |
| `oracle__paper_fig4__ga2__uniform` | 0.15 | 15.60% | 73.10% | 0.00% | 0.02607 | ok |
| `oracle__paper_fig4__ga4__logit_normal` | 0.15 | 15.10% | 71.30% | 0.00% | 0.01885 | ok |
| `oracle__paper_fig4__ga4__uniform` | 0.15 | 15.10% | 71.30% | 0.00% | 0.01885 | ok |
| `oracle__unimodal_log__ga1__logit_normal` | 0.15 | 1.00% | 96.30% | 0.00% | 0.01986 | ok |
| `oracle__unimodal_log__ga1__uniform` | 0.15 | 1.00% | 96.30% | 0.00% | 0.01986 | ok |
| `oracle__unimodal_log__ga2__logit_normal` | 0.15 | 1.00% | 96.50% | 0.00% | 0.02251 | ok |
| `oracle__unimodal_log__ga2__uniform` | 0.15 | 1.00% | 96.50% | 0.00% | 0.02251 | ok |
| `oracle__unimodal_log__ga4__logit_normal` | 0.15 | 1.00% | 96.70% | 0.00% | 0.01998 | ok |
| `oracle__unimodal_log__ga4__uniform` | 0.15 | 1.00% | 96.70% | 0.00% | 0.01998 | ok |
