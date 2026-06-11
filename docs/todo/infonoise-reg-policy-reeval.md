# InfoNoise reg 集 record 策略重评估清单

**创建于** 2026-06-05
**触发** PR #216 commit `cfbfd218` 阈值方案的 follow-up；用户在 review 中发现 `loss_weight >= 0.99` 阈值在 `reg_weight=1.0` 边界有内部矛盾，三方算法分析后改为按 `is_reg` flag 硬过滤。
**当前状态** 🟢 路线 1（is_reg hard exclude）已落地于 `runtime/training/loop.py:141-152` + `runtime/training/dataset.py`。本文档记录路线 2（loss_weight soft weighting）延后的依据 + 何时重评估。

---

## 当前实现（路线 1：is_reg 硬过滤）

```python
# runtime/training/loop.py
if "is_reg" in batch:
    _main_mask = ~batch["is_reg"].to(t.device)
    if _main_mask.any():
        ctx.timestep_sampler.record(t.detach()[_main_mask], _raw_mse[_main_mask])
else:
    ctx.timestep_sampler.record(t.detach(), _raw_mse)
```

- `MergedDataset.__getitem__` 给 main 样本写 `is_reg=False`，reg 样本写 `is_reg=True`
- `collate_fn` / `collate_fn_cached` 透传 `is_reg` 为 `torch.bool` tensor
- InfoNoise 的 FIFO/EMA 只看 main 样本的 (t, raw_mse)，CDF 学 mmse_main(t)
- reg 样本仍照常进梯度，按 `reg_weight` 加权（loop.py:154-156 那段不动）
- 对 `reg_weight` 任何取值（0.3 / 0.7 / 1.0 / 任意）都是同一种 mask 行为

## 备选路线 2（已延后：loss_weight soft 加权）

不过滤，按 loss_weight 加权 record 进 FIFO/EMA：

```python
ctx.timestep_sampler.record(t, raw_mse, weights=batch["loss_weight"])
# InfoNoise 内部 bucket aggregation 改为 weighted: sum(w*mse) / sum(w)
```

数学性质：对真实目标 `L = E_main[ℓ] + λ E_reg[ℓ]` 的 min-variance importance sampling proposal（Katharopoulos-Fleuret 等），λ 连续 unbiased。

## 选路线 1 而非路线 2 的依据（2026-06-05 三 agent 分析）

三个独立 lens：

| Lens | Q1 | 置信度 |
|---|---|---|
| 信息论 / I-MMSE | exclude_all | 72% |
| weighted ERM / IS | soft_weight_by_loss_weight | 78% |
| 训练动力学 / 用户目标 | exclude_all | 78% |

**多数（2/3）支持 hard exclude**。三方一致拒绝 `loss_weight >= 0.99` 旧方案（A: type error / B: arbitrary θ / C: identity vs weight 错配）。

**核心争点**：InfoNoise 工具的 job 是什么？
- 路线 1（A+C）："main 分布的 MMSE 估计器"，reg 是 means 不是 ends
- 路线 2（B）："你写下的 weighted ERM 的 min-variance IS proposal"

路线 1 胜出的现实依据：
1. InfoNoise 论文契约（I-MMSE）严格依赖单分布；mixture 推广未证（Lens B 自承盲点 #1）
2. 当前 AnimaLoraStudio 用户群 99% 是单角色 / 单画风 LoRA + 通用图 reg，reg 是辅助 prior 而非 co-training target
3. `reg_weight=1.0` 在社区实际语义是"强正则"，不是"multi-task 平衡训练"
4. 工程成本低（1 字段 + 1 行 mask）；路线 2 要改 InfoNoise sampler 内部 FIFO/EMA 累加器

## 重评估触发条件

**满足以下任一条件时重新评估是否切到路线 2 或混合方案**：

1. **数据规模触发**：用户群中 reg 集与 main 集"分布相似"的比例显著上升（如出现大量"用同 booru tag 范围作 reg"或"同画师其他作品作 reg"的训练 config）。判定方法：在社区调研或 wandb 上抽样近 N 个 reg 集，看其 caption 分布跟 main 集的 KL 距离分布。

2. **多 main 分布场景出现**：multi-concept LoRA / 同时训多个角色 / 同时训角色+画风 等场景成为主流。这时"单 main 分布"假设破裂，路线 1 的"hard exclude reg"也不再够用，需要 per-distribution CDF（每个分布维护一个 entropy curve）或路线 2 的 weighted aggregation。

3. **`reg_weight ≥ 0.7` 高占比**：社区调研发现用户大量配 `reg_weight ≥ 0.7`（接近 1.0），说明用户语义已经从"辅助 prior"转向"co-training"，此时路线 2 的 λ-连续 semantics 更贴合用户意图。

4. **InfoNoise 论文 / 学术界进展**：出现 I-MMSE 在 mixture distribution 下的正式推广（如 mixture-MMSE + weighted ERM 联合分析），或 Katharopoulos-Fleuret 风格 IS 与扩散 schedule 优化的对接论文。这时路线 2 有了缺失的数学保证。

5. **实测路线 1 sub-optimal**：用户反馈或对照实验显示，开 reg 集 + InfoNoise 时收敛速度 / 最终质量明显劣于路线 2 模拟（即便 reg 集分布不同）。需要 A/B 实验设计（同 config 跑两套 record 策略对比 main 任务 metric）。

## 重评估时的具体动作

- **判定继续路线 1**：本文档归档到 `docs/todo/archive/`，CHANGELOG 加一条说明重评估结论 + 数据
- **判定切换路线 2**：
  1. InfoNoise sampler 加 `record(t, mse, weights=None)` 接口
  2. FIFO bucket 改 weighted streaming average（注意数值稳定性、低权重 sample 不被掩盖）
  3. EMA 累加器同步加权
  4. loop.py:141 改为 `ctx.timestep_sampler.record(t, raw_mse, weights=batch["loss_weight"])`
  5. dataset.py 的 `is_reg` 字段保留（多 main 分布场景可能仍需要 identity label）
  6. training-tips.md reg 行重写
  7. 加测试：(a) λ=0 时 reg 样本对 schedule 影响→0；(b) λ=1 时 reg 跟 main 平权进 record；(c) weighted streaming average 数值稳定性
- **判定混合方案**（per-distribution CDF）：单独开 ADR

## 引用 / 上下文

- 触发本次讨论的 commit: `cfbfd218` (PR #216 follow-up to e7ac3e3)
- 3 agent workflow run: `wf_42e355a5-f01`（2026-06-05）
- 完整三方分析 transcript: `tmp/infonoise/` 下任何归档 + memory `infonoise-reg-policy-open.md`
- 当前实现: `runtime/training/loop.py:141-152` + `runtime/training/dataset.py` MergedDataset / collate_fn
- 相关测试: `tests/test_infonoise.py::test_record_accepts_partial_batch_after_reg_mask`
- 关联 memory: `infonoise_reg_policy_open.md`、`reg_dreambooth_alignment.md`（reg 集在现代 DiT 训练里整体定位）

## 复查节奏

- **半年一次**（2026-12-05 / 2027-06-05 ...）：抽样近 6 个月社区 / wandb 的 reg 集 config，看分布是否仍以"通用图 reg"为主
- **触发性复查**：上面 5 个触发条件任一命中
- **3 年仍无 multi-concept LoRA 主流化**（2029-06-05 后）：路线 1 可视为长期稳定方案，本文档可降级为 archive
