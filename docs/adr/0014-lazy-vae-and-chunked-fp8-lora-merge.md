# 0014 — 测试出图延迟加载 VAE，并分块合并 FP8 LoRA

**状态**：Accepted
**日期**：2026-07-20
**决策者**：项目维护者

## 背景

Krea 2 FP8 测试出图的 LoRA merge 峰值高于采样峰值，成为 24–32 GB 显卡的
阻塞点。旧顺序在 DiT 和 LoRA merge 前已经把 VAE 放进 GPU；普通 rank-32
LoRA 虽然因子很小，`up @ down` 却会为最大 `tproj.1` 层物化完整
`36864×6144` delta（FP32 为 864 MiB）。

## 候选方案

1. **只把 merge 临时精度降到 BF16**：实现简单、计算较快，但同机 A/B 的
   整卡峰值没有稳定下降，且不再保证 ComfyUI FP32 merge 数值一致。
2. **动态 adapter，不修改底模权重**：消除 merge 峰值，但采样期间需常驻
   adapter 与中间计算，抬高每一步显存；不符合本项目优先降低完整流程峰值的目标。
3. **普通 LoRA 按输出行分块 merge，并延迟 VAE 加载**：保留 FP32 语义，
   delta 工作集从整层变为 `chunk_rows×in`；VAE 仅在采样结束进入 decode 时加载。
4. **CPU merge 或双遍流式量化**：峰值更低，但 merge 时间和 CPU/RAM 压力
   明显增加，留作未来的极限省显存档。

## 决策

- 测试出图的 VAE 改为 decode 边界惰性加载。`auto` 和 `save_vram` 在每张图
  decode 后把完整 VAE（含未注册为 module buffer 的 mean/std）移到 CPU RAM；
  `performance` 保持 GPU 常驻。
- Krea 2 FP8 的普通 Linear LoRA 默认以 1024 输出行为一块计算 dense delta。
  LoHa 和 LoKr 暂不改变算法，避免未经对拍便改变其复合中间张量顺序。
- FP32/BF16 merge 精度设置保持独立；分块不改变用户选择的计算 dtype。

## 理由

在 RTX 5090、Krea 2 Turbo FP8、两份 rank-32 普通 LoRA 上，对全部 264 个
merge 层逐层执行 full/chunked A/B：

| 指标 | 完整 delta | 1024 行分块 | 变化 |
|---|---:|---:|---:|
| 最终 FP8 权重 + scale | — | 264/264 层 bit 一致 | 0 字节变化 |
| CUDA 单层峰值 allocated | 3037.4 MiB | 1832.1 MiB | -1205.2 MiB |
| 两份 LoRA 累计 merge 时间 | 10.477 s | 10.179 s | -2.8% |

对拍覆盖 13,138,657,304 个存储字节，变化数、最大/平均反量化误差均为 0。
真实 Studio 任务 #1564 也验证了生产顺序：merge 完成后采样，采样结束才加载
VAE；整卡 20 ms 监视的阶段峰值为 merge 18,699 MiB、sample 18,203 MiB、
decode 18,407 MiB，decode 清理后回落到 16,783 MiB。20 ms NVML 采样可能漏掉
只存活数毫秒的完整 delta，因此算法 A/B 以 CUDA allocator 峰值为主要指标。

## 后果

- 普通 LoRA 数量主要继续增加 merge 时间，而不会叠加完整 dense delta 峰值。
- 默认/省显存策略的第二张图需要把约 250 MiB VAE 从 RAM 搬回 GPU；这是用
  搬运时间换采样期零 VAE 显存。性能优先档不承担该搬运。
- LoHa/LoKr 仍可能物化多个完整中间矩阵，是下一阶段的峰值优化对象。
- `DeferredVAE` 依赖 VAE wrapper 的 `to(device)` 同时移动 model 与 mean/std；
  新 VAE wrapper 必须遵守该契约。

## 参考

- `runtime/training/families/krea2/lora_fp8_merge.py`
- `studio/services/inference/core.py`
- `tools/benchmark_lora_merge_chunks.py`
- `tmp/krea2-vram-chunk1024-lazy-vae.csv`（本机实验原始记录，不纳入发行包）
