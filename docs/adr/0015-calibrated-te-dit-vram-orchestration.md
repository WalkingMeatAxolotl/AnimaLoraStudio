# 0015 — 用实测峰值协调测试出图的 TE / DiT 驻留

**状态**：Accepted
**日期**：2026-07-20
**决策者**：项目维护者

## 背景

Krea 2 在第二个测试出图任务遇到新 prompt 时，需要重新把文本编码器（TE）放入
GPU。此时 DiT 已由上一个任务常驻，TE 与 DiT 同驻会把阶段峰值抬高；Windows WDDM
还可能在 CUDA 报 OOM 之前就开始迁移显存，使加载时间陡增。

一次 32 GiB 实测中，空闲显存预算显示 17.8 GiB，7.0 GiB 的 FP8 TE 理论上可与
DiT 同驻，但第二次 TE 加载从首次约 16 秒增加到 190 秒以上。20 ms 整卡采样在首次
TE、DiT/sample、VAE 阶段分别测得 10,434、17,926、18,126 MiB；第二次 TE 尚未完成
就达到 21,250 MiB。说明“尚未 OOM”不能作为默认策略的安全标准。

## 候选方案

1. 三档都强制 DiT 为 TE 让位。峰值最低，但高显存卡也承担不必要的来回搬运。
2. 只看 `torch.cuda.mem_get_info()` 能否容纳 TE。实现简单，但无法避开 WDDM 的
   residency/paging 性能悬崖。
3. 首轮实测 TE 峰值，后续按实测增量和显卡容量预留决定是否让位；省显存档始终
   让位，性能档始终同驻。

## 决策

采用方案 3：

- 第一次 Krea 2 任务从 TE 加载前到 prompt 编码后，用 CUDA allocator 记录
  `load + encode` 峰值增量，按文本模型缓存键保存在 daemon 生命周期内。
- `auto` 在 prompt cache miss 时，用该校准值（尚无校准时使用 FP8/full fallback）
  与当前 CUDA 空闲量做预算。预算要求峰值后仍保留至少整卡 40%（最低 1 GiB）的
  residency 余量；不满足时先把 DiT 和动态 adapter 移至 CPU，释放 TE 后再恢复。
- `save_vram` 在 cache miss 时总是执行上述顺序化；`performance` 总是允许同驻。
- prompt LRU 全命中时不加载 TE，因此三档都不为此移动 DiT。
- 日志输出预算来源、空闲量、TE 增量、预留量及最终决策，便于与整卡监视脚本对照。

## 理由

校准消除了不同 TE 精度、后端和权重类型带来的固定估算误差；40% 余量不是 CUDA OOM
边界，而是根据 Windows 实测增加的 residency 防线。它让当前 32 GiB 机器的默认档
选择稳定的顺序化路径，同时保留 48/64 GiB 等高显存设备在余量充足时同驻的机会。

## 后果

- 32 GiB 设备上的 `auto` 通常与 `save_vram` 一样在第二轮 cache miss 时搬运 DiT，
  以搬运时间换更低峰值和避免 WDDM 抖动；两者在更大显存设备上仍可能不同。
- CUDA allocator 校准只覆盖本 daemon 的 PyTorch 分配；桌面、浏览器、其他 CUDA
  进程和驱动保留必须由 NVML/`nvidia-smi` 整卡采样衡量。两种数值用途不同，日志中
  不把它们表述为同一口径。
- daemon 重启后需重新校准一次；首次任务本来就要求 TE 先行，不增加模型搬运。

## 验证

最终 40% 阈值在同一张 32 GiB 卡上复测：首轮 allocator 校准为 6.4 GiB；第二轮
记录 `free=17.8 GiB, TE=6.4 GiB, reserve=12.7 GiB` 并选择 DiT 让位。20 ms
整卡采样测得第二轮纯 TE 阶段峰值 10,065 MiB、DiT 恢复阶段 15,615 MiB、全流程
最高 17,745 MiB（出现在 VAE decode），未再出现约 21–22 GiB 的 TE/DiT 同驻峰值。
第二轮任务 16.6 秒完成；此前同驻实验仅 TE 加载就超过 190 秒。

## WDDM 瞬时 OOM 恢复

后续实机测试发现：DiT 从 CPU 恢复到 GPU 后，第一次 forward 偶发 OOM。此时
PyTorch 约分配 13 GiB、整卡仍空闲 17 GiB；完全相同的 prompt、分辨率和 LoRA 在
下一任务成功。因此它不属于真实容量不足，也不符合普通 caching allocator 碎片。

恢复路径现在会在采样前同步 CUDA 搬运。生成流程还会仅对 CUDA OOM 做一次有界
重试：释放可能已加载的 VAE，回收失败 forward 的临时张量，清理 CUDA allocator
和 cuBLAS 状态，恢复同一个 Python/Torch seed，然后再采样一次。其他错误不重试；
第二次 CUDA OOM 仍原样上报。

## 参考

- `runtime/anima_daemon.py`
- `tools/vram_trace.py`
- `tests/test_daemon_prompt_precache.py`
