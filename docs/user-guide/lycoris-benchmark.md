# LyCORIS eager 基准（R1，开发者 CLI）

此工具为 issue #567 的 eager 基线提供可重放证据，不是性能优化或训练质量评估。
固定 **LyCORIS 4.0.0 / resolved backend `torch`**，拒绝 auto/compile/Triton/TileLang，
不安装依赖、不修改生产 dtype/bypass/算法。生产 CLI 默认不采集基准数据；生产训练对
显式 optional backend 的启动前检查见下一节。

## 生产训练的 optional backend preflight（R2）

未设置 `LYCORIS_KERNEL_BACKEND` 时仍固定为 `torch`，不会启动 probe 子进程。显式设置
`auto`、`triton`、`tilelang` 或 `compile` 时，LoRA/LoKr/LoHa 训练会在加载大 DiT 和父进程
首次导入 LyCORIS 前启动一次隔离子进程：使用当前训练 dtype、算法和 DoRA/FP8 路径，
通过生产 `LycorisAdapter` 执行一个有界的 64×64 CUDA forward/backward，并检查输出、
输入梯度及全部 adapter 梯度有限。`tlora_use_ortho=false` 的兼容 T-LoRA 也走相同启动
检查，避免不可用 backend 在 LyCORIS 注入日志解析时才报错；默认 Ortho T-LoRA 不需要。

probe 有 120 秒硬超时，不写跨进程缓存。缺少 backend/DLL、导入或编译失败、CUDA
不可用、异常/非有限梯度、超时或无效 worker 响应都会令**本次训练进程**把 backend 改回
`torch`，并只记录一条包含 requested/configured/reason 的汇总诊断；子进程 stderr 不复制
到训练日志。普通训练随后继续使用 eager 路径，不会等到第一批数据才暴露已知的
`TritonMissing`。

当前 Ortho 与默认的 Ortho T-LoRA 使用自己的 eager 数学路径，不导入 LyCORIS，因而
不注册这项 probe。兼容 T-LoRA 虽使用 LyCORIS wrapper，但自定义 `make_weight` 不调用 v4
fused dispatcher；其 probe 只验证 backend 可解析以及兼容注入/F/B，不代表获得 kernel
加速。所有 probe 都只证明该代表性调用在当前 Python/Torch/CUDA/设备上可执行，
**不证明所有真实 shape 都支持，也不证明每个训练算子实际使用 fused kernel**；真实训练
中的 shape、OOM 或运行时错误仍会正常抛出，不能被 probe 吞掉。R2 不负责安装 optional
kernel 依赖，安装和支持矩阵属于 roadmap R3/R4。

## 三种入口，不混称训练

- `layer`：Anima 2048/5120 架构派生的 **合成单层** forward/backward。用当前 `Block`
  在 meta 上发现 Linear，随后一次仅实化一个层；复用族 `ANIMA_PRESET` 与
  `LycorisAdapter`，不加载模型资产。代表组为 self q（方形 attention）、cross k
  （1024→C 的 context）、MLP layer1（C→4C）、layer2（4C→C）。不声称测过所有层。
  self k/v/output 与 cross q/output 的等形层未逐个测量，AdaLN/完整 attention 未测。
- `replay`：校验某个合成 reference，精确恢复 input、base、完整 adapter state、
  cotangent，再比较 output、input-gradient、每个 trainable adapter gradient。
- `train`：**Anima LoRA 或 LoKr + 工具生成 RGB 图片和公开合成 captions**，通过
  原 `anima_train.run_training` 唯一 pipeline 训练。不是自写训练循环，且不接用户数据。
  Krea2/FP8、原生 TLoRA、DoRA 新矩阵和任何 optional kernel 执行不在 R1 范围。
  项目已有 TLoRA/DoRA/bypass/resume regression 仍须作为验收门禁。

## CPU 流程验证

以下使用 PowerShell；Python 必须用仓库 venv。输出是**仓库外全新目录**，不能复用：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:LYCORIS_KERNEL_BACKEND = 'torch'
$cpu = Join-Path $env:TEMP ('lycoris-cpu-' + [guid]::NewGuid())
.\venv\Scripts\python.exe tools/benchmark_lycoris.py layer `
  --config tools/benchmark_configs/lycoris_cpu_smoke_v1.json --device cpu --output $cpu
.\venv\Scripts\python.exe tools/benchmark_lycoris.py replay --device cpu `
  --manifest "$cpu/public/lora-0-layer-0/reference/reference-manifest.json"
# lokr-0-layer-0 和 loha-0-layer-0 同理。
```

CPU profile 是 32→32、`[1,4,32]`，明确标记 `synthetic_smoke`；不能用于真实规模性能结论。
CPU 没有 CUDA kernel/VRAM 数据，相应 metric 为 `null + reason`，不是 0。

## CUDA 合成层（需要另行批准 GPU 预算）

```powershell
.\venv\Scripts\python.exe tools/benchmark_lycoris.py layer `
  --config tools/benchmark_configs/lycoris_eager_v1.json `
  --profile anima-2048 --device cuda --output $env:BENCH_NEW_GPU_WORKSPACE
# 5120 规格使用 --profile anima-5120，仍须另一个全新输出目录。
```

默认 512px：WAN21_F8C16 的 spatial_stride=8、patch=2 → 图像 1024 tokens；
cross k 的 512 text tokens 是**显式合成假设**，不是观测值。MLP 保留真实普通路径
`[B,T,H,W,C]`，attention Linear 是 `[B,N,C]`，不悄悄把布局混为一类。
无 OOM 自动缩形/缩 rank/换 dtype。失败停止后续 repeats 并写 incomplete；已完成数据保留。

## Anima 隔离短训练（需要用户批准资产、工作区及 GPU 预算）

```powershell
.\venv\Scripts\python.exe tools/benchmark_lycoris.py train `
  --config tools/benchmark_configs/lycoris_eager_v1.json --scenario lora --device cuda `
  --transformer $env:BENCH_TRANSFORMER --vae $env:BENCH_VAE `
  --text-encoder $env:BENCH_TEXT_ENCODER --t5-tokenizer $env:BENCH_T5_TOKENIZER `
  --output $env:BENCH_NEW_TRAIN_WORKSPACE
# LoKr 使用 --scenario lokr 和另一全新 workspace。
```

四个本地绑定必须显式给出：transformer/VAE 为 safetensors，TE/T5 为非空本地目录。
原资产仅读取；不读取现有用户图片/caption、不导入 YAML、不接 resume checkpoint。
配置为 version=1 且未知键拒绝；可以在仓库外复制 JSON 修改**受界限校验**的实验数值。
不支持任意 TrainingConfig 字段透传。`--profile` 仅 layer 可用；训练实际形状来自诊断 hooks，
不因 layer profile 名字推断所绑定真实 transformer 的规模。

`lycoris_eager_v1.json` 的 SHORT 配置明确为：

| 项 | 值 |
|---|---|
| 训练场景 | Anima LoRA / LoKr；rank8、alpha4、LoKr factor8 |
| 数据 | PCG64 seed567 生成 8 张 512² RGB 随机像素图 + 2 条固定公开 caption 循环 |
| batch / grad_accum / effective batch | 1 / 1 / 1 |
| 精度 / attention | bf16 / `none`（现有 PyTorch SDPA，具体子 kernel 未观测） |
| checkpoint / block swap | 开 / 0 |
| VAE cache | 开，fresh dataset，在 setup 阶段建立；每个 pass/repeat 单独缓存 |
| optimizer / LR / scheduler | AdamW / 1e-4 / none |
| steps / epochs | 2 warmup + 5 **成功 update**；max_steps=7、epochs 上限2 |
| repeats | 每模式3次、每次新进程 |
| sample_steps / sample_every | 0 / 0（含 step-0 baseline 禁用） |
| periodic save | 全关；原 final save / auto-epoch backup 仍会写，全部隔离 |

每 scenario 有 throughput、F/B、kernel **三个独立模式 × 3 repeats = 9 次 pipeline**，
每次包含模型加载/cache setup/finalize。不是只执行 7 个 step 一次。NaN/跳步可能令
实际尝试批次数更多；epochs 耗尽且未完成 7 个成功更新是 incomplete，不伪造吞吐。
合成像素与 caption 不表示有语义配对，不能评估出图/模型质量或替代真实数据集性能。

### 隔离与公开边界

workspace 必须全新、父目录存在、在仓库/资产树之外；拒绝既有目录、source/models/
`studio_data`/venv 的交叠、路径别名、symlink/junction/reparse、资产硬链接和 tokenizer
嵌套绑定。生成数据只写新目录；不复制或链接用户数据。以下布局由工具创建：

```text
workspace/
  public/experiment.json, result.json
  public/<run-id>/result.json
  public/<layer-run-id>/reference/{reference-manifest.json,tensors.safetensors}
  private/<run-id>/run.log, training-args.private.json
  private/<train-run-id>/dataset/, output/, monitor/, samples/, state/, cache/, tmp/, home/
```

实际 VAE cache 写图旁，因此 dataset 本身必须隔离，不能只依赖 cache_dir。
监控路径会推导 samples/state 的 task 根，也绑定在 private run 根。
运行不继承 WandB/Studio task ID/代理/token/PYTHONPATH，禁用 HF 联网和 telemetry、
外部日志、auto-install；子进程缓存、home/temp 均指向 private。
训练 worker 另用 Python audit hook 拒绝 socket connect/DNS 和外部 subprocess/installer。
这**不是 OS sandbox**：信任本地资产与仓库代码，不抵御原生扩展或并发恶意目录替换。
不上传、不自动清理 workspace。确认磁盘空间（真实 epoch state 可很大）后再跑。

只发布 `public/`。训练原始日志、args、monitor、adapter/optimizer state 可能含私有
路径，必须留 private。公开数据仅白名单版本/device、数值、公开 config、结构元数据；
不 wholesale dump args/env/exception text。失败公开 controlled error code，详情在私有日志。
真实训练参数的临时 CPU 对比副本只存在内存，不落盘；不保存真实 activation/gradient。

## 测量口径

- **训练 headline it/s** = measured successful updates / **连续同步 wall**。
  warmup 最后一次成功 update 后开启，最后一次 measured update 后同步关闭。窗口包含
  两边界间的 dataloader fetch、TE/VAE、累积、optimizer、监控和 epoch IO；不含 setup、
  finalize、最后 update 后的尾部 IO（单独报告 loop_tail）。不使用日志 EMA 或倒数均值。
  throughput 不装 shape hooks/profiler、不做每步 F/B 同步。只在首步、warmup/终点等
  边界同步。首次成功 update latency 包含第一批 fetch，另列 phase/process wall。
- **F/B diagnostic**：单独 pass，在共享 autocast 前至 backward/非有限 loss 跳过后的
  边界同步；含模型+loss+backward及有限性检查，不含 optimizer、前置文本/VAE/噪声准备。
  不是 adapter-only 时间。记录 backward 是否真正执行；grad_accum 的每个 microbatch
  分开。shape hooks 只读第一组真实 metadata，checkpoint 重算不另算训练 batch。
- **CUDA kernel diagnostic**：单独 pass 的 `torch.profiler` CUDA leaf activity duration
  总和，排除 memcpy/memset，不加父 CPU aten。**总和不等于 critical path**，尤其有重叠
  streams 时。CUDA events 没有被冒充 kernel time。无 CUDA/CUPTI/可识别事件则 null+reason。
  profiled pass 从不填 headline training it/s。
- Layer F/B 是每步同步 wall，统计冷首步与 warmup 后重复值；不含 reference 导出、
  正确性扫描或 profiler。重复 fixture 用相同 seed/初态；训练每 repeat 从同样本地资产
  及 fresh adapter 开始。没有 OS page cache flush，不能称“全冷磁盘”。
- VRAM 报 torch allocated/reserved 与窗口 peak，warmup 后 reset；不等于 NVML 整卡用量。
  采样关闭避免生产采样重置 peak counter。CPU 不可测项为 null。
- raw repeats + median/min/max/sample stdev；n=1 时 stdev=null/insufficient_repeats，
  **不代表零波动**。eager compile/tuning = null/not_applicable_eager，不把首次慢算编译。
- 真实训练保留原有限性/跳步策略，只检查 aggregate 参数有变化且有限；正常首步某些
  因子梯度为零是合法的。无变化/非有限/不完整窗口不得发布为 complete baseline。

## Reference 完整性与容差

manifest 和 safetensors 都 version=1 口径；payload SHA-256、完整 key/shape/dtype/stride
清单、架构 source/module path、base/adapter 状态、实际 rank/分解参数 shape/bypass、
source commit/dirty/版本/device、固定 cotangent 和 loss 定义一起保存。仅合成值，无 pickle。
replay 拒绝未知 schema、丢/多 key、篡改 shape/layout/tolerance/hash 和非有限/退化值。
这不是签名信任机制：来源环境是 provenance，payload SHA 是完整性，不证明第三方真实性。

参考 fixture 显式将全部 trainable multiplicative factors 填为小幅非零随机值，确认
adapter 改变输出且每个 trainable gradient 非零；**仅 benchmark 实例**如此，不改生产初始化。
比较 output/dx/全部 adapter gradients：fp32 atol=rtol=1e-5；bf16 atol=rtol=0.02。
容差固定而不按误差动态放宽，输出 max_abs/max_rel（分母 floor=1e-8）/RMS。
bf16 容差和跨 GPU 数值行为仍需在目标 GPU 验证，不能由 CPU smoke 宣称 bit-exact。

## 验证门禁

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_lycoris_benchmark.py tests/test_lycoris_benchmark_training.py tests/test_training_observation.py -q
.\venv\Scripts\python.exe -m pytest tests/test_lycoris_backend.py tests/test_lycoris_bypass.py tests/test_lycoris_patch.py tests/test_lycoris_tlora.py tests/test_lycoris_resume.py tests/test_plugin_registry.py tests/test_anima_train_migration.py tests/test_anima_generate_xy.py -q
.\venv\Scripts\python.exe -m pytest tests/test_route_snapshot.py tests/test_studio_configs.py -q
.\venv\Scripts\python.exe -m ruff check .
```

CPU tests/CPU smoke 只证明工具流程与 seam，不是 R1 完整验收。完整基线还需指定 reviewer
源代码审阅、CUDA 两规格 layer/replay、用户授权的 Anima LoRA/LoKr synthetic-data
真实 pipeline 多 repeat 结果。未测平台/资产/GPU 组合不得用 CPU 数字补齐。
