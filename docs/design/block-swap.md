# Block swap（逐层权重换入换出）方案调研

- 状态：**Gate-0 已跑通（§5.1 实测），方案未拍板**。原理、成本模型、硬件影响评估
  与实测数据已固化；所有产品与实现选择留在 §7 开放问题，逐项走五步确认后才动代码。
- 一句话结论：**训练口径实测开销 1024² 约 +7%、1536² 约 +2.6%，换 22.6GB 显存，
  可把 K2 下限从 32GB 拉到 24GB 以下且不付精度代价；硬件损耗担忧经 1.4TB 传输实测
  排除（PCIe replay 增量 0）。**
- 日期：2026-07-21
- 上游依据：`docs/design/multi-model/00-decisions.md` D2（fp8 / block swap 搁置，
  K2 下限 32GB bf16）；`docs/design/multi-model/04-synthesis.md` §7 Phase 0 实测
  「32GB 可训但余量≈0」→ 将「K2 专属小规模 block swap 兜底」从备选提升为
  **计划内可选项（默认关、撞显存开）**。本文是那一条的展开。
- 参考实现：kohya-ss/musubi-tuner（`blocks_to_swap`，Apache-2.0）、ai-toolkit、
  diffusion-pipe、ComfyUI `--lowvram` partial load。

---

## 1. 是什么

把 DiT 的一部分 transformer block **权重常驻 CPU 内存**，只在前向/反向计算轮到
该 block 时才搬进显存，算完立即释放。换出的 block 数是唯一旋钮（musubi 生态的
`blocks_to_swap=N`）。

成立前提是 DiT 的结构事实：N 个同构 block 串行堆叠，**任一时刻真正参与计算的只有
一个**。全部权重常驻显存纯粹是为了省搬运时间，不是计算的必要条件。

本仓两族的结构（探针直接从代码读，不靠估算）：

| 族 | block 数 | 关键维度 | 单 block（bf16） | 逐 block 挂点 |
|---|---|---|---|---|
| Anima | 28（大版本 36） | features 由 ckpt 定，heads 16 / 36 版 40 | 未测（§5.3-2） | [`forward.py:31`](../../runtime/training/families/anima/forward.py) 已是逐 block 循环 |
| Krea2 | 28 | features 6144, heads 48, kvheads 12 (GQA), multiplier 4 | **828 MB**（434.2M 参数） | `modeling/krea2/krea2_modeling.py` `SingleStreamDiT`，`forward` 内已是逐 block 循环 |

Krea2 的 28 层结构与 Anima 完全同构 —— 这意味着**一套 swap 机制两族共用**，不需要
按族扇出（正是 `02-ecosystem-survey.md` §7 批评 SimpleTuner 的那类维护债）。

## 2. 原理与成本模型

### 2.1 机制

1. 权重主副本放 CPU **pinned memory**（页锁定）。非 pinned 的可分页内存无法 DMA
   异步传输，`non_blocking=True` 会静默退化为同步拷贝，带宽腰斩且完全暴露。
2. 独立 CUDA stream 做预取：计算 block *i* 的同时后台搬 block *i+1*。
3. block *i* 算完立即释放其 GPU 副本。
4. 反向传播顺序反转（N→0），需要逆序预取。

### 2.2 唯一的成败判据

```
可完全遮蔽  ⟺  T_transfer(block) < T_compute(block)
T_transfer  = bytes_per_block / effective_pcie_bandwidth
```

暴露时间 = `max(0, T_transfer − T_compute) × swapped_block_count × passes_per_step`。

由此得到两条反直觉但重要的推论：

- **分辨率越高、batch 越大，block swap 越划算。** 计算时间随 token 数增长，传输时间
  恒定（权重大小固定）。低分辨率小 batch 才是最坏工况。
- **fp8 底模让 block swap 更容易遮蔽**，因为传输字节减半而计算时间不减半（本仓
  `quant_fp8.py` 是逐层 dequant 后 bf16 matmul，计算量不变）。两者叠加是正协同。

### 2.3 LoRA 训练的额外红利

底模权重**冻结** → GPU 上那份副本用完直接丢弃，**不需要 D2H 回写**，只有单向 H2D。
全量微调则必须双向搬。所以：

| 场景 | 每步传输方向 | 相对传输量 |
|---|---|---|
| LoRA 训练（本仓唯一场景） | 前向 H2D + 反向 H2D | 2× |
| 全量微调 | 前向 H2D + 反向 H2D + D2H 回写 | 3×~4× |
| 推理（每 step） | H2D | 1× × steps |

本仓只做 LoRA → 落在最省的一档。LoRA 参数本身极小，与优化器状态一起**常驻 GPU
不参与 swap**。

### 2.4 与 gradient checkpointing 的交互（实现难点）

开启 checkpointing 后，反向阶段要**重算前向**，那时该 block 的权重必须再次在位。
朴素实现会让同一 block 在一个 step 内被搬 3 次（前向 1 + 重算 1 + 反向 1）。
musubi 的做法是把重算与反向合并在同一次驻留窗口内完成。

本仓 Anima 的 [`forward_with_optional_checkpoint`](../../runtime/training/families/anima/forward.py)
已经是逐 block 的手工展开循环，**这是天然且唯一的挂点**，无需改动架构即可插入
预取钩子。Krea2 侧需要确认 `SingleStreamDiT.forward` 是否有同等展开面。

### 2.5 与现有手段的正交关系

| 手段 | 砍什么 | 代价 | 本仓现状 |
|---|---|---|---|
| gradient checkpointing | 激活 | 重算 ≈ +30% 时间 | 已有，K2 v1 默认强制 |
| fp8 量化 | 权重**字节数** | 精度 | `quant_fp8.py` 已实施（推理 + fp8_base 训练） |
| 模型级 offload | **非活跃模型**（TE/VAE/DiT 整体） | 切换延迟 | `vram_policy` 三档已实施 |
| **block swap** | 权重**驻留位置**（字节数不变） | PCIe 时间 | 本文对象 |

**与已有 `vram_policy` 的本质区别**：模型级 offload 解决「TE + DiT 同时装不下」，
block swap 解决「**单个 DiT 自己就装不下**」。前者救不了 24GB 跑 20B，后者可以。
这是它对 K2 的唯一不可替代价值。

## 3. 硬件影响与损耗评估

用户明确关注项。结论先行：**不存在"磨损"意义的硬件损耗；真实风险全部在系统稳定性
与散热，且都可测量。**

### 3.1 不构成损耗的部分（可以放心）

- **PCIe 链路**：差分信号电气链路，无机械或存储介质磨损机制。持续满带宽是设计规格
  内的常态工况（数据中心 GPU 常年如此）。PHY 满载功耗量级为个位数瓦，相对 GPU
  整卡 300–450W 是噪声。
- **显存（GDDR6/6X）与系统内存（DDR）**：DRAM 是电容存储，读写**不产生疲劳**。
  NAND flash 的擦写寿命概念不适用。
- **半导体老化的真实机制**（electromigration、NBTI/HCI）由**温度和电压**驱动，与
  "搬运了多少字节"只有间接关系 —— 间接路径是「更多活动 → 更高功耗 → 更高温度」。
  而 block swap 期间 GPU 若出现等待气泡，**平均功耗反而下降**。

### 3.2 构成真实风险的部分（必须设护栏）

**① pinned memory 不可换页 —— 与本仓已知卡死案例同源。**

`training/sysmem.py` 的整个存在理由是 mmap 文件缓存页撑爆 working set 导致整机
换页卡死（见 `mmap_working_set_paging_freeze` 案例）。pinned memory 比那更硬：

- `trim_working_set()` 对 pinned 页**完全无效** —— 页锁定的定义就是不可被回收。
- `check_load_budget()` 现有的 RAM 预算把权重文件大小算作"可回收的 mmap 峰值"，
  而 pinned 是**永久占用**，同样字节数的危害等级不同，现有护栏语义**不覆盖**。
- Windows 对可锁定物理内存总量有系统级上限，分配失败是硬错误，必须有降级路径。

这是本方案**最大的真实风险**，远大于任何硬件担忧。

**② PCIe 链路错误（correctable error / replay）。**

满带宽持续 DMA 会暴露插槽接触、riser 线材、主板走线的边际质量问题。表现为链路
replay 重传 → 有效带宽悄悄下降，而非报错。NVML 暴露 `PcieReplayCounter`，
**探针必须采样其增量**作为链路健康判据。若本机链路本身跑在降级模式（x8 而非
x16、或走 chipset 通道而非 CPU 直连），带宽会腰斩，同样必须先测出来。

**③ 内存带宽争用与散热。**

持续 DMA 占用系统内存带宽，与 dataloader / 打标进程抢。PCH 与 GPU 板边温度会
上升，但在规格内。探针采样温度与功耗以确认无异常。

### 3.3 与 WDDM 显存崖的关系（正收益）

PR #281 记录的 190s 卡死是近满载时 WDDM 换页崖。block swap 降低常驻峰值，**天然
远离崖区**，在这条上是净正收益。

## 4. 推理侧

同样成立，且生态更成熟（ComfyUI `--lowvram` 的 partial load 本质就是它，粒度是
module 而非 block）。但算式不同：

- **更便宜**：无激活、无优化器状态、无反向，纯单向 H2D。
- **更贵**：扩散是 N 步循环，**每一步都要把整个模型搬一遍**。训练一步 = 2 遍；
  推理 30 步 = 30 遍。总传输量被 step 数放大。
- **杠杆更大**：推理显存几乎全是权重（batch=1 时激活可忽略），block swap 直接决定
  「**能不能跑**」，而非「跑得舒不舒服」。

对本仓的具体处境：32GB 上 K2 Generate 的 TE+DiT bf16 常驻超显存，当前靠
`_should_offload_te` / `_should_yield_dit` 的模型级让位解决。若之后上更高分辨率或
多 LoRA，DiT 自身驻留就是下一个瓶颈 —— 那时 DiT block swap 是同一把刀的自然延伸。

## 5. Gate-0 探针与实测结果

`tools/block_swap_probe.py`。**在写任何实现代码之前必须先跑**。

注意 Gate-0 在这里的语义**不是证伪门槛**：block swap 是「时间换显存」的确定性交易，
不存在「不值得做」，只存在「对谁值得」（详见 §8.3）。探针的作用是**标定预期**并
体检硬件。唯一仍具否决力的是 F 段的链路健康指标。

| 阶段 | 测什么 | 回答什么问题 |
|---|---|---|
| A 链路体检 | PCIe gen/width 实际 vs 最大、GPU/RAM 容量、replay 基线 | 本机链路是否已降级 |
| B 带宽矩阵 | pinned/pageable × H2D/D2H × 多种 size；pin 分配耗时 | 有效带宽实测值 |
| C 计算基准 | 真实 `SingleStreamBlock` 在真实 shape 下的前向/反向耗时 | `T_compute` |
| D 遮蔽判据 | B/C 比值 → `blocks_to_swap` × (省显存, 加时间) 曲线 | 划不划算 |
| E 端到端 | 真双 stream swap 循环 vs 全常驻循环的 wall clock（**前向口径**） | 预取是否真能遮蔽 |
| F 稳定性 | 持续负载下温度/功耗/replay 增量/可用 RAM | §3.2 三条风险的实测 |
| G 训练口径 | checkpoint + 反向逆序预取的完整一步，交错 A/B 对照 | 训练实际慢多少（B10） |

### 5.1 实测数据（2026-07-21，RTX 5090 32GB / PCIe 4.0 x16 / 37GB 可用 RAM）

**A 链路**：gen4 x16 满配（空闲时降 gen1 省电属正常），replay 基线 0。

**B 带宽**：pinned H2D **26.8 GB/s**（各 size 一致，16MB 起就跑满），pageable
18.5–23 GB/s，**pinned 提速 1.45×**。pinned 分配 **59 ms / GB** —— 实现必须预分配
复用，不能每步 alloc。

**C 规模与计算**（1024², batch 1, bf16, seq_len 4608 = text 512 + image 4096）：

| 量 | 值 |
|---|---|
| 单 block | 434.2M 参数 / **828 MB** |
| 28 层合计 | **22.64 GB**（+ txtfusion/embed/last ≈ 25.8GB 总量，与 D2 记录吻合） |
| 前向 | 29.1 ms |
| 前向+反向 | 103.7 ms（反向段 ≈ 74.5 ms） |

**D 遮蔽判据**：`T_transfer` = 828MB ÷ 26.8GB/s = **30.2 ms**。

| 口径 | 传输/计算比 | 判定 |
|---|---|---|
| 前向（推理） | **1.04** | 临界，每 block 暴露 1.1 ms |
| 反向段（训练） | **0.40** | 完全遮蔽 |

→ 训练口径的**暴露**部分仅 1.1%。但这不是全部成本 —— 见下方 E 段测出的争用项。

**E 端到端**（前向口径 = 最坏情形，双 buffer + 独立 copy stream）：

| 分辨率 | per_tensor | flat（连续 buffer） |
|---|---|---|
| 1024²（比 1.04） | +18.8% / +19.3% | +20.5% / +16.6% |
| 1536²（比 ~0.42） | +18.5% | **+11.4%** |

三条结论：

- **§2.2 的分辨率推论验证成立** —— 比值越宽裕开销越低（1024² 约 19% → 1536² 约 11%）。
- 存在一项**理论模型之外的额外成本**，扣掉暴露后仍剩：1024² 约 4.4 ms/block、
  1536² 约 8.2 ms/block。**它随激活规模上升而变大，所以不是"固定开销"，而是
  copy stream 的 DMA 写入与计算 kernel 争 HBM 带宽**（event 同步只占其中几十 μs）。
  这是理论值（3.5%）与实测（19%）差距的真正来源。
- flat 打平传输只在比值宽裕时（1536²）明显占优；在临界比下与逐张量拷贝无差别、
  甚至互有胜负（噪声范围内）。**「打平成连续 buffer」不是万灵药**。

**G 训练口径端到端实测**（B10；gradient checkpointing + 反向逆序预取，底模 frozen
= LoRA 场景。基线是同样 checkpoint 语义的全常驻）：

| 分辨率 | 基线（抖动） | swap | 开销 | 每 block 额外 |
|---|---|---|---|---|
| 1024² 第 1 次 | 920.3 ms（±0.6%） | 982.7 ms | **+6.8%** | 7.80 ms |
| 1024² 第 2 次 | 912.8 ms（±1.0%） | 983.3 ms | **+7.7%** | 8.81 ms |
| 1536² | 1963.0 ms（±0.3%） | 2013.9 ms | **+2.6%** | 8.48 ms |

三条结论：

1. **训练口径实测 1024² 约 +7%、1536² 约 +2.6%**，优于此前 9.5% 的估算。基线自身
   抖动仅 ±0.3–1.0%，数字可信。
2. **每 block 额外时间是约 8 ms 的常数**（7.80 / 8.81 / 8.48），**与分辨率无关**；
   百分比之所以在 1536² 下降，纯粹是基线计算量变大（2.25×）把它摊薄了。这修正了
   §5.1-E 基于前向口径得出的「随激活规模上升」判断 —— 那是前向口径的表象，训练
   口径下它是常数。也意味着**外推到全 28 层时开销比例不变**（每 block 常数）。
3. 每个驻留窗口约 4 ms（8ms ÷ 2 个窗口），与 E 段前向口径在 1024² 测得的 4.4ms/窗口
   互相印证。

> **方法论教训**：本段首版用「先测基线、再测 swap」的顺序测量，得到 **−2.2%** 的荒谬
> 负开销 —— GPU 时钟状态在两条路径之间不同（先跑的那条在冷态未 boost）。改成
> **交错 A/B**（两条路径同时驻留、逐轮交替计时）后才得到可复现的数字。任何
> 「A 比 B 快但物理上不可能」的测量结果，先怀疑顺序效应。

**F 硬件影响**（60s 持续负载，213 轮 × 8 block = **1704 次换入 ≈ 1.4 TB 传输**）：

| 指标 | 结果 | 判定 |
|---|---|---|
| PCIe replay 增量 | **0** | 链路零重传，§3.2 ② 通过 |
| GPU 温度 | max 70°C / mean 64°C | 正常，无热压力 |
| GPU 功耗 | max 497W / mean 484W | 与常规满载训练同量级，无异常尖峰 |
| 可用内存漂移 | −99 MB | 噪声级，pinned 无泄漏 |

**→ §3 的硬件损耗担忧全部排除**：链路零错误、温度功耗无异常、内存无漂移。剩下的
唯一真实风险仍是 §3.2 ① 的 pinned 内存预算（本次只 pin 6.5GB，未触及上限）。

### 5.2 由实测数据推出的容量结论

K2 DiT bf16 总量 ≈ 25.8GB，其中 22.64GB 是可 swap 的 28 层 block：

| blocks_to_swap | 常驻显存 | 实测训练开销（1024² / 1536²） | 意义 |
|---|---|---|---|
| 0（现状） | 25.8 GB | 0% | 32GB 余量≈0（Phase 0 实测） |
| 14 | 14.5 GB | ≈ 3.5% / 1.3% | 32GB 余量充裕；**24GB 可行** |
| 28 | 3.2 GB | ≈ 7% / 2.6% | **16GB 理论可行** |

（每 block 额外时间是常数，所以开销与 `blocks_to_swap` 成正比、与总层数无关。）

即：block swap 把 K2 训练的显存下限从 32GB 拉到 24GB 甚至更低，而训练口径的时间
代价在个位数百分比量级 —— 这正是 D2「未来下探 24GB 的唯一解锁是 fp8_scaled」当时
未考虑到的第二条路，且**不付精度代价**。

### 5.3 探针的已知局限

1. ~~E 段只测前向，训练口径未端到端实测~~ —— **已由 G 段补齐**（B10）。G 段测的是
   时序而非数值正确性：buffer 权重被轮转覆盖，梯度无意义；且 LoRA 参数（常驻、不
   参与 swap）的计算量未计入，其相对底模可忽略但非零。
2. 只覆盖 krea2。Anima 的 Block 需要 rope/adaln_lora 一串预备张量，构造成本高而收益
   低（24GB 已够用），未纳入。
3. 未测 pinned 内存逼近系统上限时的行为（§3.2 ① 的真实风险面）。
4. 与 `compile_blocks` 共存未测 —— 但按 B5 判定当前无实现，不构成阻塞。

## 6. 实现落点（若 Gate-0 通过）

按 B3，机制必须**从一开始就 family 无关**：包住一个 `nn.ModuleList`、从外部提供预取
迭代器，不要求改模型内部代码（理由见 §7.1）。

- Krea2（先行）：`modeling/krea2/krea2_modeling.py` `SingleStreamDiT.forward:513` 的
  block 循环；`01-code-layout.md:189` 已为 `families/krea2/loader.py` 预留
  "fp8/block-swap 兜底挂点"。**该文件对 ComfyUI 有逐字 parity 要求，改动要克制。**
- Anima（后补）：[`forward.py:31`](../../runtime/training/families/anima/forward.py)
  的 block 循环换一行迭代器。
- 推理侧（B4 同期）：`runtime/anima_daemon.py` 的模型栈，与现有
  `_should_yield_dit` / `_should_offload_te` 的模型级让位分层协作 —— block swap 管
  单模型内部，`vram_policy` 管模型之间。
- 护栏：`training/sysmem.py` 需要新增 **pinned 专用预算**（§3.2 ① / §8.1），不能复用
  `check_load_budget` 的 mmap 语义（那套假设内存可回收，pinned 不可）。
- 配置：`blocks_to_swap` 字段元数据一处定义（走 `config_rules.py` 双端强制，见
  `config-pipeline-refactor.md`）。字段名无单位后缀问题（"blocks" 本身即单位）。
- UI：按 B9 只给提示不给推荐数字。

## 7. 决策（2026-07-21 用户裁定第一轮）

| # | 决策 | 备注 |
|---|---|---|
| **B2** | 旋钮 = **`blocks_to_swap` 整数**，不加 `vram_policy` 档 | 与 musubi/ai-toolkit/diffusion-pipe 生态一致；用户可控优于自动推算 |
| **B3** | **先上 K2，验证成熟后再上 Anima** | 前提是迁移代价可控，见 §7.1 判定：两族 block 循环结构完全一致，机制从一开始就按 family 无关设计，Anima 后补 = 接线 |
| **B4** | **推理侧同期做** | §4 杠杆更大（决定「能不能跑」而非「快不快」） |
| **B7** | **fp8 + block swap 同一刀** | 目标明确 = K2 fp8 训练门槛下探 **12–16GB**；§2.2 已论证正协同（fp8 减半传输字节而计算量不减，比值更宽裕） |
| **B9** | 上限**只给提示、不给精确推荐数字** | 用户裁定：不同显卡的显存、PCIe 代数、DRAM 速率差异过大，精确数字会误导 |
| **B5** | `compile_blocks` 冲突 —— **当前不存在，不设计** | 查证：主线只有能力位（`FAMILY_CAPABILITIES` / `KNOWN_CAPABILITIES` 的词表占坑），**无配置字段、无 `torch.compile` 代码**；实现在未合的 `pr257-review` 分支。若 #257 先合再回来处理 |

### 7.1 B3 的迁移代价判定（支撑「先 K2 后 Anima」）

两族的 block 循环结构**完全同构**：

```
Anima  runtime/training/families/anima/forward.py:31   for block in model.blocks:  x = block(x, ...)
Krea2  modeling/krea2/krea2_modeling.py:513            for block in self.blocks:   h = block(h, ...)
```

只要机制抽成 family 无关的「包住一个 `nn.ModuleList` + 提供预取迭代器」组件（而非
写死 krea2 类型），Anima 后补 = 在它的循环里换一行迭代器，几十行 + 测试。

**唯一不对称、需要在设计时就处理好的点**：两族的**结构定义都在 `modeling/<family>/`**
（按层切架构，`01-code-layout.md` §2.1：`modeling` 结构定义 → `runtime/training/families`
行为适配 → `studio/services/models/families` 资产清单，依赖单向），但**逐 block 循环
的位置不同**：

| 族 | block 循环在哪 | 为什么 |
|---|---|---|
| Krea2 | `modeling/krea2/krea2_modeling.py:513`，**结构定义层内部**，`use_checkpoint` 是模型自带参数 | 该文件是我们按 ComfyUI 命名自己写的，可以直接内建开关 |
| Anima | `runtime/training/families/anima/forward.py:31`，**行为适配层**，手工展开模型内部 API 重写了一遍前向 | `modeling/anima/cosmos_predict2_modeling.py` 是移植的外部 Cosmos 主干（2068 行），其 `forward` 不提供 checkpoint 开关，且要保持与上游可比对，所以在外面展开而非改它 |

因此组件必须能从**外部**包住一个 `nn.ModuleList` 而不要求改模型内部代码 —— 这样
Krea2 侧不必动 parity 敏感的 `modeling/` 文件，Anima 侧也不必复制一份逻辑（后者
就是 `02-ecosystem-survey.md` §7 批评 SimpleTuner 的按族扇出）。

## 8. 第二轮裁定与剩余问题

| # | 决策 | 备注 |
|---|---|---|
| **B6** | pinned 分配失败 = **报错，不静默降级** | 用户裁定：既然失败时机可确定（§8.1：只发生在启动时的分配那一刻），就该明确报错。落 DomainError + 可操作文案，与 `check_load_budget` 两条护栏同款 |
| **B8** | HBM 争用项**不压** | 见 §8.2 三条理由；靠 B7 的 fp8 减半传输量顺路解决 |
| **B10** | **下一步 = 先补训练口径端到端实测** | 用户裁定。**已完成** —— 探针 G 段，实测 1024² +7% / 1536² +2.6%（§5.1 G） |
| **B1'** | **Gate-0 门槛不设否决语义** | 见 §8.3 —— 这不是「值不值得做」的赌，是「用时间换能不能跑」的确定性交易 |

### 8.3 门槛数字到底决定什么（B1' 的由来）

用户提问：「这个门槛数字决定了是为了什么，如果超出了这个时间就不做吗？」—— 这个
问题暴露了 Gate-0 语义被套错了模板。

LPL 那次的 Gate-0 是**证伪门槛**：新算法效果不明，不达标就没有存在价值，该 park。
block swap 不是这种东西 —— 它是**确定性交易**：拿走一段时间，换回一段显存，两边
都是可测量的已知量。对不同用户，这笔交易的价值天差地别：

| 用户显存 | 不开 | 开（慢 ~10%） | 门槛的意义 |
|---|---|---|---|
| 32GB+ | 正常训练 | 纯亏 | 默认**关** |
| 24GB | **完全跑不了** | 能跑 | 慢 30% 也必须开 |
| 12–16GB（B7 目标） | **完全跑不了** | 能跑 | 同上 |

**对跑不了的用户，任何百分比都优于「跑不了」。** 所以超出门槛不该导致「不做」。

门槛数字真正该决定的只有两件事，都不涉及否决：

1. **默认值与自动建议**：训练开销若稳定 < 15%，显存不足时可以主动建议开启；> 30%
   则只在用户显式开启时生效，不主动推荐。
2. **UI 提示的措辞强度**（B9 只给提示不给数字的前提下）：低于门槛说「会略微变慢」，
   高于门槛说「会明显变慢」。

因此 §5 里「不达门槛则方案直接否决」的表述已作废，改为标定预期用。**唯一仍具否决
力的是 F 段的硬件健康指标**（replay 持续增长 = 链路有问题，那是真该停）。

### 8.1 pinned 分配失败的时机与恢复（回答 B6 的前置问题）

- **失败发生在分配那一刻，不会在运行中途随机出现** —— `cudaHostAlloc` 要么拿到页锁定
  内存要么立即返回 `cudaErrorMemoryAllocation`（PyTorch 抛 `RuntimeError`）。一旦分配
  成功，这块内存就锁定归本进程所有，运行期不会被回收、不会"用着用着没了"。
- **但同一份配置这次成功下次失败是可能的**，因为成败取决于分配时刻的系统可用物理
  内存（另一个训练/打标进程、浏览器都会影响）。这与本仓 `check_load_budget` 面对的
  是同一类不确定性。
- **由此得出实现纪律**：**启动时一次性预分配全部 pinned buffer**，不要按需分配。
  这样失败只可能发生在训练启动阶段 —— 可预测、可 fail-fast、可给出明确错误。代价是
  启动多花约 `59ms × GB`（§5.1 B），22.6GB 约 1.3 秒，可接受。
- **自动恢复**：技术上可行（退回 pageable 慢 1.45×，或减少 `blocks_to_swap` 重试）。
  但静默降级会让用户拿到一个「莫名其妙慢了一半」的训练 —— 与
  `feedback_no_silent_magic_protection` 的纪律冲突。倾向：**DomainError 明确失败 +
  错误文案给出可操作建议**（关掉其他占内存的应用 / 调小 `blocks_to_swap`），与
  `check_load_budget` 现有的两条护栏文案同款。待用户拍板。

### 8.2 HBM 争用项要不要继续压（回答 B8）

**相对影响**（1024²，全 28 层 swap）：单步 2.90s → 3.18s。一次 2000 步的训练从约
97 分钟变成约 106 分钟，**多 9 分钟**，换来 22.6GB 显存和「24GB 卡能训 K2」。

**建议不压**，三条理由：

1. 争用的物理来源是 DMA 写入与计算 kernel 抢 HBM 带宽，**不是可以靠调度消除的开销**。
   加 buffer（3+ 轮转）解决的是抖动不是争用，而且每个 buffer 多占 828MB —— 直接抵消
   收益。copy stream 优先级影响 SM 调度，对 DMA 引擎无效。两条最便宜的招大概率无效。
2. 唯一真正有效的方向是**减少传输字节**，而那正是 B7 已经决定要做的 fp8 —— 传输量
   减半，争用同比例下降。**顺路解决，不需要单开一条优化线**。
3. 维护代价不对称：分片传输重叠会把 swap 从「一次 `copy_`」变成一个状态机，还要处理
   与 checkpoint 重算的交互。复杂度是跃升式的，而收益上限只有几个百分点。

### 8.4 第三轮裁定（进入实现）

| # | 决策 | 备注 |
|---|---|---|
| **B11** | **默认值 = 0（关闭）**，暂不做自动建议 | 用户裁定：等真实实现并训过 LoRA、有实际体验后再定默认策略。门槛数字同步搁置（B1' 已去除其否决语义，现在连"定默认值"这个用途也推后） |
| **B12** | 项目目标 = **fp8 + swap 在 16GB / 12GB 消费级卡上稳定跑 K2 LoRA 训练** | 用户明确的最大期望。这是验收标准，不是"能跑就行" —— 强调**稳定** |

## 9. 实现设计

### 9.1 必须原地换 `param.data`，不能用 buffer 轮转（探针与实现的关键差异）

探针 E/G 段用「2 个预留 block 实例作为 buffer 轮转」测时序。**真实实现不能这么做**：

LyCORIS `apply_to()`（[`utils/lycoris_adapter.py:157`](../../utils/lycoris_adapter.py)）创建的
LoRA 模块**持有原 block 内 Linear 的引用**并包住它的 forward（bypass 模式下是
`org_forward(x) + lora_up(lora_down(x))`）。若前向走的是 buffer 实例，就**完全绕过了
LoRA** —— 训练会静默地什么都没学到。

因此实现取**原地换**：module 对象始终不变，只切换每个 parameter 的 `.data` 指向。

| | buffer 轮转（探针用） | 原地换 `param.data`（实现用） |
|---|---|---|
| LoRA 兼容 | **破坏**（前向绕过 LoRA 模块） | 兼容（module 身份不变） |
| fp8 `weight_scale` | **错配**（非持久 buffer 绑在 module 上，不随权重轮转） | **自动正确**（module 不变，scale 一直配对） |
| 时序特征 | 与实现等价（传输量、同步模式相同） | —— |

探针测出的性能数字仍然有效（时序等价），但**代码形态不可照搬**。

顺带记一条：fp8 的 `weight_scale` 由 `patch_fp8_linears` 注册为**非持久 buffer**
（不进 `state_dict`）。任何基于 `state_dict()` 做搬运的设计都会漏掉它 —— 原地换
天然避开这个坑，但若将来有人改回 state_dict 路线，这是第一个会踩的雷。

### 9.2 分刀

1. **刀 1（core）**：family 无关的 swap 组件 + 单测。不接线，不改任何现有行为。
2. **刀 2（K2 接线）**：loader 支持部分 block 权重留 CPU pinned；配置字段
   `blocks_to_swap`（默认 0，B11）；pinned 预算护栏（B6 报错语义）。
3. **刀 3（fp8 叠加 + 12/16GB 验收）**：B7/B12。

Anima 接线按 B3 留到 K2 验证成熟之后。
