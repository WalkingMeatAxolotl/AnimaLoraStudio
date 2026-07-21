# Block swap（逐层权重换入换出）方案调研

- 状态：**Gate-0 已跑通（§5.1 实测），方案未拍板**。原理、成本模型、硬件影响评估
  与实测数据已固化；所有产品与实现选择留在 §7 开放问题，逐项走五步确认后才动代码。
- 一句话结论：**训练口径理论开销约 1%，可把 K2 显存下限从 32GB 拉到 24GB 以下且不
  付精度代价；硬件损耗担忧经 1.4TB 传输实测排除（PCIe replay 增量 0）。**
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

`tools/block_swap_probe.py`。**在写任何实现代码之前必须先跑**，数据不达门槛则本
方案直接否决（LPL 同款 Gate-0 纪律）。

| 阶段 | 测什么 | 回答什么问题 |
|---|---|---|
| A 链路体检 | PCIe gen/width 实际 vs 最大、GPU/RAM 容量、replay 基线 | 本机链路是否已降级 |
| B 带宽矩阵 | pinned/pageable × H2D/D2H × 多种 size；pin 分配耗时 | 有效带宽实测值 |
| C 计算基准 | 真实 `SingleStreamBlock` 在真实 shape 下的前向/反向耗时 | `T_compute` |
| D 遮蔽判据 | B/C 比值 → `blocks_to_swap` × (省显存, 加时间) 曲线 | 划不划算 |
| E 端到端 | 真双 stream swap 循环 vs 全常驻循环的 wall clock | 预取是否真能遮蔽 |
| F 稳定性 | 持续负载下温度/功耗/replay 增量/可用 RAM | §3.2 三条风险的实测 |

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

→ **训练口径理论开销仅 1.1%（全 28 层 swap，省 22.64 GB）**。

**E 端到端**（前向口径 = 最坏情形，双 buffer + 独立 copy stream）：

| 分辨率 | per_tensor | flat（连续 buffer） |
|---|---|---|
| 1024²（比 1.04） | +18.8% / +19.3% | +20.5% / +16.6% |
| 1536²（比 ~0.42） | +18.5% | **+11.4%** |

两条结论：
- **§2.2 的分辨率推论验证成立** —— 比值越宽裕开销越低（1024² 约 19% → 1536² 约 11%）。
- 存在**约 5–8 ms/block 的固定开销**（event 同步 + copy stream 与计算争 HBM 带宽），
  不随比值改善而消失。这是理论值（3.5%）与实测（19%）差距的来源，也是实现时唯一
  值得继续压的地方。
- flat 打平传输只在比值宽裕时（1536²）明显占优；在临界比下与逐张量拷贝无差别、
  甚至互有胜负（噪声范围内）。**「打平成连续 buffer」不是万灵药**。

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

| blocks_to_swap | 常驻显存 | 理论训练开销 | 意义 |
|---|---|---|---|
| 0（现状） | 25.8 GB | 0% | 32GB 余量≈0（Phase 0 实测） |
| 14 | 14.5 GB | 0.6% | 32GB 余量充裕；**24GB 可行** |
| 28 | 3.2 GB | 1.1% | **16GB 理论可行** |

即：block swap 把 K2 训练的显存下限从 32GB 拉到 24GB 甚至更低，而训练口径的时间
代价在个位数百分比量级 —— 这正是 D2「未来下探 24GB 的唯一解锁是 fp8_scaled」当时
未考虑到的第二条路，且**不付精度代价**。

### 5.3 探针的已知局限

1. **E 段只测前向**。训练口径的 swap 需要反向逆序预取 + 与 checkpoint 重算合流
   （§2.4），朴素双 buffer 会在反向时读到被覆盖的权重。表中的训练口径数字来自 D 段
   的分段公式外推，**未端到端实测** —— 这是实现阶段第一个要补的验证。
2. 只覆盖 krea2。Anima 的 Block 需要 rope/adaln_lora 一串预备张量，构造成本高而收益
   低（24GB 已够用），未纳入。
3. 未测 pinned 内存逼近系统上限时的行为（§3.2 ① 的真实风险面）。
4. 未测与 `compile_blocks` 共存（§7-5）。

## 6. 实现落点（若 Gate-0 通过）

- Anima：[`forward.py:31`](../../runtime/training/families/anima/forward.py) 的 block
  循环插预取钩子。
- Krea2：`modeling/krea2/krea2_modeling.py` `SingleStreamDiT` 需确认展开面；
  `01-code-layout.md:189` 已为 `families/krea2/loader.py` 预留 "fp8/block-swap 兜底
  挂点"。
- 护栏：`training/sysmem.py` 需要新增 pinned 专用预算（§3.2 ①），不能复用
  `check_load_budget` 的 mmap 语义。
- 配置：`blocks_to_swap` 字段元数据一处定义（走 `config_rules.py` 双端强制，见
  `config-pipeline-refactor.md`），单位后缀规范下命名待定（§7）。

## 7. 开放问题（全部待走五步确认，勿当决策）

0. **Gate-0 判读**：训练口径理论 ~1%、前向口径实测 11–19%。方案是否继续？若继续，
   下一步是「先补训练口径端到端实测」（§5.3-1）还是直接进实现？
1. **Gate-0 门槛数字**：15% / 30% 是建议值还是用户口径？在训练口径下这两个数字都
   显得过松，可能需要按口径分别定。
2. **旋钮形态**：暴露 `blocks_to_swap=N` 整数（musubi 生态一致，用户需理解含义），
   还是 `vram_policy` 加一档自动推算 N（与现有三档一致，但不可控）？
3. **两族同时上还是 K2 only**：Anima 24GB 已够用，收益仅在高分辨率；同时上则一套
   机制两族共用，只上 K2 则少一半验证面。
4. **推理侧是否同期做**：§4 的杠杆更大，但 step 放大传输量，且 Generate 已有模型级
   让位；是否等到真撞墙再做。
5. **与 `compile_blocks` 的冲突**：权重 device 变动会触发 graph break 或重编译。
   两者互斥（走 `disable_when` 元数据）还是尝试兼容？
6. **pinned 分配失败的降级语义**：静默退回 pageable（慢但能跑）还是 DomainError
   拒绝？后者符合「不加静默魔法保护」，前者符合可用性。
7. **fp8 + block swap 叠加**是否进同一刀（§2.2 有正协同），还是严格串行两刀。
8. **那 5–8 ms/block 固定开销值不值得继续压**（§5.1 E）：三 buffer 吸收抖动、
   copy stream 优先级、按 block 大小切分多次传输重叠 —— 都是可试项，但每一项都增加
   实现复杂度。1024² 下它就是 19% 与 4% 的差距，1536² 下只是 11% 与 0% 的差距。
9. **explicit `blocks_to_swap` 的上限提示**：28 层全 swap 时常驻仅 3.2GB，但那时
   传输成为绝对瓶颈。是否给 UI 一个「推荐值 = 按当前显存自动算」的提示位。
