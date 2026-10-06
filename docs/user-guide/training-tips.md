# LoRA 训练技巧（Anima / Krea 2）

> **适用范围**：数据准备、学习率 / rank、优化器、过拟合排查等内容对两个模型族通用；
> timestep / 噪声 / caption 生态相关章节以 Anima 立论，标注「仅 Anima」的功能对 Krea 2
> 不可用（UI 会按族自动隐藏）。Krea 2 专属的底模选择 / 显存 / 默认值差异集中在
> [Krea 2 训练](#krea-2-训练) 一章。
>
> **如何使用本文**：默认值与互斥规则描述当前程序契约；参数范围是实验起点，不是两族的最优值或效果保证。历史训练跨数据、配置与版本，不能把完成次数当受控对照。固定数据、训练/采样/验证划分 seed 与评估条件，一次改变一个主要变量，再判断是否改善。

## 数据准备

### 数据量与训练量：不要把张数或 repeat 当门槛

没有一个跨模型、跨目标都成立的“最低图片数”。**10–30 张经过筛选、覆盖关键变化的图片可以作为单角色或单一概念的有效小规模基线**；这不保证一次成功，也不意味着凑到 30 张才可训练。画风训练同样应先覆盖构图、角色、场景、色彩和线条变化，再决定是否扩充；重复图、近邻裁剪和低质量图不会因为数量增加而自动提供新信息。

截至 v0.28 的本地历史审计中，按每个版本保留最后一次完整任务后，可识别为画风训练谱系的 38 组记录使用 **29–100 张训练图（中位数 52）**，文件夹 repeat 只出现 **1 或 2**。这些记录证明小于旧表格所写的规模也能完整运行，**不能单凭 `done` 状态证明画面质量，也不能据此给角色 LoRA 设统一最优值**。数据口径、模型族拆分和限制见[训练建议证据审计](../reviews/2026-09-22-training-advice-audit-v0.28.0.md)。

按目标规划覆盖面，而不是先凑固定张数：

| 目标 | 小规模基线怎么选 | 扩充时优先补什么 |
|---|---|---|
| 单角色 / 单一服装 | 10–30 张高质量图即可开始对照 | 角度、表情、构图、光照，以及需要独立控制的服装或发型 |
| 画风 / 单一视觉概念 | 从能覆盖主要内容与构图变化的一组开始；本地画风记录多集中在约 30–100 张 | 角色与背景类型、色彩、线条、远近景，避免让某个常见主体冒充画风 |
| 多角色 / 复合概念 | 不设 200 或 500 张硬门槛；先为每个身份或子概念列覆盖矩阵 | 补齐弱类别并控制类别平衡；必要时拆成多个 LoRA，而不是只增加总张数 |

#### repeat、epoch 与总步数要一起算

Studio 的 **数据与训练规模**面板已经按当前实现推导：

- 文件夹有效样本 = 图片数 × repeat × 分辨率档数；正则样本也计入总量。
- 常规训练每 epoch 的 optimizer 更新步数约为 `ceil(有效样本 ÷ (batch_size × grad_accum))`；NaViT Packing 使用真实打包模拟，界面会标 `≈`。
- 总更新步数约为 `每 epoch 更新步数 × epochs`，若 `max_steps` 更小则提前截断。
- 单张训练图的计划曝光次数约为 `repeat × epochs × 分辨率档数`；它与 optimizer 更新步数不是同一个指标。

**repeat 是样本权重，不是质量旋钮。** 先从 `1` 开始；只有希望某个文件夹在每 epoch 被更频繁抽到时才试 `2` 或更高，并相应减少 epoch 或检查总步数。不要把 `repeat=10/20` 与 40–60 epoch 机械叠加，也不要为了命中某个总步数而复制重复图片。小数据集可以通过更多 epoch 或较高 repeat 获得相近曝光，但二者会改变每轮采样、保存 checkpoint 的节奏；应固定其它变量比较。

总步数也没有跨模型通用答案。本地完整画风记录中，较新的 Krea 2 任务覆盖约 **480–1740** 个更新步，Anima 任务覆盖约 **1260–3520** 个更新步；这是历史使用范围，不是推荐区间或质量保证。先设置较密的 checkpoint / sample 间隔，用固定 prompt、seed 与留出图寻找“尚未学到 → 合适 → 开始过拟合”的区间，再决定延长或缩短，而不是一次押注最终步数。

### 图片质量要求

- **分辨率**：建议 1024×1024 或更高
- **裁剪**：尽量保留完整构图，避免截断重要部位
- **多样性**：包含不同角度、表情、服装、光照
- **一致性**：如果是角色 LoRA，确保同一角色的外观一致

### 标签质量

- 使用 VLM 打标时，检查输出是否准确
- 删除明显错误的标签
- Anima 的单个 tag 内用空格、tag 之间用逗号；Krea 2 保持自然语言描述，不套用 tag 打乱规则。

### Caption 策略与正则集配合

如果希望触发词控制角色、服装或画风，可尝试将目标共有特征交给 trigger 学习，同时让 caption 描述仍需单独控制的变化。正则集不含训练触发词，是减轻非触发偏移的一种策略，**不保证 trigger off 等同原底模，也不保证严格的 on/off 开关**。

#### 按目标选择 caption 策略

| LoRA 类型 | 建议保留 | 可尝试省略 | 正则集检查 |
|---|---|---|---|
| **画风 LoRA** | 角色、动作、场景、物体等内容描述 | 按实验目标决定是否保留已有画师标签 | 排除训练触发词，核对是否含目标画师标签 |
| **人物 LoRA** | 触发词、环境、动作及希望可控的服装等变化 | 希望绑定给 trigger 的稳定身份特征 | 排除训练触发词；检查数据多样性 |
| **人物 + 衣服 LoRA** | 触发词、环境及非目标变化 | 明确希望联合学习的身份/服装特征 | 排除训练触发词；避免误把目标内容加入正则 |

省略特征描述可能增强目标绑定，也可能降低这些属性的独立可控性；不要批量删除所有发色、服装标签后就假定效果更好。保留与删除应分别建版本对照。正则图片的属性分布也需要人工检查，不因来自 Booru 或底模生成就天然均衡。

#### 如何检查是否有效

固定底模、prompt、采样 seed 和采样参数，对比三组：**不挂 LoRA**、**挂 LoRA 不写 trigger**、**挂 LoRA 写 trigger**。再替换服装、动作、场景检查可控性。无触发词时出现风格偏移可能涉及数据、学习率、训练量与 caption 共现，不能仅凭一个残留 tag 判定原因。

---

## 参数调优

### 学习率

下表仅供 AdamW 等常规学习率优化器小规模搜索，图片数不是唯一依据；adapter、训练量和数据分布同样重要。Prodigy/PPSF 使用 `1.0` 缩放语义，Automagic 的起点也不同，见[优化器指南](optimizers.md)。

| 场景 | 学习率试验范围 | 注意事项 |
|------|-----------|------|
| 小数据集 (<100 张) | 5e-5 ~ 1e-4 | 较低起点仍可能过拟合 |
| 中等数据集 (100-500 张) | 1e-4 ~ 2e-4 | 围绕基线单变量试验 |
| 大数据集 (500+ 张) | 1e-4 ~ 3e-4 | 图片更多不意味着必须加大学习率 |

**调试技巧**：
- loss 下降慢时先检查数据、有效步数及固定采样效果，不只凭 loss 提高学习率。
- loss 剧烈震荡时排查 NaN/Inf、异常样本和采样分布，再单变量试降低学习率。
- 固定条件下观察到过拟合时，尝试降低学习率或减少训练量。

### LoRA Rank

| Rank | 试验方向（非效果保证） |
|------|----------------------|
| 8–16 | 低容量基线，先验证是否足以表达目标 |
| 32 | 默认起点，可用于角色或画风基线 |
| 64–128 | 低 rank 明确不足时再试，并检查显存与过拟合 |

实际参数量和文件大小取决于模型族、目标模块、adapter、dtype 等，不能统一按 rank 换算 MB；不同算法的 rank 也不代表相同容量。先固定其他条件，再比较提高 rank 是否有效。

### LoRA vs LoKr

| 类型 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| LoRA | 简单稳定，兼容性好 | 表达力有限 | 单角色、简单画风 |
| LoKr | 表达力强，参数高效 | 需要调参 | 多角色、复杂画风 |

### LyCORIS v4 与自动训练遮罩

当前依赖固定为 `lycoris-lora==4.0.0`，LoRA / LoHa / LoKr 默认使用 eager Torch 兼容基线。普通 LoRA / LoHa 可显式启用受限的[实验性 Triton 后端](triton-backend.md)，但它不保证更快，并与 FlashAttention、DoRA、T-LoRA、Ortho、LoKr 及 adapter dropout 组合互斥；当前公开硬件验证只覆盖 Anima。旧权重的缩放语义由兼容层保留；从 v0.26 及更早版本升级时让启动器同步依赖，详见 [v0.27 升级说明](upgrading-v0.27.md)。

服装、姿态、画风训练若需降低头部区域对 loss 的贡献，可在“预处理 → 涂抹”运行[自动头部遮罩](auto-head-mask.md)，手工修正并保存，再启用 masked loss。该功能不会修改 caption；角色名等身份标签仍需自行处理。masked loss 与 Leap / NaViT Packing 不兼容。

### 优化器选择

| 优化器 | 何时用 | 关键参数 |
|--------|--------|---------|
| `adamw` | 默认。手调 lr 不嫌烦、想稳定可预期的训练 | `learning_rate` 1e-4 起步 |
| `prodigy` | 不想调 lr。**注意**：扩散 LoRA 上易出"风格突变 ep" | `prodigy_d_coef` 小数据集设 0.5 |
| `prodigy_plus_schedulefree` | 可选的自适应基线，sample/save 走 Schedule-Free averaged weights，可能缓和 checkpoint 波动，但不保证消除风格突变 | 见下方说明 |

#### ProdigyPlusScheduleFree (PPSF) 使用要点

Anima 与 Krea 2 都可使用 PPSF。它维护 averaged weights，可能缓和相邻 checkpoint 的变化；但 `d` 估计、数据、训练量与采样条件都会影响结果。本地历史没有足以证明它普遍优于普通 Prodigy 或 AdamW 的受控对照，程序默认仍是 AdamW。

- **学习率**：固定 `1.0`（PPSF 内部估计真实步长，外部 lr 只是缩放系数；UI 会强制）
- **lr_scheduler**：**必须 `none`**（Schedule-Free 自带调度，叠 cosine 会破坏 averaged
  weights 的收敛保证；UI 自动 disable，pydantic 也会拦下）
- **ppsf_d_coef**：默认 `1.0`；欠拟合可试调高（如 `2.0`），过拟合 / 小数据集可试调低（如 `0.5`）。这些是经验起点，不保证改善。
- **ppsf_prodigy_steps**：`0` 表示不冻结；若需要试验后期冻结 `d`，可从总步数的 1/4 到 1/2 开始比较（如 2000 步试 500–1000），不是防止跳档的保证。
- **ppsf_fused_back_pass**：显存吃紧时开
- **save / sample 行为**：训练代码自动在 sample 和 save 前调 `optimizer.eval()` 切到
  averaged weights、事后切回。保存的 LoRA 是 averaged 状态，直接可用

#### 如何判断 PPSF 是否改善了相邻 checkpoint 波动

保留原始数据和软件环境，固定训练 seed、采样 prompt/seed 与更新步数预算，分别跑优化器基线；记录优化器要求的学习率和 scheduler 差异。比较相邻 checkpoint 是否更平稳，以及是否损失了学习速度或最终质量。

如果仍有漂移，先排查有效更新步数、NaN/Inf、数据和采样条件；再尝试降低 `ppsf_d_coef` 等单变量调整。不要把“换了 PPSF”当成结果必然平滑，也不能只凭风格变化断定系数过大。

### Timestep 采样分布

Flow Matching 训练里每个 step 要从 `(0, 1)` 区间采一个 `t` 来构造 noisy latent。不同
分布对训练效果影响很大：

| 模式 | 描述 | 何时用 |
|------|------|--------|
| `logit_normal` | SD3/Anima 默认，偏向中间 `t`；`shift>1` 推向高噪声端 | 大部分情况、不知道选什么 |
| `uniform` | 均匀采样，覆盖结构端到细节端 | 想让模型对所有 noise level 同样关注 |
| `logit_normal_low` | logit-normal 反向 shift，偏向低噪声/细节端 | 细节强化、风格 LoRA |
| `mode` | SD3 mode-distribution，集中在某个 sigma 附近 | 论文实验复现 |
| `mixed_uniform_low` | 每样本独立按 `timestep_mix_low_prob` 概率走 `logit_normal_low`，其余走 uniform | 想细节强化但保留 uniform 覆盖度 |
| `mixed_uniform_logit` | 同上但偏置端走 `logit_normal` | 想轻度推高噪声但保留 uniform 覆盖度 |
| `krea2_shift` | 按每图 token 数动态 shift（分辨率感知） | Krea 2 默认；详见 [Krea 2 训练](#krea-2-训练) |

**`timestep_shift`**：仅 `logit_normal` / `mode` 用。`>1` 偏向高噪声（结构端），`<1` 偏向
细节端。默认 `3.0` 是 SD3 经验值。

**`timestep_mix_low_prob`**：仅 `mixed_uniform_*` 用，其他 mode 忽略。`0` = 全 uniform，
`1` = 全偏置端；典型 `0.15-0.30`。

**`timestep_schedule_shift`**：作用于**最终 t**（采样完成后再做一次 SD3/FLUX shifted
schedule 偏移，公式 `t' = (t·s) / (1 + (s-1)·t)`，Möbius 变换）；跟 `timestep_shift`
不同：后者作用于 logit-normal 内部的 sigmoid 后 u 值。默认 `1.0` 是恒等。`>1` 整体
推向高噪声端；`<1` 偏低噪声端。可跟任意 mode 叠加。

### Loss Weighting（损失加权）

加权策略改变不同 noise level 对 loss 的贡献。默认 `none` 是基线；以下机制描述不代表在本项目两族和所有数据上都有已验证的质量收益，应固定其他条件做对照：

| 模式 | 适用 | 关键参数 |
|------|------|----------|
| `none` | 默认，纯 MSE | — |
| `min_snr` | 当前实现 `SNR=((1-t)/t)^2`、`w=min(gamma/SNR,1)`，下调低 t / 高 SNR 样本权重 | `min_snr_gamma` 默认 5.0；3.0–7.0 仅作试验范围 |
| `detail_inv_t` | 细节强化，损失 ∝ 1/(t+ε)，clamp 到 `[detail_inv_t_min, detail_inv_t_max]` | 默认 `[1, 5]` 跟历史一致；雾蒙蒙/低饱和画风建议 `max=3`，激进细节 `max=8` |
| `cosmap` | SD3 风格 cosine mapping | 实验性 |

**`detail_inv_t_min` / `detail_inv_t_max`**：detail_inv_t 加权曲线的下/上限（默认 `1` / `5`）。
- `detail_inv_t_min` 必须 ≥ `1.0`——因为 `1/t` 在 `t∈(0,1)` 时恒 > 1，下限 < 1.0 是配置死区。
- `detail_inv_t_min > detail_inv_t_max` 时启动期 schema 校验直接报错（fail-fast）。
- 升 `max` 让低 t（细节端）权重更激进，但 Prodigy 用户注意：单样本权重过大易主导 `d` 估计，
  建议同时开 `weight_cap_ratio=5`。

### Loss 函数（0.8.x 新增）

`loss_type` 默认 `mse`（与历史 bit-for-bit 一致）。可选 `huber`：

| 字段 | 默认 | 说明 |
|------|------|------|
| `loss_type` | `mse` | 选 `huber` 对 outlier 鲁棒，缓解极端 sample 的梯度爆炸 |
| `huber_c` | `0.15` | huber δ 系数（仅 `loss_type=huber`）；典型 `0.1–0.3`，控制 quad/linear 转折点 |

**何时试 huber**：先排查数据与数值异常，再对极端残差样本做受控比较；换 loss 不保证修复 NaN。`huber_c=0.15` 是当前默认，0.1–0.3 是试验起点，不是本地最优值结论。

**互斥**：当前配置契约禁止 Huber 与 InfoNoise 同时开启。即使内部保留 raw MSE 通道，也不代表组合可用；见下方互斥表。

### InfoNoise 自适应采样器（0.7.1 新增）

**InfoNoise 是基于 I-MMSE 信息论的自适应 timestep 采样器**（论文 arxiv 2602.18647）：训练
过程中动态估计每个 noise 区间的"信息量"，把采样集中在有效区间，跳过极高/极低噪声的低效
段。理论上能加快收敛。

**何时开**：长训练（>2000 步）+ 你确定 baseline 已经稳定收敛后想再压榨效率。短训练 / 实验
阶段保持默认关闭即可。

**关键字段**（高级模式下可见）：

| 字段 | 默认 | 说明 |
|------|------|------|
| `infonoise_enabled` | `false` | 启用开关 |
| `infonoise_N_warm` | `0` | 热身步数（=0 自动取总步数的 1/5，最少 200）。热身期走 `timestep_sampling` 选择的 baseline 分布；之后切自适应 |
| `infonoise_K` | `64` | log-σ 空间分 bin 数；高 K = 更细 |
| `infonoise_M` | `100` | 每 M 步刷新一次采样分布 |
| `infonoise_B` | `256` | 每 bin 的 FIFO buffer 大小 |
| `infonoise_beta` | `0.9` | 自适应分布对最新 batch 的响应强度。FIFO 已做底层平滑，β 偏高合理 |
| `infonoise_N_min` | `50` | 触发刷新所需的每 bin 最小样本数（必须 ≤ `infonoise_B`） |
| `infonoise_gate_pivot_c` | `0.15` | gate 函数 pivot：低于 c 的噪声区段被压低采样。默认值取论文 §5 CIFAR 报告值；设 0 走自适应选取 |

**观察是否生效**：训练时 wandb 面板会有两个指标
- `infonoise/cdf_ready`：1 = 自适应 CDF 已就绪，0 = 还在 baseline
- `infonoise/refresh_degraded_count`：每次刷新失败次数。如果一直 0 但 cdf_ready 也是 0，
  说明你的 loss 在 log-σ 上太均匀（已收敛模型），InfoNoise 没加速空间 — 关掉即可

**注意**：InfoNoise 启用后 `timestep_sampling` 字段仅用于热身期。正式阶段由自适应 CDF 接管。

**与其他训练选项的关系**：InfoNoise 用未加权 MSE 估各噪声区间的信息量；这是论文 entropy rate 推导的必要前提。下面列出已知会跟 InfoNoise 产生干扰的配置：

| 配置 | 关系 | 处置 |
|------|------|------|
| `loss_weighting != none` (`min_snr`/`detail_inv_t`/`cosmap`) | 两个机制都在重塑 σ schedule（自适应 resample vs 手工 reweight），叠加互相消磨 | schema 互斥，保存配置时报错 |
| `loss_type=huber` | huber 削峰让 outlier 区间不学，但 InfoNoise 用 raw MSE 看到 outlier 仍高 → 推 mass 进去 → 反馈环 | schema 互斥 |
| `timestep_schedule_shift != 1.0` | shift 只在 baseline 路径生效；CDF 接管后静默失效 | schema 互斥 |
| `noise_enhancement_type != none` (`offset` / `pyramid`) | 噪声增强改变 noise 形状，InfoNoise 学到的不再是 clean entropy rate profile（I-MMSE 推导假设标准高斯 noise）| schema 互斥 |
| 正则集（`reg_data_dir != null`，任意 `reg_weight`） | reg 集与 main 集分布不同（典型 booru 通用图 vs LoRA 主题）；I-MMSE 假设单分布，混入会让 schedule 学到 mixture MMSE 而非 mmse_main。InfoNoise 按 batch 内 `is_reg` flag 硬过滤 reg 样本，仅 main 样本进 schedule 学习；reg 样本仍参与梯度（按 `reg_weight` 加权） | 透明处理，无需用户操作。未来若主流用法转向多 main 分布（multi-concept LoRA），按 `docs/todo/infonoise-reg-policy-reeval.md` 重评估 |
| LoRA dropout（`lora_dropout` / `lora_rank_dropout` / `lora_module_dropout`） | 加梯度噪声，不改 mse 形状的系统性偏移 | 可同开，FIFO + EMA 双层平滑能 absorb |

### 噪声增强

| 字段 | 默认 | 用途 |
|------|------|------|
| `noise_enhancement_type` | `none` | `none` / `offset` / `pyramid` 三选一。LoRA 训练默认保持 `none` |
| `noise_offset` | `0.0` | DC 偏置强度（0-0.2，0=关闭）。让噪声 mean 偏离 0，让模型有机会学习生成极端亮度场景（pure black / pure white / 强对比）。典型范围 0.05-0.1 |
| `pyramid_noise_iters` | `0` | 金字塔噪声层数（0-6，0=关闭）。每层在 `spatial // 2^(k+1)` 尺度注入。**实际效果强度由 `pyramid_noise_discount` 决定** —— iters 单独决定覆盖的频段范围 |
| `pyramid_noise_discount` | `0.5` | 每层相对衰减系数（0.1-0.9）。**控制低频强度的核心参数**：本训练器把整体噪声 std 归一化到 1。0.1-0.4 归一化后接近标准高斯，等价于关闭；0.5-0.7 显著改变低频结构 |

**互斥约束**：`noise_offset` 与金字塔噪声**不能同时启用**。两者都在给噪声注入低频成分（pyramid 最低分辨率那层 ≈ `noise_offset` 等价物），叠加会让低频成分双倍灌入，训练目标失真。这跟 kohya 上游 [sd-scripts PR #477](https://github.com/kohya-ss/sd-scripts/pull/477) 的硬约束一致。schema 校验会强制清零反组字段，老 yaml 同开会按 `pyramid_noise_iters > 0` 优先映射到 pyramid。

### Flip Augment + Cache Latents（双份缓存）

`flip_augment` 与 `cache_latents` 同开时，训练器按 kohya 上游 `latents` / `latents_flipped` 模式存**双份 latent**（两族通用）：

- cache 阶段对每张图 encode 两次（原图 + 镜像），分别存到 npz 的 `latent` / `latent_flipped` 键
- 训练时 `__getitem__` 50% 概率取 flipped 版本，跟非 cache 路径行为对齐
- 代价：需要两次编码并保存双份 latent，缓存主体数据量约翻倍；总缓存体积和端到端耗时还取决于压缩、I/O 与合批，不保证严格×2
- 缓存阶段会按相同 bucket 尺寸合批送入 VAE（`vae_cache_batch_size`，默认 `0` = 跟随训练 batch size，对齐 kohya）；显存不足时设为 `1` 逐张编码
- 老 cache（只有 `latent`）+ `flip_augment=true` → 自动判失效，重 encode 补全；切回 `flip_augment=false` 不会反复重 encode（双份是单份的超集）

历史 bug：旧版 0.11.x 之前同开两者会让 cache 阶段那一次随机翻转 baked 进 npz，**flip_augment 永久失效 + 50% 数据被永久镜像污染**。0.11.x 起按双份方案修复，已有的污染 cache 通过 `_is_cache_valid` 自动检测重 encode。

---

## Krea 2 训练

Krea 2 是 12.9B 单流 MMDiT（文本编码 Qwen3-VL-4B，12 层中间态条件；VAE 与 Anima 共享，
latent 缓存跨族复用）。训练配置里把「模型族」切到 Krea 2 即可用同一套流水线训练——切换会
弹确认框逐项列出将重算的权重路径与族默认值，确认后生效。

### 底模选择：训练用 Raw，测试可用 Turbo

体积列是文件大小，显卡标注只是起步参考，不是完整训练显存保证；具体峰值见下方显存参考。

| 权重 | 用途 | 体积 |
|------|------|------|
| Raw bf16 | 训练 / 训练中采样（32 GB 卡） | 26.3 GB |
| Raw fp8（官方量化） | 训练 / 推理（24 GB 级卡的训练选择） | 13.1 GB |
| Turbo bf16 / fp8 | **仅测试出图**（TDM 蒸馏，8 步 / 无 CFG） | 26.3 / 13.1 GB |

- **不要拿 Turbo 当训练底模**——它是蒸馏推理模型。使用官方目录项且未显式指定训练底模时，新版本会从 Turbo 默认选择回退到 Raw；自定义路径或重命名权重需自行核对，不保证自动识别。
- **Raw 上训出的 LoRA 可以挂 Turbo 出图**。已识别的官方 Turbo 默认 8 步 / 无 CFG，显式参数优先；能加载不等于效果或最佳强度完全相同。

### fp8 底模训练（fp8_base）

选 fp8 文件当训练底模即生效，无需额外开关（权重常驻 fp8、前向逐层反量化，底模 frozen、
LoRA 参数全精度——kohya / musubi 生态的 `fp8_base` 语义）：

- 权重显存 25.6 GB → 13.1 GB；真机参考：32 GB 卡上 1024² bs1 ga2 峰值约 25 GB
- **必须开 `grad_checkpoint`**（不开则显存反超 bf16，启动期直接报错）
- LoRA / LoHa / LoKr / rs-LoRA 有兼容路径；**DoRA 不兼容**（初始化读底模权重数值，启动期报错）。这不是所有社区变体的保证，也不表示各组合有相同显存成本；本文数字主要来自普通 LoRA。
- 文本编码器也有官方 fp8 版（5.24 GB，见设置页 TE 单选），训练 + 出图全线可用；切换精度会自动重建文本缓存

### 与 Anima 的默认值差异

切换到 Krea 2 的确认操作会重算权重路径、重置以下族相关字段，并关闭不支持的能力；**即使这些族相关字段此前被手动修改，也会进入变更清单**。这不同于普通配置载入只为缺失字段补默认。确认前核对清单，需要保留原方案时先复制版本：

| 字段 | Krea 2 默认 | 说明 |
|------|------------|------|
| `timestep_sampling` | `krea2_shift` | 按每图 token 数动态 shift（分辨率感知，mu 0.5→1.15 线性插值；1024² 约等效 discrete shift 2.5，musubi 口径）。`timestep_shift` 等 Anima 旋钮不参与 |
| `sample_sampler_name` / `sample_scheduler` | `euler` / `simple` | 与 ComfyUI 里挂 Krea 2 模型时选的 **euler + simple** 逐字对应 |
| `sample_infer_steps` / `sample_cfg_scale` | 28 / 4.5 | Raw 官方口径 |
| `text_encoder_cache` | `true` | 训练开始先预编码 caption 后释放 Qwen3-VL；bf16 TE 权重约 9 GB，fp8 不同，权重体积不等于编码峰值；关闭则在线编码 |
| `attention_backend` | SDPA（配置值为 `none`） | Krea 2 加载器固定 SDPA |
| `shuffle_caption` / `keep_tokens` / `tag_dropout` | 关闭 | tag 生态操作对自然语言 caption 不适用 |

**能力差异**：NaViT 打包、SRA、LeapAlign、compile_blocks 为仅 Anima 功能（Krea 2 下自动隐藏）；
masked loss、正则集、flip augment、各优化器 / loss / lr scheduler 跨族通用。

### caption 建议

Krea 2 走 Qwen3-VL 自然语言 caption，不是 booru tag 生态：

- 打标推荐用 **LLM 打标器**（长自然语言描述）；WD14 tag 链路机制上可用但非推荐
- 触发词随实际打标写入；默认跳过已有 caption 时不补写触发词，须在标签编辑检查。采样 prompt 中也需按实验目标填写。
- 程序不主动按 512 token 截断，超过部分仍参与训练；但仍受模型上下文、显存和缓存容量约束，不建议无目的地写长。缓存约 31 MB / 512 token 是特定编码布局的估算；DiT 注意力序列同时包含文本与图像 token（1024×1024 图像侧为 4096），完整峰值需按实际配置测量。

### Block 交换（换出到内存的层数）

把 DiT 部分层权重常驻内存，计算时换入显存，用时间换显存。搬运本身不引入额外量化，**不承诺训练数值或最终图片逐位一致**。训练入口在 **系统与性能**，出图入口在 **设置 → 测试 → 显存策略**，默认 `0`（关闭）。Anima 自 0.24.0 起与 Krea 2 一样支持。层数由底模文件决定，底模列表里标有「N 层」（Anima 1.0 等为 28 层，Anima-2.9B 为 40 层；Krea 2 为 28 层）；超过总层数按全部换出处理。

显存策略管理多个模型之间的驻留（TE / DiT），交换管理单个 DiT 内的层，两者可以配合。训练包含前向、反向与 checkpoint 重算，出图每个采样步也需搬运；开销随 PCIe、内存、分辨率和配置改变。

#### Krea 2 测量口径

RTX 5090 环境的官方 fp8、1024²、普通 LoRA rank32、batch1 / 累积4、gradient checkpointing 短实验：

| 换出层数 | 训练开始前 allocated | 训练步 allocated 峰值 | 采样 allocated 峰值 | 实验覆盖 |
|---|---|---|---|---|
| 14 | 约 7.9 GB | 此处未列 | 此处未列 | 6/10 步后取消 |
| 28 | 约 2.2 GB | 约 8.4 GB | 约 7.1 GB | 8/10 步后取消 |

这是特定短实验，不是目标小卡完整训练通过记录。28层实验训练后 reserved 约10.2GB、整卡时点约13.4GB，采样后整卡时点约6.3GB。**allocated、reserved、整卡时点与全流程峰值不能混用**；8.4GB加固定上下文不能证明12GB一定可训，采样后6.3GB也不能证明8GB从冷加载到解码都能通过。小卡可能更积极回收缓存，实际结果需实卡验证，不能据此断言必败。

较新的普通 LoRA rank64 / 14层 / 1024 / batch1 完整训练到520步，训练步 allocated 峰值约21.37GB，采样峰值约10.76GB。它不是与上表的单变量对照，不能将差异只归于rank；但说明常驻显存不代表训练峰值。

“约慢4%”来自特定**采样任务14层对28层**（52–53秒对53–55秒），不是关闭交换对28层的训练成本。旧训练时序探针有约7%/2.6%的其他测量，不能替代同配置生产训练吞吐对照。详见[原始设计与后续修订](../design/block-swap.md)；其中容量判决是历史外推，当前用户建议以本节的证据边界为准。

#### Anima 与内存成本

Anima 同样可通过交换降低 DiT 驻留量，但大卡上的分配量与某次运行的速度差不足以认证最低卡容量。**6GB实卡包含采样的完整训练仍需目标卡、完整配置与环境记录验证**；没有同配置对照时，不给出通用速度损失百分比。

换出权重需要当前可用 RAM，且会被锁定、不能供其他程序使用；估算每层 Krea 2 fp8约0.4GB / bf16约0.8GB，Anima约0.13GB。加载前护栏会尽力拦截可识别不足，但查询失败、后续峰值与其他程序占用仍可能导致OOM或换页，不是整机安全保证。

### 显存参考

下表是**试用起点，不是最低硬件认证**。适用于官方fp8、低batch并开启gradient checkpointing的训练；rank、adapter、caption长度、bucket及采样都会影响峰值。加载、TE编码、merge、训练/采样与VAE解码必须完整验证。

| 显卡 | 训练起点 | 出图起点 |
|------|------|------|
| 32 GB | 优先fp8；bf16余量可能很小，必要时交换 | fp8 + 默认策略，按实际峰值选择是否驻留 |
| 24 GB | fp8；可先不交换，峰值不足时开启 | 默认或省显存，按需交换 |
| 16 GB | fp8 + 交换14层起，不足加到28；必要时降分辨率 | 省显存 + 按需交换，不能用32GB卡上的约17GB读数当通过证明 |
| 12 GB | 交换28层试验；未完成目标实卡全流程验证 | 省显存 + 交换28层；同样待实卡验证 |
| 8–10 GB | 不保证可训练；降低分辨率试验 | 省显存 + 交换28层试验，不以约6GB时点值保证成功 |

出图显存策略位于 **设置 → 测试**：默认（用后释放）/ 省显存（强制顺序化）/ 性能优先（尽量常驻）。建议保持加载期RAM/VRAM护栏开启，但仍需预留当前可用内存。
默认与省显存档在采样结束进入decode时才加载VAE，decode后移到CPU；性能优先档让VAE留在GPU。普通Krea 2 FP8 LoRA merge默认按1024行分块；已测普通LoRA的full/chunked merge逐层一致，不是所有adapter或跨软件最终图片一致的保证。

Krea 2 遇到未缓存的新 prompt 时，省显存档总会先把 DiT 移到 CPU RAM，再加载并运行
TE；默认档会记录首次 TE `load + encode` 的实际 CUDA 峰值，并结合当前空闲显存和
WDDM 预留量决定是否执行同样的顺序化。32 GB 显卡通常会顺序化，48/64 GB 显卡在余量
充足时可让 TE 与 DiT 同驻。性能优先档不移动 DiT。prompt LRU 命中时不重新加载 TE，
因此不会触发 DiT 搬运。

---

## 训练算法选项

可选的 loss / 采样 / 优化器 / adapter，在 Advanced 模式按需配置。

- **Loss 函数**：MSE / Huber，可配置权重曲线（`min_snr` / `cosmap` / `detail_inv_t` 等）。
- **Timestep 采样**：`uniform` / `logit_normal` / `mode` / `mixed_uniform` 等，含可配置 schedule shift；Krea 2 另有分辨率感知的 `krea2_shift`。
- **InfoNoise 自适应采样**（可选）：基于 I-MMSE 的反 CDF 时间步采样器。
- **自蒸馏 / 表征对齐**（可选，进阶，**仅 Anima**）：LeapAlign 两步跳跃自蒸馏（含 FlowBP 四变体）、SRA v2 中间表征对齐 VAE latent。
- **优化器**：AdamW / Lion / Automagic / CAME / Prodigy / Prodigy+ScheduleFree / SOAP / Schedule-Free SOAP（起步参数 / 切换换算见 [optimizers.md](optimizers.md)）。
- **Adapter**：LoRA / LoHa / LoKr（走 [lycoris-lora](https://github.com/KohakuBlueleaf/LyCORIS) v4 eager 基线），以及 OrthoLoRA / T-LoRA；DoRA、rs-LoRA 与 dropout 的可用性按 adapter 和模型族过滤。
- **分层 rank**：`lora_rank_rules` 按层名正则配不同 rank，便于按模块重要性差异化分配参数预算。
- **Attention backend**：xformers / flash_attn / PyTorch SDPA（Krea 2 固定 SDPA）。

---

## Schema 简单/高级模式 与历史字段迁移

Train 页和 Presets 页提供 **简单/高级** 切换，共享同一份浏览器偏好。简单模式只显示常用字段，高级模式展开进阶选项；实际可见字段还取决于模型族、adapter 和当前启用的功能，不以固定字段数为准。

以下分组和默认值表是 **0.7.x → 0.8.0 的历史迁移记录**，不是当前版本的完整 schema；当前字段定义以 `studio/domain/training.py` 和页面显示为准。

### 字段位置变化（0.7.0 → 0.8.0）

以下字段移动了所属分组，但 yaml/TOML preset key 名没变，老 preset 仍兼容加载：

| 字段 | 旧分组 | 新分组 |
|------|--------|--------|
| `kv_trim` | training | **system** |
| `mixed_precision` | training | **system** |
| `attention_backend` | training | **system** |
| `num_workers` | training | **system** |
| `grad_checkpoint` | system | **training**（紧贴 `grad_accum`） |
| `noise_offset` / `pyramid_noise_*` / `timestep_*` / `infonoise_*` / `loss_weighting` 等 | training | **noise_schedule**（新增分组） |

`lora` 分组的 UI 标签从 "LoRA / LoKr" 改为 "网络设置"（key 仍是 `lora`）。

### 默认值变化（0.7.0 → 0.8.0）

以下字段默认值改了。**老 preset 显式写过值不受影响**；走默认的需要注意节奏变化：

| 字段 | 旧默认 | 新默认 | 影响 |
|------|--------|--------|------|
| `save_every_epochs` | 0 | **2** | 每 2 epoch 保存 LoRA |
| `save_every_steps` | 500 | **0** | step-based save 默认关 |
| `save_state_every_steps` | 1000 | **0** | step-based state save 默认关 |
| `sample_every` | 5 | **2** | epoch 采样频率翻倍 |
| `sample_max_side` | 1024 | **1216** | 采样图分辨率提升 |

`save_every_epochs` / `save_state_every_epochs`（epoch 版）和 `save_every_steps` /
`save_state_every_steps`（step 版）是双轨设计：step 版写 `..._step{N}.{ext}`，
epoch 版写 `..._epoch{N}.{ext}`，文件名互不覆盖，可同时启用。老 yaml 用 `save_every` /
`save_state_every` 仍能加载（schema 自动迁移到新名）。

---

## CLI 与启动

### `--torch=<tag>` 强制指定 PyTorch CUDA 版本

`./studio.sh --torch=cu128`（也支持 `--torch=cu126 / cu124 / cu118 / cpu`）

适合场景：**CPU-only 租赁机预装 GPU torch**，方便后续切到 GPU 实例时不用再装。Ctrl+C 可
跳过本次安装；marker 文件保留到下次启动重试。

若想永久跳过 pending 重试，删 `studio_data/.pending-pip-install.json` 即可。

### `dev` 子命令的 `--fe-port`

`./studio.sh dev --fe-port 5174` —— Vite dev server 默认 5173 与其他服务冲突时用这个改端口。

---

## 常见问题

### 过拟合

**症状**：
- 训练 loss 很低，但采样图质量下降
- 生成的图和训练集几乎一样
- 无法响应新的提示词变化

**解决方案**：
1. 减少 epochs
2. 降低学习率
3. Anima tag caption 可试 `tag_dropout`（如5–15%，需检查关键标签保护）；Krea 2 不使用此选项
4. 降低 LoRA rank
5. 增加数据多样性

### 欠拟合

**症状**：
- 训练 loss 居高不下
- 采样图完全没有学到特征
- 角色/画风不像目标

**解决方案**：
1. 增加 epochs
2. 提高学习率
3. 提高 LoRA rank
4. 检查标签是否正确
5. 检查数据是否正确加载

### 角色崩坏

**症状**：
- 角色特征不稳定
- 有时正确有时错误
- 多角色混淆

**解决方案**：
1. 确保每个角色的标签一致
2. 增加角色名标签的权重（推理时）
3. Anima 的 TXT caption 可用 `keep_tokens` 保护最终文本前缀；Krea 2 不使用此 tag 选项
4. 增加训练数据

### 显存不足

**症状**：
- CUDA out of memory
- 训练中断

**解决方案**：
1. 启用 `grad_checkpoint: true`
2. 减小 `batch_size`（改用 `grad_accum` 补偿）
3. 降低 `resolution`
4. 开启 Block 交换（Anima / Krea 2 通用，用时间换显存，见「Krea 2 训练 → Block 交换」节）
5. 关闭 `cache_latents`（会变慢）
6. 使用 `mixed_precision: bf16`

### 训练中途速度骤降、显存占满但不报 OOM（Windows）

**症状**：
- 前几十步速度正常，某一步之后 it/s 掉一半以上且不再恢复
- 任务管理器 / `nvidia-smi` 显示专用显存打满，「共享 GPU 内存」开始上涨，但没有 CUDA OOM
- 日志里 torch「已分配」不高，「保留」和全卡已用远高于它

**原因**：Windows WDDM 允许 CUDA 分配超出专用显存时溢到系统内存（「Sysmem Fallback」），
之后每步都经 PCIe 搬数据。Linux 上同样场景会触发 PyTorch 分配器的「OOM → 释放缓存 → 重试」
自愈，Windows 下分配不失败，自愈永不触发。多分辨率桶（ARB）训练最容易踩到：切换到新
桶时旧桶留下的缓存段叠加新桶的激活峰值，reserved 可能一次性涨 30%。

**处理**：
- 训练器在切桶时会自动归还上一个桶的分配器缓存，多桶训练下不需要额外设置。
- 出图、评估等其他路径若仍出现同样症状，可在 NVIDIA 控制面板 → 管理 3D 设置 → 程序设置
  中，把 Python 的「CUDA - 系统内存回退策略」设为「首选无系统内存回退」：分配超限时改为直接
  报 OOM，PyTorch 的自愈路径随之生效；代价是原本「慢但能跑」的真超限场景会直接失败。

---

## 监控训练

### Loss 曲线解读

loss 只适合在数据、timestep 分布、加权方式与记录口径一致时观察趋势，不能独立判断视觉质量或过拟合。持续下降可能仍伴随泛化变差；震荡也可能来自不同噪声等级或样本。不要跨任务仅按绝对 loss 排名。

任务完成后同时核对**有效更新步数 / 目标步数**和日志中的 NaN/Inf、跳步、采样失败。历史上存在 `done / exit 0` 但有效步严重不足的记录；监控曲线只含有限 loss，也不能排除被跳过的异常更新。

### 采样图检查

每隔若干有效更新步或 epoch 保存并检查固定 prompt / 采样 seed 的样图：

1. 目标特征是否逐步出现，是否仍能响应不同动作、服装和场景。
2. 相邻 checkpoint 是否漂移，非触发提示词与不挂 LoRA 的基线差异如何。
3. 后续训练是否继续改善，还是开始复制训练图、丢失多样性。

不要把 1–5 / 5–15 / 15+ epoch 当通用阶段；数据量、repeats、batch 和梯度累积都会改变每个 epoch 的更新数和曝光量。保留配置快照之外，还需保存数据/caption、底模与环境版本；配置快照不冻结这些文件。

### 使用训练监控

走 Studio 的监控页：启动训练后打开 <http://127.0.0.1:8765/tools/monitor>，
或在训练 / 队列页里点任务进入 **任务详情 → 监控** 标签。

监控面板显示：
- 实时 loss 曲线
- 学习率变化
- 采样图预览
- 训练速度

> 旧的 `python train_monitor.py` 自带 HTTP server 已删除（详见
> `runtime/train_monitor.py` 顶部 docstring）；现在它只是个状态写入器，由
> `anima_train` 调用，不需要单独启动。

### 查看日志与调试开关

任务日志（训练 / 打标 / 预处理 / 评估 / 出图）在 **任务详情 → 日志** 标签，各步骤页
底部的日志抽屉显示的是同一份内容；出图页的「日志」抽屉显示推理进程的输出。所有日志视图
长一个样：

- 每行有时间、级别、来源；ERROR 红、WARNING 黄、DEBUG 弱化。报错先看红行，再展开它下面
  的缩进续行（traceback）。
- 默认不显示 DEBUG 行，视图左上角的 **调试** 开关可以临时打开——日志文件里始终记录了
  调试行，打开开关就能看到，不需要重跑任务。这个开关不保存；全局默认值在
  **设置 → 系统 → 日志 → 默认显示调试日志**。
- 长日志默认显示最后 2000 行；如果还有更早记录，顶部 **加载全部** 会一次取回完整历史；
  **下载** 拿到原始 `run.log`。刷新页面或网络断开重连后会自动补齐漏掉的行。
- 任务失败时详情页红框里是日志中最后一个错误块，和日志标签里看到的一致。
- 报 issue 时点详情页顶部的 **诊断包**：一个 zip，里面是该任务的 `run.log`、配置快照、训练指标快照、
  任务起止时间窗内的 `studio.log` 片段和环境摘要（版本 / 驱动 / CUDA / PyTorch），已做密钥脱敏、不含
  `secrets.json` / `credentials.json` 等敏感存储文件；发出前自己过一眼即可。不针对某个任务的诊断包在 **设置 → 系统 → 日志 → 导出诊断包**。

日志文件本身在 `studio_data/tasks/<任务 id>/run.log`；随任务删除一起删，不单独清理。

---

## 最佳实践

### 训练前

1. ✅ 验证模型文件完整
   ```bash
   python tools/validate_local_models.py
   ```

2. ✅ 检查数据集
   - 图片是否正确加载
   - 标签文件是否存在
   - 标签格式是否正确

3. ✅ 小批量测试
   ```bash
   python runtime/anima_train.py --config config.yaml --epochs 3 --save-every-epochs 1
   ```

### 训练中

1. ✅ 监控 loss 曲线
2. ✅ 定期检查采样图
3. ✅ 保存多个 checkpoint（便于回退）

### 训练后

1. ✅ 核对有效更新步数与目标步数，检查 NaN/Inf、跳步和采样失败；不能只看 done
2. ✅ 固定采样条件，对比不挂 LoRA、挂 LoRA 无 trigger、挂 LoRA 有 trigger
3. ✅ 测试不同提示词及与其他 LoRA 的兼容性，并保留数据/环境记录

---

## ComfyUI 使用

两族输出的 LoRA 都是 `lora_unet_*` 键名，直接拖进 ComfyUI 用，无需转换。

### 加载 LoRA

使用 `LoraLoader` 或 `LoraLoaderModelOnly` 节点：

```
模型路径：models/loras/my_lora.safetensors
strength_model: 0.8-1.0
strength_clip: 0.8-1.0
```

### 推荐参数

以下是比较用起点或当前默认，不是已证明的最佳质量参数。跨软件对照还需固定底模、TE、LoRA顺序/强度、seed、精度、采样实现与版本；名称或merge精度相同不保证最终图片逐位一致。

**Anima**：

| 参数 | 推荐值 |
|------|--------|
| Steps | 25-50 |
| CFG | 4-5 |
| Sampler | er_sde |
| Scheduler | simple |

**Krea 2**（与本程序测试页的默认逐字对应）：

| 参数 | Raw | Turbo |
|------|-----|-------|
| Steps | 28 | 8 |
| CFG | 4.5（Krea 官方口径） | 无 CFG |
| Sampler | euler | euler |
| Scheduler | simple | simple |

### 提示词格式

```
masterpiece, best quality, newest, safe, 
1girl, [角色名], [作品名], @[画师], 
[外观标签], [动作标签], [环境标签]
```

---

## 硬件优化

以下 yaml 是 **Anima 的起步示例，不保证任意配置都适配对应显卡**；Krea 2 的显存试用配置见 [Krea 2 训练 → 显存参考](#显存参考)。

### RTX 3090/4090 (24GB)

```yaml
batch_size: 1
grad_accum: 4
resolution: 1024
grad_checkpoint: true
mixed_precision: "bf16"
cache_latents: true
```

### RTX 5090 (32GB)

```yaml
batch_size: 2
grad_accum: 2
resolution: 1024
grad_checkpoint: true
mixed_precision: "bf16"
attention_backend: "none"  # 用 PyTorch SDPA（也可选 "xformers" / "flash_attn"）
cache_latents: true
```

### 多 GPU

目前脚本不支持多 GPU 并行，建议单卡训练。
