# Anima LoRA 变体说明与推荐

本文说明当前训练分支中三个自研/半自研 LoRA 变体的定位、设计思路、参数作用和推荐配置：

- T-LoRA
- Ortho-Hydra
- LucidLoRA

它们不是互相替代的同一种东西，而是分别解决不同训练痛点：T-LoRA 解决不同噪声阶段容量需求不同；Ortho-Hydra 解决单一低秩子空间表达不足和专家塌缩；LucidLoRA 解决 DiT 结构中不同投影角色容量需求不同，并把 T-LoRA 风格的动态 rank、LoRA+、辅助正则和可选 FFN LoKr 合在一个更易用的入口里。

## 快速选型

| 场景 | 推荐变体 | 理由 |
| --- | --- | --- |
| 想要稳定、简单、兼容性好 | `lokr` 或 `lora` | 基线方案，参数少，最容易排查问题 |
| 数据较少，担心高噪声阶段过拟合 | `tlora` | 高噪声阶段自动降低有效 rank，减少无意义容量 |
| 想让模型在不同样本/噪声阶段走不同专家 | `orthohydra` | 多专家路由，适合更复杂的风格/概念混合 |
| 想用新的默认实验路线 | `lucid` | DiT 角色感知 rank、LoRA+、动态 rank、aux loss、LyCORIS-compatible metadata |
| 需要最大兼容和最少变量 | 不建议直接用 `orthohydra` / 高级 Lucid | 复杂变体调参成本高，先用简单路线验证数据和学习率 |

推荐默认路线：

1. 先用 `lokr` 或 `lora` 验证数据集、caption、学习率和训练流程正常。
2. 如果想试自研路线，优先试 `lucid` 的 simple 模式。
3. 如果明确需要时间步容量调度，但不想引入 Lucid 的其他机制，用 `tlora`。
4. 如果任务是多风格、多概念、多域混合，且愿意承担更高调参成本，再试 `orthohydra`。

## 共同背景：为什么 DiT 训练不一定适合固定 LoRA

Anima 这类 DiT/flow matching 训练中，同一 batch 的 timestep 表示不同噪声强度：

- `t` 接近 1：更接近纯噪声，模型主要学习粗结构和去噪方向。
- `t` 接近 0：更接近干净样本，模型更需要细节、纹理、局部风格。

固定 rank 的普通 LoRA 在所有时间步都使用同样容量，这会带来两个问题：

1. 高噪声阶段给太多 rank，容易把容量浪费在不稳定信号上。
2. 低噪声阶段 rank 不够时，细节和风格表达不足。

T-LoRA 和 LucidLoRA 都围绕这个问题做了 timestep rank schedule。Ortho-Hydra 则从另一个方向出发：不是只调 rank，而是把适配器拆成多个正交专家，用 router 决定当前样本/时间步更该使用哪个专家。

## T-LoRA

### 期望解决的问题

T-LoRA 主要解决固定 rank 在不同噪声阶段容量不匹配的问题。

普通 LoRA 在所有 timestep 都启用完整 rank。对 flow matching 来说，高噪声阶段的目标更粗，完整 rank 未必有必要；低噪声阶段才更需要完整容量表达细节。T-LoRA 希望做到：

- 高噪声阶段使用较低 rank，减少过拟合和无效更新。
- 低噪声阶段使用较高 rank，保留细节表达能力。
- 在不改变基础训练流程的前提下，给每个 timestep 一个动态有效 rank。

### 设计思路

T-LoRA 注入普通 LoRA 结构，但每个 LoRA rank 分量前面会乘一个 mask。训练时根据 batch 的 timestep 均值构造 mask：

```text
t 接近 1 -> active rank 接近 min_rank
t 接近 0 -> active rank 接近 lora_rank
```

rank 曲线由 `alpha_rank_scale` 控制：

```text
active_rank = int((1 - t) ^ alpha_rank_scale * (rank - min_rank)) + min_rank
```

因此：

- `alpha_rank_scale = 1.0`：线性变化。
- `alpha_rank_scale > 1.0`：低噪声阶段更长时间保留高 rank，到高噪声端更陡降。
- `alpha_rank_scale < 1.0`：更早降 rank，高 rank 使用区间变短。

T-LoRA 还支持按模块名正则覆盖 rank 和学习率：

- `tlora_reg_dims`：匹配模块名后覆盖 rank。
- `tlora_reg_lrs`：匹配模块名后覆盖学习率。

这适合对 `q_proj`、`k_proj`、`v_proj`、`mlp` 等模块做差异化容量配置。

### 参数说明

| 参数 | 作用 | 默认/建议 |
| --- | --- | --- |
| `lora_rank` | 最大 rank，也是低噪声阶段可用的最高容量 | 16-64 起步 |
| `lora_alpha` | LoRA scale，影响 delta 权重强度 | 通常与 rank 相同或略低 |
| `tlora_alpha_rank_scale` | timestep rank 曲线幂次 | 默认 1.0；想保细节可试 1.5-2.0 |
| `tlora_sig_type` | up/B 权重初始化方式：`random`、`last`、`first` | 默认 `last`；不稳定时试 `random` |
| `tlora_reg_dims` | 按模块名正则覆盖 rank | 高级项，先不用 |
| `tlora_reg_lrs` | 按模块名正则覆盖学习率 | 高级项，先不用 |

### 推荐配置

保守起步：

```yaml
lora_type: tlora
lora_rank: 32
lora_alpha: 16
tlora_alpha_rank_scale: 1.0
tlora_sig_type: last
```

更偏细节/风格：

```yaml
lora_type: tlora
lora_rank: 48
lora_alpha: 24
tlora_alpha_rank_scale: 1.5
tlora_sig_type: last
```

数据很少、容易过拟合：

```yaml
lora_type: tlora
lora_rank: 16
tlora_alpha_rank_scale: 0.8
```

### 注意事项

- T-LoRA 的核心收益来自动态 rank，不是参数越多越好。
- `tlora_reg_dims` 和 `tlora_reg_lrs` 很强，但也很容易把实验复杂度拉高。没有明确目标时先不要用。
- 如果你只是想要 LyCORIS LoKr/LoHa 兼容路线，不应该用 T-LoRA。

## Ortho-Hydra

### 期望解决的问题

Ortho-Hydra 主要解决单一 LoRA 子空间表达不足的问题。

普通 LoRA 学到的是一个低秩更新子空间。如果数据里混有多个风格、多个概念、多个构图模式，单一子空间可能会把这些信号挤在一起，导致：

- 不同概念互相干扰。
- rank 增大后仍然不是最优，因为容量没有结构化分配。
- 某些方向过强，另一些方向学不到。

Ortho-Hydra 的思路是：把一个 LoRA 拆成多个专家，每个专家对应不同的左奇异子空间，并通过 router 按样本和 timestep 选择专家组合。

### 设计思路

Ortho-Hydra 每个目标 Linear 层包含：

- 共享右基 `Q_basis`
- 多个左基 `P_bases`
- 可学习 Cayley rotation 参数 `S_q`、`S_p`
- 可学习幅度 `lambda_layer`
- router：根据 bottleneck 激活和 sigma feature 输出专家权重

核心流程：

1. 用 SVD 初始化共享/专家基。
2. 用 Cayley rotation 保持近似正交结构。
3. 对输入经过共享 bottleneck。
4. router 根据 RMS pooled bottleneck 和 sigma 特征输出专家 gate。
5. 多个专家输出按 gate 加权求和。

它还有 balance loss，类似 Switch Transformer 的专家均衡项，用来避免所有样本都路由到同一个专家。

### 参数说明

| 参数 | 作用 | 默认/建议 |
| --- | --- | --- |
| `lora_rank` | 每个专家的 rank | 16-32 起步 |
| `lora_alpha` | delta scale | 通常与 rank 接近 |
| `orthohydra_num_experts` | 专家数量 | 默认 8；显存/稳定性不足可降到 4 |
| `orthohydra_balance_loss_weight` | 专家均衡 loss 权重 | 默认 `5e-7`；0 可关闭均衡项 |
| `orthohydra_balance_warmup_ratio` | 前多少比例训练步不加 balance loss | 默认 0.4 |
| `orthohydra_router_lr_scale` | router 学习率倍数 | 默认 10；不稳定可降到 2-5 |

### 推荐配置

保守起步：

```yaml
lora_type: orthohydra
lora_rank: 16
lora_alpha: 16
orthohydra_num_experts: 4
orthohydra_balance_loss_weight: 5e-7
orthohydra_balance_warmup_ratio: 0.4
orthohydra_router_lr_scale: 5.0
```

更强表达：

```yaml
lora_type: orthohydra
lora_rank: 32
lora_alpha: 32
orthohydra_num_experts: 8
orthohydra_balance_loss_weight: 5e-7
orthohydra_router_lr_scale: 10.0
```

如果专家塌缩：

```yaml
orthohydra_balance_loss_weight: 1e-6
orthohydra_router_lr_scale: 5.0
```

如果训练不稳定：

```yaml
orthohydra_num_experts: 4
orthohydra_router_lr_scale: 2.0
orthohydra_balance_loss_weight: 0
```

### 注意事项

- Ortho-Hydra 比 T-LoRA 和 Lucid 更复杂，不建议作为第一个测试变体。
- 专家数量越多不一定越好。小模型层宽不够时，代码会自动降低实际 experts。
- `balance_loss_weight = 0` 可以关闭专家均衡项，但 router 仍然存在。
- router 有单独参数组，训练脚本会用 `router_lr_scale` 放大学习率。

## LucidLoRA

### 期望解决的问题

LucidLoRA 是当前分支推荐的自研实验入口，用来替代旧 StyleK 路线。它主要解决三类问题：

1. DiT 中不同模块角色需要不同容量。
   - Q/K 更偏路由和匹配，未必需要完整 rank。
   - V/out/FFN 更直接影响内容和细节，保留完整 rank 更合理。
2. 不同噪声阶段需要不同有效 rank。
   - 继承 T-LoRA 风格的 timestep rank mask。
3. LoRA rank 分量容易冗余或纠缠。
   - 用正交正则和幅值正则引导 rank 分量更分散、更可裁剪。

此外 LucidLoRA 还加入：

- LoRA+：up/B 参数组使用更高学习率。
- 可选 FFN LoKr：只在 FFN 层使用自研 LoKr 结构。
- LyCORIS-compatible metadata：默认保存为更容易被外部工具识别的 metadata 形式。
- simple/advanced UI：simple 默认隐藏容易误调的高级项。

### 设计思路

LucidLoRA 注入以下目标模块：

- self-attn q/k/v/out
- cross-attn q/k/v/out
- mlp layer1/layer2

其中：

- Q/K 使用 `rank * lucid_qk_rank_ratio`，最少为 1。
- V/out/FFN 使用完整 `lora_rank`。
- timestep mask 根据 `lucid_min_rank_ratio` 和 `lucid_alpha_rank_scale` 计算每步 active rank。
- down/A 和 up/B 分成两个 optimizer group，up/B 组学习率乘 `lucid_lora_plus_ratio`。
- 如果启用 `lucid_use_lokr_ffn`，FFN 层改用 `LucidLoKrLinear`。

辅助 loss 包括：

- `ortho_reg`：约束 B 矩阵列方向接近正交。
- `mag_reg`：对 rank 分量幅值做稀疏/放大约束。
- `mag_amplify`：控制幅值 soft mask 的陡峭程度。
- `aux_loss_weight`：整体辅助 loss 权重。
- `aux_warmup_ratio`：前多少比例训练步不启用辅助 loss。

### 参数说明

| 参数 | 作用 | 默认/推荐 | 如何关闭/中性值 |
| --- | --- | --- | --- |
| `lucid_ui_mode` | UI 参数模式：`simple` 或 `advanced` | 默认 `simple` | 不是训练参数 |
| `lucid_min_rank_ratio` | timestep 高噪声阶段最小 rank 比例 | 默认 0.1 | 设 1.0 等于不随 timestep 降 rank |
| `lucid_qk_rank_ratio` | Q/K rank 比例 | 默认 0.25 | 设 1.0 等于 Q/K 不降 rank；0 仍至少 rank 1 |
| `lucid_lora_plus_ratio` | up/B 学习率倍数 | 默认 16.0 | 中性值 1.0；不建议设 0 |
| `lucid_alpha_rank_scale` | timestep rank 曲线幂次 | 默认 2.0 | 不是开关；最小 0.1 |
| `lucid_sig_type` | up/B 初始化方式：`random`、`last`、`first` | 默认 `last` | 不是开关 |
| `lucid_ortho_reg` | 正交正则强度 | 默认 0.01 | 设 0 关闭正交项 |
| `lucid_mag_reg` | 幅值正则强度 | 默认 0.001 | 设 0 关闭幅值正则和 amplify 相关项 |
| `lucid_mag_amplify` | 幅值 soft mask 陡峭度 | 默认 2.0 | 单独设 0 不等于关闭；关闭用 `mag_reg=0` |
| `lucid_aux_loss_weight` | Lucid 辅助 loss 总权重 | 默认 1.0 | 设 0 关闭全部 aux loss |
| `lucid_aux_warmup_ratio` | aux loss 热身比例 | 默认 0.1 | 设 0 表示不热身，立即启用 |
| `lucid_use_lokr_ffn` | FFN 层使用自研 LoKr | 默认 false | false 关闭 |
| `lucid_lokr_factor` | FFN LoKr 分解因子 | 默认 None，内部按 8 处理 | 仅启用 FFN LoKr 时生效 |
| `lucid_export_mode` | 保存 metadata 模式 | 默认 `lycoris_compat` | 不是训练开关 |

注意：当前训练脚本已修正 Lucid 参数读取逻辑，显式填 `0` 不会再被默认值覆盖。也就是说 `ortho_reg=0`、`mag_reg=0`、`aux_loss_weight=0` 这类关闭方式会按预期生效。

### Simple 模式推荐

Simple 模式建议只关注：

```yaml
lora_type: lucid
lora_rank: 32
lora_alpha: 16
lucid_ui_mode: simple
lucid_min_rank_ratio: 0.1
```

默认行为：

- Q/K 使用 0.25 倍 rank。
- V/out/FFN 使用完整 rank。
- LoRA+ 默认开启，up/B 学习率倍率 16。
- FFN LoKr 默认关闭。
- 导出 metadata 默认 LyCORIS-compatible。

如果想更稳一点：

```yaml
lucid_lora_plus_ratio: 8.0
lucid_aux_loss_weight: 0.5
```

如果想先排除辅助 loss 影响：

```yaml
lucid_aux_loss_weight: 0
```

### Advanced 调参建议

保守配置：

```yaml
lora_type: lucid
lora_rank: 32
lora_alpha: 16
lucid_min_rank_ratio: 0.1
lucid_qk_rank_ratio: 0.25
lucid_lora_plus_ratio: 8.0
lucid_alpha_rank_scale: 1.5
lucid_sig_type: last
lucid_ortho_reg: 0.005
lucid_mag_reg: 0.0005
lucid_mag_amplify: 2.0
lucid_aux_loss_weight: 0.5
lucid_aux_warmup_ratio: 0.1
lucid_use_lokr_ffn: false
lucid_export_mode: lycoris_compat
```

更激进配置：

```yaml
lora_type: lucid
lora_rank: 64
lora_alpha: 32
lucid_min_rank_ratio: 0.1
lucid_qk_rank_ratio: 0.25
lucid_lora_plus_ratio: 16.0
lucid_alpha_rank_scale: 2.0
lucid_ortho_reg: 0.01
lucid_mag_reg: 0.001
lucid_mag_amplify: 2.0
lucid_aux_loss_weight: 1.0
lucid_use_lokr_ffn: true
lucid_lokr_factor: 8
```

排查配置：

```yaml
lucid_lora_plus_ratio: 1.0
lucid_ortho_reg: 0
lucid_mag_reg: 0
lucid_aux_loss_weight: 0
lucid_use_lokr_ffn: false
```

这会尽量把 Lucid 简化成“角色感知 rank + timestep mask”的形式，适合判断问题是否来自辅助正则或 LoRA+。

### 注意事项

- `lucid_qk_rank_ratio=0` 不会禁用 Q/K LoRA，只会在 adapter 内被压到最少 rank 1。
- 如果想让 Q/K 和其他层一样完整 rank，设 `lucid_qk_rank_ratio=1.0`。
- `lucid_lora_plus_ratio=0` 会让 up/B 组学习率变成 0，相当于冻结关键参数，不推荐。关闭 LoRA+ 效果应设 1.0。
- `lucid_mag_amplify=0` 不是关闭幅值正则；关闭幅值正则用 `lucid_mag_reg=0`。
- `lucid_aux_loss_weight=0` 是最干净的 aux loss 总开关。
- `lucid_use_lokr_ffn=true` 会改变 FFN 层结构，兼容性和训练行为都更实验，不建议默认打开。

## 参数关闭速查

| 目标 | 推荐设置 |
| --- | --- |
| 关闭 T-LoRA 动态降 rank | 不使用 `tlora`，或在 Lucid 中设 `lucid_min_rank_ratio=1.0` |
| 关闭 Ortho-Hydra balance loss | `orthohydra_balance_loss_weight=0` |
| 关闭 Ortho-Hydra router | 不能单独关闭；不用 `orthohydra` |
| 关闭 Lucid 正交项 | `lucid_ortho_reg=0` |
| 关闭 Lucid 幅值项 | `lucid_mag_reg=0` |
| 关闭 Lucid 全部 aux loss | `lucid_aux_loss_weight=0` |
| 关闭 Lucid LoRA+ 放大 | `lucid_lora_plus_ratio=1.0` |
| 关闭 Lucid FFN LoKr | `lucid_use_lokr_ffn=false` |
| 让 Lucid Q/K 不降 rank | `lucid_qk_rank_ratio=1.0` |
| 保持 LyCORIS-compatible metadata | `lucid_export_mode=lycoris_compat` |

## 推荐实验顺序

### 第一轮：验证训练链路

```yaml
lora_type: lokr
lora_rank: 32
lora_alpha: 16
```

目标：确认数据、模型路径、采样、保存、恢复都正常。

### 第二轮：测试 Lucid simple

```yaml
lora_type: lucid
lora_rank: 32
lora_alpha: 16
lucid_ui_mode: simple
lucid_min_rank_ratio: 0.1
```

目标：看 Lucid 默认角色感知 rank 和动态 rank 是否带来收益。

### 第三轮：关闭 aux 排查

```yaml
lucid_lora_plus_ratio: 1.0
lucid_aux_loss_weight: 0
lucid_use_lokr_ffn: false
```

目标：确认问题是否来自 LoRA+ 或辅助正则。

### 第四轮：逐个打开高级项

推荐顺序：

1. `lucid_lora_plus_ratio`: 1 -> 8 -> 16
2. `lucid_ortho_reg`: 0 -> 0.005 -> 0.01
3. `lucid_mag_reg`: 0 -> 0.0005 -> 0.001
4. `lucid_use_lokr_ffn`: false -> true

不要一次改一堆参数。一次只改一个变量，采样图和 loss 才有可解释性。

### 第五轮：复杂数据再试 Ortho-Hydra

```yaml
lora_type: orthohydra
lora_rank: 16
lora_alpha: 16
orthohydra_num_experts: 4
orthohydra_balance_loss_weight: 5e-7
orthohydra_router_lr_scale: 5.0
```

目标：测试多专家是否能缓解多风格/多概念混合时的互相污染。

## 当前建议默认

如果只保留一个推荐入口，建议：

```yaml
lora_type: lucid
lora_rank: 32
lora_alpha: 16
lucid_ui_mode: simple
lucid_min_rank_ratio: 0.1
lucid_export_mode: lycoris_compat
```

如果训练明显不稳定，先改成：

```yaml
lucid_lora_plus_ratio: 1.0
lucid_aux_loss_weight: 0
```

如果图像细节不足，再逐步增加：

```yaml
lora_rank: 48
lucid_lora_plus_ratio: 8.0
lucid_alpha_rank_scale: 1.5
```

如果风格混杂严重，再考虑 Ortho-Hydra，不要一开始就上最复杂方案。
