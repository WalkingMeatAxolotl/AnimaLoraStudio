# Anima LoRA 训练技巧

## 数据准备

### 数据量建议

| 场景 | 最少图片数 | 推荐图片数 | repeats |
|------|-----------|-----------|---------|
| 单角色 LoRA | 30 | 50-100 | 10-20 |
| 画风 LoRA | 50 | 100-300 | 5-10 |
| 多角色 LoKr | 200 | 500+ | 1-3 |

### 图片质量要求

- **分辨率**：建议 1024×1024 或更高
- **裁剪**：尽量保留完整构图，避免截断重要部位
- **多样性**：包含不同角度、表情、服装、光照
- **一致性**：如果是角色 LoRA，确保同一角色的外观一致

### 标签质量

- 使用 VLM 打标时，检查输出是否准确
- 删除明显错误的标签
- 保持标签风格一致（全小写，空格分隔）

### Caption 策略与正则集配合

想要训完的 LoRA 有「trigger off → 退回 base 默认 / trigger on → 出训练学的画风或角色」这种**开关式**控制效果，caption 必须把「要被 trigger 吸收的特征 tag」从 train caption 里删掉；正则集再去掉触发词，两边配合才能让 trigger 成为真正的 on/off 开关。

#### 核心原则

| LoRA 类型 | train caption 保留 | train caption 删掉 | 正则集排除 |
|---|---|---|---|
| **画风 LoRA** | 内容 tag（角色 / 场景 / 物体） | —（保持原样） | 触发词 + 画师名 |
| **人物 LoRA** | 触发词（人物） + 环境（场景 / 动作） | 角色特征（发色 / 眼睛 / 体型 / 标志性服饰） | 触发词（角色名） |
| **人物 + 衣服 LoRA** | 触发词（人物 + 衣服） + 环境 | 角色特征 + 衣服细节 | 触发词（角色名 + 衣服名） |

#### 为什么这样配合

- train 集 caption 删了特征 tag → 模型把这些特征**绑定到 trigger** 上
- 正则集 caption 不含 trigger（builder 自动排除 `based_on_version`；正则集排除列表里再手动加触发词 / 角色名 / 衣服名）
- 正则集图片来自 booru 自然分布或 base 模型自生成，发色 / 眼睛 / 衣服**随机分布**
- 训练时正则集提供「trigger 不在时这组环境 tag 该长什么样」的监督信号
- 结果：trigger off 时输出退向 base 模型的自然分布；trigger on 时才出训练学到的角色 / 画风

#### 不遵守的后果

如果 caption 里残留了本来应被吸收的特征 tag（如训角色没删 `silver_hair`），那 trigger off 时 prompt 写 `silver_hair` 仍会 leak 出训练集风格 —— trigger 不再是干净的 on/off 开关，更接近「加权融合」，且 LoRA 容易在「不写 silver_hair 反而劣化」上踩坑。

---

## 参数调优

### 学习率

| 场景 | 推荐学习率 | 说明 |
|------|-----------|------|
| 小数据集 (<100 张) | 5e-5 ~ 1e-4 | 防止过拟合 |
| 中等数据集 (100-500 张) | 1e-4 ~ 2e-4 | 标准范围 |
| 大数据集 (500+ 张) | 1e-4 ~ 3e-4 | 可以激进一些 |

**调试技巧**：
- 如果 loss 下降太慢 → 提高学习率
- 如果 loss 震荡剧烈 → 降低学习率
- 如果过拟合（采样图变差）→ 降低学习率或减少 epoch

### LoRA Rank

| Rank | 参数量 | 适用场景 |
|------|--------|----------|
| 8 | ~1MB | 简单画风微调 |
| 16 | ~2MB | 画风 LoRA |
| 32 | ~4MB | 单角色 LoRA |
| 64 | ~8MB | 复杂角色/多角色 |
| 128 | ~16MB | 极复杂场景（很少用） |

**经验法则**：从低 rank 开始，如果效果不够再提高。

### LoRA vs LoKr

| 类型 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| LoRA | 简单稳定，兼容性好 | 表达力有限 | 单角色、简单画风 |
| LoKr | 表达力强，参数高效 | 需要调参 | 多角色、复杂画风 |

### 优化器选择

| 优化器 | 何时用 | 关键参数 |
|--------|--------|---------|
| `adamw` | 默认。手调 lr 不嫌烦、想稳定可预期的训练 | `learning_rate` 1e-4 起步 |
| `prodigy` | 不想调 lr。**注意**：扩散 LoRA 上易出"风格突变 ep" | `prodigy_d_coef` 小数据集设 0.5 |
| `prodigy_plus_schedulefree` | **DiT LoRA 推荐**。在 Prodigy 基础上加 Schedule-Free averaged weights，sample/save 走 averaged 权重，**风格突变现象基本消失** | 见下方说明 |

#### ProdigyPlusScheduleFree (PPSF) 使用要点

Anima 是 Cosmos DiT + Flow Matching，跟 Flux/Qwen-Image 同型问题。这些社区已经把 PPSF
作为 LoRA 训练事实默认，原因是 Prodigy 在 timestep 随机性 + 小数据集场景下 `d` 估计抖动，
表现为某些 epoch 的 sample 风格突变。PPSF 通过维护 averaged weights 平滑了这个观感。

- **学习率**：固定 `1.0`（PPSF 内部估计真实步长，外部 lr 只是缩放系数；UI 会强制）
- **lr_scheduler**：**必须 `none`**（Schedule-Free 自带调度，叠 cosine 会破坏 averaged
  weights 的收敛保证；UI 自动 disable，pydantic 也会拦下）
- **ppsf_d_coef**：小数据集（<50 张）建议 `0.5`；正常 `1.0`；过拟合可试 `2.0`
- **ppsf_prodigy_steps**：建议设为总步数的 1/4 到 1/2（如总 2000 步设
  500-1000），后期冻结 `d`、防跳档；不确定就留 `0`（不冻结）
- **ppsf_fused_back_pass**：显存吃紧时开
- **save / sample 行为**：训练代码自动在 sample 和 save 前调 `optimizer.eval()` 切到
  averaged weights、事后切回。保存的 LoRA 是 averaged 状态，直接可用

#### 怎么知道 Prodigy → PPSF 是不是真的能解决我的"突变 ep"问题

切换后再训一遍同样的 dataset，对比相邻 ep 的 sample：
- 之前：某些 ep 风格突然偏离，下一个 ep 又回来或漂到新位置
- PPSF 后：sample 应该平滑过渡，没有"跳档"的视觉断层

如果 PPSF 之后仍有跳档，多半是 `ppsf_d_coef` 过大或数据集本身太小 — 降到 `0.5` 或
`0.3` 再试。

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

**`timestep_shift`**：仅 `logit_normal` / `mode` 用。`>1` 偏向高噪声（结构端），`<1` 偏向
细节端。默认 `3.0` 是 SD3 经验值。

**`timestep_mix_low_prob`**：仅 `mixed_uniform_*` 用，其他 mode 忽略。`0` = 全 uniform，
`1` = 全偏置端；典型 `0.15-0.30`。

**`timestep_schedule_shift`**：作用于**最终 t**（采样完成后再做一次 SD3/FLUX shifted
schedule 偏移，公式 `t' = (t·s) / (1 + (s-1)·t)`，Möbius 变换）；跟 `timestep_shift`
不同：后者作用于 logit-normal 内部的 sigmoid 后 u 值。默认 `1.0` 是恒等。`>1` 整体
推向高噪声端；`<1` 偏低噪声端。可跟任意 mode 叠加。

### Loss Weighting（损失加权）

加权策略改变模型在不同 noise level 上的关注度。**只有 `min_snr` 是经过广泛验证的**，其他
都是实验性的：

| 模式 | 适用 | 关键参数 |
|------|------|----------|
| `none` | 默认，纯 MSE | — |
| `min_snr` | **强烈推荐**。SD3 论文方案，缓解 t→1 端的 loss 主导 | `min_snr_gamma` 默认 5.0，可在 3.0~7.0 调 |
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

**何时用 huber**：训练时偶发 NaN / loss 跳变剧烈、或数据集有少量极端 sample 时。
EDM/Karras 论文里 δ=0.15 是常用经验值。

**注意**：启用 huber 后 InfoNoise 仍收到纯 MSE（解耦设计），所以两者可同时开。

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
| `pyramid_noise_discount` | `0.5` | 每层相对衰减系数（0.1-0.9）。**控制低频强度的核心参数**：anima 把整体噪声 std 归一化到 1。0.1-0.4 归一化后接近标准高斯，等价于关闭；0.5-0.7 显著改变低频结构 |

**互斥约束**：`noise_offset` 与金字塔噪声**不能同时启用**。两者都在给噪声注入低频成分（pyramid 最低分辨率那层 ≈ `noise_offset` 等价物），叠加会让低频成分双倍灌入，训练目标失真。这跟 kohya 上游 [sd-scripts PR #477](https://github.com/kohya-ss/sd-scripts/pull/477) 的硬约束一致。Anima 的 schema 校验会强制清零反组字段，老 yaml 同开会按 `pyramid_noise_iters > 0` 优先映射到 pyramid。

### Flip Augment + Cache Latents（双份缓存）

`flip_augment` 与 `cache_latents` 同开时，Anima 按 kohya 上游 `latents` / `latents_flipped` 模式存**双份 latent**：

- cache 阶段对每张图 encode 两次（原图 + 镜像），分别存到 npz 的 `latent` / `latent_flipped` 键
- 训练时 `__getitem__` 50% 概率取 flipped 版本，跟非 cache 路径行为对齐
- 代价：**编码时间和 cache 大小都 ×2**（小数据集无感知，大数据集要权衡）
- 缓存阶段会按相同 bucket 尺寸合批送入 VAE（`vae_cache_batch_size`，默认 `0` = 跟随训练 batch size，对齐 kohya）；显存不足时设为 `1` 逐张编码
- 老 cache（只有 `latent`）+ `flip_augment=true` → 自动判失效，重 encode 补全；切回 `flip_augment=false` 不会反复重 encode（双份是单份的超集）

历史 bug：旧版 0.11.x 之前同开两者会让 cache 阶段那一次随机翻转 baked 进 npz，**flip_augment 永久失效 + 50% 数据被永久镜像污染**。0.11.x 起按双份方案修复，已有的污染 cache 通过 `_is_cache_valid` 自动检测重 encode。

---

## Schema 简单/高级模式 与 字段位置（0.7.1 改动）

0.7.1 引入了 Train 页和 Presets 页的 **简单/高级** 切换：

- **简单模式**：只显示常用字段（学习率、rank、optimizer、采样间隔等），约 30 个字段
- **高级模式**：显示全部字段（约 65 个），包含 dropout、scheduler tuning、PPSF 细节、噪声
  schedule、InfoNoise 等

两个页面共享同一个 toggle 状态（localStorage），打开任一页面切换都会同步到另一个。

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
3. 增加 tag_dropout（5-15%）
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
3. 使用 keep_tokens 保护角色名
4. 增加训练数据

### 显存不足

**症状**：
- CUDA out of memory
- 训练中断

**解决方案**：
1. 启用 `grad_checkpoint: true`
2. 减小 `batch_size`（改用 `grad_accum` 补偿）
3. 降低 `resolution`
4. 关闭 `cache_latents`（会变慢）
5. 使用 `mixed_precision: bf16`

---

## 监控训练

### Loss 曲线解读

```
理想曲线：
  快速下降 → 缓慢下降 → 趋于平稳
  
过拟合曲线：
  快速下降 → 继续下降 → 非常低（接近 0）
  
欠拟合曲线：
  缓慢下降 → 停滞 → 居高不下
```

### 采样图检查

每隔几个 epoch 检查采样图：

1. **早期** (1-5 epoch)：应该开始出现目标特征的雏形
2. **中期** (5-15 epoch)：特征应该越来越明显
3. **后期** (15+ epoch)：质量应该稳定，注意过拟合

### 使用训练监控

走 Studio 的监控页：启动训练后打开 <http://127.0.0.1:8765/studio/tools/monitor>，
或在 ⑥ 训练 / 队列页里点任务进入 **任务详情 → 监控** 标签。

监控面板显示：
- 实时 loss 曲线
- 学习率变化
- 采样图预览
- 训练速度

> 旧的 `python train_monitor.py` 自带 HTTP server 已删除（详见
> `runtime/train_monitor.py` 顶部 docstring）；现在它只是个状态写入器，由
> `anima_train` 调用，不需要单独启动。

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

1. ✅ 在 ComfyUI 中测试
2. ✅ 测试不同提示词
3. ✅ 测试与其他 LoRA 的兼容性

---

## ComfyUI 使用

### 加载 LoRA

使用 `LoraLoader` 或 `LoraLoaderModelOnly` 节点：

```
模型路径：models/loras/my_lora.safetensors
strength_model: 0.8-1.0
strength_clip: 0.8-1.0
```

### 推荐参数

| 参数 | 推荐值 |
|------|--------|
| Steps | 25-50 |
| CFG | 4-5 |
| Sampler | er_sde |
| Scheduler | simple |

### 提示词格式

```
masterpiece, best quality, newest, safe, 
1girl, [角色名], [作品名], @[画师], 
[外观标签], [动作标签], [环境标签]
```

---

## 硬件优化

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
