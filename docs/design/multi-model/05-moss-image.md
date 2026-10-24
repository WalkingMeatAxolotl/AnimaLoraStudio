# 多模型支持 · 05 MOSS-Image（第 3 个模型族）

- 状态：设计稿（对齐结论已与作者确认）。PR-1 随本稿一起提交，其余按 §9 切刀实施。
- 日期：2026-10-18
- 前置阅读：[`04-synthesis.md`](04-synthesis.md)（族接口冻结面与执行红线）、[`runtime/training/families/README.md`](../../../runtime/training/families/README.md)（加族步骤）。本文不改动 04 的冻结面，只在其框架内回答「MOSS 怎么接」。
- 相关：#605（int8 量化底模识别）、[`docs/design/block-swap.md`](../block-swap.md)。

## TL;DR

- MOSS-Image = 5B 单流 DiT + 冻结 Qwen3.5-9B（TE）+ FLUX.2 VAE，Apache-2.0。官方用户指南推荐 booru tag 串（Quality→Score→Meta→Year→Character→Series→Artist→General，画师写 `@名字`），civitai 已有画风 LoRA——与本项目二次元 LoRA 主线相关。
- **两种存储格式都是一等公民**：DiT 与 TE 都同时支持 bf16 和 ComfyUI `int8_tensorwise + convrot`。int8 以 int8 原样常驻显存、前向逐层反量化（与 Krea2 fp8 路径同构），**不在加载时转成 bf16**——否则显存与 bf16 一致，量化就白做了。
- **latent 布局选「BN 归一化后、未 patchify 的 32ch f8」**：它与 ComfyUI `Flux2` latent（128ch/16）只差一次纯 reshape，却让 `LatentSpec` 取值与现有共享代码的 stride 8 / patch 2 / 对齐 16px 假设完全一致，dataset / 缓存 / masked loss 零修改（§5.1）。
- LoRA 按 HF 命名输出（`lora_unet_layers_{i}_proj_{q,k,v,o,ffn_up_gate,ffn_down}`），ComfyUI 核心（#17095）会把分开的 q/k/v 映射到融合 qkv 的切片，bf16 / int8、融合 / 不融合底模都能加载（§5.7）。
- 能力集对齐 Krea2（`masked_loss` / `text_cache` / `block_swap`）；出图对齐 ComfyUI；显存不做档位承诺，复用 Krea2 的显存编排与 block swap，优化到哪儿算哪儿。
- ComfyUI 对 int8 底模**不做常驻反量化**：无 LoRA 时直接跑 W8A8 int8 矩阵乘；挂 LoRA 时把 LoRA 合进权重后**重新量化回 int8**（逐行重算 scale + 随机舍入），再跑 int8 矩阵乘（§5.9）。所以训练走 W8A16（LoRA 旁路、数值更准），studio 出图走 Comfy parity（merge → requant → W8A8），与 Krea2 fp8 的「训练 dequant / 出图 merge parity」同一套分工（M9）。

---

## 1. 背景与目标

在 Anima、Krea 2 之后接入 MOSS-Image 作为第 3 个模型族：支持在 Base / SFT 上训练 LoRA，并在 studio 内出图、做评估。作者实际的推理环境是 ComfyUI（int8 TE + int8 DiT），训练出的 LoRA 必须能直接放进这套环境使用。

## 2. MOSS-Image 档案（训练视角）

| 组件 | 内容 |
|---|---|
| 主干 | 5.08B 单流 DiT：42 层、dim 3072、24 头 × 128、FFN 8192（SiTU-GLU，`ffn_up_gate` 融合）、QK RMSNorm、共享 modulation（全模型一个 `shared_block_modulation`）、前置 2 层文本 refiner（`text_embedder`，带 bias） |
| 位置编码 | 4D RoPE（r 参考图序号 / f 序列位置 / h / w），每轴 32 维。无参考图时 r≡0，前 32 维不旋转。**序列顺序 = 图像在前、文本在后**；图像 token 的 f = **该样本的文本长度**，h/w 以 0 为中心 |
| 训练目标 | rectified flow 速度预测 v = ε − x₀；timestep = σ × 1000（正弦嵌入 cos 在前）；loss 按 token 平均，无 SNR 加权（技术报告式 3） |
| 训练 t 采样（官方预训练） | Stage-3 末段：S = 1 − sigmoid(Z)、Z ~ N(−1.09, 0.95²)，截断 a = 0.95 后拉伸回 σ∈[0,1]，混 1% 均匀，1000 格离散网格，σ 中位数 ≈ 0.78（技术报告附录 A.2 式 4、表 6）。scheduler 配置 `training_shift: 1.0` 指「在此之外不再 shift」，偏移已含在 μ 里 |
| 推理（官方） | Euler，50 步，CFG 3，shift 5（静态），负分支跳第 22 层（0-index，每步仍两次 forward），σ 吸附到 1000 步训练网格 |
| 文本条件 | Qwen3.5-9B（GDN 线性注意力 + 全注意力 3:1 混合），取 last_hidden_state（4096 维）；chat template `<|im_start|>user\n{caption}<|im_end|>\n<|im_start|>assistant\n<think>\n`；**每条 caption 单独编码、不 padding**（官方注释：避免 GDN 受 padding 影响）；上限 4096 token；**官方要求 TE 冻结** |
| VAE | FLUX.2 VAE（`AutoencoderKLFlux2`）：32ch、f8；2×2 patchify 成 128ch 后，用 VAE 自带 `bn.running_mean/var` 归一化 |
| 许可 | 权重与推理代码 Apache-2.0；comfy-kitchen（convrot 实现来源）Apache-2.0 |
| 参考实现 | HF 仓库自带 `pipeline_moss_image.py`（diffusers 0.38 custom pipeline）；ComfyUI 核心 `comfy/ldm/moss_image/`（10-16 合入，#17095；10-17 #17099 融合 situ） |

## 3. 分发形态与资产（2026-10-18 核实）

| 来源 | DiT | TE | VAE |
|---|---|---|---|
| HF `OpenMOSS-Team/MOSS-Image-{Base,SFT}` | diffusers 分片 bf16，q/k/v 分开 | `text_encoder/` bf16 分片 16.8GB（**Base 与 SFT 逐文件 sha256 相同**） | `vae/` 小 decoder 版 0.25GB（两 repo 相同） |
| `Comfy-Org/MOSS-Image`（HF / ModelScope） | 单文件 bf16 10.16GB、int8_convrot 5.41GB（Base / SFT 各一），qkv 融合 | **只有** `qwen3.5_9b_int8_convrot` 9.75GB 和 `w4a8_gptq` 7.34GB | `flux2-vae` 标准 decoder 0.34GB |
| civitai 官方页 | 同 Comfy-Org 一套（各文件字节数逐一对上），默认下载为 int8 | 同上 | 同上 |

作者本地（`G:\ComfyUI-aki-v1.6`）用的是 `mossImage_sft_pruned_int8` + `mossImage_sft_txt_int8` + `flux2-vae`，header 与 Comfy-Org 文件一致：

- DiT：168 个量化 Linear 全部是 `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`；`text_embedder` 与各 embedder / modulation / 输出头保持 bf16。
- TE：317 个量化 Linear，组大小有 256 / 64 / 16 三种（按 in_features 可整除性逐层取）；GDN 的 `in_proj_a/b` 保持 bf16；文件里含 `model.visual.*` 视觉塔（我们用不到）。

## 4. 已确认决策

| # | 决策 | 理由 / 备注 |
|---|---|---|
| M1 | **TE 同时支持 bf16（HF 目录）与 int8_convrot（Comfy 单文件）** | 见下方「TE 显存」。作者推理用 int8 TE，训练用同一个 TE，缓存的 embedding 与部署环境一致 |
| M2 | **DiT 支持 Comfy 单文件 bf16 与 int8_convrot**；不收 diffusers 分片目录（loader 键映射仍容忍两种命名，便于测试） | bf16 10.2GB 超出多数消费卡的可用显存；int8 与 Krea2 的 fp8 同属「量化底模常驻」方案。在 int8 底模上训练、在 int8 底模上推理，环境一致 |
| M3 | **默认训 Base，SFT 同列可选** | 见下方「Base vs SFT」 |
| M4 | **不承诺显存档位**；接入 Krea2 的显存编排（让位 / 三档）与 block swap，能优化到哪儿算哪儿 | — |
| M5 | **能力集对齐 Krea2**：`masked_loss`、`text_cache`、`block_swap` | navit / sra / leap / compile_blocks 首版不开。`caption_tag_ops` 与 `cached_varlen` 被 `validate_spec` 判为互斥（缓存键 = caption hash），所以 tag shuffle / dropout 首版不可用，与 Krea2 相同 |
| M6 | **训练 t 采样默认 logit_normal(0,1) + shift 3.5**，逼近技术报告 Stage-3 末段分布（见下方「训练 t 分布」）。2026-10-24 修订：原定 shift 1 误把 `training_shift=1.0` 当成官方 shift | 只是族默认 overlay（`FAMILY_CONFIG_DEFAULTS`，仅在字段缺失时填入），`timestep_sampling`（logit_normal / uniform / mode / mixed_*）、`timestep_shift`、`timestep_schedule_shift`、`timestep_shift_resolution_aware` 照常可改。已落盘 `timestep_shift` 的存量 version 不受影响 |
| M7 | **出图对齐 ComfyUI**（用户实际部署环境），而不是官方 diffusers 的 σ 吸附调度 | 负分支跳层对齐 Comfy `SkipLayerGuidanceDiTSimple`（只改 uncond，`single_layers`） |
| M8 | **TE 缓存指纹包含存储形态**（bf16 / int8_convrot） | 两种 TE 产出的 embedding 有微小差异，切换 TE 时应重建缓存而不是混用。Krea2 现在的指纹不区分 fp8 / bf16，MOSS 不沿用这个做法 |
| M9 | **int8 底模的两条前向**：训练 = W8A16（int8 常驻、逐层临时反量化、LoRA 旁路）；studio 出图 / 评估 = ComfyUI parity（LoRA merge 进 int8 → requant → W8A8） | 由 M7 推出。训练要的是梯度与数值准确，出图要的是「和用户在 ComfyUI 里看到的一样」。Krea2 fp8 已是同一分工（`lora_fp8_merge.py` 逐位复刻 Comfy merge） |

**TE 显存（M1 的依据）**：TE 只在文本缓存阶段存活——按 Krea2 的模式，先缓存全部 caption、释放 TE，再加载 DiT 开始训练。所以**训练阶段的显存与 TE 格式无关**，TE 格式只影响缓存阶段的峰值：bf16 权重约 16.8GB；int8 原样常驻约 9.75GB（不加载视觉塔还会更小）。如果加载时把 int8 反量化成 bf16，缓存阶段又回到 17GB，所以 int8 必须常驻、前向逐层反量化。LoRA 训练本来就不训 TE，官方也明确要求 TE 冻结。

**训练 t 分布（M6 的依据）**：报告末段 S = 1 − sigmoid(Z)、Z ~ N(μ, s²)，即 σ ≈ sigmoid(−Z)，−Z 的均值为 +1.09。本项目 `logit_normal` 的 shift 是 Möbius 变换 `t' = s·t / (1 + (s−1)·t)`，作用在 sigmoid 之后等价于 logit 上加 ln s，所以 μ = −1.09 约等于 shift e^1.09 ≈ 3。报告首区间（第 1–140k 步）的配方就是 logit-normal(0,1) + shift 3。末段的截断拉伸（σ = S / 0.95）又把分布往高噪声端推了一点，用现有旋钮逼近时 shift 3.5 最贴（σ 中位数解析值 3.5·0.5 / (1 + 2.5·0.5) = 0.778）：

| 分布 | σ 中位数 | P(σ<0.2) | P(σ>0.9) | 与报告末段的总变差距离 |
|---|---|---|---|---|
| 报告末段（μ = −1.09，s = 0.95，a = 0.95，1% 均匀） | 0.781 | 0.6% | 21.5% | — |
| logit_normal + shift 1（修订前默认） | 0.500 | 8.3% | 1.4% | 0.46 |
| logit_normal + shift 3（报告首区间） | 0.750 | 0.6% | 13.6% | 0.08 |
| **logit_normal + shift 3.5（默认）** | 0.778 | 0.4% | 17.2% | 0.058 |

剩余差距来自截断拉伸与离散网格，对 LoRA 训练影响很小，不为此新增 MOSS 专用采样器。`tests/test_moss_family.py` 按式 4 算出报告末段中位数，锁定族默认与它的差距。

**Base vs SFT（M3 的依据）**：用 HTTP Range 从两个官方仓库取同名权重比较，`‖W_sft − W_base‖ / ‖W_sft‖` 在第 2 / 20 层约 1.4–1.5%，第 40 层约 0.7–0.8%，和 int8 量化误差同一量级（见 §5.4 表）。也就是说，SFT 是在 Base 上做的一次轻量微调。经验上：
- LoRA 学的是「相对于所训底模的残差」。主要在 SFT 上推理的话，在 SFT 上训更贴合（Base 上训出的 LoRA 部分是在修 Base 自身的缺陷，SFT 已经修过一部分，叠上去可能过量）。
- 在 Base 上训的 LoRA 更容易迁移到各种衍生模型（SFT、社区 finetune）。
- 两种底模结构完全相同，UI 让用户随时切换；默认 Base 与官方「Base 留给社区训 LoRA」的定位一致。

## 5. 关键设计

### 5.1 latent 空间：归一化后、未 patchify 的 32ch f8

共享代码对 latent 布局有硬编码：`runtime/training/dataset.py` 直接用 `_ANIMA_LATENT`（`ANIMA_SPEC.latent`）的 `align_px` / `spatial_stride` / `patch_spatial` 计算对齐、mask 与 token 数，只有缓存指纹用的是注入的 `latent_spec`。

候选布局有两种：

| 布局 | LatentSpec | 共享代码 |
|---|---|---|
| A. ComfyUI 原样：patchify + BN 后的 128ch、f16 | channels 128 / stride 16 / patch 1 | dataset 里 stride 8 的硬编码全错，要改成按族注入 |
| **B. BN 归一化后再 unpatchify 回 32ch、f8** | channels 32 / stride 8 / patch 2 | 与 Anima 取值相同，**零修改** |

**选 B。** 依据：
- BN 归一化是逐 (通道, 子像素位置) 的仿射变换，patchify 是纯 reshape，两者可交换：B 的张量就是 A 的张量换个排布，信息完全等价。
- reshape 保持 iid，所以在 B 布局里加的高斯噪声，换到 A 布局仍是标准高斯——共享循环里的 `noisy = (1−t)·x₀ + t·ε`、`target = ε − x₀` 不需要任何改动。
- DiT 的 `forward_train` 入口做 2×2 patchify（与 Anima DiT 内部 patchify 同构），出口 unpatchify 回 32ch，masked loss 在 f8 分辨率上算，和现有逻辑一致。
- 预览直接用 ComfyUI `latent_formats.Flux2` 的 32 行 `latent_rgb_factors`——Comfy 自己也是先 `latent_rgb_factors_reshape` 回 32ch f8 再乘这组系数。

新增 `latent_spaces.FLUX2_F8C32_BN`（指纹建议 `flux2-f8c32-bn`），rgb 系数取自 ComfyUI `Flux2`（GPL-3.0，THIRD_PARTY_NOTICES 同 Wan21 的处理方式）。这个空间不与 anima / krea2 共享缓存；以后再有 FLUX.2 VAE 的族，直接引用同一实例。

**技术债备注**：`dataset.py` 部分走注入的 `latent_spec`、部分走 `_ANIMA_LATENT`，B 布局下两者数值相同所以能工作，但这不是结构上的保证。本次不顺手修（`docs/AGENTS.md` §4.1 一个 PR 一件事），在 §10 记作遗留项。

### 5.2 VAE 包装（PR-1 已实施）

- **结构**：`modeling/flux2/autoencoder.py` 原样移植 BFL 官方 `black-forest-labs/flux2` 的 `src/flux2/autoencoder.py`（Apache-2.0）。模块路径就是 checkpoint 键名，Comfy / civitai 分发的 `flux2-vae.safetensors`（`encoder.*` / `decoder.*` / `bn.*`）可以 strict 加载，不需要键名转换。
  - 没有用 diffusers `AutoencoderKLFlux2`：仓库目前不 import diffusers，而 `requirements.txt` 的下限是 0.32，远低于 FLUX.2 VAE 所需的版本；diffusers 也没有 FLUX.2 VAE 的单文件转换器。
- **适配层**：`runtime/training/vae_flux2.py` 的 `Flux2LatentAdapter` 暴露与 WAN VAE 相同的 `encode(x, scale)` / `decode(z, scale)` / `z_dim` 接口，交给现有 `VAEWrapper` 复用整图 / 分块决策与 OOM 回退。归一化必须放在适配层：共享循环（`vae.model.encode`）和缓存分块（`vae._tiled_encode`）都绕过 `VAEWrapper.encode` 直接调底层模型。`scale` 是 `[bn_mean, 1/bn_std]`（128 维，patchify 后的通道序），与 WAN 的 `scale` 参数同构。
- **`VAEWrapper` 的两处泛化**：分块 encode 的累加器通道数改为读底层模型的 `z_dim`（原先写死 16）；新增 `latent_align`（FLUX.2 = 2），分块尺寸须为 `8 × latent_align` px 的整倍数——FLUX.2 的归一化按 2×2 子像素位置取统计量，块起点落在奇数格会让块内 patch 分组与全图错位。
- **encode 取 mean（mode）**：BFL 官方 `AutoEncoder.encode` 就是取 `chunk(moments, 2)[0]`，diffusers FLUX.2 pipeline 编码参考图也用 `sample_mode="argmax"`（§11 原第 2 项已确认）。
- **只支持 BFL / Comfy 格式单文件**。HF 仓库里的 diffusers 格式小 decoder 版（0.25GB）不支持：它的键名是 diffusers 命名，decoder 通道数也不同（96/192/384/384），资产清单（§7）本来就只列 Comfy 的 `flux2-vae`。
- **实测验证**（作者本地 `flux2-vae.safetensors`，RTX 5090，512×384 真实图片）：

  | 对比 | 结果 |
  |---|---|
  | 我们 fp32 vs ComfyUI fp32 VAE encode（patchify 后逐元素） | 相对误差 0.04%，最大绝对差 0.0021 |
  | 重建 PSNR：我们 fp32 / ComfyUI fp32 | 39.76 dB / 39.76 dB |
  | 我们 bf16 vs ComfyUI fp32 | 相对误差 2.3%，重建 39.58 dB |
  | 分块 encode（256px 块、64px 重叠）vs 整图 | 相对误差约 15% |

  分块 encode 本来就是近似（encoder 的 mid block 有全局 attention），15% 的偏差偏大。MOSS 族接入时缓存分块的默认值应取更大的块，或者默认不分块；这一点留到 PR-4 定族默认值时一并决定。
- VAE 以 fp32 运行（官方默认）。bf16 下 BN 统计量也会跟着降精度，上表的 2.3% 包含了这一部分。

### 5.3 DiT 建模与加载

- `modeling/moss/`：从官方 `pipeline_moss_image.py`（Apache-2.0）移植结构定义，模块路径沿用 HF 命名（`layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`），保证 kohya 键名与 ComfyUI 的 LoRA 映射一致（§5.7）。
- `rope.bands` 是派生常量：建模侧 `persistent=False`，加载时剥掉（#601 的教训；ComfyUI 的 `process_unet_state_dict` 也会 pop 掉它）。
- loader 接受两种单文件：
  - **bf16（qkv 融合）**：`proj.qkv.weight [9216, 3072]` 按行三等分，拆成 q/k/v。
  - **int8_convrot（qkv 融合）**：int8 权重和逐行 `weight_scale [9216, 1]` 都按行三等分。旋转只作用在输入维（列），行与行互相独立，所以按行拆分是精确的。每层的组大小从 `comfy_quant` blob 读取。
  - 层数从 header 数（`count_blocks` 同口径），不查表。
- 其他量化格式（w4a8、nvfp4、int4、非 convrot 的 int8 等）沿用 #605 的识别与报错文案；#605 的判定函数在 MOSS 族里从「拒绝」改为「按格式分流」。
- `forward_train`：v1 照官方实现**逐样本循环**（每个样本按真实文本长度构造 RoPE，图像 f 坐标 = 文本长度，text refiner 与联合注意力都只看有效 token）。pad + mask 的批量化是后续性能优化，不进 v1。

### 5.4 int8_convrot 运行时

**格式约定**（取自 comfy-kitchen 0.2.38 `tensor/int8_utils.py` 与 `backends/eager/quantization.py`）：

- H = 归一化的规则 Hadamard 矩阵（由 4×4 基元 kron 展开，除以 √g），对称且自逆（H = Hᵀ，H² = I）。
- 存储：`W_rot = W @ Hᵀ`（按输入维分组，每组 g 列），再逐行 int8 量化，`weight_scale` 为逐行 F32。
- 反量化：`W = (q · s) @ Hᵀ`（因为 H 自逆）。
- Comfy 前向：`x_rot = x @ H`（分组）→ 动态逐行 int8 量化 → `int8_mm`（W8A8）。

**实测验证**（作者本地 int8 DiT vs 官方 HF bf16，逐层相对误差 `‖Ŵ − W‖ / ‖W‖`）：

| 层 | int8 反量化 vs SFT bf16 | Base vs SFT |
|---|---|---|
| layers.2.proj.o | 0.89% | 1.46% |
| layers.2.proj.ffn_down | 0.91% | 1.49% |
| layers.20.proj.o | 0.88% | 1.37% |
| layers.20.proj.ffn_down | 0.91% | 1.49% |
| layers.40.proj.o | 0.88% | 0.69% |
| layers.40.proj.ffn_down | 0.91% | 0.82% |

如果约定理解错了（比如旋转方向反了），反量化结果会接近随机正交变换，误差在 ~140% 量级。实测 ~0.9% 正是逐行 int8 量化的正常误差，约定确认无误。

**我们的前向（W8A16）**：int8 原样常驻；每层前向时 `x_rot = x @ H`（分组），`y = F.linear(x_rot, (q·s).to(compute_dtype), bias)`，反量化出的权重是临时张量、随用随释放。
- 反向不需要自定义 autograd：`x @ H` 本身是可微的 matmul，底模权重 detach，autograd 会自动给出 `grad_x`。
- 显存收益依赖梯度检查点（临时权重随重算段释放），沿用 Krea2 fp8 在 `phases/models` 启动期的强制校验。
- 与 Comfy W8A8 的差异：训练前向不量化激活，数值上更接近 bf16。studio 出图另走 Comfy parity 路径（§5.9、M9），不复用这条训练前向。
- H 的构造约 30 行，从 comfy-kitchen 移植并在 THIRD_PARTY_NOTICES 署名（Apache-2.0）。

**LyCORIS**：`utils/lycoris_adapter.py` 现在只对 fp8 底模的 LoKr 切 bypass，规则要扩展到 int8 底模（判定条件从「fp8 dtype」改为「量化底模」）。DoRA（`weight_decompose`）需要底模权重范数，v1 在量化底模上拒绝并给出明确提示。

**block swap**（PR-5 实施）：换出层的 int8 权重与浮点参数由 loader 直接落 CPU pinned、不经过显存；`weight_scale` 是非持久 buffer、不归 swap 管，且每层只有几 KB，恒放计算设备（跟着权重落 CPU 会让前向每步多一次 H2D）。`PinnedPacker` 按实际 dtype 计算字节（int8 换出量约为 bf16 的一半）；swap 比例沿用「按参数比例、不按字节」的口径（见 Krea2 `swapped_param_ratio` 的注释）。层容器叫 `layers`，共享代码经族接口 `dit_blocks(model)` 取，不读 `model.blocks`。

### 5.5 TE 加载与文本缓存

- **bf16**：HF 目录（`OpenMOSS-Team/MOSS-Image-Base` 的 `text_encoder/` + `tokenizer/`，Base 与 SFT 共用同一份资产）。只加载语言模型部分，跳过视觉塔。
- **int8_convrot**：Comfy 单文件。照 Krea2 `_load_comfy_single_file_te` 的结构走：meta 构造、前缀映射（`model.language_model.*`）、丢弃 `comfy_quant` blob、收集 `weight_scale`、给量化 Linear 挂 int8 前向（§5.4，组大小逐层读取）；跳过 `model.visual.*`。tokenizer 需要随资产一起下载（单文件里不带）。
- 编码：chat template（见 §2）→ 逐条不 padding 前向 → last_hidden_state → 截断到 4096。TextSpec `strategy="cached_varlen"`，缓存变长 `[L, 4096]`。
- 缓存指纹 = TE 身份 + 存储形态（M8）+ 模板版本。
- 依赖：本地已满足 transformers 5.8（官方要求 ≥5.3，含 `qwen3_5`）、torch 2.11。**但 `requirements.txt` 的 transformers 下限是 4.57**，用户的旧 venv 不一定有 `qwen3_5`；PR-3 需要决定是抬全局下限，还是只在 MOSS 族加载 TE 时检查版本并给出明确提示。fla 与 causal-conv1d 没装，Qwen3.5 的 GDN 会走 torch 慢路径；这只影响一次性的缓存阶段，速度在 PR-3 实测，**不新增依赖**（新增依赖需要 ADR）。

### 5.6 采样（对齐 ComfyUI）

- 模型采样：`ModelSamplingDiscreteFlow`，shift 5.0、multiplier 1.0（DiT 内部 `timestep_embedding` 的 `time_factor=1000`），与 ComfyUI `supported_models.MOSSImage` 一致。
- 默认值取官方推荐：euler / simple / 50 步 / CFG 3 / uncond 跳第 22 层，全程生效。sampler / scheduler 白名单写进 `FAMILY_SAMPLING["moss"]`，至少要包含 er_sde（作者本地工作流在用）。
- uncond 跳层：语义对齐 Comfy `SkipLayerGuidanceDiTSimple`——只在 uncond 分支跳过指定的 `single_block`，并带 start / end 百分比区间。schema 暴露层号列表与区间两个参数。
- **parity 验收用例**取作者本地 `MOSS.json` 工作流：int8 DiT + int8 TE + flux2-vae，er_sde / simple / 30 步 / CFG 10 / `SkipLayerGuidanceDiTSimple(single_layers=22, 0–0.5)`、832×1216。同 seed 出图，与 Comfy 结果做对比。

### 5.7 LoRA 产物

- target：`layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`（42 × 6 = 252 个 Linear）。`text_embedder`、各 embedder、modulation、输出头首版不进默认 preset。
- 键名：`lora_unet_layers_{i}_proj_{name}`（`LoraOutputSpec(prefix="lora_unet")`）。ComfyUI `comfy/lora.py` 的 MOSS 分支会把 `...proj.q/k/v` 映射到融合 `proj.qkv` 的行切片 `(0, i·3072, 3072)`；非融合底模走通用映射。所以同一个 LoRA 文件能在 bf16 / int8、融合 / 非融合四种底模上加载。v1 用测试锁定：拿 Comfy 的映射规则对我们产出的键名做覆盖检查。
- 元数据：沿用 #503 的层数兼容契约（写入层数，出图侧用 `check_lora_compat` 判定）。

### 5.8 训练配置默认值（`FAMILY_CONFIG_DEFAULTS["moss"]`）

在 Krea2 的基础上调整：`text_encoder_cache=True`、`shuffle_caption=False`、`tag_dropout=0.0`（与缓存互斥，见 M5）、`timestep_sampling="logit_normal"` + `timestep_shift=3.5`（M6）、`timestep_shift_resolution_aware=False`，采样默认值见 §5.6。所有值都只在字段缺失时填入，显式配置不改写。

### 5.9 ComfyUI 运行时对 int8 的处理（部署环境事实）

依据：作者本地 ComfyUI（`G:\ComfyUI-aki-v1.6`，commit `5cc4b8700`，2026-10-16）的 `comfy/ops.py`、`comfy/model_patcher.py`，以及 comfy-kitchen 0.2.38。

| 场景 | ComfyUI 实际做法 | 代码位置 |
|---|---|---|
| 无 LoRA | **不反量化权重**。激活按组乘 H 旋转 → 逐行动态 int8 量化 → int8 矩阵乘（W8A8）→ 乘回 `weight_scale × x_scale` | `MixedPrecisionOps.forward` 的 `_use_quantized` 分支 → ck `int8_linear` |
| 挂 LoRA（默认 dynamic VRAM） | LoRA 挂成 `weight_lowvram_function`：取权重时**反量化**到 compute dtype → `calculate_weight` 把 ΔW 加进去 → `requantize_from_float(scale="recalculate", stochastic_rounding=seed)` **重新量化回 int8** → 结果写回常驻缓存 → 仍走 W8A8 | `ops.resolve_cast_module_with_vbar.post_cast` |
| 挂 LoRA（非 dynamic 的 ModelPatcher） | 加载时 `convert_weight`（反量化）→ 加 ΔW → `set_weight` 重新量化（同样重算 scale + 随机舍入） | `model_patcher.patch_weight_to_device` / `ops.set_weight` |
| `full_precision_mm` 或层带 `weight_function` | 回退：每次前向反量化成 compute dtype 再算普通 `F.linear`（W8A16） | `forward_comfy_cast_weights` |

对我们的含义：
- **LoRA 在部署时会被重新量化**。int8 的逐行量化步长约为该行最大值的 1/127（实测相对误差约 0.9%）。ΔW 中小于半个步长的分量会被随机舍入：期望上无偏，但会叠加噪声。低强度使用（例如 strength 0.3）时，LoRA 效果更接近噪声底。studio 出图如果走训练用的 W8A16 旁路，看到的效果会比 ComfyUI 里略「干净」——这正是 M9 要求出图走 parity 路径的原因。
- **parity 实现**照搬 Krea2 `lora_fp8_merge.py` 的做法：以 comfy-kitchen eager 路径为 oracle，逐步复刻 dequant（`q.float() × scale`，再按组乘 Hᵀ）→ merge（Comfy 的 `lora_compute_dtype` 与 delta 公式）→ requant（旋转 → 逐行 amax/127 → `_round_int8` 随机舍入，seed = Comfy `string_to_seed("diffusion_model.{layer}.weight")`）→ W8A8 前向（`torch._int_mm`）。CUDA 与 CPU 的随机数序列不同，逐位 parity 只在 GPU 上成立（与 Krea2 相同）。
- TE 不挂 LoRA，所以 ComfyUI 编码时 TE 恒走 W8A8；我们缓存用的 W8A16 与之只差激活量化。§11-3 会量化这个差异。

**PR-6 实施结论（作者本地 ComfyUI 0.41.0 + comfy-kitchen 0.2.38 + torch 2.7 cu128 实测）**：

- ComfyUI 在 torch < cu130 时禁用 comfy-kitchen 的 cuda / triton 后端，int8 线性层、量化、RoPE、AdaLN 全部走 eager 纯 PyTorch 实现——这条路径可以逐位复刻。cuda 后端的 `int8_linear` 与 eager 约 31% 的输出不同，所以 oracle 必须是 eager。
- **DiT 注意力**：Comfy-Org 的 MOSS 文件每层带 `{"attention": "comfy_kitchen_int8"}`，走 comfy-kitchen 自带的 INT8 SageAttention CUDA 内核（闭源二进制），不受后端禁用影响。本项目 `attention_backend=none` 时用 SDPA，并照搬 `comfy/ops.py` 的后端优先级（≥128K 元素时 FLASH → CUDNN → EFFICIENT → MATH）；选 flash_attn / xformers 时走对应的包（`modeling/attention.py`，训练前向同样适用），两者都不在逐位范围内。
- **TE 不是 W8A8**：编码时 `full_precision_mm`，每次前向把 int8 权重反量化成 compute dtype 再普通 `F.linear`。本项目 TE 改走同一条反量化路径，int8 TE 的缓存指纹改后缀 `-int8convrot-dq`（旧缓存自动重编）。剩余差异来自 Qwen3.5 实现不同（HF transformers vs Comfy 自己的实现）。
- **LoRA**：作者环境（`--highvram`、torch < 2.8）走经典 ModelPatcher 路径而非 dynamic VRAM：fp16 反量化 → fp32 算 ΔW → fp16 相加 → fp16 旋转、逐行 amax、随机舍入；种子 = `crc32("diffusion_model.{Comfy 权重键}.weight")`，融合 qkv 按整块 merge。dynamic VRAM 路径（bf16 merge、种子取模块名）在作者环境不启用，未实现。
- **DiT 推理前向**：上游前向与 Comfy 数学相同、舍入路径不同（调制在 fp32 vs bf16、RoPE 先转 bf16 vs fp32 里旋转、RMSNorm 写法、残差 addcmul）。出图另走 `comfy_forward.py` 逐算子复刻；训练与训练中采样仍走上游前向。
- **采样**：er_sde 的 half-log-SNR / alpha 改用 Comfy 原式（Anima 共用，出图更贴近 Comfy）；CFG 在去噪结果上组合；`percent_to_sigma` 用 double——SLG 窗口闭区间，fp32 边界会在 end_percent 那一步多跳一层。

**验收**（同 torch 2.7 环境、注意力统一 SDPA）：DiT 42 层前向、LoRA merge 后的 int8 权重与 scale、8 步 er_sde + CFG + SLG 的端到端 latent，全部与 ComfyUI 逐位一致（max diff 0）。W8A8 线性层在 torch 2.11 下喂 Comfy 的真实输入同样逐位一致。

**剩余差异**（作者实际环境，不在 PR-6 范围）：
- torch 2.11 的 `F.rms_norm` 是融合内核、2.7 是组合实现，`mean` 归约顺序不同，1200 万元素约 70 个差 1 个 bf16 最小单位；单次 DiT 前向对这种扰动的放大约 7.7%（只改本项目自己的 RMSNorm 写法也是 7.7%），所以跨 torch 版本逐位不可能。
- 个别形状的 fp32 批量 GEMM 跨版本选不同 cuBLAS 算法（merge 时 `o` 层 0.01% 元素差 1）。
- Comfy 的 INT8 注意力内核；VAE 解码两边实现不同（最多差 4 级像素）。
- 整图（作者工作流 832×1216 / 30 步 / CFG 10 / SLG 22 @ 0–0.5）vs Comfy：er_sde 16.30 → 16.60 dB，euler 18.57 → 20.94 dB。

## 6. 能力矩阵与显存

| 项 | v1 |
|---|---|
| masked_loss | 开（f8 分辨率，布局 B 下零改动） |
| text_cache | 开（必需） |
| block_swap | 开（42 层） |
| navit / sra / leap / compile_blocks / caption_tag_ops | 关 |

权重常驻量（不含激活，仅供参考，不作承诺）：DiT bf16 10.2GB / int8 5.4GB；TE（只在缓存阶段存活）bf16 16.8GB / int8 9.75GB（含视觉塔，跳过后更小）。

**PR-5 实测（只记录，不承诺）**：RTX 5090，int8 DiT + int8 TE，读数为 torch 峰值已分配显存（与显卡型号无关；WDDM 驱动开销另计）。每层约 1.13 亿参数，换出时锁页内存约 0.11GB/层（int8）/ 0.21GB/层（bf16）。

| 场景 | swap 0 | swap 21 | swap 42 |
|---|---|---|---|
| 出图 832×1216 采样+解码（TE 已预编码释放） | 7.27 GiB | 5.27 GiB | 3.05 GiB |
| 出图 DiT 常驻 | 5.05 GiB | 3.05 GiB | 0.83 GiB |
| 出图 8 步耗时 | 7.3 s | 7.6 s | 7.9 s |
| 训练 512px batch 1 训练步（grad checkpoint） | 7.05 GB | 4.90 GB | — |
| 训练中采样 | 6.63 GB | 4.48 GB | — |

TE 阶段与 swap 无关：出图预编码 7.51 GiB，训练文本缓存 8.23 GB——开 swap 后它成为整条流程的峰值。不同 swap 下出图逐像素一致；daemon 开 swap 21 与不开 swap 的出图逐像素一致。

## 7. 资产清单（`studio/services/models/families/moss.py`）

| 条目 | 来源（HF 主，ModelScope 镜像） | 用途 |
|---|---|---|
| `base_bf16` / `sft_bf16` | `Comfy-Org/MOSS-Image` `diffusion_models/moss_image_{base,sft}_bf16.safetensors` | 训练 / 推理 |
| `base_int8` / `sft_int8` | 同 repo `moss_image_{base,sft}_int8_convrot.safetensors` | 训练 / 推理 |
| `te_bf16` | `OpenMOSS-Team/MOSS-Image-Base` 的 `text_encoder/` + `tokenizer/` | 缓存 |
| `te_int8` | `Comfy-Org/MOSS-Image` `text_encoders/qwen3.5_9b_int8_convrot.safetensors` + tokenizer 小文件 | 缓存 |
| `vae` | `Comfy-Org/MOSS-Image` `vae/flux2-vae.safetensors` | 全流程 |

磁盘落点按 04-synthesis §2-9（按资产类型组织；TE 目录落 `text_encoders/<safe_dir_name(repo)>/`）。用户已有的 ComfyUI 文件通过自定义路径接入，不强制重新下载。

## 8. 不在范围内

Layout JSON 编辑器、prompt enhancer、参考图 / 编辑能力、少步蒸馏版（社区已有 8 步 DMD 版）、W8A8 int8 出图提速、TE 微调（官方明确禁止）、`dataset.py` 的 `_ANIMA_LATENT` 硬编码清理（§5.1 技术债）。

## 9. 切刀计划

每刀一个 PR，从 `dev` 切出、squash 回 `dev`。

| 刀 | 内容 | 验收 |
|---|---|---|
| PR-1 | `FLUX2_F8C32_BN` latent 空间 + FLUX.2 VAE（BFL 结构移植 + 布局 B 适配层 + `VAEWrapper` 通道 / 对齐泛化） | 适配器与 BFL encode/decode 互为 reshape；patchify 通道序与 Comfy 一致；分块对齐；strict 加载；真实权重与 ComfyUI fp32 对比（§5.2 表） |
| PR-2 | `modeling/moss/` + loader（bf16 / int8 单文件、qkv 拆分、`rope.bands` 剥离）+ int8_convrot 运行时 + `forward_train` | 小配置随机权重与官方 `MOSSImageTransformer` 做 forward 数值对齐；int8 反量化与 bf16 的误差测试（合成数据）；融合 / 非融合键映射测试；梯度能回传到 LoRA |
| PR-3 | TE（bf16 目录 / int8 单文件）+ 文本缓存 | 模板与 tokenize 结果对齐官方 / Comfy；int8 TE 与 bf16 TE 的 embedding 相似度；GDN 慢路径的缓存耗时实测 |
| PR-4 | 族注册、schema（`model_family` Literal、能力矩阵、族默认值、采样白名单）、资产清单、采样（Comfy parity + uncond 跳层）、LyCORIS 量化底模规则、UI | `test_model_family_gating` 三方对齐；§5.6 parity 用例出图对比；内置浏览器做视觉验证 |
| PR-5 | 显存编排（让位 / 三档）+ block swap 接入 | 12 / 16GB 卡实测峰值（只记录，不承诺） |
| PR-6 | int8 出图 Comfy parity：LoRA merge → requant（随机舍入）→ W8A8 前向（§5.9） | 以 comfy-kitchen eager 为 oracle 的逐步数值测试；GPU 上与作者本地 ComfyUI 同 seed 出图逐位 / 近似对比 |

依赖关系：PR-2 依赖 PR-1（`forward_train` 要用布局 B 的 patchify 约定）；PR-3 与 PR-2 相互独立，可以并行；PR-4 依赖 PR-1～3；PR-5、PR-6 依赖 PR-4，两者之间相互独立。PR-4 的出图先走 W8A16 旁路，功能上可用；PR-6 再把 int8 出图切到 parity 路径。旁路本来就是训练中采样要用的路径，PR-6 合入后也不删，所以不属于「后一刀会删掉的临时功能」。

## 10. 遗留 / 后续

- `dataset.py` 改为全部经注入的 `latent_spec` 计算（去掉 `_ANIMA_LATENT` 直连）。
- 批量化 forward（pad + mask，替代逐样本循环）。
- bf16 底模出图的 parity：如果用户在 ComfyUI 里加载 bf16 文件，Comfy 的 LoRA merge 是浮点直加，我们的旁路与之只差加法顺序，暂不单做。
- Krea2 的 TE 缓存指纹也区分存储形态（与 M8 对齐）。

## 11. 待验证事项

1. ~~HF 小 decoder VAE 与 Comfy `flux2-vae` 的 encoder 是否相同~~：v1 只支持 BFL / Comfy 格式（§5.2），不再需要。将来要支持 diffusers 格式时再比对，决定能否共享 latent 指纹。
2. ~~VAE 编码取 mode 还是 sample~~：已确认取 mean（BFL 官方实现，§5.2）。
3. int8 TE 输出的 embedding 与 bf16 TE 的差异量级（PR-3 测量，写进测试阈值）。
4. Qwen3.5 GDN 在没有 fla 时的缓存速度（PR-3）。
5. `SkipLayerGuidanceDiTSimple` 的区间语义（start / end 百分比按 σ 还是按步数），PR-4 对照源码确认。
6. comfy-kitchen 的 CUDA int8 kernel 与 eager 路径数值是否一致（决定 PR-6 用哪个做 oracle；Krea2 fp8 用的是 eager）。
7. 作者本地 ComfyUI 实际走的是 dynamic VRAM 分支还是传统 ModelPatcher 分支（两者都会 requant，但 seed 的取法和时机需要逐一核对）。

## 参考

- 技术报告：<https://openmoss.ai/MOSS-Image/paper/MOSS_Image_Technical_Report.pdf>
- 官方仓库：<https://github.com/OpenMOSS/MOSS-Image>（`USERGUIDE_ZH.md`、`comfyui/`）
- HF：<https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base>、<https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT>
- Comfy-Org 重打包：<https://huggingface.co/Comfy-Org/MOSS-Image>
- ComfyUI 核心支持：comfyanonymous/ComfyUI#17095、comfyanonymous/ComfyUI#17099（`comfy/ldm/moss_image/model.py`、`comfy/lora.py` MOSS 分支、`comfy/supported_models.py:MOSSImage`、`comfy/latent_formats.py:Flux2`）
- comfy-kitchen 0.2.38（Apache-2.0）：`tensor/int8_utils.py`、`backends/eager/quantization.py`
- ConvRot 论文：<https://arxiv.org/abs/2512.03673>
