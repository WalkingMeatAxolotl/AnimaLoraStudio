# NaViT / Patch-n-Pack 块对角打包训练

> 状态：**模型内核 + 数据层 + 训练循环接线已落地**（本地单测通过）。
> opt-in / default-off。关掉时与改动前逐字节等价。

## 1. 解决什么问题

小数据集 + 多分辨率 + 想开高 batch 时，现有 ARB 分桶按**精确 `(h, w)`** 分桶，分辨率一多
→ 每桶几张图 → 填不满高 batch。

**NaViT/Patch-n-Pack（arXiv 2307.06304）** 用真正的 *example packing*：把**多张异构图**拼进
一条序列，用**块对角注意力掩码**让每图只注意自己的 token（self）与自己的 caption（cross），
每图带**自己的 timestep**。于是"每步处理多少图"与"单图形状"彻底解耦——小数据集也能用
固定 token 预算填满任意有效 batch，且**零 padding、走 xformers varlen 快内核**。

## 2. 怎么开

```yaml
navit_packing: true            # 总开关（默认 false）
navit_token_budget: 16384      # 一个 pack 的 token 数之和上限（按显存定，必须 ≥ 最大单图 token 数）
navit_max_images_per_pack: 0   # 单 pack 最多几张图，0=不限
```

### 提速 / 打包旋钮（均 opt-in，关时逐字节等价）

```yaml
navit_text_trim_padding: false # cross-attn 按每图有效 T5 长度打包文本，去 512-pad
navit_pack_strategy: next_fit  # next_fit（默认，顺序贪心）/ ffd（窗口内 FFD，包更满）
navit_pack_ffd_window: 256     # ffd 窗口大小（0=全局 FFD；>0=窗口内 FFD + 跨 epoch reshuffle）
navit_drop_last: false         # 丢弃每 epoch 最后未满预算的包
```

### 缓存分块 encode（`cache_encode_tiled`，opt-in）

```yaml
cache_encode_tiled: true        # 超大图按 tile_px 分块 VAE encode + latent 羽化拼接
cache_encode_tile_px: 1024      # 块边长（16 的整倍数）
cache_encode_tile_overlap: 128  # 相邻块重叠（16 的整倍数）
cache_encode_max_pixels: 0      # 单次 encode 像素预算（含翻转份）；0=内置 4M 默认
```

峰值显存从 ∝ 整图像素降到 ∝ 单块像素。接缝处是**近似**（VAE conv 感受野越过块边界），
overlap 越大误差越小。

### 显存 ↔ token_budget 对照（grad_checkpoint=true 下的保守起点）

| 显存 | 起步 token_budget | 约等于（4096-token/张） |
|---|---|---|
| 16 GB | 16384 | ~4 张 |
| 24 GB | 32768 | ~8 张 |
| 48 GB | 65536 | ~16 张 |
| 80 GB | 98304 | ~24 张 |

**务必首跑观察峰值显存再调**——你的卡、rank、底模大小都会左右它。

## 3. v1 支持范围与门控

NaViT 改变了 batch 语义（一包异构图、逐图 t），许多按"批量网格 + 逐 batch 单 timestep"
假设写的特性会语义错位。v1 的策略：

**支持**：basic flow-matching（逐图 t 采样 + 噪声 + per-image loss）、`grad_accum`、
逐块梯度检查点（`grad_checkpoint`）、LoRA 保存/恢复、`loss_weighting`（min_snr / cosmap /
detail_inv_t，按 per-image t 算权重）、正则集降权（`loss_weight`，逐图应用）。

**互斥（同时开 → 启动即 fail-fast 报错）**：`leap_enabled`、`infonoise_enabled`、
`lora_type=tlora`、`sra_enabled`。这些在 NaViT 路径会被跳过——为避免"开着却悄悄不生效"
的隐性行为改变，一律 fail-fast 要求显式关闭。

**前置要求**：`cache_latents=true`（打包按 latent token 数预算分包，需要预编码缓存）、
`navit_token_budget > 0`。

## 4. 实现地图

| 层 | 文件 |
|---|---|
| 注意力 op 块对角分支 | `modeling/cosmos_predict2_modeling.py` — `torch_attention_op` 接 `BlockDiagonalMask` |
| 逐图 timestep 调制 | 同上 — `Block.forward_tokens` / `FinalLayer.forward_tokens` 的 `token_wise_mod` |
| 打包前向 | 同上 — `MiniTrainDIT.forward_packed_navit`（块对角 self/cross + 逐图 RoPE + 逐图 AdaLN） |
| token 预算打包 | `runtime/training/dataset.py` — `NavitPackBatchSampler` / `collate_fn_navit_pack` |
| 分块 VAE encode | 同上 — `tiled_vae_encode` |
| 训练步核心 | `runtime/training/navit.py` — `navit_packed_forward_and_loss`（逐图加噪 → 打包前向 → 逐图 loss） |
| 训练循环接线 | `runtime/training/loop.py` — navit 分支（检测 `navit_latents` → pack cross → 打包前向） |
| config 键 | `studio/domain/training.py` — `navit_*` / `cache_encode_*` 字段 + 互斥校验 |

### 验证状态

本地 GPU（CUDA + xformers 0.0.30）+ 纯 Python 单测，**已通过**（50 test）：

- `test_packed_block_diag_attention`：块对角 self/cross attention ≡ 各图独立 attention 拼接。
- `test_packed_navit_forward`：`forward_packed_navit` ≡ 各图单独前向拼接；grad checkpoint ≡ 非检查点。
- `test_navit_pack_sampler`：打包预算 / 覆盖 / 超大图 / 张数上限 / 跨 epoch 重洗。
- `test_navit_packed_objective`：训练步前向 + loss + 反向梯度有限；grad checkpoint 等价。
- `test_navit_per_image_adaln`：per-image AdaLN（`mod_index`）≡ legacy per-token 布局。
- `test_navit_per_image_weights`：per-image 权重（正则集 loss_weight × loss_weighting）按权重缩放 loss。
- `test_navit_multiscale` / `test_navit_native_resolution`：定尺寸 / 多尺度副本规划。

**尚未本地验证（云端 smoke 必做）**：训练循环端到端接线、真模型（head_dim 128）首跑、
与 LoRA/LoKr 注入的实跑交互。
