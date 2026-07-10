# TODO：NaViT 多尺度阶梯（multiscale，暂缓）

> 状态：**原生分辨率（`navit_native_resolution`）已实现**（opt-in / default-off，见
> `docs/navit-packing.md`「原生分辨率」小节）；**多尺度阶梯（multiscale）仍有意暂缓**。

## 已落地：原生分辨率

navit 打包（`navit_packing=true`）默认仍只做 **batching 优化**（latent 来自 ARB 分桶）；
开 `navit_native_resolution=true` 后单图改按**原生尺寸** floor 对齐 16px 定尺寸、零 padding，
绕过桶量化。超大图按 `navit_native_over_budget`（`downscale` 默认 / `fail`）处理，正面消除
了此前顾虑的 token 爆炸 / RoPE capacity / 显存：

- **token 爆炸 / 超预算**：`downscale` 等比缩到 ≤ `navit_token_budget`（`fail` 报错）——不再 OOM。
- **RoPE capacity**：单边 token 数在数据层封顶到 `max_img_h/w // patch_spatial`（默认 512 token
  ≈ 8192px/边）；`_packed_rope_from_grid` 前向也有 fail-fast 兜底。
- **显存**：超大原生图缓存 encode 复用已有 `cache_encode_tiled` 分块封顶峰值。

接线：`ImageDataset._target_size_for` / `plan_native_fit_image`（`runtime/training/dataset.py`）+
`studio/domain/training.py` 两个 config 字段 + `runtime/training/phases/dataset.py` 传参；
token 数从缓存 latent shape 推，原生尺寸天然跟随（打包器 / 前向 varlen 无需改）。

## 仍暂缓：多尺度阶梯（multiscale）

大图追加低 token 档的**等比缩放副本**拼入打包序列（填满大图包剩余预算 + 缓解 train-large /
infer-small 尺度偏移）+ per-copy loss 降权。建立在原生分辨率之上。

**为什么暂缓**（边际价值，建议 native 在真训练稳定后再评估）：

- native 已解开"单图尺寸被桶量化"这一主要诉求；multiscale 是叠加的"同图多尺度"增强。
- 复杂度：`.ms<T>.npz` sidecar 缓存链路 + `ms_flags` per-image 权重 + multiscale→native 校验 +
  确定性展开（每图每档每 epoch 恰一次），改动面比 native 本身还大。
- 收益偏特定场景（超大图相对预算很大、想同时喂多尺度统计），非典型 anime LoRA 刚需。

若要做：对原生 token 数超档位的图，缓存阶段额外编码等比缩小副本（floor 16px、resize-cover +
center-crop → 零 padding），作为正式数据集条目参与打包；`navit_pack_strategy: ffd` 会自然装出
"1 大图 + N 小副本"的高填充包。副本 caption 与原生共享，只降不升采样。

## 关联

- 设计文档：`docs/navit-packing.md`
- 定尺寸实现：`runtime/training/dataset.py::plan_native_fit_image` / `ImageDataset._target_size_for`
- 前向天然支持异构 token：`runtime/training/navit.py::navit_packed_forward_and_loss`
- 决策讨论：PR #371 review（native 删除）+ 原生分辨率接线 PR
