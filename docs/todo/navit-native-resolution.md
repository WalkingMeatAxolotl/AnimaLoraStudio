# NaViT 原生分辨率 / 多尺度：状态

> 状态：**原生分辨率（`navit_native_resolution`）与多尺度阶梯（`navit_multiscale`）均已实现**
> （opt-in / default-off，见 `docs/navit-packing.md`）。本文保留为设计/历史记录。

## 已落地：原生分辨率

开 `navit_native_resolution=true` 后单图按**原生尺寸** floor 对齐 16px 定尺寸、零 padding，绕过 ARB 桶量化。
超大图 `navit_native_over_budget`（`downscale` 默认 / `fail`）处理，消除 token 爆炸 / RoPE capacity / 显存顾虑：

- **token 爆炸 / 超预算** → `downscale` 等比缩到 ≤ `navit_token_budget`（`fail` 报错）。
- **RoPE capacity** → 数据层封顶单边 ≤ `max_img_h/w // patch_spatial`（默认 512 token）+ `_packed_rope_from_grid` 前向 fail-fast。
- **显存** → 复用 `cache_encode_tiled` 分块 encode。

接线：`ImageDataset._target_size_for` / `plan_native_fit_image`（`runtime/training/dataset.py`）+
`studio/domain/training.py` config + `runtime/training/phases/dataset.py` 传参；token 数从缓存 latent shape 推。

## 已落地：多尺度阶梯

`navit_multiscale=true`（需 native）为原生大图追加低 token 档等比缩小副本参与打包：填满大图包剩余预算 +
缓解 train-large / infer-small 尺度偏移。确定性展开（每图每档每 epoch 恰一次），只降不升，副本走独立
`<stem>.ms<T>.npz` sidecar。`navit_multiscale_loss_weight`（默认 1.0=每图等权，贴合本仓库设计；<1.0 副本降权）。

接线：`ImageDataset._expand_multiscale_samples` / `plan_multiscale_copy` + `CachedLatentDataset` 缓存穿线
`ms_tokens_target` + `collate_fn_navit_pack` 折 `ms_weight` 进 per-image 权重（与正则集 loss_weight 正交相乘）。

## 可能的将来（未实现）

- **随机分辨率采样**（NaViT 论文 arXiv 2307.06304 的 resolution-sampling）：多尺度阶梯的随机版
  （每 step 对每图随机抽一个尺度）。当前实现是确定性阶梯（可复现、可归因），随机版是否更优待评估。
- fractional PE：RoPE3D 整数网格外推不需要（不做）。

## 关联

- 设计文档：`docs/navit-packing.md`
- 定尺寸 / 副本：`runtime/training/dataset.py::plan_native_fit_image` / `plan_multiscale_copy` / `ImageDataset._expand_multiscale_samples`
- 前向天然支持异构 token：`runtime/training/navit.py::navit_packed_forward_and_loss`
- 决策讨论：PR #371 review（骨架删除）→ 原生分辨率接线 PR → 多尺度阶梯 PR
