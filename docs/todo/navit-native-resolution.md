# TODO：NaViT 原生分辨率 / 多尺度（暂缓）

> 状态：**未实现，有意暂缓**（非遗忘）。PR #371 删除了未接线的骨架。

## 当前状态

navit 打包（`navit_packing=true`）目前只做 **batching 优化**：把多张图的 token 按预算
打包进一条序列（块对角注意力隔离、每图独立 timestep、varlen 快内核），替代 ARB
的"分桶凑同尺寸 batch"。**打包的 latent 仍来自 ARB 分桶**（`ImageDataset` 按
`resolution` resize/crop 到桶），所以 navit 目前是"半个 NaViT"——有打包填 batch，
无原生分辨率。

PR #371 删除的骨架：`navit_native_resolution` / `navit_multiscale` /
`navit_multiscale_token_ladder` / `navit_multiscale_loss_weight` 四个 config 字段 +
`plan_native_fit_image` / `plan_multiscale_copy` + `.ms<T>.npz` sidecar + `ms_flags`
链路 + 校验器 multiscale→native 约束。这些在 base 版**也从未接线**（`ImageDataset`
恒走 ARB，plan 函数只被测试调，`ms_tokens_target` 从不写入）。

## 为什么暂缓

从典型 anima LoRA 角度，native_resolution 的增量价值有限、复杂度高：

- navit 对 LoRA 值钱的是"打包填 batch"（大 batch + 多分辨率 + 小数据集时），已工作、不依赖 native。
- ARB 已处理宽高比；LoRA 数据集通常已 crop/preprocess；底模在固定分辨率范围工作最好。
- native 引入超大图 token 爆炸：单张 4000×3000 原生 ≈ 47k token，可能超 `token_budget`
  / 超 `_packed_rope_from_grid` 的 RoPE capacity（报错）/ 显存尖峰。原生细节保留的收益
  抵不过这些复杂度——除非有极端宽高比数据集或超高清细节 LoRA 的特定需求。

## 将来接线要点（若要做）

1. **数据层**：`ImageDataset` 在 navit 模式下按源图**原生尺寸** floor 对齐 16px（VAE 8 ×
   patch 2）缓存 latent，不走 ARB resize/crop。原 `plan_native_fit_image` 逻辑（floor/pad
   对齐 + `max_tokens` 封顶 + over-budget 处理）可参考 PR #371 之前的实现。
2. **打包器**：`NavitPackBatchSampler` 的 token 数按原生尺寸算（当前 `token_count_for_index`
   从缓存 latent shape 推，改原生后自然跟随）。
3. **前向**：**不需要** token-bucket 路径——`forward_packed_navit`（varlen）天然支持异构
   token 数（`latents_list` 每个 `[1,C,T,h_i,w_i]` 逐图 patchify），原生尺寸 latent 直接喂。
   （这也是 PR #371 删 `forward_packed_tokens` 系列的依据：varlen 覆盖了 token-bucket。）
4. **超大图策略**：`max_tokens` 封顶后，超预算的图 fail / skip / 降采样三选一；降采样即
   multiscale（大图追加低 token 档副本），依赖本功能先落地。
5. **RoPE capacity**：原生大图的 token 网格不能超 `max_img_h/max_img_w`（模型构造参数），
   否则 `_packed_rope_from_grid` 报错——接线时要么放大 capacity 要么在 plan 阶段封顶。
6. **多尺度**（multiscale）：建立在 native 之上，大图追加降采样副本拼入打包序列 +
   per-copy loss 降权。边际价值，建议 native 稳定后再评估。

## 关联

- 设计文档：`docs/navit-packing.md`
- 前向天然支持异构 token：`runtime/training/navit.py::navit_packed_forward_and_loss`
- 决策讨论：PR #371 review
