# 多分辨率训练设计

> Issue: [#246 多分辨率训练支持可能性](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/issues/246)
> 状态: 设计已定稿，实现待排期。

## 1. 背景

官方示范 LoRA 采用多个分辨率混合训练（典型如 512 / 768 / 1024）。掺入更高分辨率样本可改善推理时高分辨率的表现（更少糊化 / 重复 / 构图崩坏），本质是多尺度增强。

炉子当前只支持**单一基准分辨率**：`BucketManager(args.resolution)` 围绕一个 `base_reso` 做 ARB 分桶，所有图最终都落在 ≈base² 这一个像素预算上。本设计在保留 ARB 的前提下，加入"一次训练里多个分辨率"。

### 概念澄清

- **ARB（长宽比分桶）**：不同长宽比、**相同像素面积**的图分到不同桶。炉子已有。
- **多分辨率 / 多尺度**：真正用**不同基准分辨率**（不同像素预算）混训。本设计要加的。

业界两条路线，本设计**两者都做**且可组合：
- **全局分辨率列表**（diffusion-pipe / ai-toolkit 风格）：每张图在每个分辨率各训一遍。
- **按组覆盖**（kohya TOML / OneTrainer concept 风格）：每组各自一个分辨率。本设计用 kohya 文件夹方言编码。

## 2. 目标 / 非目标

**目标**
- 单次训练支持多个分辨率。
- 两个独立、可组合的指定方式：全局列表 + 按文件夹覆盖。
- 桶长宽比上限可配（默认 2.0），支持大长宽比作品不被强行裁剪。
- 右栏可视化实际桶分布（每桶分辨率 + 命中图数），辅助 batch size 决策。
- 默认行为不变（不填列表、文件夹无前缀、配比留 2.0 → 跟今天完全一样）。

**非目标**
- crop 页改造（crop 只定构图/AR，与绝对分辨率无关，见 §9）。
- 自动决定哪些图该用哪个分辨率（始终由用户显式指定）。

## 3. 两个机制

### 机制 A — 全局分辨率列表

`resolution` 字段从标量扩展为"标量或列表"：

```yaml
resolution: 1024            # 标量：行为同今天
resolution: [512, 768, 1024]  # 列表：多分辨率
```

语义：**对没有分辨率前缀的文件夹**，列表里每个分辨率各生成一套 ARB 桶，每张图在每套桶里**各训练一遍**。即 `[512,768,1024]` 下，一张无前缀文件夹里的图 → 3 个样本（512 / 768 / 1024 各一份）。

### 机制 B — 按文件夹覆盖（`px` 前缀）

沿用现有 Kohya 风格文件夹名约定（`5_concept` = repeat 5），扩展出分辨率前缀：

```
1024px_2_data      → 分辨率 1024, repeat 2, label "data"
768px_5_concept    → 分辨率 768,  repeat 5, label "concept"
1024px_data        → 分辨率 1024, repeat 1, label "data"
5_concept          → 分辨率=config, repeat 5         （向后兼容 Kohya）
concept            → 分辨率=config, repeat 1
```

- **带 `px` 前缀的文件夹 = 单一分辨率，覆盖全局列表**。即使 `resolution: [512,768,1024]`，`1024px_xxx` 里的图也只在 1024 训一遍（不参与列表 fan-out）。`px` 前缀是显式的"我就要这个分辨率"。
- 默认创建的文件夹**不带**前缀，需用户手动重命名添加。

### 为什么用 `px` 单位而非裸数字

现有文法 `^(\d+)_(.*)$` 把第一个数字当 repeat。裸 `1024_2_data` 会与之相撞（被解析成 repeat=1024），且 `1024_data`（单数字想表达"分辨率 1024、repeat 1"）会变成 repeat=1024 的陷阱。`px` 单位让分辨率 token 永不与 repeat 混淆，且自带文档性、新手一眼懂。

## 4. 文件夹名文法

按 `_` 切 token，从左往右：
- token 匹配 `^\d+px$` → **分辨率**（去掉 `px` 取数字）。
- token 匹配 `^\d+$` → **repeat**。
- 其余 → label（剩余部分整体作为标签，可含 `_`）。

规则：
- 分辨率与 repeat 都可选；分辨率在前、repeat 在后（`1024px_2_data`）。
- 单数字（无 `px`）= repeat，保持与 Kohya 完全兼容。
- 解析必须 Python（`runtime/training/dataset.py`）与前端展示（`studio/web/src/pages/project/steps/Train.tsx:parseFolderRepeat`）两处同步。

## 5. 样本展开语义

每张**唯一图**展开成的样本数 = `repeat × 该文件夹的分辨率数`：

| 文件夹 | config `resolution` | 每张图样本数 |
|---|---|---|
| `2_data` | `1024`（标量） | 2 × 1 = 2 |
| `2_data` | `[512,768,1024]` | 2 × 3 = 6 |
| `1024px_2_data` | `[512,768,1024]` | 2 × 1 = 2（px 覆盖，不 fan-out） |
| `data` | `[512,768,1024]` | 1 × 3 = 3 |

展开在 `ImageDataset._scan` / 样本构建阶段完成（与现有 repeat 展开同层）。每个展开样本携带一个 `target_reso` 字段，决定它走哪套 ARB 桶。

`DatasetStatsPanel`（Train.tsx）的"有效样本数 / steps 估算"公式要相应乘上分辨率数。

## 6. ARB / BucketManager 改造

### 6.1 多 base 支持

`ImageDataset` 持有一个按分辨率缓存的 `dict[int, BucketManager]`，每个样本按其 `target_reso` 取对应 manager 做 `get_bucket(w, h)`。

### 6.2 桶长宽比上限可配（宽 AR 支持）

当前 AR 上限是写死的 `2.0`（`dataset.py:57`）。`get_bucket` 不丢图，AR 超过 2:1 的图会被 snap 到 2.0 桶并**居中裁掉超出的边**——大长宽比作品因此被强行裁剪、丢失左右/上下内容。

改为暴露**一个对称配比 R**（schema 字段 `aspect_ratio_limit`，默认 `2.0`）：
- R 同时管两头：最宽 `R:1`、最高 `1:R`（等价 `min_ar = 1/R`、`max_ar = R`，一个数字即可，不拆 min/max_ar）。
- `_generate` 里的 `2.0` 字面量换成 R；默认 2.0 → 桶集与今天完全一致。
- **作用域（v1 决策）**：全局一个值，默认 2.0。per-folder 覆盖（同 px 机制）留作后续——全局放宽到 3.0 会让所有文件夹的桶都摊到 3:1；其碎片化代价由 §10 的桶分布预览显性化后，再决定是否细化到 per-folder。

> 放宽 R 的代价（写给用户的提醒）：面积恒定下 AR 越大短边有效分辨率越低；极端 AR 桶样本少 → 单独成小/短 batch、梯度噪声大 → 需足够样本量。§10 预览可直接看每桶命中数来权衡。

### 6.3 min/max 由 (base, R) 派生（取代写死值）

当前 `BucketManager(base, min_reso=512, max_reso=2048, step=64)` 的 min/max 写死。这对 base=1024 没问题，但**对小 base 会退化**：base=512 时最小边卡在 512 → 唯一满足 ±10% 面积的就是 512×512，**只剩一个方桶，无任何长宽比变化**。

修复：min/max **不再暴露**，由 `(base, R)` 内部派生——

- 恒定面积 base² 下最极端的桶是 `base·√R × base/√R`，故 `min_reso ≈ base/√R`、`max_reso ≈ base·√R`（向外取整到 step 并留一格余量，防量化误裁；面积 ±10% + AR≤R 做真正切割）。
- R=2、base=1024 → 派生边界仍包住那 37 个桶，**默认行为不变**（`trainBuckets.test.ts` 的 `count==37` 不破）。
- 与 diffusion-pipe 模型一致：用户只调 AR 范围（R），reso 边界是派生量、不暴露（详见对照核对结论）。

`step`（64）、面积容差（0.1）保持不变。

## 7. latent cache 改造

`CachedLatentDataset` 当前每张图一个 npz：`img.with_suffix(".npz")`。多分辨率 fan-out 下，同一张图需要同时存在多份不同分辨率的 latent（spatial shape 不同），单一路径会互相覆盖。

方案：
- 图只有**单一目标分辨率**（标量 config，或单一 px 文件夹）→ 维持 `img.npz`（不动现有缓存）。
- 图 fan-out 到**多个分辨率** → 每分辨率独立 npz：`img.r{reso}.npz`（如 `img.r512.npz` / `img.r768.npz` / `img.r1024.npz`）。
- `_is_cache_valid` 已校验 npz 内 `bucket_w/bucket_h` 与期望桶一致，不匹配即失效重 encode；多 npz 路径方案叠加这层即可。

> 一次性代价：仅对启用多分辨率列表的图首建缓存时多 encode 几份；单分辨率用户零影响。

## 8. batch sampler（已天然支持）

`BucketBatchSampler` / `CachedLatentDataset._fill_bucket_for_index` 已经**按 latent 的实际 spatial shape `(h, w)` 分桶**。不同分辨率天然产生不同 shape → 自动落入不同桶 → 同 batch 形状一致。**这一层无需改动。**

## 9. 放大页 prep UX（`Preprocess.tsx`）

放大页是与分辨率真正相关的页面（保证 px 文件夹里的源图有足够真实细节）。现状：

- **目标分辨率**：已经是 dropdown（`TARGET_PRESETS` 的 `<select>` + 自定义边长，`Preprocess.tsx:564`）。
- **按分辨率范围 filter**：pixel-bin chips（`FilterMode`，按图当前像素面积档过滤 grid）。
- 无文件夹维度。

改动（小）：
1. **新增文件夹 filter** —— 把 grid 与"全部放大"范围限定到单个文件夹（唯一 load-bearing 的新增）。图片路径已带文件夹段，数据现成。
2. **选中带 px 的文件夹 → 目标分辨率自动跟随**（解析前缀）。列表 config 下的无前缀文件夹，目标分辨率取列表 `max`（确保最大那档也有真实细节）。
3. pixel-bin chips 可选收成 dropdown 腾地方（纯 cosmetic）。

### crop 页：不动

crop 定的是构图 / 长宽比；裁出的区域最终是多少像素由文件夹分辨率 + trainer 决定。`cropClustering` 里写死的 1024 只是 AR 参考网格，跨分辨率的桶 AR 仅因 step=64 量化差极小，可忽略。

## 10. 右栏桶分布预览（`DatasetStatsPanel`）

现状：Train 页右栏只算"有效样本数 / steps 估算"。新增"实际桶分布"，把多分辨率 + 可配 R 的形态显性化（每个分辨率档命中了哪些桶、各多少图），辅助判断数据在各尺度/长宽比上的分布。

按"分辨率档 → 桶 → 有效图数"分组展示，0 命中桶不显示：

```
1024 档
  1024×1024   128
  1152×896     34
  1408×704      2
768 档
  768×768      96
  ...
```

设计要点：
1. **后端权威计算**：直接用真正的 `BucketManager`（Python）+ 每张图尺寸算直方图，保证与实际训练逐桶一致，绕开 TS 预测的同步脆弱性。后端已有每张图尺寸（放大页像素直方图同源）。新增 `GET /versions/{vid}/bucket-distribution` 返回 `{reso: [{w, h, count}]}`。
2. **按分辨率档分组**：多分辨率 fan-out 下同一张图进多档，分组展示让 fan-out 形态可见。
3. **计数用"有效数"**（含 repeat / fan-out 展开后），即一个 epoch 该桶实际样本数。
4. **不做 drop_last 丢图警告**：trainer 缓存路径写死 `drop_last=False`（`phases/dataset.py`），桶不满只出**短 batch**、不丢图（diffusion 的 Norm 对动态 batch 不敏感，loop.py 按 `latents.shape[0]` 读 bs）。故预览只展示分布、不标红。
5. 隐藏 0 命中桶（37 个桶通常只命中一小撮）。
6. **只统计训练集**：reg 不裁剪、无需在预览展示（但 reg 仍参与 fan-out 与实际批处理，见 §14）。

## 11. schema / 迁移

- `studio/domain/training.py:65` `resolution: int` → `resolution: int | list[int]`（或 `Union[int, conlist(int)]`），保留 `ge=256, le=4096` 对元素生效。
- 加 pydantic before-validator：标量与列表都归一到内部 `list[int]`；建议各值 snap 到 64 的倍数并 clamp 到 `[256, 4096]`。
- 新增 `aspect_ratio_limit: float = Field(2.0, ge=1.0)`（桶长宽比上限 R，见 §6.2）；i18n description 写"桶最大长宽比，2.0=最宽 2:1；调大支持长图但会增加碎片化、降低短边分辨率"。
- `bootstrap.apply_yaml_config` 侧做读旧（标量 resolution、无 aspect_ratio_limit）兼容。
- SchemaForm（`studio/web/src/components/SchemaForm.tsx`）需支持列表输入（逗号分隔或多值），i18n description 写"单值=单一分辨率；多值=每张图在各分辨率各训一遍"。

## 12. 前后端同步不变量

`runtime/training/dataset.py:BucketManager` ↔ `studio/web/src/lib/trainBuckets.ts` 必须保持算法/默认一致（见两文件顶部 docstring）。

- §6.2 的 AR 上限 R、§6.3 的 min/max 派生逻辑若在 Python 实现，TS 镜像也应同步（即便 crop 页只用默认参数、不触发非默认 base/R）。
- 默认参数（base=1024、R=2.0）下两边输出仍 byte-for-byte 一致，`trainBuckets.test.ts` 断言不变。
- §10 的桶分布预览改为后端权威计算后，此预览路径**不依赖** TS 镜像，进一步降低同步风险（TS 镜像仅 crop 页仍在用）。

## 13. 要改的 surface 清单

| 层 | 文件 | 改动 |
|---|---|---|
| 训练-解析 | `runtime/training/dataset.py` | 文件夹名解析 `px` 前缀 + repeat；样本带 `target_reso`；`dict[reso]→BucketManager`；AR 上限改读 R；min/max 由 (base,R) 派生 |
| 训练-装配 | `runtime/training/phases/dataset.py` | 按 `target_reso` 取 BucketManager；传入 R；reg 集随列表 fan-out（见 §14） |
| 训练-cache | `runtime/training/dataset.py:CachedLatentDataset` | 多分辨率图按 `img.r{reso}.npz` 分别缓存 |
| 后端-stats | version stats builder（`train_folders` 同源处） | 用真 `BucketManager` 算桶直方图 `{reso:[{w,h,count}]}`（§10） |
| schema | `studio/domain/training.py` | `resolution` 标量→标量/列表 + validator + 迁移；新增 `aspect_ratio_limit` |
| 前端-展示 | `studio/web/src/pages/project/steps/Train.tsx` | `parseFolderRepeat` 扩展显示 px；有效样本数 ×分辨率数；渲染桶分布预览（分辨率分组 / 有效数 / 隐藏 0） |
| 前端-放大 | `studio/web/src/pages/project/steps/Preprocess.tsx` | 文件夹 filter + 目标分辨率跟随文件夹 |
| 前端-schema | `studio/web/src/components/SchemaForm.tsx` | 列表输入控件 + `aspect_ratio_limit` + i18n |
| 镜像 | `studio/web/src/lib/trainBuckets.ts` | min/max + AR 上限(R) 派生同步（如改 Python） |
| crop | — | 不动 |

## 14. 决策记录（全部已决）

- 配比 R：暴露**一个对称值**（`aspect_ratio_limit`，默认 2.0），**不**暴露 min/max_ar、**不**暴露 min/max reso（由 base+R 派生）。
- 配比作用域：**v1 全局**，per-folder 留作后续（§6.2）。
- 桶分布预览：后端权威计算、按分辨率分组、用有效数、隐藏 0（§10）。trainer `drop_last=False`（桶不满出短 batch、不丢图），故不做丢图警告。
- **reg 正则集参与 fan-out**：与训练集一致，每张 reg 图在列表各分辨率各训一遍；桶分布预览只统计训练集、不含 reg（reg 不裁剪、无需展示，但仍参与实际批处理）。
- **bucket 步长固定 64、不暴露**：受架构约束（VAE ÷8 × DiT patchify ÷2 = latent ÷16 下限，64 是安全超集）；调低会崩，暴露只挖坑。
- **放大目标（列表 config + 无前缀文件夹）= 列表 `max`**：图会在所有分辨率训，放大到最大那档才保证最大尺度有真实细节，小档从它缩小。路由不变：无前缀→配置分辨率、有前缀→前缀。
- **repeat × 分辨率 的相乘规则**：
  - **无 px 前缀**文件夹 → `repeat × 分辨率列表` 相乘（`2_data` + `[512,768,1024]` = 每图 6 样本）。
  - **有 px 前缀**文件夹 → 单一分辨率、**不做分辨率 fan-out**；Kohya `N_` repeat 照常（`1024px_2_data` = 2 样本）。文件夹分辨率（局部）**覆盖**配置页（全局）。
- **用户填的分辨率值自动 snap /64**：validator 自动纠（1000→1024）+ clamp 到 `[256,4096]`，避免偏心桶。
- **npz 命名混合方案**：单分辨率保 `img.npz`、多分辨率用 `img.r{reso}.npz`；首次启用多分辨率时那些图一次性重 encode、单分辨率用户零影响。

> 设计闭环：所有 open question 已决，可直接照 §13 surface 清单实现。
