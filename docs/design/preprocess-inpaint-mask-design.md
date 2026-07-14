# 预处理 · 涂抹 + 训练 Mask — 功能设计

> 临时设计文档，整理**逻辑模型 + 用户场景 + 数据契约 + 分期计划**。实现细节看后续 PR。
>
> 状态：设计已讨论定稿（2026-07-12），未实现。UI 已有占位：
> `PreprocessToolsBar` 的 `inpaint` tab（disabled）+ `?tool=inpaint` fallback。

## 0. 目的与非目的

**目的**：补齐预处理工具栏第五个工具「涂抹」，覆盖两个使用场景：

- **场景 1（改图）**：图片上的文字 / 水印 / 杂物，用取色笔刷实心涂抹去除，避免 LoRA 学入。
- **场景 2（不学）**：手部等易崩坏部位，用 mask 笔刷标记，训练时该区域 loss 置零（masked loss），LoRA 对该区域不产生梯度，生成时由 base 模型先验接管。

**非目的**：

- **不做高斯模糊笔刷**。diffusion 训练中像素即监督信号，模糊区域=教模型生成模糊，
  是反效果（同 JPEG 痕迹 / 水印被学入机制）。i18n 占位文案「取色 + 高斯」中的
  高斯设想作废。
- 不做 AI inpainting（LaMa 等模型补全）。第一期只做取色实心涂抹；AI 补全留作
  未来可选增强（见 §9）。
- 不引入新的备份目录。`download/` 仍是唯一备份，场景 1 出错走现有 restore。
- 不自动联动 caption（不删不加 tag），对齐「不加静默魔法保护」原则。
- 不强制 stage 时序（涂抹与裁剪 / 放大平级，可任意穿插）。

## 1. 已定决策

| # | 决策 | 理由 |
|---|---|---|
| D1 | 分两期：PR-A 场景 1（纯预处理），PR-B 场景 2（含训练器） | 场景 1 训练管线零改动，可独立交付 |
| D2 | 场景 1 = 破坏性像素编辑，原地覆盖 `train/1_data/X.png` | 符合现有 stage 覆盖模型，restore 机制现成 |
| D3 | 场景 2 = 非破坏性 mask sidecar + 训练器 masked loss | kohya sd-scripts / diffusion-pipe 生态标准做法 |
| D4 | 前端在**原图分辨率** canvas 上编辑，整图上传写回 | 笔画后端重放难以精确复现前端抗锯齿 / 软边渲染；整图上传所见即所得，本地 10–20MB PNG 无压力 |
| D5 | ~~mask 存**平行目录** `train/masks/`~~ → **修订（2026-07-13）**：mask 与图**同目录同 stem**，后缀恒 `.mask`（内容仍是灰度 PNG 字节） | 原方案否决的是 `.mask.png` 双扩展名（`suffix==".png"` 被 IMAGE_EXTS 误收），但平行目录实际是**开放集合雷区**——每个递归扫 train/ 的消费点都要记得豁免，reg_ai 漏豁免就把 mask 当训练图（生成数虚高）。`.mask` 单一非图片后缀让所有扫描点天然不命中，与 .txt/.json caption sidecar 同构、共享删除/导出/跟随基建（OneTrainer `-masklabel.png` 同思路，后缀更稳）。老布局 lazy 迁移（`migrate_legacy_masks`），`TRAIN_RESERVED_DIRS` 豁免保留防未迁移数据 |
| D6 | mask 为单通道灰度 PNG，255=正常学习、0=不学，中间值=部分权重 | `loss * mask` 对 [0,1] 连续值天然成立，软边笔刷自动成为过渡权重 |
| D7 | masked loss 是训练 config 显式开关，不因 masks/ 非空而自动开启 | 显式旋钮是用户责任；UI 检测到 masks 非空时提示但不代开 |
| D8 | restore（从 download 复原）时**删除该图 mask** | restore 语义=回到 download 原点、派生一律作废，对齐「不做 partial undo」 |

## 2. 数据模型

### 目录

```
versions/{label}/train/
  manifest.json
  1_data/X.png + X.txt + X.mask   训练 bytes + caption + mask sidecar（同目录同 stem）
```

- mask 路径 = `train/{N_label}/{stem}.mask`（D5 修订，2026-07-13）：与 .txt/.json
  caption 同构的 sidecar；后缀不在 IMAGE_EXTS，所有图片扫描点天然不命中。
- **只有画过 mask 的图片才有 mask 文件**；无文件 = 全 255（正常训练）。
- mask 位于 `train/` 之内 → fork 版本 `_copytree("train")` 自动跟随，零改动。
- **legacy**：首发版布局 `train/masks/{folder}/{stem}.png`（保留目录）由
  `migrate_legacy_masks` lazy 搬迁（masks.py 各入口 + bundle 导出前触发）；
  `TRAIN_RESERVED_DIRS` / trainer `RESERVED_SUBDIRS` 豁免保留，防未迁移老
  数据被递归扫描当训练图。
- manifest **不新增字段**（mask 有无以文件系统为准，过程信息不落盘）。场景 1
  的涂抹写回走现有 entry touch（`mtime/size/processed=true`）。

### 格式

- PIL `L` mode 灰度 PNG，尺寸 == 对应训练图当前尺寸。
- 训练器载入后 `/255` 得 [0,1] 权重图。

### 尺寸校验（fail-safe 兜底）

训练器加载 mask 时校验 `mask.size == image.size`；不匹配 → 记 warning、该图按
无 mask 处理，**不 crash 训练**。这是 §7 变换跟随漏网情形的最后防线。

## 3. User cases

| # | 场景 | 笔刷 | 输出 |
|---|---|---|---|
| U1 | 图上有字幕 / 水印，取周围颜色涂掉 | 取色笔刷（吸管 + 实心圆刷） | 覆盖 `1_data/X.png` |
| U2 | 手崩，训练时不想学手 | mask 笔刷（半透明红 overlay） | 写 `1_data/X.mask` |
| U3 | 画错了撤销几笔 | — | 前端笔画级 undo/redo（会话内） |
| U4 | 整张图改坏了想重来 | — | 现有 restore（download 复原，mask 同时作废，见 D8） |
| U5 | mask 画了一半想清掉 | — | 「清除 mask」按钮 = 删 mask 文件 |

## 4. PR-A 范围（场景 1，纯预处理）

**前端**（新组件，仓库内无可复用 canvas 积木，`FreeCropEditor` 是 DOM overlay）：

- 涂抹编辑器：原图分辨率离屏 canvas + CSS 缩放显示；pointer events 画圆刷；
  笔刷大小滑条；吸管取色（canvas `getImageData`）；zoom / pan（涂小文字必需）；
  笔画栈 undo/redo。
- 图片源用 `thumb?raw=1` 取原图，不能用 1024px thumb（精度）。
- filmstrip 沿用裁剪页模式（256px thumb + 虚拟滚动）。

**后端**：

- `POST .../versions/{vid}/preprocess/inpaint/save`：multipart PNG 整图上传，
  校验 rel path（`_validate_rel_name` 两段格式），tmp + atomic rename 覆盖，
  manifest touch。缩略图 mtime 失效机制自动生效。
- 无新 job 类型（同步写盘，非任务队列操作）。

**其他**：i18n 占位文案去掉「高斯」；`inpaint` tab 解禁。

### PR-A 页面布局（2026-07-12 定稿，对齐裁剪页骨架）

```
StepShell
├─ Header actions（次=ghost / 主=primary+icon 最右，镜像裁剪页三按钮）
│    [还原本图 ghost·confirm]  [保存全部 (n) ghost]  [▶ 保存当前图 primary]
├─ belowHeader: PreprocessToolsBar（inpaint active）
└─ 内容区 —— 单 section 卡片占满全宽（无统计 RightRail，涂抹无统计语义）
   └─ section
      ├─ header 行: [全部/未改/已改] pills · 文件名 · W×H    [清除本图笔画 ghost]
      └─ 内层三栏 grid '220px minmax(0,1fr) 260px'（同裁剪页参数）
         ├─ Filmstrip（自裁剪页抽出共享；「已改」dot 角标）
         ├─ InpaintCanvas：原图分辨率离屏 canvas + CSS 缩放；滚轮 zoom /
         │    空格拖拽 pan；笔刷圆形光标；底部细条 readout（zoom%·坐标）
         └─ ToolPanel：模式切换占位（涂抹|Mask，PR-B 启用）· 颜色 swatch
              + 吸管（Alt 临时吸管）+ 最近用色 · 笔刷大小/硬度 slider ·
              undo/redo · 底部固定栏提示文案（保存覆盖原图，可还原）
```

**状态模型**：`strokesByImage: Record<name, Stroke[]>` 存笔画矢量（点序列 +
颜色 + 大小 + 硬度），镜像裁剪页 `cropsByImage`。只有活动图持有真实 canvas
（避免多张 4K 位图驻留内存），切图对目标图重放笔画——前端同引擎重放无一致性
问题。交互心智与裁剪页一致：随便切图、改动留在内存、统一提交，无切图守卫。

- 保存当前图 = 活动 canvas `toBlob` 整图上传；成功后清该图笔画（对齐裁剪页
  job done 清 `cropsByImage`），mtime cache-buster 自动刷缩略图。
- 保存全部 = 串行逐图 加载原图 → 重放 → 上传，进度 toast。
- 「清除本图笔画」（section header，清内存）与「还原本图」（header actions，
  restore 磁盘）语义分开，两者都保留。
- filter 判据 = 内存有无笔画（同裁剪页以内存 rect 数为准的语义）。
- 无 logSources：保存为同步写盘，不走 job 队列。
- workspace 数据复用 crop workspace 端点（图列表 + w/h + mtime）。

## 5. PR-B 范围（场景 2，mask + 训练联动）

**前端**：同一编辑器加 mask 笔刷模式（红色半透明 overlay 叠加显示 / 橡皮擦），
保存导出灰度 PNG 上传；画廊 / filmstrip 上有 mask 的图显示角标；显示单图 mask
覆盖率（>50% 时警示，见 §8 质量红线）。

**后端**：

- mask 读写 endpoint（GET 返回 mask 文件或 404 / PUT 写入 / DELETE 清除）。
- 预处理变换跟随（见 §7 矩阵）。
- 删图 / restore 的 sidecar 跟随清理：`.mask` 与 `.txt`/`.json` 同为同 stem
  sidecar，走 `delete_mask` 跟随。
- bundle 导出可选打包 `.mask` sidecar（对齐 PR #391 latent 缓存打包的做法）。

**训练器**（触点已调研核实）：

| 触点 | 位置 | 改动 |
|---|---|---|
| 几何变换同步 | `dataset.py` `get_with_flip`（唯一变换实现点） | mask 用相同 scale / crop 坐标 / flip，NEAREST 插值 |
| latent 缓存 | `_encode_and_save` / `_is_cache_valid` / `CachedLatentDataset.__getitem__` | mask 下采样到 latent /8 分辨率存进同一 npz（`mask` / `mask_flipped` 键，仿 `latent_flipped` 双份）；缓存校验加 mask 键检查否则旧缓存不失效 |
| collate | `collate_fn` / `collate_fn_cached` | stack mask 字段（无 mask 图填全 1） |
| loss | `loop.py` 标准路径（loss 为 per-element `(B,C,T,H,W)`，现有权重均为 per-sample 广播） | mask 逐元素乘入，reduction 由 `.mean()` 改 `(loss*mask).sum()/mask.sum()` 加权均值；与 reg weight / timestep weighting 正交叠加 |
| InfoNoise | `loop.py` `_raw_mse` 记录 | per-sample MSE 同样用 mask 加权（否则 mask 区域噪声污染 CDF 统计） |
| NaViT | `navit.py`（缓存路径显式假设无 mask） | 第一版闸掉：NaViT + masked_loss 共存时 warning + 忽略 mask |
| config | 训练 schema | `masked_loss: bool`（显式开关，D7） |

## 6. Mask × 打标

核心事实：**mask 非破坏性，tagger 永远看到完整像素**，打标与画 mask 先后顺序
无关，天然无冲突。

- **caption 不自动联动**（不删 "hands" 等 tag）。训练语义：mask 区域无梯度，
  caption token 与该区域像素的关联本来就学不到；tag 保留与否影响甚微，属用户
  显式选择。
- 场景 1（涂抹改图，破坏性）与打标有顺序交互：**先涂抹后打标**则 caption 干净；
  先打标后涂抹则可能残留 "text / watermark" tag，重新打标即可解决。现有 phase
  顺序（curating → preprocessing → tagging）天然引导正确顺序，不强制。
- 打标页对有 mask 的图不做特殊渲染（P2 nice-to-have：角标提示）。

## 7. Mask × 其他预处理（变换跟随矩阵）

关键前提：所有破坏性变换都在后端执行且几何参数已知，mask 跟随可集中实现。
策略 = **变换跟随为主（B）+ 尺寸校验兜底（C，§2）**。

| 操作 | 对 mask 的处理 |
|---|---|
| 裁剪（单框原地覆盖） | 同 rect 裁 mask（NEAREST），原地覆盖对应 `.mask` 文件 |
| 裁剪（multi-crop fan-out `X_c0/_c1`） | mask 同步 fan-out 为 `X_c0.mask` 等，删原 mask |
| 放大 | mask 按相同目标尺寸 NEAREST resize（**不走** RealESRGAN） |
| 涂抹（场景 1 改像素） | 不改几何 → mask 不动，天然对齐 |
| restore（download 复原） | **删 mask**（D8）；即便尺寸恰好吻合也删，语义可预测优先 |
| 去重 / 删图 | mask 跟随删除（sidecar 清理登记，防孤儿） |
| fork 版本 | `_copytree("train")` 自动带走，零改动 |
| 训练时 bucket resize / crop / flip | 训练器侧 `get_with_flip` 同变换（§5），与未来多分辨率 fan-out（issue #246）兼容——变换按 bucket 在训练时计算，mask 跟同一路径 |

漏网情形（外部工具直接改图等）由训练器尺寸校验兜底降级为无 mask。

## 8. 质量红线与训练语义备注

- masked loss 为成熟技术，正确实现不破坏 LoRA 质量；风险仅在 **mask 面积**：
  mask 掉的区域无监督，单图 mask 过半等效于该图有效数据减半。UI 显示覆盖率
  并在 >50% 警示。
- 加权均值 reduction（除以 `mask.sum()`）保证不同 mask 面积的样本在 batch 内
  权重一致，避免 kohya 朴素 `mean()` 实现中大 mask 图被隐性降权的问题。
- 场景 1 提示（文档 / tooltip 级）：取色实心涂抹优于模糊；纯色块位置颜色
  不一致时学入风险低。

## 9. PR-B 实现决策（2026-07-12 讨论定稿）

1. **拆两个 PR**：
   - **B1（studio 层，数据面闭环）**：mask 笔刷 UI + mask 读写端点 + 预处理
     变换跟随（§7 矩阵）+ bundle 导出打包。合并后 mask 能画能存能跟预处理走，
     训练不消费——先合先用，攒 mask 数据。
   - **B2（训练器）**：dataset 变换同步、npz 缓存、loss 加权、config 开关、
     NaViT gate。对齐 #389/#390 拆写读路径的先例。
2. **保存交互**（B1 实装后修订，2026-07-13）：~~按当前模式分发 + 双桶独立~~
   → **统一编辑历史**。双桶模型让切模式改变按钮文案 / 可用性，体感割裂成两个
   页面；改为 `historyByImage` 单一时间线（`{kind: paint|mask, stroke}` 混合
   序列），dirty / undo / redo / 撤销修改跨模式共用，「保存当前图」一次写两个
   数据面（涂抹先行获取可能的改名，mask 用新 name 写入）。模式只是笔刷。
3. **npz 缓存失效三条**（漏一条就训到旧 mask，全部进 `_is_cache_valid`）：
   mask 存在但 npz 无 mask 键（新画）；mask mtime 新于 npz（重画）；mask 已删
   但 npz 有 mask 键（清除）。下采样时机 = 编码期存 latent 分辨率进 npz
   （对齐 `latent_flipped` 先例）。
4. **latent 下采样插值 = area**（每 latent 位置 = 8×8 像素块 mask 均值），与
   灰度连续权重语义自洽。不用 NEAREST（边缘跳变）/ max-pool（一像素吞整个
   latent 位置）。注意与 §7 预处理跟随中 mask 图像本身的几何变换（NEAREST，
   防灰度值被插值污染）是两回事。
5. **config**：`masked_loss: bool` 放 loss 相关 group（与 `loss_weighting`
   相邻）。UI：存在 `.mask` 文件且开关关闭 → 提示不代开；开关开但无 mask
   文件 → 训练器 log 一条，不报错。
6. **B2 合并门槛**：本地小 A/B（同数据集同 seed，手部 mask vs 无 mask），
   确认 (a) loss 曲线正常无 NaN；(b) 出图 mask 区域不带崩坏特征。不达标则
   B2 挂待判（先例：FFL）。

## 10. Open questions（不阻塞）

1. AI inpainting（LaMa / sd inpaint）作为第三种笔刷的可能性——依赖新模型下载，
   留待场景 1 用出真实需求后评估。
2. 多分辨率训练（issue #246）落地后 `1024px_` 覆盖文件夹内的 `.mask` sidecar
   跟随关系需要在该 feature 实现时一并确认。
3. 打标页 / 画廊的 mask 角标展示范围（B1 内做最小版还是单独 polish）。
