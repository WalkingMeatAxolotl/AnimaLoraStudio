# 预处理 · 裁剪 — 功能设计

> 临时设计文档，整理**逻辑模型 + 用户场景 + 数据契约**。实现细节看后续 PR。

## 0. 目的与非目的

**目的**：在预处理的「放大」之后增加「裁剪」stage，允许用户对 `preprocess/` 工作集做手动 / 智能两种切图。

**非目的**：
- 不引入第三类工作目录。`download/` 仍是唯一备份，`preprocess/` 是唯一工作集。
- 不强制 stage 时序（放大 → 裁剪 → 放大 → 裁剪 都合法）。
- 不做 partial undo，还原只回到 download。
- 不持久化裁剪过程信息（rect 坐标、target AR、cluster id）—— 一旦写盘，过程丢。

---

## 1. 数据模型

### 文件夹

| 目录 | 角色 | 是否可变 |
|---|---|---|
| `download/` | 唯一原图备份 | **永不动** |
| `preprocess/` | 当前工作集，每次 stage 直接覆盖 | 可变 |

### 文件命名

- 默认 1:1：`preprocess/X.png` 对应 `download/X.png`
- multi-crop 派生：`preprocess/X_c0.png`、`preprocess/X_c1.png`，origin 都指 `download/X.png`
- 多次裁剪：`X_c0.png` 再分多裁 → `X_c0_c0.png` / `X_c0_c1.png`（origin 仍 `X.png`）

### Manifest schema（新版）

```jsonc
{
  "images": {
    "X.png":     { "origin": "X.png",  "mtime": 1731000000, "size": 1234567 },
    "Y_c0.png":  { "origin": "Y.png",  "mtime": ...,        "size": ...     },
    "Y_c1.png":  { "origin": "Y.png",  "mtime": ...,        "size": ...     }
  }
}
```

只记 `{origin, mtime, size}`。**不**记 `kind / model / scale / action / target_area / src_size / dst_size / elapsed_seconds`（这些都属于"过程信息"，写盘后丢弃）。

### 老 schema 兼容

ADR 0004 的 `{kind, model, scale, action, target_area, src_size, dst_size, elapsed_seconds, source}` 字段：
- **读时兼容**：缺 `origin` 时，取 `source ?? entry_key`
- **写时丢弃**：保存只写新字段
- 几个 version 后逐步 deprecate 老字段读支持

### resolve(download_name)

下游（curation / thumbnail / copy_to_train）：

- manifest 里存在 origin = `download_name` 的 entry → 返回这些 preprocess/ 文件（可能多个）
- 否则 → 返回 `download/{download_name}`

### 还原

删 preprocess 文件 + 所有 `origin == download_name` 的 entry。下游回看 download 原图。**不**做单 stage undo。

---

## 2. User cases

| # | 场景 | 模式 | 输出 |
|---|---|---|---|
| U1 | 一张全身图保头像 + 全身 | 手动 / 自由 AR / 多框 | `X_c0.png` (头像) + `X_c1.png` (全身) |
| U2 | 训练桶统一 1:1 / 2:3 | 手动 / 锁 AR / 单框 | `X.png` (覆盖) |
| U3 | 数据集 AR 杂乱想分桶 | 智能聚类 → 微调 | 每图 `X.png` (覆盖) |
| U4 | 自定义比例 5:7 | 手动 / 自定义 W:H | 同 U2 |
| U5 | 某图直通 | 不画框 | preprocess 文件保持原样 / 还原后用 download |
| U6 | 回到原图 | 还原 | 删 preprocess entry + 文件 |

---

## 3. 模式

### A. 手动裁剪

**AR 下拉**：`自由(不锁)` / `1:1` / `4:3` / `3:2` / `16:9` / `3:4` / `2:3` / `9:16` / `4:5` / `自定义…`

- 自由 → 拖动新建任意 AR 的框
- 锁定 → 新框按 AR；resize handle 等比；移动不影响 AR
- 自定义 → 弹两个数字输入 W、H
- 一图多裁：N 个框 → N 个产物

**画布交互**：8 handle（4 角 + 4 边） + 三分网格 + 暗色 dim 框外 + live 像素尺寸 / AR readout。

**右侧 rect list**：缩略 + 可编辑 label + 输出像素 + 复制 / 删除。

**filter chips**：全部 / 待裁剪 / 已裁剪（按 manifest entry 数量过滤）。

**主操作**：`裁剪选中(n)` / `▶ 裁剪全部(N)`。

### B. 智能聚类

**参数**：`max_crop ∈ [0, 0.30]`（最大允许裁面积比）、`k_min ∈ [1, 10]`、`k_max ∈ [2, 15]`。

**算法（前端 JS）**：

1. 对 `preprocess/` 所有图算 AR = w/h
2. 1-D k-means 在 `[k_min, k_max]` 区间，用 elbow / silhouette 挑 k
3. 每 cluster 取一个 target AR（cluster 中位 + 贴近常用桶 1:1 / 4:3 / 3:2 / 16:9 等）
4. 每张成员按 target AR 居中裁剪到最大可填矩形
5. `max_crop` 约束：裁掉面积比 > max_crop 则不加框（用户可手动处理）

**结果**：写入 cropsByImage（每图 1 个 ✦ 标记的 cluster 来源框）。

**后续**：自动切回手动编辑器，用户可任意微调 / 删 / 加。

**主操作**：`▶ 开始聚类` / `裁剪全部(N)`。

---

## 4. 后端契约

### 新增 endpoint

```
POST /api/projects/:id/preprocess/crop
body: {
  mode: 'all' | 'selected',
  crops: {
    "IMG_2741.png": [
      { x: 0.10, y: 0.05, w: 0.55, h: 0.45, label: "头像" },
      { x: 0.12, y: 0.42, w: 0.72, h: 0.55, label: "全身" }
    ],
    "IMG_2742.png": [ { x: 0.25, y: 0.12, w: 0.50, h: 0.78, label: "" } ]
  }
} → Job
```

### Worker 逻辑

对每个 source name：

1. resolve source path → preprocess/source 或 download/source
2. PIL 打开
3. 对每个 rect：`crop()` → 写 `preprocess/{stem}_c{n}.png`（n>1 时）或 `preprocess/{stem}.png`（n=1，覆盖）
4. n>1 时删原 `preprocess/{stem}.png`（如果存在）
5. manifest 加 N 条 entry，origin 指源 download 名

### SSE 事件

- `crop_progress`：单图完成推一次（per image）
- `job_state_changed`：状态变化

---

## 5. 前端结构

### 路由

- `/projects/:id/preprocess` → 现有放大页（status quo）
- `/projects/:id/preprocess/crop` → 新增裁剪页

### 入口

放大页的 OperationPanel stage pills `[放大 ✓] [裁剪 ●] [涂抹]`：现在 `裁剪` 是 disabled placeholder。改为 link 跳 `/preprocess/crop`。反之裁剪页 stage pills 的 `放大` 可点回去。

### 页面布局（沿用放大页同模式）

```
StepShell (title / subtitle / stepper)
└─ grid 1fr / 260px
   ├─ 左
   │  ├─ OperationPanel (compact)
   │  │  ├─ [手动][聚类] segmented + 主操作
   │  │  ├─ AR 下拉 (或 3 sliders)
   │  │  └─ stage pills
   │  └─ WorkArea
   │     ├─ filter chips · 当前图 meta · 清空本图
   │     ├─ canvas + rect list (260px)
   │     └─ filmstrip
   └─ 右 RightRail (裁剪进度 / 预估产物 / AR 分布 / 盘占用)
```

---

## 6. 实施切分

| Step | 工作量 | 说明 |
|---|---|---|
| 1 | M | 后端 endpoint + worker + manifest 读兼容 / 写新 schema |
| 2 | M | 前端：CropPage 容器 + 路由 + OperationPanel |
| 3 | L | 前端：FreeCropEditor 画布 + 手势 + AR-lock |
| 4 | M | 前端：rect 列表 + filmstrip + filter chips + RightRail |
| 5 | S | 前端：聚类 JS（k-means + elbow + max_crop 约束） |
| 6 | S | 放大页 stage pill 改 link + i18n 补字 |
| 7 | S | 测试（pytest crop endpoint + manifest，vitest editor + k-means） |

---

## 7. ARB 桶对齐（裁剪与训练桶一致）

### 7.1 问题

训练时 `runtime/training/dataset.py:BucketManager` 按 (base_reso=1024, step=64,
area_tol=0.10, max_ar=2.0) 派生 ~30 个 (w, h) 桶；每张图按 **AR 绝对距离** 落到最近桶
并 resize 到该桶尺寸。聚类裁剪如果挑 "4:3 = 1.333" 这种 pretty AR 当 target，裁出来
的图 trainer 会再二次 resize 到 (1152, 896) = 1.286 或 (1216, 832) = 1.461 —— 引入额外
失真。

裁剪聚类的 target AR 应当**和 trainer 实际会落的桶完全一致**，trainer 拿到图就不再
做第二次 resize。

### 7.2 UX 原则（用户不需要知道 ARB 内部）

底层 ARB 桶（"1024×1024"、"1216×832"、桶数量、面积带、step 这些）**永远不暴露给用户**：

- **不懂 ARB 的用户**：默认值 work，看到的标签都是 `1:1` / `4:3` / `3:2` / `16:9` 这种
  熟悉的比例，照常用
- **略懂的用户**：知道 4:3 是横向比例，照常用
- **深懂的用户**：他想知道底层细节自己去看源码 `runtime/training/dataset.py`，UI 不替他展示

另：手动模式支持"裁掉烂的部分"用例（自由 AR + 拖动），不强制对齐训练桶。

### 7.3 实现

**Internal**（不暴露）：
- 前端 `studio/web/src/lib/trainBuckets.ts` 把 Python `BucketManager` 算法 1:1 移植成 TS
- 默认参数硬编码 `base_reso=1024, min_reso=512, max_reso=2048, step=64,
  area_tolerance=0.10, max_ar_ratio=2.0` —— 与 backend 默认 100% 一致
- `generateBuckets()` 生成桶网格；`snapToBucket(aspect, buckets)` 按绝对 AR 距离 snap

**接入点**：
- 聚类目标 AR：cluster 中心 → `snapToBucket()` → 训练桶 (w, h)，rect 用这个比例算
- 聚类卡片 label 显示：取训练桶 AR 的"最近 pretty AR"作显示标签（如 `聚类 3:2`），
  内部 rect 严格按训练桶比例

**不接入**：
- Histogram (`arBucket`)：保持现状 snap 到 11 个 pretty AR，按 aspect 排序
- 手动模式 AR 下拉：保持现状（`1:1` / `4:3` / ... / `自定义 W:H`），UX 优先

### 7.4 防漂移

backend `runtime/training/dataset.py:BucketManager` 和 frontend `lib/trainBuckets.ts`
是两套独立实现的同一算法，最怕改一边忘另一边。

- 两边文件顶部互引注释，明示"改算法 / 默认参数 → 必须两边同 commit"
- review 阶段把这俩文件列为联动文件
- 后续可加跨语言同步测试（option，先不做）

### 7.5 base_reso 从哪取

**硬编码 1024**。理由：
- 覆盖 SDXL / Flux / Anima 默认场景（90%+ 用户）
- 用户在 preprocess 阶段还没必要去想训练分辨率
- SD1.5 用户（少数）即使桶预测略偏，trainer 也会按其真实参数 re-bucket，最多多一次
  轻微 resize，不影响训练
- 加 UI 控件等同于暴露 ARB 概念，违反 §7.2 原则

base_reso 可调当 follow-up 处理（如果出现项目级痛点）。

---

## 8. 不做的事

- **rect 不持久化**：写盘后过程信息丢，重画从头
- **stage 时序约束**：无（放大 ↔ 裁剪任意顺序、任意次）
- **partial undo**：无（只能整图还原到 download）
- **多 manifest / 多目录**：无（仍单 manifest + 单 preprocess/）
- **后端跑聚类**：无（前端 JS）
- **保留旧 upscale 产物作为裁剪备份**：无（每 stage 覆盖）
- **暴露 ARB 底层（base_reso / step / 桶数 / 桶 (w,h)）给用户**：无（见 §7.2，UX 原则）
- **base_reso 项目级可调**：无（硬编码 1024，见 §7.5）
- **手动模式 AR 下拉换成训练桶**：无（保 pretty AR，UX 优先）
