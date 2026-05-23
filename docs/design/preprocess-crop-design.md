# 预处理 · 裁剪 — 功能设计

> 临时设计文档，整理**逻辑模型 + 用户场景 + 数据契约**。实现细节看后续 PR。
>
> 落地后做了多轮 UI 迭代，详见 [§9 Addendum 1](#addendum-1--ui-演进2026-05-21)。

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

## 3. 功能

> **版本注**：v1 设计稿把"手动 / 聚类"做成 segmented tab，落地后迭代去掉
> 了 tab —— 裁剪只有一个概念，"智能聚类"是个**可选预填工具**而非独立模式。
> 详见 §9 Addendum 1。

### 主裁剪能力

**AR 下拉**：`自由(不锁)` / `1:1` / `4:3` / `3:2` / `16:9` / `3:4` / `2:3` / `9:16` / `4:5` / `自定义…`

- 自由 → 拖动新建任意 AR 的框
- 锁定 → 新框按 AR；resize handle 等比；移动不影响 AR
- 自定义 → 弹两个数字输入 W、H
- 一图多裁：N 个框 → N 个产物

**画布交互**：8 handle（4 角 + 4 边） + 三分网格 + 暗色 dim 框外 + live 像素尺寸 / AR readout。

**右侧 rect list**：缩略 + 可编辑 label + 输出像素 + 复制 / 删除（选中时 header 出 icon）。

**filter chips**：全部 / 待裁剪 / 已裁剪（按本 session 内 `cropsByImage` 状态过滤）。

**主操作**：`裁剪当前图` / `▶ 裁剪全部(N)`。

### 智能聚类（可选预填）

OperationPanel 下方独立 section，**默认折叠**；点 `▸ 智能聚类` 展开。

**参数**：`max_crop ∈ [0, 0.30]`（最大允许裁面积比）、`k_min ∈ [1, 10]`、`k_max ∈ [2, 15]`。

**算法（前端 JS）**：

1. 对 `preprocess/` 所有图算 AR = w/h
2. 1-D k-means 在 `[k_min, k_max]` 区间，用 elbow 挑 k
3. 每 cluster 中心 → snap 到训练桶网格（见 §7 ARB 对齐）
4. 显示 label 取最近常用 pretty AR；rect 用训练桶 AR 算
5. 每张成员按 target AR 居中裁剪到最大可填矩形
6. `max_crop` 约束：裁掉面积比 > max_crop 则不加框（用户可手动处理）

**结果**：写入 cropsByImage（每图 1 个 ✦ 标记的 cluster 来源框）。聚类后用户可任意微调 / 删 / 加（用同一个主画布）。

**主操作**：`▶ 开始聚类` — section 内独立按钮，**不是** 提交到磁盘。提交还走外层 `裁剪全部`。

---

## 4. 后端契约

### 新增 endpoint

```
POST /api/projects/:id/preprocess/crop
body: {
  crops: {
    "IMG_2741.png": [
      { x: 0.10, y: 0.05, w: 0.55, h: 0.45, label: "头像" },
      { x: 0.12, y: 0.42, w: 0.72, h: 0.55, label: "全身" }
    ],
    "IMG_2742.png": [ { x: 0.25, y: 0.12, w: 0.50, h: 0.78, label: "" } ]
  }
} → Job
```

辅助 endpoint：

- `GET /api/projects/:id/preprocess/crop/workspace` —— 列出所有可裁剪的图
  （preprocess/ 已处理 + download/ 未处理合并），含 PIL 读图头返回的 w/h；
  前端裁剪页 filmstrip 用
- `POST /api/projects/:id/preprocess/files/reset` —— 总览 tab 的"撤销全部"
  调用，清空 manifest + 删 preprocess/ 所有 PNG

### Worker 逻辑

对每个 source name：

1. resolve source path → preprocess/source 或 download/source
2. PIL 打开
3. 对每个 rect：`crop()` → 写 `preprocess/{stem}_c{n}.png`（n>1 时）或 `preprocess/{stem}.png`（n=1，覆盖）
4. n>1 时删原 `preprocess/{stem}.png`（如果存在）
5. manifest 加 N 条 entry，origin 指源 download 名

### SSE 事件

- `crop_progress`：单图完成推一次，但 worker 端**节流 ≥ 1Hz**（首末 / skip / fail 强发，其余 done 跨 ≥ 1s 才 emit）。避免 264 张数据集刷 ~500 个事件淹没事件流。
- `job_state_changed`：状态变化

---

## 5. 前端结构

> **版本注**：v1 设计稿是 `/preprocess/crop` 子路由 + 横向 filmstrip + stage
> pills 内嵌 OperationPanel。落地后改成 query string `?tool=crop` + 共享工具栏
> + 竖向 filmstrip，详见 §9 Addendum 1。

### 路由

预处理工具共用 `/projects/:pid/preprocess` 单路由 + `?tool=` query：

- `/preprocess` （默认）/ `?tool=upscale` → 放大工具
- `?tool=overview` → 总览（多选 + 撤销）
- `?tool=crop` → 裁剪工具
- `?tool=inpaint` → 占位（未实现）

入口由 `PreprocessHub.tsx` 调度。query 切换不卸载父路由，工具切换更顺；侧栏 `/preprocess` 匹配也不被打断。

### 入口

页面顶部独立**工具栏**（`PreprocessToolsBar.tsx`）三个 / 四个 pill，左侧首位是「总览」，pill 即工具，点了变 `<Link to="?tool=...">`。**没有完成 ✓ 徽章** —— 工具不是 pipeline 节点。

### 页面布局（共享框架）

```
StepShell (title / subtitle)
└─ grid 1fr / 260px
   ├─ 左
   │  ├─ PreprocessToolsBar  [总览][放大][裁剪][涂抹]
   │  ├─ OperationPanel (工具专属配置)
   │  │  ├─ AR 下拉 + 主操作按钮
   │  │  └─ 智能聚类 section（默认折叠）
   │  ├─ PreprocessJobStrip （job 在跑 / 有 logs 时才显示）
   │  └─ WorkArea (裁剪)
   │     ├─ filter chips · 当前图 meta · 清空本图
   │     └─ grid: filmstrip 220px / canvas 1fr / rect list 260px
   └─ 右 RightRail (裁剪进度 / 预估产物 / AR 分布 / 盘占用)
```

WorkArea 内部三列：filmstrip 竖排（3 col 正方 cover thumbs）/ canvas 容器测量自适应 / rect list 选中时 header 出 ⎘ ✕ icon。

### 总览 tab（overview）

独立页 `PreprocessOverview.tsx`：所有 preprocess workspace 图 grid + 单击预览 modal + ctrl/shift 多选 + `撤销选中` + `↶ 撤销全部`。撤销逻辑从放大页移到这里，所有工具都不再单独处理撤销，UX 心智模型统一。

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

---

## 9. Addendum 1 — UI 演进（2026-05-21）

设计稿落地后多轮 UI 迭代，记录主要偏离原稿的决策：

### 9.1 去掉「手动 / 智能聚类」segmented tabs

v1 把这两个做成 segmented tab 互斥切换。用户反馈"它们不是互斥关系"：聚类后生成的框可以手动改，根本就是同一个裁剪能力。

落地：
- 删 mode tabs
- 主裁剪能力（AR 下拉 + 画布 + 主操作）始终可见
- "智能聚类"降级为 OperationPanel 下方独立 section，默认折叠 `▸ 智能聚类`，点开看 sliders + `开始聚类` 按钮
- 状态保留：聚类完成后 section 顶上挂 `✓ k=N` 徽章

### 9.2 URL 从 `/preprocess/crop` 切到 `/preprocess?tool=crop`

子路径模型有两个问题：(1) 侧栏 `/preprocess` 路径匹配被 `/crop` 后缀打断，高亮丢失；(2) 工具切换导致父路由卸载，状态全丢。

落地：
- 单一路由 `/projects/:pid/preprocess`
- query string `?tool=overview|upscale|crop|inpaint` 调度
- 新建 `PreprocessHub.tsx` 调度器
- 工具切换不卸载父路由

### 9.3 "阶段" 改 "工具"，去掉 ✓ 完成态

stage pills 暗示 pipeline 时序，但放大 / 裁剪 / 涂抹都不是 stage，是任意顺序可用、可重复用的工具。

落地：
- 文案 "阶段" → "工具"
- 共享组件 `PreprocessToolsBar.tsx` 放在每个工具页顶部
- pill 没有完成徽章

### 9.4 总览 tab — 撤销统一入口

放大页本来有"还原 N 张"按钮。但裁剪等其他工具也需要撤销，每个工具各自加是冗余。

落地：
- 新建 `PreprocessOverview.tsx` 总览页
- 工具栏左侧 `[总览]` pill
- ImageGrid + shift/ctrl 多选 + 单图 preview modal
- "撤销选中 N" + "↶ 撤销全部" + confirm modal
- 放大页移除还原控件，image grid 保留
- 后端新增 `POST /preprocess/files/reset` 路由到 `preprocess_manifest.clear_all()`

### 9.5 Filmstrip 从底部横排改左侧竖排

264 张图横排会挤成 5px 一条根本看不见。改竖排 3-col 正方 cover thumb，给画布让出更多上下空间。

CSS 注意：`<button>` 直接挂 `aspect-ratio: 1` 在 grid 里会塌缩（Chromium/WebKit anonymous flow-root 影响 `::before` padding-top）。包一层 div 做 padding-top trick 才稳。

### 9.6 画布按容器测量自适应

固定 maxWidth/maxHeight 在不同视口要么浪费要么溢出。改 ResizeObserver 测父容器，maxWidth/maxHeight 退为兜底上限。

### 9.7 AR-lock resize 两个坑

- **缩塌成全图**：超出画布时独立 clamp w/h 会破 AR（1:1 锁定的 rect 在 2:3 源图上变全图 = 2:3）。修：按锚定角缩比例，永远保 AR
- **磁吸感**：拖出后反方向拉要先消化累计 dxN/dyN 才动。修：每帧重锚定，delta 始终是上一帧到现在的增量

### 9.8 Multi-crop 缩略图寻址

`bucket=download` + `resolve_origin` 取 `[0]` 在 multi-crop 后多个派生共享 origin 时永远落到同一张缩略图。

落地：
- thumb endpoint 加 `bucket=preprocess`，直接按 preprocess 文件名寻址
- 兜底：`bucket=download` 找不到文件且 name 是 manifest entry key 时也走 preprocess/
- 裁剪页 / 总览页 / 放大页都按"已处理走 preprocess bucket + im.name，未处理走 download" 寻址

### 9.9 SSE 节流

crop 速度比 upscale 快（单图 300-700ms），264 张 ~500 个事件淹没 EventSource。

落地：worker 内 `emit_throttled(force=...)`：done 事件 ≥1s 间隔；首末 / skip / fail 强发。

### 9.10 像素分布 + 训练桶对齐

右栏统计原本只是裁剪的 AR 分布。放大页移植了像素面积 histogram（6 bin），跟 sd-scripts ARB 训练桶语义对齐（见 §7 ARB 对齐）。

放大页 filter chips 也从 `全部 / 未处理 / 已处理` 改成 `全部 + 像素 bins`（同 sidebar histogram），UX 上「未处理 / 已处理」对放大无意义。

### 9.11 JobStrip 不持久化日志

刷新页面后 status endpoint 还在返回历史 job + log_tail，结果空 JobStrip 蹲在页面上没意义。

落地：
- 不再从 `status.log_tail` 初始化 logs，logs 仅本 session SSE 累积
- JobStrip 渲染条件加 `(isLive || logs.length > 0)`，无活跃 job + 无 session log 时整块隐藏
