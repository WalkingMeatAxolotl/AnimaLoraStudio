# 0004 — 预处理状态用单 manifest 替代「双 bucket + per-image sidecar」

**状态**：Accepted（amended 2026-05-21 — 见 Addendum 1）
**日期**：2026-05-15
**决策者**：@WalkingMeatAxolotl

## 背景

PR #69 给 AnimaLoraStudio 加了"预处理"阶段（放大器 + 智能跳过策略）。落地后产生
两个 UX / 架构问题：

### 1. UX：磁盘结构被原样翻译成 UI

当前流水线在磁盘上是两层平行目录：

```
projects/{id}-{slug}/
  download/      原图
  preprocess/    产物 PNG + 每张图旁 {name}.preprocess.json sidecar
```

前端 `Preprocess.tsx` 跟磁盘一一映射：「未处理 grid」对应 `download/` 里没有产物的图，
「已处理 grid」对应 `preprocess/` 里有产物的图。用户必须**打开预处理 tab 才能"发现"
有多少图需要处理**，而且页面上"两个网格"的视觉模型暗示了"两份图"，跟用户脑里
「我的数据集 = 一套图」的心智模型不符。

下游接缝更暴露这个问题。`studio/curation.py:98-109` 的 `_active_left_dir`：

```python
def _active_left_dir(pdir):
    pre = pdir / "preprocess"
    if pre.exists() and any(f.is_file() ... for f in pre.iterdir()):
        return pre, "preprocess"
    return pdir / "download", "download"
```

「preprocess/ 非空 → 用 preprocess/，否则用 download/」**全有或全无**。
意味着用户没法 mix——比如「100 张里只想 upscale 那 20 张小图，剩下 80 张用原图」
做不到（一旦 preprocess/ 有产物，整批左侧都切到 preprocess/，那 80 张就缺源
被过滤掉了）。

`left_source` 字段被泄漏到前端 API（`tests/test_curation_endpoints.py:78` assert
这个字段），前端拿来切 thumbnail URL builder——又一处把磁盘结构暴露给客户端。

### 2. 架构：状态散在 N 个 sidecar 里

`studio/services/upscaler.py:353` 每张处理完写 `{name}.png.preprocess.json`：

```json
{"source": "...", "model": "...", "scale": 4, "src_size": [W,H], "dst_size": [W,H],
 "action": "upscale", "elapsed_seconds": 12.3, ...}
```

`studio/preprocess.py:list_processed` 列图时逐张读 sidecar 拼成响应。这套有几个问题：

- **没有"用户决定"状态**：sidecar 只在 worker 跑完才落，但 sidecar 缺失 ≠ "用户决定
  跳过"——可能只是没跑过、也可能跑失败被删了。无法表达「这张图我**故意**不处理，
  用原图」的语义
- **批量操作贵**：1000 张图列一次要 1000 次 stat + 1000 次 JSON parse
- **没有原子性**：单张状态没事，但「重置该项目预处理状态」要 1000 次 unlink，中途
  ctrl-c 会留下半干净的目录
- **「还原」逻辑分散**：要删 PNG + 删 sidecar 两处，少删一个就状态错乱

### 3. 用户提出的目标心智模型

用户原话：

> 创建版本的时候直接在 preprocess 文件夹创建副本，没改过的图可以用 json 表示状态
> 来占位，后续筛选从 download 拿去实际文件，修改过的图创建副本，筛选从 preprocess 拿，
> 在用户的眼里没有 preprocess，只有一个文件夹。

要的是**一份图 + 每张图的处理状态**，磁盘结构对用户不可见。

设计讨论中追加的约束：

- preprocess/ 保持 **project 级**（不下沉到 version 级）——理由：upscale 几百张是
  真贵，跨版本复用预处理结果远比"每版独立预处理"更符合实际工作流（用户更多在
  iterate 数据集组合 + 训练超参，预处理决定通常一次确定）
- **一个 manifest 文件**记录所有状态，**不是** N 个 sidecar
- **没有"删除"状态**——筛选阶段不勾即可，预处理页不引入"删图"语义，避免混淆"删原图
  还是删产物"
- **无版本号字段**——YAGNI，真要 migrate schema 那天再加
- **隐式 original**：manifest 里没记的图默认 = 用原图，manifest 只存非默认决定

## 候选方案

### 方案 A — 现状维持

继续用 `_active_left_dir` 双 bucket + sidecar。

- 优点：0 工时
- 缺点：上述 UX + 架构问题都不动；将来加"用户跳过该图"语义无处放
- **拒绝**

### 方案 B — preprocess 下沉到 version 级 + 同样 N 个 sidecar

`versions/{label}/preprocess/`，每版独立。

- 优点：v1 不预处理 / v2 全 upscale 这种对照实验可做；筛选/打标/训练全 version 级一致
- 缺点：跨版本复用预处理结果**做不了**——同一批图在 v1 处理完，v2 还要重处理一遍
  几百张 upscale，每次几十分钟。这是预处理这一阶段最贵的成本
- **拒绝**（被用户明确否决：「不改 version 级，不然每次都要重新处理」）

### 方案 C — project 级单 manifest + 隐式 original（**选中**）

- 一个 `projects/{id}/preprocess/manifest.json` 记录**非默认决定**
- 「manifest 没记」= 用 `download/` 原图（隐式 original）
- 「manifest 里有 `kind: processed`」= 用 `preprocess/{name}.png`（实际副本）
- Resolver 一个函数，所有下游（thumbnail / curation / 打标 / 训练 materialize）调它
- 优点：UX 单 grid + 状态徽章；下游接缝从「双 bucket fallback」坍缩成「resolver 查表」；
  per-image 状态可表达；批量列图一次 JSON read 完事
- 代价：需要 supervisor 串行写 manifest（防并发）；migration 老 sidecar
- 详细落地见下节

## 决策

选**方案 C**。

### Manifest schema

`projects/{id}/preprocess/manifest.json`：

```json
{
  "images": {
    "bar.png": {
      "kind": "processed",
      "source": "bar.jpg",
      "model": "RealESRGAN_x4",
      "scale": 4,
      "action": "upscale",
      "target_area": 1048576,
      "src_size": [512, 512],
      "dst_size": [2048, 2048],
      "elapsed_seconds": 12.3,
      "mtime": 1731000000
    }
  }
}
```

- key = 产物文件名（始终 `.png`）
- value.kind 当前只有 `"processed"` 一种；未来可扩展（如 `"cropped"`、`"masked"`）
- 隐式 original 不写 entry
- **无 `version` 字段**：要扩 schema 那天，老文件无字段 = 当 v0

### Resolver 单点

`studio/services/preprocess_manifest.py:resolve(project, name)` → `Path | None`：

```python
def resolve(project_dir, name):
    m = load_manifest(project_dir)
    entry = m["images"].get(name)
    if entry is None:
        return project_dir / "download" / name   # 隐式 original
    if entry["kind"] == "processed":
        return project_dir / "preprocess" / name  # 副本
    raise ValueError(f"unknown kind: {entry['kind']}")
```

所有读图入口（thumbnail / curation 左侧 / copy_to_train）走这一个函数。
**删除 `_active_left_dir` / `list_left_source` / API 响应里的 `left_source`**。

### 并发写：supervisor 单消费者

所有 manifest mutation 通过 `studio/services/preprocess_manifest.py:_with_lock`
（threading.Lock 进程内串行 + 原子 tmp+rename 落盘）。两个写源都走这个：

- 预处理 worker 跑完一张 → `add_processed(project_id, name, meta)`
- 用户点"还原" → `restore(project_id, name)`（删 entry + 删 PNG）

进程内 lock 而非文件锁的理由：服务端单进程，CLI 不会绕过去写 manifest（CLI 没有
预处理工作流）。如果未来需要跨进程，再升级到 `portalocker`。

### 「未列出」语义

list_pending / list_processed 改为读 manifest + download dir 做 diff：

- `download/foo.png` 存在 + manifest 无 entry → "未处理"（pending）
- `download/foo.png` 存在 + manifest 有 entry → "已处理"（processed），缩略图走 preprocess/foo.png
- `download/foo.png` 不存在 + manifest 有 entry → 孤儿（产物但源已删），UI 标 orphan
- `download/foo.png` 不存在 + manifest 无 entry → 不存在的图，不返回

### Migration

旧项目里有 `*.preprocess.json` sidecar（按 `studio/preprocess.py:48` 的 `SIDECAR_SUFFIX`
约定写在 preprocess/ 下）。第一次访问该 project 的 preprocess 数据时：

1. 检测 `manifest.json` 是否存在；存在 → 跳过 migration
2. 扫 `preprocess/*.preprocess.json` → 聚合写成 manifest.json
3. 老 sidecar 文件**保留不删**（防御性回滚 + 0 删除风险）；新代码不再读它们

Migration 是幂等的：manifest 存在就直接返回，不再尝试。

### UI 模型

`Preprocess.tsx` 从「双 grid」改成「单 grid + 状态徽章」：

```
共 N 张 · 未处理 X · 已处理 Y                    [全部] [未处理] [已处理]

┌─[img]─┐ ┌─[img]─┐ ┌─[img]─┐
│  ⊘ 待 │ │ ✓ 4x │ │  ⊘ 待 │
└───────┘ └───────┘ └───────┘

┌─ 待处理 X 张 · [模型 ▾] [tile 256] [开始预处理] ─┐
```

「还原」按钮放在已处理图的预览/编辑 overlay 里；同时支持多选批量还原。

### 外部删 download 文件不做特殊处理

用户在 OS 文件管理器里删 `download/foo.png`：

- manifest 里没 entry → resolver 返 None → 下游 skip，不报错
- manifest 里有 entry → preprocess/foo.png 仍存在，下游照常拿；UI 标 orphan

**不做主动 reconcile**——这是用户行为，不做"自动清理"，避免吃掉用户没想删的状态。

## 理由

**为什么 single manifest 而非保留 N sidecar 但补一个总览文件**：DRY，状态只有一处真理。
两套并存意味着 update 时要两边同步，必然漂移。

**为什么 project 级而不下沉到 version 级**：用户的真实工作流决定的——「预处理一次，
跨多版本复用数据集组合 + 训练超参」是常态，「为不同版本做不同预处理」是稀有需求。
后者将来真要做可以加 version override 字段（manifest 内嵌 per-version branches），
但 v1 不预投资。

**为什么隐式 original 而不全显式占位**：

- 节省 manifest 体积（1000 张图里改 50 张 → 50 条 entry）
- 「没决定」和「决定用原图」两种状态自然合并——用户语义上没区别
- 「新增图自动算未处理」不需要额外回填代码

**为什么没有「删除」状态**：用户明确说 v1 不要。"在预处理页删图"会让用户困惑
"删的是哪个目录"。删图是筛选阶段的事：不勾它，不进 train/。

**为什么没有 version 字段**：YAGNI。Schema 改是稀有事件，真要改可以加字段并按
「没字段 = v0」处理老文件。提前加 `"version": 1` 是占位仪式感，没解决任何真问题。

**为什么 supervisor 串行而非文件锁**：服务端单进程，没跨进程写者。文件锁是为了
应对多进程 / 外部 CLI 同时写——这个场景不存在。降级到 threading.Lock 简单 90%。

## 后果

### 好处

- 用户视角磁盘抽象消失：UI 只有「一套图 + 状态徽章」
- 下游 (`curation.py` / `server.py` 缩略图 / `copy_to_train`) 接缝坍缩到 `resolver()` 单点
- per-image 状态有了显式 schema，将来加「裁剪 / mask / 标记跳过」都是 manifest 加 kind
- 批量列图从 O(N) stat+JSON-parse 降到 O(1) 读 manifest
- 「重置该项目预处理」原子化（rm preprocess/ + 重写空 manifest）

### 代价 / 新增约束

- **所有 manifest 写必须经 `_with_lock`**——任何旁路写都会丢更新。代码 review
  时需要把住这条
- **Migration 入口要正确**：第一次访问该 project 预处理数据时触发；如果有路径绕过
  入口直接读 manifest，需要再补 migration 调用
- **老 sidecar 不删但被忽略**：磁盘占用增加（每张 ~500B），可接受。下次大版本可
  考虑 v2 一并清理
- **测试覆盖面变大**：需要单测 manifest schema / 原子写 / migration / 并发写

### 还的债 / 未来扩展

- 如果真要做 version 级预处理 override（用户改主意），manifest 内嵌一层
  `version_overrides: {v1: {bar.png: {kind: "skip"}}}` 即可，project 级 entry 仍为
  baseline——不需要重新设计存储
- 如果真要做"用户故意不预处理"显式占位（区分"还没决定"和"决定跳过"），
  加 `kind: "skip"`——但 v1 不引入
- 跨进程写如果将来真出现（如独立的预处理 daemon），从 `threading.Lock` 升到
  `portalocker.Lock`，逻辑外壳不变

## 参考

- 触发讨论的 PR：[#69 feat(preprocess): 预处理 stage（放大器）](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/pull/69)
- 影响的代码：`studio/curation.py` `_active_left_dir`、`studio/preprocess.py` `list_processed`、
  `studio/services/upscaler.py:353` sidecar 写入、`studio/web/src/pages/project/steps/Preprocess.tsx` 双 grid
- 设计讨论：本 session 2026-05-15

---

## Addendum 1 — 裁剪 stage 引入 + manifest schema 简化（2026-05-21）

**触发**：PR 引入预处理第二阶段「裁剪」（一图可多裁，需要 `X.png → X_c0.png / X_c1.png`
fan-out 派生），原 schema 的"产物名 → 处理过程 meta"模型不够用：派生项不是直接 1:1
对应一张 download 原图，需要更明确的 origin 追溯。详见
[`docs/design/preprocess-crop-design.md`](../design/preprocess-crop-design.md)。

讨论中用户明确表态过度设计：rect 坐标 / target AR / action 分类 / VRAM 估算这类
"过程信息"一旦写盘就丢，不应该 persist。最终选了极简 schema 起手，老字段读时兼容、
写时丢弃，几个 minor 版本后逐步 deprecate。

### Schema 演进（v0 → v1，read-compat 双跑）

**v1（新写）**：entry 只剩三个字段。

```json
{
  "images": {
    "X.png":     { "origin": "X.png",  "mtime": 1731000000, "size": 1234567 },
    "Y_c0.png":  { "origin": "Y.png",  "mtime": ...,         "size": ... },
    "Y_c1.png":  { "origin": "Y.png",  "mtime": ...,         "size": ... }
  }
}
```

- `origin` = download/ 下原图名。**Multi-crop 派生**的 entry 多个共享同一 origin
- `kind` 字段**不再写入**。entry 存在即"已处理"（v0 里 kind 一直只有 `processed` 一种，
  本来就是死字段；将来真有"未来状态"再加 `state` 字段，不要复活 `kind`）
- 老的 `source / model / scale / action / target_area / src_size / dst_size /
  elapsed_seconds` 字段**全部不再写**——属于过程信息，写盘后应该丢

**v0（老读）**：保留 read-compat 至少 3 个 minor 版本。

- 缺 `origin` → 用 `source` 字段顶上（语义相同，只是更名）
- 缺 `source` → 用 entry key 兜底（= 1:1 同名场景）
- `kind` 字段如果存在且 ≠ `"processed"`，仍按 v0 规则视为未来扩展，resolver 返 None
- 老的 model/scale/... 字段读到就透传给前端（sidebar 还能显示），但下次任何 mutation
  会把 entry 重写成 v1 schema，老字段丢失

旧 sidecar `*.preprocess.json` 的迁移逻辑（`ensure_manifest`）保留不动——已经聚合成
manifest entry 的老格式仍在 read-compat 覆盖范围内。

### 新增 API

`studio/services/preprocess_manifest.py` 加三个：

| 函数 | 作用 |
|---|---|
| `entry_origin(entry, fallback)` | 从 entry 提取 origin，处理 v0/v1 schema 兼容 |
| `resolve_origin(project_dir, download_name)` | 反向 resolve：给 download/X，返回 preprocess/ 里所有 origin 匹配的派生（multi-crop 一对多）；无匹配则回退 download |
| `replace_with_crops(project_dir, source_name, outputs)` | 原子替换：把 source_name 自身 + 所有 origin 指向 source_name 的 entry 整批删，写入 N 个新 entry。供裁剪 worker 在 fan-out 时调用 |

`resolve(name)` 接口本身不变。

### 下游消费者改动

- `studio/preprocess.py:list_pending` —— 按 origin 集合比对（不再按 stem），handle
  multi-crop fan-out 的"已处理"判定（任一 entry origin == download/X.jpg → 不算 pending）
- `studio/preprocess.py:summary` —— 同 origin 集合判定
- `studio/curation.py:copy_to_train` —— 用 `resolve_origin()` fan-out 复制多裁剪产物
  到 train/{folder}，文件名按 preprocess 实际产物名（含 `_c0`/`_c1` 后缀），metadata
  按 stem 匹配
- `studio/server.py` thumbnail endpoint —— 同样 `resolve_origin()`，multi-crop 时取第一个
  preprocess 派生展示

### 文件命名约定（fan-out）

- N=1 裁剪 → 覆盖 `preprocess/{stem}.png`（同名覆盖，origin 不变）
- N>1 裁剪 → 写 `preprocess/{stem}_c0.png` / `{stem}_c1.png` ...，删原 `{stem}.png`
- `{stem}_c0.png` 再多裁 → `{stem}_c0_c0.png` / `{stem}_c0_c1.png`（链式后缀；origin
  始终指向最初的 download 原图）

### 还原（restore）语义不变

`restore(names)` 仍是删 entry + 删对应 preprocess/ 文件。multi-crop 派生的 entry 各自
独立可还原（删 `Y_c0.png` 不影响 `Y_c1.png`）。整图回退到 download 原图 = 删该图所有
origin 派生的 entry。

### Stage 不强制时序

「放大 → 裁剪 → 放大 → 裁剪」是合法链路：每个 stage 都是对 `preprocess/` 当前状态的
覆盖，没有 partial undo。原 ADR §「未来扩展」里假设的 `kind: cropped` / `kind: masked`
分类**不采用**——所有产物 entry 是同一类 (`origin + mtime + size`)，stage 信息只在
worker 运行时存在，不进 manifest。

### 跟训练 ARB 桶的对齐

裁剪聚类的 target AR 内部 snap 到训练桶（`runtime/training/dataset.py:BucketManager`
的前端 TS 镜像 `studio/web/src/lib/trainBuckets.ts`），但 UI 标签仍显示 pretty AR。
这个跟 manifest schema 无关，纯前端逻辑，仅在此点一笔。详见
[`preprocess-crop-design.md §7`](../design/preprocess-crop-design.md)。

### 防漂移

backend Python `BucketManager` 和 frontend TS `trainBuckets.ts` 必须保持算法 + 默认
参数同步。两边代码顶部互引注释 + TS 单测断言桶数严格 == 37（默认参数下 Python 输出
也是 37）。改一边漏改另一边 → CI 红。

### 老 schema 读支持的退役时间

`origin` v1 schema 在 dev 跑通后 dev → master 合并发版即可观测。Read-compat for v0：

- v1 schema 引入后 3 个 minor 版本仍读 v0 字段（兜底兼容 0.7-0.9 时期写的老 entry）
- 老用户首次访问 preprocess 数据触发 `ensure_manifest` 迁移；之后任何 mutation 把
  老 entry 重写成 v1
- 之后版本（约 1.0+）可清理 `entry_origin()` 里的 `source` 兜底分支 + 删
  `_scan_legacy_sidecars()`

不写明确的 deprecation timeline 在代码里，避免用户被"必须升级"压力。等到要清的那天
看实际使用数据决定。
