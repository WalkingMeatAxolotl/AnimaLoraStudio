# 0010 — preprocess scope 从项目级 download 下沉到 version 级 train

**状态**：Proposed
**日期**：2026-06-03
**决策者**：@WalkingMeatAxolotl
**Supersedes**：[ADR 0004 — 预处理状态用单 manifest 替代「双 bucket + per-image sidecar」](0004-preprocess-manifest.md)（含 Addendum 1）

## 背景

ADR 0004 把预处理状态固化为 **项目级单 manifest + 双 bucket resolver + 隐式 original**，工作流是：

```
download/  ──(对全集预处理)──>  preprocess/  ──(curate 选入)──>  versions/{label}/train/
   ↑ 项目级                    ↑ 项目级                          ↑ version 级
```

跨版本「预处理结果复用」由项目级 `preprocess/` 目录直接承担——这是 ADR 0004 §74-83 选项目级 scope 的核心论据。

beta 用户使用后揭示了**四个真实痛点**：

1. **前置时间浪费**：booru scrape 进来几百-几千张图，最终只用很小一部分；用户对全集每张图都要操作（放大要等 / 裁剪要逐张跳过），其中大部分图不会进 train
2. **统计指标无意义**：分辨率分布 / 宽高比分布基于 `download/` 全集，不是最终 train 集——对训练决策没有指导价值
3. **智能聚类失效**：聚类目标是统一 ARB 桶，但聚类在 download 全集做完后下游 curate 还要再筛，**桶又乱了**
4. **scope 错位**：用户心智上"我想训练的图"是 train 集合；preprocess 在 download 全集上做，跟心智不对齐

ADR 0004 项目级 scope 的两条事实假设也已变化：

- **ADR 0007** 落地后 `create_version(fork_from_version_id=...)` (`studio/services/projects/versions.py:407-498`) 通过 `_copytree("train")` 实现**整树 fork**。train/ 含已 upscale 产物在 fork 时自然跟随到子 version——**不需要项目级缓存**就能实现跨版本复用
- 用户调研：**新 version 绝大多数从上一个 version 复制 + 微调**，"v2 完全重做 preprocess"这种最坏情况几乎不发生

ADR 0004 的优化目标（跨版本复用）在新事实下由 ADR 0007 fork 机制承接，scope 错位反而成了主要成本。

## 候选方案

### A — 维持 ADR 0004 现状

- 优点：0 工时
- 缺点：四个痛点不解决；统计 / 聚类基于错误集合是**正确性 bug**而不是 UX 缺陷
- **否决**

### B — UI filter 折中（Preprocess 页加"只看 train"过滤）

保留项目级磁盘结构，只在 UI 加 filter 让用户视图聚焦到 train 集合。成本 ~2.5d，不动任何 ADR。

逐条 check 痛点解决度：

| 痛点 | filter 解了吗 |
|---|---|
| 前置时间浪费（对不会用的图也操作） | **否**——filter 是事后过滤，时间已花掉 |
| 统计指标基于 download 全集无意义 | **否**——统计源不变 |
| 聚类失效（统一 ARB 桶又被下游筛乱） | **否**——聚类源不变 |
| booru 素材只用很小一部分 | 部分（视觉上隐藏，不影响数据流） |

**0/4 真解**。filter 解的是"视觉干扰"——这是症状不是根因。**否决**。

### C — preprocess scope 下沉到 version 级 train/（**采纳**）

新工作流：

```
download/  ──(curate 选入)──>  versions/{label}/train/  ──(对 train 处理)──>  tag / train
   ↑ 项目级 source-of-truth     ↑ version 级（预处理 + 训练同位）
```

预处理在 curate **之后**，只对最终训练集生效。跨版本复用通过 fork 整树复制承接。

详 §决策。

## 决策

选**方案 C**。

### 磁盘结构

```
projects/{id}-{slug}/
  download/                       # 项目级，仍是 source-of-truth（不变）
    X.jpg, Y.jpg, ...
  preprocess/                     # 老目录，保留不动（fallback 重建源；不主动删）
    manifest.json                 # 老 v0/v1 schema，只读
    *.png                         # 老产物，只读
  versions/{label}/
    train/
      manifest.json               # 新增 per-version（v2 schema）
      X.png + X.txt               # 训练 bytes + caption（caption 不进 manifest）
      Y_c0.png + Y_c0.txt         # multi-crop 派生
      Y_c1.png + Y_c1.txt
    reg/ samples/ output/ config.yaml
```

**关键不变量**：项目级 `preprocess/` 目录**永远不主动删**（详 §不变量）。

### Manifest schema v2

`versions/{label}/train/manifest.json`：

```json
{
  "version": 2,
  "images": {
    "X.png":    { "origin": "X.jpg", "mtime": 1731000000, "size": 1234567 },
    "Y_c0.png": { "origin": "Y.jpg", "mtime": 1731000000, "size": 800000 },
    "Y_c1.png": { "origin": "Y.jpg", "mtime": 1731000000, "size": 850000 }
  }
}
```

**字段语义**：

- `version` (int): schema 版本，当前固定 `2`
- `images` (map): key = train 内**POSIX 相对路径** `"{N_label}/{image}"`（LoRA
  repeat folder 结构：`train/1_data/X.png` → entry key `"1_data/X.png"`），
  value = entry
- `entry.origin` (string): `download/` 下对应原图的**平铺文件名**（download/
  无 sub-folder 结构），用于 restore 反查
- `entry.mtime` / `entry.size` (number): 产物文件元数据，可检测外部修改

**为什么 rel path 而不是平铺**：LoRA 训练 dataset config 把 `train/{N_label}/`
当 repeat folder（N = 重复次数）；同一张图可在多个 folder 出现（罕见但合法）。
平铺 entry key 无法表达跨 folder 唯一性 + 同名歧义。POSIX 形式 `/` 跨平台一致。

`train/` 根目录直接放的图**被忽略**（LoRA 训练只读 sub-folder 内）；validator
`_validate_rel_name` 强制 `folder/image` 两段格式，防 path traversal。

**状态从字段差异隐含推断**（承袭 ADR 0004 极简原则）：

- entry key rel path 末段 `"X.png"` + `origin="X.jpg"` → 处理过（扩展名变 =
  upscale / crop / 转码之一）
- 末段 `"X.jpg"` + `origin="X.jpg"` + mtime/size 匹配 → 原样未处理
- 末段 `"X.jpg"` + `origin="X.jpg"` + mtime/size 不匹配 → 外部修改

**不存的字段**（明确否定项，跟 ADR 0004 Addendum 1 一致）：

- `kind`（老字段：`processed` / `cropped` / `masked`）
- `source`（老字段名，由 `origin` 取代）
- `model / scale / action / target_area / src_size / dst_size / elapsed_seconds`（过程信息，写盘后应丢）
- `state`（新枚举形如 `upscale-done`）—— 从字段差异推断
- `caption_mtime / tags / caption`（caption 是 tagging stage owner，独立 lifecycle）
- `kind: duplicate_removed`（人工去重审核状态——本 ADR 下沉去重 scope 到 train 集，每个 version 独立审核，状态不跨 version 共享，详 §去重 scope）

### Fallback 重建机制：`ensure_train_manifest`

老项目里 train/ 已经有 preprocess 产物（curate 阶段已复制），唯一丢失的是 train ↔ download 的 origin 关系。`ensure_train_manifest(project_dir, version_label)` 在所有 manifest read/write 入口防御性调用，幂等。

**重建规则**（按优先级）：

1. 目标 `versions/{label}/train/manifest.json` 已存在 → 直接返回（O(1) stat，热路径 0 开销）
2. 不存在 + 老 `projects/{id}/preprocess/manifest.json` 存在 → 递归扫
   `train/` 一级 sub-folder 收集图片相对路径集合（仅图像后缀；根目录直接放
   的图忽略）；按文件名（rel path 末段）匹配老平铺 entry name → 重建 v2
   schema，entry key 用 rel path
3. 老 manifest 不存在 / 损坏（非 dict / 非法 JSON） → 写空 v2 manifest（`{"version": 2, "images": {}}`）
4. 老 entry 标 `kind: duplicate_removed` → **跳过**（人工去重审核状态不跨模型迁移，新模型下用户在 train scope 重新去重）
5. 跨 sub-folder 同名图（罕见但合法，如 `1_data/X.png` + `5_extra/X.png`） →
   各自独立 entry，都继承同一老 origin

并发：`threading.Lock` 串行 + 原子 `tmp+rename` 落盘 + 双检查防竞态。

**为什么 fallback 而不是显式迁移脚本**：用户 train/ 物理 bytes 已经是 preprocess 复制过来的产物，**删 project 级 `preprocess/` 不影响 train/ 内容**。唯一损失是 origin 反查，30 行代码 lazy 重建即可。比显式脚本 + UI 弹窗省 1 人日 + 零用户感知 + 零失败回滚成本（重建失败下次调用会重试）。

### Resolver 命运

ADR 0004 的 `resolve(name) → Path` resolver 是为消除"双 bucket fallback（download/ vs preprocess/）"而设计的中心抽象。**新模型下 train/ 是 self-contained**——thumbnail / curation / tagging / training materialize 全部直接读 `train/{name}`，没有歧义。

| 函数 | 命运 |
|---|---|
| `resolve(project_dir, name)` | **删**（双 bucket 概念消失） |
| `resolve_origin(project_dir, download_name)` | **删**（反向 resolve 无业务调用方） |
| `list_pending()` | **删**（"未处理 / 已处理"二元概念消失） |
| `ensure_manifest()` + `_scan_legacy_sidecars()` | **删**（不再迁老 sidecar；老 manifest 由 `ensure_train_manifest` 一次性读） |
| `entry_origin()` | **保留**（fallback 读老 entry / restore 反查都要用） |
| `add_processed / restore / mark_duplicate_removed / clear_all / replace_with_crops` | **保留并加 `version_label` 参数** |

### Restore 语义

`restore(project_dir, version_label, name)`：

1. `ensure_train_manifest` 前置调用
2. 查 `manifest.images[name].origin`
3. 从 `download/{origin}` 复制覆盖 `train/{name}`
4. 更新 manifest entry 的 mtime/size

**`download/{origin}` 缺失时**：明确失败 + UI 显式提示。具体图列出 + 提供三选项：

- **拖入替换** — 文件选择器选本地图覆盖 `download/{origin}`，重试 restore
- **保留处理后版本** — 忽略失败，train/{name} 不变
- **从 train/ 移除** — 删 train/{name} + 删 manifest entry（破坏性，二次确认）

承袭 ADR 0004 §215-219 "外部删 download/ 不主动 reconcile"原则：**没有隐藏备份字节**（不做 per-version `.backup/`，违反 "download 唯一备份" 不变量）。复原失败是有意的接受现状，UI 显式失败优于静默假复原。

### 跨版本复用：fork 整树复制

通过 ADR 0007 现有 fork 机制承接：

- `create_version(fork_from_version_id=...)` 调用 `_copytree("train")` 递归复制 train 子树
- `train/manifest.json` **自动随复制带过去**（递归复制目标，0 代码改动）
- fork 后调用一次 `ensure_train_manifest(new_label)` 兜底（万一源 manifest 损坏可重建）
- v2 起手 train/ 跟 v1 完全一致 → preprocess phase 自动跳过（产物全继承）

代价：fork 时复制几 GB train/（含 upscale 产物）。**用户接受**——"v3 通常只改训练参数不改 train" 是最常见 case，磁盘代价摊销低；想完全重做 preprocess 的少数场景用户显式触发"重做 stage"按钮。

### Phase 状态机改动（ADR 0007 amendment 配套）

`VersionPhase.ORDER` 加 `preprocessing`：

```python
# 改前（ADR 0007 §132）：
ORDER = (curating, tagging, editing, regularizing, ready)         # 5 个
SKIPPABLE = {regularizing}

# 改后：
ORDER = (curating, preprocessing, tagging, editing, regularizing, ready)   # 6 个
SKIPPABLE = {preprocessing, regularizing}
```

`check_preprocessing(conn, version_id)` 跟 `check_regularizing` 同 pattern：

- 校验 = 无 preprocess job 处于 pending / running
- 不强校验"是否处理过任何图"——可跳过，跟 `regularizing` 一致心智
- UI 文案在 phase 名后加"（可选）"，跟 `regularizing` 现有 `nav.reg = "正则集（可选）"` 完全同 pattern

### Migration 两层机制

**Layer 1 — `_v11_preprocessing_phase` DB migration**（隐式 add-only，跟 `_v8_version_status_phase.py` 同 pattern）：

跟 lifecycle PR-5 (`_v9 destructive`) 之后跑。回填规则：

| 现存 phase | train/ 状态 | 新 phase |
|---|---|---|
| `curating` | 空 | `curating`（保持） |
| `curating` | 非空 | **`preprocessing`** |
| 其他（tagging / editing / regularizing / ready） | 任意 | 保持不变 |

依据：用户原话"preprocessing 的图片已经复制到现有 train"——train/ 非空意味着 curating 已实质完成。零用户感知。

**Layer 2 — `ensure_train_manifest` 隐式 fallback**（见 §Fallback 重建机制）：

DB migration 处理 phase 字段，fallback 处理 manifest 文件。两层解耦。

### 去重 / blur / 聚类 scope

全部跟随下沉到 **train 集合**：

- `studio/services/preprocess/duplicates.py:_resolve_download_sources` → `_resolve_train_sources(version_label)`，scope 改 `versions/{label}/train/`
- **不拆模块**（R1/R2 设计阶段曾考虑"去重上提到 ingestion 阶段"作为前置硬条件，最终否决）——下沉到 train 后去重就是为了清理 train 集本身，跟"项目级清理素材池"是不同心智，拆模块反而增加复杂度
- 人工去重审核状态在新模型下**每个 version 独立**（fork 时跟随 train manifest 复制，不跨 version 共享）

### 不动的部分

- `download/` 仍是项目级 source-of-truth；curate 阶段从 download 复制进 train 的语义不变
- `services/preprocess/` 模块（worker / upscale / crop / blur / dedup 核心逻辑）保留——本 ADR 改的是**状态存储位置**，不是计算职责
- ADR 0007 §70 "数据集归属 = 项目级"决议**不撕**——train 集合 source-of-truth 仍是项目级 download 池，本 ADR 只把**预处理产物**与 train 同位

## 不变量（未来修改时必读）

以下约束是本 ADR 数据模型的硬约束。后续相关方向重构时**必须保留**或显式撕掉并写新 ADR：

1. **`download/` 是唯一持久原图备份**——除非用户外部删，否则永远存在；不发明任何 `.backup/` / 影子目录复制 download bytes
2. **老 `projects/{id}/preprocess/` 目录永远不主动删**——作 fallback 重建源 + 老数据备份；用户自决何时清理；只有 next next minor release 才考虑加用户引导清理
3. **Manifest 只存"现状反查关系"，不存"过程信息"**——schema 三字段 `{origin, mtime, size}` 之外都属于过程信息（kind / model / scale / action / state / ops 链 / rect / target_area / 等等）。写盘后过程信息应丢
4. **状态从字段差异隐含推断**——不显式存"是否 upscale 过 / 是否 crop 过"等 bool；增加状态枚举字段时必须先论证为什么差异推断不够
5. **train/ 是 self-contained**——所有下游消费者（thumbnail / curation / tagging / training materialize）直接读 `train/{name}`，**不**走双 bucket fallback。manifest 用于反查 origin（restore / 派生关系），不用于"指路径"
6. **caption 跟 manifest 解耦**——caption (`.txt`) 是 tagging stage owner，不进 manifest；preprocess 改图不动 caption
7. **跨版本复用走 fork 整树复制**，不走项目级缓存——这是 ADR 0007 现有机制承接，不重复发明
8. **复原失败显式可见**——`download/{origin}` 缺失时 restore **必须明确失败**给用户三选项，**禁止**静默成功 / 假复原
9. **DB migration 跟 manifest fallback 解耦**——`_v11` 改 phase 字段，`ensure_train_manifest` 重建 manifest 文件；两套机制独立、各自幂等
10. **进程内 `threading.Lock` 即可**——服务端单进程，没跨进程写者。未来真出现跨进程写（独立 preprocess daemon），升级到 `portalocker.Lock`，逻辑外壳不变

## 理由

**为什么 ADR 0004 项目级 scope 被推翻**：

不是决策当时论证错了，是两个**事实假设**变了：

1. ADR 0007 落地的 fork 机制让"跨版本复用"无需项目级缓存就能实现——ADR 0004 §74-83 的核心论据由 fork 承接
2. 用户实际工作流是「v2 从 v1 复制 + 微调」远多于「v2 重做」——重做 upscale 几十分钟这种最坏 case 几乎不发生

ADR 0004 §227 "跨版本复用"论据在新事实下**不构成项目级 scope 的支撑**，反而 scope 错位（preprocess 跨 train 内外）成了主要成本。

**为什么不选方案 B（UI filter）**：

filter 只换视图——前置时间已经花掉 / 统计源不变 / 聚类源不变。**0/4 真解**。把 filter 当方案 = 把视觉症状当根因。

**为什么 fallback 而不是显式迁移**：

train/ 物理 bytes 已经是 preprocess 复制过来的产物（curate 阶段早已发生），唯一丢失的是 origin 反查。30 行代码 lazy 重建就够，零用户感知，比显式脚本 + UI 弹窗省 ~1d 工时 + 零失败回滚成本。

**为什么 schema 极简化**：

承袭 ADR 0004 Addendum 1 原则：状态从字段差异隐含推断，过程信息写盘后应丢。这套原则在 0004 已经过两轮迭代验证，本 ADR 继承不引入新规则。

**为什么加 phase 而不是只动 Sidebar UI**：

加 phase = 让状态机更严格符合 ADR 0007 设计原则；前提是只有隐式 DB migration 无用户感知——这条满足（`_v8_version_status_phase.py` 已经证明 add-only migration 是干净 pattern）。`check_preprocessing` 跟 `check_regularizing` 同 pattern，零设计代价。

**为什么不撕 ADR 0007 §70 "数据集归属 = 项目级"**：

§70 否决的是"每 version 独立的 train 集（数据集本身 version 级）"——本 ADR 下 train 集合的 source-of-truth 仍是项目级 `download/` 池（curating phase 从 download 复制进 train）；本 ADR 下沉的是**预处理产物**，不是数据集归属。ADR 0007 加 amendment 精确化措辞即可。

## 后果

### 好处

- 四个用户痛点 4/4 真解：前置时间 / 统计 / 聚类 / scope 错位全部修复
- 跟用户心智对齐：preprocess 是"对我要训练的图做精细处理"
- ADR 0007 fork 机制自然承接跨版本复用，不重复发明缓存
- manifest 模块大幅瘦身（25 函数 → 一半瘦身 + 极简 schema）
- 老项目零感知升级（隐式 fallback）
- 去重 / 聚类指标 finally 对训练有意义

### 代价 / 新增约束

- `_copytree("train")` fork 时复制几 GB train + manifest 是真实磁盘代价。用户已接受（"v3 通常只改参数"频次最高）
- `restore` 在 download 缺失时**真的失败**——有意的，跟 ADR 0004 原则一致，UI 显式失败优于静默假复原
- 老 `projects/{id}/preprocess/` 目录长期占磁盘——next minor release notes 提醒可手动清，不强制
- 11 个 preprocess API endpoint URL 从 pid → (pid, vid) **breaking change**（beta 心智 + 前后端同 PR 切换，不做 redirect 兼容期）
- `_v11_preprocessing_phase` migration 是 ADR 0007 `_v9 destructive` 之后第二次动 phase 列——必须严格在 _v9 之后跑

### 还的债 / 未来扩展

- 老 `preprocess/` 目录的清理：next minor release notes 提醒用户可删；几个 release 后可考虑在 `ensure_train_manifest` 里加"老 manifest 不存在则直接返空"分支（一旦绝大多数老项目已 lazy 重建过，老 fallback 路径就是死代码可清）
- 如果未来出现"v1 复制到 v2 后想重做某 stage"的高频需求：当前是用户进 v2 显式点"重做 upscale"按钮（手动），将来如果需要可加 fork 时 dialog 询问。但不预投资
- 如果未来真要支持人工去重审核状态跨 version 共享：本 ADR 显式跳过老 manifest 的 `duplicate_removed` entry。重新引入需要新 schema 字段 + ADR

## 参考

- 被取代的 ADR：[ADR 0004](0004-preprocess-manifest.md)（含 Addendum 1）
- 牵连的 ADR：[ADR 0007](0007-project-version-lifecycle-refactor.md)（加 Addendum 1：`preprocessing` phase）
- 不动的 ADR：[ADR 0008](0008-studio-restructure-0.11.0.md)（模块边界）/ [ADR 0009](0009-logging-error-system.md)（日志体系）
- 实施细节（file:line 改动清单 + 4 PR 切片 + 测试 case + 风险清单）：[`docs/design/preprocess-train-scope-plan.md`](../design/preprocess-train-scope-plan.md)
- 关键代码文件：
  - `studio/services/preprocess/manifest.py`（schema + `ensure_train_manifest` + `entry_origin` / `restore`）
  - `studio/services/projects/versions.py:41-58`（`VersionPhase.ORDER` / `SKIPPABLE`）
  - `studio/services/projects/versions.py:407-498`（`create_version` fork 流程）
  - `studio/services/projects/phase.py`（`check_preprocessing`）
  - `studio/infrastructure/migrations/_v11_preprocessing_phase.py`（DB migration）
