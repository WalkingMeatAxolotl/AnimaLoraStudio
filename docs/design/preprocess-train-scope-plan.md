# Preprocess scope 重定位实施计划

**状态**：Plan（实施前定稿）
**日期**：2026-06-03
**作者**：@WalkingMeatAxolotl 主推 + Claude (Opus 4.7) 主审
**配套 ADR**：[ADR 0010](../adr/0010-preprocess-train-scope.md) supersedes ADR 0004 + Addendum 1
（注：ADR 0009 已被 "统一日志 + 错误体系" 占用，本提案是 0010）

---

## 0. 一句话目标

把 preprocess 阶段的 scope 从**项目级 `download/` 全集**收窄到 **per-version `versions/{label}/train/` 已选集合**，统计 / 聚类 / 去重一并跟随；解决用户痛点："对最终不会用的图也做预处理是浪费时间 + 统计/聚类基于 download 全集对训练无意义"。

---

## 1. 设计决策一览（每个讨论点的最终结果）

每条都附"来源"标注出处（用户原话 / R3 共识 / 主审推荐）。

| # | 讨论点 | 决议 | 来源 |
|---|---|---|---|
| D1 | 数据流方向 | **`download → curate（进 train/）→ preprocess（对 train/ 处理）→ tag → train`**，预处理移到筛选**之后** | 用户 2026-06-03 投票 |
| D2 | preprocess 产物落盘位置 | **`versions/{label}/train/` 内**，跟训练 bytes / caption 同位 | R3 三方共识 |
| D3 | 项目级 `preprocess/` 目录 | **保留不动，永远不主动删** — 作为 fallback 重建源 + 老数据备份 | 用户 2026-06-03 二轮反馈 |
| D4 | manifest 文件名 | **`train/manifest.json`**（不隐藏，不带前缀点） | 用户 2026-06-03 二轮反馈："不应该隐藏" |
| D5 | manifest schema | **v2 极简：`{origin, mtime, size}`**，不存 ops / rect / action / scale / model 等过程信息 | MEMORY `feedback_preprocess_data_model_simple` + R3 三方共识 |
| D6 | 跨版本复用机制 | **靠 fork 整树复制**（`versions.py:_copytree("train")` 已支持，新增 manifest.json 自动跟随递归复制），**不**做项目级 cache | 用户："新版本都不会从空 template 创建，而是从上一个 version 复制" |
| D7 | 复原 (`restore`) 语义 | **从 `download/{entry.origin}` 复制覆盖回 `train/{name}`**；download 缺失则失败 + UI 显式提示 | 用户："download 作为唯一真相，删掉了，就没办法复原" |
| D8 | 复原 fallback 设计 | **不做** per-version `.backup/` 备份字节（违反 "download 唯一备份" 原则） | 用户接受现状限制 |
| D9 | 去重 / blur / 模糊检测 scope | **全部下沉到 train 集合**，不拆模块（`duplicates.py:_resolve_download_sources` → `_resolve_train_sources`） | 用户："统计基于 download 全集对训练无意义，聚类失效" |
| D10 | 统计指标 (分辨率分布 / 宽高比) | **scope 改 train/**，移到 Preprocess 页 Overview 子页；Download / Curation 页删除这些指标 | 用户原话 |
| D11 | 智能聚类 / ARB 桶 | **scope 改 train/**，下游不再筛选所以桶稳定 | 用户："聚类是为了统一 arb 桶，但是统一后再筛选导致 arb 桶再次乱掉" |
| D12 | 老项目兼容策略 | **隐式 lazy fallback**：`ensure_train_manifest()` 函数，train/manifest.json 缺失时按老 `preprocess/manifest.json` 反查 train/ 实际文件名隐式重建；**零用户感知** | 用户 2026-06-03 二轮反馈（详 §3.2） |
| D13 | 迁移触发点 | **所有 manifest read 入口防御性调用 `ensure_train_manifest`**（幂等代价极低）+ `create_version(fork_from=...)` 时调用一次 | 用户："所有 manifest" |
| D14 | 迁移目标 version | **不做迁移**（train/ 已经是处理后的图，删 preprocess/ 不影响 train） | 用户："不需要做迁移目标 version 了" |
| D15 | ADR 0007 加 `preprocessing` phase | **加**（`VersionPhase.ORDER` 加新值），前提是只有隐式 DB migration 无用户感知 | 用户："加上状态机更严格符合我们的设计" |
| D16 | preprocessing phase 可跳过 | **可跳过**（`SKIPPABLE` 加 `PREPROCESSING`），跟 `regularizing` 同 pattern | 用户："可跳过，和正则一样" |
| D17 | UI 文案 | Sidebar 标签 `"预处理（可选）"` / `"Preprocess (optional)"` — 跟现有 `nav.reg = "正则集（可选）"` 完全同 pattern | 用户："名字后面可以加（可选）" + 现状 `i18n/locales/zh.json:13` |
| D18 | Sidebar Stepper 顺序 | `download → curate → preprocess → tag → edit → reg → train`（preprocess 从原 `idx=②`（project scope）改成 version scope，移到 curate 之后） | R3 impl §2.4 + 状态机入 phase 后必然 |
| D19 | _v11 DB migration 回填 | phase=curating + train/ **非空** → 推进到 preprocessing；phase=curating + train/ 空 → 保持 curating；其他 phase 不动 | 用户原话："preprocessing 的图片已经复制到现有 train" → train/ 非空意味着已过 curating |
| D20 | API endpoint URL 改造 | 11 个 preprocess endpoint pid → (pid, vid) | R3 impl §1.3 |
| D21 | 老 API URL 兼容期 | **不做兼容 redirect**（前端 PR 同步切换；beta 心智） | beta + 用户接受 breaking |
| D22 | 老 sidecar `*.preprocess.json` | **彻底不支持**（`manifest.py:_scan_legacy_sidecars` + `ensure_manifest` 删除） | beta，不留 dead read |
| D23 | dead-code 兼容层 | **不保留**（违反 `tagger_config_two_surfaces` 教训：两 surface 不一致是测试地狱） | R3 ux §5.3 + impl §3.1 |
| D24 | feature flag | **不加** | beta + 一次性 + 老文件保留备份足够 |
| D25 | ADR 修订路径 | 新建 **ADR 0010** supersede 0004；ADR 0004 顶部加 `Status: Superseded by ADR 0010`；ADR 0007 加 amendment 加 PREPROCESSING phase；ADR 0008 / 0009 不动 | R3 arch §5（编号修正：0009 已被日志错误体系占用） |
| D26 | multi-crop fan-out | **保留**（派生图 `Y_c0.png / Y_c1.png` 仍在 train/ 内 + origin 共享，origin 反查仍成立） | R3 arch §3.2 |
| D27 | caption (.txt) 落盘 | manifest **不记 caption**（caption 是 tagging stage owner，跟 preprocess origin 是两个 lifecycle，不混进 manifest） | R3 arch §6.3 |
| D28 | `copy_to_train` 命运 | **大幅简化**：删 "preprocess 派生 vs download 原图" 双分支（`curation.py:210-275`），变成纯 download → train 复制 | R3 impl §2.1 |
| D29 | PR 切片 | **4 个 PR**：PR-1 ADR 0010 + fallback（0.5d）/ PR-2 后端 manifest+core+worker（3.0d）/ PR-3 _v11 migration + phase + API（1.5d）/ PR-4 前端 + Sidebar（2.5d） | R3 impl §7 |
| D30 | PR 时序依赖 | PR-1 独立可立刻开；PR-2/3/4 需要 lifecycle PR 链 + `feat/preprocess-multitool` 合 dev 后开 | 用户："新的 pr 我都看过，功能上和当前没有冲突" |

---

## 2. 数据模型

### 2.1 磁盘结构

```
projects/{id}-{slug}/
  project.json
  download/                        # 项目级 source-of-truth（不变）
    X.jpg, Y.jpg, ...
  preprocess/                      # 老目录，保留不删（D3）
    manifest.json                  # 老 v1 schema，只读，作 fallback 重建源
    *.png                          # 老产物，只读
  versions/{label}/
    version.json                   # ADR 0007 现状不变
    train/
      manifest.json                # 新增 per-version（D4）
      X.png                        # train 图（可能是 upscale 后产物，也可能是原图）
      X.txt                        # caption（不进 manifest，D27）
      Y_c0.png + Y_c0.txt          # multi-crop 派生
      Y_c1.png + Y_c1.txt
    reg/
    samples/
    output/
```

### 2.2 Manifest v2 schema

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

- `version` (int): schema 版本，当前固定 `2`。未来再变 schema 用这个判断
- `images` (map): key = train 内**实际文件名**（含扩展名），value = entry
- `entry.origin` (string): `download/` 下对应原图的**文件名**（含扩展名）。用于 restore 反查
- `entry.mtime` (number): 产物文件 mtime（秒级 Unix timestamp）。差异可推断"用户在外部改过"
- `entry.size` (number): 产物文件大小（字节）

**状态从字段差异隐含推断**：

- `name="X.png"` + `origin="X.jpg"` → 处理过（扩展名变，必经过转码 / upscale / crop 之一）
- `name="X.jpg"` + `origin="X.jpg"` 且 `mtime/size` 匹配 train/X.jpg 实际文件 → 原样未处理
- `name="X.jpg"` + `origin="X.jpg"` 但 `mtime/size` 不匹配 → 外部修改过
- 不存"是否处理"显式 bool。这是 ADR 0004 Addendum 1 的"不存过程信息"原则继承

### 2.3 不存的字段（明确否定项）

跟 ADR 0004 Addendum 1 一致。**不存**：

- `kind` (老 v0 字段：`processed` / `cropped` / `masked`) — 已被否决
- `source` (老 v0 字段名) — 由 `origin` 取代
- `model` / `scale` / `action` / `target_area` / `src_size` / `dst_size` / `elapsed_seconds` — 过程信息，写盘后应丢
- `state` (新枚举如 `"upscale-done"`) — 从字段差异推断，不显式
- `caption_mtime` / `tags` — caption 是 tagging stage owner（D27）

---

## 3. 数据流与生命周期

### 3.1 完整流程图

```
[创建项目]
   │
   ▼
1. Download 页 (project scope)
   │   导入图到 download/  (booru scrape / 拖入 / 上传)
   │   状态显示：原图总数 / 总磁盘
   │   不显示：分辨率分布 / 宽高比分布 / ARB 桶（这些挪到 Preprocess）
   ▼
2. 创建 / 选 version
   │
   ▼
3. ① Curate 页 (version scope, phase=curating)
   │   从 download/ 池子挑图，"入选" = 复制 download/X.jpg → train/X.jpg
   │   ※ copy_to_train 简化：纯 1:1 文件复制 + 产生 manifest entry { origin: "X.jpg" }
   │   完成校验：train/ ≥ 1 张图（不变）
   │   推进按钮 → phase: curating → preprocessing
   ▼
4. ② Preprocess 页 (version scope, phase=preprocessing, **可选**)
   │   ┌─ Overview 子页：基于 train/ 计算的统计 / ARB 桶预览 (D10/D11)
   │   ├─ Upscale 子页：批量 RealESRGAN 替代训练默认放大
   │   │   ※ worker 读 train/X.jpg → 写 train/X.png + 更新 manifest entry
   │   ├─ Crop 子页：multi-crop fan-out (X.png → X_c0.png / X_c1.png)
   │   │   ※ worker 删原 X.png 的 manifest entry + 写 X_c0.png / X_c1.png 共享 origin
   │   └─ Dedup/Blur 子页：scope train/
   │   "下一步" 按钮永远可点（D16 可跳过）
   │   skip 推进：phase: preprocessing → tagging（无 concurrent preprocess job 即可）
   ▼
5. ③ Tag 页 (version scope, phase=tagging)
   │   给 train/ 打标 (现状不变)
   ▼
6. ④ Edit 页 / ⑤ Reg / ⑥ Train (现状不变)
```

### 3.2 `ensure_train_manifest` 隐式 fallback 重建（D12 / D13）

**目标函数**（新增）：`studio/services/preprocess/manifest.py:ensure_train_manifest()`

**触发**：所有需要读 train manifest 的入口防御性调用（D13），幂等。具体包括：

| 调用点 | 文件:行 | 触发时机 |
|---|---|---|
| `load_manifest(project_dir, version_label)` | `manifest.py` 入口 | 任何 read 前置 |
| `restore(project_dir, version_label, names)` | `manifest.py` | 复原前 |
| `add_processed / mark_duplicate_removed / replace_with_crops` | `manifest.py` | 任何 write 前置 |
| 缩略图 endpoint | `api/routers/projects/curation.py:117-186` | 列图前 |
| `list_train_images` | `services/preprocess/core.py` | 列图前 |
| `create_version(fork_from_version_id=...)` | `services/projects/versions.py:407-498` | fork 时主动调一次（保证 v2/v3 起手就有 manifest，避免 lazy 不一致） |

**重建规则**（伪代码）：

```python
def ensure_train_manifest(project_dir: Path, version_label: str) -> Path:
    """如果 versions/{label}/train/manifest.json 缺失，按老 project 级 preprocess
    manifest 隐式重建一份。零用户感知，幂等。

    Rules (D12):
      1. 目标已存在 → 直接返回路径（O(1) stat 检查）
      2. 目标不存在 + 老 preprocess/manifest.json 不存在 → 写空 manifest 返回
      3. 目标不存在 + 老 manifest 存在 → 按 train/ 实际文件名匹配老 entry origin 重建
    """
    train_dir = project_dir / "versions" / version_label / "train"
    target = train_dir / "manifest.json"
    if target.exists():
        return target

    train_dir.mkdir(parents=True, exist_ok=True)
    new_data = {"version": 2, "images": {}}

    legacy_manifest = project_dir / "preprocess" / "manifest.json"
    if legacy_manifest.exists():
        try:
            legacy = json.loads(legacy_manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            legacy = {"images": {}}

        train_files = {
            f.name for f in train_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        }

        for name, entry in legacy.get("images", {}).items():
            # 只为 train/ 里实际存在的图重建 entry
            if name not in train_files:
                continue
            new_data["images"][name] = {
                "origin": entry.get("origin") or entry.get("source") or name,
                "mtime": entry.get("mtime", 0),
                "size": entry.get("size", 0),
            }

    # 原子写入：tmp + rename
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(new_data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)
    return target
```

**正确性论证**：

- **train/ 内有图但老 manifest 漏记的图**（用户手动拖入 train/）→ 不写 entry。后续这些图被认为 `origin = name`（restore 时从 download/{name} 找；找不到就失败）— **这是有意的兜底**，跟 D7 一致
- **老 manifest 里有但 train/ 没的图**（用户在 curation 时没勾入）→ 不写 entry。正确，新模型下这张图就不该在 train scope
- **multi-crop 派生**（`Y_c0.png` / `Y_c1.png` 共享 origin `Y.jpg`）→ 老 manifest 里两个 entry，按规则各自检查 train/ 是否存在并写入

### 3.3 `restore(name)` 语义（D7 / D8）

```python
def restore(project_dir, version_label, name):
    """复原 train/{name} 为 download/{entry.origin} 原图副本。"""
    ensure_train_manifest(project_dir, version_label)
    manifest = load_manifest(project_dir, version_label)
    entry = manifest["images"].get(name)
    if entry is None:
        raise RestoreError(f"No manifest entry for {name}")
    origin = entry["origin"]
    src = project_dir / "download" / origin
    if not src.exists():
        raise RestoreError(
            f"Original download/{origin} not found — cannot restore. "
            "Likely deleted from disk."
        )
    dst = project_dir / "versions" / version_label / "train" / name
    shutil.copy2(src, dst)
    # 更新 manifest entry：origin 不变，但 mtime/size 同步成 download 文件的
    manifest["images"][name] = {
        "origin": origin,
        "mtime": int(src.stat().st_mtime),
        "size": src.stat().st_size,
    }
    save_manifest(project_dir, version_label, manifest)
```

**失败 UX**（D8）：前端收到 `RestoreError` → toast "X 张图无法恢复：原图已从 download/ 删除"，列出具体 origin 文件名 + 三选项 [拖入替换 / 保留处理后版本 / 从 train/ 移除]。

### 3.4 Fork version 行为（D6）

现行 `create_version(fork_from_version_id=...)` (`studio/services/projects/versions.py:407-498`) 调用 `_copytree("train")` 递归复制 train 子树。

**新方案**：

- `train/manifest.json` 因为是递归复制目标，**自动跟随**（0 代码改动）
- 调用 `ensure_train_manifest(project_dir, new_version_label)`（防御性，万一源 manifest 损坏可重建）
- v2 起手 train/ 跟 v1 完全一致，preprocess 产物全继承 → preprocessing phase 自动跳过（UI Sidebar 显示已完成）

### 3.5 Phase advance 行为

按 ADR 0007 §11.5-A pattern：

- 当前 phase = curating → 校验 train/ ≥ 1 → 推进到 preprocessing
- 当前 phase = preprocessing → 校验无 preprocess job pending/running → skip 推进到 tagging（不强求处理过任何图）
- 当前 phase = tagging → 校验 caption 100% → 推进到 editing
- 后续不变

---

## 4. Phase 状态机变更

### 4.1 `VersionPhase` 枚举（`studio/services/projects/versions.py:41-58`）

**改前**：

```python
class VersionPhase:
    CURATING = "curating"
    TAGGING = "tagging"
    EDITING = "editing"
    REGULARIZING = "regularizing"
    READY = "ready"
    ORDER = (CURATING, TAGGING, EDITING, REGULARIZING, READY)
    VALUES = frozenset(ORDER)
    SKIPPABLE = frozenset({REGULARIZING})
```

**改后**：

```python
class VersionPhase:
    CURATING = "curating"
    PREPROCESSING = "preprocessing"    # ← 新增
    TAGGING = "tagging"
    EDITING = "editing"
    REGULARIZING = "regularizing"
    READY = "ready"
    ORDER = (CURATING, PREPROCESSING, TAGGING, EDITING, REGULARIZING, READY)  # 6 个
    VALUES = frozenset(ORDER)
    SKIPPABLE = frozenset({PREPROCESSING, REGULARIZING})  # 加 PREPROCESSING
```

### 4.2 `check_preprocessing`（`studio/services/projects/phase.py`）

新增函数，跟 `check_regularizing` 同 pattern：

```python
def check_preprocessing(conn: sqlite3.Connection, version_id: int) -> CheckResult:
    """无 preprocess job 处于 pending/running（D16；可跳过，不强校验完成度）。"""
    row = conn.execute(
        "SELECT COUNT(*) FROM project_jobs "
        "WHERE version_id = ? AND kind = 'preprocess' "
        "  AND status IN ('pending', 'running')",
        (version_id,),
    ).fetchone()
    if int(row[0]) > 0:
        return CheckResult(False, "预处理任务进行中，请等待完成")
    return CheckResult(True)
```

`check_phase` dispatcher 加分支 `if phase == P.PREPROCESSING: return check_preprocessing(conn, version_id)`。

### 4.3 `_v11_preprocessing_phase` migration

文件：`studio/infrastructure/migrations/_v11_preprocessing_phase.py`

**回填规则**（D19）：

| 现存 phase | train/ 状态 | 新 phase |
|---|---|---|
| `curating` | 空 | `curating`（保持） |
| `curating` | 非空 | **`preprocessing`**（用户原话：train/ 已是处理后） |
| `tagging` | * | `tagging`（保持） |
| `editing` | * | `editing`（保持） |
| `regularizing` | * | `regularizing`（保持） |
| `ready` | * | `ready`（保持） |

**伪代码**：

```python
"""v10 → v11: ADR-0009 加 preprocessing phase。

VersionPhase 从 5 → 6 个，新增 preprocessing 介于 curating / tagging 之间。

回填策略（all silent，跟 _v8 同样 add-only 模式）：
- phase=curating + train/ 非空 → 推进到 preprocessing
- phase=curating + train/ 空 → 保持 curating
- 其他 phase 不动
"""
from __future__ import annotations
import sqlite3
from pathlib import Path


def migrate(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT v.id, v.label, p.id AS project_id, p.dir_path "
        "FROM versions v JOIN projects p ON v.project_id = p.id "
        "WHERE v.phase = 'curating'"
    ).fetchall()
    for vid, label, _, project_dir in rows:
        train_dir = Path(project_dir) / "versions" / label / "train"
        if not train_dir.exists():
            continue
        has_files = any(
            f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            for f in train_dir.iterdir()
        )
        if has_files:
            conn.execute(
                "UPDATE versions SET phase = 'preprocessing' WHERE id = ?",
                (vid,),
            )
    conn.commit()
```

**测试 case**（`tests/test_migration_v11.py`）：

- v=curating + train/ 空 → 不动
- v=curating + train/ 有 .png → preprocessing
- v=curating + train/ 有 .txt 但无图 → 不动
- v=tagging + 任意 train/ → tagging（保持）
- v=ready + 任意 train/ → ready（保持）

### 4.4 前端 phase enum 同步

`studio/web/src/api/client.ts` 里 `PHASE_ORDER` / `PHASE_SKIPPABLE` / `VersionPhase` 类型：

```typescript
export type VersionPhase =
  | 'curating'
  | 'preprocessing'   // 新增
  | 'tagging'
  | 'editing'
  | 'regularizing'
  | 'ready'

export const PHASE_ORDER: VersionPhase[] = [
  'curating',
  'preprocessing',    // 新增
  'tagging',
  'editing',
  'regularizing',
  'ready',
]

export const PHASE_SKIPPABLE: VersionPhase[] = ['preprocessing', 'regularizing']  // 加 preprocessing
```

---

## 5. 模块改造清单

### 5.1 后端：删除

| 文件:行 | 函数 / 段落 | 删除理由 |
|---|---|---|
| `manifest.py:132-152` | `resolve()` 双 bucket fallback | 新模型 train/ self-contained，不需要 fallback |
| `manifest.py:155-180` | `resolve_origin()` (反查 download → 派生) | train/ 不是派生源，反查无意义 |
| `manifest.py:443-460` | `ensure_manifest()` 老 sidecar 迁移 + `_scan_legacy_sidecars` | D22 不支持老 sidecar |
| `core.py:106-151` | `list_pending()` | "未处理 / 已处理" 二元概念消失，只有 train/ 物理文件 |
| `curation.py:109-163` | `list_download()` 的 multi-crop fan-out 行展开 | download 是只读快照，不展示派生 |

### 5.2 后端：改造（带 version_label 参数）

| 文件:行 | 现有签名 | 新签名 | 备注 |
|---|---|---|---|
| `manifest.py:70-71` | `manifest_path(project_dir)` | `manifest_path(project_dir, version_label)` → `versions/{label}/train/manifest.json` | path 变 |
| `manifest.py:62` | 全局 `_LOCK = threading.Lock()` | `_LOCKS: dict[str, Lock]`（按 `(pid, vid)` key） | 防 N version 并发争用 |
| `manifest.py:194-410` | `add_processed / restore / mark_duplicate_removed / replace_with_crops / clear_all / all_processed / duplicate_removed_origins` | 全加 `version_label` 参数 | |
| `manifest.py` | 新增 `ensure_train_manifest(project_dir, version_label) → Path` | 见 §3.2 |
| `core.py:154-219` | `list_processed(project)` | `list_train_images(project, version_label)`：扫 train/ 物理文件 + manifest entry 拼元数据 | |
| `core.py:260-300` | `resolve_targets(project, ...)` | 加 `version_label`，从 train_dir 列 names | |
| `core.py:353-466` | `list_crop_workspace / list_duplicate_removed_workspace` | 加 `version_label` | |
| `duplicates.py:430-451` | `_resolve_download_sources` | `_resolve_train_sources(version_label)`，scope train/ | D9 |
| `curation.py:210-275` | `copy_to_train` 双分支 | 单分支：`shutil.copy2(download_dir / name, train_dir / name)` + 写 manifest entry `{origin: name}` | D28 |
| `preprocess_worker.py:189-216` | `add_processed` 写盘到 `preprocess/{name}.png` | 写盘到 `train/{name}.png` + 删 `train/{origin}` 老文件（如有） | |
| `preprocess_worker.py:241-265` | `_resolve_crop_source` source 路径 | source = `train/{folder}/{name}` | |
| `preprocess_worker.py:351-399` | crop 写盘到 `preprocess/{name}.png` | 写盘到 `train/{name}.png` | |

### 5.3 后端：新增

- `studio/infrastructure/migrations/_v11_preprocessing_phase.py`（§4.3）
- `studio/services/projects/phase.py:check_preprocessing`（§4.2）
- `studio/services/preprocess/manifest.py:ensure_train_manifest`（§3.2）
- `studio/infrastructure/migrations/__init__.py` 注册 _v11

### 5.4 前端：改造

| 文件:行 | 改动 |
|---|---|
| `studio/web/src/api/client.ts` | `VersionPhase` 加 `'preprocessing'`；`PHASE_ORDER` 加；`PHASE_SKIPPABLE` 加 |
| `studio/web/src/components/Sidebar.tsx:11-29` | `STEP_KEY_TO_PHASE` 加 `preprocess: 'preprocessing'`；`PHASE_TO_STEP_KEY` 加 `preprocessing: 'preprocess'` |
| `Sidebar.tsx:266-274` | `STEPS` 顺序换：`preprocess` 从 `idx=''` (project) 改成 `idx: '2'` (version)，移到 curate 之后；scope `'version'` | 
| `Sidebar.tsx:266-274` | 旧 curate idx '1' / tag '2' / edit '3' / reg '4' / train '5' → curate '1' / preprocess '2' / tag '3' / edit '4' / reg '5' / train '6' |
| `Sidebar.tsx:472` | `projectScopeStep` regex `/^\/projects\/[^/]+\/(download|preprocess)$/` → `/^\/projects\/[^/]+\/download$/`（preprocess 不再 project scope） |
| `studio/web/src/i18n/locales/zh.json:15` | `"preprocess": "预处理工具"` → `"preprocess": "预处理（可选）"`（D17，跟 `reg: "正则集（可选）"` 同 pattern） |
| `studio/web/src/i18n/locales/zh.json:149` | `"preprocess": "② 预处理"` → 删除（不再 project scope）或重定向到 version step 的 idx |
| `studio/web/src/i18n/locales/en.json` 对应位置 | `"preprocess": "Preprocess (optional)"` |
| `studio/web/src/pages/project/steps/Preprocess.tsx`（905 行） | 数据源切 train/；删 pending/processed 双概念；改成"train 视图"单 grid + 状态徽章 |
| `PreprocessOverview.tsx`（316 行） | 统计指标 scope 改 train/，新增 ARB 桶分布卡片 |
| `PreprocessCrop.tsx`（982 行） | source 路径全 train/ |
| `PreprocessDuplicates.tsx`（778 行） | scope train/ |
| `PreprocessHub.tsx`（31 行） | 加 `useActiveVersion` 拿 vid，子路由透传 |
| Curation.tsx | 顶部加"对选中跳预处理 →"按钮（次要，可放 PR-4 followup） |

### 5.5 i18n 文案改动汇总

`zh.json`:
```diff
-  "preprocess": "预处理工具",
+  "preprocess": "预处理（可选）",
```

`en.json`:
```diff
-  "preprocess": "Preprocess Tools",
+  "preprocess": "Preprocess (optional)",
```

`zh.json:149-153`（Sidebar idx 编号区）现状：

```
"download": "① 数据集",
"preprocess": "② 预处理",
"curate": "③ 筛选",
...
```

改成：

```
"download": "① 数据集",
"curate": "② 筛选",
"preprocess": "③ 预处理（可选）",
"tag": "④ 打标",
"edit": "⑤ 标签编辑",
"reg": "⑥ 正则集（可选）",
"train": "⑦ 训练",
```

---

## 6. API endpoint 改造

11 个 endpoint 路径 pid → (pid, vid)（D20 / D21：**不做 redirect**，PR-3 后端 + PR-4 前端同步切换）：

| 旧 URL | 新 URL | 文件:行 |
|---|---|---|
| `POST /api/projects/{pid}/preprocess/start` | `POST /api/projects/{pid}/versions/{vid}/preprocess/start` | `ingestion.py:222` |
| `GET /api/projects/{pid}/preprocess/status` | `GET /api/projects/{pid}/versions/{vid}/preprocess/status` | `ingestion.py:275` |
| `GET /api/projects/{pid}/preprocess/files` | `GET /api/projects/{pid}/versions/{vid}/preprocess/files` | `ingestion.py:301` |
| `GET /api/projects/{pid}/preprocess/duplicates/removed` | `GET /api/projects/{pid}/versions/{vid}/preprocess/duplicates/removed` | `ingestion.py:315` |
| `GET /api/projects/{pid}/preprocess/crop/workspace` | `GET /api/projects/{pid}/versions/{vid}/preprocess/crop/workspace` | `ingestion.py:331` |
| `POST /api/projects/{pid}/preprocess/crop` | `POST /api/projects/{pid}/versions/{vid}/preprocess/crop` | `ingestion.py:348` |
| `POST /api/projects/{pid}/preprocess/files/reset` | `POST /api/projects/{pid}/versions/{vid}/preprocess/files/reset` | `ingestion.py:380` |
| `POST /api/projects/{pid}/preprocess/files/restore` | `POST /api/projects/{pid}/versions/{vid}/preprocess/files/restore` | `ingestion.py:397` |
| `GET /api/projects/{pid}/preprocess/thumb` | `GET /api/projects/{pid}/versions/{vid}/preprocess/thumb` | `ingestion.py:421` |
| `POST /api/projects/{pid}/preprocess/duplicates/scan` | `POST /api/projects/{pid}/versions/{vid}/preprocess/duplicates/scan` | `curation.py:271` |
| `POST /api/projects/{pid}/preprocess/duplicates/apply` | `POST /api/projects/{pid}/versions/{vid}/preprocess/duplicates/apply` | `curation.py:342` |

**额外**：thumb endpoint (`curation.py:117-186`) 的 `bucket` 参数语义：

- `bucket=download` 保留（curation 页用）
- `bucket=preprocess` 删除（不再存在 project 级 preprocess）
- 新增 `bucket=train`（缩略图从 train/ 取）

---

## 7. UI 改造

### 7.1 Sidebar Stepper 顺序（D18）

```
当前:  ① 数据集 → ② 预处理(project) → ③ 筛选 → ④ 打标 → ⑤ 编辑 → ⑥ 正则(可选) → ⑦ 训练

新:    ① 数据集 → ② 筛选 → ③ 预处理(可选,version) → ④ 打标 → ⑤ 编辑 → ⑥ 正则(可选) → ⑦ 训练
```

Phase cursor 映射：preprocess step idx '③' → version phase `preprocessing`（version scope）。

可选 step 视觉跟 reg 完全一致（i18n 标签加"（可选）"，无其他特殊样式）。

### 7.2 Preprocess 页 grid

数据源 = `versions/{label}/train/` 整目录 + 同位 manifest.json。

每张图卡片：

```
┌──────────────┐
│  [缩略图]    │   X.png  (origin: X.jpg)
│              │   2048×2048 · 4MB
│   [✓ upscale]│
│   [✓ crop]   │
└──────────────┘
```

徽章逻辑（基于 entry 字段差异，§2.2 状态隐含推断）：

- `name.suffix != origin.suffix` → 显示 `✓ upscale`（扩展名变 = 处理过）
- `"_c" in name` → 显示 `✓ crop` 派生标记
- `mtime/size` 跟 train/{name} 物理文件不一致 → 显示 `⚠ 外部修改`

顶部统计区（D10 / D11）：`X 张图 · 平均 W×H · ARB 桶分布柱状图`。

### 7.3 错误恢复 UX（D8 详化）

`restore` 失败 toast（前端 `PreprocessOverview.tsx` 复原按钮失败处理）：

```
✕ 3 张图无法恢复
─────────────────────────
原图已从 download/ 删除：
• X.jpg
• Y.jpg
• Z.jpg

[拖入替换] [保留处理后版本] [从训练集移除]
```

三选项语义：

- **拖入替换**：打开文件选择器，用户选本地图覆盖 `download/{origin}`，再触发 restore
- **保留处理后版本**：忽略失败，train/{name} 保持现状（manifest entry 不变）
- **从训练集移除**：删 train/{name} + 删 manifest entry（不可恢复，需二次确认 dialog）

---

## 8. 测试计划

### 8.1 单测重点（PR-1 + PR-2）

| 测试文件 | 内容 |
|---|---|
| `tests/test_train_manifest_fallback.py`（**新增**） | `ensure_train_manifest` 8 个 case：目标已存在 / 老 manifest 不存在 / 老 manifest 损坏 / multi-crop 派生匹配 / train/ 部分图无 entry / 重复调用幂等 / 并发安全 / `_LOCKS` per-version 隔离 |
| `tests/test_preprocess_manifest.py` | 删 ~60% 老 case（v0/v1 read-compat / `resolve` / `resolve_origin` / `_scan_legacy_sidecars`）；改 ~20% 加 `version_label` 参数；新增 ~20% 测 v2 schema |
| `tests/test_preprocess.py` | 删 `list_pending` 测试（~80 行）；改 `list_train_images` 加 vid；保留 worker 行为测试 |
| `tests/test_preprocess_crop_worker.py` | 改源路径常量到 train/；保留 multi-crop fan-out 行为测试 |
| `tests/test_preprocess_endpoints.py` | 改 URL pattern + 加 vid path param；测 thumb endpoint bucket=train |
| `tests/test_curation.py` | 删 `test_copy_to_train_uses_processed_bytes_when_available:158-168`（双分支已合并）；删 `test_curation_view_expands_multi_crop_derivatives:183-204`（fan-out 行展开删除） |
| `tests/test_curation_endpoints.py` | URL pattern 改 |
| `tests/test_migration_v11.py`（**新增**） | _v11 回填 5 个 case（§4.3 末尾列表） |
| `tests/test_phase.py` | 加 `check_preprocessing` 测试：job pending → 失败；无 job → 通过；测 ORDER 包含 preprocessing；测 SKIPPABLE 包含 preprocessing |

### 8.2 集成测试

- e2e：创建 project → download 导入 → 创建 version → curate 入选 → preprocess upscale → tag → train（确认 phase 推进正确）
- fork：v1 完整跑完到 ready → fork v2 → 确认 v2 train/manifest.json 自动复制 + phase 跟随
- 老项目兼容：起 fixture 项目（含老 `preprocess/manifest.json` + train/ 已有图）→ 启动 → 第一次访问 → 确认 `ensure_train_manifest` 隐式重建成功

### 8.3 前端测试

vitest 现状无 Preprocess 单测（grep 确认）。PR-4 不强求新增 vitest，靠端到端。

---

## 9. PR 切片 + 时序（D29 / D30）

### 9.1 PR 拆分

| # | PR title | 范围 | 工时 | 阻塞依赖 |
|---|---|---|---|---|
| **PR-1** | `feat(preprocess): add ensure_train_manifest fallback + ADR 0010 draft` | ADR 0010 草案（supersede 0004）+ `manifest.py:ensure_train_manifest` + `tests/test_train_manifest_fallback.py` | **0.5d** | 无（独立） |
| **PR-2** | `refactor(preprocess): move manifest scope from project to version train/` | `manifest.py` 瘦身（删 §5.1 + 改 §5.2）+ `core.py` / `duplicates.py` / `preprocess_worker.py` / `curation.py:copy_to_train` 改造 + 测试调整 | **3.0d** | lifecycle PR 链 + `feat/preprocess-multitool` 合 dev；PR-1 合 dev |
| **PR-3** | `feat(lifecycle): add preprocessing phase + _v11 migration + endpoint vid routing` | `VersionPhase.ORDER` 加 PREPROCESSING + `SKIPPABLE` 加 + `check_preprocessing` + `_v11_preprocessing_phase.py` + 11 个 API endpoint 加 vid path | **1.5d** | PR-2 合 dev |
| **PR-4** | `feat(ui): preprocess as version phase + sidebar reorder + train-scope grid` | 前端 Preprocess hub + 4 子页接 vid 上下文 + Sidebar STEPS 重排 + i18n "（可选）" + Preprocess.tsx 数据契约 | **2.5d** | PR-3 合 dev |
| | i18n review / a11y check（跟 PR-4 并行） | | +0.5d | |
| | **总计** | | **8.0d** | |

### 9.2 时序图

```
[现在]
  ├── PR-1 (0.5d) — 可立刻开
  │
  └── 等 in-flight 链合 dev:
       feat/lifecycle-v8-schema (PR-2 lifecycle)
       └── feat/lifecycle-business-logic (PR-3/4 lifecycle)
            └── feat/lifecycle-v9-destructive (PR-5 lifecycle)
                 └── feat/lifecycle-frontend-v3 (PR-6 lifecycle)
                      └── feat/preprocess-multitool
                           └── fix/preprocess-stage-and-crop-thumb
                                │
                                └── PR-2 (3.0d)
                                     └── PR-3 (1.5d)
                                          └── PR-4 (2.5d)
                                               → MERGE
```

### 9.3 并行机会

- PR-1 跟 lifecycle PR 链 100% 并行（不冲突）
- PR-2 / PR-3 / PR-4 串行（schema 改 → migration + API → 前端）
- 后端 PR-2 在 review 期间可以**起草** PR-3 的 _v11 migration + check_preprocessing 代码（不依赖 PR-2 合 dev，只在自己分支基于 PR-2 commit 开发）

---

## 10. ADR 协调

### 10.1 ADR 0010 起草（PR-1 范围内）

**关键章节**（详见单独 ADR 文件，本 plan §10 只列大纲）：

- Status: Proposed
- Supersedes: ADR 0004（含 Addendum 1）
- **编号说明**：跳过 0009（已被"统一日志 + 错误体系"占用，详 `docs/adr/README.md`）
- 背景：用户反馈四个痛点（前置时间 / 统计无意义 / 聚类失效 / booru 素材特性）
- 候选方案：A 维持现状 / B UI filter / **C 本方案**
- 决策：方案 C（preprocess scope → train 集 + per-version manifest + fallback 重建）
- 数据模型（§2 内容）
- 兼容策略（§3.2 ensure_train_manifest）
- 跟 ADR 0007 关系：amendment 加 PREPROCESSING phase
- 跟 ADR 0008 关系：模块边界不动
- 后果 / 新债

### 10.2 ADR 0004 改动（ADR 0010 accept 后才动）

**ADR 0010 Proposed 期间 0004 状态不变**——README 索引和 ADR 0004 文件本身都保持现状，仅 README 加 hint "待 ADR 0010 accept 后转 Superseded"。ADR 0010 accept 时（PR-2 或 PR-3 合 dev 时）再做下列改动：

文件顶部状态行改为：

```markdown
**状态**：Superseded by [ADR 0010](0010-preprocess-train-scope.md) (YYYY-MM-DD)
```

正文不删（保留作历史决策记录），但加一段 banner：

> **Note**: ADR 0010 把 preprocess scope 从项目级下沉到 version 级 train/。本 ADR 的 schema / resolver / 项目级 manifest 设计**已弃用**；老项目通过 `ensure_train_manifest()` 隐式 fallback 重建到新模型。本 ADR 仍保留作历史决策记录。

### 10.3 ADR 0007 amendment

`docs/adr/0007-project-version-lifecycle-refactor.md` 加 Addendum：

```markdown
## Addendum 1 — 加 `preprocessing` phase（2026-06-03，ADR 0010 配套）

ADR 0010 把 preprocess scope 下沉到 version 级 train/，作为新的可选 phase 加入 cursor。

VersionPhase.ORDER 改为 6 个：

```
curating → preprocessing → tagging → editing → regularizing → ready
```

VersionPhase.SKIPPABLE 加 `preprocessing`（跟 `regularizing` 一致心智）。

phase 校验 `check_preprocessing`：无 preprocess job pending/running 即可推进（不强求处理过任何图）。

§70 "数据集 version 级否决" 不撕——train 集合 source-of-truth 仍是项目级 `download/` 池（curating phase 从 download 复制进 train）；本 amendment 仅把 preprocess 产物跟随 train 一起落 version 级，不反转数据集归属决议。

详 ADR 0010。
```

### 10.4 ADR 0008 / 0009 不动

`services/preprocess/` 模块仍存在（核心逻辑 upscale / crop / blur / dedup worker 不变），只是状态存储位置从项目级搬到 version/train/。模块边界不变。

---

## 11. 风险清单

| 风险 | 类别 | 影响 | 缓解 |
|---|---|---|---|
| `ensure_train_manifest` 重建逻辑错误（漏图 / 错 origin） | 中 | 用户 restore 找错图 | 单测覆盖 8 个 case + 整合测试用真实老项目 fixture |
| _v11 migration 错把 phase=curating + 空 train/ 推到 preprocessing | 低 | UX：用户进入 v 后被跳过 curating | 严格 check `any(train_dir.iterdir())` + 单测 |
| 老 preprocess/ 目录长期占磁盘 | 低 | 用户存储压力 | 0.13.0 release notes 提醒可手动删；不主动删 |
| 多 version 并发写 manifest 死锁 | 低 | manifest 损坏 | `_LOCKS: dict[(pid, vid), Lock]` per-version 隔离 + 原子 tmp+rename 落盘 |
| Multi-crop fan-out 在 fork 后跨 version 同步问题 | 低 | v2 改 crop 不影响 v1 | 测试 case：v1 crop 后 fork v2，v2 改 crop 不写回 v1 |
| 11 个 endpoint URL 改后老前端 / 第三方调用 404 | 中 | 用户更新到新版本 + 浏览器缓存老 JS → 报错 | beta 心智 + frontend 同 PR 切换；前端缓存靠 hash 失效 |
| 一次性 i18n 文案改动遗漏 | 低 | 文案不一致 | grep 全量 i18n key 后批改 |
| ADR 0007 PR-7 destructive `_v9` 跟 _v11 顺序问题 | 低 | _v11 假设 phase 列存在 | _v11 必须在 _v9 之后；migrations/__init__.py 顺序保证 |

---

## 12. Open Questions（已决 / 待 implementer 注意）

### 12.1 已决（D1-D30 全部，本 plan 是 ground truth）

### 12.2 待 implementer 在 PR 期间注意的小决策

1. **`_LOCKS` 实现细节**：用 `WeakValueDictionary[(pid, vid), Lock]` 还是常规 dict？前者 GC 友好，后者实现简单。**主审推荐常规 dict**（version 数 << 1000，内存压力可忽略）。
2. **`ensure_train_manifest` 调用是否 short-circuit**：第一次 stat 已经 ensure，后续调用的 stat 是否 cache？**推荐不 cache**（O(1) stat 代价极低，cache 反而引入失效问题）。
3. **fork 时 manifest 复制**：`_copytree("train")` 已经递归复制；是否在 fork 后**显式**调一次 `ensure_train_manifest(new_label)` 兜底（万一源 manifest 损坏）？**推荐显式调**（防御性，cost 极低）。
4. **i18n idx 编号**：现 zh.json:149-153 有 `① ② ③ ④ ⑤` 编号嵌在 i18n value 里。如果 implementer 决定改成"label 不带编号 + Sidebar 单独渲染编号"，那是个独立的小重构，**不在本 plan scope**。维持现状 inline 编号即可。

### 12.3 跟其他 in-flight work 协调

- `feat/preprocess-multitool` 已经把 Preprocess 拆成 hub + 4 子页结构 → PR-4 完全复用，不需要重新设计
- `feat/lifecycle-frontend-v3` PR-6 落地 phase header → PR-4 在 phase header 加 preprocessing step 应该是 trivial（跟 reg 同 pattern）
- `feat/lifecycle-v9-destructive` 删 `stage` 字段 → 必须在 _v11 之前合 dev（_v11 假设 stage 列已删，只动 phase 列）

---

## 13. 引用

### 关键 ADR / 代码
- `docs/adr/0004-preprocess-manifest.md`（含 Addendum 1）— 待 supersede
- `docs/adr/0007-project-version-lifecycle-refactor.md` — 加 Addendum 1（PREPROCESSING phase）
- `docs/adr/0010-preprocess-train-scope.md` — 待起草（PR-1 范围）
- `studio/services/preprocess/manifest.py` — 主要改动目标（瘦身 + 加 `ensure_train_manifest`）
- `studio/services/preprocess/core.py:106-219` — `list_pending` 删；`list_processed` 改
- `studio/services/preprocess/duplicates.py:430-451` — scope 改 train/
- `studio/services/dataset/curation.py:210-275` — `copy_to_train` 大幅简化
- `studio/workers/preprocess_worker.py:189-216,241-265,351-399` — 写盘路径改 train/
- `studio/services/projects/versions.py:41-58` — `VersionPhase.ORDER` / `SKIPPABLE` 加新值
- `studio/services/projects/versions.py:407-498` — `create_version` fork 流程（加 `ensure_train_manifest` 兜底）
- `studio/services/projects/phase.py` — 加 `check_preprocessing`
- `studio/infrastructure/migrations/_v11_preprocessing_phase.py` — 新文件
- `studio/api/routers/projects/ingestion.py:217-441` — 11 个 endpoint URL 改
- `studio/api/routers/projects/curation.py:117-186,271,342` — thumb endpoint bucket=train + 2 个 duplicates endpoint URL
- `studio/web/src/components/Sidebar.tsx:11-29,266-274,472` — STEP 顺序 / phase 映射 / project scope regex
- `studio/web/src/api/client.ts` — `VersionPhase` / `PHASE_ORDER` / `PHASE_SKIPPABLE`
- `studio/web/src/i18n/locales/{zh,en}.json` — "（可选）" 文案 + idx 编号
- `studio/web/src/pages/project/steps/Preprocess*.tsx` — 数据契约 + 4 子页
