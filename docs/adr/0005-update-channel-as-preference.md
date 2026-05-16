# 0005 — 更新通道（master / dev）作为用户视图偏好，与 git 工作树状态解耦

**状态**：Accepted
**日期**：2026-05-16
**决策者**：@WalkingMeatAxolotl

## 背景

ADR 0002（webui 自更新）落地后，Settings → 系统 → 版本面板有两个长期暗坑，在
v0.8.0 release 直后被触发：

### 现象

用户做了 v0.8.0 release（dev → master + PR merge + 打 tag）后，本地 master
还没 `git pull`。打开版本面板看到：

| UI 元素 | 显示 | 用户解读 |
|---|---|---|
| master 卡「你在这里」 | 我在 master | branch 名 = master |
| master 卡 `v0.8.0` | 我装的是 v0.8.0 | `__version__` 字符串 |
| master 卡「↑落后 2 commits」 | 我落后了 | git `rev-list --count` |
| dev 卡「HEAD f6f202b」 | dev 顶端是这个 | 远端 |
| dev 卡列表「● 当前」 | 我装的就是这个 | commit hash 比较 |
| 「切到 dev (f6f202b)」按钮 | 可以切 | onDev=false |
| `v0.8.0 → v0.8.0` 箭头 | 升级前后版本号一样？ |  |

点「切到 dev (f6f202b)」 → preview 出现 → 确认 → 重启 → UI **没有变化**，
仍然显示「master · 你在这里」。

### 两个根因

1. **UI 用 git 词汇而不是产品语言**。
   - `commits_ahead` 直接从后端 `git rev-list --count` 出来，前端原样显示
     「↑落后 N commits」。
   - 但用户视角的"落后"只有一个维度：版本号。版本号 v0.8.0 == v0.8.0 时根本
     不存在"落后"。`__version__` 字符串与 commit hash 是两套独立维度，UI 把
     两套维度混着展示就产生了 `v0.8.0 → v0.8.0` + 「落后 2 commits」的矛盾。

2. **"通道"概念绑死在 git 工作树状态上**。
   - 「我在哪个通道」判定 = `git rev-parse --abbrev-ref HEAD`（branch 名）
   - 「切换通道」实现 = `git reset --hard <ref>`（不动 branch 名）
   - 两个机制大多数场景下"凑巧"能用：dev 通常比 master 领先，reset 后 commit
     hash 变了，UI 看上去就"动了"。
   - **release 直后例外**：本地 commit `f6f202b` 已经 == origin/dev HEAD（release
     commit 本身在 dev 上做），切到 dev 是 no-op，branch 名也不变 → UI 永远
     卡在「master · 你在这里」。

更深的：UI 把 master 卡 + dev 卡**并排同屏**渲染，本应互斥的两个通道在视觉上
变成两张同时活着的卡，让用户陷入「我究竟在哪个通道」的矛盾解读。

## 决策

**通道是用户视图偏好，不是 git 工作树状态**：

1. **用户偏好持久化为 `system.update_channel`**（`"stable"` / `"dev"`），存
   `secrets.json`。**切 toggle 不触发任何 git 操作**，纯 UI 视图切换。
2. **真正"切到 dev HEAD" / "更新到 vX.Y.Z" / "回滚"是独立按钮**，跟通道偏好
   解耦 —— 用户可以"订阅 dev 通道做研发"但暂时"装着稳定版"。
3. **同屏只展示当前选中通道的卡片**（不再 master + dev 并排）。
4. **后端引入「装了什么」分类 `installed_kind`**（`stable` / `dev` / `custom`），
   按 commit hash 比对推断，**取代前端依赖 `branch` 字段做产品判断**。
5. **前端文案只用版本号 / 状态语言**，不出现 `commits` / `sha` / `branch` 等
   git 词汇。"落后 N commits" → "有新稳定版 vX.Y.Z" / "已是最新" / dev 通道
   改"N 项新更新"。

### 后端 API 形态

`VersionInfo`（`/api/system/version`）：

```python
@dataclass
class VersionInfo:
    version: str
    commit: str
    commit_short: str
    commit_time_iso: str
    branch: str              # debug 用，前端不再做产品判断
    tag: Optional[str]
    is_dirty: bool
    # ---- ADR 0005 ----
    installed_kind: str      # "stable" / "dev" / "custom"
    installed_label: str     # "v0.8.0" / "dev @ f6f202b · 2026-05-16" / "自定义（feat/foo @ a1b2c3d）"
    stable_version: Optional[str]  # "vX.Y.Z" 仅 stable 时填
```

分类规则（优先级）：

1. HEAD 命中 `vX.Y.Z` release tag → `stable`
2. `__version__` 匹某 vX.Y.Z tag 且当前 commit 与 tag commit 的 **tree 一致** →
   `stable`（覆盖 release 直后 release commit 在 dev、tag 在 merge commit 的场景）
3. commit == `origin/dev` HEAD → `dev`
4. else → `custom`（feature branch / detached）

`UpdateCheckResult`（`/api/system/update_check`）：

```python
@dataclass
class UpdateCheckResult:
    channel: str
    current_commit: str
    latest_commit: str
    commits_ahead: int       # 内部 debug 保留
    has_update: bool         # 兼容字段 = (state == "update_available")
    latest_tag: Optional[str]
    checked_at: float
    # ---- ADR 0005 ----
    state: str               # "up_to_date" / "update_available" / "ahead" / "detached"
    installed_version: Optional[str]
    latest_version: Optional[str]
    behind_count: int        # 给前端用的"N 项更新"数字
    error: Optional[str]
```

state 推断：
- **master 通道**：版本号优先（`installed_version == latest_version` → up_to_date），
  没版本号（custom / dev）回落到 commit 比较
- **dev 通道**：直接 commit hash 比较 + ahead/behind 计数

### 前端

- `formatMasterStateText(check)` / `formatDevStateText(check)`：纯函数把 state
  + check 数据 → 用户可读文案
- `shouldShowMasterUpdateButton(check)`：state=update_available 且有 latest_version
- `isDevSwitchButtonDisabled(check)`：state=up_to_date → disabled
- toggle 视觉切换走 `<button role="radio">`，写 `secrets.json` 但**不**调
  `/api/system/update`

### 迁移

旧 `system.show_dev_channel`：保留 pydantic 字段做兼容；
`_migrate_legacy_schema` 一次性映射 `show_dev_channel=true → update_channel="dev"`，
前端 PATCH 时同时写两个字段保持旧版本回滚兼容。

## 后果

### 正面

- release 直后用户的版本面板能正确显示「已是最新 v0.8.0」+「与 dev HEAD 一致」，
  不再有"落后 2 commits"+「切到 dev 没反应」的矛盾
- UI 文案彻底脱离 git 词汇，对非 git 用户更友好
- 「装了什么」与「订阅哪条通道」解耦后，未来支持"装稳定版但跟 dev 看预览"等
  灵活组合时不需要再改架构

### 负面 / 待评估

- `installed_kind=stable` 判定靠 `git diff --quiet` tree 比较，每次 `current_version()`
  会多 1-2 个 git 调用。release 后短期内本地 tag 没 fetch 全时可能误判为 custom，
  靠下次 `check_update` fetch tag 后自动纠正
- 后端 `commits_ahead` / `has_update` 字段保留作兼容，但前端不再读 —— 任何外部
  脚本读这俩字段的还能跑，可逐步迁移

## 不在范围

- "切到此 commit"（点 dev 列表里旧 commit 切过去）保留，但属于 dev 通道展开
  区里的二级操作
- 自动检查 + Topbar 红点维持 ADR 0002 的"只看 master"决定，不因通道偏好改
- 完整翻译 Settings 页里其他地方的 git 词汇（譬如 secrets 里 wandb 同步状态）
  不在此 ADR 范围
