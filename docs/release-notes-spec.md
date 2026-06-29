# Release Notes 编写规范（AI agent 用）

> **⚠️ 2026-06-28 格式迁移（ADR 0013）**：release notes 已从 `release_notes.yaml`
> 单文件迁成 `docs/announcements/<date>-v<version>.md` 一版一文件（`tag: release`、
> markdown 正文、双语双文件）。**文件格式 / frontmatter / tag / `bump_version` 命令**
> 以 [`docs/announcements/README.md`](announcements/README.md) 为准；下文涉及
> `release_notes.yaml` / 结构化 `kind`+`summary`+`pr_refs` / `summary≤80` 的**机制部分已过时**。
> 本文**仍然适用的是与格式无关的写作原则**：语气、内容取舍、粒度、diff 视角、GFM 坑等。
>
> **📌 写作规范已统一升级到 [`docs/announcements/CONTENT-GUIDE.md`](announcements/CONTENT-GUIDE.md)**
> （覆盖 release / notice / migration 等所有公告类型，综合 Apple HIG + Keep a Changelog +
> 业界实践）。新写公告以 CONTENT-GUIDE 为准；本文保留为历史参考与更细的 release-notes 例子。

本文是给写 release notes 的 agent 看的——人 / Claude / 任何 LLM。读完应当
**不需要任何额外问问题**就能：(1) 知道改哪个文件、(2) 从哪儿拿内容、
(3) 用什么结构 / 风格 / 语气写、(4) 跑哪个工具校验。

> **核心原则 —— 三 surface 共享文本，全部 user-facing**
>
> `release_notes.yaml` 的 `summary` + `detail` 文本会被同时渲染到三个用户
> 可见 surface：
>
> 1. **Studio Web UI** Settings → 系统 → 版本卡片：summary 一行展示 + 点开
>    detail modal
> 2. **`CHANGELOG.md`** 仓库根 + GitHub repo 主页可达；`render-changelog` 从
>    yaml 派生，summary + detail 都进
> 3. **GitHub Release body**（`/releases` 每个 tag 一页）；maintainer 手动从
>    CHANGELOG 复制对应段
>
> 三处共享同一份 yaml，**summary 和 detail 都得 user 视角写**。旧说法
> 「summary 给用户，detail 给开发者技术词随便用」错了 —— detail 在 CHANGELOG
> 和 Release body 两个用户表面同样可见。技术 ref（实现路径 / 内部符号名 /
> 重构理由 / 设计权衡 / 替代方案对比）属于 **commit message + PR
> description**，不进 yaml。
>
> 4 层 archeology 分工：
>
> | 层 | 受众 | 内容 |
> |---|---|---|
> | Commit message | git blame / 工程师 | atomic 改动的根因 / 修法 |
> | PR description | reviewer / 历史考古 | 设计权衡、替代方案、scope |
> | CHANGELOG / Release body | 升级的用户 | 「升上去看到什么变化」 |
> | ADR | 架构决策回溯 | 「为啥选这条路不选另一条」 |

## 1. 单一来源：`release_notes.yaml`

**永远改 `release_notes.yaml`；永远不要手改 `CHANGELOG.md`。**

`CHANGELOG.md` 由 `tools/bump_version.py render-changelog` 从 yaml **派生**，
任何手改都会在下次 `render-changelog` 时被覆盖。GitHub release page 看的
是 `CHANGELOG.md`，所以 yaml 才是 source of truth。

`studio/__init__.py:__version__` 和 `studio/web/package.json:version` 也由
`bump_version.py bump --version X.Y.Z` 统一改写，不要手动改。

## 2. 数据模型（yaml schema）

```yaml
- version: "0.6.1"              # str, semver；与 git tag / __version__ 对齐
  date: "2026-05-13"            # str, ISO YYYY-MM-DD
  summary: "..."                # str?, optional. 整版本一句话总览
                                # （CHANGELOG 顶部段落用）；可省，agent 通常写
  entries:                      # list, ≥ 1 条
    - kind: added               # enum, 必填，见 §4
      summary: "..."            # str, 必填, ≤ 80 字符, plain text, user-facing
      pr_refs: [18, 34]         # list[int], optional; 关联的 PR 号
      detail: |                 # str, optional; markdown 允许；多行
        Detail block, 可多段。
```

**版本顺序**：yaml 是 list, **最新版本排第一**（top）。`bump_version.py bump`
会自动 prepend 新 block 到列表头。

## 3. 工作流

agent 被叫来做 release notes 时，按这个顺序：

### Step 1: 找出 last release commit / tag

```bash
# yaml 现有的最新版本（agent 之前的 release 标记点）
tail -n +1 release_notes.yaml | grep -m1 '^- version:'

# 仓库里对应的 tag（如果有打 tag）
git describe --tags --abbrev=0 2>/dev/null
```

如果 yaml 里写的是 `0.6.0` 但 repo 没 `v0.6.0` tag，以**当时记录的最后一个
commit 作为分界点**（通常是 `chore(release): 0.6.0` 这个 commit；用 git
log 找）：

```bash
git log --oneline --grep='chore(release): 0.6.0' -1
```

### Step 2: 拉取自上次 release 以来的所有合并 PR

```bash
# PR 列表（base 通常是 dev 或 master，按你们的 release flow 选）
gh pr list --state merged --base dev --limit 100 \
  --search 'merged:>=2026-05-12' \
  --json number,title,body,labels,author,mergedAt
```

`--search` 的日期 = 上次 release 的日期；`--base` 跟你们 release 拉的分支
（默认 dev）。如果你们 release 是从 master 拉，base 就改成 master。

补充材料（PR 描述有时太简单）：

```bash
# 该 PR 关联的所有 commits
gh pr view <num> --json commits --jq '.commits[].messageHeadline'

# 整个版本区间的 commit 全貌
git log <last_release_sha>..HEAD --pretty='%h %s'
```

### Step 3: 按 PR 一条一条决定 kind + 写 summary + detail

**默认每个 PR 写一个 entry**。例外：

- **纯 chore PR**（依赖 bump / 格式化 / 内部 refactor，对用户行为零影响）：**不写 entry**
- **纯 docs PR**（README / 注释）：**不写 entry**
- **跨多个领域的大 PR**：可拆成多条 entry，每条对应 PR 一部分；`pr_refs` 都填同一个 PR 号
- **同主题的多个 PR 串**（feature 主 PR + 后续 followup fix PR）：可合并成一个 entry，
  `pr_refs` 列出所有相关 PR（agent 判断）。比如 `[18, 34, 35]`：主 feature + P0 修 + 重做

### Step 4: 追加到 yaml

新版本 block prepend 到 list 顶；entries 内部按重要性排（user 最关心的在前），
**不要**按 PR 时间顺序排。

### Step 5: 校验 + 派生

```bash
python tools/bump_version.py validate
python tools/bump_version.py bump --version 0.6.1 --date 2026-05-13
```

`bump` 会：(1) 重跑 validate (2) 改 `studio/__init__.py:__version__` +
`studio/web/package.json` (3) 调 `render-changelog` 重写 `CHANGELOG.md` (4)
打印 commit 建议。**不会** 自动 commit / tag —— 那是人的活。

## 4. `kind` 分类（标准 + 我们的扩展）

参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，加 `improved`。

| `kind` | 含义 | 何时用 |
| --- | --- | --- |
| `added` | 完全新的 feature / 端点 / UI / 文件 | 用户能"发现"一个新东西 |
| `changed` | 行为 / 接口 / 默认值 / UI 流程**变了** | 用户原来这么用，现在那么用 |
| `improved` | 现有功能性能 / UX / 文案 / 错误信息**优化** | 表面行为不变，体感更好 |
| `fixed` | bug 修 | 之前坏的，现在好了 |
| `removed` | 删了 feature / 端点 / 字段 / 文件 | 用户能"感觉到"少了东西 |
| `deprecated` | 标记将删，但还能用 | 通常预告下下个版本会 `removed` |
| `security` | CVE / auth / 凭证泄漏类修复 | **优先级最高**，单独写 |

**含糊判断**：

- "重写 UI 但用户操作流程没变" → `changed`（用户能感知 UI 变了）
- "重写后端 service 内部，对前端 API 不变" → 通常**不写**（agent 跳过这条 PR）
- "之前慢，优化后快 5x" → `improved`
- "之前会偶尔报错，修了" → `fixed`
- "新加配置项，default 同旧行为" → `added`（默认行为不变，但用户能发现）
- "把 feature 默认关改成默认开" → `changed`（默认值变化是行为变化）

## 5. `summary` 写作规范

**目的**：UI release notes 面板一行展示。用户扫一眼就知道这版干了啥。

### Do

- **≤ 80 字符**（中文每字算 1，含括号和 PR 号）。`bump_version.py validate` 会拒长的
- **user-facing 语气**：从用户视角写，不是开发者视角
- **现在时 / 命令式**：「修复 X」「新增 Y」「优化 Z」（不要"我们做了"「已经实现」）
- **末尾带 `（#N）`**：链接 PR；多 PR 就 `（#N, #M）`，最多 3 个
- **specific 而不是 generic**：「修复 Danbooru 403」比「修复一些 bug」好
- **避开技术术语**（除非用户必须知道）：「训练监控加 GPU 占用」比「`_StatsThread` 推 SSE event」好
- **三 surface 一致 user 视角**：写完想象这条 entry 同时出现在 Studio 版本
  卡片、CHANGELOG、GitHub Release body —— 都是给升级用户看的，不是给
  reviewer 或 git blame 来的人。技术 ref 进 commit / PR，不进这里

### Don't

- ❌ Markdown 格式（`**bold**` / `code` / 链接）——summary 是 plain text
- ❌ 引号包标题：`「LLM tagger」新增` 不要，直接 `新增 LLM tagger`
- ❌ 不在末尾的 PR 号：`#18 LLM tagger` ❌，`LLM tagger（#18）` ✓
- ❌ 多句话 / 句号串句：`新增 X。修复 Y。` ❌，拆两个 entry
- ❌ 模糊词："优化体验" / "改进若干" / "杂项修复"——具体什么体验？

### 例

| ❌ Bad | ✓ Good |
| --- | --- |
| `**LLM tagger + WandB 监控**（#18）` | `LLM tagger 第二打标器 + WandB 训练监控（#18）` |
| `修复一些 UI bug` | `修复 Settings 系统 tab 切换不刷新（#52）` |
| `重构 services/booru_api` | （不写——纯内部） |
| `优化体验（#33）` | `Queue 输出下载改直链 + 批量 + 排序（#33）` |
| `**onnxruntime-gpu 静默降级 CPU**(#29 Windows / #30 Linux)` | `onnxruntime-gpu 在 Win/Linux 静默降级 CPU（#29, #30）` |

## 6. `detail` 写作规范

**目的**：用户点开 entry 想了解"具体怎么变的"时看；CHANGELOG.md
派生时也用 detail 填二级信息。

### Do

- **Markdown 允许**：列表、代码 fence、行内 code、链接、粗体
- **写 _why_ 不只 _what_**：summary 已经说了 what，detail 说为什么这么改 / 影响 / 边界
- **包含具体路径 / 命令 / 配置 key**：让用户能定位
- **多个 sub-point 用 bullet**：`-` 顶格

### Don't

- ❌ 重复 summary 已经说的话
- ❌ 把 commit message 原样 paste 进来 —— detail 也是 user-facing surface
  （CHANGELOG / Release body 都展示它）
- ❌ 写实现细节而不写用户能感知的部分（"`_StatsThread` 用 nvidia-ml-py 而不是 pynvml" → 用户不在乎，跳过）
- ❌ 写「实现路径」/「内部类名」/「重构理由」/「替代方案对比」/「设计权衡」
  —— 这些属于 commit message 或 PR description，不进 yaml。例如
  「`useEffect` deps 用 `[task]` 对象引用，React 浅 clone 触发重拉」就是
  典型踩线 —— 用户不知道 `useEffect` 是什么
- ❌ 单行 detail（一行能写完的内容直接进 summary）

### 例

```yaml
- kind: added
  summary: "训练监控加 Topbar 系统资源 pill（CPU / GPU / MEM / VRAM）（#37, #42）"
  pr_refs: [37, 42]
  detail: |
    - Topbar 永远显示 4 个等宽 pill（min-w 96px）；从 `nvidia-ml-py` 拉，
      老 `pynvml` 已停维护
    - Backend `_StatsThread` 每 2.5s 通过 SSE `system_stats_updated` 推到前端
    - Monitor 视图改增量协议（步进式 delta 取代每秒 snapshot），10k 步训练
      payload 从 O(N) 降到 O(1)
    - Cold-start 默认 `max_points=0` 不降采样，前端 cap 5000 → 50000 与
      backend `train_monitor` 对齐
```

## 7. End-to-end 完整例子

**场景**：上次 release `0.6.0` 在 2026-05-12；现在准备发 `0.6.1`。

### Step 1-2: gather

```bash
$ gh pr list --state merged --base dev \
    --search 'merged:>=2026-05-12' \
    --json number,title,body --limit 50

[
  {"number": 51, "title": "feat: webui 自更新（ADR 0002）", "body": "..."},
  {"number": 52, "title": "feat(version-section): 双通道升级面板", "body": "..."},
  {"number": 53, "title": "chore: bump nvidia-ml-py to 12.0.3", "body": "..."}
]
```

agent 思考：
- #51 → 大 feature，对应 ADR 0002，应当写 entry（`added` 或 `changed`？webui 内可视化升级是新功能 → `added`）
- #52 → UI 重设计，原版可点击但样式 / 状态机变了 → `changed`
- #53 → 纯 chore dep bump，**跳过**

### Step 3: write entries

```yaml
- version: "0.6.1"
  date: "2026-05-13"
  summary: "webui 内一键升级 + 系统设置版本面板重设计"
  entries:
    - kind: added
      summary: "webui 内一键升级 + 重启 + 回滚（ADR 0002）（#51）"
      pr_refs: [51]
      detail: |
        Studio 不再需要 CLI `git pull` + 重启，直接在 Settings → 系统 →
        版本卡片点更新即可：

        - `git fetch` + `git reset --hard origin/master` 在 cli.py 启动期
          完成（避开 server 进程持有 native module 锁的问题）
        - `tmp/restart` flag + studio.sh/bat wrapper loop 触发重启
        - 训练 / 打标任务在跑时拒绝 update（precondition 校验返 422）
        - 失败自动留在原版本；`.last_version` 记录上一 commit 支持一键回滚
        - PR-D 加 installer 自检（cli.py / studio.sh / studio.bat sha256
          变化 → exit 42 → wrapper exec self）+ dev 通道 toggle
        - 详见 [`docs/adr/0002-webui-self-update.md`](docs/adr/0002-webui-self-update.md)
    - kind: changed
      summary: "Settings 系统 → 版本卡片改双通道布局 + inline preview（#52）"
      pr_refs: [52]
      detail: |
        - master / dev 并排显示；当前 channel 高亮（"你在这里"）
        - master 卡显示 release tag（v0.6.0）prominently，不再露 commit hash
        - dev 卡显示 commit 时间线；任意 commit 可点击切换（不只是 HEAD）
        - 操作不再走 dialog 模态：单击 → inline preview 面板（含 release
          notes + pre-flight 检查 + 取消/确认）
        - dev 通道 toggle 下移到卡片之后（demoted），当前在 dev 时强制
          开 + 锁定
```

### Step 4-5: validate + bump

```bash
$ python tools/bump_version.py validate
✓ 0.6.1 (2026-05-13) 2 entries
✓ 0.6.0 (2026-05-12) 5 entries
✓ ... (older versions)
validate ok

$ python tools/bump_version.py bump --version 0.6.1 --date 2026-05-13
[bump] studio/__init__.py: 0.6.0 → 0.6.1
[bump] studio/web/package.json: 0.6.0 → 0.6.1
[bump] CHANGELOG.md re-rendered from release_notes.yaml
[bump] git diff --stat:
   studio/__init__.py        | 2 +-
   studio/web/package.json   | 2 +-
   CHANGELOG.md              | 38 ++++++++++++++++++++
   release_notes.yaml        | 24 ++++++++++++

next: review changes, then:
   git add -A && git commit -m 'chore(release): 0.6.1'
   git tag v0.6.1
   git push --tags
```

## 8. 校验规则（`bump_version.py validate` 会跑）

会拒 yaml 上线的硬错误：

- `version` 不是合法 semver（`X.Y.Z` 或 `X.Y.Z-suffix`）
- `version` 重复（同一版本号出现两次）
- `version` 不单调递减（list 应当 latest 在 top）
- `date` 不是 ISO `YYYY-MM-DD`
- `entries` 为空 list
- `kind` 不在白名单（added/changed/improved/fixed/removed/deprecated/security）
- `summary` 缺失 / 空 / 长度 > 80 字符
- `summary` 含 markdown 字符（`*`、`` ` ``、`[`）—— 提示用 detail 而不是 summary
- `pr_refs` 不是 list[int]，或单个 int > 9999

只警告的 soft check（CI 不拒）：

- `summary` < 10 字符（可能写太简短）
- `detail` < 20 字符（这种情况建议把 detail 内容合并到 summary）
- 同 `version` 块里有 ≥ 10 个 entries（可能没好好分类整理）
- 一个 PR 出现在多条 entry 的 `pr_refs` 里超过 3 次（可能拆得太细）

## 9. 常见 anti-pattern

| ❌ | 为什么 | ✓ |
| --- | --- | --- |
| `summary: "依赖"` | 一个标签词，用户看不懂改了啥 | `summary: "升级 nvidia-ml-py 0.6.0 → 12.0.3，替代已停维护的 pynvml"` |
| `summary: "**bold**"` | summary 不允许 markdown | 把粗体去掉，强调放到 detail 里 |
| `pr_refs: ["18", "34"]` | 必须是 int 不是 str | `pr_refs: [18, 34]` |
| `kind: refactor` | 不在白名单；refactor 通常用户不感知 | 跳过这条 PR，或者用 `changed` / `improved` |
| `summary: "新增 LLM tagger（#18）。修复 Danbooru 403（#41）。"` | 两个独立改动塞一行 | 拆成两条 entry |
| `detail: "useEffect deps 用 [task] 对象引用，React 浅 clone 触发重拉..."` | detail 是用户 surface（CHANGELOG / Release body），不应露 React 内部实现 | `detail: "关联配置 tab 训练运行期间每 2 秒重拉，浏览器卡顿..."` |
| `detail: "ADR 0003 PR-A 把 utils/ 搬到根后 Path(__file__).parent.parent 少回溯一层..."` | 实现路径细节属于 commit message / PR | `detail: "v0.10.0 引入的路径错算让训练找不到 JSON caption 工具文件..."` |
| 写完 yaml 后直接 commit | 没跑 `bump_version.py validate` | 先校验，错了 CI 也会拒 |

## 10. 维护这份文档

如果某个规则不合理 / 实际跑下来 agent 老写错 / 团队约定变了 → 改这份
文档而不是绕开它。改完同步更新 `bump_version.py` 的 validate 逻辑保持
"文档说啥，工具拒啥"一致。
