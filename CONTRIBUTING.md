# 贡献指南

适用对象：
- **Contributor** — 加新功能 / 修 bug 想 PR 进来
- **Maintainer** — 决定合并 / 发版
- **AI agent**（Claude Code / 类似）— 帮上面任一角色干活时

---

## 30 秒概览

```
contributor 切 feature branch from dev
    ↓ 提多个 commit（noise OK，开发用）
PR 到 dev：squash 合并（noise 压成 1 个 PR-title commit）
    ↓ dev 累积一段时间
maintainer 攒到一组功能 → bump version → PR dev → master：merge commit（保留每 feature 颗粒度）
    ↓
打 git tag v0.X.0 → GitHub Release
```

**为什么这套**：feature branch 上的 noise commit 没价值；master 上每个 commit = 一个完整 feature；release 边界靠 git tag 标，不靠 squash。

---

## 分支策略

| 分支 | 角色 | 谁能 push |
|---|---|---|
| `master` | release 分支，永远 stable | **没人直接 push**，只接 dev 的 release PR |
| `dev` | integration 分支，长期存在 | 默认只接 contributor PR（squash 合并）；**例外**：maintainer 的小型 chore（docs / typo / 配置 / `.gitignore` 等）可直接 push |
| `feat/<topic>` `fix/<topic>` `refactor/<topic>` `docs/<topic>` `chore/<topic>` | contributor 的临时分支 | 自己 push，从 `dev` 切出 |

切分支：

```bash
git checkout dev
git pull origin dev
git checkout -b feat/cool-thing
```

---

## Commit message 约定

格式参考 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)：

```
type(scope): subject

可选的多行 body，解释 why（不解释 what — 代码就是 what）。

可选 footer（如 BREAKING CHANGE / Co-Authored-By）。
```

**type**：`feat` / `fix` / `refactor` / `docs` / `chore` / `test` / `perf` / `style`

**scope**（可选）：模块名，`train` / `ui` / `web` / `tagger` / `setup` / `release` 等

示例：

```
feat(train): 加 attention_backend 三选一字段
fix(ui): SchemaForm path 字段渲染错位
refactor: 仓库目录重组 — runtime/ 收纳 anima_* 运行时核心
docs: 重构 docs/ 为 user-guide / architecture / adr 三块
chore(release): v0.5.0 — 版本控制系统 + bump 0.4 → 0.5.0
```

**PR 标题** = 最终 squash 后落到 dev 的 commit message，所以**写 PR 标题要按上面格式**。

---

## PR 流程（Contributor）

### 1. 开始之前

- 确认 issue / 想做的事跟 maintainer 对齐过（避免做了不收）
- 跑一遍现有测试通过：`python -m studio test`

### 2. 开发

- 在自己的 `feat/<topic>` 分支上自由提 commit（`再调一下` `lint 修了` 这种 noise 都可以，最后会被 squash 掉）
- **小步走**：一个 PR = 一个完整 unit of work，别把 3 件不相关的事塞一个 PR
- 改了功能要补 / 改对应测试

### 3. 提交前自检

本地**不要求跑全量**（Windows 上全量 pytest 十几分钟；CI 会在 PR 上自动跑全量兜底），只跑两类：

```bash
# (1) 相关测试：改了哪个模块就跑对应 tests/test_<module>*.py
python -m pytest tests/test_xxx.py -q

# (2) 横切安全网（秒级，改后端必带——专抓"改 A 坏 B"）：
python -m pytest tests/test_route_snapshot.py tests/test_studio_configs.py -q

# 改了前端：
cd studio/web
npx vitest run                 # 前端全量也只要 ~20s
npm run lint && npx tsc --noEmit
```

想本地全量也可以：`python -m studio test`。

**测试卫生三条**（违反的测试在别人机器 / CI 上会假红，等于拆掉安全网）：

1. **不依赖机器状态**：secrets.json / 环境变量 / 已安装的可选包 / 网络都要
   monkeypatch 掉。反例教训：`_get_download_source()` 读真实 secrets，配了
   ModelScope 的机器上测试常年红。
2. **不依赖平台分支**：代码里有 `sys.platform` / GPU 检测分支的，测试要么
   mock 掉分支入口，要么显式参数化两个平台路径。CI 是 Linux 无 GPU，本地是
   Windows 有 GPU——两边都得绿。
3. **flake 即修，不许习惯**：超时类断言要留并发余量（CI runner 2 核且邻居
   吵）。一旦大家习惯"红的是 flake，重跑就行"，门禁就废了。

### 4. 开 PR

- **Base 分支选 `dev`**（不是 master）
- **PR 标题用 conventional commits 格式** —— 这就是 squash 后的 commit message
- **PR 描述**至少含：
  - 背景 / 动机（为什么做）
  - 改动概要（做了什么）
  - 测试方式（手测过 / 自动测试覆盖）
  - 截图（如果是 UI 改动）

### 5. Review

- Maintainer review，可能让你改
- 改完不需要重开 PR，在原 branch 继续 push 就行
- **CI 必须全绿才能合并**（`.github/workflows/test.yml`：全量 pytest +
  vitest + lint + tsc，PR 推上来自动跑，几分钟出结果）。CI 红了先看是不是
  自己改挂的；确认是 flake / 既有问题就修它或单独开 issue，**不要默默重跑
  到绿为止**

### 6. 合并

- Maintainer 用 **Squash and merge**（GitHub 默认）
- 你的 noise commits 被压成 1 个 PR-title commit 进 dev
- 你的 feature branch 可以删了

---

## 移植 / 参考外部代码的规矩

任何 PR 如果**移植、改写或实质参考**了外部仓库的代码（哪怕只是一个函数、一段公式实现），必须：

1. **PR 描述里声明来源**：上游仓库 URL + 许可证 + 大致对应的文件/类/函数。
   「参考了思路」和「搬了代码」都要写清楚是哪种——判断标准：如果对方仓库删库后
   你写不出一样的代码，那就是搬了代码。
2. **保留 / 补上版权声明**：MIT / BSD / Apache 等许可都要求在副本中保留版权与许可声明。
   派生文件加文件头注明来源 + Copyright + 许可证，并在
   [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 增补对应条目。
3. **确认许可兼容**：本仓库整体 GPL-3.0。MIT / BSD / Apache-2.0 → 可以并入；
   GPL-3.0 → 可以；AGPL / 专有 / 无 LICENSE 的仓库 → **不能搬**，没有 LICENSE
   文件 ≠ 随便用，恰恰相反默认保留所有权利。拿不准先问 maintainer。
4. **算法实现注明研究归属**：基于论文自实现的，docstring 引论文（参考
   `utils/optimizer_utils.py` 的 Lion 写法）；对照过 reference 实现的，注明
   「对照校对，未复制代码」或如实写派生关系。

**为什么这么严**：开源社区里"用了不说"比"用了"的杀伤力大得多——许可合规是底线，
显眼的 credit 才是对上游作者的尊重，也是项目自己的信誉。漏写出处事后被发现，
解释成本远高于 PR 时写一行来源。

Maintainer review 时会问"这段是不是从哪来的"——主动写比被问出来好。

---

## Release 流程（Maintainer）

### 1. 决定版本号

dev 攒到一组改动后，**看最重的改动**决定 bump 哪一位：

| dev 上的改动 | bump | 例 |
|---|---|---|
| 只有 bug fix / 内部 refactor / docs / 配置 | **PATCH** | v0.5.0 → v0.5.1 |
| 含**任何**新功能 / schema 改 / API 改 / 行为变更 | **MINOR** | v0.5.0 → v0.6.0 |

> 0.x 阶段把 SemVer 收紧用：MINOR 视作「可能不向后兼容」；PATCH **严格不动**既有 config / API / 行为。
>
> 「feature 少」很正常——大多数 release 是 PATCH，攒到一个 feat 时再 bump MINOR。

**对应 release_notes.yaml 的 `kind`**：本周期 entries 全是 `fixed` / `improved` /
`removed` / `deprecated` → PATCH；含任一 `added` / `changed` → MINOR；纯
`security` hotfix → PATCH（紧急 patch 即使含 `changed` 也走 PATCH，让用户
敢升）。

Pre-release 用 `-rc1` / `-beta1` 后缀（v0.6.0-rc1）。

### 2. 在 `dev` 上准备 release

```bash
git checkout dev
git pull origin dev
```

**写 release notes（结构化 yaml，agent 友好）**

source of truth 是 [`release_notes.yaml`](release_notes.yaml)；`CHANGELOG.md`
由工具从 yaml 派生，**不要手改** CHANGELOG（下次 render 会被覆盖）。

按 [`docs/release-notes-spec.md`](docs/release-notes-spec.md) 在 yaml 顶部
插入新版本 block，每个 user-facing PR 一条 entry（纯 chore / docs / 内部
refactor PR 跳过）：

```yaml
- version: "0.X.0"
  date: "YYYY-MM-DD"
  summary: "一句话总览"
  entries:
    - kind: added            # added/changed/improved/fixed/removed/deprecated/security
      summary: "user-facing 一行，≤ 80 字符，结尾带 PR 号（#NN）"
      pr_refs: [NN]
      detail: |              # 可选，markdown 多行
        markdown 多行细节
```

拉本周期所有 merge 的 PR（命令样例）：

```bash
gh pr list --state merged --base dev \
  --search 'merged:>=<上次 release 日期>' \
  --json number,title,body,labels,author,mergedAt --limit 100
```

详细 do/don't / kind 分类规则 / good vs bad 例子见 release-notes-spec.md。

**`bump_version.py` 一键同步版本号 + 重写 CHANGELOG.md**

```bash
python tools/bump_version.py validate                 # 先校验 yaml schema
python tools/bump_version.py bump --version 0.X.0     # 同步版本号 + 重写 CHANGELOG.md
```

工具自动改：

1. `studio/__init__.py` — `__version__ = "0.X.0"`
2. `studio/web/package.json` — `"version": "0.X.0"`
3. `CHANGELOG.md` — 从 yaml 派生

`bump` 跑前自动跑 `validate`，schema 错（kind 不在白名单 / summary > 80 字 /
版本顺序错 / pr_refs 不是 int / etc.）会直接拒。

**还需要手动改一处**：`README.md` 顶部 shields.io badge URL + 「## 版本」
段「当前版本 **0.X.0**」（README 风格用户偏好强，工具暂不动）。

> 前端 Sidebar 的版本号从 `/api/health` 拉，**不要去 Sidebar.tsx 硬编码**。

跑 grep 兜底找漏（架构文档示意图、ADR 引用等）：

```bash
grep -rn "<旧版本号，例 0.5.0>" --include="*.md" --include="*.json" \
  --include="*.py" --exclude-dir=node_modules --exclude-dir=venv
```

输出里属于「历史记录」的保留不动（CHANGELOG 历史段、ADR 引用上次发版
的事实、release_notes.yaml 历史 entries）；属于「当前展示」的更新到新版本
（README badge / docs 示意图 / 任何描述「当前版本」的句子）。

**审阅 README + docs 全量内容**（不只是版本号 — 内容本身是否还对得上）：

版本号是「指针」，指针对了内容也可能漂；release 是把内容跟当前实际功能
重新对齐的窗口。本周期改了的地方，文档里相应的描述 / 示例 / 截图 / 数据
模型 / 字段名 / UI 措辞都要顺手核对。

- **README.md 通读一遍**：项目概览 / 特性列表 / 截图 / 流水线 7 步 / Studio
  Web 工作台描述 / 快速开始 / 系统先决条件 / 项目结构 / 致谢，每一段对照
  本周期 PR 看是否还成立（功能被删 / 改名 / 行为变了 / 新增大功能没提到）
- **docs/ 通读**：
  - `docs/user-guide/` — 用户向（标签格式 / 训练 tips / caption 格式 等）
    本周期 schema / 默认值 / UI 流程改了的话，对应章节要同步
  - `docs/architecture/` — 开发者向架构总览，**示意图里的版本号 / 模块路径
    / 字段名**对照新代码看是否漂了（grep 抓不到结构性漂移）
  - `docs/adr/` — 历史决策记录，**保留原状不改**（ADR 是 point-in-time 快照，
    引用「当时的版本号」属于历史记录）。除非本周期落地了某个 ADR，可在那条
    ADR 顶部 status 从 Proposed → Accepted
- **截图 / GIF**：UI 大改的话（本周期如 Settings / 训练页 / 自更新面板）
  README / docs 里的截图可能已陈旧，必要时重拍

发现漂移**就在本 release commit 里一起改**，不要拖到下次。文档跟版本号
脱钩的程度往往比想象的大。

提一个 commit：

```bash
git add release_notes.yaml CHANGELOG.md \
        studio/__init__.py studio/web/package.json README.md
git commit -m "chore(release): v0.X.0"
git push origin dev
```

**Release commit body 默认空 / 极简**。release commit 是机械的 version bump
+ render，不是 fix 真正发生的地方。fix 的根因 / 修法已经在原 fix commit 的
message + PR description 里；CHANGELOG.md / GitHub Release body 承担用户视角
描述。Release commit body 重复这些信息反而模糊 git log archeology 的层次
（commit message 给工程师、release notes 给用户、各司其职）。例：v0.9.1 /
v0.10.0 release commit body 都是空。

### 3. 开 Release PR：dev → master

- **Base = master，compare = dev**
- **标题**：`feat: v0.X.0 — <主题>`
- **描述**：复制 CHANGELOG 的对应段（让 PR 页面就能看到完整改动）

### 4. 合并

- 选 **Create a merge commit**（**不要 squash**）—— 保留 dev 上每个 feature 的 commit 颗粒度
- Merge commit message 默认 `Merge pull request #N from .../dev`，extended description 简短写 `Version: v0.X.0 — 主题` 即可

### 5. 打 tag

```bash
git checkout master
git pull origin master
git tag -a v0.X.0 -m "v0.X.0 — <主题>"
git push origin v0.X.0
```

### 6. 发 GitHub Release

- 去 `https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/releases`
- 找到刚 push 的 tag → **Create release from tag**
- **Title**：`vX.Y.Z`（纯版本号，**不**带主题 —— 仓库历史 release title 都是
  这风格：v0.10.0 / v0.9.1 / v0.8.3 ...。tag 的 message 才带主题，见 §5）
- **Body**：复制 [`CHANGELOG.md`](CHANGELOG.md) 对应段
- 勾选 **Set as the latest release**
- Publish

---

---

## Hotfix 加急路径（绕过 dev）

**只在等不了下次 release 时用**——线上炸了 / 安全问题 / dev 上有不该跟着发的功能但 bug 必须立即修。平时的 bug fix 走 dev 正常流程，**不算 hotfix**。

```bash
# 1. 从 master 切（不是从 dev）
git checkout master && git pull origin master
git checkout -b hotfix/<topic>

# 2. 修 + 测 + commit + 开 PR
#    - Base 选 master（不是 dev）
#    - 标题：fix(scope): ... 描述紧急 bug
#    - 描述说清楚为什么不能等下次 release

# 3. Squash and merge 进 master

# 4. 按上面 release 流程的第 2 / 5 / 6 步：
#    - bump PATCH（__init__.py + package.json + CHANGELOG.md + README.md）
#    - 直接在 master 上 commit version bump（也算 hotfix 例外）
#    - 打 v0.X.Y tag + 发 GitHub Release

# 5. 必须 sync 回 dev！否则 dev 缺这个修复，下次 release 会回归
git checkout dev
git merge master
git push origin dev
```

第 5 步是关键——很容易忘。Hotfix 之后 dev 必须包含这个修复，不然下次 release（dev → master）的 PR 会显示「这个 bug 又回来了」。

---

## 版本号规则

按 [SemVer](https://semver.org/lang/zh-CN/)：`MAJOR.MINOR.PATCH`

| 阶段 | 规则 |
|---|---|
| **0.x**（当前）| MINOR 视为破坏性升级（schema 改 / API 改 / 大重构都进 MINOR）；PATCH 留给 hotfix |
| **1.0+** | PATCH = bug fix / 内部重构；MINOR = 向后兼容新功能；MAJOR = 破坏性改动 |

唯一来源：`studio/__init__.py:__version__`。FastAPI app version 从这派生（`/api/health` 暴露），前端 Sidebar fetch `/api/health` 拿。

---

## 文档

| 在哪 | 内容 |
|---|---|
| [README.md](README.md) | 项目概览 + 快速开始 + 项目结构 |
| [CHANGELOG.md](CHANGELOG.md) | 版本变更（Keep a Changelog 格式） |
| [docs/README.md](docs/README.md) | 文档总入口 |
| [docs/user-guide/](docs/user-guide/) | 用户向：标签格式 / 训练 tips / 正则集 / caption 格式 |
| [docs/architecture/](docs/architecture/) | 开发者向：跨模块架构总览 |
| [docs/adr/](docs/adr/) | 架构决策记录（ADR） |

**改了功能**：同步更新对应文档。

**架构层面的「我们选 X 而不选 Y」决定**：写一份新 ADR（`docs/adr/NNNN-title.md`），模板见 [docs/adr/README.md](docs/adr/README.md)。例：[ADR 0001](docs/adr/0001-lokr-via-lycoris-lora.md) 决定 LoKr 走 lycoris-lora 而不切 sd-scripts。

---

## 仓库目录约定

```
modeling/    模型架构定义（vendored diffusion-pipe 子集 + Anima 包装；按路径动态加载）
models/      下载的模型权重 / tokenizer 数据落点（gitignored、按需创建，见 .gitignore）
utils/       lycoris_adapter / optimizer / 训练共享工具
runtime/     Anima 运行时核心（独立进程；Studio subprocess 拉起 / 也可单独 CLI）
studio/      Web 后端 + 前端
tools/       用户 CLI + setup helper（含 dev bench）
docs/        三块：user-guide / architecture / adr
tests/       后端 pytest + 前端 vitest（前端测试在 studio/web/src/**/*.test.ts*）
```

依赖方向单向：`modeling → utils → runtime → studio → tools`。不要反向 import。
（`models/` 是纯数据目录，不含代码、不在依赖链里——模型架构代码在 `modeling/`。）

---

## 给 AI agent 的额外说明

**AI agent 接到本仓库任务时必读** [`docs/AGENTS.md`](docs/AGENTS.md) — 代码质量、一致性、可维护性、AI 协作的完整公约（含开工前对齐协议、单一权威源清单、陷阱清单、PR 自检）。本节是简版速查。

如果你是 Claude Code / Cursor / 类似 agent 在帮 maintainer 或 contributor 干活：

**遵守的**：

- 一个 PR = 一个完整 unit of work，别把不相关的改动塞一起
- 改 version 时**四处必须同步**：`studio/__init__.py` + `studio/web/package.json` + `CHANGELOG.md` 顶部新段 + `README.md`（顶部 badge + 「## 版本」段当前版本句）
- 修 bug 不要顺手 refactor 别的代码（除非 maintainer 明确要求）
- 测试覆盖：bug fix → 加 regression test；feat → 新功能 test
- 中文 commit message 和文档没问题（仓库已有惯例），但 conventional commits 的 `type(scope):` 前缀用英文
- 最终 commit message / PR description 走 maintainer 给你说的格式

**不要的**：

- 不要直接 push `master`（不管什么情况）
- 不要打 git tag / 创建 GitHub Release（这是 maintainer 决定，授权了再做）
- 不要 squash / rebase / force-push 已经 push 到 origin 的分支（除非 maintainer 明确要求）
- 不要绕过测试（`--no-verify` / 跳测试 / disable lint），先修根因
- 不要为了「漂亮」改无关代码 / 改风格 / 改命名（无关改动会让 review 难做）

**优先 reuse 现存代码**：

- 写新组件前 grep 仓库看有没有现成的（`useProjectCtx` / `Toast` / `PathPicker` / `SchemaForm` 等都已存在）
- 写新 CLI 前看 `tools/` 有没有重叠
- 写新测试前看 `tests/conftest.py` 的 fixture
