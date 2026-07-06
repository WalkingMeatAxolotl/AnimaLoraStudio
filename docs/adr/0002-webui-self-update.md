# 0002 — Webui 内自更新（flag + shell wrapper loop）

**状态**：Accepted（已落地：`studio/services/runtime/updater.py` + Settings 版本卡）
**日期**：2026-05-12
**决策者**：@WalkingMeatAxolotl

## 背景

当前用户升级必须走 CLI：`git pull` + 重启 `studio.sh` / `studio.bat`。痛点：

- 技术不熟的用户卡在 git 操作
- 跨 minor 版本 `requirements.txt` 变化要手动 `pip install`；CHANGELOG 写在那但不直观
- 我们这一类长任务工具的特殊点：训练任务跑到一半时升级会丢进度，没有暂停-恢复的话用户不敢点更新

讨论顺序：

1. 暂停 + resume 训练（[Ctrl+C 现成机制](../../runtime/anima_train.py)，看 `signal_handler` / `signal.SIGINT`；PR-A 之后会搬到 `runtime/training/phases/resume.py`。差 supervisor 发对信号 + UI 接线）
2. webui 一键更新（本 ADR）

两者配套出"无痛升级 + 长任务护盘" 完整故事。

## 关键现状盘点

仓库已经隐式具备"启动期 bootstrap"层，只是不循环、不接受外部触发：

- `studio.sh` / `studio.bat` shell wrapper（一次性 venv / 系统级 setup，单次调用 `python -m studio`）
- `studio/cli.py:cmd_run` Python 启动器：`_ensure_python_deps` / `_web_dist_is_stale` 触发前端 rebuild / `_apply_pending_install` / `_check_torch_cuda` / `_check_onnxruntime`，最后 `subprocess.call(server)` 阻塞
- `studio/services/pending_install.py` 实现"延迟到下次启动执行的 pip 操作"（为 torch 重装设计）
- `cli.py:_web_dist_is_stale` 已实现 git HEAD 比对 + mtime 比对的 stale 检测
- `studio.sh:135-148` 已实现 `requirements.txt` sha256 marker 比对 → 增量 `pip install -r`

## 业界调研

详细见对话记录（2026-05-12）。三大流派：

| 流派 | 代表 | 重启机制 | 优劣 |
| --- | --- | --- | --- |
| A. flag + shell wrapper loop | A1111, SwarmUI | 服务写 `tmp/restart` + `os._exit(0)`，外层 wrapper 检测后跳回入口 | 最稳，跨平台一致；要求用 wrapper 启动 |
| B. `os.execv` 进程内重启 | ComfyUI-Manager, SD.Next | Python 自己 `os.execv(sys.executable, ['python'] + sys.argv)` 原地变身 | 不依赖 wrapper；Linux+CUDA VRAM 不释放（issue #576）、Windows 含空格路径找不到 sys.executable |
| C. 外部启动器代管 | InvokeAI Launcher, StabilityMatrix, Pinokio | 桌面应用（Electron / Avalonia / etc）spawn/kill 子进程 | 最干净，Python 零自更新代码；要做独立桌面应用 |

**业界共识**：

- 没人热替换代码或 native module（torch / onnxruntime dlopen 后锁文件，Windows 直接装不上）
- 依赖更新永远走两步：pull → 重启 → entry 比对 → 再 pip
- 几乎没人保护运行中任务，按下 update 直接强杀
- ComfyUI-Manager 的 `install-scripts.txt`（LAZY 脚本）和我们的 `pending_install.py` 是同一个 pattern
- oobabooga 的 `update_wizard` pull 后检查 installer 自身 hash，变了就 abort 让用户重跑 wizard —— 防"半成品启动器"，值得借鉴

## 决策

采用 **A 流派（flag + shell wrapper loop）为主，B 流派（execv）为 fallback**。差异化点 ——
**强制约束：所有 task 必须 paused / done / canceled 才允许 update**。

### 配套决策

| 维度 | 决策 |
| --- | --- |
| Scope | **仅 git clone 部署用户**；未来 PyPI / Docker 走另外的 update 路径，单独 ADR |
| 自动检查 | **默认开**；启动期**异步**触发（不阻塞冷启动），结果写 `studio_data/.update_cache` (TTL 24h)；后续启动 24h 内复用 cache。失败静默 |
| 检查目标 | **永远只检查 master**（即使 toggle 开了 / 当前本地是 dev） |
| 更新通知 | **Topbar 加 update badge**（小红点），仅 master 有新版时显示；点击跳 Settings → 系统 section |
| 通道 | **默认仅 master**；Settings 高级 toggle「显示开发版更新」开启后解锁"手动检查 dev" + "更新到 dev"按钮；自动检查行为不变 |
| Telemetry | **不上报**。失败 UI 提供"复制 `.update_log` 到 GitHub Issue 模板"按钮代替 |

## 理由

- 仓库 `studio.sh`/`studio.bat` + `cli.py` + `server.py` 三层结构正好对齐 A1111；改动最小
- 现有 `pending_install` / `_STALE` / `_web_dist_is_stale` 三大块本来就是为"启动期 bootstrap"设计的，循环复用即可
- B 的 VRAM 不释放 / 含空格路径 bug 对训练工具特别致命（显存泄漏 = 下一轮 OOM）；保留 fallback 给开发者用
- C 需要做桌面应用，超出当前项目 scope
- 任务保护是行业空白点；训练任务损失代价远高于出图，必做

## 架构

### 进程层

```
studio.sh / studio.bat                    ← OS 级 setup（一次性）
    while true; do                        ← 新增 loop
        python -m studio                  ← cli.py 主循环（基本不变）
        rc=$?
        if [[ ! -f tmp/restart ]]; then break; fi
        rm tmp/restart
        if [[ $rc -eq 42 ]]; then         ← installer 自身变化的退出码
            echo "[studio] launcher updated, re-exec wrapper"
            exec "$0" "$@"                ← wrapper 本身也重新执行
        fi
    done
    exit $rc
```

`studio.bat` 同等改造（goto 语法）。

### Flag 协议

| 文件 | 含义 | 作者 → 读者 |
| --- | --- | --- |
| `tmp/restart` | 需要重启 | server → wrapper |
| `studio_data/.update_pending` | 启动期要 git pull，内容是 target ref（默认 `origin/master`） | server → cli.py |
| `studio_data/.last_version` | 上一版 HEAD（rollback 用） | cli.py（每次 pull 前写） |
| `studio_data/.update_log` | 最近一次 update 的日志（pull / pip / 错误） | cli.py |

两层 flag 解耦的原因：`tmp/restart` 只管"重启"，`.update_pending` 管"重启后做什么"。`tmp/` 用 git 忽略，重启即焚；`studio_data/` 是持久化的，保留历史。

### Server 端点

```
GET  /api/system/version              当前 HEAD / tag / commit 时间 / dirty 状态 / current_branch
GET  /api/system/update_check?channel=master|dev
                                       git fetch + 比对，返回 {has_update, latest_tag, latest_sha,
                                       commits_ahead, changelog_md}
                                       默认 channel=master；channel=dev 仅当 settings.show_dev_channel=true
                                       才接受请求（前端 toggle 关时连查询都不发）
POST /api/system/update               body: {target: "origin/master" | "origin/dev" | "<sha>"}
                                       前置 has_running_task() 校验 → 写两 flag → 触发 graceful shutdown
POST /api/system/restart              仅写 tmp/restart（不 pull）
POST /api/system/rollback             以 .last_version 内容为 target 重新 update
GET  /api/system/update_log           tail .update_log
```

每个写操作端点都先调 `supervisor.has_running_task()`，true 则返回 422 + 当前 running task 列表。

**自动检查路径**：cli.py 启动期异步线程 → `updater.check(channel="master")` → 结果写 `studio_data/.update_cache` (含 `checked_at` 时间戳)。下次启动若 cache 未过期（mtime < 24h）则跳过 fetch。Settings 页"手动重新检查"覆盖 cache。

**Dev 通道路径**：toggle 关闭时 UI 完全不暴露 dev 入口；toggle 开后 Settings 加一个"手动检查 dev"按钮（不自动触发，避免开发者 flush master 信号）。点击后单次 `update_check?channel=dev`，结果**不写入** cache（不影响 master 的 update_available 状态）。Topbar badge **仅由 master cache 驱动**，dev 永远不亮 badge。

### Cli.py 改动

`cmd_run` 主流程：

```python
def cmd_run(args):
    while True:
        rc = _ensure_python_deps()
        if rc != 0: return rc

        # 新增：检测 update pending，先 git pull 再走 bootstrap
        _apply_update_pending()             # 读 .update_pending, git pull, 写 .last_version, 清 flag

        # 现有：build / pending_install / torch / onnx
        ...

        # 新增：installer 自身变化检测（学 oobabooga）
        if _installer_changed_since_start():
            print("[studio] launcher 代码已更新，请用 wrapper 重启")
            return 42                       # 特殊退出码，wrapper 看到会 exec 自己

        rc = subprocess.call([sys.executable, "-m", "studio.server", ...])

        # 现有：阻塞结束后判断是否要重启
        flag = REPO_ROOT / "tmp" / "restart"
        if not flag.exists():
            return rc
        flag.unlink()
        # loop continue
```

`_installer_changed_since_start()`：进程启动期记录 `cli.py` + `studio.sh` + `studio.bat` 的 sha256，server 退出后再算，变化即 true。**必须在 wrapper loop 之外退出**（exit 42），否则跑的还是旧 wrapper / 旧 cli.py。

### Updater 模块

`studio/services/updater.py`：

```python
def check() -> UpdateInfo: ...
    # git fetch origin
    # git rev-list HEAD..origin/master --count
    # 拉 latest tag + changelog 段落

def request_update(target: str = "origin/master") -> None:
    # 1. precondition: 无 running task
    # 2. precondition: git status --porcelain 干净
    # 3. write .update_pending = target
    # 4. write tmp/restart
    # 5. trigger uvicorn graceful shutdown (server.should_exit = True)

def apply_pending() -> ApplyResult:                # cli.py 启动期调
    if not has_pending(): return
    target = read_pending()
    write(LAST_VERSION, current_head())

    rc = subprocess.call(["git", "pull", "--ff-only", "origin", target])
    if rc != 0: return failed("git pull failed")

    if requirements_sha_changed():
        new_native = diff_requirements_for_native_packages()  # torch / onnx
        for pkg in new_native:
            pending_install.queue(pkg)                        # 复用现有机制
        # pure python deps 直接装
        subprocess.call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    if web_package_json_changed():
        subprocess.call(["npm", "install"], cwd=WEB_DIR)

    clear_pending()
    return ok()

def rollback() -> None:
    target = read(LAST_VERSION)
    request_update(target)                          # 走相同的 update 流程
```

`apply_pending` 在 `cli.py:cmd_run` 主流程顶端调，**先于** `_ensure_python_deps` 之外的所有 bootstrap 步（因为新版可能改了 deps 列表）。

### UI

Settings → 新加「系统」section（或独立 tab，跟 #36 重排后的 6 tab 结构契合）。

**默认状态**（toggle 关，绝大多数用户看到的样子）：

```
┌─────────────────────────────────────────────────────┐
│ 版本                                                │
│   当前  v0.6.0  (commit 879378e · 2 days ago)       │
│   最新  v0.6.1  (commit abc1234 · 3h ago)           │
│   ↑ 5 commits ahead                                 │
│                                                     │
│   [查看 CHANGELOG]  [立即更新]  [回滚到 v0.6.0]    │
│                                                     │
│   自动检查更新  ☑   每 24 小时                      │
│                                                     │
│   ─────────────────────────────                     │
│   ▸ 高级设置                                        │
└─────────────────────────────────────────────────────┘
```

**Toggle 开启后**（点开"高级设置"折叠 → 勾选"显示开发版更新"）：

```
┌─────────────────────────────────────────────────────┐
│ 版本                                                │
│   当前  v0.6.0  (commit 879378e · 2 days ago)       │
│   最新  v0.6.1  (commit abc1234 · 3h ago)           │
│   ↑ 5 commits ahead                                 │
│                                                     │
│   [查看 CHANGELOG]  [立即更新 (稳定)]               │
│   [回滚到 v0.6.0]                                   │
│                                                     │
│   自动检查更新  ☑   每 24 小时（仅检查稳定通道）    │
│                                                     │
│   ─────────────────────────────                     │
│   ▾ 高级设置                                        │
│   显示开发版更新  ☑                                 │
│                                                     │
│   开发版通道（dev，用于开发 / 测试）                │
│   [手动检查 dev]                                    │
│   ─ 检查后显示 ─                                    │
│   dev  commit ef56789                               │
│   ↑ 12 commits ahead of master (2h ago)             │
│   [查看 dev diff]  [更新到 dev]                     │
│   ⚠️ dev 未经发布测试，可能引入崩溃 / 训练异常       │
└─────────────────────────────────────────────────────┘
```

**「立即更新」点击流程**（master 通道）：

1. 前端检查 `/api/queue/running` —— 有 running task 则禁用按钮 + tooltip 显示当前 task
2. 弹模态："将关闭并重启 studio。预计 1-3 分钟。期间 webui 不可访问。"
3. POST `/api/system/update` `{target: "origin/master"}`，server 写 flag 后 200 响应
4. 前端进入"更新中"状态卡片：阶段（git pull → pip → 重启）+ 自动轮询 `/api/health`
5. server 起来后 reconnect → toast "已更新到 v0.6.1" + 自动刷新页面
6. 失败时：拉 `/api/system/update_log` 显示 + 提示"请用 shell wrapper 重启 / 联系开发者"

**「更新到 dev」点击流程**：

1. 同样的 running task 检查
2. 弹模态额外加红色警告："这是开发版，可能不稳定。回滚需要单独操作。"
3. POST `/api/system/update` `{target: "origin/dev"}`，其余路径同上
4. 成功后 toast "已更新到 dev (commit ef56789)" —— 不显示 tag（dev 没有）

**Topbar badge**：

- 仅 master cache `has_update=true` 时显示小红点
- Dev 即使有新 commit 也不亮 badge（避免持续骚扰开发者）
- 点击 badge 直接跳 Settings → 系统 section

### Pending install 复用

新版 `requirements.txt` 含 native module 改版（torch / onnxruntime）时，**不在 update 当下 install**（旧 venv 进程仍 import 着），写到 `pending_install` 队列。下次启动期 `_apply_pending_install` 阶段执行 —— 那时新 venv 进程 freshly 起，没有 torch 已 loaded 的问题。

这条逻辑已经现成（[`studio/services/pending_install.py`](../../studio/services/pending_install.py)），不需要新代码。

## 状态机

```
Idle ──[check]──> HasUpdate ──[update click]──> CheckPrecondition
                                                    │
                              has running task ─────┤ → 422 "暂停所有任务后再试"
                                                    │
                              local dirty ──────────┤ → 422 "本地有未提交修改"
                                                    │
                                              all clean
                                                    ↓
                                          WriteFlag + ServerShutdown
                                                    ↓
                                      CliLoop._apply_update_pending
                                                    ↓
                                            git pull, write .last_version
                                                    ↓
                                  requirements 改？──── yes ──> pip install / pending_install.queue
                                                    │
                                                    ↓
                                  installer 改？──── yes ──> exit 42 → wrapper exec self
                                                    │
                                                    no
                                                    ↓
                                       subprocess.call(server) → Idle
```

## 失败模式

| 故障 | 影响 | 兜底 |
| --- | --- | --- |
| `git pull` 冲突（本地未提交） | 不更新，server 重新起 | precondition 检查阶段拒绝；UI 显示 dirty 文件列表 |
| `git pull` 网络失败 | 同上 | `.update_log` 记录，UI 显示 |
| `pip install` 失败 | server 可能起不来 | `.last_version` 自动 rollback + UI 提示 |
| Native module 锁定 | 改装 venv 时 .pyd 占用 | 走 `pending_install` 延迟到 fresh 进程；新版 entry 检测不到必装件则 fail-fast |
| 用户 direct `python -m studio` 启动 | restart 不工作（执行 `tmp/restart` 创建但无 wrapper loop） | cli.py 检测 `parent_pid` 是不是 studio.sh / studio.bat，否则 update 端点返回 400 提示用 wrapper |
| 更新到坏版本（server 起不来） | webui 一直黑屏 | wrapper 检测连续 N 次 server 启动失败后 abort + 命令行 hint 用户 `git reset --hard <last_version>` |
| 训练任务跑一半被强杀 | LoRA 进度丢失 | precondition 强制约束（必须前置完成「暂停训练」feature） |
| `os.execv` 路径含空格（Windows） | execv fallback 走不通 | cli.py 启动期检测路径含空格则禁用 execv，强制要求用 wrapper |

## 实施步骤（按 PR 拆分）

1. **PR-A：基础重启链路**（不含 git pull）
   - `tmp/restart` flag + `cli.py` loop + `studio.sh`/`studio.bat` 改 wrapper loop
   - `/api/system/restart` 端点 + Settings UI「重启 server」按钮
   - 验证：webui 重启 server，shell wrapper / direct python 两路径都通
   - 工作量 ~1 天

2. **PR-B：update 主路径**
   - `updater.py` + `/api/system/update_check` / `/update` 端点
   - `apply_pending` 在 cli.py 启动期 hook
   - Settings UI 版本 section
   - 工作量 ~1.5 天

3. **PR-C：回滚 + 日志 + 失败处理**
   - `.last_version` rollback + `/api/system/rollback`
   - `update_log` tail + UI 失败提示
   - 工作量 ~1 天

4. **PR-D：installer 自检**
   - cli.py / studio.sh / studio.bat sha256 比对
   - 退出码 42 协议 + wrapper exec self 路径
   - 工作量 ~半天

5. **PR-E：跨平台 QA**
   - Windows + Linux + macOS（如有）wrapper loop / 进程退出 / flag 文件竞态
   - 含空格路径 / 中文路径 / 异常退出 / 持续故障的 fallback
   - 工作量 ~1 天

**总 ~5 天**。前置依赖：**暂停训练 feature 必须先落地**（PR-B 的 precondition 才有意义）。

## 已决细节

讨论后确认的设计选择（与「配套决策」表对应的扩展说明）：

1. **自动检查（合并原 Q1 + Q6）**：默认开 + 异步触发 + 24h cache。Cache 写到 `studio_data/.update_cache`，TTL 用 mtime 判断。失败静默（公司网络 / 国内访问 github 不稳，不应阻塞启动）。Settings 提供"手动重新检查"覆盖 cache
2. **通知**：Topbar 加 update badge（小红点），点击跳 Settings → 系统 section。仅 master 触发 badge
3. **dev 通道**：默认隐藏在 Settings 高级设置；toggle 开后解锁"手动检查 dev" + "更新到 dev"两个按钮。自动检查仍只 fetch master，避免开发者被高频 commit 持续骚扰。Rollback 协议**必须兼容 commit hash**（dev 没有 tag）
4. **Telemetry**：不做。失败 UI 给"复制 update_log 到 GitHub Issue 模板"按钮代替
5. **Scope**：仅 git clone 用户。PyPI / Docker 路径不在本 ADR scope，未来另起 ADR

## 后果

**收益**：

- 用户不再走 CLI 即可升级
- 跟暂停 feature 配套形成完整"无痛升级"故事，差异化竞争点
- 复用现有 bootstrap / pending_install / stale_check 三大块，最大化代码复用

**新增约束**：

- 必须用 `studio.sh` / `studio.bat` 启动才能享受完整 update 流程（已经是文档推荐路径，但现在变成事实必须）
- 自动 git pull 意味着 master commit 即用户的生产代码，回归测试压力变大（建议未来引入 staging：dev → master 前内部跑一轮全 e2e）
- 训练任务暂停 feature 成为前置 hard dependency

**未来的债**：

- 大版本（0.x → 1.0 含 db schema 迁移）的 update 路径要单独设计，本 ADR 不覆盖
- 多用户场景（如果以后做 server 共享）update 触发权限要加 ACL
- update 失败 rollback 路径如果 venv 也坏了（pending_install 卸载半完成态）不工作 —— 真坏的情况只能让用户走 shell `./studio.sh --reinstall`，UI 端不补救

## 参考

- 业界调研：[A1111 restart.py](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/restart.py) / [ComfyUI-Manager manager_server.py](https://github.com/Comfy-Org/ComfyUI-Manager/blob/main/glob/manager_server.py) / [oobabooga one_click.py](https://github.com/oobabooga/text-generation-webui/blob/main/one_click.py) / [SwarmUI AdminAPI.cs](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/src/WebAPI/AdminAPI.cs)
- 仓库现有机制：[`studio/cli.py:cmd_run`](../../studio/cli.py) / [`studio/services/pending_install.py`](../../studio/services/pending_install.py) / [`studio.sh:135-148`](../../studio.sh) / [`runtime/anima_train.py`](../../runtime/anima_train.py) 的 `signal_handler`（grep `signal.SIGINT`）；ADR 0003 重构后位置改到 `runtime/training/phases/resume.py`
- 暂停训练前置 feature：单独 ADR / PR 跟进
