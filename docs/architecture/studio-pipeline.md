# Studio 架构总览

跨步骤的横切关注点：数据模型、目录布局、SQLite schema、secrets、Sidebar、SSE 事件、Tagger 抽象、Preset 关系。Studio 内部模块结构见 [`studio/README.md`](../../studio/README.md)。

---

## 1. Pipeline 流程

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
│ 创建项目 │ → │ 下载数据 │ → │ 预处理   │ → │ 筛选数据 │ → │ 标签生成 │ → │ 正则集生成 │ → │ 配置/入队 │
│ Project  │   │ Download │   │ Upscale  │   │ Curation │   │ Tagging  │   │ Reg-build  │   │ Train    │
│ 含 v1    │   │ 项目级   │   │ 项目级(可选)│ │ 版本级   │   │ 版本级   │   │ 版本级     │   │ 版本级   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └────────────┘   └──────────┘
                                                        ↓ 可循环（新建 v2 重新筛选/打标）
```

每个版本（version）独立维护 `train/` `reg/` `output/` `samples/` `monitor_state.json`；`download/` 和 `preprocess/`（v0.8 新）在项目级共享，**永远不删**（永远是「全量来源」）。预处理是项目级的可选步骤（ADR 0004），状态走单 manifest，详见 §2 物理布局。

---

## 2. 物理目录布局

```
studio_data/
├── secrets.json                          ★ 全局服务配置（gelbooru token 等）
│                                         studio_data/ 已被 .gitignore，自然安全
├── presets/                              ★ 全局预设池
│   ├── train_baseline.yaml
│   └── proj_42_baseline.yaml             从某 version 推回的预设
├── projects/{id}-{slug}/
│   ├── project.json                      title / stage / active_version_id / ts
│   ├── download/                         project 级共享，全量备份
│   │   ├── 12345.png
│   │   └── 12345.json                    Gelbooru 元数据，可选
│   ├── preprocess/                       项目级共享工作集（v0.8 / ADR 0004 + Addendum 1）
│   │   ├── manifest.json                 状态唯一真理：{images:{name:{origin,mtime,size}}}
│   │   ├── 12345.png                     单产物：origin=12345.png（1:1 同名）
│   │   ├── 67890_c0.png                  多裁剪派生：origin=67890.png
│   │   └── 67890_c1.png                  同 origin，另一裁剪框（multi-crop fan-out）
│   └── versions/
│       └── {label}/                      ★ label 用户填："baseline" / "high-lr"
│           ├── version.json              config_name / stage / note
│           ├── train/
│           │   └── 5_concept/            Kohya 风格 N_xxx
│           │       ├── 12345.png         复制自 ../../../download/
│           │       ├── 12345.txt         打标产物
│           │       └── 12345.json        分类 caption（可选）
│           ├── reg/                      ★ version 级（train 变就重生）
│           │   ├── meta.json             {generated_at, source_version, target_count, source_tags, generation_method}
│           │   └── 1_general/
│           │       ├── reg_001.png
│           │       └── reg_001.txt
│           ├── output/                   训练产物
│           │   ├── lora_step500.safetensors
│           │   ├── lora_final.safetensors
│           │   └── state_step1000.pt
│           ├── samples/
│           │   └── step500_p0.png
│           └── monitor_state.json        该 version 训练 loss/lr 曲线
```

> v0.8 起 项目 / 版本删除是直接 `rmtree`（无回收站）— 之前有过 `_trash/` 软删但无恢复 UI / 无定期清理，等同硬删却 silently 累积孤儿目录，故移除（详见 [release notes 0.8.0](../../release_notes.yaml)）。

**slug 规则**：title 转 ASCII 小写 + 连字符；冲突时加 `-2` `-3` 后缀。
**id**：自增，与 slug 一起组成目录名 `{id}-{slug}`。

---

## 3. SQLite Schema

DB 落在 `studio_data/studio.db`。Migrations 在 `studio/migrations/` 顺序应用（`PRAGMA user_version` 控制）。

```sql
CREATE TABLE projects (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    slug                TEXT UNIQUE NOT NULL,
    title               TEXT NOT NULL,
    active_version_id   INTEGER REFERENCES versions(id) ON DELETE SET NULL,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    note                TEXT
);
CREATE INDEX idx_projects_slug ON projects(slug);

CREATE TABLE versions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id           INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    label                TEXT NOT NULL,             -- baseline / high-lr / ...
    config_name          TEXT,                       -- 引用 presets/{config_name}.yaml
    -- ADR-0007: status + phase 双正交字段（v8 加；v9 删 stage）
    status               TEXT NOT NULL DEFAULT 'preparing',
                         -- preparing | training | completed | failed | canceled
    phase                TEXT NOT NULL DEFAULT 'curating',
                         -- curating | tagging | editing | regularizing | ready
                         -- 仅 status=preparing 时有业务意义
    last_failure_reason  TEXT,                       -- task=failed 时来自 task.error_msg
    trigger_word         TEXT NOT NULL DEFAULT '',
    created_at           REAL NOT NULL,
    output_lora_path     TEXT,                       -- 训练完回填主产物
    note                 TEXT,
    UNIQUE(project_id, label)
);
CREATE INDEX idx_versions_project ON versions(project_id);

CREATE TABLE project_jobs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id          INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    version_id          INTEGER REFERENCES versions(id) ON DELETE CASCADE,
                        -- NULL = project 级（download / preprocess）
                        -- 非 NULL = version 级（tag, reg_build, generate）
    kind                TEXT NOT NULL,             -- download | preprocess | tag | reg_build | generate
    params              TEXT NOT NULL,             -- JSON 序列化的输入参数
    status              TEXT NOT NULL,             -- pending | running | done | failed | canceled
    started_at          REAL,
    finished_at         REAL,
    pid                 INTEGER,
    log_path            TEXT,                       -- studio_data/jobs/{job_id}.log
    error_msg           TEXT
);
CREATE INDEX idx_jobs_project ON project_jobs(project_id);
CREATE INDEX idx_jobs_status ON project_jobs(status);

-- tasks 表（训练任务），含 project_id / version_id 外键
```

**Version 状态推进规则**（ADR-0007 §11.3-B / §11.5-A）：

| 维度 | 谁推 | 何时 |
|---|---|---|
| `status` (5 enum) | supervisor | task 启动 / 终态：`pending/running/paused → training`；`done → completed`；`failed → failed`；`canceled → canceled` |
| `phase` (5 enum, 仅 preparing) | 用户按 PhaseHeaderNav 的 "下一步" | 校验通过（详 ADR §11.5-B 每 phase 完成判定）→ cursor 前进 |
| `phase` 跳过 | 用户按 "下一步"（仅 `regularizing` 可跳） | 无 reg job running → cursor 直接跳到下一个 |
| **不主动回退** | — | user 删数据后下次 next 校验失败时提示，由 user redo（§11.5-C） |

**Project 无 stage**：ADR-0007 PR-5 删除。项目级数据集状态（download / preprocess 文件数）由 UI 实时扫派生（§6.10）。

---

## 4. 全局服务配置 `studio_data/secrets.json`

```jsonc
{
  "gelbooru": {
    "user_id": "",
    "api_key": "",
    "save_tags": false,                   // 是否同时保存 booru 自带标签
    "convert_to_png": true,
    "remove_alpha_channel": false
  },
  "huggingface": {
    "token": "",                           // WD14 公开模型不强制；私有/限速时填
    "endpoint": ""                         // 空 = HF 官方；可粘贴自建反代 URL。0.8.2 起 hf-mirror 暂时隐藏，见 docs/todo/hf-mirror-recheck.md
  },
  "joycaption": {
    "base_url": "http://localhost:8000/v1",
    "model": "fancyfeast/llama-joycaption-beta-one-hf-llava",
    "prompt_template": "Descriptive Caption"
  },
  "wd14": {
    "model_id": "SmilingWolf/wd-vit-tagger-v3",
    "local_dir": null,                    // null = models/wd14/{model_id}/
    "threshold_general": 0.35,
    "threshold_character": 0.85,
    "blacklist_tags": []
  }
}
```

Pydantic 模型在 `studio/secrets.py`；GET / PUT `/api/secrets` 操作；敏感字段（`token` / `api_key`）GET 时显示 `"***"`，PUT 时客户端发 `"***"` 表示「保持不变」。

前端 `/tools/settings` 表单分 7 个 tab（数据集 / 打标 / 训练 / 监控 / 测试 / 页面 / 系统），密码字段用 `<input type="password">`。系统 tab 含 webui 自更新版本卡片（详见 [ADR 0002](../adr/0002-webui-self-update.md)）和服务重启。

---

## 5. Sidebar 与路由

```
┌──────────────────────┐
│  Anima                │
│  lora studio · 0.11.1 │ ← 版本号从 /api/health 拉，single source of truth
├──────────────────────┤
│ ▶ 项目 (Projects)    │ /
│   队列 (Queue)       │ /queue
├──────────────────────┤
│   工具               │
│ ──────               │
│   预设 (Presets)     │ /tools/presets
│   测试 (Generate)    │ /tools/generate
│   监控 (Monitor)     │ /tools/monitor
│   设置 (Settings)    │ /tools/settings
└──────────────────────┘

进入项目后侧栏切换为 Stepper（仅当处于 /projects/:pid/* 下）：

┌──────────────────────┐
│ ← 返回项目列表        │
│ 项目: Cosmic Kaguya  │
│ 版本: [baseline ▾]   │ ← VersionTabs
├──────────────────────┤
│ ① 下载  ✓            │ /projects/:pid/download
│ ② 预处理（可选）✓    │ /projects/:pid/preprocess?tool=…  ← v0.8 新；v0.9.x 加裁剪
│ ③ 筛选  ✓            │ /projects/:pid/v/:vid/curate
│ ④ 打标  ✓            │ /projects/:pid/v/:vid/tag
│ ⑤ 标签编辑 ✓         │ /projects/:pid/v/:vid/edit
│ ⑥ 正则集（可选）●    │ /projects/:pid/v/:vid/reg        ← 当前
│ ⑦ 训练  ○            │ /projects/:pid/v/:vid/train
└──────────────────────┘
```

状态符号：✓ 完成 / ● 进行中 / ○ 未开始（按 stage + version.stats 派生）。

「② 预处理」内部按 query param 切多个**工具**子页（无 stage 顺序）：

- `?tool=overview` — 总览（gallery + 多选 + 撤销）
- 默认 / `?tool=upscale` — 放大（ESRGAN 等 spandrel 模型）
- `?tool=crop` — 裁剪（手动 / 智能聚类预填）
- `?tool=inpaint` — 涂抹（占位，未实现）

详见 [crop design](../design/preprocess-crop-design.md) §5 / §9。

---

## 6. SSE 事件目录

复用 `studio.event_bus.bus`：

| type | 字段 | 触发 |
|---|---|---|
| `task_state_changed` | `task_id`, `status`, `project_id?`, `version_id?` | 训练任务状态变 |
| `monitor_state_updated` | `task_id`, `state` | anima_train 写 monitor_state.json，全量 state 塞 payload |
| `project_state_changed` | `project_id`, `stage` | 项目 stage 推进 |
| `version_state_changed` | `project_id`, `version_id`, `stage` | 版本 stage 推进 |
| `job_state_changed` | `job_id`, `project_id`, `version_id?`, `kind`, `status` | download / tag / reg_build / generate / preprocess job |
| `job_log_appended` | `job_id`, `text`, `seq` | worker 写日志 → 推增量到前端 |
| `generate_progress` | `job_id`, `step`, `total_steps` | 推理 daemon 出图进度 |
| `preprocess_progress` | `job_id`, `project_id`, `idx`, `total`, `name`, `status`, `action?`, `succeeded`, `failed`, `skipped` | preprocess_worker 放大每张图完成 → 前端实时刷 files / 进度 / 盘占 |
| `crop_progress` | `job_id`, `project_id`, `idx`, `total`, `name`, `status`, `n_out?`, `outputs?`, `succeeded`, `failed`, `skipped` | preprocess_worker 裁剪每张图完成；worker 端节流 ≥1Hz（done 事件聚合，skip/fail/首末强发）|
| `system_stats_updated` | `cpu`, `gpu`, `mem`, `vram` | `_StatsThread` 2.5s 周期推 Topbar 系统资源 pill（v0.6） |

前端 `useEventStream.ts` 共享一条 `EventSource`，多个组件订阅不会重复连。

### 6.1 Worker → Supervisor 事件标记约定

子进程 worker 想发自定义 typed SSE 事件（而不是只发普通日志行）时，往 stdout 写：

```
__EVENT__:<event_type>:<json_payload>
```

例如：

```python
print('__EVENT__:preprocess_progress:{"idx":5,"total":73,"status":"done"}', flush=True)
```

`studio/supervisor.py:_EVENT_MARKER` 识别该前缀后：

1. 解析 `event_type` 和 JSON payload
2. **自动注入** `job_id` / `project_id` / `version_id` / `kind`（worker 不用知道也不能伪造）
3. `bus.publish` 成 typed SSE 事件给前端
4. 该行**不**进 `job_log_appended`（前端日志窗口不显示这种内部标记）

设计取舍：
- 比专门搭 IPC（队列 / socket / 状态文件）轻 — 复用现成的 stdout → log_tail 通道
- 比让前端文本 grep 日志靠谱 — 显式 schema、字段稳定
- 解析失败时 supervisor 写 `logger.exception` 但不会崩；标记行被丢弃，主流程不受影响
- 不要把敏感信息塞 payload — 任何看得到 SSE 流的客户端都能拿到

谁可以用：任何在 `studio/workers/` 下的子进程 worker。当前只 `preprocess_worker` 用了；
download / tag / reg_build worker 暂时只走 `job_log_appended`，如果将来需要细粒度进度，
按同样的约定加事件类型即可（同时需要在上面的 SSE 事件目录里登记一行）。

---

## 7. Tagger 抽象

```python
# studio/services/tagger.py
class TagResult(TypedDict):
    image: Path
    tags: list[str]                       # 排序好的（按概率降）
    raw_scores: dict[str, float]          # 可选：每 tag 的概率

class Tagger(Protocol):
    name: str                              # "wd14" / "cltagger" / "joycaption"
    requires_service: bool                 # 本地 ONNX False；JoyCaption True (vLLM)

    def is_available(self) -> tuple[bool, str]: ...
    def prepare(self) -> None: ...
    def tag(
        self,
        image_paths: list[Path],
        on_progress: Callable[[int, int], None] = None,
    ) -> Iterator[TagResult]: ...
```

ONNX 类 tagger（WD14 / CLTagger）继承 `OnnxTaggerBase`，自动获得线程池调度、GPU EP fallback、模型解析（local → HF 自动下载）。新增 ONNX tagger 注册到 `tagger registry`，UI 自动列出。

`<name>_overrides` 是统一持久化键约定（如 `wd14_overrides` / `cltagger_overrides`），前端按 tagger name 派生字段。

---

## 8. Preset 池关系

```
                  ┌──── presets/ (全局池) ────┐
                  │  train_baseline.yaml      │
                  │  high-lr.yaml             │
                  │  proj_42_baseline.yaml    │  ← 项目推回的命名
                  └──────────────────────────┘
                         ↑               ↓
                   save_as_preset    from_preset
                         │               │
                  ┌──────┴───────────────┴────────┐
                  │  versions/baseline/           │
                  │    config_name = "..."        │
                  └──────────────────────────────┘
```

| 操作 | 流程 |
|---|---|
| 创建版本 | 用户选「从预设 fork」或「从空白开始」 |
| Fork preset | 复制 `presets/{name}.yaml` → 自动重命名为 `proj_{pid}_{label}.yaml` 写回 `presets/` → version.config_name 指向它 |
| 编辑 config | 走 `/api/presets/{name}` PUT；version 共享所引用的 yaml |
| 推回预设 | `save_as_preset {target_name}` → 复制 yaml，**清空项目特定字段**：`data_dir` `reg_data_dir` `output_dir` `output_name` `resume_lora` `resume_state` |
| 切到另一预设 | `from_preset` 覆盖 version.config_name；旧的 `proj_*` 不删，可手动清理 |

「项目特定字段」清单在 `studio/services/version_config.py` 的 `PROJECT_SPECIFIC_FIELDS` 常量里。

---

## 9. 测试体系

| 类型 | 工具 | 范围 |
|---|---|---|
| 后端单元 | pytest | `projects.py` `versions.py` `services/*` `secrets.py` |
| 后端集成 | pytest + TestClient | API 端点全覆盖（200/4xx 路径） |
| 后端进程 | pytest + 假 cmd_builder | supervisor 调度 project_jobs |
| 前端单元 | Vitest | 纯函数 `lib/*` |
| 前端组件 | Vitest + RTL | 关键交互组件（ImageGrid 多选、TagEditor、Stepper、PreviewXYGrid） |

入口：

```bash
python -m studio test    # pytest + vitest
```

---

## 10. 已知约束

| 项 | 说明 |
|---|---|
| Slug 不可改 | title 可改，slug 一旦确定写死，避免目录搬迁 |
| Windows num_workers=0 | 多进程 spawn 易崩，dataloader worker 在 Windows 下强制单进程 |
| 单 GPU | 训练循环未实现 DDP/FSDP；多 GPU 需切训练后端（详见 [ADR 0001](../adr/0001-lokr-via-lycoris-lora.md)） |
| JoyCaption 需用户自起 vLLM | Studio 不在 Win 下管 vLLM 进程，只通过 base_url 调用 |
| 用户手动改磁盘 | 每次进 step 重扫，以磁盘为准（不维护 stale 缓存） |

---

## 11. 前端样式约定

**现阶段不做全局响应式**，布局以桌面端宽屏为主。单个页面 / 组件如有窄屏拥挤，可加简单单点适配，但必须遵守下述约定，方便未来做全局响应式时统一升级：

- 所有媒体查询集中在 `studio/web/src/styles/responsive.css`，不要散落到组件里
- 断点统一 `max-width: 1280px`（< laptop 中屏阈值），不要自创新断点
- 单点适配只动 padding / 尺寸 / 显隐 label，**不改布局结构**（结构性改造留给未来的全局响应式）
- 给目标元素加专属 className（如 `.banner-shell` / `.phase-timeline-label`），CSS 用 className 命中，不写到全局选择器
