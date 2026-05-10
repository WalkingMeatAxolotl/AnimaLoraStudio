# Studio 架构总览

跨步骤的横切关注点：数据模型、目录布局、SQLite schema、secrets、Sidebar、SSE 事件、Tagger 抽象、Preset 关系。Studio 内部模块结构见 [`studio/README.md`](../../studio/README.md)。

---

## 1. Pipeline 流程

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
│ 创建项目 │ → │ 下载数据 │ → │ 筛选数据 │ → │ 标签生成 │ → │ 正则集生成 │ → │ 配置/入队 │
│ Project  │   │ Download │   │ Curation │   │ Tagging  │   │ Reg-build  │   │ Train    │
│ 含 v1    │   │ 项目级   │   │ 版本级   │   │ 版本级   │   │ 版本级     │   │ 版本级   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └────────────┘   └──────────┘
                                       ↓ 可循环（新建 v2 重新筛选/打标）
```

每个版本（version）独立维护 `train/` `reg/` `output/` `samples/` `monitor_state.json`；`download/` 在项目级共享，**永远不删**（永远是「全量来源」）。

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
└── _trash/projects/{id}-{slug}/          软删；UI 提供「清空 trash」按钮
```

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
    stage               TEXT NOT NULL DEFAULT 'created',
                        -- created | downloading | curating | tagging | regularizing | configured | training | done
    active_version_id   INTEGER REFERENCES versions(id) ON DELETE SET NULL,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    note                TEXT
);
CREATE INDEX idx_projects_slug ON projects(slug);

CREATE TABLE versions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id          INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    label               TEXT NOT NULL,             -- 用户填：baseline / high-lr / ...
    config_name         TEXT,                       -- 引用 presets/{config_name}.yaml
    stage               TEXT NOT NULL DEFAULT 'curating',
                        -- curating | tagging | regularizing | ready | training | done
    created_at          REAL NOT NULL,
    output_lora_path    TEXT,                       -- 训练完回填主产物
    note                TEXT,
    UNIQUE(project_id, label)
);
CREATE INDEX idx_versions_project ON versions(project_id);

CREATE TABLE project_jobs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id          INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    version_id          INTEGER REFERENCES versions(id) ON DELETE CASCADE,
                        -- NULL = project 级（download）
                        -- 非 NULL = version 级（tag, reg_build, generate）
    kind                TEXT NOT NULL,             -- download | tag | reg_build | generate
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

**Stage 推进规则**（前端只读，后端权威；写在 `studio/projects.py:advance_stage()` 里）：

| 当前 stage | 触发条件 | 下一个 stage |
|---|---|---|
| `created` | download job 启动 | `downloading` |
| `downloading` | download job 完成 | `curating` |
| `curating` | version 的 train/ 有图 | `tagging` (active version) |
| `tagging` | tag job 完成 | `regularizing`（如果用户跳过则进 `configured`） |
| `regularizing` | reg job 完成 / 用户跳过 | `configured` |
| `configured` | task 入队 | `training` |
| `training` | task done | `done` |

stage 只是「展示性」字段——前端 Stepper 高亮用，跳步骤不强制。

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
    "endpoint": "https://hf-mirror.com"    // 可在 Settings 切到 huggingface.co 或自建反代
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

前端 `/tools/settings` 表单分 4 个 tab（数据集 / 打标 / 训练 / 页面），密码字段用 `<input type="password">`。

---

## 5. Sidebar 与路由

```
┌──────────────────────┐
│  Anima                │
│  lora studio · 0.5.0  │ ← 版本号从 /api/health 拉，single source of truth
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
│ ② 筛选  ✓            │ /projects/:pid/v/:vid/curate
│ ③ 打标  ✓            │ /projects/:pid/v/:vid/tag
│ ④ 标签编辑 ✓         │ /projects/:pid/v/:vid/edit
│ ⑤ 正则集 ●           │ /projects/:pid/v/:vid/reg        ← 当前
│ ⑥ 训练  ○            │ /projects/:pid/v/:vid/train
└──────────────────────┘
```

状态符号：✓ 完成 / ● 进行中 / ○ 未开始（按 stage + version.stats 派生）。

---

## 6. SSE 事件目录

复用 `studio.event_bus.bus`：

| type | 字段 | 触发 |
|---|---|---|
| `task_state_changed` | `task_id`, `status`, `project_id?`, `version_id?` | 训练任务状态变 |
| `monitor_state_updated` | `task_id`, `state` | anima_train 写 monitor_state.json，全量 state 塞 payload |
| `project_state_changed` | `project_id`, `stage` | 项目 stage 推进 |
| `version_state_changed` | `project_id`, `version_id`, `stage` | 版本 stage 推进 |
| `job_state_changed` | `job_id`, `project_id`, `version_id?`, `kind`, `status` | download / tag / reg_build / generate job |
| `job_log_appended` | `job_id`, `text`, `seq` | worker 写日志 → 推增量到前端 |
| `generate_progress` | `job_id`, `step`, `total_steps` | 推理 daemon 出图进度 |

前端 `useEventStream.ts` 共享一条 `EventSource`，多个组件订阅不会重复连。

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
