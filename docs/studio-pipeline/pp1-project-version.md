# PP1 — Project + Version 数据模型

**状态**：已完成
**前置依赖**：PP0
**预估工作量**：2 工作日

## 目标

落地 Project / Version 的核心数据模型与 CRUD：
- SQLite `projects` `versions` 表（迁移）
- 后端 `studio/projects.py` `studio/versions.py`
- 物理目录创建器（`studio_data/projects/{id}-{slug}/...`）
- 前端 Projects 列表页、Project Layout（Stepper + VersionTabs 占位）、Overview
- 软删（移到 `_trash/`）

不在本 PP 范围：下载 / 筛选 / 打标 / 正则 / 训练（各自后续 PP）；Stepper 5 步骤页面只放占位。

## 后端

### A. 数据库迁移

新增 `studio/migrations/` 目录与启动时 schema 升级机制：

```python
# studio/migrations/__init__.py
import sqlite3
from typing import Callable

# 按顺序应用；user_version 跟踪当前版本
MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [
    # v1 = 当前 base schema (tasks 表)，在 db.init_db 已建
    # v2 = PP1：projects / versions / project_jobs + tasks 扩字段
    _migrate_v2_projects,
]

def apply_all(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA user_version").fetchone()
    current = cur[0]
    for i, migrate in enumerate(MIGRATIONS, start=1):
        if current < i:
            migrate(conn)
            conn.execute(f"PRAGMA user_version = {i}")
    conn.commit()
```

`db.init_db()` 改成：先 executescript SCHEMA（创建基础表），再 `apply_all()`。

```python
# studio/migrations/_v2.py
def _migrate_v2_projects(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            stage TEXT NOT NULL DEFAULT 'created',
            active_version_id INTEGER,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            note TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_projects_slug ON projects(slug);

        CREATE TABLE IF NOT EXISTS versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            config_name TEXT,
            stage TEXT NOT NULL DEFAULT 'curating',
            created_at REAL NOT NULL,
            output_lora_path TEXT,
            note TEXT,
            UNIQUE(project_id, label),
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_versions_project ON versions(project_id);

        CREATE TABLE IF NOT EXISTS project_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            version_id INTEGER,
            kind TEXT NOT NULL,
            params TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at REAL,
            finished_at REAL,
            pid INTEGER,
            log_path TEXT,
            error_msg TEXT,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
            FOREIGN KEY(version_id) REFERENCES versions(id) ON DELETE CASCADE
        );

        ALTER TABLE tasks ADD COLUMN project_id INTEGER;
        ALTER TABLE tasks ADD COLUMN version_id INTEGER;
    """)
```

### B. `studio/projects.py`

```python
import re
import time
from pathlib import Path

from . import db
from .paths import STUDIO_DATA

PROJECTS_DIR = STUDIO_DATA / "projects"
TRASH_DIR = STUDIO_DATA / "_trash" / "projects"

VALID_STAGES = {
    "created", "downloading", "curating", "tagging",
    "regularizing", "configured", "training", "done"
}

class ProjectError(Exception): pass

# slug 算法
_slug_pat = re.compile(r"[^a-z0-9]+")
def slugify(title: str) -> str:
    s = _slug_pat.sub("-", title.lower()).strip("-")
    return s or "project"

def _unique_slug(conn, base: str) -> str:
    """如果 base 已被占用，加 -2 -3 后缀。"""
    n = 1
    candidate = base
    while conn.execute("SELECT 1 FROM projects WHERE slug=?", (candidate,)).fetchone():
        n += 1
        candidate = f"{base}-{n}"
    return candidate

def project_dir(project_id: int, slug: str) -> Path:
    return PROJECTS_DIR / f"{project_id}-{slug}"

def create_project(conn, *, title: str, slug: str | None = None,
                   note: str | None = None) -> dict:
    base_slug = slug or slugify(title)
    final_slug = _unique_slug(conn, base_slug)
    now = time.time()
    cur = conn.execute(
        "INSERT INTO projects(slug, title, stage, created_at, updated_at, note) "
        "VALUES (?, ?, 'created', ?, ?, ?)",
        (final_slug, title, now, now, note)
    )
    conn.commit()
    pid = cur.lastrowid
    # 创建目录
    pdir = project_dir(pid, final_slug)
    (pdir / "download").mkdir(parents=True, exist_ok=True)
    (pdir / "versions").mkdir(parents=True, exist_ok=True)
    (pdir / "project.json").write_text(...)  # 同步 json
    return get_project(conn, pid)

def list_projects(conn, include_trashed: bool = False) -> list[dict]: ...
def get_project(conn, project_id: int) -> dict | None: ...
def update_project(conn, project_id: int, **fields) -> dict: ...
def soft_delete_project(conn, project_id: int) -> None:
    """把目录移到 _trash/，db 里删（CASCADE 删 versions / project_jobs）。"""
def empty_trash() -> int: ...

def write_project_json(p: dict) -> None:
    """同步 project.json 到磁盘。stage / active_version_id 等字段冗余存。"""

def advance_stage(conn, project_id: int, target: str) -> None:
    """推进项目 stage，发 SSE 事件。"""
```

### C. `studio/versions.py`

```python
import time
from pathlib import Path

from . import db, projects

VALID_STAGES = {
    "curating", "tagging", "regularizing",
    "ready", "training", "done"
}

class VersionError(Exception): pass

def version_dir(project_id: int, slug: str, label: str) -> Path:
    return projects.project_dir(project_id, slug) / "versions" / label

def create_version(conn, *, project_id: int, label: str,
                   fork_from_version_id: int | None = None,
                   note: str | None = None) -> dict:
    """
    label 由用户提供；项目内唯一。
    fork_from_version_id 给了：
      - 复制源 version 的 train/ 整树（硬链接，省空间；Win 下 fallback 到 copy）
      - 复制 config_name 到新名 proj_{pid}_{label}（先调用 presets fork）
    """
    # 1. label 校验：只允许 [A-Za-z0-9_-]+
    # 2. 写 db
    # 3. 创建目录：versions/{label}/{train,reg,output,samples}/
    # 4. 写 version.json
    # 5. 如果是项目第一个版本，自动设 active_version_id

def list_versions(conn, project_id: int) -> list[dict]: ...
def get_version(conn, version_id: int) -> dict | None: ...
def update_version(conn, version_id: int, **fields) -> dict: ...
def delete_version(conn, version_id: int) -> None:
    """目录搬到 _trash/projects/{slug}/versions/{label}/，db 删。
    若是 active_version，自动选剩下的最新一个为 active。"""

def activate_version(conn, version_id: int) -> dict: ...

def advance_stage(conn, version_id: int, target: str) -> None: ...
```

### D. `studio/server.py` 端点

```python
class ProjectCreate(BaseModel):
    title: str
    slug: Optional[str] = None
    note: Optional[str] = None
    initial_version_label: Optional[str] = "v1"  # 默认建一个 v1

class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    note: Optional[str] = None
    stage: Optional[str] = None
    active_version_id: Optional[int] = None

class VersionCreate(BaseModel):
    label: str
    fork_from_version_id: Optional[int] = None
    note: Optional[str] = None

class VersionUpdate(BaseModel):
    label: Optional[str] = None
    note: Optional[str] = None
    stage: Optional[str] = None

# 项目
@app.get("/api/projects")
@app.post("/api/projects")
@app.get("/api/projects/{pid}")
@app.patch("/api/projects/{pid}")
@app.delete("/api/projects/{pid}")
@app.post("/api/projects/_trash/empty")

# 版本
@app.get("/api/projects/{pid}/versions")
@app.post("/api/projects/{pid}/versions")
@app.get("/api/projects/{pid}/versions/{vid}")
@app.patch("/api/projects/{pid}/versions/{vid}")
@app.delete("/api/projects/{pid}/versions/{vid}")
@app.post("/api/projects/{pid}/versions/{vid}/activate")
```

详情接口返回示例：

```jsonc
GET /api/projects/42
{
  "id": 42,
  "slug": "cosmic-kaguya",
  "title": "Cosmic Kaguya",
  "stage": "curating",
  "active_version_id": 7,
  "created_at": 1700000000,
  "updated_at": 1700000050,
  "note": null,
  "versions": [
    {
      "id": 7,
      "label": "baseline",
      "config_name": null,
      "stage": "curating",
      "created_at": 1700000005,
      "output_lora_path": null,
      "note": null,
      "stats": {                          // 由 server 即时计算
        "train_image_count": 0,
        "train_folders": [],
        "reg_image_count": 0,
        "has_output": false
      }
    }
  ],
  "download_image_count": 0
}
```

### E. SSE 事件

```python
bus.publish({"type": "project_state_changed", "project_id": pid, "stage": new_stage})
bus.publish({"type": "version_state_changed", "project_id": pid, "version_id": vid, "stage": new_stage})
```

## 前端

### A. 路由扩展

```tsx
<Route path="/" element={<ProjectsList />} />
<Route path="/projects/:pid" element={<ProjectLayout />}>
  <Route index element={<ProjectOverview />} />
  <Route path="download" element={<DownloadPlaceholder />} />            {/* PP2 */}
  <Route path="v/:vid">
    <Route path="curate" element={<CurationPlaceholder />} />            {/* PP3 */}
    <Route path="tag" element={<TaggingPlaceholder />} />                {/* PP4 */}
    <Route path="reg" element={<RegPlaceholder />} />                    {/* PP5 */}
    <Route path="train" element={<TrainPlaceholder />} />                {/* PP6 */}
  </Route>
</Route>
```

`*Placeholder` 组件统一一个：「PP{n} 阶段实现」+ 链接到 plan 文档（dev 模式下）。

### B. `pages/Projects.tsx`

```tsx
function ProjectsList() {
  const [items, setItems] = useState<Project[]>([])
  const [creating, setCreating] = useState(false)
  // GET /api/projects
  // 卡片布局：title / stage badge / 版本数 / 创建时间 / 删除按钮
  // 「+ 新建项目」按钮 → 弹对话框：title, note
}

interface NewProjectDialog {
  title: string
  initial_version_label: string  // 默认 "v1"
}
```

### C. `pages/Project/Layout.tsx`

```tsx
function ProjectLayout() {
  const { pid } = useParams()
  const [project, setProject] = useState<ProjectDetail>()
  const [activeVid, setActiveVid] = useState<number | null>(null)

  // 监听 SSE: project_state_changed / version_state_changed → reload
  // 渲染：
  //   左侧：返回 + ProjectStepper + VersionTabs
  //   右侧：<Outlet />
}
```

### D. `components/VersionTabs.tsx`

```tsx
interface Props {
  versions: Version[]
  activeId: number | null
  onSelect: (vid: number) => void
  onCreate: () => void          // 弹「新建版本」对话框
  onDelete: (vid: number) => void
}
```

横向 tab，加号在最右。tab 上显示 label 和 stage 小圆点。

### E. `components/ProjectStepper.tsx`

```tsx
const STEPS = [
  { key: 'download',  label: '① 下载',  scope: 'project' },
  { key: 'curate',    label: '② 筛选',  scope: 'version' },
  { key: 'tag',       label: '③ 打标',  scope: 'version' },
  { key: 'reg',       label: '④ 正则集', scope: 'version' },
  { key: 'train',     label: '⑤ 训练',  scope: 'version' },
]

function ProjectStepper({ project, activeVersion }: Props) {
  // 根据 project.stage / activeVersion.stage 决定 ✓ ● ○
  // 点击导航到 /projects/:pid[/v/:vid]/<step>
}
```

### F. `pages/Project/Overview.tsx`

```tsx
function ProjectOverview() {
  // 当前项目的版本卡片网格
  // 每个版本：label / stage / train_image_count / output 是否有 / note
  // 点击切到该版本（activate + 跳第一个未完成 step）
  // 「+ 新建版本」按钮 → fork-from? 选源版本（可选）
}
```

### G. API client

```tsx
api.listProjects()
api.getProject(pid)
api.createProject({ title, slug?, note?, initial_version_label? })
api.updateProject(pid, partial)
api.deleteProject(pid)

api.listVersions(pid)
api.getVersion(pid, vid)
api.createVersion(pid, { label, fork_from_version_id?, note? })
api.updateVersion(pid, vid, partial)
api.deleteVersion(pid, vid)
api.activateVersion(pid, vid)
```

## 测试

### 后端 pytest

- `tests/test_projects.py`：
  - `slugify` / `_unique_slug` 各种 corner case
  - `create_project` 创建后磁盘目录正确（download/, versions/, project.json）
  - `soft_delete_project` 把目录移到 `_trash/`
  - `empty_trash` 物理删
- `tests/test_versions.py`：
  - `create_version` label 唯一约束（同 project 内重名报错）
  - `fork_from` 复制 train 树（mock 一些假图）
  - `activate_version` 更新 project.active_version_id
  - 删 active version 自动切换
- `tests/test_project_endpoints.py`：完整 CRUD + 错误码
- `tests/test_migrations.py`：从 v1 schema 升到 v2，旧 tasks 数据不丢

### 前端 Vitest

- `Sidebar`：新建项目链接到 `/`；活动路由高亮
- `ProjectStepper`：按 stage 显示 ✓ ● ○
- `VersionTabs`：点 tab 触发 onSelect

### 手测剧本

1. 启动 Studio
2. 进首页 → 看到空列表 + 「+ 新建项目」按钮
3. 创建 3 个项目（title: A / B / 测试中文标题）→ 列表里出现，含 stage badge
4. 进 A → 看到默认 v1 版本卡片，stepper 第一步「下载」高亮
5. 在 A 里新建 v2（label: high-lr）→ VersionTabs 出现 v1, v2，可切
6. 切到 v1 → 删除 v2 → VersionTabs 只剩 v1
7. 软删项目 B → 列表消失；磁盘 `studio_data/_trash/projects/{B-id}-b/` 存在
8. 重启 Studio → 项目列表数据保留
9. （DB 检查）`sqlite3 studio_data/studio.db "SELECT * FROM projects;"` 看到 A 和「测试中文标题」，B 已删

## 风险

| 风险 | 应对 |
|---|---|
| Windows 上 hard link 失败 | versions fork 默认 copy；hard link 仅尝试 + 失败回退 |
| 标题含特殊字符 | slugify 严格化；前端预览生成的 slug |
| 已有用户已经有 task 数据，迁移加列 | `ALTER TABLE ... ADD COLUMN` 默认 NULL，安全 |
| 活动版本 = 删除中 | 删除前在 db 里自动 reassign，再删行 |
