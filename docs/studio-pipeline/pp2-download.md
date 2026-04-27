# PP2 — Gelbooru/Danbooru 下载集成

**状态**：已完成
**前置依赖**：PP1
**预估工作量**：2 工作日
**外部脚本来源**：`C:\Users\Mei\Desktop\SD\danbooru\dev\danbooru_downloader.py`

## 目标

- 把现有 `danbooru_downloader.py` 库化为 `studio/services/downloader.py`
- 用 `project_jobs` 表 + supervisor 扩展异步执行
- 实时日志推送（SSE `job_log_appended`）
- 前端 Download.tsx：tag 输入 + 数量 + 选项 → 启动 → 进度面板
- 完成后 `download/` 写入图片，project.stage 推进

不在范围：图片预览（PP3 处理）、自动打标（PP4）。

## 后端

### A. `studio/services/downloader.py`

把脚本 OO 类拆成纯函数 + 配置 dataclass。设计要点：
- 不用 input()，所有参数函数签名传入
- 配置从 `secrets.gelbooru` 取，不再从 `danbooru_config.json`
- 进度回调，便于流式日志输出

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

ProgressFn = Callable[[str], None]   # 接受日志一行

@dataclass
class DownloadOptions:
    tag: str                          # 主搜索 tag（多 tag 逗号分隔）
    count: int                        # 目标图片数
    api_source: str = "gelbooru"      # 或 "danbooru"
    save_tags: bool = False
    convert_to_png: bool = True
    remove_alpha_channel: bool = False
    skip_existing: bool = True        # download/ 里已存在的跳过
    user_id: str = ""                 # 从 secrets.gelbooru 注入
    api_key: str = ""

def estimate(opts: DownloadOptions) -> int:
    """先调 API 看候选数量；可选用于 UI 给「预计 N 张」。"""

def download(opts: DownloadOptions, dest_dir: Path,
             on_progress: ProgressFn = print,
             on_image_saved: Callable[[Path], None] = None) -> int:
    """
    阻塞式下载。
    - 返回成功保存的图片数
    - 文件命名：{post_id}.{ext}（如 12345.png）
    - 同时（save_tags=True）写 {post_id}.json：原始 tag 列表
    - 进度 / 错误通过 on_progress 推送到调用方（worker 转写到 log）
    """
```

实现注意：
- 速率限制：每秒最多 1 请求（gelbooru 推荐）
- 失败重试 3 次，指数退避
- 中途取消：worker 通过设置 `cancel_event: threading.Event` 注入；download 内部循环检查

### B. `studio/workers/__init__.py` + `studio/workers/download_worker.py`

```python
# python -m studio.workers.download_worker --job-id 7
import argparse, json, sys, time
from pathlib import Path

from studio import db, secrets
from studio.services.downloader import DownloadOptions, download
from studio.projects import project_dir, get_project

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", type=int, required=True)
    args = p.parse_args()

    with db.connection_for() as conn:
        job = conn.execute("SELECT * FROM project_jobs WHERE id=?", (args.job_id,)).fetchone()
    params = json.loads(job["params"])

    project = get_project(...)
    dest = project_dir(project["id"], project["slug"]) / "download"

    sec = secrets.load()
    opts = DownloadOptions(
        tag=params["tag"], count=params["count"],
        save_tags=sec.gelbooru.save_tags,
        convert_to_png=sec.gelbooru.convert_to_png,
        remove_alpha_channel=sec.gelbooru.remove_alpha_channel,
        user_id=sec.gelbooru.user_id, api_key=sec.gelbooru.api_key,
    )

    log_path = Path(job["log_path"])
    with open(log_path, "a", encoding="utf-8") as log_fp:
        def on_progress(line: str):
            log_fp.write(line + "\n")
            log_fp.flush()
            print(line)  # supervisor 也会捕获 stdout 写日志

        try:
            saved = download(opts, dest, on_progress=on_progress)
            on_progress(f"[done] saved {saved} images")
            sys.exit(0)
        except Exception as e:
            on_progress(f"[error] {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
```

### C. supervisor 扩展

`studio/supervisor.py` 现在只调度训练 task；扩展为同时调度 `project_jobs`：

```python
class Supervisor:
    def _tick(self):
        if self._current_proc:
            self._poll_current()
            return

        # 优先级：project_jobs 优先于 training tasks（避免下载被堵）
        # 单进程串行：要么跑 job 要么跑 task
        with db.connection_for() as conn:
            job = self._next_pending_job(conn)
        if job:
            self._spawn_job(job)
            return

        with db.connection_for() as conn:
            task = db.next_pending(conn)
        if task:
            self._spawn_task(task)

    def _spawn_job(self, job):
        # cmd_builder_for_job：根据 kind 选 worker
        cmd = self._job_cmd_builder(job)  # 默认: [python, "-m", "studio.workers.{kind}_worker", "--job-id", id]
        # log_path = LOGS_DIR / f"job_{id}.log"
        # status → running, started_at, pid
        # publish job_state_changed
        # 启动 tail 线程，读 log 增量推 job_log_appended
```

### D. Log tail 线程

```python
# studio/log_tail.py
import threading, time
from pathlib import Path
from typing import Callable

class LogTailer:
    """跟踪一个 log 文件，把新增行通过回调推出去。"""
    def __init__(self, path: Path, on_line: Callable[[str], None],
                 poll_interval: float = 0.5):
        self.path, self.on_line = path, on_line
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._poll = poll_interval
        self._offset = 0

    def start(self): self._t.start()
    def stop(self): self._stop.set(); self._t.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            if self.path.exists():
                with open(self.path, "rb") as f:
                    f.seek(self._offset)
                    chunk = f.read()
                    if chunk:
                        text = chunk.decode("utf-8", errors="replace")
                        for line in text.splitlines(keepends=False):
                            self.on_line(line)
                        self._offset += len(chunk)
            self._stop.wait(self._poll)
```

supervisor 在 `_spawn_job` 里：

```python
def _on_log_line(line):
    bus.publish({
        "type": "job_log_appended",
        "job_id": job["id"], "text": line, "seq": next(self._log_seq)
    })
self._tailer = LogTailer(log_path, _on_log_line)
self._tailer.start()
# _finish_job 时 stop
```

### E. 端点

```python
class DownloadRequest(BaseModel):
    tag: str
    count: int = Field(20, ge=1, le=10000)
    api_source: Literal["gelbooru", "danbooru"] = "gelbooru"

@app.post("/api/projects/{pid}/download")
def start_download(pid: int, body: DownloadRequest):
    # 检查 project 存在
    # 检查 secrets.gelbooru.api_key 已配（否则报 400 + 提示去 Settings）
    # 创建 project_jobs row, status=pending, kind=download, params=json
    # 推送 job_state_changed
    # 推进 project.stage → downloading
    # 不直接 spawn，由 supervisor 下个 tick 调起
    return {"job_id": ..., "project_id": pid}

@app.get("/api/projects/{pid}/download/status")
def download_status(pid: int):
    # 返回最近一次 download job 的状态（含 log 末尾 20 行）
    return {...}

@app.get("/api/projects/{pid}/files")
def list_files(pid: int, bucket: str = "download"):
    # bucket: download | (PP3 加 train)
    # 列出文件名 + size + 可选 metadata
    return {"items": [{"name": "12345.png", "size": 102400, "has_meta": true}, ...]}

# /api/jobs 通用
@app.get("/api/jobs/{jid}")
def get_job(jid: int): ...

@app.get("/api/jobs/{jid}/log")
def get_job_log(jid: int, tail: int = 200):
    # 整文件 or 末 N 行
    ...

@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: int):
    # supervisor.cancel_job(jid)：当前跑的就 SIGTERM
    ...
```

## 前端

### A. `pages/Project/steps/Download.tsx`

```tsx
function DownloadPage() {
  const { pid } = useParams()
  const [project, setProject] = useState<ProjectDetail>()
  const [job, setJob] = useState<Job | null>(null)
  const [form, setForm] = useState({ tag: "", count: 20 })
  const [logs, setLogs] = useState<string[]>([])

  // 加载最新 job 状态
  // 监听 SSE：
  //   job_state_changed (job_id 匹配) → setJob
  //   job_log_appended (job_id 匹配) → setLogs(prev => [...prev, line])

  const start = async () => {
    if (!secretsHasGelbooru()) {
      toast("请先在「设置」配置 Gelbooru 账户", "error")
      return
    }
    const r = await api.startDownload(pid, form)
    setJob(r)
  }

  return (
    <div>
      <h1>下载数据</h1>
      <p>从 {form.api_source} 拉取 tag <code>{form.tag}</code> 共 {form.count} 张</p>
      {/* 表单 */}
      <input value={form.tag} ... placeholder="例如：character_x">
      <input type="number" value={form.count} min={1} max={10000}>
      <select value={form.api_source}>...</select>
      <button onClick={start} disabled={job?.status === 'running'}>开始下载</button>

      {job && <JobProgress job={job} logs={logs} onCancel={...} />}

      {/* 已下载图片预览（缩略图 grid，最多 50 张） */}
      <FileList pid={pid} bucket="download" />
    </div>
  )
}
```

### B. `components/JobProgress.tsx`

```tsx
interface Props {
  job: Job
  logs: string[]
  onCancel?: () => void
}

function JobProgress({ job, logs, onCancel }: Props) {
  // 状态条 + 已耗时 + 取消按钮
  // 日志面板（滚动 pre，最多保留 1000 行；超出截断头部）
  // 自动滚到底
}
```

### C. `components/FileList.tsx`

```tsx
interface Props {
  pid: number
  bucket: 'download' | 'train' | 'reg' | 'samples'
  vid?: number      // train/reg/samples 用
  folder?: string   // train/5_concept 用
}

// 调 /api/projects/{pid}/files 或对应 version 端点
// 缩略图网格（aspect-ratio square）+ 文件名 + 总数
```

### D. API client

```tsx
api.startDownload(pid, body)
api.getDownloadStatus(pid)
api.listFiles(pid, bucket)
api.getJob(jid)
api.getJobLog(jid, tail?)
api.cancelJob(jid)
```

## 测试

### 后端 pytest

- `tests/test_downloader.py`：
  - `DownloadOptions` 校验
  - mock `requests`：返回假 booru API 响应，确认下载循环 + 文件落盘
  - 速率限制（quasi-时间断言）
  - 取消 event 中途生效
- `tests/test_download_worker.py`：构造假 job + 假 secrets，验证 worker exit code
- `tests/test_supervisor_jobs.py`：supervisor 能调度 project_jobs（参考 `test_supervisor.py` 模式）
- `tests/test_log_tail.py`：写文件 → tail 推送 → 收到对应行
- `tests/test_download_endpoints.py`：完整 HTTP

### 手测剧本

1. Settings 页填 Gelbooru user_id + api_key
2. 项目 A → Download 页
3. tag = `character_x`，count = 5（小量测试）
4. 点开始 → 看到「下载中」+ 实时日志：`[1/5] downloading 12345.png ...`
5. 5 张全下完 → status: done，缩略图列表出现
6. 再次点开始（同 tag, count=10）→ 已存在的跳过，新增 5 张
7. project.stage 推进到 `curating`（数据库 / 前端 stepper 都验证）
8. 中途取消：count=20 启动后立即点取消 → 子进程退出，job status=canceled

### 真实集成测试（可选）

如果有 Gelbooru 账户：用一个冷门 tag（如 `aki_(fish)`，count=2）验证完整链路。

## 风险

| 风险 | 应对 |
|---|---|
| Gelbooru API 限流（HTTP 429） | downloader 已带退避；UI 给「重试」按钮 |
| 网络断开 | worker 重试 3 次；最终失败标 status=failed，error_msg 记错 |
| 用户在下载中删除项目 | supervisor 取消当前 job + 清 trash |
| 大量小图 → 文件系统压力 | 不做特殊处理；download/ 是平铺，PP3 缩略图按需加载 |
| 中文 tag 编码 | URL encode + UTF-8；测试一个中文 tag |
