import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  api,
  type Task,
  type TaskOutputs,
  type TaskStatus,
} from '../api/client'
import { useToast } from '../components/Toast'
import { useEventStream } from '../lib/useEventStream'

type Tab = 'overview' | 'log' | 'monitor' | 'outputs'

const STATUS_STYLE: Record<TaskStatus, string> = {
  pending: 'bg-slate-700 text-slate-300',
  running: 'bg-cyan-600/30 text-cyan-300 animate-pulse',
  done: 'bg-emerald-700/40 text-emerald-300',
  failed: 'bg-red-700/40 text-red-300',
  canceled: 'bg-slate-700/40 text-slate-400',
}

const STATUS_LABEL: Record<TaskStatus, string> = {
  pending: '待运行',
  running: '运行中',
  done: '完成',
  failed: '失败',
  canceled: '已取消',
}

const TERMINAL: ReadonlyArray<TaskStatus> = ['done', 'failed', 'canceled']

function fmtTime(ts: number | null | undefined): string {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleString('zh-CN', { hour12: false })
}

function fmtDuration(start?: number | null, end?: number | null): string {
  if (!start) return '—'
  const e = end ?? Date.now() / 1000
  const sec = Math.max(0, e - start)
  if (sec < 60) return `${sec.toFixed(0)}s`
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  if (m < 60) return `${m}m ${s}s`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}

export default function QueueDetailPage() {
  const { id } = useParams<{ id: string }>()
  const taskId = Number(id)
  const navigate = useNavigate()
  const { toast } = useToast()

  const [task, setTask] = useState<Task | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [tab, setTab] = useState<Tab>(() => {
    if (typeof window === 'undefined') return 'overview'
    const v = window.location.hash.replace(/^#/, '')
    return (['overview', 'log', 'monitor', 'outputs'] as const).includes(
      v as Tab
    )
      ? (v as Tab)
      : 'overview'
  })
  const [confirmDelete, setConfirmDelete] = useState(false)

  // hash 同步当前 tab，方便分享 / 收藏 / 浏览器后退
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const h = `#${tab}`
      if (window.location.hash !== h) {
        window.history.replaceState(null, '', h)
      }
    }
  }, [tab])

  const reload = useCallback(async () => {
    if (!Number.isFinite(taskId)) return
    try {
      const t = await api.getTask(taskId)
      setTask(t)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [taskId])

  useEffect(() => {
    void reload()
  }, [reload])

  useEventStream((evt) => {
    if (evt.type === 'task_state_changed' && evt.task_id === taskId) {
      void reload()
    }
  })

  // running 时刷新时长（每 2s 重渲染）
  useEffect(() => {
    if (task?.status !== 'running') return
    const tick = window.setInterval(() => setTask((t) => (t ? { ...t } : t)), 2000)
    return () => window.clearInterval(tick)
  }, [task?.status])

  if (!Number.isFinite(taskId)) {
    return <p className="text-red-400">无效任务 ID</p>
  }

  const status = task?.status
  const isLive = status === 'running' || status === 'pending'
  const isTerminal = !!status && TERMINAL.includes(status)

  const cancel = async () => {
    if (!task) return
    setBusy(true)
    try {
      await api.cancelTask(task.id)
      toast('已发送取消信号', 'success')
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const retry = async () => {
    if (!task) return
    setBusy(true)
    try {
      const newTask = await api.retryTask(task.id)
      toast(`重试已入队 #${newTask.id}`, 'success')
      navigate(`/queue/${newTask.id}`)
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const remove = async () => {
    if (!task) return
    setBusy(true)
    try {
      await api.deleteTask(task.id)
      toast('已删除', 'success')
      navigate('/queue')
    } catch (e) {
      toast(String(e), 'error')
      setBusy(false)
      setConfirmDelete(false)
    }
  }

  return (
    <div className="flex flex-col h-full min-h-0 w-full gap-2 overflow-hidden">
      {/* Header */}
      <header className="flex items-baseline gap-3 shrink-0">
        <Link
          to="/queue"
          className="text-xs text-slate-400 hover:text-slate-200"
        >
          ← 队列
        </Link>
        <h1 className="text-base font-semibold">
          任务 <span className="font-mono text-slate-500">#{taskId}</span>
        </h1>
        {task && (
          <>
            <span className="text-sm text-slate-300">{task.name}</span>
            <code className="text-[11px] text-slate-500 font-mono">
              {task.config_name}.yaml
            </code>
          </>
        )}
        {status && (
          <span
            className={
              'px-2 py-0.5 rounded text-xs font-mono ' + STATUS_STYLE[status]
            }
          >
            {STATUS_LABEL[status]}
          </span>
        )}
        {error && (
          <span className="text-xs text-red-400 ml-2">{error}</span>
        )}
      </header>

      {/* Tabs */}
      <nav className="flex items-center gap-1 border-b border-slate-700 shrink-0 text-xs">
        {(
          [
            ['overview', '📋 详情'],
            ['log', '📜 日志'],
            ['monitor', '📊 监控'],
            ['outputs', '📦 输出'],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={
              'px-3 py-1.5 border-b-2 -mb-px transition-colors ' +
              (tab === key
                ? 'border-cyan-500 text-cyan-200'
                : 'border-transparent text-slate-400 hover:text-slate-200')
            }
          >
            {label}
          </button>
        ))}
      </nav>

      {/* Tab body */}
      <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
        {tab === 'overview' && task && <OverviewTab task={task} />}
        {tab === 'overview' && !task && (
          <p className="text-slate-500 text-xs p-2">加载...</p>
        )}
        {tab === 'log' && (
          <LogTab taskId={taskId} status={status ?? null} />
        )}
        {tab === 'monitor' && <MonitorTab taskId={taskId} />}
        {tab === 'outputs' && <OutputsTab taskId={taskId} />}
      </div>

      {/* Footer actions — 危险操作集中在这里，跟内容区视觉隔开 */}
      <footer className="flex items-center gap-2 pt-2 border-t border-slate-700 shrink-0 text-xs">
        <Link
          to="/queue"
          className="px-3 py-1.5 text-slate-400 hover:text-slate-200"
        >
          ← 返回队列
        </Link>
        <span className="flex-1" />
        {isLive && (
          <button
            onClick={cancel}
            disabled={busy}
            className="px-3 py-1.5 rounded bg-amber-700/60 hover:bg-amber-700 text-amber-100 disabled:opacity-50"
          >
            ✕ 取消任务
          </button>
        )}
        {isTerminal && (
          <>
            <button
              onClick={retry}
              disabled={busy}
              className="px-3 py-1.5 rounded bg-cyan-600 hover:bg-cyan-500 text-white disabled:opacity-50"
            >
              🔁 重试
            </button>
            <button
              onClick={() => setConfirmDelete(true)}
              disabled={busy}
              className="px-3 py-1.5 rounded bg-red-700/80 hover:bg-red-700 text-red-100 disabled:opacity-50"
            >
              🗑 删除记录
            </button>
          </>
        )}
      </footer>

      {confirmDelete && task && (
        <ConfirmDialog
          title="删除任务记录"
          message={
            <>
              将永久删除任务{' '}
              <code className="text-slate-200">
                #{task.id} {task.name}
              </code>{' '}
              的数据库记录。
              <br />
              <span className="text-slate-400 text-[11px]">
                LoRA / 训练日志 / 监控状态文件不会被删，仍在磁盘上。
              </span>
            </>
          }
          confirmLabel="删除"
          danger
          onConfirm={remove}
          onCancel={() => setConfirmDelete(false)}
          busy={busy}
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Overview tab
// ---------------------------------------------------------------------------

function OverviewTab({ task }: { task: Task }) {
  const rows: Array<[string, React.ReactNode]> = [
    ['ID', <code key="id" className="font-mono">{task.id}</code>],
    ['名称', task.name],
    ['Config', <code key="cn" className="font-mono">{task.config_name}.yaml</code>],
    [
      '状态',
      <span
        key="st"
        className={
          'px-2 py-0.5 rounded text-xs font-mono ' + STATUS_STYLE[task.status]
        }
      >
        {STATUS_LABEL[task.status]}
      </span>,
    ],
    ['优先级', task.priority],
    ['入队时间', fmtTime(task.created_at)],
    ['开始时间', fmtTime(task.started_at)],
    ['结束时间', fmtTime(task.finished_at)],
    ['运行时长', fmtDuration(task.started_at, task.finished_at)],
    ['退出码', task.exit_code ?? '—'],
    ['PID', task.pid ?? '—'],
  ]
  if (task.project_id || task.version_id) {
    rows.push([
      '来源',
      task.project_id && task.version_id ? (
        <Link
          to={`/projects/${task.project_id}/v/${task.version_id}/train`}
          className="text-cyan-400 hover:underline font-mono"
        >
          项目 #{task.project_id} / v#{task.version_id}
        </Link>
      ) : (
        '—'
      ),
    ])
  }
  if (task.config_path) {
    rows.push([
      'Config 路径',
      <code key="cp" className="font-mono break-all text-[11px]">
        {task.config_path}
      </code>,
    ])
  }
  if (task.monitor_state_path) {
    rows.push([
      '监控文件',
      <code key="msp" className="font-mono break-all text-[11px]">
        {task.monitor_state_path}
      </code>,
    ])
  }
  if (task.error_msg) {
    rows.push([
      '错误',
      <code
        key="err"
        className="font-mono break-all text-red-300 text-[11px]"
      >
        {task.error_msg}
      </code>,
    ])
  }

  return (
    <div className="overflow-y-auto">
      <table className="w-full text-xs">
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label} className="border-b border-slate-800/60 last:border-0">
              <th className="px-2 py-1.5 text-left font-normal text-slate-500 w-32 align-top">
                {label}
              </th>
              <td className="px-2 py-1.5 text-slate-200 align-top">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Log tab — 之前 pages/Log.tsx 的核心逻辑搬过来
// ---------------------------------------------------------------------------

function LogTab({
  taskId,
  status,
}: {
  taskId: number
  status: TaskStatus | null
}) {
  const [content, setContent] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const preRef = useRef<HTMLPreElement>(null)

  const refresh = useCallback(async () => {
    try {
      const log = await api.getLog(taskId)
      setContent(log.content)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [taskId])

  useEffect(() => {
    void refresh()
  }, [refresh])

  // 状态变化时刷新
  useEventStream((evt) => {
    if (evt.type === 'task_state_changed' && evt.task_id === taskId) {
      void refresh()
    }
  })

  // running 时每 2s 拉一次
  useEffect(() => {
    if (status !== 'running') return
    const tick = window.setInterval(() => void refresh(), 2000)
    return () => window.clearInterval(tick)
  }, [status, refresh])

  useEffect(() => {
    if (autoScroll && preRef.current) {
      preRef.current.scrollTop = preRef.current.scrollHeight
    }
  }, [content, autoScroll])

  return (
    <div className="flex flex-col flex-1 min-h-0">
      <div className="flex items-center gap-2 text-xs pb-2 shrink-0">
        <label className="text-slate-400 flex items-center gap-1.5 cursor-pointer">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
            className="h-3 w-3"
          />
          自动滚动
        </label>
        <span className="flex-1" />
        <button
          onClick={() => void refresh()}
          className="text-slate-400 hover:text-slate-200"
        >
          刷新
        </button>
      </div>
      {error && (
        <div className="mb-2 p-2 rounded bg-red-900/40 border border-red-700 text-red-300 text-xs font-mono">
          {error}
        </div>
      )}
      <pre
        ref={preRef}
        className="flex-1 min-h-0 overflow-auto bg-black/60 border border-slate-800 rounded p-3 text-xs font-mono text-slate-300 whitespace-pre-wrap break-all"
      >
        {content || <span className="text-slate-600">（尚无日志）</span>}
      </pre>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Monitor tab — iframe 嵌 monitor_smooth.html
// ---------------------------------------------------------------------------

function MonitorTab({ taskId }: { taskId: number }) {
  return (
    <div className="flex flex-col flex-1 min-h-0">
      <div className="flex items-center gap-2 text-xs pb-2 shrink-0">
        <span className="text-slate-500">嵌入 monitor_smooth.html</span>
        <span className="flex-1" />
        <a
          href={`/monitor_smooth.html?task_id=${taskId}`}
          target="_blank"
          rel="noopener"
          className="text-slate-400 hover:text-cyan-400"
        >
          独立窗口打开 ↗
        </a>
      </div>
      <iframe
        src={`/monitor_smooth.html?task_id=${taskId}`}
        title={`monitor-task-${taskId}`}
        className="flex-1 w-full border border-slate-800 rounded bg-slate-950"
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Outputs tab — 之前 OutputsModal 的逻辑搬过来
// ---------------------------------------------------------------------------

function OutputsTab({ taskId }: { taskId: number }) {
  const { toast } = useToast()
  const [data, setData] = useState<TaskOutputs | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [refreshKey, setRefreshKey] = useState(0)

  useEffect(() => {
    let alive = true
    void api
      .getTaskOutputs(taskId)
      .then((r) => alive && setData(r))
      .catch((e) => alive && setError(String(e)))
    return () => {
      alive = false
    }
  }, [taskId, refreshKey])

  const sortedFiles = useMemo(
    () =>
      data
        ? [...data.files].sort((a, b) => b.mtime - a.mtime)
        : [],
    [data]
  )

  const openFolder = async () => {
    setBusy(true)
    try {
      const r = await api.openTaskFolder(taskId)
      toast(`已打开 ${r.opened}`, 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const copyPath = async () => {
    if (!data?.output_dir) return
    try {
      await navigator.clipboard.writeText(data.output_dir)
      toast('路径已复制', 'success')
    } catch {
      toast('复制失败（浏览器拒绝）', 'error')
    }
  }

  return (
    <div className="flex flex-col flex-1 min-h-0 gap-2">
      {data?.output_dir ? (
        <div className="flex items-center gap-2 text-xs shrink-0 border-b border-slate-800 pb-2">
          <span className="text-slate-500 shrink-0">目录</span>
          <code className="flex-1 min-w-0 truncate text-slate-200 font-mono">
            {data.output_dir}
          </code>
          <button
            onClick={copyPath}
            className="px-2 py-0.5 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-800 shrink-0"
            title="复制路径"
          >
            复制
          </button>
          {data.supports_open_folder ? (
            <button
              onClick={openFolder}
              disabled={busy || !data.exists}
              className="px-2 py-0.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-200 disabled:opacity-50 shrink-0"
              title="在文件管理器打开"
            >
              📁 打开
            </button>
          ) : (
            <span
              className="text-[10px] text-slate-500 shrink-0"
              title="服务端不在本机，远程打开文件夹无意义"
            >
              （远程，无法打开）
            </span>
          )}
          <button
            onClick={() => setRefreshKey((k) => k + 1)}
            className="text-slate-400 hover:text-slate-200 shrink-0"
          >
            刷新
          </button>
          {data.exists && data.files.length > 0 && (
            <a
              href={api.taskOutputsZipUrl(taskId)}
              download={`task_${taskId}_outputs.zip`}
              className="px-2 py-0.5 rounded bg-cyan-700 hover:bg-cyan-600 text-white shrink-0"
              title="打包 output 目录里全部文件为 zip 下载"
            >
              ⤓ 全量 zip
            </a>
          )}
        </div>
      ) : (
        <p className="text-slate-500 text-xs shrink-0">
          该任务没有 project / version 关联，找不到 output 目录
        </p>
      )}

      <div className="flex-1 min-h-0 overflow-y-auto">
        {error ? (
          <p className="text-red-300 font-mono text-xs">{error}</p>
        ) : !data ? (
          <p className="text-slate-500 text-xs">加载...</p>
        ) : !data.exists ? (
          <p className="text-amber-300 text-xs">目录不存在</p>
        ) : sortedFiles.length === 0 ? (
          <p className="text-slate-500 text-xs">目录为空</p>
        ) : (
          <table className="w-full text-xs">
            <thead className="text-slate-500 border-b border-slate-800 sticky top-0 bg-slate-900">
              <tr>
                <th className="px-2 py-1 text-left font-normal">文件</th>
                <th className="px-2 py-1 text-right font-normal">大小</th>
                <th className="px-2 py-1 text-right font-normal">修改时间</th>
                <th className="px-2 py-1 text-right font-normal">操作</th>
              </tr>
            </thead>
            <tbody>
              {sortedFiles.map((f) => (
                <tr
                  key={f.name}
                  className="border-b border-slate-800/60 last:border-0 hover:bg-slate-800/30"
                >
                  <td className="px-2 py-1.5">
                    <code className="font-mono text-slate-200">{f.name}</code>
                    {f.is_lora && (
                      <span className="ml-2 text-[10px] px-1 py-0.5 rounded bg-emerald-700/40 text-emerald-200">
                        LoRA
                      </span>
                    )}
                  </td>
                  <td className="px-2 py-1.5 text-right text-slate-400 font-mono">
                    {fmtBytes(f.size)}
                  </td>
                  <td className="px-2 py-1.5 text-right text-slate-500 font-mono text-[11px]">
                    {fmtTime(f.mtime)}
                  </td>
                  <td className="px-2 py-1.5 text-right">
                    <a
                      href={api.taskOutputDownloadUrl(taskId, f.name)}
                      download={f.name}
                      className="text-cyan-400 hover:text-cyan-300"
                    >
                      ↓ 下载
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Confirm dialog — 替代 native window.confirm，破坏性操作专用
// ---------------------------------------------------------------------------

function ConfirmDialog({
  title,
  message,
  confirmLabel = '确认',
  cancelLabel = '取消',
  danger = false,
  busy = false,
  onConfirm,
  onCancel,
}: {
  title: string
  message: React.ReactNode
  confirmLabel?: string
  cancelLabel?: string
  danger?: boolean
  busy?: boolean
  onConfirm: () => void
  onCancel: () => void
}) {
  return (
    <div
      onClick={onCancel}
      className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="bg-slate-900 border border-slate-700 rounded-lg shadow-2xl w-full max-w-md"
      >
        <header className="px-4 py-3 border-b border-slate-700">
          <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
        </header>
        <div className="px-4 py-3 text-xs text-slate-300">{message}</div>
        <footer className="px-4 py-3 border-t border-slate-700 flex items-center gap-2 justify-end text-xs">
          <button
            onClick={onCancel}
            disabled={busy}
            className="px-3 py-1.5 rounded text-slate-400 hover:text-slate-200 disabled:opacity-50"
          >
            {cancelLabel}
          </button>
          <button
            onClick={onConfirm}
            disabled={busy}
            className={
              'px-3 py-1.5 rounded text-white disabled:opacity-50 ' +
              (danger
                ? 'bg-red-700 hover:bg-red-600'
                : 'bg-cyan-600 hover:bg-cyan-500')
            }
          >
            {busy ? '...' : confirmLabel}
          </button>
        </footer>
      </div>
    </div>
  )
}
