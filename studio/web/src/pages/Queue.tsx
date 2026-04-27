import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type ConfigSummary, type Task, type TaskStatus } from '../api/client'
import { useToast } from '../components/Toast'
import { useEventStream } from '../lib/useEventStream'

async function downloadJson(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: 'application/json',
  })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

async function pickJsonFile(): Promise<unknown | null> {
  return new Promise((resolve) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json,application/json'
    input.onchange = async () => {
      const f = input.files?.[0]
      if (!f) {
        resolve(null)
        return
      }
      try {
        const text = await f.text()
        resolve(JSON.parse(text))
      } catch {
        alert('JSON 解析失败')
        resolve(null)
      }
    }
    input.click()
  })
}

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

function fmtTime(ts: number | null): string {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleString('zh-CN', { hour12: false })
}

function fmtDuration(start: number | null, end: number | null): string {
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

export default function QueuePage() {
  const [tasks, setTasks] = useState<Task[]>([])
  const [configs, setConfigs] = useState<ConfigSummary[]>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const reloadTimer = useRef<number | null>(null)
  const { toast } = useToast()

  const reload = useCallback(async () => {
    try {
      const [items, cfgList] = await Promise.all([
        api.listQueue(),
        api.listConfigs(),
      ])
      setTasks(items)
      setConfigs(cfgList)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  // SSE：收到 task_state_changed 事件就重拉。多事件合并到 100ms 内只刷一次。
  useEventStream((evt) => {
    if (evt.type !== 'task_state_changed') return
    if (reloadTimer.current) return
    reloadTimer.current = window.setTimeout(() => {
      reloadTimer.current = null
      void reload()
    }, 100)
  })

  // running 任务存在时每 2s 刷新一次时长显示（time-only re-render）
  useEffect(() => {
    void reload()
  }, [reload])
  useEffect(() => {
    const hasRunning = tasks.some((t) => t.status === 'running')
    if (!hasRunning) return
    const tick = window.setInterval(() => setTasks((ts) => [...ts]), 2000)
    return () => window.clearInterval(tick)
  }, [tasks])

  const enqueue = async (configName: string) => {
    setBusy(true)
    setError(null)
    try {
      await api.enqueue({ config_name: configName })
      await reload()
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  const cancel = async (id: number) => {
    if (!window.confirm('取消任务？')) return
    setBusy(true)
    try {
      await api.cancelTask(id)
      await reload()
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  const retry = async (id: number) => {
    setBusy(true)
    try {
      await api.retryTask(id)
      await reload()
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  const remove = async (id: number) => {
    if (!window.confirm('删除任务记录？')) return
    setBusy(true)
    try {
      await api.deleteTask(id)
      await reload()
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-4">
      {/* 入队栏 */}
      <section className="rounded-xl border border-slate-700 bg-slate-800/40 p-4">
        <h2 className="text-sm font-semibold text-slate-200 mb-3">入队新任务</h2>
        {configs.length === 0 ? (
          <p className="text-slate-500 text-sm">
            还没有配置。先去{' '}
            <Link
              to="/configs"
              className="text-cyan-400 hover:underline"
            >
              配置
            </Link>{' '}
            页面新建一个。
          </p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {configs.map((c) => (
              <button
                key={c.name}
                disabled={busy}
                onClick={() => void enqueue(c.name)}
                className="px-3 py-1.5 rounded text-sm bg-slate-700 hover:bg-cyan-600
                  disabled:opacity-50 transition-colors"
              >
                + {c.name}
              </button>
            ))}
          </div>
        )}
      </section>

      {error && (
        <div className="p-3 rounded bg-red-900/40 border border-red-700 text-red-300 text-sm font-mono">
          {error}
        </div>
      )}

      {/* 队列表格 */}
      <section className="rounded-xl border border-slate-700 bg-slate-800/40 overflow-hidden">
        <header className="px-4 py-3 border-b border-slate-700 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-200">
            队列 <span className="text-slate-500 ml-1">({tasks.length})</span>
          </h2>
          <div className="flex gap-2 text-xs">
            <button
              disabled={busy || tasks.length === 0}
              onClick={async () => {
                try {
                  const data = await api.exportQueue()
                  await downloadJson(
                    `queue_${new Date()
                      .toISOString()
                      .slice(0, 19)
                      .replace(/[:T]/g, '-')}.json`,
                    data
                  )
                } catch (e) {
                  setError(String(e))
                }
              }}
              className="text-slate-400 hover:text-slate-200 disabled:opacity-40"
            >
              导出
            </button>
            <button
              disabled={busy}
              onClick={async () => {
                const payload = await pickJsonFile()
                if (!payload) return
                setBusy(true)
                try {
                  const r = await api.importQueue(payload)
                  const renamed = Object.keys(r.renamed).length
                  toast(
                    `已导入 ${r.imported_count} 个任务` +
                      (renamed ? `（${renamed} 个改名）` : ''),
                    'success'
                  )
                  await reload()
                } catch (e) {
                  setError(String(e))
                } finally {
                  setBusy(false)
                }
              }}
              className="text-slate-400 hover:text-slate-200 disabled:opacity-40"
            >
              导入
            </button>
            <button
              onClick={() => void reload()}
              className="text-slate-400 hover:text-slate-200"
            >
              刷新
            </button>
          </div>
        </header>

        {tasks.length === 0 ? (
          <p className="px-4 py-8 text-center text-slate-500 text-sm">
            队列为空
          </p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-xs text-slate-500 border-b border-slate-800">
              <tr>
                <th className="px-3 py-2 text-left font-normal">#</th>
                <th className="px-3 py-2 text-left font-normal">名称 / 配置</th>
                <th className="px-3 py-2 text-left font-normal">状态</th>
                <th className="px-3 py-2 text-left font-normal">入队时间</th>
                <th className="px-3 py-2 text-left font-normal">运行时长</th>
                <th className="px-3 py-2 text-right font-normal">操作</th>
              </tr>
            </thead>
            <tbody>
              {tasks.map((t) => (
                <tr
                  key={t.id}
                  className="border-b border-slate-800 last:border-0 hover:bg-slate-800/30"
                >
                  <td className="px-3 py-2 text-slate-500 font-mono">{t.id}</td>
                  <td className="px-3 py-2">
                    <div className="text-slate-200">{t.name}</div>
                    <div className="text-xs text-slate-500 font-mono">
                      {t.config_name}.yaml
                    </div>
                  </td>
                  <td className="px-3 py-2">
                    <span
                      className={
                        'inline-block px-2 py-0.5 rounded text-xs ' +
                        STATUS_STYLE[t.status]
                      }
                    >
                      {STATUS_LABEL[t.status]}
                    </span>
                    {t.error_msg && (
                      <div
                        className="text-xs text-red-400 mt-1 truncate max-w-xs"
                        title={t.error_msg}
                      >
                        {t.error_msg}
                      </div>
                    )}
                  </td>
                  <td className="px-3 py-2 text-slate-400 text-xs">
                    {fmtTime(t.created_at)}
                  </td>
                  <td className="px-3 py-2 text-slate-400 text-xs">
                    {fmtDuration(t.started_at, t.finished_at)}
                  </td>
                  <td className="px-3 py-2 text-right space-x-1">
                    <Link
                      to={`/queue/${t.id}/log`}
                      className="text-xs text-slate-400 hover:text-cyan-400"
                    >
                      日志
                    </Link>
                    {(t.status === 'pending' || t.status === 'running') && (
                      <button
                        disabled={busy}
                        onClick={() => void cancel(t.id)}
                        className="text-xs text-slate-400 hover:text-amber-400 ml-2"
                      >
                        取消
                      </button>
                    )}
                    {(t.status === 'failed' ||
                      t.status === 'done' ||
                      t.status === 'canceled') && (
                      <>
                        <button
                          disabled={busy}
                          onClick={() => void retry(t.id)}
                          className="text-xs text-slate-400 hover:text-cyan-400 ml-2"
                        >
                          重试
                        </button>
                        <button
                          disabled={busy}
                          onClick={() => void remove(t.id)}
                          className="text-xs text-slate-400 hover:text-red-400 ml-2"
                        >
                          删除
                        </button>
                      </>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}
