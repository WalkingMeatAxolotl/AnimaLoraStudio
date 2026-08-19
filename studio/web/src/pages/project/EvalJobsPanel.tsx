// 概览「评估」tab —— 该版本的评估作业列表。
//
// 和旁边的「任务」一样是一张 task table：一次评估 = 一个作业（#465），结果不在这里
// 铺开，点进作业详情看（那里有指标 / 样图两个 tab）。发起动作在表格上方 ——
// 「创建新评估」弹 modal 填参数。
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type EvalSessionSummary } from '../../api/client'
import CreateEvalModal from '../../components/CreateEvalModal'
import TaskLogDrawer, { type LogSource, type LogSourceStatus } from '../../components/TaskLogDrawer'
import { useEventStream } from '../../lib/useEventStream'

const STATUS_BADGE: Record<string, string> = {
  pending: 'neutral', running: 'accent', done: 'ok',
  partial: 'warn', failed: 'err', canceled: 'neutral',
}

const TRIGGER_LABEL: Record<string, string> = {
  manual: '手动', after_training: '训练后自动',
}

function fmtTime(ts: number | null | undefined): string {
  return ts ? new Date(ts * 1000).toLocaleString() : '—'
}

/** 正在跑的那次评估的日志 —— 概览页发起后就地能看，不必先跳去作业详情。 */
function useRunningEvalLog(sessions: EvalSessionSummary[]): LogSource | null {
  const active = useMemo(
    () => sessions.find((s) => s.status === 'pending' || s.status === 'running') ?? null,
    [sessions],
  )
  const taskId = active?.task_id ?? null
  const [baseLines, setBaseLines] = useState<string[]>([])
  const [liveLines, setLiveLines] = useState<string[]>([])

  useEffect(() => {
    if (!taskId) { setBaseLines([]); setLiveLines([]); return }
    let alive = true
    void api.getLog(taskId, { tail: 1000 })
      .then((log) => { if (alive) { setBaseLines(log.lines.map((l) => l.text)); setLiveLines([]) } })
      .catch(() => {})
    return () => { alive = false }
  }, [taskId])

  useEventStream((evt) => {
    const isMine =
      (evt.type === 'job_log_appended' && evt.job_id === taskId)
      || (evt.type === 'task_log_appended' && evt.task_id === taskId)
    if (!isMine || taskId == null) return
    const text = typeof evt.text === 'string' ? evt.text : ''
    if (text) setLiveLines((prev) => [...prev, ...text.split('\n')])
  })

  return useMemo(() => {
    if (!active) return null
    const status: LogSourceStatus = active.status === 'running' ? 'running' : 'pending'
    return {
      key: `eval-session-${active.id}`,
      label: `评估 #${active.id}`,
      status,
      lines: [...baseLines, ...liveLines],
    }
  }, [active, baseLines, liveLines])
}

export default function EvalJobsPanel({
  pid, vid,
}: {
  pid: number
  vid: number | null
}) {
  const navigate = useNavigate()
  const [sessions, setSessions] = useState<EvalSessionSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)

  const load = useCallback(async () => {
    if (!vid) return
    try {
      const { sessions: list } = await api.listEvalSessions(pid, vid)
      setSessions(list)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setSessions([])
    }
  }, [pid, vid])

  useEffect(() => { setSessions(null); void load() }, [load])

  // 有在跑的就轮询（阶段推进不发独立事件，靠拉）
  const hasActive = (sessions ?? []).some(
    (s) => s.status === 'pending' || s.status === 'running',
  )
  useEffect(() => {
    if (!hasActive) return
    const id = window.setInterval(() => void load(), 5000)
    return () => window.clearInterval(id)
  }, [hasActive, load])

  const logSource = useRunningEvalLog(sessions ?? [])

  if (!vid) {
    return (
      <div className="p-6 text-fg-tertiary text-sm italic">
        先选一个版本。评估的对象是该版本 output/ 下的 LoRA 文件。
      </div>
    )
  }

  return (
    <div className="relative flex flex-col flex-1 min-h-0">
      <div className="flex-1 min-h-0 overflow-y-auto p-6 flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold">评估</span>
          <span className="text-xs text-fg-tertiary">
            一次评估一个作业；点进去看指标和样图
          </span>
          <span className="flex-1" />
          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={() => setCreating(true)}
          >
            创建新评估
          </button>
        </div>

        {error && (
          <div className="rounded-md border border-err bg-err-soft px-3 py-2 text-sm text-err">
            评估列表读取失败：{error}
          </div>
        )}

        {sessions == null ? (
          <div className="text-fg-tertiary text-sm">读取中…</div>
        ) : sessions.length === 0 ? (
          <div className="text-fg-tertiary text-sm italic">
            还没有评估。点「创建新评估」选 LoRA 文件。
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-fg-tertiary text-xs">
              <tr className="border-b border-subtle">
                <th className="text-left py-2 px-3 font-normal">评估</th>
                <th className="text-left py-2 px-3 font-normal">状态</th>
                <th className="text-left py-2 px-3 font-normal">被测对象</th>
                <th className="text-left py-2 px-3 font-normal">触发</th>
                <th className="text-left py-2 px-3 font-normal">创建</th>
                <th className="text-left py-2 px-3 font-normal">结束</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr
                  key={s.id}
                  className="border-b border-subtle cursor-pointer hover:bg-overlay"
                  onClick={() => s.task_id && navigate(`/queue/${s.task_id}`)}
                  title={s.task_id ? `作业 #${s.task_id}` : '该评估没有关联作业'}
                >
                  <td className="py-2 px-3 font-mono">#{s.id}</td>
                  <td className="py-2 px-3">
                    <span className={`badge badge-${STATUS_BADGE[s.status] ?? 'neutral'}`}>
                      {s.status === 'running' && <span className="dot dot-running" />}
                      {s.status}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-fg-secondary text-xs font-mono">
                    {s.candidate_count} 个 · {s.validation_images} 张验证图
                  </td>
                  <td className="py-2 px-3 text-fg-tertiary text-xs">
                    {TRIGGER_LABEL[s.trigger] ?? s.trigger}
                  </td>
                  <td className="py-2 px-3 text-fg-tertiary text-xs">{fmtTime(s.created_at)}</td>
                  <td className="py-2 px-3 text-fg-tertiary text-xs">{fmtTime(s.finished_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* 概览发起后就地看日志（issue #251 统一抽屉） */}
      <TaskLogDrawer sources={[logSource]} />

      {creating && (
        <CreateEvalModal
          pid={pid}
          vid={vid}
          onClose={() => setCreating(false)}
          onCreated={(sid) => {
            setCreating(false)
            void load()
            void sid
          }}
        />
      )}
    </div>
  )
}
