// 概览「评估」tab —— 该版本的评估作业列表。
//
// 和旁边的「任务」一样是一张 task table：一次评估 = 一个作业（#465），结果不在这里
// 铺开，点进作业详情看（那里有指标 / 样图两个 tab）。发起动作在表格上方 ——
// 「创建新评估」弹 modal 填参数。
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { api, type EvalSessionInfo, type EvalSessionSummary } from '../../api/client'
import Alert from '../../components/Alert'
import Badge, { type BadgeTone } from '../../components/Badge'
import Button, { buttonClassName } from '../../components/Button'
import Card from '../../components/Card'
import CreateEvalModal from '../../components/CreateEvalModal'
import EmptyState from '../../components/EmptyState'
import TaskLogDrawer, { type LogSource, type LogSourceStatus } from '../../components/TaskLogDrawer'
import { useTaskLog } from '../../lib/useTaskLog'

const STATUS_TONE: Record<string, BadgeTone> = {
  pending: 'neutral', running: 'accent', done: 'success',
  partial: 'warning', failed: 'danger', canceled: 'neutral',
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
  const log = useTaskLog(active?.task_id ?? null)

  return useMemo(() => {
    if (!active) return null
    const status: LogSourceStatus = active.status === 'running' ? 'running' : 'pending'
    return {
      key: `eval-session-${active.id}`,
      label: `评估 #${active.id}`,
      status,
      lines: log.lines,
      downloadUrl: log.downloadUrl,
      hasMoreBefore: log.hasMoreBefore,
      loadingAll: log.loadingAll,
      onLoadAll: log.loadAll,
    }
  }, [active, log.lines, log.downloadUrl, log.hasMoreBefore, log.loadingAll, log.loadAll])
}

export default function EvalJobsPanel({
  pid, vid,
}: {
  pid: number
  vid: number | null
}) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [sessions, setSessions] = useState<EvalSessionSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [createdSession, setCreatedSession] = useState<EvalSessionInfo | null>(null)

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
      <EmptyState
        embedded
        size="sm"
        className="m-page"
        description="先选一个版本。评估的对象是该版本 output/ 下的 LoRA 文件。"
      />
    )
  }

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <div className="flex min-h-0 flex-1 flex-col gap-section overflow-y-auto p-page">
        <div className="flex items-center gap-related">
          <div className="min-w-0 flex-1">
            <h2 className="type-panel-title">评估</h2>
            <p className="type-page-description mt-1">一次评估一个作业；进入任务详情查看指标和样图。</p>
          </div>
          <Button variant="primary" size="sm" onClick={() => setCreating(true)}>
            创建新评估
          </Button>
        </div>

        {createdSession && (
          <Alert
            tone="success"
            size="sm"
            role="status"
            title={t('eval.createdTitle')}
            action={createdSession.task_id ? (
              <Link
                to={`/queue/${createdSession.task_id}`}
                className={buttonClassName({ variant: 'ghost', size: 'sm' })}
              >
                {t('eval.viewTask')}
              </Link>
            ) : undefined}
          >
            {t('eval.createdDescription', { id: createdSession.id })}
          </Alert>
        )}

        {error && (
          <Alert tone="danger" size="sm" role="alert">
            评估列表读取失败：{error}
          </Alert>
        )}

        {sessions == null ? (
          <div role="status" className="text-sm text-fg-secondary">读取中…</div>
        ) : sessions.length === 0 ? (
          <EmptyState
            embedded
            size="sm"
            description="还没有评估。选择 LoRA 文件创建第一项评估。"
          />
        ) : (
          <Card padding="none" className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="type-data-label text-fg-secondary">
                <tr className="border-b border-subtle">
                  <th className="px-field py-related text-left font-normal">评估</th>
                  <th className="px-field py-related text-left font-normal">状态</th>
                  <th className="px-field py-related text-left font-normal">被测对象</th>
                  <th className="px-field py-related text-left font-normal">触发</th>
                  <th className="px-field py-related text-left font-normal">创建</th>
                  <th className="px-field py-related text-left font-normal">结束</th>
                </tr>
              </thead>
              <tbody>
                {sessions.map((s) => (
                  <tr
                    key={s.id}
                    className={`border-b border-subtle last:border-0 hover:bg-overlay ${s.task_id ? 'cursor-pointer' : 'cursor-default'} ${createdSession?.id === s.id ? 'bg-accent-soft' : ''}`}
                    onClick={() => s.task_id && navigate(`/queue/${s.task_id}`)}
                    title={s.task_id ? `作业 #${s.task_id}` : '该评估没有关联作业'}
                  >
                    <td className="px-field py-related font-mono">#{s.id}</td>
                    <td className="px-field py-related">
                      <Badge
                        tone={STATUS_TONE[s.status] ?? 'neutral'}
                        size="sm"
                        active={s.status === 'running'}
                      >
                        {s.status}
                      </Badge>
                    </td>
                    <td className="px-field py-related font-mono text-xs text-fg-secondary">
                      {s.candidate_count} 个 · {s.validation_images} 张验证图
                    </td>
                    <td className="px-field py-related text-xs text-fg-secondary">
                      {TRIGGER_LABEL[s.trigger] ?? s.trigger}
                    </td>
                    <td className="px-field py-related text-xs text-fg-secondary">{fmtTime(s.created_at)}</td>
                    <td className="px-field py-related text-xs text-fg-secondary">{fmtTime(s.finished_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        )}
      </div>

      {/* 概览发起后就地看日志（issue #251 统一抽屉） */}
      <TaskLogDrawer sources={[logSource]} />

      {creating && (
        <CreateEvalModal
          pid={pid}
          vid={vid}
          onClose={() => setCreating(false)}
          onCreated={(session) => {
            setCreating(false)
            setCreatedSession(session)
            void load()
          }}
        />
      )}
    </div>
  )
}
