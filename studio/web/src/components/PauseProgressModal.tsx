// PauseProgressModal —— ADR 0006 PR-4 §4.3 暂停过程 modal。
//
// 设计动机：用户点"暂停"后 handle_interrupt 写盘要几秒到十几秒（state +
// LoRA + wandb finish），期间 UI 必须给透明反馈、阻止误操作。点暂停就锁屏
// modal 全程引导，看到 __EVENT__:pause_state 才算"暂停完成"（rc=0 不够 —
// rc 在 Windows wrapper 改写场景下不可靠）。
//
// 状态机（modal 内自管，跟 task.status 解耦）：
//   - 'saving'：发了 pause 请求，等子进程 emit pause_state
//   - 'saved'：成功，task_state_changed → paused，关闭 modal + toast
//   - 'timeout'：30s 还没收到事件，弹三选一（再等 / 强退保进度 / 终止）
//   - 'failed'：子进程异常退出（rc != 0 + status='failed'）
//
// SSE 订阅：用全局 useEventStream，过滤同 task_id 的 pause_state /
// task_state_changed / pause_failed 事件。
import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api } from '../api/client'
import { useEventStream, type StudioEvent } from '../lib/useEventStream'
import { useToast } from './Toast'

const TIMEOUT_MS = 30_000

type PhaseState =
  | { phase: 'saving' }
  | { phase: 'saved'; step: number }
  | { phase: 'timeout' }
  | { phase: 'failed'; exitCode: number | null }

export interface PauseProgressModalProps {
  taskId: number
  /** 任务显示名（modal title 用） */
  taskName?: string
  /** modal 关闭回调 */
  onClose: () => void
}

export function PauseProgressModal({ taskId, taskName, onClose }: PauseProgressModalProps) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [state, setState] = useState<PhaseState>({ phase: 'saving' })
  const [elapsedSec, setElapsedSec] = useState(0)
  const startedAt = useRef(Date.now())
  // 跨 onEvent / timer 共享 phase，避免闭包陷阱。
  const phaseRef = useRef<PhaseState['phase']>('saving')
  phaseRef.current = state.phase

  // ── elapsed 计数（用于文案显示 + timeout 判定）
  useEffect(() => {
    const id = window.setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - startedAt.current) / 1000))
    }, 1000)
    return () => window.clearInterval(id)
  }, [])

  // ── 30s 超时 → 升级到 timeout 状态（仅 saving 阶段触发）
  useEffect(() => {
    const id = window.setTimeout(() => {
      if (phaseRef.current === 'saving') {
        setState({ phase: 'timeout' })
      }
    }, TIMEOUT_MS)
    return () => window.clearTimeout(id)
  }, [])

  // ── SSE 监听
  useEventStream(
    useCallback((evt: StudioEvent) => {
      if (evt.task_id !== taskId) return
      if (evt.type === 'pause_state') {
        // 子进程已落盘 .pt + snapshot → 标 saved；step 来自 payload（PR-2 emit）
        const step = typeof evt.step === 'number' ? evt.step : 0
        setState({ phase: 'saved', step })
        return
      }
      if (evt.type === 'task_state_changed') {
        if (evt.status === 'paused' && phaseRef.current !== 'saved') {
          // 兜底：万一 pause_state 事件丢了，task_state_changed='paused' 也算成功
          setState({ phase: 'saved', step: 0 })
        } else if (evt.status === 'failed' || evt.status === 'canceled') {
          // 子进程异常退出 / 用户从外面 cancel 了 → 失败态
          if (phaseRef.current === 'saving' || phaseRef.current === 'timeout') {
            setState({ phase: 'failed', exitCode: null })
          }
        }
      }
    }, [taskId]),
  )

  // ── 用户操作
  const handleClose = () => onClose()

  const handleWaitMore = () => {
    // 再等 30 秒：重置 timer + 把状态回退到 saving
    startedAt.current = Date.now()
    setElapsedSec(0)
    setState({ phase: 'saving' })
  }

  const handleForceCancelKeep = async () => {
    // ADR §4.3 "强制取消保存进度"：发硬中断；如果磁盘上 pause 文件已落盘
    // 仍标 paused，否则降级 canceled — 这条降级逻辑在 supervisor _finish_slot
    // 三元分流里已实现，前端只需调 cancel。
    try {
      await api.cancelTask(taskId)
      toast(t('queue.pauseSent'), 'info')
    } catch (e) {
      toast(String(e), 'error')
    }
    onClose()
  }

  const handleTerminate = async () => {
    try {
      await api.cancelTask(taskId)
    } catch (e) {
      toast(String(e), 'error')
    }
    onClose()
  }

  const handleViewLogs = () => {
    // 打开 QueueDetail 日志 tab
    window.location.href = `/queue/${taskId}?tab=logs`
  }

  // ── 渲染
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="pause-progress-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-md"
      data-testid="pause-progress-modal"
    >
      <div className="w-[90%] max-w-[520px] flex flex-col gap-6 p-7 bg-elevated border border-dim rounded-lg shadow-xl">
        <h2
          id="pause-progress-title"
          className="m-0 text-lg font-semibold text-fg-primary"
        >
          {t('queue.pauseProgress.title', { id: taskId })}
          {taskName ? ` · ${taskName}` : ''}
        </h2>

        {state.phase === 'saving' && (
          <div className="flex flex-col gap-3" data-testid="pause-saving">
            <div className="flex items-center gap-3">
              <span className="inline-block w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-fg-secondary">{t('queue.pauseProgress.saving')}</span>
            </div>
            <span className="text-xs text-fg-tertiary">
              {t('queue.pauseProgress.elapsedSeconds', { n: elapsedSec })}
            </span>
          </div>
        )}

        {state.phase === 'saved' && (
          <div className="flex flex-col gap-3" data-testid="pause-saved">
            <span className="text-sm text-ok">
              ✓ {t('queue.pauseProgress.saved', { step: state.step })}
            </span>
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={handleClose}
                className="px-3 py-1.5 text-sm rounded bg-accent text-accent-on hover:bg-accent-hover"
              >
                {t('queue.pauseProgress.ok')}
              </button>
            </div>
          </div>
        )}

        {state.phase === 'timeout' && (
          <div className="flex flex-col gap-4" data-testid="pause-timeout">
            <span className="text-sm font-medium text-warn">
              ⚠️ {t('queue.pauseProgress.timeoutTitle')}
            </span>
            <span className="text-xs text-fg-secondary">
              {t('queue.pauseProgress.timeoutDesc', { n: elapsedSec })}
            </span>
            <div className="flex flex-col gap-2">
              <button
                type="button"
                onClick={handleWaitMore}
                className="px-3 py-2 text-sm rounded border border-dim bg-surface hover:bg-surface-hover text-left"
              >
                {t('queue.pauseProgress.waitMore')}
              </button>
              <button
                type="button"
                onClick={handleForceCancelKeep}
                className="px-3 py-2 text-sm rounded border border-dim bg-surface hover:bg-surface-hover text-left"
                title={t('queue.pauseProgress.forceCancelKeepHint')}
              >
                {t('queue.pauseProgress.forceCancelKeep')}
              </button>
              <button
                type="button"
                onClick={handleTerminate}
                className="px-3 py-2 text-sm rounded border border-err text-err hover:bg-err-soft text-left"
                title={t('queue.pauseProgress.terminateHint')}
              >
                {t('queue.pauseProgress.terminate')}
              </button>
            </div>
          </div>
        )}

        {state.phase === 'failed' && (
          <div className="flex flex-col gap-3" data-testid="pause-failed">
            <span className="text-sm font-medium text-err">
              ✗ {t('queue.pauseProgress.failedTitle')}
            </span>
            <span className="text-xs text-fg-secondary">
              {t('queue.pauseProgress.failedDesc', { code: state.exitCode ?? '?' })}
            </span>
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={handleViewLogs}
                className="px-3 py-1.5 text-sm rounded border border-dim bg-surface hover:bg-surface-hover"
              >
                {t('queue.pauseProgress.viewLogs')}
              </button>
              <button
                type="button"
                onClick={handleClose}
                className="px-3 py-1.5 text-sm rounded bg-accent text-accent-on hover:bg-accent-hover"
              >
                {t('queue.pauseProgress.ok')}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
