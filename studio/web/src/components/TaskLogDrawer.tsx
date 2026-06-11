import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

export type LogSourceStatus =
  | 'pending'
  | 'running'
  | 'done'
  | 'failed'
  | 'canceled'
  | 'paused'

/** 一条可回放的任务日志流（job / task / 前端合成日志通吃）。 */
export interface LogSource {
  key: string
  label: string
  status: LogSourceStatus
  lines: string[]
  /** 秒级 epoch；多 source 都终态时按它选「最近」，也用于条上的耗时显示。 */
  startedAt?: number | null
  finishedAt?: number | null
  /** 缺省 = 不可取消（如去重扫描的前端合成日志）。 */
  onCancel?: () => void
}

const STATUS_BADGE: Record<LogSourceStatus, string> = {
  pending: 'badge badge-neutral',
  running: 'badge badge-warn',
  done: 'badge badge-ok',
  failed: 'badge badge-err',
  canceled: 'badge badge-neutral',
  paused: 'badge badge-neutral',
}

const isLiveStatus = (s: LogSourceStatus) => s === 'pending' || s === 'running'

/** 多 source 单显（issue #251 拍板）：活着的优先，否则最近启动的。
 *  旧任务的产物已被新任务覆盖，历史日志不提供多入口。 */
function pickActive(sources: LogSource[]): LogSource | null {
  if (sources.length === 0) return null
  const live = sources.find((s) => isLiveStatus(s.status))
  if (live) return live
  return [...sources].sort((a, b) => (b.startedAt ?? 0) - (a.startedAt ?? 0))[0]
}

/**
 * 任务日志抽屉 —— 全 app 统一的任务进度/日志 UI（issue #251）。
 *
 * 三态：
 * - 无任务：整体隐藏
 * - 收起条（默认）：一行 status 徽标 + 最后一行日志 + 耗时，点击展开
 * - 展开：日志面板 overlay 在内容区上方（不挤压页面布局），自动滚底
 *
 * 开合状态机（手动开合随时生效）：
 * - 进入 live（含挂载即 running 的回放场景）→ 自动展开
 * - 任务结束**不**自动收起 —— 用户要回看结果/错误；只有切页（组件卸载）
 *   或手动点击才收
 * - 挂载即终态（历史回放）→ 默认收起
 *
 * 由 StepShell 统一挂载（`logSources` prop），页面只声明 source。
 */
export default function TaskLogDrawer({
  sources,
}: {
  sources: Array<LogSource | null | undefined | false>
}) {
  const { t } = useTranslation()
  const list = sources.filter((s): s is LogSource => !!s)
  const active = pickActive(list)

  const [expanded, setExpanded] = useState(false)
  const preRef = useRef<HTMLPreElement>(null)
  const prevRef = useRef<{ key: string; status: LogSourceStatus } | null>(null)

  const activeKey = active?.key ?? null
  const activeStatus = active?.status ?? null
  useEffect(() => {
    if (!activeKey || !activeStatus) return
    const prev = prevRef.current
    const wasLive = prev?.key === activeKey && isLiveStatus(prev.status)
    if (isLiveStatus(activeStatus) && !wasLive) setExpanded(true)
    prevRef.current = { key: activeKey, status: activeStatus }
  }, [activeKey, activeStatus])

  // live 时 1s tick 刷新耗时显示（同原 JobProgress）
  const live = !!activeStatus && isLiveStatus(activeStatus)
  const [, setTick] = useState(0)
  useEffect(() => {
    if (!live) return
    const id = window.setInterval(() => setTick((n) => n + 1), 1000)
    return () => window.clearInterval(id)
  }, [live])

  // 展开时跟随日志滚到底
  const lineCount = active?.lines.length ?? 0
  useEffect(() => {
    if (expanded && preRef.current) {
      preRef.current.scrollTop = preRef.current.scrollHeight
    }
  }, [expanded, lineCount])

  if (!active) return null

  const elapsed = active.startedAt
    ? (active.finishedAt ?? Date.now() / 1000) - active.startedAt
    : null
  const lastLine = active.lines[active.lines.length - 1] ?? ''

  return (
    <div className="relative shrink-0 mt-2">
      {expanded && (
        <div className="absolute bottom-full inset-x-0 z-30 mb-1 rounded-md border border-subtle bg-surface shadow-lg overflow-hidden flex flex-col">
          <pre
            ref={preRef}
            className="m-0 px-3 py-2 text-[11px] leading-relaxed font-mono text-fg-secondary bg-sunken max-h-[min(40vh,480px)] overflow-y-auto whitespace-pre-wrap break-words"
          >
            {active.lines.length === 0
              ? t('jobProgress.waitingLogs')
              : active.lines.slice(-1000).join('\n')}
          </pre>
        </div>
      )}
      <div
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        onClick={() => setExpanded((v) => !v)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            setExpanded((v) => !v)
          }
        }}
        className="cursor-pointer select-none rounded-md border border-subtle bg-surface px-2.5 py-1.5 flex items-center gap-2 text-sm"
      >
        <span
          className={`inline-block transition-transform text-fg-tertiary w-3 ${expanded ? 'rotate-90' : ''}`}
        >
          ▸
        </span>
        <span className={STATUS_BADGE[active.status]}>{active.status}</span>
        <span className="text-fg-secondary shrink-0">{active.label}</span>
        {elapsed != null && elapsed > 0 && (
          <span className="text-fg-tertiary text-xs shrink-0">· {Math.round(elapsed)}s</span>
        )}
        <span className="mono truncate flex-1 min-w-0 text-fg-secondary text-xs">{lastLine}</span>
        {live && active.onCancel && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              active.onCancel?.()
            }}
            className="btn btn-ghost btn-sm text-err"
          >
            {t('common.cancel')}
          </button>
        )}
      </div>
    </div>
  )
}
