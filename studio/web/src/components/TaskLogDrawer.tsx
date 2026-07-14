import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

export type LogSourceStatus =
  | 'pending'
  | 'running'
  | 'done'
  | 'failed'
  | 'canceled'
  | 'paused'
  | 'scheduled'

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
  /** failed 时 header 右侧显示重试按钮；缺省 = 不可重试。 */
  onRetry?: () => void
}

const STATUS_BADGE: Record<LogSourceStatus, string> = {
  pending: 'badge badge-neutral',
  running: 'badge badge-warn',
  done: 'badge badge-ok',
  failed: 'badge badge-err',
  canceled: 'badge badge-neutral',
  paused: 'badge badge-neutral',
  // 0.17 P-B：scheduled（计划任务）还没跑，无日志，视觉同 pending。
  scheduled: 'badge badge-neutral',
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
 * 形态：页面级 footer。收起时一行 header 全宽贴在页面最底（status 徽标 +
 * 最后一行日志 + 耗时）；点击后日志面板从 header 下方升起（200ms 高度动画，
 * overlay 不挤压页面布局），header 骑在面板顶上充当与内容区的分隔条。
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
    <>
      {/* 占位：与 footer header 同高，让页面内容不被贴底的 header 盖住 */}
      <div className="shrink-0 h-9" aria-hidden />
      {/* footer 抽屉本体：贴页面底、全宽、无圆角无 margin；anchored bottom，
          body 高度动画 0 ↔ 40vh 时 header 随抽屉上升，充当内容/日志分隔条 */}
      <div
        className={`absolute bottom-0 inset-x-0 z-30 flex flex-col ${expanded ? 'shadow-2xl' : ''}`}
      >
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
          className="h-9 shrink-0 cursor-pointer select-none border-t-2 border-accent bg-surface px-4 flex items-center gap-2 text-sm"
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
          {active.status === 'failed' && active.onRetry && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                active.onRetry?.()
              }}
              className="btn btn-ghost btn-sm text-accent"
            >
              {t('common.retry')}
            </button>
          )}
        </div>
        <div
          data-testid="log-drawer-body"
          className="overflow-hidden bg-sunken transition-[height] duration-200 ease-out"
          style={{ height: expanded ? '40vh' : '0px' }}
        >
          <pre
            ref={preRef}
            className="m-0 h-full px-4 py-2 text-[11px] leading-relaxed font-mono text-fg-secondary overflow-y-auto whitespace-pre-wrap break-words"
          >
            {active.lines.length === 0
              ? t('jobProgress.waitingLogs')
              : active.lines.slice(-1000).join('\n')}
          </pre>
        </div>
      </div>
    </>
  )
}
