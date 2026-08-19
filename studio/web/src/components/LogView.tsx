import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'

import { isLineVisible, levelClass, parseLogLines, type LogLine } from '../lib/logLines'
import { useLogDebugDefault } from '../lib/logDebugPref'
import { useOptionalToast } from './Toast'

export type LogViewStatus = 'loading' | 'waiting' | 'live' | 'finished' | 'error'

export interface LogViewProps {
  /** 原文行（run.log / daemon ring / 前端合成），按行契约解析着色 */
  lines: readonly string[]
  /** waiting = 活着但还没有行；live = 在跑；finished = 终态；error 配 `error` 文案 */
  status?: LogViewStatus
  error?: string | null
  /** 顶部「加载更早」：由数据源提供（useTaskLog） */
  hasMoreBefore?: boolean
  loadingEarlier?: boolean
  onLoadEarlier?: () => void
  /** 出错重试 / 手动刷新 */
  onRefresh?: () => void
  /** 原始文件下载地址 */
  downloadUrl?: string | null
  /** 关掉工具栏（设置页里的小面板 / 只读回放） */
  toolbar?: boolean
  /** 工具栏右侧额外按钮（如 daemon 抽屉的「清屏」） */
  extraActions?: ReactNode
  /** 没有任何行时的占位文案；缺省按 status 选 */
  emptyText?: string
  /** 容器 class（高度由父级给：flex-1 min-h-0 或固定 h-*） */
  className?: string
  /** 每行渲染上限（默认不限；数据源已封顶 5000） */
  maxRender?: number
  /** 行头的 时间/级别/来源 列；小面板可关 */
  showMeta?: boolean
}

const META_TIME_CH = 12   // HH:MM:SS.mmm
const META_LEVEL_CH = 5
const META_LOGGER_CH = 22

function shortTime(t?: string): string {
  return t ?? ''
}

function Row({ line, showMeta }: { line: LogLine; showMeta: boolean }) {
  const cls = levelClass(line.level)
  if (!showMeta || line.kind !== 'header') {
    // 续行缩进到与行头 msg 对齐；老格式 / 裸前缀行不缩进
    const indent = showMeta && line.kind === 'continuation'
    return (
      <div
        className={`whitespace-pre-wrap break-all ${cls}`}
        style={indent ? { paddingLeft: `${META_TIME_CH + META_LEVEL_CH + META_LOGGER_CH + 3}ch` } : undefined}
      >
        {line.raw}
      </div>
    )
  }
  return (
    <div className="flex gap-[1ch] whitespace-pre-wrap break-all">
      <span className="shrink-0 text-fg-tertiary" style={{ width: `${META_TIME_CH}ch` }}>{shortTime(line.time)}</span>
      <span className={`shrink-0 font-semibold ${cls}`} style={{ width: `${META_LEVEL_CH}ch` }}>{(line.level ?? '').slice(0, 5)}</span>
      <span
        className="shrink-0 text-fg-tertiary overflow-hidden text-ellipsis whitespace-nowrap"
        style={{ width: `${META_LOGGER_CH}ch` }}
        title={line.logger}
      >
        {line.logger}
      </span>
      <span className={`min-w-0 flex-1 ${cls}`}>{line.msg}</span>
    </div>
  )
}

/**
 * 统一日志视图（docs/design/logging-target-state.md §3.3）。
 *
 * 所有「看一段任务 / 进程日志」的地方都用它：QueueDetail 日志 tab、TaskLogDrawer
 * 内容区、Generate 的 daemon 抽屉、评估面板。按行契约解析：行头 时间/级别/来源 +
 * 消息，续行缩进继承级别；ERROR/CRITICAL 红、WARNING 黄、DEBUG 弱化。
 *
 * 「调试」开关是**视图级**、不持久化，初值取全局默认（设置页「默认显示调试日志」，
 * 后端字段）；用户在视图里改过之后，全局默认再变也不回推（open question 3：不同步）。
 */
export default function LogView({
  lines,
  status = 'finished',
  error = null,
  hasMoreBefore = false,
  loadingEarlier = false,
  onLoadEarlier,
  onRefresh,
  downloadUrl = null,
  toolbar = true,
  extraActions,
  emptyText,
  className = '',
  maxRender,
  showMeta = true,
}: LogViewProps) {
  const { t } = useTranslation()
  const { toast } = useOptionalToast()
  const globalDefault = useLogDebugDefault()
  const [showDebug, setShowDebug] = useState<boolean>(globalDefault ?? false)
  const touchedRef = useRef(false)
  // 全局默认晚于挂载才到（首次 getSecrets）：用户没动过开关就采纳它
  useEffect(() => {
    if (!touchedRef.current && globalDefault !== null) setShowDebug(globalDefault)
  }, [globalDefault])

  const [autoScroll, setAutoScroll] = useState(true)
  const bodyRef = useRef<HTMLDivElement | null>(null)

  const parsed = useMemo(() => parseLogLines(lines), [lines])
  const visible = useMemo(() => {
    const v = parsed.filter((l) => isLineVisible(l, showDebug))
    return maxRender && v.length > maxRender ? v.slice(-maxRender) : v
  }, [parsed, showDebug, maxRender])
  const hiddenDebug = parsed.length - parsed.filter((l) => isLineVisible(l, showDebug)).length

  // 跟随到底
  useEffect(() => {
    if (!autoScroll) return
    const el = bodyRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [visible.length, autoScroll, showDebug])

  // 「加载更早」前插后保持视口位置不跳
  const prevHeightRef = useRef<number | null>(null)
  const handleLoadEarlier = useCallback(() => {
    const el = bodyRef.current
    prevHeightRef.current = el ? el.scrollHeight : null
    onLoadEarlier?.()
  }, [onLoadEarlier])
  useLayoutEffect(() => {
    const el = bodyRef.current
    const prev = prevHeightRef.current
    if (el && prev !== null && !loadingEarlier) {
      el.scrollTop += el.scrollHeight - prev
      prevHeightRef.current = null
    }
  }, [loadingEarlier, lines.length])

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(visible.map((l) => l.raw).join('\n'))
      toast(t('logView.copied'), 'success')
    } catch {
      toast(t('logView.copyFailed'), 'error')
    }
  }

  const empty = lines.length === 0
  const placeholder = emptyText
    ?? (status === 'loading' ? t('logView.loading')
      : status === 'waiting' || status === 'live' ? t('logView.waiting')
        : t('logView.empty'))

  return (
    <div className={`flex flex-col min-h-0 ${className}`} data-testid="log-view">
      {toolbar && (
        <div className="flex items-center gap-3 text-xs pb-2 shrink-0 flex-wrap">
          <label className="text-fg-tertiary flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showDebug}
              onChange={(e) => { touchedRef.current = true; setShowDebug(e.target.checked) }}
              style={{ width: 14, height: 14, accentColor: 'var(--accent)' }}
            />
            {t('logView.debug')}
            {!showDebug && hiddenDebug > 0 && (
              <span className="text-fg-tertiary">（{t('logView.hiddenDebug', { n: hiddenDebug })}）</span>
            )}
          </label>
          <label className="text-fg-tertiary flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              style={{ width: 14, height: 14, accentColor: 'var(--accent)' }}
            />
            {t('logView.autoScroll')}
          </label>
          <span className="flex-1" />
          {extraActions}
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => void handleCopy()} disabled={visible.length === 0}>
            {t('logView.copy')}
          </button>
          {downloadUrl && (
            <a className="btn btn-ghost btn-sm" href={downloadUrl} download>
              {t('logView.download')}
            </a>
          )}
          {onRefresh && (
            <button type="button" className="btn btn-ghost btn-sm" onClick={onRefresh}>
              {t('common.refresh')}
            </button>
          )}
        </div>
      )}
      {status === 'error' && error && (
        <div className="mb-2 p-2.5 rounded-md bg-err-soft border border-err text-err text-xs font-mono shrink-0 flex items-center gap-2">
          <span className="flex-1 min-w-0 break-all">{error}</span>
          {onRefresh && (
            <button type="button" className="btn btn-ghost btn-sm" onClick={onRefresh}>{t('common.retry')}</button>
          )}
        </div>
      )}
      <div
        ref={bodyRef}
        className="flex-1 min-h-0 overflow-auto bg-sunken border border-subtle rounded-md px-3 py-2 text-[11px] font-mono leading-relaxed"
        data-testid="log-view-body"
      >
        {hasMoreBefore && onLoadEarlier && (
          <div className="pb-1.5 text-center">
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={handleLoadEarlier}
              disabled={loadingEarlier}
            >
              {loadingEarlier ? t('logView.loadingEarlier') : t('logView.loadEarlier')}
            </button>
          </div>
        )}
        {empty ? (
          <div className="text-fg-tertiary italic">{placeholder}</div>
        ) : (
          visible.map((l, i) => <Row key={i} line={l} showMeta={showMeta} />)
        )}
      </div>
    </div>
  )
}
