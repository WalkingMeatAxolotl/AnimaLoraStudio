import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api } from '../../../api/client'
import { useEventStream } from '../../../lib/useEventStream'
import LogView from '../../../components/LogView'

interface LogEntry {
  ts: number
  seq: number
  line: string
}

const RING_MAX = 2000

/** daemon stderr ring buffer 抽屉。
 *
 * - 从底部向上滑出 40vh，z-index 高，挡住下方 Generate 页表面但 layout 不占空间
 * - 隐藏时 translateY(100%) 完全不可见（不只是 visibility:hidden，整块抽屉离开视口）
 * - 首次打开 GET /api/generate/daemon/logs 拉历史，之后靠 SSE daemon_log_line 增量；
 *   SSE 重连（onOpen）按 since_seq 补拉，断线期间的行不丢
 * - 关闭后再开：只显示历史 + 此后增量；不会丢内容（ring buffer maxlen=2000）
 * - 内容区是统一 LogView：daemon 经 setup_logging 写 stderr，行契约同 run.log，
 *   按级别着色 / 调试开关 / 复制；「清屏」只清客户端显示
 */
export default function DaemonLogDrawer({
  open, onClose,
}: {
  open: boolean
  onClose: () => void
}) {
  const { t } = useTranslation()
  const [entries, setEntries] = useState<LogEntry[]>([])
  // LogView 工具栏 portal 进 header（与 TaskLogDrawer 同构，避免两层工具栏行）
  const [toolbarEl, setToolbarEl] = useState<HTMLElement | null>(null)
  const seqRef = useRef(0)
  const openRef = useRef(open)
  openRef.current = open

  // 从 since_seq 补拉（首次打开 / 重连）
  const fill = useCallback(() => {
    void api.getDaemonLogs(seqRef.current).then((r) => {
      if (r.entries.length > 0) {
        const minSeq = seqRef.current
        setEntries((prev) => {
          const next = [...prev, ...r.entries.filter((e) => e.seq >= minSeq)]
          return next.length > RING_MAX ? next.slice(-RING_MAX) : next
        })
        seqRef.current = r.next_seq
      } else if (seqRef.current === 0) {
        seqRef.current = r.next_seq
      }
    }).catch(() => { /* 不阻塞 */ })
  }, [])

  // 打开时拉历史；关闭不清空（保留下次打开时立即可见）
  useEffect(() => {
    if (open) fill()
  }, [open, fill])

  // SSE 增量 + 重连补拉
  useEventStream(useCallback((evt) => {
    if (evt.type !== 'daemon_log_line') return
    const seq = typeof evt.seq === 'number' ? evt.seq : seqRef.current
    if (seq < seqRef.current) return  // 老事件忽略
    seqRef.current = seq + 1
    setEntries((prev) => {
      const next = [...prev, {
        ts: Number(evt.ts) || Date.now() / 1000,
        seq,
        line: String(evt.line ?? ''),
      }]
      // 保护内存：客户端也限 2000 行
      return next.length > RING_MAX ? next.slice(-RING_MAX) : next
    })
  }, []), { onOpen: () => { if (openRef.current) fill() } })

  const lines = entries.map((e) => e.line)

  return (
    <div
      aria-hidden={!open}
      className={`fixed inset-x-0 bottom-0 h-[40vh] z-[60] flex flex-col bg-elevated border-t border-subtle transition-transform duration-200 ease-out ${
        open ? 'translate-y-0 shadow-2xl pointer-events-auto' : 'translate-y-full pointer-events-none'
      }`}
    >
      <header className="flex items-center gap-2.5 px-4 py-2 border-b border-subtle shrink-0">
        <span className="text-sm font-semibold text-fg-primary">{t('generate.logDrawerTitle')}</span>
        <span className="text-xs font-mono text-fg-tertiary">{entries.length}</span>
        <span className="flex-1" />
        <div ref={setToolbarEl} className="flex items-center shrink-0" />
        <button className="btn btn-ghost btn-sm" onClick={onClose}>
          {t('generate.logDrawerClose')}
        </button>
      </header>
      {/* 关着时不渲染内容区，省掉隐藏抽屉的解析 / 滚动 */}
      {open && (
        <LogView
          className="flex-1 min-h-0"
          lines={lines}
          status="live"
          emptyText={t('generate.logDrawerEmpty')}
          toolbar={!!toolbarEl}
          toolbarContainer={toolbarEl}
          frameless
          extraActions={
            <button className="btn btn-ghost btn-sm" onClick={() => setEntries([])} disabled={entries.length === 0}>
              {t('generate.logDrawerClear')}
            </button>
          }
        />
      )}
    </div>
  )
}
