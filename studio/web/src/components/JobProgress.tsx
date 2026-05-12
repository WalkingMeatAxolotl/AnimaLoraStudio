import { useEffect, useRef, useState } from 'react'
import type { Job } from '../api/client'

interface Props {
  job: Job
  logs: string[]
  onCancel?: () => void
}

const STATUS_COLOR: Record<Job['status'], string> = {
  pending:  'bg-overlay text-fg-secondary',
  running:  'bg-warn-soft text-warn',
  done:     'bg-ok-soft text-ok',
  failed:   'bg-err-soft text-err',
  canceled: 'bg-overlay text-fg-secondary',
}

export default function JobProgress({ job, logs, onCancel }: Props) {
  const logRef = useRef<HTMLPreElement>(null)
  const [, setTick] = useState(0)
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logs])

  const isLive = job.status === 'running' || job.status === 'pending'
  useEffect(() => {
    if (!isLive) return
    const id = window.setInterval(() => setTick((n) => n + 1), 1000)
    return () => window.clearInterval(id)
  }, [isLive])

  const elapsed =
    job.started_at && (job.finished_at ?? Date.now() / 1000) - job.started_at

  return (
    <section className="rounded-lg border border-subtle bg-surface overflow-hidden">
      <header className="flex items-center gap-2 px-3 py-2 border-b border-subtle">
        <span
          className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${STATUS_COLOR[job.status]}`}
        >
          {job.status}
        </span>
        <span className="text-xs text-fg-secondary font-mono">
          job #{job.id}
        </span>
        {elapsed && elapsed > 0 && (
          <span className="text-xs text-fg-tertiary">
            · {Math.round(elapsed)}s
          </span>
        )}
        <span className="flex-1" />
        {isLive && onCancel && (
          <button
            onClick={onCancel}
            className="btn btn-ghost btn-sm text-err hover:bg-err-soft"
          >
            取消
          </button>
        )}
      </header>
      <pre
        ref={logRef}
        className="p-3 text-[11px] font-mono text-fg-secondary bg-sunken max-h-72 overflow-y-auto whitespace-pre-wrap"
      >
        {logs.length === 0 ? '(等待日志...)' : logs.slice(-1000).join('\n')}
      </pre>
    </section>
  )
}
