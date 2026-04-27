import { useEffect, useRef } from 'react'
import type { Job } from '../api/client'

interface Props {
  job: Job
  logs: string[]
  onCancel?: () => void
}

const STATUS_COLOR: Record<Job['status'], string> = {
  pending: 'bg-slate-700/60 text-slate-300',
  running: 'bg-amber-700/40 text-amber-200',
  done: 'bg-emerald-700/40 text-emerald-200',
  failed: 'bg-red-800/50 text-red-200',
  canceled: 'bg-slate-700/60 text-slate-300',
}

export default function JobProgress({ job, logs, onCancel }: Props) {
  const logRef = useRef<HTMLPreElement>(null)
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logs])

  const elapsed =
    job.started_at && (job.finished_at ?? Date.now() / 1000) - job.started_at
  const isLive = job.status === 'running' || job.status === 'pending'

  return (
    <section className="rounded-lg border border-slate-700 bg-slate-800/40 overflow-hidden">
      <header className="flex items-center gap-2 px-3 py-2 border-b border-slate-700">
        <span
          className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${STATUS_COLOR[job.status]}`}
        >
          {job.status}
        </span>
        <span className="text-xs text-slate-400 font-mono">
          job #{job.id}
        </span>
        {elapsed && elapsed > 0 && (
          <span className="text-xs text-slate-500">
            · {Math.round(elapsed)}s
          </span>
        )}
        <span className="flex-1" />
        {isLive && onCancel && (
          <button
            onClick={onCancel}
            className="text-xs px-2 py-1 rounded text-red-300 hover:text-red-200 hover:bg-red-900/30"
          >
            取消
          </button>
        )}
      </header>
      <pre
        ref={logRef}
        className="p-3 text-[11px] font-mono text-slate-300 bg-slate-950/40 max-h-72 overflow-y-auto whitespace-pre-wrap"
      >
        {logs.length === 0 ? '(等待日志...)' : logs.slice(-1000).join('\n')}
      </pre>
    </section>
  )
}
