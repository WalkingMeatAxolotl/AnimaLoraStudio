import { useTranslation } from 'react-i18next'
import { type Job } from '../../api/client'

const STATUS_COLOR: Record<Job['status'], string> = {
  pending: 'badge badge-neutral',
  running: 'badge badge-warn',
  done: 'badge badge-ok',
  failed: 'badge badge-err',
  canceled: 'badge badge-neutral',
}

/** Compact, collapsible progress strip shared by all preprocess tools.
 *
 *  Default open while live, collapsed once terminal — matches the upscale
 *  page's UX. Last log line is visible in the summary so users see motion
 *  without expanding. Click summary to expand and read full tail.
 */
export default function PreprocessJobStrip({
  job,
  logs,
  onCancel,
}: {
  job: Job
  logs: string[]
  onCancel?: () => void
}) {
  const { t } = useTranslation()
  const elapsed =
    job.started_at && (job.finished_at ?? Date.now() / 1000) - job.started_at
  const isLive = job.status === 'running' || job.status === 'pending'
  const lastLine = logs[logs.length - 1] ?? ''
  return (
    <details
      open={isLive}
      className="group rounded-md border border-subtle bg-surface overflow-hidden shrink-0"
    >
      <summary className="cursor-pointer flex items-center gap-2 list-none px-2.5 py-1.5 text-sm select-none">
        <span className="inline-block transition-transform group-open:rotate-90 text-fg-tertiary w-3">▸</span>
        <span className={STATUS_COLOR[job.status]}>{job.status}</span>
        <span className="mono text-fg-secondary">job #{job.id}</span>
        {elapsed && elapsed > 0 && (
          <span className="text-fg-tertiary">· {Math.round(elapsed)}s</span>
        )}
        <span className="mono truncate flex-1 min-w-0 text-fg-secondary text-xs">
          {lastLine}
        </span>
        {isLive && onCancel && (
          <button
            onClick={(e) => {
              e.preventDefault()
              onCancel()
            }}
            className="btn btn-ghost btn-sm text-err"
          >{t('common.cancel')}</button>
        )}
      </summary>
      <pre className="px-3 py-2 text-xs font-mono text-fg-secondary bg-sunken max-h-[224px] overflow-auto whitespace-pre-wrap border-t border-subtle m-0">
        {logs.length === 0 ? t('jobProgress.waitingLogs') : logs.slice(-1000).join('\n')}
      </pre>
    </details>
  )
}
