/**
 * UploadProgressBar — browser upload status composed from the shared ProgressBar.
 * Numeric XHR ticks stay in progressbar value semantics; only phase transitions
 * (processing, complete, failure) enter a live region.
 */
import { useTranslation } from 'react-i18next'

import type { UploadProgressState } from '../lib/useUploadProgress'
import { formatBytes, formatEta, formatSpeed } from '../lib/useUploadProgress'
import Alert from './Alert'
import ProgressBar from './ProgressBar'

interface Props {
  state: UploadProgressState
  className?: string
}

export default function UploadProgressBar({ state, className }: Props) {
  const { t } = useTranslation()
  if (state.phase === 'idle') return null

  const determinate = state.total > 0
  const pct = determinate
    ? Math.min(100, Math.round((state.loaded / state.total) * 100))
    : null

  if (state.phase === 'error') {
    return (
      <Alert
        tone="danger"
        size="sm"
        icon={false}
        role="alert"
        className={className}
      >
        {t('upload.failed')}: {state.error ?? ''}
      </Alert>
    )
  }

  const processing = state.phase === 'processing'
  const done = state.phase === 'done'
  const label = processing
    ? t('upload.processing')
    : done
      ? t('upload.complete')
      : t('upload.progressLabel')
  const detail = determinate
    ? `${formatBytes(state.loaded)} / ${formatBytes(state.total)}`
    : formatBytes(state.loaded)

  return (
    <div className={['flex flex-col gap-related', className].filter(Boolean).join(' ')}>
      <ProgressBar
        label={label}
        value={processing ? null : (done ? 100 : pct)}
        valueText={processing ? t('upload.processing') : (pct == null ? undefined : `${pct}% · ${detail}`)}
        tone={done ? 'success' : 'accent'}
        size="sm"
      />

      {processing ? (
        <div className="flex items-center gap-related text-xs text-fg-secondary" role="status" aria-live="polite">
          <span className="dot dot-running" aria-hidden="true" />
          <span>{t('upload.processing')}</span>
        </div>
      ) : done ? (
        <div className="text-xs text-ok" role="status" aria-live="polite">
          {t('upload.complete')}
        </div>
      ) : (
        <div className="flex items-center justify-between gap-related font-mono text-xs text-fg-tertiary">
          <span className="truncate">
            {detail}
            {state.speedBps > 0 && <> · {formatSpeed(state.speedBps)}</>}
          </span>
          {(state.etaSec != null || pct != null) && (
            <span className="shrink-0">
              {state.etaSec != null && <>{t('upload.etaPrefix')} {formatEta(state.etaSec)}{pct != null && ' · '}</>}
              {pct != null && `${pct}%`}
            </span>
          )}
        </div>
      )}
    </div>
  )
}
