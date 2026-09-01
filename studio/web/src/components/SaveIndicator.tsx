import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import type { SaveStatus } from '../lib/SettingsData'

/**
 * Stable autosave live region shared by Settings, Train, and Presets.
 *
 * The element stays mounted while idle so assistive technology can announce
 * later state changes. Autosave failures remain non-urgent here; callers may
 * provide a separate nearby recovery surface when one exists.
 */
export interface SaveIndicatorProps {
  status: SaveStatus
  /** Set false when the same failure is already announced by an error Toast. */
  announceError?: boolean
}

export default function SaveIndicator({
  status,
  announceError = true,
}: SaveIndicatorProps) {
  const { t } = useTranslation()

  let content: ReactNode = null
  let toneClass = 'text-fg-tertiary'
  let title: string | undefined

  if (status.state === 'saving') {
    content = t('settings.saveStatus.saving')
  } else if (status.state === 'saved') {
    const time = new Date(status.at).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
    content = (
      <span key={status.at} className="settings-saved-flash inline-flex items-center gap-1">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M20 6L9 17l-5-5" />
        </svg>
        {t('settings.saveStatus.saved', { time })}
      </span>
    )
  } else if (status.state === 'error') {
    toneClass = 'text-err'
    title = status.error
    content = t('settings.saveStatus.error')
  }

  return (
    <span
      role="status"
      aria-live={status.state === 'error' && !announceError ? 'off' : 'polite'}
      aria-atomic="true"
      data-state={status.state}
      className={`inline-flex min-w-0 max-w-72 items-center gap-1.5 whitespace-nowrap text-xs ${toneClass}`}
      title={title}
    >
      {content}
    </span>
  )
}
