import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { InfoButton } from '../../../components/InfoButton'
import PathPicker from '../../../components/PathPicker'
import { normalizeLoraPath } from '../generate/loraSelection'
import { SettingsSection } from './fields'

export default function LoraSourcesSection({
  modelsRoot,
  directories,
  onChange,
}: {
  modelsRoot: string
  directories: string[]
  onChange: (directories: string[]) => void
}) {
  const { t } = useTranslation()
  const [pickerOpen, setPickerOpen] = useState(false)
  const defaultPath = `${modelsRoot.replace(/[\\/]+$/, '')}/loras`

  const add = (path: string) => {
    const trimmed = path.trim()
    if (!trimmed) return
    const key = normalizeLoraPath(trimmed)
    if (!directories.some((item) => normalizeLoraPath(item) === key)
        && normalizeLoraPath(defaultPath) !== key) {
      onChange([...directories, trimmed])
    }
    setPickerOpen(false)
  }

  return (
    <SettingsSection
      id="lora-sources"
      title={t('settings.loraSources.title')}
      headerExtras={(
        <>
          <InfoButton><p>{t('settings.loraSources.defaultHelp')}</p></InfoButton>
          <button type="button" className="btn btn-secondary btn-sm ml-auto" onClick={() => setPickerOpen(true)}>
            <span aria-hidden="true">+</span>
            <span>{t('settings.loraSources.add')}</span>
          </button>
        </>
      )}
    >
      <div className="flex flex-col gap-2">
        <div
          className="rounded-md border border-subtle bg-sunken px-2.5 py-2 flex items-center gap-2"
          style={{ minHeight: 44 }}
        >
          <code className="text-xs text-fg-primary flex-1 min-w-0 break-all">{defaultPath}</code>
        </div>
        {directories.map((path) => (
          <div
            key={normalizeLoraPath(path)}
            className="rounded-md border border-subtle bg-sunken px-2.5 py-2 flex items-center gap-2"
            style={{ minHeight: 44 }}
          >
            <code className="text-xs text-fg-primary flex-1 min-w-0 break-all">{path}</code>
            <button
              type="button"
              className="btn btn-ghost btn-sm text-err shrink-0 inline-flex items-center gap-1"
              onClick={() => onChange(directories.filter((item) => item !== path))}
              aria-label={`${t('settings.loraSources.remove')} ${path}`}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                <path d="M18 6 6 18M6 6l12 12" />
              </svg>
              <span>{t('settings.loraSources.remove')}</span>
            </button>
          </div>
        ))}
      </div>

      {pickerOpen && (
        <PathPicker
          dirOnly
          initialPath={directories[directories.length - 1] ?? defaultPath}
          onPick={add}
          onClose={() => setPickerOpen(false)}
        />
      )}
    </SettingsSection>
  )
}
