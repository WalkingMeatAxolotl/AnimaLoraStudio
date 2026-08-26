import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { LoraEntry } from '../../../api/client'
import {
  applyLoraText,
  LoraTextError,
  loraTextName,
  serializeLoraText,
  type LoraUiState,
} from './loraSelection'

export default function SidebarLoras({
  loras,
  ui,
  onChange,
}: {
  loras: LoraEntry[]
  ui: LoraUiState[]
  onChange: (loras: LoraEntry[], ui: LoraUiState[]) => void
}) {
  const { t } = useTranslation()
  const [text, setText] = useState(() => serializeLoraText(loras, ui))
  const [textError, setTextError] = useState<string | null>(null)
  const textFocused = useRef(false)
  const summary = useMemo(() => serializeLoraText(loras, ui), [loras, ui])

  useEffect(() => {
    if (!textFocused.current) setText(summary)
  }, [summary])

  const updateEntry = (index: number, patch: Partial<LoraEntry>) => {
    onChange(loras.map((entry, i) => (i === index ? { ...entry, ...patch } : entry)), ui)
  }

  const removeEntry = (index: number) => {
    onChange(loras.filter((_, i) => i !== index), ui.filter((_, i) => i !== index))
  }

  const applyText = () => {
    try {
      const result = applyLoraText(text, loras, ui)
      onChange(result.loras, result.ui)
      setText(serializeLoraText(result.loras, result.ui))
      setTextError(null)
    } catch (error) {
      if (error instanceof LoraTextError) {
        setTextError(t(`generate.loraTextError.${error.code}`, { name: error.value }))
      } else {
        setTextError(String(error))
      }
    }
  }

  return (
    <div className="flex flex-col gap-3" data-testid="current-lora-panel">
      <div>
        <textarea
          className={`input input-mono w-full text-xs resize-y ${textError ? 'border-err' : ''}`}
          style={{ minHeight: 82 }}
          value={text}
          onFocus={() => { textFocused.current = true }}
          onBlur={() => { textFocused.current = false; applyText() }}
          onChange={(event) => { setText(event.target.value); setTextError(null) }}
          onKeyDown={(event) => {
            if (event.ctrlKey && event.key === 'Enter') {
              event.preventDefault()
              applyText()
            }
          }}
          placeholder="<lora:name:1>"
          aria-label={t('generate.loraText')}
          aria-invalid={Boolean(textError)}
        />
        {textError && <div className="text-xs text-err mt-1" role="alert">{textError}</div>}
      </div>

      {loras.length === 0 && (
        <div className="rounded-md border border-dashed border-subtle p-5 text-center text-xs text-fg-tertiary">
          {t('generate.currentLorasEmpty')}
        </div>
      )}

      {loras.map((entry, index) => {
        const state = ui[index]
        const id = state?.id ?? `missing-${index}`
        const missing = !entry.path.trim()
        const enabled = !missing && state?.enabled !== false
        const name = loraTextName(entry) || t('generate.unknownLora')
        return (
          <div
            key={id}
            className={`rounded-md border p-2.5 transition-all ${missing ? 'border-err bg-err-soft' : 'border-subtle bg-overlay'}`}
            style={{ opacity: enabled || missing ? 1 : 0.62 }}
            data-lora-id={id}
          >
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={enabled}
                disabled={missing}
                onChange={(event) => onChange(loras, ui.map((item, i) => (
                  i === index ? { ...item, enabled: event.target.checked } : item
                )))}
                title={t('generate.loraEnabled')}
                aria-label={`${t('generate.loraEnabled')} ${name}`}
                className="shrink-0"
              />
              <div className="flex-1 min-w-0">
                <div className="font-mono text-xs text-fg-primary truncate" title={name}>{name}</div>
                {missing && <div className="text-2xs text-err truncate mt-0.5">{t('generate.loraNotFoundHint')}</div>}
              </div>
              {!missing && (
                <input
                  type="number"
                  min={0}
                  max={1.5}
                  step={0.05}
                  value={entry.scale}
                  onChange={(event) => updateEntry(index, { scale: Number(event.target.value) })}
                  aria-label={`${t('generate.weightValue')} ${name}`}
                  className="input input-mono text-xs shrink-0"
                  style={{ width: 70, padding: '3px 5px' }}
                />
              )}
              <button
                type="button"
                className="btn btn-ghost btn-sm text-err shrink-0"
                onClick={() => removeEntry(index)}
                title={t('generate.removeLora')}
                aria-label={`${t('generate.removeLora')} ${name}`}
              >
                ×
              </button>
            </div>
          </div>
        )
      })}
    </div>
  )
}
