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

/** XY 模式专用的固定 LoRA 表单；普通 LoRA tab 继续使用 SidebarLoras 的既有样式。 */
export default function SidebarFixedLoras({
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
    onChange(loras.map((entry, itemIndex) => (
      itemIndex === index ? { ...entry, ...patch } : entry
    )), ui)
  }

  const removeEntry = (index: number) => {
    onChange(
      loras.filter((_, itemIndex) => itemIndex !== index),
      ui.filter((_, itemIndex) => itemIndex !== index),
    )
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
    <div className="flex flex-col gap-3" data-testid="current-fixed-lora-panel">
      {loras.length === 0 ? (
        <div className="rounded-md border border-dashed border-subtle p-5 text-center text-xs text-fg-tertiary">
          {t('generate.currentLorasEmpty')}
        </div>
      ) : (
        <div className="flex flex-col" data-testid="selected-lora-list">
          {loras.map((entry, index) => {
            const state = ui[index]
            const id = state?.id ?? `missing-${index}`
            const missing = !entry.path.trim()
            const enabled = !missing && state?.enabled !== false
            const name = loraTextName(entry) || t('generate.unknownLora')
            return (
              <div
                key={id}
                className="border-b border-subtle py-3 first:pt-0 last:border-b-0 last:pb-0 transition-opacity"
                style={{ opacity: enabled || missing ? 1 : 0.58 }}
                data-lora-id={id}
              >
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={enabled}
                    disabled={missing}
                    onChange={(event) => onChange(loras, ui.map((item, itemIndex) => (
                      itemIndex === index ? { ...item, enabled: event.target.checked } : item
                    )))}
                    title={t('generate.loraEnabled')}
                    aria-label={`${t('generate.loraEnabled')} ${name}`}
                    className="shrink-0"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="font-mono text-xs text-fg-primary truncate" title={name}>{name}</div>
                    {missing && <div className="text-2xs text-err truncate mt-0.5">{t('generate.loraNotFoundHint')}</div>}
                  </div>
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm text-err shrink-0"
                    onClick={() => removeEntry(index)}
                    title={t('generate.removeLora')}
                    aria-label={`${t('generate.removeLora')} ${name}`}
                  >
                    {t('common.delete')}
                  </button>
                </div>

                {!missing && (
                  <div className="flex items-center gap-2 mt-2 pl-6">
                    <span className="caption shrink-0">{t('generate.weight')}</span>
                    <input
                      type="range"
                      min={0}
                      max={1.5}
                      step={0.05}
                      value={entry.scale}
                      onChange={(event) => updateEntry(index, { scale: Number(event.target.value) })}
                      aria-label={`${t('generate.weightSlider')} ${name}`}
                      className="flex-1 min-w-0"
                      style={{ accentColor: 'var(--accent)' }}
                    />
                    <input
                      type="number"
                      min={0}
                      max={1.5}
                      step={0.05}
                      value={entry.scale}
                      onChange={(event) => updateEntry(index, { scale: Number(event.target.value) })}
                      aria-label={`${t('generate.weightValue')} ${name}`}
                      className="input input-mono text-xs text-center shrink-0"
                      style={{ width: 58, padding: '3px 5px' }}
                    />
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      <details className="border-t border-subtle pt-2">
        <summary className="text-xs text-fg-tertiary cursor-pointer select-none py-1">
          {t('generate.loraTextEdit')}
        </summary>
        <div className="mt-2">
          <textarea
            className={`input input-mono w-full text-xs resize-y ${textError ? 'border-err' : ''}`}
            style={{ minHeight: 72 }}
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
      </details>
    </div>
  )
}
