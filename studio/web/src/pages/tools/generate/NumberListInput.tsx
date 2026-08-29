import { useTranslation } from 'react-i18next'

/** Numeric axis values are edited as one comma-separated draft and echoed as
 * removable chips below it. Keeping the text field controlled by `raw` makes
 * the current matrix definition visible when the editor drawer opens, while
 * the chips provide a quick way to remove one value without retyping the list.
 */
export default function NumberListInput({
  raw, onChange,
  placeholder = '0.85',
}: {
  raw: string
  onChange: (raw: string) => void
  placeholder?: string
}) {
  const { t } = useTranslation()
  const values = raw.split(/[,，]+/).map((s) => s.trim()).filter(Boolean)

  const removeAt = (index: number) => {
    onChange(values.filter((_, valueIndex) => valueIndex !== index).join(', '))
  }

  return (
    <div className="flex flex-col gap-1.5">
      <textarea
        className="input font-mono text-xs w-full"
        rows={3}
        inputMode="decimal"
        placeholder={placeholder}
        value={raw}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
            event.preventDefault()
            onChange(event.currentTarget.value)
          }
        }}
        aria-label={t('generate.axisDirectInput')}
      />
      {values.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {values.map((value, index) => (
            <span
              key={`${value}-${index}`}
              className="font-mono inline-flex items-center gap-1"
              style={{
                fontSize: 11,
                padding: '2px 4px 2px 8px',
                borderRadius: 999,
                background: 'var(--accent-soft)',
                color: 'var(--accent)',
                border: '1px solid transparent',
              }}
            >
              {value}
              <button
                type="button"
                onClick={() => removeAt(index)}
                style={{
                  width: 14, height: 14,
                  display: 'grid', placeItems: 'center',
                  borderRadius: 999,
                  border: 0,
                  background: 'transparent',
                  color: 'inherit',
                  cursor: 'pointer',
                  fontSize: 11,
                  lineHeight: 1,
                  padding: 0,
                }}
                title={t('common.delete')}
                aria-label={t('common.delete')}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
