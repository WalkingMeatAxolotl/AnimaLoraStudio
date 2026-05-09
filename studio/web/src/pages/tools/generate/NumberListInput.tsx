import { useState } from 'react'

/** 数值列表输入：[输入框] [+ 添加] + 已加 chips（× 删除）。
 *
 * 替代 "0.6, 0.8, 1.0" 这种逗号分隔的 text input —— 用户反馈打错容易（少
 * 个逗号、多个空格、混入字母）。chips 输入避免格式错误，并直观看到已加值。
 *
 * 内部仍把 chips 序列化成 ", " 分隔的 raw 字符串往外抛，与 schema
 * parseAxisValues 兼容（不改 backend）。
 */
export default function NumberListInput({
  raw, onChange,
  min = 0, max = 1, step = 0.05,
  placeholder = '0.85',
}: {
  raw: string
  onChange: (raw: string) => void
  min?: number
  max?: number
  step?: number
  placeholder?: string
}) {
  const [draft, setDraft] = useState('')

  const values = raw.split(',').map((s) => s.trim()).filter(Boolean)

  const addValue = () => {
    const t = draft.trim()
    if (!t) return
    const n = Number(t)
    if (!Number.isFinite(n)) return
    if (values.includes(t)) {
      setDraft('')
      return
    }
    onChange([...values, t].join(', '))
    setDraft('')
  }

  const removeAt = (i: number) => {
    onChange(values.filter((_, j) => j !== i).join(', '))
  }

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex gap-1.5 items-stretch">
        <input
          type="number"
          className="input font-mono text-xs flex-1"
          min={min}
          max={max}
          step={step}
          placeholder={placeholder}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              addValue()
            }
          }}
        />
        <button
          type="button"
          onClick={addValue}
          disabled={!draft.trim() || !Number.isFinite(Number(draft))}
          className="font-mono"
          style={{
            border: '1px solid var(--border-subtle)',
            background: 'var(--bg-sunken)',
            borderRadius: 'var(--r-md)',
            padding: '0 12px',
            fontSize: 12,
            color: 'var(--fg-secondary)',
            cursor: 'pointer',
            whiteSpace: 'nowrap',
          }}
        >
          + 添加
        </button>
      </div>
      {values.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {values.map((v, i) => (
            <span
              key={i}
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
              {v}
              <button
                type="button"
                onClick={() => removeAt(i)}
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
                title="删除"
                aria-label="删除"
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
