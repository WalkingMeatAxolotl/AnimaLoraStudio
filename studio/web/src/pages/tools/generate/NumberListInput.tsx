import { useState } from 'react'
import { useTranslation } from 'react-i18next'

/** 数值列表输入：[输入框] [+ 添加] + 已加 chips（× 删除）。
 *
 * 用 chips 直观看到已加的每个值，避免裸 "0.6, 0.8, 1.0" text input 打错难发现。
 * 但输入框本身支持一次性粘贴 / 输入逗号（中英文）或空格分隔的多个值 ——
 * 不在 onChange 拦截，点「+ 添加」/ Enter 时才拆分 + 逐个 Number() 校验，
 * 有限数才入列（无效 token 直接忽略）。
 *
 * 内部仍把 chips 序列化成 ", " 分隔的 raw 字符串往外抛，与 schema
 * parseAxisValues 兼容（不改 backend）。
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
  const [draft, setDraft] = useState('')

  const values = raw.split(',').map((s) => s.trim()).filter(Boolean)

  // 转化时校验：按中英文逗号 / 空白拆 token，逐个 Number() 留有限数（保留原文本，
  // 不强转规范化，"0.50" 仍显示 "0.50"）。一次性输入 "0.2, 0.3 0.4" → [0.2,0.3,0.4]。
  const parseDraft = (s: string): string[] =>
    s.split(/[,，\s]+/).map((x) => x.trim()).filter((x) => x && Number.isFinite(Number(x)))

  const addValue = () => {
    const parsed = parseDraft(draft)
    if (parsed.length === 0) return
    const merged = [...values]
    for (const v of parsed) {
      if (!merged.includes(v)) merged.push(v)
    }
    onChange(merged.join(', '))
    setDraft('')
  }

  const removeAt = (i: number) => {
    onChange(values.filter((_, j) => j !== i).join(', '))
  }

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex gap-1.5 items-stretch">
        <input
          type="text"
          inputMode="decimal"
          className="input font-mono text-xs flex-1"
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
          disabled={parseDraft(draft).length === 0}
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
          {t('generate.addValue')}
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
