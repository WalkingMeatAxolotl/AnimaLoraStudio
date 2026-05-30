import { useEffect, useRef, useState } from 'react'

/** 逗号分隔字符串 → 规范化 tag 数组（去首尾空格 / 丢空段）。 */
export function parseTags(s: string): string[] {
  return s.split(',').map((t) => t.trim()).filter(Boolean)
}

/** 逗号分隔 tag 列表输入（不带 label）。两态：
 *
 * - **编辑态（focus）**：纯文本 `<input>`，逗号 / 空格随便打。受控的是文本而非
 *   数组，避免「每键 join 回填」把正在敲的逗号 / 尾随空格当场抹掉。
 * - **静止态（blur）**：把 tag 渲染成 chip 一眼可扫；点击 / 聚焦回到编辑态。
 *
 * blur 时文本归整成 `tags.join(', ')`，再进编辑态看到的是规范形式。
 * 给自带外层 label 的场景（Settings 的 SettingsField）直接用这个；要 140px
 * grid label 的用下面的 {@link TagsInput}。 */
export function TagListInput({ value, onChange, placeholder, disabled, className = '' }: {
  value: string[]
  onChange: (v: string[]) => void
  placeholder?: string
  disabled?: boolean
  className?: string
}) {
  const [text, setText] = useState(value.join(', '))
  const [editing, setEditing] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  // 外部改 value（restore 默认 / 切表单）且与当前文本解析结果不一致 → 重新同步
  // 文本。自己打字触发的 value 变化进不来（那时 parseTags(text) 恒等 value）。
  useEffect(() => {
    if (JSON.stringify(parseTags(text)) !== JSON.stringify(value)) setText(value.join(', '))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value])

  // 进入编辑态 → 把光标放进 input。
  useEffect(() => {
    if (editing) inputRef.current?.focus()
  }, [editing])

  if (editing && !disabled) {
    return (
      <input
        ref={inputRef}
        type="text"
        value={text}
        placeholder={placeholder}
        onChange={(e) => { setText(e.target.value); onChange(parseTags(e.target.value)) }}
        onBlur={() => { setText(value.join(', ')); setEditing(false) }}
        disabled={disabled}
        className={className}
      />
    )
  }

  // 静止态：chip 展示，点击 / 聚焦进入编辑。
  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      onClick={() => { if (!disabled) setEditing(true) }}
      onFocus={() => { if (!disabled) setEditing(true) }}
      className={`${className} flex flex-wrap items-center gap-1 ${disabled ? '' : 'cursor-text'}`}
    >
      {value.length === 0
        ? <span className="text-fg-tertiary">{placeholder}</span>
        : value.map((tag, i) => (
            <span
              key={`${tag}-${i}`}
              className="inline-flex items-center px-2 py-0.5 rounded-full bg-overlay border border-subtle text-xs font-mono text-fg-primary"
            >
              {tag}
            </span>
          ))}
    </div>
  )
}

/** 带 140px label 的版本（打标页 grid 布局用）。 */
export default function TagsInput({ label, value, placeholder, disabled, onChange, modified, className = '' }: {
  label: string
  value: string[]
  placeholder?: string
  disabled: boolean
  onChange: (v: string[]) => void
  modified?: boolean
  className?: string
}) {
  return (
    <label className={'grid grid-cols-[140px_1fr] items-center gap-2 ' + className}>
      <span className="text-fg-tertiary font-mono text-xs">{label}</span>
      <TagListInput
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        disabled={disabled}
        className={`input input-mono ${modified ? 'border-warn' : ''}`}
      />
    </label>
  )
}
