import { useEffect, useState } from 'react'

/** 逗号分隔字符串 → 规范化 tag 数组（去首尾空格 / 丢空段）。 */
export function parseTags(s: string): string[] {
  return s.split(',').map((t) => t.trim()).filter(Boolean)
}

/** 逗号分隔 tag 列表的裸 <input>（不带 label）。focus 中显示原始文本（逗号 /
 * 空格随便打），onChange 实时解析数组，blur 时归整成 tags.join(', ')。
 *
 * 受控的是**文本**而非数组——避免「每次按键把数组 join 回填到 value」把正在
 * 敲的逗号 / 尾随空格当场抹掉（直接绑 `arr.join(', ')` + 每键 split/trim/filter
 * 就是那个「打不了逗号和空格」的 bug）。规范化只在 blur 发生，不碰逐键输入。
 *
 * 给自带外层 label 的场景（如 Settings 的 SettingsField）直接用这个；需要
 * 140px grid label 的用下面的 {@link TagsInput}。 */
export function TagListInput({ value, onChange, placeholder, disabled, className = '' }: {
  value: string[]
  onChange: (v: string[]) => void
  placeholder?: string
  disabled?: boolean
  className?: string
}) {
  const [text, setText] = useState(value.join(', '))
  // 外部改 value（restore 默认 / 切表单）且与当前文本解析结果不一致 → 重新同步
  // 文本。自己打字触发的 value 变化进不来（那时 parseTags(text) 恒等 value）。
  useEffect(() => {
    if (JSON.stringify(parseTags(text)) !== JSON.stringify(value)) setText(value.join(', '))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value])
  return (
    <input
      type="text"
      value={text}
      placeholder={placeholder}
      onChange={(e) => { setText(e.target.value); onChange(parseTags(e.target.value)) }}
      onBlur={() => setText(value.join(', '))}
      disabled={disabled}
      className={className}
    />
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
