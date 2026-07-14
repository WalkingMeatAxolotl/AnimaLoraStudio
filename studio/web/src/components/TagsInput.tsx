import { useEffect, useRef, useState } from 'react'

import { TranslatedTag } from './tagDisplay/TranslatedTag'
import { TagSuggestList } from './tagSuggest/TagSuggestList'
import { useTagSuggest } from './tagSuggest/useTagSuggest'

/** 逗号分隔字符串 → 规范化 tag 数组（去首尾空格 / 丢空段）。 */
export function parseTags(s: string): string[] {
  return s.split(',').map((t) => t.trim()).filter(Boolean)
}

/** 逗号分隔 tag 列表输入（不带 label）。两态：
 *
 * - **编辑态（focus）**：纯文本 `<input>`，逗号 / 空格随便打。受控的是文本而非
 *   数组，避免「每键 join 回填」把正在敲的逗号 / 尾随空格当场抹掉。autocomplete
 *   popover 在 input 下方浮出；键盘 ↑↓ + Enter/Tab 选中。
 * - **静止态（blur）**：把 tag 渲染成 chip 一眼可扫；命中翻译时显示中文。
 *
 * blur 时文本归整成 `tags.join(', ')`，再进编辑态看到的是规范形式。
 * 给自带外层 label 的场景（Settings 的 SettingsField）直接用这个；要 140px
 * grid label 的用下面的 {@link TagsInput}。 */
export function TagListInput({ value, onChange, placeholder, disabled, className = '', style, commitOnBlur = false }: {
  value: string[]
  onChange: (v: string[]) => void
  placeholder?: string
  disabled?: boolean
  className?: string
  /** 内联样式透传（打标页 TagField 块用它对齐训练配置页控件视觉）。 */
  style?: React.CSSProperties
  /** true：编辑过程只更新本地文本，blur 时才上抛父一次（instant-apply 设置页用，
   *  避免逐字 commit → 逐字 PUT）。默认 false 保持逐字上抛（打标页等本地编辑场景）。 */
  commitOnBlur?: boolean
}) {
  const [text, setText] = useState(value.join(', '))
  const [editing, setEditing] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const suggest = useTagSuggest({
    value: text,
    inputRef,
    disabled,
    onPick: ({ suggestion, range }) => {
      const before = text.slice(0, range.start)
      const after = text.slice(range.end)
      // 把 token range 替换为 `tag, `；后面已有的内容紧接其后，前导空格归整一次
      const cleanAfter = after.replace(/^[,，]\s*/, '')
      const next = `${before}${suggestion.tag}, ${cleanAfter}`
      setText(next)
      if (!commitOnBlur) onChange(parseTags(next))
      // cursor 移动到新插入 tag 后的 ", " 之后
      const newCursor = before.length + suggestion.tag.length + 2
      // 等 React 刷完再 setSelectionRange，否则受控更新会把 cursor 拉到末尾
      requestAnimationFrame(() => {
        const el = inputRef.current
        if (el) { el.focus(); el.setSelectionRange(newCursor, newCursor) }
      })
    },
  })

  // 外部改 value（restore 默认 / 切表单）且与当前文本解析结果不一致 → 重新同步
  // 文本。自己打字触发的 value 变化进不来（那时 parseTags(text) 恒等 value）。
  useEffect(() => {
    if (editing) return  // 编辑中不被外部 value 冲掉本地文本（commitOnBlur 下尤其必要）
    if (JSON.stringify(parseTags(text)) !== JSON.stringify(value)) setText(value.join(', '))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, editing])

  // 进入编辑态 → 把光标放进 input。
  useEffect(() => {
    if (editing) inputRef.current?.focus()
  }, [editing])

  if (editing && !disabled) {
    return (
      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          value={text}
          placeholder={placeholder}
          onChange={(e) => {
            setText(e.target.value)
            if (!commitOnBlur) onChange(parseTags(e.target.value))
            suggest.notifyChange()
          }}
          onKeyDown={(e) => { suggest.handleKeyDown(e) }}
          onKeyUp={() => { suggest.notifySelect() }}
          onClick={() => { suggest.notifyClick() }}
          onFocus={() => { suggest.notifyFocus() }}
          onBlur={() => {
            suggest.notifyBlur()
            // blur 归整：下划线→空格（跟训练 caption 同形，后端匹配也已 _/空格不敏感），
            // chip 统一展示空格形式。commitOnBlur 模式此时才把完整 tags 上抛父一次。
            const source = commitOnBlur ? parseTags(text) : value
            const canon = source.map((t) => t.replace(/_/g, ' '))
            if (JSON.stringify(canon) !== JSON.stringify(value)) onChange(canon)
            setText(canon.join(', '))
            setEditing(false)
          }}
          disabled={disabled}
          className={className}
          style={style}
        />
        <TagSuggestList
          open={suggest.open}
          suggestions={suggest.suggestions}
          activeIdx={suggest.activeIdx}
          onPick={(s) => suggest.pickAt(suggest.suggestions.indexOf(s))}
          onHover={suggest.setActiveIdx}
          inputRef={inputRef}
          cursor={suggest.cursor}
          positionDeps={[text]}
        />
      </div>
    )
  }

  // 静止态：chip 展示，点击 / 聚焦进入编辑。min-h-[1.75rem] 兜底空列表行高，
  // 防止只渲染 nbsp 时容器塌缩到 0；与 `.input` 类的 padding 叠加后高度跟普通
  // input 一致。
  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      onClick={() => { if (!disabled) setEditing(true) }}
      onFocus={() => { if (!disabled) setEditing(true) }}
      className={`${className} flex flex-wrap items-center gap-1 min-h-[1.75rem] ${disabled ? '' : 'cursor-text'}`}
      style={style}
    >
      {value.length === 0
        // nbsp 占位：撑住一行行高，空列表时容器不塌缩成一条线（必须是真实
        // U+00A0，ASCII 空格在 flex 容器里会被压成 0 宽）
        ? <span className="text-fg-tertiary">{placeholder || ' '}</span>
        : value.map((tag, i) => (
            <span
              key={`${tag}-${i}`}
              className="inline-flex items-center px-2 py-0.5 rounded-full bg-overlay border border-subtle text-xs font-mono text-fg-primary"
            >
              <TranslatedTag tag={tag} />
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
