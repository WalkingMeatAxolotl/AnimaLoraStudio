import { useRef } from 'react'

import { TagSuggestList } from '../../../components/tagSuggest/TagSuggestList'
import { useTagSuggest } from '../../../components/tagSuggest/useTagSuggest'

/** 负向提示词输入：接 tag autocomplete，跟 PromptList 同 UX。 */
export default function NegPromptInput({ value, onChange }: {
  value: string
  onChange: (v: string) => void
}) {
  const taRef = useRef<HTMLTextAreaElement>(null)
  const suggest = useTagSuggest({
    value,
    inputRef: taRef,
    onPick: ({ suggestion, range }) => {
      const before = value.slice(0, range.start)
      const after = value.slice(range.end)
      const cleanAfter = after.replace(/^[,，]\s*/, '')
      const next = `${before}${suggestion.tag}, ${cleanAfter}`
      onChange(next)
      const newCursor = before.length + suggestion.tag.length + 2
      requestAnimationFrame(() => {
        const el = taRef.current
        if (el) { el.focus(); el.setSelectionRange(newCursor, newCursor) }
      })
    },
  })
  return (
    <div className="relative">
      <textarea
        ref={taRef}
        className="input w-full font-mono text-xs resize-y"
        rows={5}
        value={value}
        onChange={(e) => { onChange(e.target.value); suggest.notifyChange() }}
        onKeyDown={(e) => { suggest.handleKeyDown(e) }}
        onKeyUp={() => suggest.notifySelect()}
        onClick={() => suggest.notifySelect()}
        onFocus={() => suggest.notifyFocus()}
        onBlur={() => suggest.notifyBlur()}
      />
      <TagSuggestList
        open={suggest.open}
        suggestions={suggest.suggestions}
        activeIdx={suggest.activeIdx}
        onPick={(s) => suggest.pickAt(suggest.suggestions.indexOf(s))}
        onHover={suggest.setActiveIdx}
        inputRef={taRef}
        cursor={suggest.cursor}
        positionDeps={[value]}
      />
    </div>
  )
}
