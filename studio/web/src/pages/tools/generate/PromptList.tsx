import { useRef } from 'react'
import { useTranslation } from 'react-i18next'

import { TagSuggestList } from '../../../components/tagSuggest/TagSuggestList'
import { useTagSuggest } from '../../../components/tagSuggest/useTagSuggest'
import { useAutoGrowTextarea } from '../../../lib/useAutoGrowTextarea'
import { useTokenCount } from '../../../lib/useTokenCount'

/** 正向提示词输入。
 *
 * 之前支持多 prompt 轮换（"+ 添加 prompt"），用户决策"隐藏前端轮换功能"
 * → 简化成单 textarea。后端仍接 list[str]，发请求时仍然包成数组。
 *
 * 接入 tag autocomplete：cursor 所在 token 触发建议；↑↓/Tab/Enter 选中插入。
 */
export default function PromptList({ prompts, onChange, modelFamily = 'anima' }: {
  prompts: string[]
  onChange: (p: string[]) => void
  /** token 计数用的族（选对应 tokenizer）；不传默认 anima。 */
  modelFamily?: string
}) {
  const { t } = useTranslation()
  const taRef = useRef<HTMLTextAreaElement>(null)
  // 当前只显示第一条 prompt；用户编辑时同步成 [value]
  const value = prompts[0] ?? ''
  const suggest = useTagSuggest({
    value,
    inputRef: taRef,
    onPick: ({ suggestion, range }) => {
      const before = value.slice(0, range.start)
      const after = value.slice(range.end)
      const cleanAfter = after.replace(/^[,，]\s*/, '')
      const next = `${before}${suggestion.tag}, ${cleanAfter}`
      onChange([next])
      const newCursor = before.length + suggestion.tag.length + 2
      requestAnimationFrame(() => {
        const el = taRef.current
        if (el) { el.focus(); el.setSelectionRange(newCursor, newCursor) }
      })
    },
  })
  useAutoGrowTextarea(taRef, value)
  const tokenCount = useTokenCount(value, modelFamily)
  return (
    <div className="relative">
      <textarea
        ref={taRef}
        className="input w-full font-mono text-sm resize-none overflow-hidden"
        rows={5}
        value={value}
        onChange={(e) => { onChange([e.target.value]); suggest.notifyChange() }}
        onKeyDown={(e) => { suggest.handleKeyDown(e) }}
        onKeyUp={() => suggest.notifySelect()}
        onClick={() => suggest.notifyClick()}
        onFocus={() => suggest.notifyFocus()}
        onBlur={() => suggest.notifyBlur()}
        placeholder={t('generate.positivePlaceholder')}
      />
      {tokenCount != null && (
        <span className="absolute bottom-1.5 right-2 text-2xs text-fg-tertiary pointer-events-none select-none">
          {tokenCount} tokens
        </span>
      )}
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
