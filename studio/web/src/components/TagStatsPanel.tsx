import { useEffect, useMemo, useRef, useState, type RefObject } from 'react'
import { useTranslation } from 'react-i18next'

import { TranslatedTag } from './tagDisplay/TranslatedTag'
import { TagSuggestList } from './tagSuggest/TagSuggestList'
import { useTagSuggest } from './tagSuggest/useTagSuggest'

type Sort = 'count_desc' | 'count_asc' | 'name_asc' | 'name_desc'

interface Props {
  cache: Map<string, string[]>
  selectedKeys: string[]
  /** 点 tag 文字 = 选中所有含该 tag 的图（保留旧行为，仍是 tag 浏览的主路径）。 */
  onPickTag: (tag: string) => void
  /** 行内 × 按钮：从当前选中图中删除该 tag。未传时不渲染 ×（向后兼容）。 */
  onRemoveTag?: (tag: string) => void
  /** 行内 ✎ → inline edit：从当前选中图中把 oldTag 替换为 newTag。 */
  onReplaceTag?: (oldTag: string, newTag: string) => void
}

export default function TagStatsPanel({
  cache,
  selectedKeys,
  onPickTag,
  onRemoveTag,
  onReplaceTag,
}: Props) {
  const { t } = useTranslation()
  const [filter, setFilter] = useState('')
  const [sort, setSort] = useState<Sort>('count_desc')
  // inline replace state：editingTag = 正在 inline 改名的 tag；editValue 是
  // input 当前值。点 ✎ 进入；按 ✓ / 回车提交；按 ✕ / Esc / 失焦取消。
  const [editingTag, setEditingTag] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const editInputRef = useRef<HTMLInputElement>(null)

  const usingSelection = selectedKeys.length > 0

  const items = useMemo(() => {
    const counter = new Map<string, number>()
    // 仅按选中图统计 —— 跟"操作 = 给选中图做"语义一致。未选时下面会显示
    // hint 而不是空列表，所以这里 selectedKeys 为空也不需要 fallback。
    for (const k of selectedKeys) {
      const tags = cache.get(k) ?? []
      for (const tag of tags) counter.set(tag, (counter.get(tag) ?? 0) + 1)
    }
    return Array.from(counter.entries())
  }, [cache, selectedKeys])

  const sorted = useMemo(() => {
    const out = [...items]
    if (sort === 'count_desc') out.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    else if (sort === 'count_asc') out.sort((a, b) => a[1] - b[1] || a[0].localeCompare(b[0]))
    else if (sort === 'name_asc') out.sort((a, b) => a[0].localeCompare(b[0]))
    else if (sort === 'name_desc') out.sort((a, b) => b[0].localeCompare(a[0]))
    return out
  }, [items, sort])

  const filtered = useMemo(() => {
    if (!filter.trim()) return sorted
    const f = filter.trim().toLowerCase()
    return sorted.filter(([tag]) => tag.toLowerCase().includes(f))
  }, [sorted, filter])

  const maxCount = filtered.length > 0 ? filtered[0][1] : 1

  // 进入 edit mode 时 focus + select 输入框
  useEffect(() => {
    if (editingTag && editInputRef.current) {
      editInputRef.current.focus()
      editInputRef.current.select()
    }
  }, [editingTag])

  const startEdit = (tag: string) => {
    setEditingTag(tag)
    setEditValue(tag)
  }
  const cancelEdit = () => {
    setEditingTag(null)
    setEditValue('')
  }
  const commitEdit = () => {
    const o = editingTag
    const n = editValue.trim()
    if (!o || !n || n === o) { cancelEdit(); return }
    onReplaceTag?.(o, n)
    cancelEdit()
  }

  // 表头点击：在同列内切换方向；跨列默认进 desc（数字默认从多到少看，名字
  // 默认 A→Z）。两列各自记忆方向，跟 Excel / Numbers 等表格软件的直觉一致。
  const toggleNameSort = () => setSort(sort === 'name_asc' ? 'name_desc' : 'name_asc')
  const toggleCountSort = () => setSort(sort === 'count_desc' ? 'count_asc' : 'count_desc')
  const nameArrow = sort === 'name_asc' ? ' ↑' : sort === 'name_desc' ? ' ↓' : ''
  const countArrow = sort === 'count_desc' ? ' ↓' : sort === 'count_asc' ? ' ↑' : ''

  return (
    <section className="flex flex-col min-h-0 flex-1 min-w-0 overflow-hidden">
      <div className="px-2.5 py-1.5 border-b border-subtle flex items-center gap-2 text-xs shrink-0 flex-wrap">
        <span className="font-semibold text-fg-primary">{t('tagStats.title')}</span>
        {usingSelection && (
          <span className="px-1.5 py-px rounded-sm text-[10px] bg-accent-soft text-accent">
            {t('tagStats.selected', { n: selectedKeys.length })}
          </span>
        )}
        {usingSelection && (
          <span className="text-fg-tertiary">{t('tagStats.tagCount', { n: items.length })}</span>
        )}
        <span className="flex-1" />
        {usingSelection && (
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder={t('common.filter')}
            className="input"
            style={{ fontSize: 'var(--t-xs)', padding: '1px 8px', width: 120 }}
          />
        )}
      </div>

      {/* table header：仅选中时显示。点击 header 切换该列排序方向 —— 类比
       * 表格软件「同列切方向 / 跨列默认 desc」直觉。活跃列旁带 ↑/↓ 指示
       * 当前方向，非活跃列不显示箭头避免视觉噪声。 */}
      {usingSelection && filtered.length > 0 && (
        <div className="px-2 py-1 border-b border-subtle flex items-center text-[10px] font-medium text-fg-tertiary shrink-0 gap-2">
          <button
            type="button"
            onClick={toggleNameSort}
            className="flex-1 min-w-0 text-left bg-transparent border-none cursor-pointer p-0 hover:text-fg-primary"
          >
            {t('tagStats.headerTag')}{nameArrow}
          </button>
          <button
            type="button"
            onClick={toggleCountSort}
            className="shrink-0 bg-transparent border-none cursor-pointer p-0 hover:text-fg-primary"
          >
            {t('tagStats.headerCount')}{countArrow}
          </button>
          {/* 占位给 ✎ × 按钮列，让 header 跟下面行的列对齐（即使按钮 hover 才显示） */}
          {(onReplaceTag || onRemoveTag) && (
            <span className="shrink-0" style={{ width: (onReplaceTag ? 16 : 0) + (onRemoveTag ? 16 : 0) }} />
          )}
        </div>
      )}

      <div className="flex-1 min-h-0 overflow-y-auto">
        {!usingSelection ? (
          // 未选图：默认空状态 + 提示 user 先选图。比起"显示全部 tag 但操作
          // 不可用"，明确提示更不容易让 user 误以为 panel 坏了。
          <p className="px-2.5 py-3 text-xs text-fg-tertiary m-0 leading-relaxed">
            {t('tagStats.pleaseSelectFirst')}
          </p>
        ) : filtered.length === 0 ? (
          <p className="px-2.5 py-2 text-xs text-fg-tertiary m-0">
            {filter.trim() ? t('tagStats.noMatch') : t('tagStats.noTagsSelected')}
          </p>
        ) : (
          <div className="p-1.5 flex flex-col gap-px">
            {filtered.map(([tag, c]) => {
              const pct = Math.max((c / maxCount) * 100, 3)
              const isEditing = editingTag === tag
              if (isEditing) {
                return (
                  <div
                    key={tag}
                    className="flex items-center gap-1 px-2 py-0.5 rounded-sm bg-overlay"
                  >
                    <EditTagInput
                      editInputRef={editInputRef}
                      value={editValue}
                      setValue={setEditValue}
                      onCommit={commitEdit}
                      onCancel={cancelEdit}
                    />
                    {/* onMouseDown 而不是 onClick：input 的 onBlur 会先触发，
                     * onClick 来时 commitEdit 已经被 cancelEdit 覆盖。
                     * mousedown 在 blur 之前触发，能拿到正确的 editValue。 */}
                    <button
                      onMouseDown={(e) => { e.preventDefault(); commitEdit() }}
                      className="btn btn-primary btn-sm"
                      style={{ padding: '1px 6px' }}
                      aria-label={t('common.confirm')}
                    >✓</button>
                    <button
                      onMouseDown={(e) => { e.preventDefault(); cancelEdit() }}
                      className="btn btn-ghost btn-sm"
                      style={{ padding: '1px 6px' }}
                      aria-label={t('common.cancel')}
                    >✕</button>
                  </div>
                )
              }
              return (
                <div
                  key={tag}
                  className="group flex items-center gap-1 px-2 py-0.5 rounded-sm bg-transparent text-xs min-w-0 relative hover:bg-overlay transition-colors"
                >
                  <span style={{
                    position: 'absolute', inset: 0, borderRadius: 'var(--r-sm)',
                    background: 'var(--accent-soft)', opacity: pct / 100 * 0.35,
                    width: `${pct}%`, zIndex: 0,
                  }} />
                  <button
                    onClick={() => onPickTag(tag)}
                    title={t('tagStats.pickTagTitle', { tag })}
                    className="font-mono text-fg-primary flex-1 min-w-0 overflow-hidden text-ellipsis whitespace-nowrap relative z-[1] text-left bg-transparent border-none cursor-pointer p-0"
                  >
                    <TranslatedTag tag={tag} />
                  </button>
                  <span className="text-fg-tertiary font-mono text-[10px] relative z-[1] shrink-0">
                    {c}
                  </span>
                  {onReplaceTag && (
                    <button
                      onClick={() => startEdit(tag)}
                      title={t('tagStats.replaceTitle', { tag })}
                      className="opacity-0 group-hover:opacity-100 relative z-[1] shrink-0 text-fg-tertiary hover:text-accent px-1 cursor-pointer bg-transparent border-none"
                      aria-label={t('tagStats.replaceTitle', { tag })}
                    >✎</button>
                  )}
                  {onRemoveTag && (
                    <button
                      onClick={() => onRemoveTag(tag)}
                      title={t('tagStats.removeTitle', { tag })}
                      className="opacity-0 group-hover:opacity-100 relative z-[1] shrink-0 text-fg-tertiary hover:text-danger px-1 cursor-pointer bg-transparent border-none"
                      aria-label={t('tagStats.removeTitle', { tag })}
                    >×</button>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </section>
  )
}

/** Inline tag rename input + 翻译 autocomplete。抽出来是为了能放 useTagSuggest hook —
 *  父组件 .map 内部不能调 hook。 */
function EditTagInput({ editInputRef, value, setValue, onCommit, onCancel }: {
  editInputRef: RefObject<HTMLInputElement>
  value: string
  setValue: (v: string) => void
  onCommit: () => void
  onCancel: () => void
}) {
  const suggest = useTagSuggest({
    value,
    inputRef: editInputRef,
    wholeAsToken: true,
    onPick: ({ suggestion }) => setValue(suggestion.tag),
  })
  return (
    <div className="relative flex-1 min-w-0">
      <input
        ref={editInputRef}
        value={value}
        onChange={(e) => { setValue(e.target.value); suggest.notifyChange() }}
        onKeyDown={(e) => {
          if (suggest.handleKeyDown(e)) return
          if (e.key === 'Enter') onCommit()
          else if (e.key === 'Escape') onCancel()
        }}
        onClick={() => suggest.notifyClick()}
        onFocus={() => suggest.notifyFocus()}
        onBlur={() => { suggest.notifyBlur(); onCancel() }}
        className="input input-mono w-full"
        style={{ fontSize: 'var(--t-xs)', padding: '1px 6px' }}
      />
      <TagSuggestList
        open={suggest.open}
        suggestions={suggest.suggestions}
        activeIdx={suggest.activeIdx}
        onPick={(s) => suggest.pickAt(suggest.suggestions.indexOf(s))}
        onHover={suggest.setActiveIdx}
        inputRef={editInputRef}
        cursor={suggest.cursor}
        positionDeps={[value]}
      />
    </div>
  )
}
