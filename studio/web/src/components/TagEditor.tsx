import { useEffect, useMemo, useRef, useState } from 'react'
import {
  DndContext,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core'
import {
  SortableContext,
  arrayMove,
  rectSortingStrategy,
  useSortable,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useTranslation } from 'react-i18next'

import { TranslatedTag } from './tagDisplay/TranslatedTag'
import { TagSuggestList } from './tagSuggest/TagSuggestList'
import { useTagSuggest } from './tagSuggest/useTagSuggest'

interface Props {
  tags: string[]
  natural?: boolean
  onChange: (tags: string[]) => void
  onSave?: () => void | Promise<void>
  saving?: boolean
  dirty?: boolean
}

type Mode = 'chip' | 'text'

const parseLine = (raw: string): string[] =>
  raw.split(/[,，\n]/).map((t) => t.trim()).filter(Boolean)

export default function TagEditor({
  tags, natural, onChange, onSave, saving, dirty,
}: Props) {
  const { t } = useTranslation()
  const [draft, setDraft] = useState('')
  const tagsJoined = useMemo(() => tags.join(', '), [tags])
  const [mode, setMode] = useState<Mode>(natural ? 'text' : 'chip')
  const [textBuf, setTextBuf] = useState(() => tagsJoined)
  const draftInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // PointerSensor + 6px 启动距离：拖拽手感不会跟「点 × 删除」/ 误触冲突。
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
  )

  // Reset draft when image switches
  useEffect(() => { setDraft('') }, [tags])

  // Sync textBuf when tags change WHILE in text mode (image switch)
  const prevTagsJoinedRef = useRef(tagsJoined)
  useEffect(() => {
    if (mode === 'text' && tagsJoined !== prevTagsJoinedRef.current) {
      setTextBuf(tagsJoined)
    }
    prevTagsJoinedRef.current = tagsJoined
  }, [tagsJoined, mode])

  const addTag = (raw: string) => {
    const t = raw.trim().replace(/^[,，]+|[,，]+$/g, '')
    if (!t) return
    if (tags.includes(t)) { setDraft(''); return }
    // 加到末尾：跟 chip 拖拽重排的心智一致（新东西落在底部，用户拖到想要的位置）
    onChange([...tags, t])
    setDraft('')
  }

  // chip 模式 input：draft 整体当一个 token；选中候选直接 addTag。
  const draftSuggest = useTagSuggest({
    value: draft,
    inputRef: draftInputRef,
    wholeAsToken: true,
    onPick: ({ suggestion }) => { addTag(suggestion.tag) },
  })

  // text 模式 textarea：根据 cursor 算 token range，替换为 `tag, ` 并保持光标。
  const textSuggest = useTagSuggest({
    value: textBuf,
    inputRef: textareaRef,
    onPick: ({ suggestion, range }) => {
      const before = textBuf.slice(0, range.start)
      const after = textBuf.slice(range.end)
      const cleanAfter = after.replace(/^[,，]\s*/, '')
      const next = `${before}${suggestion.tag}, ${cleanAfter}`
      setTextBuf(next)
      const newCursor = before.length + suggestion.tag.length + 2
      requestAnimationFrame(() => {
        const el = textareaRef.current
        if (el) { el.focus(); el.setSelectionRange(newCursor, newCursor) }
      })
    },
  })

  const removeTag = (t: string) => {
    onChange(tags.filter((x) => x !== t))
  }

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event
    if (!over || active.id === over.id) return
    const oldIndex = tags.indexOf(String(active.id))
    const newIndex = tags.indexOf(String(over.id))
    if (oldIndex < 0 || newIndex < 0) return
    onChange(arrayMove(tags, oldIndex, newIndex))
  }

  const commitText = () => {
    const next: string[] = []
    const seen = new Set<string>()
    for (const t of parseLine(textBuf)) {
      if (seen.has(t)) continue
      seen.add(t); next.push(t)
    }
    if (JSON.stringify(next) !== JSON.stringify(tags)) onChange(next)
  }

  const switchToText = () => {
    if (mode === 'text') return
    setTextBuf(tagsJoined) // sync immediately, no double-render via effect
    setMode('text')
  }

  const switchToChip = () => {
    if (mode === 'chip') return
    commitText()
    setMode('chip')
  }

  if (natural) {
    return (
      <div className="flex flex-col gap-2 flex-1 min-h-0">
        <textarea
          value={tags[0] ?? ''}
          onChange={(e) => onChange([e.target.value])}
          placeholder={t('tagEditor.naturalPlaceholder')}
          className="input input-mono text-sm flex-1 resize-none"
        />
        {onSave && (
          <button
            disabled={saving || !dirty}
            onClick={onSave}
            className={`self-start ${dirty ? 'btn btn-primary btn-sm' : 'btn btn-secondary btn-sm'}`}
          >
            {saving ? t('common.saving') : dirty ? t('common.save') : t('saveBar.saved')}
          </button>
        )}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1.5 flex-1 min-h-0">
      {/* mode switch */}
      <div className="flex items-center gap-1.5 text-xs shrink-0">
        <ModeBtn active={mode === 'chip'} onClick={switchToChip}>{t('tagEditor.modeChip')}</ModeBtn>
        <ModeBtn active={mode === 'text'} onClick={switchToText}>{t('tagEditor.modeText')}</ModeBtn>
        <span className="flex-1" />
        <span className="text-fg-tertiary">{t('tagEditor.tagCount', { n: tags.length })}</span>
      </div>

      {/* content area — both modes use flex:1 so no height jitter */}
      {mode === 'chip' ? (
        <>
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragEnd={handleDragEnd}
          >
            <SortableContext items={tags} strategy={rectSortingStrategy}>
              <div className="flex flex-wrap gap-1 overflow-y-auto flex-1 min-h-0 content-start py-1">
                {tags.length === 0 && (
                  <span className="text-xs text-fg-tertiary">{t('tagEditor.empty')}</span>
                )}
                {tags.map((t) => (
                  <SortableChip key={t} id={t} onRemove={() => removeTag(t)} />
                ))}
              </div>
            </SortableContext>
          </DndContext>
          <div className="flex items-center gap-1.5 shrink-0">
            <div className="relative flex-1">
              <input
                ref={draftInputRef}
                value={draft}
                onChange={(e) => { setDraft(e.target.value); draftSuggest.notifyChange() }}
                onKeyDown={(e) => {
                  if (draftSuggest.handleKeyDown(e)) return
                  if (e.key === 'Enter' || e.key === ',' || e.key === '，') {
                    e.preventDefault(); addTag(draft)
                  }
                }}
                onClick={() => draftSuggest.notifyClick()}
                onFocus={() => draftSuggest.notifyFocus()}
                onBlur={() => draftSuggest.notifyBlur()}
                placeholder={t('tagEditor.addPlaceholder')}
                className="input input-mono text-xs w-full"
              />
              <TagSuggestList
                open={draftSuggest.open}
                suggestions={draftSuggest.suggestions}
                activeIdx={draftSuggest.activeIdx}
                onPick={(s) => draftSuggest.pickAt(draftSuggest.suggestions.indexOf(s))}
                onHover={draftSuggest.setActiveIdx}
                inputRef={draftInputRef}
                cursor={draftSuggest.cursor}
                positionDeps={[draft]}
              />
            </div>
            {onSave && (
              <button
                disabled={saving || !dirty}
                onClick={onSave}
                className={dirty ? 'btn btn-primary btn-sm' : 'btn btn-secondary btn-sm'}
              >
                {saving ? t('common.saving') : dirty ? t('common.save') : t('saveBar.saved')}
              </button>
            )}
          </div>
        </>
      ) : (
        <>
          <div className="relative flex-1 min-h-0 flex flex-col">
            <textarea
              ref={textareaRef}
              value={textBuf}
              onChange={(e) => { setTextBuf(e.target.value); textSuggest.notifyChange() }}
              onKeyDown={(e) => { textSuggest.handleKeyDown(e) }}
              onKeyUp={() => textSuggest.notifySelect()}
              onClick={() => textSuggest.notifyClick()}
              onFocus={() => textSuggest.notifyFocus()}
              onBlur={() => { textSuggest.notifyBlur(); commitText() }}
              placeholder={t('tagEditor.textPlaceholder')}
              className="input input-mono text-xs flex-1 resize-none"
            />
            <TagSuggestList
              open={textSuggest.open}
              suggestions={textSuggest.suggestions}
              activeIdx={textSuggest.activeIdx}
              onPick={(s) => textSuggest.pickAt(textSuggest.suggestions.indexOf(s))}
              onHover={textSuggest.setActiveIdx}
              inputRef={textareaRef}
              cursor={textSuggest.cursor}
              positionDeps={[textBuf]}
            />
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <button onClick={commitText} className="btn btn-ghost btn-sm">{t('tagEditor.sync')}</button>
            {onSave && (
              <button
                disabled={saving || !dirty}
                onClick={async () => { commitText(); await onSave() }}
                className={dirty ? 'btn btn-primary btn-sm' : 'btn btn-secondary btn-sm'}
              >
                {saving ? t('common.saving') : dirty ? t('common.save') : t('saveBar.saved')}
              </button>
            )}
          </div>
        </>
      )}
    </div>
  )
}

function ModeBtn({ active, onClick, children }: {
  active: boolean; onClick: () => void; children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className={[
        'px-2 py-0.5 rounded-sm text-xs border transition-colors cursor-pointer',
        active
          ? 'bg-accent border-accent text-accent-fg'
          : 'bg-overlay border-subtle text-fg-secondary hover:bg-surface',
      ].join(' ')}
    >
      {children}
    </button>
  )
}

/** 单个可拖拽 chip。dnd-kit 用 useSortable 给我们 setNodeRef / 拖拽 listeners /
 * transform / transition;CSS.Transform.toString 把 dnd-kit 算出的 (x,y,scale)
 * 翻译成 CSS transform 字符串。
 *
 * × 删除按钮要 stopPropagation onPointerDown —— 否则 6px 移动阈值过后 × 也成了
 * 拖拽起点,点 × 反而触发拖拽。
 */
function SortableChip({ id, onRemove }: { id: string; onRemove: () => void }) {
  const { t } = useTranslation()
  const {
    attributes, listeners, setNodeRef, transform, transition, isDragging,
  } = useSortable({ id })
  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
    zIndex: isDragging ? 1 : undefined,
  }
  return (
    <span
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-overlay border border-subtle text-sm font-mono text-fg-primary cursor-grab active:cursor-grabbing select-none touch-none"
    >
      <TranslatedTag tag={id} />
      <button
        onPointerDown={(e) => e.stopPropagation()}
        onClick={onRemove}
        aria-label={t('tagEditor.deleteTag', { tag: id })}
        className="bg-transparent border-none text-fg-tertiary hover:text-err cursor-pointer p-0 text-sm leading-none"
      >
        ×
      </button>
    </span>
  )
}
