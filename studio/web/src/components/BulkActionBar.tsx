import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useToast } from './Toast'

/** add 拆成 -front / -back 两个显式 op：把"插入位置"从隐藏 state 提到按钮上，
 * user 不用先选 dropdown 再点确认 —— 看到"加到首部 / 加到尾部"两个按钮直接
 * 点哪个就是哪个意图。 */
type Op = 'add-front' | 'add-back' | 'remove' | 'replace' | 'dedupe'

interface Props {
  cache: Map<string, string[]>
  selectedKeys: string[]
  onApply: (updates: Map<string, string[]>) => void
  onSelectAll: () => void
  onClearSelection: () => void
  tagSuggestions?: string[]
  /** 用来给"未选时"的 hint 显示总数。 */
  totalCount: number
}

/** 重新设计的批量操作面板。
 *
 * 跟旧版的关键差异：
 * - **零 popover / 零隐藏 state**：所有 input 常驻可见，user 不会"输到一半
 *   popover 关掉，输入清空"
 * - **零 scope 选择**：永远操作 selectedKeys。要"对全部图操作" → 用顶部「全选」
 *   按钮先选起来再点 op。心智模型从"调 scope dropdown + 调 op"二维降到"操
 *   作就是给选中的"一维
 * - **add/remove 共享一个 input**：最常见的"输入若干 tag 然后加/删"场景统一
 *   到一个输入框，下面三个按钮决定动作（加首部/加尾部/删除）
 * - **replace 单独一行**："old → new"是约定俗成的特殊语义，单独行避免跟 add/
 *   remove 的 input 混淆
 * - **dedupe 顶部按钮**：无参数操作不需要 input，跟全选/取消同排
 *
 * 不再处理 filterTag —— TagStatsPanel 的标签分布已经是 tag 浏览的入口，配合
 * 行内 × / ✎ 单 tag 快捷。这里专心做 batch。 */
export default function BulkActionBar({
  cache,
  selectedKeys,
  onApply,
  onSelectAll,
  onClearSelection,
  tagSuggestions = [],
  totalCount,
}: Props) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [tagsInput, setTagsInput] = useState('')
  const [oldTag, setOldTag] = useState('')
  const [newTag, setNewTag] = useState('')

  const parseTags = (raw: string): string[] =>
    raw.split(/[,，\n]/).map((t) => t.trim()).filter(Boolean)

  const apply = (op: Op) => {
    const keys = selectedKeys
    if (keys.length === 0) {
      toast(t('bulkAction.noFiles'), 'error')
      return
    }
    const updates = new Map<string, string[]>()

    if (op === 'add-front' || op === 'add-back') {
      const ts = parseTags(tagsInput)
      if (ts.length === 0) { toast(t('bulkAction.enterTag'), 'error'); return }
      const insertFront = op === 'add-front'
      for (const k of keys) {
        const cur = cache.get(k) ?? []
        const have = new Set(cur)
        const toAdd = ts.filter((t) => !have.has(t))
        if (toAdd.length === 0) continue
        updates.set(k, insertFront ? [...toAdd, ...cur] : [...cur, ...toAdd])
      }
    } else if (op === 'remove') {
      const ts = parseTags(tagsInput)
      if (ts.length === 0) { toast(t('bulkAction.enterTag'), 'error'); return }
      const drop = new Set(ts)
      for (const k of keys) {
        const cur = cache.get(k) ?? []
        const next = cur.filter((t) => !drop.has(t))
        if (next.length !== cur.length) updates.set(k, next)
      }
    } else if (op === 'replace') {
      const o = oldTag.trim(); const n = newTag.trim()
      if (!o || !n) { toast(t('bulkAction.replaceNeedsOldNew'), 'error'); return }
      for (const k of keys) {
        const cur = cache.get(k) ?? []
        if (!cur.includes(o)) continue
        const next: string[] = []
        const seen = new Set<string>()
        for (const t of cur) {
          const out = t === o ? n : t
          if (seen.has(out)) continue
          seen.add(out); next.push(out)
        }
        updates.set(k, next)
      }
    } else if (op === 'dedupe') {
      for (const k of keys) {
        const cur = cache.get(k) ?? []
        const seen = new Set<string>(); const next: string[] = []
        for (const t of cur) { if (seen.has(t)) continue; seen.add(t); next.push(t) }
        if (next.length !== cur.length) updates.set(k, next)
      }
    }

    if (updates.size === 0) {
      toast(t('bulkAction.noChanges', { op }), 'success')
      return
    }
    onApply(updates)
    toast(t('bulkAction.applyDone', { op, n: updates.size }), 'success')
    // 清掉对应 input 让 user 知道"已应用、可以输下一组"
    if (op === 'add-front' || op === 'add-back' || op === 'remove') setTagsInput('')
    if (op === 'replace') { setOldTag(''); setNewTag('') }
  }

  const noneSelected = selectedKeys.length === 0
  const opDisabled = noneSelected

  return (
    <div className="px-2.5 py-2 flex flex-col gap-2 text-xs shrink-0 border-b border-subtle">
      {/* 顶部 toolbar：选中信息 + 全选/取消/去重 */}
      <div className="flex items-center gap-1.5 flex-wrap">
        <span className={noneSelected ? 'text-fg-tertiary' : 'text-accent font-mono'}>
          {t('bulkAction.selectedTotal', { n: selectedKeys.length, total: totalCount })}
        </span>
        <span className="flex-1" />
        <button
          onClick={onSelectAll}
          disabled={totalCount === 0}
          className="btn btn-ghost btn-sm"
        >{t('common.selectAll')}</button>
        <button
          onClick={onClearSelection}
          disabled={noneSelected}
          className="btn btn-ghost btn-sm"
        >{t('common.deselect')}</button>
        <button
          onClick={() => apply('dedupe')}
          disabled={opDisabled}
          className="btn btn-ghost btn-sm"
          title={t('bulkAction.dedupeHint')}
        >{t('bulkAction.dedupe')}</button>
      </div>

      {/* add / remove 共享一个 input：下面 3 个按钮决定动作 */}
      <div className="flex flex-col gap-1">
        <TagsField
          value={tagsInput}
          onChange={setTagsInput}
          placeholder={t('bulkAction.tagPlaceholder')}
          suggestions={tagSuggestions}
        />
        <div className="flex gap-1 flex-wrap">
          <button
            onClick={() => apply('add-front')}
            disabled={opDisabled}
            className="btn btn-secondary btn-sm"
            title={t('bulkAction.addFrontHint')}
          >+ {t('bulkAction.addFront')}</button>
          <button
            onClick={() => apply('add-back')}
            disabled={opDisabled}
            className="btn btn-secondary btn-sm"
            title={t('bulkAction.addBackHint')}
          >+ {t('bulkAction.addBack')}</button>
          <button
            onClick={() => apply('remove')}
            disabled={opDisabled}
            className="btn btn-secondary btn-sm"
            title={t('bulkAction.removeHint')}
          >− {t('bulkAction.removeTag')}</button>
        </div>
      </div>

      {/* replace：old → new 是约定语义，单独成行 */}
      <div className="flex items-center gap-1 flex-wrap">
        <span className="text-fg-tertiary shrink-0">{t('bulkAction.replace')}</span>
        <TagsField
          value={oldTag}
          onChange={setOldTag}
          placeholder={t('bulkAction.replaceOldPlaceholder')}
          suggestions={tagSuggestions}
          width={120}
        />
        <span className="text-fg-tertiary shrink-0">→</span>
        <TagsField
          value={newTag}
          onChange={setNewTag}
          placeholder={t('bulkAction.replaceNewPlaceholder')}
          suggestions={tagSuggestions}
          width={120}
        />
        <button
          onClick={() => apply('replace')}
          disabled={opDisabled}
          className="btn btn-secondary btn-sm"
        >✓</button>
      </div>
    </div>
  )
}

interface TagsFieldProps {
  value: string
  onChange: (v: string) => void
  placeholder: string
  suggestions: string[]
  width?: number
}

function TagsField({ value, onChange, placeholder, suggestions, width }: TagsFieldProps) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const tail = (() => {
    const m = value.match(/([^,，\n]*)$/)
    return (m ? m[1] : value).trim().toLowerCase()
  })()
  const matches = tail
    ? suggestions.filter((s) => s.toLowerCase().includes(tail) && s.toLowerCase() !== tail).slice(0, 8)
    : []

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  const pick = (s: string) => {
    const head = value.replace(/([^,，\n]*)$/, '')
    onChange(head + s); setOpen(false)
  }

  return (
    <div className="relative flex-1 min-w-0" ref={ref}>
      <input
        value={value}
        onChange={(e) => { onChange(e.target.value); setOpen(true) }}
        onFocus={() => setOpen(true)}
        placeholder={placeholder}
        className="input input-mono w-full"
        style={{ fontSize: 'var(--t-xs)', width: width ? `${width}px` : undefined }}
      />
      {open && matches.length > 0 && (
        <ul
          className="absolute left-0 top-full mt-0.5 z-30 bg-elevated border border-subtle rounded-sm shadow-lg max-h-[180px] overflow-y-auto min-w-[200px] list-none p-1 m-0"
          role="listbox"
        >
          {matches.map((s) => (
            <li
              key={s}
              onMouseDown={(e) => { e.preventDefault(); pick(s) }}
              className="px-2.5 py-1 text-xs font-mono text-fg-primary cursor-pointer hover:bg-overlay rounded-sm"
            >
              {s}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

