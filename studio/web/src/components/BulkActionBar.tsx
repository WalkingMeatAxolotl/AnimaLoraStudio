import { useEffect, useRef, useState, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { useDialog } from './Dialog'
import { useToast } from './Toast'

type Op = 'add' | 'remove' | 'replace' | 'dedupe'
type Position = 'front' | 'back'

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

/** 批量操作面板 — V2「行式」布局。
 *
 * 四个操作 (添加 / 删除 / 替换 / 去重) 各占一行，按钮在同一条竖线上：
 * `[icon·label] [input(s)] [toggle / spacer] [action button]`。
 * 节奏统一，按钮归属明确（首部/尾部 只挂在添加行内）。
 *
 * - **零 popover**：所有 input 常驻可见。
 * - **零 scope**：永远操作 selectedKeys（要全部 → 先「全选图片」按钮）。
 * - **add / remove 各有自己的 input**：避免「一个输入框两个按钮」的归属歧义。
 * - **每个 op 都过 useDialog().confirm**：影响张数预计算（pre-compute
 *   updates → 拿 size），用户看到的"N 张"是真实数。
 */
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
  const { confirm } = useDialog()
  const [addInput, setAddInput] = useState('')
  const [removeInput, setRemoveInput] = useState('')
  const [oldTag, setOldTag] = useState('')
  const [newTag, setNewTag] = useState('')
  const [position, setPosition] = useState<Position>('front')

  const parseTags = (raw: string): string[] =>
    raw.split(/[,，\n]/).map((s) => s.trim()).filter(Boolean)

  const computeUpdates = (op: Op): Map<string, string[]> => {
    const updates = new Map<string, string[]>()
    const keys = selectedKeys

    if (op === 'add') {
      const ts = parseTags(addInput)
      const insertFront = position === 'front'
      for (const k of keys) {
        const cur = cache.get(k) ?? []
        const have = new Set(cur)
        const toAdd = ts.filter((tag) => !have.has(tag))
        if (toAdd.length === 0) continue
        updates.set(k, insertFront ? [...toAdd, ...cur] : [...cur, ...toAdd])
      }
    } else if (op === 'remove') {
      const drop = new Set(parseTags(removeInput))
      for (const k of keys) {
        const cur = cache.get(k) ?? []
        const next = cur.filter((tag) => !drop.has(tag))
        if (next.length !== cur.length) updates.set(k, next)
      }
    } else if (op === 'replace') {
      const o = oldTag.trim()
      const n = newTag.trim()
      for (const k of keys) {
        const cur = cache.get(k) ?? []
        if (!cur.includes(o)) continue
        const next: string[] = []
        const seen = new Set<string>()
        for (const tag of cur) {
          const out = tag === o ? n : tag
          if (seen.has(out)) continue
          seen.add(out); next.push(out)
        }
        updates.set(k, next)
      }
    } else if (op === 'dedupe') {
      for (const k of keys) {
        const cur = cache.get(k) ?? []
        const seen = new Set<string>(); const next: string[] = []
        for (const tag of cur) { if (seen.has(tag)) continue; seen.add(tag); next.push(tag) }
        if (next.length !== cur.length) updates.set(k, next)
      }
    }
    return updates
  }

  const opLabel = (op: Op): string => {
    if (op === 'add') {
      return t(position === 'front' ? 'bulkAction.opLabelAddFront' : 'bulkAction.opLabelAddBack',
        { tags: addInput.trim() })
    }
    if (op === 'remove') return t('bulkAction.opLabelRemove', { tags: removeInput.trim() })
    if (op === 'replace') return t('bulkAction.opLabelReplace', { from: oldTag.trim(), to: newTag.trim() })
    return t('bulkAction.opLabelDedupe')
  }

  const apply = async (op: Op) => {
    if (selectedKeys.length === 0) {
      toast(t('bulkAction.noFiles'), 'error')
      return
    }
    if (op === 'add' && parseTags(addInput).length === 0) {
      toast(t('bulkAction.enterTag'), 'error'); return
    }
    if (op === 'remove' && parseTags(removeInput).length === 0) {
      toast(t('bulkAction.enterTag'), 'error'); return
    }
    if (op === 'replace' && (!oldTag.trim() || !newTag.trim())) {
      toast(t('bulkAction.replaceNeedsOldNew'), 'error'); return
    }

    const updates = computeUpdates(op)
    if (updates.size === 0) {
      toast(t('bulkAction.noChanges', { op }), 'success')
      return
    }

    const ok = await confirm(
      t('bulkAction.confirmMessage', { op: opLabel(op), n: updates.size }),
      { tone: op === 'remove' || op === 'replace' ? 'danger' : 'warn',
        title: t('bulkAction.confirmTitle') },
    )
    if (!ok) return

    onApply(updates)
    toast(t('bulkAction.applyDone', { op, n: updates.size }), 'success')
    if (op === 'add') setAddInput('')
    if (op === 'remove') setRemoveInput('')
    if (op === 'replace') { setOldTag(''); setNewTag('') }
  }

  const noneSelected = selectedKeys.length === 0
  const opDisabled = noneSelected

  return (
    <div className="px-2.5 py-2 flex flex-col gap-2 text-xs shrink-0 border-b border-subtle">
      {/* selection summary + selection management — kept as before. */}
      <div className="flex items-center gap-1.5 flex-wrap">
        <span className={noneSelected ? 'text-fg-tertiary' : 'text-accent font-mono'}>
          {t('bulkAction.selectedTotal', { n: selectedKeys.length, total: totalCount })}
        </span>
        <span className="flex-1" />
        <button
          onClick={onSelectAll}
          disabled={totalCount === 0}
          className="btn btn-ghost btn-sm"
          title={t('bulkAction.selectAllImagesHint')}
        >{t('bulkAction.selectAllImages')}</button>
        <button
          onClick={onClearSelection}
          disabled={noneSelected}
          className="btn btn-ghost btn-sm"
        >{t('common.deselect')}</button>
      </div>

      {/* V2 行式：四操作各一行，按钮列右对齐。 */}
      <div className="rounded-md border border-subtle overflow-hidden">
        <BulkRow icon={ICON.plus} label={t('bulkAction.add')}>
          <TagsField
            value={addInput}
            onChange={setAddInput}
            placeholder={t('bulkAction.tagPlaceholder')}
            ariaLabel={t('bulkAction.addAria')}
            suggestions={tagSuggestions}
          />
          <PositionToggle position={position} onChange={setPosition} t={t} />
          <RowButton
            onClick={() => void apply('add')}
            disabled={opDisabled}
            tone="primary"
            title={t(position === 'front' ? 'bulkAction.addFrontHint' : 'bulkAction.addBackHint')}
          >{t('bulkAction.add')}</RowButton>
        </BulkRow>

        <BulkRow icon={ICON.minus} label={t('bulkAction.removeTag')} tone="danger">
          <TagsField
            value={removeInput}
            onChange={setRemoveInput}
            placeholder={t('bulkAction.tagPlaceholder')}
            ariaLabel={t('bulkAction.removeAria')}
            suggestions={tagSuggestions}
          />
          <ToggleSpacer />
          <RowButton
            onClick={() => void apply('remove')}
            disabled={opDisabled}
            tone="danger"
            title={t('bulkAction.removeHint')}
          >{t('bulkAction.removeTag')}</RowButton>
        </BulkRow>

        <BulkRow icon={ICON.replace} label={t('bulkAction.replace')}>
          <TagsField
            value={oldTag}
            onChange={setOldTag}
            placeholder={t('bulkAction.replaceOldPlaceholder')}
            suggestions={tagSuggestions}
          />
          <span className="text-fg-tertiary shrink-0">→</span>
          <TagsField
            value={newTag}
            onChange={setNewTag}
            placeholder={t('bulkAction.replaceNewPlaceholder')}
            suggestions={tagSuggestions}
          />
          <RowButton
            onClick={() => void apply('replace')}
            disabled={opDisabled}
            tone="ghost"
          >{t('bulkAction.replace')}</RowButton>
        </BulkRow>

        <BulkRow icon={ICON.dedupe} label={t('bulkAction.dedupe')} last>
          <span className="flex-1 min-w-0 text-fg-tertiary font-mono truncate">
            {t('bulkAction.dedupeRowHint')}
          </span>
          <ToggleSpacer />
          <RowButton
            onClick={() => void apply('dedupe')}
            disabled={opDisabled}
            tone="ghost"
            title={t('bulkAction.dedupeHint')}
          >{t('bulkAction.dedupe')}</RowButton>
        </BulkRow>
      </div>
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────── */
/* row scaffolding                                                            */
/* ────────────────────────────────────────────────────────────────────────── */

function BulkRow({
  icon,
  label,
  tone,
  last,
  children,
}: {
  icon: ReactNode
  label: string
  tone?: 'danger'
  last?: boolean
  children: ReactNode
}) {
  const toneClass = tone === 'danger' ? 'text-err' : 'text-fg-secondary'
  return (
    <div
      className={
        'flex items-center gap-2 px-2.5 py-2 ' +
        (last ? '' : 'border-b border-subtle')
      }
    >
      <div className={'inline-flex items-center gap-1.5 shrink-0 w-16 ' + toneClass}>
        <RowIcon>{icon}</RowIcon>
        <span className="text-xs font-medium">{label}</span>
      </div>
      {children}
    </div>
  )
}

function RowIcon({ children }: { children: ReactNode }) {
  return (
    <svg
      width="13"
      height="13"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  )
}

/** 在「删除」「去重」行里占位，宽度对齐到「添加」行的首部/尾部 toggle 列，
 * 保证四行的右按钮在同一条竖线上。 */
function ToggleSpacer() {
  // 64px ≈ 首部/尾部 segmented 的渲染宽度（含 padding + border）。
  return <span aria-hidden="true" className="shrink-0" style={{ width: 64 }} />
}

function RowButton({
  onClick,
  disabled,
  tone,
  title,
  children,
}: {
  onClick: () => void
  disabled?: boolean
  tone: 'primary' | 'ghost' | 'danger'
  title?: string
  children: ReactNode
}) {
  // 设计稿 V2：只有「添加」是 filled primary，其它三个（删除 / 替换 / 去重）都是
  // outline。删除走 err 着色（text + 边），保持和 替换 / 去重 同等视觉份量 —
  // 不再 filled，避免抢走 caption 列表的注意力。
  const baseCls = 'btn btn-sm shrink-0 justify-center'
  const cls =
    tone === 'primary'
      ? `btn-primary ${baseCls}`
      : `btn-secondary ${baseCls}`
  const dangerStyle =
    tone === 'danger'
      ? {
          color: 'var(--err)',
          borderColor:
            'color-mix(in oklch, var(--err) 35%, var(--border-default))',
        }
      : {}
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={cls}
      style={{ minWidth: 64, ...dangerStyle }}
    >
      {children}
    </button>
  )
}

const ICON = {
  plus: <path d="M8 3v10M3 8h10" />,
  minus: <path d="M3 8h10" />,
  replace: (
    <g>
      <path d="M3 5h7" />
      <path d="M6 2L3 5l3 3" />
      <path d="M13 11H6" />
      <path d="M10 14l3-3-3-3" />
    </g>
  ),
  dedupe: (
    <g>
      <rect x="3" y="3" width="6" height="6" rx="1" />
      <rect x="7" y="7" width="6" height="6" rx="1" />
    </g>
  ),
}

/* ────────────────────────────────────────────────────────────────────────── */
/* position toggle (segmented control: 首部 / 尾部)                          */
/* ────────────────────────────────────────────────────────────────────────── */

function PositionToggle({
  position,
  onChange,
  t,
}: {
  position: Position
  onChange: (p: Position) => void
  t: (k: string) => string
}) {
  // 设计稿 V2：container 是 bg-sunken 的小坑，激活态是「弹出的小台」—
  // bg-canvas + 仅 accent 文字 + 一道 accent 着色的 inset 边，整体很克制，
  // 不抢「添加」主按钮的颜色。
  const activeStyle = {
    background: 'var(--bg-canvas)',
    color: 'var(--accent)',
    boxShadow:
      'inset 0 0 0 1px color-mix(in oklch, var(--accent) 30%, var(--border-default))',
  }
  const segClass =
    'px-1.5 py-0.5 text-[11px] font-mono cursor-pointer border-0 bg-transparent'
  return (
    <div
      className="inline-flex rounded-sm shrink-0 p-[2px] gap-[2px]"
      style={{
        background: 'var(--bg-sunken)',
        border: '1px solid var(--border-subtle)',
      }}
    >
      <button
        type="button"
        onClick={() => onChange('front')}
        className={segClass + ' rounded-[3px]'}
        style={position === 'front' ? activeStyle : { color: 'var(--fg-secondary)' }}
        aria-pressed={position === 'front'}
      >{t('bulkAction.posFront')}</button>
      <button
        type="button"
        onClick={() => onChange('back')}
        className={segClass + ' rounded-[3px]'}
        style={position === 'back' ? activeStyle : { color: 'var(--fg-secondary)' }}
        aria-pressed={position === 'back'}
      >{t('bulkAction.posBack')}</button>
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────── */
/* tags input with autocomplete                                               */
/* ────────────────────────────────────────────────────────────────────────── */

interface TagsFieldProps {
  value: string
  onChange: (v: string) => void
  placeholder: string
  suggestions: string[]
  ariaLabel?: string
}

function TagsField({ value, onChange, placeholder, suggestions, ariaLabel }: TagsFieldProps) {
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
        aria-label={ariaLabel}
        className="input input-mono w-full"
        style={{ fontSize: 'var(--t-xs)' }}
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
