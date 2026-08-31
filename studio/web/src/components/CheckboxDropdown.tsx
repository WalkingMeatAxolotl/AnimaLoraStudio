// 下拉包裹的多选 checkbox 列表。
//
// 评估的样图矩阵要选 prompt 和 LoRA，两者都可能是几十上百项 —— 平铺会把控制区撑爆，
// 原生 <select multiple> 又没法看清勾了哪些。收进一个 popover：按钮上显示「已选 n/m」，
// 展开才是完整清单。
import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import Button from './Button'

export interface CheckboxDropdownOption {
  value: string
  label: string
  /** hover 时的完整内容（prompt 常是一长串 tag）。 */
  title?: string
}

export default function CheckboxDropdown({
  label, options, selected, onChange, emptyHint,
}: {
  label: string
  options: CheckboxDropdownOption[]
  selected: Set<string>
  onChange: (next: Set<string>) => void
  emptyHint?: string
}) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLDivElement | null>(null)

  // 点外面收起（popover 覆盖在网格上，不收起会挡住内容）
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  const allValues = useMemo(() => options.map((o) => o.value), [options])
  const allPicked = allValues.length > 0 && allValues.every((v) => selected.has(v))

  const toggle = (value: string) => {
    const next = new Set(selected)
    if (next.has(value)) next.delete(value)
    else next.add(value)
    onChange(next)
  }

  return (
    <div ref={wrapRef} className="relative">
      <Button
        variant="secondary"
        size="xs"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label={label}
        className="font-mono"
      >
        <span className="font-sans font-semibold">{label}</span>
        <span className="text-fg-tertiary">
          {selected.size}/{options.length}
        </span>
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className={`text-fg-tertiary transition-transform ${open ? 'rotate-180' : ''}`}
          aria-hidden="true"
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </Button>

      {open && (
        <div
          // 右对齐：这两个下拉挂在 header 的最右端，左对齐会整块伸出屏幕外。
          // maxWidth 用 vw 兜底，防止超长 prompt 把 popover 撑出视口。
          className="absolute right-0 z-30 mt-1 rounded-md border border-subtle bg-elevated shadow-xl flex flex-col"
          style={{ minWidth: 240, maxWidth: 'min(420px, 80vw)' }}
        >
          <div className="flex items-center gap-2 px-2.5 py-1.5 border-b border-subtle">
            <span className="text-xs text-fg-tertiary flex-1">
              {t('common.selectedCount', { selected: selected.size, total: options.length })}
            </span>
            <Button
              variant="ghost"
              size="xs"
              onClick={() => onChange(allPicked ? new Set() : new Set(allValues))}
            >
              {allPicked ? t('common.deselect') : t('common.selectAll')}
            </Button>
          </div>
          <div className="overflow-y-auto py-1" style={{ maxHeight: 300 }}>
            {options.length === 0 ? (
              <div className="px-2.5 py-2 text-xs text-fg-tertiary">
                {emptyHint ?? t('common.noOptions')}
              </div>
            ) : options.map((o) => (
              <label
                key={o.value}
                className="flex items-center gap-1.5 px-2.5 py-1 text-xs cursor-pointer hover:bg-overlay"
                title={o.title ?? o.label}
              >
                <input
                  type="checkbox"
                  checked={selected.has(o.value)}
                  onChange={() => toggle(o.value)}
                />
                <span className="truncate font-mono text-fg-secondary">{o.label}</span>
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
