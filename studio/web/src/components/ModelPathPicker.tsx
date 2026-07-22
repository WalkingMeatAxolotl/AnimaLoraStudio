import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type ModelPathChoice } from '../api/client'

/**
 * 模型路径字段的 dropdown picker：点字段旁的「选择模型」按钮触发，列出模型设置里
 * 已就绪的模型，选中后写绝对路径回字段。范式同 ResumeFieldPicker（点外 / Esc
 * 关闭、选中项打勾、label 语义化而值仍是绝对路径）。
 *
 * 候选按 `family` 从后端取——哪个字段能选哪些资产是族知识，前端不拼。没下载的
 * 资产不在候选里（选了也训不起来，下载是 Settings 页的职责）；候选为空时
 * 调用方不渲染触发按钮，用户仍可手填或用「浏览」。
 */
export default function ModelPathPicker({
  field,
  family,
  value,
  onChange,
  onClose,
  anchorRef,
}: {
  field: string
  family: string
  value: string
  onChange: (path: string) => void
  onClose: () => void
  anchorRef: React.RefObject<HTMLElement | null>
}) {
  const { t } = useTranslation()
  const [choices, setChoices] = useState<ModelPathChoice[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const popRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    let alive = true
    api.getModelPathChoices(family)
      .then((r) => { if (alive) setChoices(r.choices[field] ?? []) })
      .catch((e) => { if (alive) setError(String(e)) })
    return () => { alive = false }
  }, [family, field])

  // 点外面关闭 + Esc 关闭
  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (popRef.current?.contains(e.target as Node)) return
      if (anchorRef.current?.contains(e.target as Node)) return
      onClose()
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [onClose, anchorRef])

  const groups: Array<{ key: string; items: ModelPathChoice[] }> = []
  for (const c of choices ?? []) {
    const g = groups.find((x) => x.key === c.group)
    if (g) g.items.push(c)
    else groups.push({ key: c.group, items: [c] })
  }

  return (
    <div
      ref={popRef}
      className="absolute z-40 mt-1 max-h-[360px] overflow-y-auto rounded-md border border-dim bg-elevated shadow-xl text-xs"
      style={{ top: '100%', left: 0, width: '100%', minWidth: 420 }}
    >
      {error ? (
        <div className="px-3 py-2 text-err">{error}</div>
      ) : choices === null ? (
        <div className="px-3 py-2 text-fg-tertiary italic">{t('modelPicker.loading')}</div>
      ) : choices.length === 0 ? (
        <div className="px-3 py-2 text-fg-tertiary italic">{t('modelPicker.empty')}</div>
      ) : (
        groups.map((g) => (
          <div key={g.key} className="border-b border-subtle last:border-0">
            <div className="px-3 py-1 bg-canvas text-fg-tertiary font-mono uppercase tracking-wider text-[10px] sticky top-0 border-b border-subtle">
              {t(`modelPicker.group.${g.key}`)}
            </div>
            {g.items.map((it) => (
              <PickRow
                key={it.path}
                label={it.label}
                note={it.note ? t(`modelPicker.note.${it.note}`, { defaultValue: it.note }) : ''}
                selected={it.path === value}
                onPick={() => { onChange(it.path); onClose() }}
              />
            ))}
          </div>
        ))
      )}
    </div>
  )
}

function PickRow({
  label, note, selected, onPick,
}: {
  label: string
  note: string
  selected: boolean
  onPick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onPick}
      className={
        'w-full text-left px-4 py-1.5 font-mono cursor-pointer transition-colors flex items-center gap-2 ' +
        (selected
          ? 'bg-accent-soft text-accent font-semibold'
          : 'text-fg-primary hover:bg-overlay')
      }
    >
      <span className={'w-3 inline-block ' + (selected ? '' : 'opacity-0')}>✓</span>
      <span>{label}</span>
      {note && <span className="text-fg-tertiary text-[10px] uppercase">{note}</span>}
    </button>
  )
}
