import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core'
import {
  SortableContext,
  arrayMove,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { LoraEntry } from '../../../api/client'
import {
  applyLoraText,
  LoraTextError,
  loraTextName,
  serializeLoraText,
  type LoraUiState,
} from './loraSelection'

export function reorderLoraSelection(
  loras: LoraEntry[],
  ui: LoraUiState[],
  activeId: string,
  overId: string,
): { loras: LoraEntry[]; ui: LoraUiState[] } | null {
  const ids = loras.map((_, index) => ui[index]?.id ?? `missing-${index}`)
  const from = ids.indexOf(activeId)
  const to = ids.indexOf(overId)
  if (from < 0 || to < 0 || from === to) return null
  return {
    loras: arrayMove(loras, from, to),
    ui: arrayMove(ui, from, to),
  }
}

function SortableLoraCard({
  id,
  entry,
  state,
  onEnabledChange,
  onWeightChange,
  onRemove,
}: {
  id: string
  entry: LoraEntry
  state: LoraUiState | undefined
  onEnabledChange: (enabled: boolean) => void
  onWeightChange: (scale: number) => void
  onRemove: () => void
}) {
  const { t } = useTranslation()
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id })
  const missing = !entry.path.trim()
  const enabled = !missing && state?.enabled !== false
  const name = loraTextName(entry) || t('generate.unknownLora')

  return (
    <div
      ref={setNodeRef}
      className={`lora-sortable-card group rounded-md border p-2.5 transition-[background-color,border-color,box-shadow,opacity] ${
        missing
          ? 'border-err bg-err-soft'
          : isDragging
            ? 'border-accent bg-overlay shadow-md'
            : 'border-subtle bg-overlay'
      }`}
      style={{
        transform: CSS.Transform.toString(transform),
        transition,
        opacity: isDragging ? 0.5 : enabled || missing ? 1 : 0.62,
        zIndex: isDragging ? 1 : undefined,
      }}
      data-lora-id={id}
    >
      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={enabled}
          disabled={missing}
          onPointerDown={(event) => event.stopPropagation()}
          onChange={(event) => onEnabledChange(event.target.checked)}
          title={t('generate.loraEnabled')}
          aria-label={`${t('generate.loraEnabled')} ${name}`}
          className="shrink-0"
        />
        <div
          className="flex-1 min-w-0 cursor-grab active:cursor-grabbing select-none touch-none"
          title={t('generate.axisDrag')}
          aria-label={`${t('generate.axisDrag')} ${name}`}
          {...attributes}
          {...listeners}
        >
          <div className="font-mono text-xs text-fg-primary truncate" title={name}>{name}</div>
          {missing && <div className="text-2xs text-err truncate mt-0.5">{t('generate.loraNotFoundHint')}</div>}
        </div>
        <button
          type="button"
          className="btn btn-ghost btn-sm text-err shrink-0 opacity-0 pointer-events-none transition-opacity group-hover:opacity-100 group-hover:pointer-events-auto group-focus-within:opacity-100 group-focus-within:pointer-events-auto"
          onPointerDown={(event) => event.stopPropagation()}
          onClick={onRemove}
          title={t('generate.removeLora')}
          aria-label={`${t('generate.removeLora')} ${name}`}
        >
          ×
        </button>
        {!missing && (
          <input
            type="number"
            min={0}
            max={1.5}
            step={0.05}
            value={entry.scale}
            onPointerDown={(event) => event.stopPropagation()}
            onChange={(event) => onWeightChange(Number(event.target.value))}
            aria-label={`${t('generate.weightValue')} ${name}`}
            className="input input-mono text-xs shrink-0"
            style={{ width: 70, padding: '3px 5px' }}
          />
        )}
      </div>
    </div>
  )
}

export default function SidebarLoras({
  loras,
  ui,
  onChange,
}: {
  loras: LoraEntry[]
  ui: LoraUiState[]
  onChange: (loras: LoraEntry[], ui: LoraUiState[]) => void
}) {
  const { t } = useTranslation()
  const [text, setText] = useState(() => serializeLoraText(loras, ui))
  const [textError, setTextError] = useState<string | null>(null)
  const textFocused = useRef(false)
  const summary = useMemo(() => serializeLoraText(loras, ui), [loras, ui])
  const ids = loras.map((_, index) => ui[index]?.id ?? `missing-${index}`)
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  )

  useEffect(() => {
    if (!textFocused.current) setText(summary)
  }, [summary])

  const updateEntry = (index: number, patch: Partial<LoraEntry>) => {
    onChange(loras.map((entry, i) => (i === index ? { ...entry, ...patch } : entry)), ui)
  }

  const removeEntry = (index: number) => {
    onChange(loras.filter((_, i) => i !== index), ui.filter((_, i) => i !== index))
  }

  const handleDragEnd = ({ active, over }: DragEndEvent) => {
    if (!over) return
    const next = reorderLoraSelection(loras, ui, String(active.id), String(over.id))
    if (next) onChange(next.loras, next.ui)
  }

  const applyText = () => {
    try {
      const result = applyLoraText(text, loras, ui)
      onChange(result.loras, result.ui)
      setText(serializeLoraText(result.loras, result.ui))
      setTextError(null)
    } catch (error) {
      if (error instanceof LoraTextError) {
        setTextError(t(`generate.loraTextError.${error.code}`, { name: error.value }))
      } else {
        setTextError(String(error))
      }
    }
  }

  return (
    <div className="flex flex-col gap-3" data-testid="current-lora-panel">
      <div>
        <textarea
          className={`input input-mono w-full text-xs resize-y ${textError ? 'border-err' : ''}`}
          style={{ minHeight: 82 }}
          value={text}
          onFocus={() => { textFocused.current = true }}
          onBlur={() => { textFocused.current = false; applyText() }}
          onChange={(event) => { setText(event.target.value); setTextError(null) }}
          onKeyDown={(event) => {
            if (event.ctrlKey && event.key === 'Enter') {
              event.preventDefault()
              applyText()
            }
          }}
          placeholder="<lora:name:1>"
          aria-label={t('generate.loraText')}
          aria-invalid={Boolean(textError)}
        />
        {textError && <div className="text-xs text-err mt-1" role="alert">{textError}</div>}
      </div>

      {loras.length === 0 && (
        <div className="rounded-md border border-dashed border-subtle p-5 text-center text-xs text-fg-tertiary">
          {t('generate.currentLorasEmpty')}
        </div>
      )}

      <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
        <SortableContext items={ids} strategy={verticalListSortingStrategy}>
          <div className="flex flex-col gap-2" data-testid="current-lora-list">
            {loras.map((entry, index) => (
              <SortableLoraCard
                key={ids[index]}
                id={ids[index]}
                entry={entry}
                state={ui[index]}
                onEnabledChange={(enabled) => onChange(loras, ui.map((item, i) => (
                  i === index ? { ...item, enabled } : item
                )))}
                onWeightChange={(scale) => updateEntry(index, { scale })}
                onRemove={() => removeEntry(index)}
              />
            ))}
          </div>
        </SortableContext>
      </DndContext>
    </div>
  )
}
