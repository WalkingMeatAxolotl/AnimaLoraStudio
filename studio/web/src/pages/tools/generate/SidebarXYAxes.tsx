import { useState } from 'react'
import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
  type DragEndEvent,
  type DragOverEvent,
  type DragStartEvent,
} from '@dnd-kit/core'
import {
  SortableContext,
  arrayMove,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useTranslation } from 'react-i18next'
import { normalizeLoraPath } from './loraSelection'
import { SegmentedControl, SidebarToolIcon, ToolbarAction } from './SidebarToolbar'
import { axisLabel, axisView, cellCount, formatAxisValue, splitAxisRaw, type XYAxisDraft } from './xy'

function checkpointName(path: string): string {
  return path.split(/[\\/]/).pop()?.replace(/\.safetensors$/i, '') ?? path
}

function withCheckpointPaths(draft: XYAxisDraft, paths: string[]): XYAxisDraft {
  const currentAnchor = draft.checkpointAnchor ?? null
  const anchorStillSelected = currentAnchor
    ? paths.some((path) => normalizeLoraPath(path) === normalizeLoraPath(currentAnchor.path))
    : false
  return {
    ...draft,
    raw: paths.join(', '),
    checkpointAnchor: anchorStillSelected
      ? currentAnchor
      : paths[0]
        ? { path: paths[0], scale: 1, project_id: null, version_id: null }
        : null,
  }
}

function updateAxisValues(draft: XYAxisDraft, values: string[]): XYAxisDraft {
  return draft.axis === 'lora_ckpt'
    ? withCheckpointPaths(draft, values)
    : { ...draft, raw: values.join(', ') }
}

function SortableAxisValueCard({
  id,
  label,
  dropEdge,
  onRemove,
}: {
  id: string
  label: string
  dropEdge: 'before' | 'after' | null
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

  return (
    <div
      ref={setNodeRef}
      className="relative py-1"
      style={{
        transform: CSS.Transform.toString(transform),
        transition,
        opacity: isDragging ? 0.5 : 1,
        zIndex: isDragging ? 1 : undefined,
      }}
    >
      {dropEdge === 'before' && (
        <span
          className="absolute inset-x-1 top-0 h-0.5 rounded-full bg-accent pointer-events-none"
          data-testid="xy-axis-drop-indicator"
        />
      )}
      <div className="relative group">
        <div
          className={`xy-axis-sortable-row flex items-center rounded-md border bg-overlay p-2.5 pr-12 select-none touch-none cursor-grab active:cursor-grabbing transition-[background-color,border-color,box-shadow,opacity] ${isDragging ? 'border-accent shadow-md' : 'border-subtle'}`}
          data-testid="xy-axis-selected-value"
          title={t('generate.axisDrag')}
          aria-label={`${t('generate.axisDrag')} ${label}`}
          {...attributes}
          {...listeners}
        >
          <span className="font-mono text-xs text-fg-primary flex-1 min-w-0 truncate" title={label}>
            {label}
          </span>
        </div>
        <button
          type="button"
          className="btn btn-ghost btn-sm text-err absolute right-2 top-1/2 -translate-y-1/2 z-[2] opacity-0 pointer-events-none transition-opacity group-hover:opacity-100 group-hover:pointer-events-auto group-focus-within:opacity-100 group-focus-within:pointer-events-auto"
          onPointerDown={(event) => event.stopPropagation()}
          onKeyDown={(event) => event.stopPropagation()}
          onClick={onRemove}
          title={t('common.delete')}
          aria-label={`${t('common.delete')} ${label}`}
        >×</button>
      </div>
      {dropEdge === 'after' && (
        <span
          className="absolute inset-x-1 bottom-0 h-0.5 rounded-full bg-accent pointer-events-none"
          data-testid="xy-axis-drop-indicator"
        />
      )}
    </div>
  )
}

function AxisValueList({
  draft,
  onChange,
  onManualReorder,
}: {
  draft: XYAxisDraft
  onChange: (draft: XYAxisDraft) => void
  onManualReorder: () => void
}) {
  const { t } = useTranslation()
  const values = splitAxisRaw(draft.raw)
  const ids = values.map((value, index) => `${index}:${normalizeLoraPath(value)}`)
  const [dragTarget, setDragTarget] = useState<{ activeId: string; overId: string } | null>(null)
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  )

  const commit = (next: string[], reordered = false) => {
    if (reordered && draft.axis === 'lora_ckpt') onManualReorder()
    onChange(updateAxisValues(draft, next))
  }
  const handleDragStart = ({ active }: DragStartEvent) => {
    const activeId = String(active.id)
    setDragTarget({ activeId, overId: activeId })
  }
  const handleDragOver = ({ active, over }: DragOverEvent) => {
    setDragTarget(over ? { activeId: String(active.id), overId: String(over.id) } : null)
  }
  const handleDragEnd = ({ active, over }: DragEndEvent) => {
    setDragTarget(null)
    if (!over || active.id === over.id) return
    const from = ids.indexOf(String(active.id))
    const to = ids.indexOf(String(over.id))
    if (from < 0 || to < 0) return
    commit(arrayMove(values, from, to), true)
  }
  const handleDragCancel = () => setDragTarget(null)

  const activeIndex = dragTarget ? ids.indexOf(dragTarget.activeId) : -1
  const overIndex = dragTarget ? ids.indexOf(dragTarget.overId) : -1
  const dropEdge = activeIndex >= 0 && overIndex >= 0 && activeIndex !== overIndex
    ? activeIndex < overIndex ? 'after' : 'before'
    : null

  if (values.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-subtle px-3 py-5 text-center text-xs text-fg-tertiary">
        {t('generate.axisNoValues')}
      </div>
    )
  }

  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      onDragStart={handleDragStart}
      onDragOver={handleDragOver}
      onDragEnd={handleDragEnd}
      onDragCancel={handleDragCancel}
    >
      <SortableContext items={ids} strategy={verticalListSortingStrategy}>
        <div className="flex flex-col" data-testid="xy-axis-selected-values">
          {values.map((value, index) => {
            const label = draft.axis === 'lora_ckpt'
              ? checkpointName(value)
              : formatAxisValue(draft.axis, value)
            return (
              <SortableAxisValueCard
                key={ids[index]}
                id={ids[index]}
                label={label}
                dropEdge={dragTarget?.overId === ids[index] ? dropEdge : null}
                onRemove={() => commit(values.filter((_, valueIndex) => valueIndex !== index))}
              />
            )
          })}
        </div>
      </SortableContext>
    </DndContext>
  )
}

export function XYAxisToolbar({
  xDraft,
  yDraft,
  activeAxis,
  onSelectAxis,
  onSwap,
}: {
  xDraft: XYAxisDraft
  yDraft: XYAxisDraft
  activeAxis: 'X' | 'Y'
  onSelectAxis: (axis: 'X' | 'Y') => void
  onSwap: () => void
}) {
  const { t } = useTranslation()

  return (
    <div className="flex items-center gap-1.5" data-testid="xy-axis-toolbar">
      <SegmentedControl
        items={[
          {
            value: 'X',
            label: `X · ${axisLabel(xDraft.axis)}`,
            controls: 'xy-active-axis-panel',
          },
          {
            value: 'Y',
            label: `Y · ${axisLabel(yDraft.axis)}`,
            controls: 'xy-active-axis-panel',
          },
        ]}
        value={activeAxis}
        onChange={onSelectAxis}
        ariaLabel={t('generate.xyAxes')}
        density="compact"
        idPrefix="xy-axis-tab"
        className="flex-1"
      />
      <ToolbarAction
        label={t('generate.swapAxes')}
        icon={<SidebarToolIcon name="swap" />}
        iconOnly
        onClick={onSwap}
      />
    </div>
  )
}

export default function SidebarXYAxes({
  xDraft,
  yDraft,
  yEnabled,
  activeAxis,
  fp8BaseModel,
  onAxisChange,
  onManualReorder,
}: {
  xDraft: XYAxisDraft
  yDraft: XYAxisDraft
  yEnabled: boolean
  activeAxis: 'X' | 'Y'
  fp8BaseModel: boolean
  onAxisChange: (axis: 'X' | 'Y', draft: XYAxisDraft) => void
  onManualReorder: (axis: 'X' | 'Y') => void
}) {
  const { t } = useTranslation()
  const activeDraft = activeAxis === 'X' ? xDraft : yDraft
  const total = cellCount(
    axisView(xDraft).values.length,
    yEnabled ? axisView(yDraft).values.length : null,
  )
  const fp8MergeHeavy = fp8BaseModel
    && yEnabled
    && [xDraft.axis, yDraft.axis].includes('lora_ckpt')
    && [xDraft.axis, yDraft.axis].includes('lora_scale')

  return (
    <div className="flex flex-col gap-3" data-testid="xy-axes-panel">
      <div
        id="xy-active-axis-panel"
        role="tabpanel"
        aria-labelledby={`xy-axis-tab-${activeAxis.toLocaleLowerCase()}`}
      >
        <AxisValueList
          draft={activeDraft}
          onChange={(next) => onAxisChange(activeAxis, next)}
          onManualReorder={() => onManualReorder(activeAxis)}
        />
      </div>

      {total > 50 && (
        <div className="text-2xs text-warn">{t('generate.xyLargeMatrixWarning')}</div>
      )}
      {fp8MergeHeavy && (
        <div className="text-2xs text-warn">{t('generate.xyFp8MergeWarning')}</div>
      )}
    </div>
  )
}
