import type { LoraEntry, XYAxisType } from '../../../api/client'
import { AXIS_LABELS, AXIS_VALUE_TYPE, REQUIRES_LORA_INDEX, type XYAxisDraft } from './xy'

const ALL_AXES: XYAxisType[] = ['steps', 'cfg_scale', 'seed', 'sampler_name', 'lora_scale']

function placeholderFor(axis: XYAxisType): string {
  const t = AXIS_VALUE_TYPE[axis]
  if (axis === 'sampler_name') return 'er_sde, euler_a, ...'
  if (t === 'int') return '20, 25, 30'
  return '0.6, 0.8, 1.0'
}

function AxisCard({
  label, draft, onChange, onRemove, loras,
}: {
  label: 'X' | 'Y'
  draft: XYAxisDraft
  onChange: (d: XYAxisDraft) => void
  onRemove?: () => void
  loras: LoraEntry[]
}) {
  const needsLora = REQUIRES_LORA_INDEX.has(draft.axis)
  return (
    <div className="bg-sunken border border-subtle rounded-md px-2.5 py-2 flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold text-fg-secondary shrink-0 w-4">{label}</span>
        <select
          className="input text-xs flex-1"
          value={draft.axis}
          onChange={(e) => {
            const newAxis = e.target.value as XYAxisType
            onChange({
              ...draft,
              axis: newAxis,
              loraIndex: REQUIRES_LORA_INDEX.has(newAxis)
                ? (loras.length > 0 ? 0 : null)
                : null,
            })
          }}
        >
          {ALL_AXES.map((a) => (
            <option key={a} value={a}>{AXIS_LABELS[a]}</option>
          ))}
        </select>
        {onRemove && (
          <button
            onClick={onRemove}
            className="btn btn-ghost btn-sm text-fg-tertiary hover:text-err shrink-0 px-1.5"
            title="移除 Y 轴（退化到单轴）"
            aria-label="移除 Y 轴"
          >
            ×
          </button>
        )}
      </div>
      <input
        type="text"
        className="input font-mono text-xs"
        placeholder={placeholderFor(draft.axis)}
        value={draft.raw}
        onChange={(e) => onChange({ ...draft, raw: e.target.value })}
      />
      {needsLora && (
        <select
          className="input text-xs"
          value={draft.loraIndex ?? ''}
          onChange={(e) =>
            onChange({ ...draft, loraIndex: e.target.value === '' ? null : Number(e.target.value) })
          }
        >
          {loras.length === 0 ? (
            <option value="">— 先在上方添加一个 LoRA —</option>
          ) : (
            loras.map((l, i) => (
              <option key={i} value={i}>
                LoRA #{i + 1} · {l.path.split(/[\\/]/).pop() ?? l.path}
              </option>
            ))
          )}
        </select>
      )}
    </div>
  )
}

/** Sidebar 的 XY 轴配置区（仅 mode=xy 时渲染）。 */
export default function SidebarXYAxes({
  xDraft, yDraft, onXChange, onYChange, loras,
}: {
  xDraft: XYAxisDraft
  yDraft: XYAxisDraft | null
  onXChange: (d: XYAxisDraft) => void
  /** null = 移除 Y 轴退化到单轴 N×1；非 null = 添加/修改 Y */
  onYChange: (d: XYAxisDraft | null) => void
  loras: LoraEntry[]
}) {
  return (
    <div className="card" style={{ padding: 18 }}>
      <div className="flex items-center justify-between mb-3">
        <div className="text-md font-semibold">XY 轴</div>
      </div>
      <div className="flex flex-col gap-2">
        <AxisCard label="X" draft={xDraft} onChange={onXChange} loras={loras} />
        {yDraft ? (
          <AxisCard
            label="Y"
            draft={yDraft}
            onChange={onYChange}
            onRemove={() => onYChange(null)}
            loras={loras}
          />
        ) : (
          <button
            onClick={() => onYChange({ axis: 'cfg_scale', raw: '3.0, 4.0, 5.0', loraIndex: null })}
            className="btn btn-ghost btn-sm self-start text-xs text-fg-tertiary"
          >
            + 添加 Y 轴
          </button>
        )}
      </div>
    </div>
  )
}
