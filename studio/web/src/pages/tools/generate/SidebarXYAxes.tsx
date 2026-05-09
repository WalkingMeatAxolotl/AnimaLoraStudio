import { useEffect, useState } from 'react'
import { api, type LoraCkpt, type LoraEntry, type XYAxisType } from '../../../api/client'
import { AXIS_LABELS, AXIS_VALUE_TYPE, REQUIRES_LORA_INDEX, type XYAxisDraft } from './xy'

const ALL_AXES: XYAxisType[] = ['steps', 'cfg_scale', 'lora_scale', 'lora_ckpt']

function placeholderFor(axis: XYAxisType): string {
  const t = AXIS_VALUE_TYPE[axis]
  if (t === 'int') return '20, 25, 30'
  return '0.6, 0.8, 1.0'
}

/** 把 raw 字符串解析成 path 数组。"/a.safetensors, /b.safetensors" → ["/a.safetensors", "/b.safetensors"] */
function parsePathList(raw: string): string[] {
  return raw.split(',').map((s) => s.trim()).filter(Boolean)
}

function CkptMultiPicker({
  versionId, projectId, raw, onChange,
}: {
  versionId: number
  projectId: number
  raw: string
  onChange: (raw: string) => void
}) {
  const [ckpts, setCkpts] = useState<LoraCkpt[]>([])
  const [loaded, setLoaded] = useState(false)
  useEffect(() => {
    let cancelled = false
    setLoaded(false)
    void api.listVersionLoraCkpts(projectId, versionId)
      .then((items) => {
        if (cancelled) return
        setCkpts(items); setLoaded(true)
      })
      .catch(() => { if (!cancelled) { setCkpts([]); setLoaded(true) } })
    return () => { cancelled = true }
  }, [projectId, versionId])

  const selected = new Set(parsePathList(raw))
  const toggle = (p: string) => {
    const next = new Set(selected)
    if (next.has(p)) next.delete(p); else next.add(p)
    onChange(Array.from(next).join(', '))
  }

  if (!loaded) {
    return <div className="text-2xs text-fg-tertiary">加载 ckpt…</div>
  }
  if (ckpts.length === 0) {
    return <div className="text-2xs text-fg-tertiary">该 LoRA 没扫到 ckpt 文件（output/ 下需有 *.safetensors）</div>
  }
  return (
    <div className="flex flex-wrap gap-1">
      {ckpts.map((c) => {
        const on = selected.has(c.path)
        return (
          <button
            key={c.path}
            type="button"
            onClick={() => toggle(c.path)}
            className="font-mono"
            style={{
              fontSize: 11,
              padding: '3px 9px',
              borderRadius: 999,
              border: on ? '1px solid transparent' : '1px solid var(--border-subtle)',
              background: on ? 'var(--accent-soft)' : 'var(--bg-sunken)',
              color: on ? 'var(--accent)' : 'var(--fg-secondary)',
              cursor: 'pointer',
            }}
            title={c.path}
          >
            {on ? '✓ ' : '+ '}{c.label}
          </button>
        )
      })}
    </div>
  )
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
  const isCkpt = draft.axis === 'lora_ckpt'
  const boundLora =
    draft.loraIndex !== null && draft.loraIndex < loras.length
      ? loras[draft.loraIndex]
      : null
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
              raw: newAxis === 'lora_ckpt' ? '' : draft.raw,
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

      {/* lora_ckpt 用 checkbox 多选；其他轴用 text input */}
      {isCkpt ? (
        boundLora && boundLora.version_id && boundLora.project_id ? (
          <CkptMultiPicker
            projectId={boundLora.project_id}
            versionId={boundLora.version_id}
            raw={draft.raw}
            onChange={(raw) => onChange({ ...draft, raw })}
          />
        ) : (
          <div className="text-2xs text-fg-tertiary">
            选一个绑定到项目的 LoRA（外部文件不行）
          </div>
        )
      ) : (
        <input
          type="text"
          className="input font-mono text-xs"
          placeholder={placeholderFor(draft.axis)}
          value={draft.raw}
          onChange={(e) => onChange({ ...draft, raw: e.target.value })}
        />
      )}

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
