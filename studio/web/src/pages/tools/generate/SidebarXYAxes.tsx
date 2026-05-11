import { useEffect, useMemo, useState } from 'react'
import { api, type LoraCkpt, type LoraEntry, type XYAxisType } from '../../../api/client'
import PathPicker from '../../../components/PathPicker'
import InlineLoraPicker from './InlineLoraPicker'
import NumberListInput from './NumberListInput'
import type { ProjectLora } from './types'
import { AXIS_LABELS, AXIS_VALUE_TYPE, REQUIRES_LORA_INDEX, ckptStemFromPath, type XYAxisDraft } from './xy'

const ALL_AXES: XYAxisType[] = ['lora_ckpt', 'lora_scale', 'cfg_scale', 'steps']

function placeholderFor(axis: XYAxisType): string {
  const t = AXIS_VALUE_TYPE[axis]
  if (t === 'int') return '20, 25, 30'
  return '0.6, 0.8, 1.0'
}

function parsePathList(raw: string): string[] {
  return raw.split(',').map((s) => s.trim()).filter(Boolean)
}

/** axis=lora_ckpt 时绑定的 LoRA 在该 axis 内显示 + ckpt 多选区。 */
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
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let cancelled = false
    setLoaded(false)
    setError(null)
    void api.listVersionLoraCkpts(projectId, versionId)
      .then((items) => { if (!cancelled) { setCkpts(items); setLoaded(true) } })
      .catch((e) => {
        if (cancelled) return
        const msg = e instanceof Error ? e.message : String(e)
        console.error('listVersionLoraCkpts failed', { projectId, versionId, error: msg })
        setCkpts([])
        setError(msg)
        setLoaded(true)
      })
    return () => { cancelled = true }
  }, [projectId, versionId])

  const selected = new Set(parsePathList(raw))
  const toggle = (p: string) => {
    const next = new Set(selected)
    if (next.has(p)) next.delete(p); else next.add(p)
    // 按 ckpts 列表顺序输出（server 已 final → step desc → epoch desc → 自然序）
    // 而非 set 插入序，让 XY 网格列顺序与可选列表一致，避免点击次序污染。
    const ordered = ckpts.map((c) => c.path).filter((path) => next.has(path))
    onChange(ordered.join(', '))
  }

  if (!loaded) {
    return (
      <div className="text-2xs text-fg-tertiary font-mono">
        加载 ckpt…（{projectId}/{versionId}）
      </div>
    )
  }
  if (error) {
    return <div className="text-2xs text-err font-mono">加载 ckpt 失败：{error}</div>
  }
  if (ckpts.length === 0) {
    return <div className="text-2xs text-fg-tertiary">该 LoRA 没扫到 ckpt 文件（output/ 下需 *.safetensors）</div>
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
              fontSize: 11, padding: '3px 9px', borderRadius: 999,
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

/** Axis 内嵌 LoRA picker：选完一个 LoRA 后加进 loras + 设 axis.lora_index。 */
function AxisLoraPicker({
  loras, onLorasChange, projectLoras,
  loraIndex, onLoraIndexChange,
}: {
  loras: LoraEntry[]
  onLorasChange: (l: LoraEntry[]) => void
  projectLoras: ProjectLora[]
  loraIndex: number | null
  onLoraIndexChange: (i: number | null) => void
}) {
  const [pickerOpen, setPickerOpen] = useState(false)
  const [externalOpen, setExternalOpen] = useState(false)
  const selectedPaths = useMemo(() => new Set(loras.map((l) => l.path)), [loras])

  const addAndBind = (path: string, projectId: number | null, versionId: number | null) => {
    const idx = loras.findIndex((l) => l.path === path)
    if (idx >= 0) {
      onLoraIndexChange(idx)
    } else {
      onLorasChange([
        ...loras,
        { path, scale: 1.0, project_id: projectId, version_id: versionId },
      ])
      onLoraIndexChange(loras.length)
    }
    setPickerOpen(false)
    setExternalOpen(false)
  }

  // 已绑 LoRA：show summary + 「换 LoRA」按钮
  const bound = loraIndex !== null && loraIndex < loras.length ? loras[loraIndex] : null
  if (bound && !pickerOpen) {
    const matched = projectLoras.find((p) => p.versionId === bound.version_id)
    const label = matched
      ? `${matched.projectTitle} / ${matched.versionLabel}`
      : ckptStemFromPath(bound.path)
    return (
      <div
        className="flex items-center gap-2 text-xs"
        style={{
          padding: '6px 10px',
          borderRadius: 'var(--r-md)',
          border: '1px solid var(--border-subtle)',
          background: 'var(--bg-elevated)',
        }}
      >
        <span className="text-fg-tertiary shrink-0">绑定:</span>
        <span className="font-medium truncate flex-1" title={bound.path}>{label}</span>
        <button
          type="button"
          onClick={() => setPickerOpen(true)}
          className="btn btn-ghost btn-sm font-mono text-2xs shrink-0"
          style={{ padding: '2px 8px' }}
        >
          换 LoRA
        </button>
      </div>
    )
  }

  return (
    <>
      {!pickerOpen ? (
        <button
          type="button"
          onClick={() => setPickerOpen(true)}
          className="font-mono inline-flex items-center gap-1.5 self-start"
          style={{
            border: '1px solid var(--border-subtle)',
            background: 'var(--bg-sunken)',
            borderRadius: 'var(--r-md)',
            padding: '6px 10px',
            fontSize: 12,
            color: 'var(--fg-tertiary)',
            cursor: 'pointer',
          }}
        >
          + 添加 LoRA
        </button>
      ) : (
        <InlineLoraPicker
          projectLoras={projectLoras}
          selectedPaths={selectedPaths}
          onPick={(path) => {
            const matched = projectLoras.find((p) => p.path === path)
            addAndBind(path, matched?.projectId ?? null, matched?.versionId ?? null)
          }}
          onClose={() => setPickerOpen(false)}
          onPickExternal={() => setExternalOpen(true)}
        />
      )}
      {externalOpen && (
        <PathPicker
          dirOnly={false}
          onPick={(p) => addAndBind(p, null, null)}
          onClose={() => setExternalOpen(false)}
        />
      )}
    </>
  )
}

function AxisCard({
  label, draft, onChange, onRemove, loras, onLorasChange, projectLoras,
}: {
  label: 'X' | 'Y'
  draft: XYAxisDraft
  onChange: (d: XYAxisDraft) => void
  onRemove?: () => void
  loras: LoraEntry[]
  onLorasChange: (l: LoraEntry[]) => void
  projectLoras: ProjectLora[]
}) {
  const needsLora = REQUIRES_LORA_INDEX.has(draft.axis)
  const isCkpt = draft.axis === 'lora_ckpt'
  const bound =
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

      {/* 需绑 LoRA 的轴：内嵌 LoRA picker 选 / 切 */}
      {needsLora && (
        <AxisLoraPicker
          loras={loras}
          onLorasChange={onLorasChange}
          projectLoras={projectLoras}
          loraIndex={draft.loraIndex}
          onLoraIndexChange={(i) => onChange({ ...draft, loraIndex: i })}
        />
      )}

      {/* axis=lora_ckpt：ckpt 多选 chip；axis=lora_scale：+ 添加 数字 chip；
          其他数值轴（steps/cfg）：text input（范围广用逗号分隔够直观）。 */}
      {isCkpt ? (
        bound && bound.version_id && bound.project_id ? (
          <CkptMultiPicker
            projectId={bound.project_id}
            versionId={bound.version_id}
            raw={draft.raw}
            onChange={(raw) => onChange({ ...draft, raw })}
          />
        ) : bound ? (
          <div className="text-2xs text-fg-tertiary">外部文件 LoRA 没法列 ckpt（要 picker 选项目里的）</div>
        ) : null
      ) : draft.axis === 'lora_scale' ? (
        <NumberListInput
          raw={draft.raw}
          onChange={(raw) => onChange({ ...draft, raw })}
          min={0}
          max={1}
          step={0.05}
          placeholder="0.85"
        />
      ) : (
        <input
          type="text"
          className="input font-mono text-xs"
          placeholder={placeholderFor(draft.axis)}
          value={draft.raw}
          onChange={(e) => onChange({ ...draft, raw: e.target.value })}
        />
      )}
    </div>
  )
}

/** Sidebar 的 XY 轴配置区（仅 mode=xy 时渲染）。
 *
 * mode=xy 下整页 LoRA 选择都收纳在这里，无独立 LoRA 卡片：
 *   - axis 选项：LoRA / 权重 / CFG / 步数（前两个最常用）
 *   - axis=LoRA / 权重 时，轴卡片内嵌 LoRA picker（+ 添加 LoRA）
 *   - 添加完会 mutate Generate.tsx 的 loras state（共用 lora_configs）
 */
export default function SidebarXYAxes({
  xDraft, yDraft, onXChange, onYChange,
  loras, onLorasChange, projectLoras,
}: {
  xDraft: XYAxisDraft
  yDraft: XYAxisDraft | null
  onXChange: (d: XYAxisDraft) => void
  onYChange: (d: XYAxisDraft | null) => void
  loras: LoraEntry[]
  onLorasChange: (l: LoraEntry[]) => void
  projectLoras: ProjectLora[]
}) {
  return (
    <div className="card" style={{ padding: 18 }}>
      <div className="flex items-center justify-between mb-3">
        <div className="text-md font-semibold">XY 轴</div>
      </div>
      <div className="flex flex-col gap-2">
        <AxisCard
          label="X" draft={xDraft} onChange={onXChange}
          loras={loras} onLorasChange={onLorasChange} projectLoras={projectLoras}
        />
        {yDraft ? (
          <AxisCard
            label="Y" draft={yDraft}
            onChange={(d) => onYChange(d)}
            onRemove={() => onYChange(null)}
            loras={loras} onLorasChange={onLorasChange} projectLoras={projectLoras}
          />
        ) : (
          <button
            onClick={() => onYChange({
              axis: 'lora_scale',
              raw: '1',  // 默认就一个值，用户自己 + 添加；不再带 3,4,5 误导
              loraIndex: loras.length > 0 ? 0 : null,
            })}
            className="btn btn-ghost btn-sm self-start text-xs text-fg-tertiary"
          >
            + 添加 Y 轴
          </button>
        )}
      </div>
    </div>
  )
}
