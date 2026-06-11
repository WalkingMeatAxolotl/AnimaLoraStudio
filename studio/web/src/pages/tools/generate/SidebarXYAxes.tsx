import { useMemo, useState } from 'react'
import type { LoraEntry, XYAxisType } from '../../../api/client'
import PathPicker from '../../../components/PathPicker'
import InlineLoraPicker from './InlineLoraPicker'
import NumberListInput from './NumberListInput'
import type { ProjectLora } from './types'
import { AXIS_VALUE_TYPE, REQUIRES_LORA_INDEX, axisLabel, ckptStemFromPath, type XYAxisDraft } from './xy'

const ALL_AXES: XYAxisType[] = ['lora_ckpt', 'lora_scale', 'cfg_scale', 'steps']

function placeholderFor(axis: XYAxisType): string {
  const t = AXIS_VALUE_TYPE[axis]
  if (t === 'int') return '20, 25, 30'
  return '0.6, 0.8, 1.0'
}

/** lora_ckpt 轴的内嵌 picker：多选 ckpt + 自动 push / 更新 loras[] + 设 axis.raw + axis.loraIndex。
 *
 * 流程：用户点 picker chip 多选 → 确认 → 取所有 picks 的 (pid, vid) —— 多选始
 * 终在同一 (pid, vid) 下（picker 切 pid/vid 会清 picked）；在 loras[] 找匹配的
 * (project_id, version_id)；没有就 push 一条（path 用 picks[0]，scale=1.0）；
 * axis.loraIndex 指向那个槽；axis.raw 是 picks.map(path).join(', ')。
 */
function AxisLoraCkptPicker({
  draft, onDraftChange, loras, onLorasChange, projectLoras,
}: {
  draft: XYAxisDraft
  onDraftChange: (d: XYAxisDraft) => void
  loras: LoraEntry[]
  onLorasChange: (l: LoraEntry[]) => void
  projectLoras: ProjectLora[]
}) {
  const [externalOpen, setExternalOpen] = useState(false)

  const commitPicks = (picks: { path: string; projectId: number | null; versionId: number | null }[]) => {
    if (picks.length === 0) {
      // live 模式下，picker 把所有 chip 反选 / 切换 pid/vid 都会送空集合过来 →
      // 清掉 axis 绑定。loras[] 里之前那条 entry 不动（picker 可能复用它）。
      onDraftChange({ ...draft, loraIndex: null, raw: '' })
      return
    }
    const pid = picks[0].projectId
    const vid = picks[0].versionId
    // 在 loras[] 里找已绑过这个 (pid, vid) 的槽
    let idx = loras.findIndex(
      (l) => l.project_id === pid && l.version_id === vid && pid !== null && vid !== null
    )
    let nextLoras = loras
    if (idx < 0) {
      // 没绑过 → push 一条作 anchor（path 用 picks[0]，cell 内 backend 会 mutate path）
      const newEntry: LoraEntry = {
        path: picks[0].path,
        scale: 1.0,
        project_id: pid,
        version_id: vid,
      }
      nextLoras = [...loras, newEntry]
      idx = loras.length
      onLorasChange(nextLoras)
    }
    onDraftChange({
      ...draft,
      loraIndex: idx,
      raw: picks.map((p) => p.path).join(', '),
    })
  }

  // 显示已绑的 LoRA 摘要（如果 raw 非空 + loraIndex 有效）
  const bound = draft.loraIndex !== null && draft.loraIndex < loras.length
    ? loras[draft.loraIndex]
    : null
  const matched = bound ? projectLoras.find((p) => p.versionId === bound.version_id) : null
  const pickedCount = draft.raw.trim() ? draft.raw.split(',').filter((s) => s.trim()).length : 0

  // 受控同步给 InlineLoraPicker：raw 字符串当 path/basename 列表喂过去 + 锚定
  // 到 bound LoRA 的 (pid, vid)，让历史回填 picker chip 高亮、basename → 全 path
  // 自动 upgrade。
  const selectedPaths = useMemo(
    () => draft.raw.split(',').map((s) => s.trim()).filter(Boolean),
    [draft.raw],
  )

  return (
    <div className="flex flex-col gap-1.5">
      {bound && pickedCount > 0 && (
        <div
          className="flex items-center gap-2 text-2xs"
          style={{
            padding: '4px 8px',
            borderRadius: 'var(--r-sm)',
            background: 'var(--bg-sunken)',
            color: 'var(--fg-tertiary)',
          }}
        >
          <span>扫:</span>
          <span className="font-medium" style={{ color: 'var(--fg-secondary)' }}>
            {matched ? `${matched.projectTitle} / ${matched.versionLabel}` : ckptStemFromPath(bound.path)}
          </span>
          <span className="font-mono">· {pickedCount} 个 ckpt</span>
        </div>
      )}
      <InlineLoraPicker
        mode="multi"
        live
        projectLoras={projectLoras}
        existingPaths={new Set()}
        showWeight={false}
        selectedPaths={selectedPaths}
        initialPid={bound?.project_id ?? null}
        initialVid={bound?.version_id ?? null}
        onPick={commitPicks}
        onClose={() => { /* 常驻在 axis card 里，nothing to close */ }}
        onPickExternal={() => setExternalOpen(true)}
      />
      {externalOpen && (
        <PathPicker
          dirOnly={false}
          onPick={(p) => {
            commitPicks([{ path: p, projectId: null, versionId: null }])
            setExternalOpen(false)
          }}
          onClose={() => setExternalOpen(false)}
        />
      )}
    </div>
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
  const isCkpt = draft.axis === 'lora_ckpt'

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
            <option key={a} value={a}>{axisLabel(a)}</option>
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

      {/* axis=lora_ckpt：多选 chip picker 直接产出 raw + loraIndex；
          axis=lora_scale：纯数字（chip 列表），不再绑 LoRA；
          axis=steps/cfg：文本输入。 */}
      {isCkpt ? (
        <AxisLoraCkptPicker
          draft={draft}
          onDraftChange={onChange}
          loras={loras}
          onLorasChange={onLorasChange}
          projectLoras={projectLoras}
        />
      ) : draft.axis === 'lora_scale' ? (
        <NumberListInput
          raw={draft.raw}
          onChange={(raw) => onChange({ ...draft, raw })}
          min={0}
          max={1.5}
          step={0.05}
          placeholder="0.6, 0.8, 1.0"
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
 * 4 个 axis 类型：
 *   - LoRA（lora_ckpt）：picker 多选 ckpt 作格点，cell 内 mutate path 重 inject
 *   - 权重（lora_scale）：纯数字轴，全局 cell 内覆盖所有 LoRA 的 multiplier
 *   - CFG / 步数：文本输入数字列表
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
              raw: '1',  // 一个值起步，用户自己 + 添加
              loraIndex: null,
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
