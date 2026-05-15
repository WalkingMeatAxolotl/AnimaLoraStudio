import { useMemo, useState } from 'react'
import type { LoraEntry } from '../../../api/client'
import PathPicker from '../../../components/PathPicker'
import InlineLoraPicker, { type PickedLora } from './InlineLoraPicker'
import type { ProjectLora } from './types'

/** Sidebar 的 LoRA 区：每个 LoRA = 一个常驻 picker 槽（项目下拉 + ckpt chip + 权重 + ×）。
 *
 * 之前是「N 张 LoraCard 摘要 + 1 个折叠 picker」两层；现在统一为「N 个 picker
 * 实例」，新增 LoRA 就 push 一个空槽。删除走 picker 自己的 × 按钮。 */
export default function SidebarLoras({
  loras, onChange, projectLoras,
}: {
  loras: LoraEntry[]
  onChange: (l: LoraEntry[]) => void
  projectLoras: ProjectLora[]
}) {
  // 外部文件 picker 当前给哪个槽用（loras 索引；null = 没在选）
  const [externalForIdx, setExternalForIdx] = useState<number | null>(null)
  // 「新增空槽」按钮触发后的临时槽数：UI 上 render 多出 N 个空 picker，挑了 ckpt
  // 才落进 loras[]。否则空 picker 不会污染 loras 数组。
  const [emptySlots, setEmptySlots] = useState<number>(0)

  // 已选 path（用于在所有 picker 的 chip 列表上互相 disable 重复 path）
  const existingPaths = useMemo(() => new Set(loras.map((l) => l.path)), [loras])

  const handleSlotChange = (i: number, picked: PickedLora | null, weight: number) => {
    if (picked === null) {
      // picker 内单选模式不会再发 null（点同 chip = no-op）；这里保留兜底
      // 但不删槽 —— 只更新 weight，path 保持原样
      onChange(loras.map((l, idx) => (idx === i ? { ...l, scale: weight } : l)))
      return
    }
    const entry: LoraEntry = {
      path: picked.path,
      scale: weight,
      project_id: picked.projectId,
      version_id: picked.versionId,
    }
    onChange(loras.map((l, idx) => (idx === i ? entry : l)))
  }

  const handleSlotRemove = (i: number) => {
    onChange(loras.filter((_, idx) => idx !== i))
  }

  const handleEmptySlotPick = (slotIdx: number, picked: PickedLora | null, weight: number) => {
    if (picked === null) {
      // 空槽里被点了又取消 —— 啥都不变
      return
    }
    // 空槽确认选了 ckpt → 进 loras[]
    const entry: LoraEntry = {
      path: picked.path,
      scale: weight,
      project_id: picked.projectId,
      version_id: picked.versionId,
    }
    onChange([...loras, entry])
    // 占位的空槽数 -1（被「具象化」成 loras 里一条）
    setEmptySlots((n) => Math.max(0, n - 1))
    if (externalForIdx === loras.length + slotIdx) setExternalForIdx(null)
  }

  const handleEmptySlotRemove = (slotIdx: number) => {
    // 空槽 × 直接撤
    setEmptySlots((n) => Math.max(0, n - 1))
    if (externalForIdx === loras.length + slotIdx) setExternalForIdx(null)
  }

  return (
    <div className="flex flex-col gap-2">
      {/* 已具象的 LoRA 槽 */}
      {loras.map((l, i) => (
        <InlineLoraPicker
          // key 只用 index：避免 ckpt 切换 (path 变) 时整个 picker remount，
          // 否则会重新拉 ckpt 列表 + 内部 state 重置 → 视觉闪烁
          key={`lora-${i}`}
          mode="single"
          projectLoras={projectLoras}
          value={{ path: l.path, projectId: l.project_id ?? null, versionId: l.version_id ?? null }}
          weight={l.scale}
          onChange={(p, w) => handleSlotChange(i, p, w)}
          onClose={() => handleSlotRemove(i)}
          onPickExternal={() => setExternalForIdx(i)}
        />
      ))}

      {/* 临时空槽：用户按了「+ 添加 LoRA」但还没确认 ckpt 的占位 picker */}
      {Array.from({ length: emptySlots }, (_, k) => k).map((slotIdx) => (
        <InlineLoraPicker
          key={`empty-${slotIdx}`}
          mode="single"
          projectLoras={projectLoras}
          value={null}
          weight={1.0}
          onChange={(p, w) => handleEmptySlotPick(slotIdx, p, w)}
          onClose={() => handleEmptySlotRemove(slotIdx)}
          onPickExternal={() => setExternalForIdx(loras.length + slotIdx)}
        />
      ))}

      <button
        onClick={() => setEmptySlots((n) => n + 1)}
        className="font-mono inline-flex items-center gap-1.5 self-start"
        style={{
          border: '1px solid var(--border-subtle)',
          background: 'var(--bg-sunken)',
          borderRadius: 'var(--r-md)',
          padding: '6px 10px',
          fontSize: 12,
          color: 'var(--fg-tertiary)',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.color = 'var(--fg-primary)'
          e.currentTarget.style.borderColor = 'var(--border-default)'
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.color = 'var(--fg-tertiary)'
          e.currentTarget.style.borderColor = 'var(--border-subtle)'
        }}
      >
        + 添加 LoRA
      </button>

      {externalForIdx !== null && (
        <PathPicker
          dirOnly={false}
          onPick={(p) => {
            const entry: LoraEntry = {
              path: p,
              scale: 1.0,
              project_id: null,
              version_id: null,
            }
            // 已具象槽 → 覆盖；空槽 → push（emptySlots 也 -1）
            if (externalForIdx < loras.length) {
              onChange(loras.map((l, idx) => (idx === externalForIdx ? entry : l)))
            } else {
              onChange([...loras, entry])
              setEmptySlots((n) => Math.max(0, n - 1))
            }
            // 不在 existingPaths 里强查 —— 外部文件可以叠多次
            void existingPaths
            setExternalForIdx(null)
          }}
          onClose={() => setExternalForIdx(null)}
        />
      )}
    </div>
  )
}
