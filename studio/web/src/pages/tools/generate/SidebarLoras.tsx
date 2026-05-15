import { useMemo, useState } from 'react'
import type { LoraEntry } from '../../../api/client'
import PathPicker from '../../../components/PathPicker'
import InlineLoraPicker, { type PickedLora } from './InlineLoraPicker'
import type { ProjectLora } from './types'

/** Sidebar 的 LoRA 区：每个 LoRA = 一个常驻 picker 槽（项目下拉 + ckpt chip + 权重 + ×）。
 *
 * 数据模型统一：所有 picker 槽都用 loras[] 表示，path='' 表示「空槽」（用户
 * 点了 + 添加 LoRA 但还没挑 ckpt）。这样：
 *   - 「+ 添加 LoRA」push 一条空 entry → 新增一个 picker（key 稳定，不会闪）
 *   - 反选 (点已选 chip) = 槽 path 设回 ''，picker 自己仍渲染 → 跟初次打开一样
 *   - × = 真正把 entry 从数组里删掉
 * Generate.tsx handleGenerate 在送 backend 前会 `loras.filter((l) => l.path.trim())`
 * 过滤空槽，不影响 enqueue。 */
export default function SidebarLoras({
  loras, onChange, projectLoras,
}: {
  loras: LoraEntry[]
  onChange: (l: LoraEntry[]) => void
  projectLoras: ProjectLora[]
}) {
  const [externalForIdx, setExternalForIdx] = useState<number | null>(null)

  // 已选 path（互相 disable，避免重复添加）—— 排除空槽
  const existingPaths = useMemo(
    () => new Set(loras.filter((l) => l.path).map((l) => l.path)),
    [loras],
  )

  const handleSlotChange = (i: number, picked: PickedLora | null, weight: number) => {
    const entry: LoraEntry = picked
      ? {
          path: picked.path,
          scale: weight,
          project_id: picked.projectId,
          version_id: picked.versionId,
        }
      : {
          // 反选：槽保留但 path 清空，picker 仍渲染（视觉等同初次打开的空槽）
          path: '',
          scale: weight,
          project_id: null,
          version_id: null,
        }
    onChange(loras.map((l, idx) => (idx === i ? entry : l)))
  }

  const handleSlotRemove = (i: number) => {
    onChange(loras.filter((_, idx) => idx !== i))
    if (externalForIdx === i) setExternalForIdx(null)
  }

  const handleAddSlot = () => {
    onChange([
      ...loras,
      { path: '', scale: 1.0, project_id: null, version_id: null },
    ])
  }

  return (
    <div className="flex flex-col gap-2">
      {loras.map((l, i) => {
        const hasCkpt = !!l.path
        return (
          <InlineLoraPicker
            // key 只用 index：避免 ckpt 切换 / 反选时整个 picker remount
            key={`lora-${i}`}
            mode="single"
            projectLoras={projectLoras}
            value={
              hasCkpt
                ? { path: l.path, projectId: l.project_id ?? null, versionId: l.version_id ?? null }
                : null
            }
            weight={l.scale}
            onChange={(p, w) => handleSlotChange(i, p, w)}
            onClose={() => handleSlotRemove(i)}
            onPickExternal={() => setExternalForIdx(i)}
          />
        )
      })}

      <button
        onClick={handleAddSlot}
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
            // 覆盖目标槽的内容；existingPaths 已排除空 path，外部文件可叠加
            void existingPaths
            onChange(loras.map((l, idx) => (idx === externalForIdx ? entry : l)))
            setExternalForIdx(null)
          }}
          onClose={() => setExternalForIdx(null)}
        />
      )}
    </div>
  )
}
