import { useMemo, useState } from 'react'
import type { LoraEntry } from '../../../api/client'
import PathPicker from '../../../components/PathPicker'
import InlineLoraPicker from './InlineLoraPicker'
import LoraCard from './LoraCard'
import type { ProjectLora } from './types'

/** Sidebar 的 LoRA 区：已添加卡片列表 + 「+ 选 LoRA」inline 展开 picker。
 *
 * 替换原 LoraList（text input + 浏览图标 + 「最近」浮层）。 */
export default function SidebarLoras({
  loras, onChange, projectLoras,
}: {
  loras: LoraEntry[]
  onChange: (l: LoraEntry[]) => void
  projectLoras: ProjectLora[]
}) {
  const [pickerOpen, setPickerOpen] = useState(false)
  const [pathPickerOpen, setPathPickerOpen] = useState(false)

  // path → "项目 / 版本" label 反查（picker 选过的有；外部文件用 basename 兜底）
  const labelOf = useMemo(() => {
    const map = new Map<string, string>()
    for (const l of projectLoras) {
      map.set(l.path, `${l.projectTitle} / ${l.versionLabel}`)
    }
    return (path: string): string => map.get(path) ?? ''
  }, [projectLoras])

  // path → stage 反查（训练中卡片要描边 accent）
  const stageOf = useMemo(() => {
    const map = new Map<string, string>()
    for (const l of projectLoras) map.set(l.path, l.stage)
    return (path: string): string | undefined => map.get(path)
  }, [projectLoras])

  const selectedPaths = useMemo(() => new Set(loras.map((l) => l.path)), [loras])

  const addLora = (path: string) => {
    if (!path || selectedPaths.has(path)) return
    onChange([...loras, { path, scale: 1.0 }])
  }
  const removeAt = (i: number) => onChange(loras.filter((_, idx) => idx !== i))
  const setScaleAt = (i: number, scale: number) =>
    onChange(loras.map((l, idx) => (idx === i ? { ...l, scale } : l)))

  return (
    <div className="flex flex-col gap-2">
      {loras.map((l, i) => (
        <LoraCard
          key={`${l.path}-${i}`}
          lora={l}
          label={labelOf(l.path)}
          stage={stageOf(l.path)}
          onScaleChange={(s) => setScaleAt(i, s)}
          onRemove={() => removeAt(i)}
        />
      ))}

      {!pickerOpen ? (
        <button
          onClick={() => setPickerOpen(true)}
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
          + 从项目添加 LoRA
        </button>
      ) : (
        <InlineLoraPicker
          projectLoras={projectLoras}
          selectedPaths={selectedPaths}
          onPick={(path) => addLora(path)}
          onClose={() => setPickerOpen(false)}
          onPickExternal={() => setPathPickerOpen(true)}
        />
      )}

      {pathPickerOpen && (
        <PathPicker
          dirOnly={false}
          onPick={(p) => {
            addLora(p)
            setPathPickerOpen(false)
          }}
          onClose={() => setPathPickerOpen(false)}
        />
      )}
    </div>
  )
}
