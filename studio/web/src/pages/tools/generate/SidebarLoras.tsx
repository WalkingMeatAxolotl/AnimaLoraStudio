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

  // version_id → "项目 / 版本" label 反查（picker 选过的有；切 step ckpt 后
  // path 会变，所以不能用 path 反查）
  const labelOf = useMemo(() => {
    const map = new Map<number, string>()
    for (const l of projectLoras) {
      map.set(l.versionId, `${l.projectTitle} / ${l.versionLabel}`)
    }
    return (entry: LoraEntry): string =>
      entry.version_id != null ? (map.get(entry.version_id) ?? '') : ''
  }, [projectLoras])

  // version_id → stage 反查（训练中卡片要描边 accent）
  const stageOf = useMemo(() => {
    const map = new Map<number, string>()
    for (const l of projectLoras) map.set(l.versionId, l.stage)
    return (entry: LoraEntry): string | undefined =>
      entry.version_id != null ? map.get(entry.version_id) : undefined
  }, [projectLoras])

  const selectedPaths = useMemo(() => new Set(loras.map((l) => l.path)), [loras])

  const addLora = (path: string) => {
    if (!path || selectedPaths.has(path)) return
    // 找 picker 里这个 path 对应的 project/version，绑定到 LoraEntry
    const matched = projectLoras.find((l) => l.path === path)
    onChange([...loras, {
      path,
      scale: 1.0,
      project_id: matched?.projectId ?? null,
      version_id: matched?.versionId ?? null,
    }])
  }
  const removeAt = (i: number) => onChange(loras.filter((_, idx) => idx !== i))
  const replaceAt = (i: number, next: LoraEntry) =>
    onChange(loras.map((l, idx) => (idx === i ? next : l)))

  return (
    <div className="flex flex-col gap-2">
      {loras.map((l, i) => (
        <LoraCard
          key={`${l.version_id ?? 'ext'}-${i}`}
          lora={l}
          label={labelOf(l)}
          stage={stageOf(l)}
          onChange={(next) => replaceAt(i, next)}
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
          onRemove={(path) => onChange(loras.filter((l) => l.path !== path))}
          onClearAll={() => onChange([])}
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
