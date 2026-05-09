import { useMemo, useState } from 'react'
import type { ProjectLora } from './types'

/** 项目缩写图标（2 字符 uppercase，从 title 提字母数字派生）。 */
export function projectAbbr(title: string): string {
  const cleaned = title.replace(/[^a-zA-Z0-9]/g, '')
  return (cleaned.slice(0, 2) || '??').toUpperCase()
}

function ProjectIcon({ title }: { title: string }) {
  return (
    <div className="shrink-0 w-7 h-7 rounded bg-sunken text-fg-tertiary text-2xs font-mono flex items-center justify-center border border-subtle">
      {projectAbbr(title)}
    </div>
  )
}

/** Image 8 「PICKER B · 内嵌展开」—— "+ 选 LoRA" 就地展开成挑选区，关闭后折叠。
 *
 * 不是 modal —— 在 sidebar 内部 inline 展开；选中一个不自动关闭，方便连续选多个。
 * 「外部文件…」打开 PathPicker 走兜底（自由路径）。 */
export default function InlineLoraPicker({
  projectLoras, selectedPaths, onPick, onClose, onPickExternal,
}: {
  projectLoras: ProjectLora[]
  selectedPaths: Set<string>
  onPick: (path: string) => void
  onClose: () => void
  onPickExternal: () => void
}) {
  const [search, setSearch] = useState('')

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return projectLoras
    return projectLoras.filter(
      (l) =>
        l.projectTitle.toLowerCase().includes(q) ||
        l.versionLabel.toLowerCase().includes(q)
    )
  }, [projectLoras, search])

  const grouped = useMemo(() => {
    const map = new Map<number, { projectTitle: string; loras: ProjectLora[] }>()
    for (const l of filtered) {
      let g = map.get(l.projectId)
      if (!g) {
        g = { projectTitle: l.projectTitle, loras: [] }
        map.set(l.projectId, g)
      }
      g.loras.push(l)
    }
    return Array.from(map.values())
  }, [filtered])

  const projectCount = grouped.length
  const versionCount = filtered.length

  return (
    <div
      className="rounded-md border border-subtle bg-overlay p-2.5 flex flex-col gap-2"
      data-testid="inline-lora-picker"
    >
      {/* header: search + count + 外部文件 + close（image 2 design：外部文件靠右与搜索同行） */}
      <div className="flex items-center gap-2">
        <input
          type="text"
          className="input flex-1 text-xs"
          placeholder="搜索项目 / 版本…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          autoFocus
        />
        <span className="text-2xs text-fg-tertiary whitespace-nowrap">
          {projectCount} 项目 · {versionCount} 版本
        </span>
        <button
          onClick={onPickExternal}
          className="btn btn-ghost btn-sm text-2xs text-fg-tertiary"
          title="选系统中任意 .safetensors 文件"
        >
          外部文件
        </button>
        <button
          onClick={onClose}
          className="btn btn-ghost btn-sm text-fg-tertiary px-1.5"
          title="关闭"
          aria-label="关闭挑选区"
        >
          ×
        </button>
      </div>

      {/* list — 按 project 分组 */}
      <div className="flex flex-col gap-2 overflow-y-auto" style={{ maxHeight: 320 }}>
        {grouped.map((group) => (
          <div key={group.projectTitle} className="flex flex-col gap-px">
            <div className="caption text-2xs text-fg-tertiary px-1 uppercase tracking-wider">
              {group.projectTitle}
            </div>
            {group.loras.map((l) => {
              const added = selectedPaths.has(l.path)
              return (
                <button
                  key={l.versionId}
                  disabled={added}
                  onClick={() => onPick(l.path)}
                  className={`flex items-center gap-2 px-2 py-1.5 rounded text-xs text-left border-none transition-colors ${
                    added
                      ? 'bg-sunken text-fg-tertiary cursor-not-allowed'
                      : 'bg-transparent hover:bg-surface text-fg-secondary cursor-pointer'
                  }`}
                >
                  <ProjectIcon title={l.projectTitle} />
                  <div className="flex-1 min-w-0">
                    <div className="font-medium truncate flex items-center gap-1.5">
                      <span>{l.projectTitle} / {l.versionLabel}</span>
                      {l.stage === 'training' && (
                        <span className="badge badge-info" style={{ fontSize: 10 }}>训练中</span>
                      )}
                    </div>
                    <div className="text-2xs text-fg-tertiary truncate">
                      {l.stage} · {new Date(l.createdAt * 1000).toLocaleDateString()}
                    </div>
                  </div>
                  <span className="font-mono text-2xs shrink-0">
                    {added ? '已添加' : '+'}
                  </span>
                </button>
              )
            })}
          </div>
        ))}

        {!grouped.length && (
          <div className="text-fg-tertiary text-xs px-2 py-4 text-center">
            {search ? '没有匹配的 LoRA' : '还没有训练好的 LoRA —— 先去训练一个，或用「外部文件」'}
          </div>
        )}
      </div>

    </div>
  )
}
