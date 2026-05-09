import { useMemo, useState } from 'react'
import type { ProjectLora } from './types'
import { ckptStemFromPath } from './xy'

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

/** 内嵌 LoRA 多选挑选器：扁平列表 + 搜索 + 一键清空 + 短名显示。
 *
 * 列表每行：[项目 icon] [项目 / 版本] [训练中 pill] [短名] [✓ / +]
 * 不再按 project 分组（用户反馈分组冗余）；显示 stem 名（去 .safetensors）。
 *
 * onRemove 可选：传了就支持点击已添加项取消勾选（多选 toggle）。
 * onClearAll 可选：传了就显示一键清空按钮。
 */
export default function InlineLoraPicker({
  projectLoras, selectedPaths,
  onPick, onRemove, onClearAll,
  onClose, onPickExternal,
}: {
  projectLoras: ProjectLora[]
  selectedPaths: Set<string>
  onPick: (path: string) => void
  onRemove?: (path: string) => void
  onClearAll?: () => void
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
        l.versionLabel.toLowerCase().includes(q) ||
        ckptStemFromPath(l.path).toLowerCase().includes(q)
    )
  }, [projectLoras, search])

  const selectedCount = selectedPaths.size

  return (
    <div
      className="rounded-md border border-subtle bg-overlay p-2.5 flex flex-col gap-2"
      data-testid="inline-lora-picker"
    >
      {/* header: 搜索 + 计数 + 一键清空 + 外部文件 + 关闭 */}
      <div className="flex items-center gap-2">
        <input
          type="text"
          className="input flex-1 text-xs"
          placeholder="搜索项目 / 版本 / 文件名…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          autoFocus
        />
        <span className="text-2xs text-fg-tertiary whitespace-nowrap">
          已选 {selectedCount} / {filtered.length}
        </span>
        {onClearAll && selectedCount > 0 && (
          <button
            onClick={onClearAll}
            className="btn btn-ghost btn-sm text-2xs text-fg-tertiary"
            title="清空所有已选 LoRA"
          >
            一键清空
          </button>
        )}
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

      {/* 列表：扁平，每行一个 LoRA */}
      <div className="flex flex-col gap-px overflow-y-auto" style={{ maxHeight: 360 }}>
        {filtered.map((l) => {
          const added = selectedPaths.has(l.path)
          const stem = ckptStemFromPath(l.path)
          const handleClick = () => {
            if (added) {
              if (onRemove) onRemove(l.path)
            } else {
              onPick(l.path)
            }
          }
          return (
            <button
              key={`${l.projectId}-${l.versionId}`}
              onClick={handleClick}
              disabled={added && !onRemove}
              className={`flex items-center gap-2 px-2 py-1.5 rounded text-xs text-left border-none transition-colors ${
                added
                  ? (onRemove
                      ? 'bg-accent-soft text-accent cursor-pointer hover:opacity-80'
                      : 'bg-sunken text-fg-tertiary cursor-not-allowed')
                  : 'bg-transparent hover:bg-surface text-fg-secondary cursor-pointer'
              }`}
              style={{
                background: added
                  ? 'var(--accent-soft)'
                  : undefined,
              }}
            >
              <ProjectIcon title={l.projectTitle} />
              <div className="flex-1 min-w-0 flex flex-col gap-px">
                <div className="font-medium truncate flex items-center gap-1.5">
                  <span>{l.projectTitle} / {l.versionLabel}</span>
                  {l.stage === 'training' && (
                    <span className="badge badge-info" style={{ fontSize: 10 }}>训练中</span>
                  )}
                </div>
                <div className="text-2xs text-fg-tertiary font-mono truncate" title={l.path}>
                  {stem}
                </div>
              </div>
              <span className="font-mono text-2xs shrink-0" style={{ minWidth: 16, textAlign: 'right' }}>
                {added ? '✓' : '+'}
              </span>
            </button>
          )
        })}

        {filtered.length === 0 && (
          <div className="text-fg-tertiary text-xs px-2 py-4 text-center">
            {search ? '没有匹配的 LoRA' : '还没有训练好的 LoRA —— 先去训练一个，或用「外部文件」'}
          </div>
        )}
      </div>
    </div>
  )
}
