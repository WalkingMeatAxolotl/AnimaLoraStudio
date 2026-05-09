/**
 * commit 16：右侧竖排图片历史栏（design image 1）。
 *
 * - 按当前 mode 过滤显示（single/xy/compare 各一桶）
 * - 64-72px 宽，垂直堆叠缩略图，溢出滚动
 * - XY/对比 entry 右下角带 badge（"XY 5×5" / "2×"）
 * - 点击 → onSelect(entry) 给父组件，由父组件决定如何"回看"
 *   （单图：拉原图覆盖主预览；XY/对比：弹封面缩略图 modal）
 */
import type { HistoryEntry, HistoryMode } from './useGenerateHistory'

interface Props {
  entries: HistoryEntry[]
  mode: HistoryMode
  onSelect: (entry: HistoryEntry) => void
  onRemove?: (id: string) => void
  onClear?: () => void
}

export default function PreviewHistoryRail({
  entries, mode, onSelect, onRemove, onClear,
}: Props) {
  const list = entries.filter((e) => e.mode === mode)
  if (list.length === 0) {
    return (
      <div
        className="text-fg-tertiary text-2xs text-center"
        style={{ width: 64, paddingTop: 16 }}
      >
        暂无历史
      </div>
    )
  }

  return (
    <div
      className="flex flex-col gap-1"
      style={{ width: 64, maxHeight: 720, overflowY: 'auto' }}
    >
      {onClear && (
        <button
          className="btn btn-ghost text-2xs"
          style={{ padding: '1px 4px' }}
          onClick={onClear}
          title={`清空当前 ${mode} 历史`}
        >
          清空
        </button>
      )}
      {list.map((entry) => (
        <HistoryItem
          key={entry.id}
          entry={entry}
          onSelect={() => onSelect(entry)}
          onRemove={onRemove ? () => onRemove(entry.id) : undefined}
        />
      ))}
    </div>
  )
}

interface ItemProps {
  entry: HistoryEntry
  onSelect: () => void
  onRemove?: () => void
}

function HistoryItem({ entry, onSelect, onRemove }: ItemProps) {
  return (
    <div
      className="relative rounded-sm border border-subtle hover:border-strong cursor-pointer overflow-hidden"
      style={{ width: 56, height: 56 }}
      onClick={onSelect}
      title={`#${entry.taskId} · ${new Date(entry.createdAt).toLocaleString()}`}
    >
      <img
        src={entry.thumbnailDataUrl}
        alt={`#${entry.taskId}`}
        style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
      />
      {entry.badge && (
        <span
          className="absolute bottom-0 right-0 bg-canvas/80 text-fg-primary text-[9px] px-1 rounded-tl"
        >
          {entry.badge}
        </span>
      )}
      {onRemove && (
        <button
          className="absolute top-0 right-0 bg-canvas/80 text-fg-tertiary hover:text-err text-xs leading-none"
          style={{ padding: '1px 4px', lineHeight: 1 }}
          onClick={(e) => {
            e.stopPropagation()
            onRemove()
          }}
          title="删除"
        >
          ×
        </button>
      )}
    </div>
  )
}
