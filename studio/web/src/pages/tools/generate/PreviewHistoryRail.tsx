/**
 * commit 16：右侧竖排图片历史栏（design image 1）。
 *
 * - 按当前 mode 过滤显示（single/xy/compare 各一桶）
 * - 64-72px 宽，垂直堆叠缩略图，溢出滚动
 * - XY/对比 entry 右下角带 badge（"XY 5×5" / "2×"）
 * - 点击 → onSelect(entry) 给父组件，由父组件决定如何"回看"
 *   （单图：拉原图覆盖主预览；XY/对比：弹封面缩略图 modal）
 */
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { HistoryEntry, HistoryMode } from './useGenerateHistory'

interface Props {
  entries: HistoryEntry[]
  mode: HistoryMode
  onSelect: (entry: HistoryEntry) => void
  onClear?: () => void
  /** 清理失效（server cache 已没的 entry） */
  onPruneStale?: () => Promise<number>
}

export default function PreviewHistoryRail({
  entries, mode, onSelect, onClear, onPruneStale,
}: Props) {
  const { t } = useTranslation()
  const list = entries.filter((e) => e.mode === mode)
  const [pruning, setPruning] = useState(false)
  const [pruneResult, setPruneResult] = useState<string | null>(null)

  const handlePrune = async () => {
    if (!onPruneStale || pruning) return
    setPruning(true)
    setPruneResult(null)
    try {
      const n = await onPruneStale()
      setPruneResult(n > 0 ? t('generate.prunedCount', { n }) : t('generate.historyAllAlive'))
      setTimeout(() => setPruneResult(null), 3000)
    } finally {
      setPruning(false)
    }
  }

  return (
    <div
      className="card flex flex-col gap-1 self-stretch"
      style={{ width: 80, padding: 8, overflowY: 'auto' }}
    >
      {list.length > 0 && onClear && (
        <button
          className="btn btn-ghost text-2xs"
          style={{ padding: '1px 4px' }}
          onClick={onClear}
          title={t('generate.clearCurrentHistoryTitle', { mode })}
        >
          {t('common.delete')}
        </button>
      )}
      {list.length > 0 && onPruneStale && (
        <button
          className="btn btn-ghost text-2xs"
          style={{ padding: '1px 4px' }}
          onClick={() => void handlePrune()}
          disabled={pruning}
          title={t('generate.pruneStaleTitle')}
        >
          {pruning ? t('generate.checkingShort') : pruneResult ?? t('generate.pruneStale')}
        </button>
      )}
      {list.length === 0 ? (
        <div className="text-fg-tertiary text-2xs text-center pt-3">{t('generate.noHistory')}</div>
      ) : (
        list.map((entry) => (
          <HistoryItem
            key={entry.id}
            entry={entry}
            onSelect={() => onSelect(entry)}
          />
        ))
      )}
    </div>
  )
}

interface ItemProps {
  entry: HistoryEntry
  onSelect: () => void
}

function HistoryItem({ entry, onSelect }: ItemProps) {
  return (
    <div
      className="relative rounded-sm border border-subtle hover:border-strong cursor-pointer overflow-hidden"
      // flexShrink:0：父容器是 flex-col，默认 shrink=1 会在历史满时把高度压扁（56xN，N<56），
      //   破坏 1:1 长宽比 + 让 overflowY:auto 失效（永远没溢出）。
      style={{ width: 56, height: 56, flexShrink: 0 }}
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
    </div>
  )
}
