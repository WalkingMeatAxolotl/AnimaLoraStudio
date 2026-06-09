/** 右侧竖排图片历史栏。Step 4 重写：用 entryAdapter helper 不直接 switch source。
 *
 * - 按当前 mode 过滤显示（single / xy / compare）
 * - 64-72px 宽，垂直堆叠缩略图，溢出滚动
 * - DiskEntry 用 server thumb URL（带 ETag + HTTP cache）；CacheEntry 直接
 *   用 imageUrl + CSS 自缩（session 期间不多）
 * - XY entry 右下角 badge ("XY 5×3"); single 无
 * - 点击 → onSelect(entry) 给父组件
 * - 顶部 [刷新] 按钮 → 调 refresh() 重拉 disk-history（多 tab 同步 / 外部改
 *   studio_data 后用户主动同步）
 */
import { useTranslation } from 'react-i18next'
import { entryBadge, entryThumbUrl, type HistoryEntry } from './entryAdapter'

interface Props {
  entries: HistoryEntry[]
  mode: 'single' | 'xy' | 'compare'
  onSelect: (entry: HistoryEntry) => void
  onRefresh?: () => Promise<void>
  loading?: boolean
}

export default function PreviewHistoryRail({
  entries, mode, onSelect, onRefresh, loading,
}: Props) {
  const { t } = useTranslation()
  const list = entries.filter((e) => e.mode === mode)

  return (
    <div
      className="card flex flex-col gap-1 self-stretch"
      style={{ width: 80, padding: 8, overflowY: 'auto' }}
    >
      {onRefresh && (
        <button
          className="btn btn-ghost text-2xs"
          style={{ padding: '1px 4px' }}
          onClick={() => void onRefresh()}
          disabled={loading}
          title={t('generate.refreshHistoryTitle')}
        >
          {loading ? t('generate.checkingShort') : t('generate.refreshHistory')}
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
  const badge = entryBadge(entry)
  return (
    <div
      className="relative rounded-sm border border-subtle hover:border-strong cursor-pointer overflow-hidden"
      style={{ width: 56, height: 56, flexShrink: 0 }}
      onClick={onSelect}
      title={`${entry.source === 'disk' ? entry.filename : `#${entry.taskId}`} · ${new Date(entry.createdAt).toLocaleString()}`}
    >
      <img
        src={entryThumbUrl(entry)}
        alt=""
        style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
        loading="lazy"
      />
      {badge && (
        <span
          className="absolute bottom-0 right-0 bg-canvas/80 text-fg-primary text-[9px] px-1 rounded-tl"
        >
          {badge}
        </span>
      )}
    </div>
  )
}
