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
 *
 * 虚拟滚动（#2）：图多了（落盘历史累积成百上千张）整列全量渲染会卡 —— 一次性
 * 挂出所有 thumbnail 节点 + 触发全部缩略图请求。改成定高窗口化：只渲染滚动可见
 * 区间（+overscan）的项，外层用 total×stride 的占位高度撑出正确滚动条。item 用
 * 绝对定位放到 idx×stride，省去 flex gap 的累计误差。
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { entryBadge, entryDisplayLabel, entryThumbUrl, type HistoryEntry } from './entryAdapter'

// item 56px 方形 + 4px 间距 = 60px stride（与旧 gap-1 视觉一致）
const ITEM_SIZE = 56
const ITEM_STRIDE = 60
// 上下各多渲染几行，滚动时不露白
const OVERSCAN = 4
// mount 前 / jsdom 无布局（clientHeight=0）时的视口兜底高度：先从宽渲染，
// ResizeObserver 测到真实高度后收敛（仿 PreviewXYGrid 的 maxW 兜底）。
const VIEWPORT_FALLBACK = 2000

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
  const list = useMemo(() => entries.filter((e) => e.mode === mode), [entries, mode])

  const scrollRef = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewportH, setViewportH] = useState(VIEWPORT_FALLBACK)

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const update = () => setViewportH(el.clientHeight || VIEWPORT_FALLBACK)
    update()
    // jsdom 没有 ResizeObserver；测试 env 降级为 window resize 监听
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(update)
      ro.observe(el)
      return () => ro.disconnect()
    }
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [])

  const total = list.length
  const start = Math.max(0, Math.floor(scrollTop / ITEM_STRIDE) - OVERSCAN)
  const end = Math.min(total, Math.ceil((scrollTop + viewportH) / ITEM_STRIDE) + OVERSCAN)
  const slice = list.slice(start, end)

  return (
    <div
      className="card flex flex-col gap-1 self-stretch"
      style={{ width: 80, padding: 8 }}
    >
      {onRefresh && (
        <button
          className="btn btn-ghost text-2xs shrink-0"
          style={{ padding: '1px 4px' }}
          onClick={() => void onRefresh()}
          disabled={loading}
          title={t('generate.refreshHistoryTitle')}
        >
          {loading ? t('generate.checkingShort') : t('generate.refreshHistory')}
        </button>
      )}
      {total === 0 ? (
        <div className="text-fg-tertiary text-2xs text-center pt-3">{t('generate.noHistory')}</div>
      ) : (
        <div
          ref={scrollRef}
          className="flex-1 min-h-0 overflow-y-auto"
          onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
        >
          {/* 占位高度 = total×stride 撑出真实滚动条；可见项绝对定位到各自 idx */}
          <div style={{ height: total * ITEM_STRIDE, position: 'relative' }}>
            {slice.map((entry, i) => {
              const idx = start + i
              return (
                <div
                  key={entry.id}
                  style={{
                    position: 'absolute',
                    top: idx * ITEM_STRIDE,
                    left: 0,
                    right: 0,
                    display: 'flex',
                    justifyContent: 'center',
                  }}
                >
                  <HistoryItem entry={entry} onSelect={() => onSelect(entry)} />
                </div>
              )
            })}
          </div>
        </div>
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
      style={{ width: ITEM_SIZE, height: ITEM_SIZE, flexShrink: 0 }}
      onClick={onSelect}
      title={`${entryDisplayLabel(entry)} · ${new Date(entry.createdAt).toLocaleString()}`}
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
