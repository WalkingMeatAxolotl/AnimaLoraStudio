/** 右侧竖排出图时间线（0.17 P-I：从「只列 done 历史」升级成统一时间线）。
 *
 * item 两种：
 *  - live（pending / running，来自队列）：占位卡。pending 灰底「排队中」+ ✕；running
 *    脉冲边框「生成中」，点击回到实时视图。
 *  - done（来自 cache/disk 扫盘）：缩略图，点击回看（onSelect(entry)）。
 * live 恒在最上（最新提交的），done 按 createdAt 往下——天然一条时间线。
 *
 * - 按当前 mode 过滤（single / xy / compare）；live item 的 mode 由父组件按 runsRef 定。
 * - 定高窗口化虚拟滚动（item 同尺寸，uniform stride）：只渲染可见区间 + overscan，
 *   外层 total×stride 占位撑滚动条，item 绝对定位到 idx×stride。
 * - 顶部 [刷新] 重拉 disk-history（多 tab / 外部改盘后手动同步）。
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { Task } from '../../../api/client'
import { entryBadge, entryDisplayLabel, entryThumbUrl, type HistoryEntry } from './entryAdapter'

// item 84px 方形 + 6px 间距 = 90px stride（0.17：从 56 放大 1.5× —— 原来太小、✕ 易误触、
// 内部小字看不清）
const ITEM_SIZE = 84
const ITEM_STRIDE = 90
const OVERSCAN = 4
const VIEWPORT_FALLBACK = 2000

/** 统一时间线一项：进行中（队列 task）或已完成（历史 entry）。 */
export type TimelineItem =
  | { kind: 'live'; task: Task; mode: 'single' | 'xy' | 'compare' }
  | { kind: 'done'; entry: HistoryEntry }

function itemMode(it: TimelineItem): 'single' | 'xy' | 'compare' {
  return it.kind === 'live' ? it.mode : it.entry.mode
}
function itemKey(it: TimelineItem): string {
  return it.kind === 'live' ? `live:${it.task.id}` : it.entry.id
}

interface Props {
  items: TimelineItem[]
  mode: 'single' | 'xy' | 'compare'
  /** 点 done 项回看 / 点 running 项回到实时视图。pending 项不触发（只可取消）。 */
  onSelect: (it: TimelineItem) => void
  /** 取消某条 pending/running 的 generate。 */
  onCancel: (taskId: number) => void
  onRefresh?: () => Promise<void>
  loading?: boolean
}

export default function PreviewHistoryRail({
  items, mode, onSelect, onCancel, onRefresh, loading,
}: Props) {
  const { t } = useTranslation()
  const list = useMemo(() => items.filter((it) => itemMode(it) === mode), [items, mode])

  const scrollRef = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewportH, setViewportH] = useState(VIEWPORT_FALLBACK)

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const update = () => setViewportH(el.clientHeight || VIEWPORT_FALLBACK)
    update()
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
      style={{ width: 110, padding: 8 }}
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
          <div style={{ height: total * ITEM_STRIDE, position: 'relative' }}>
            {slice.map((it, i) => {
              const idx = start + i
              return (
                <div
                  key={itemKey(it)}
                  style={{
                    position: 'absolute',
                    top: idx * ITEM_STRIDE,
                    left: 0,
                    right: 0,
                  }}
                >
                  {it.kind === 'done' ? (
                    <HistoryItem entry={it.entry} onSelect={() => onSelect(it)} />
                  ) : (
                    <LiveItem task={it.task} onSelect={() => onSelect(it)} onCancel={() => onCancel(it.task.id)} />
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

function HistoryItem({ entry, onSelect }: { entry: HistoryEntry; onSelect: () => void }) {
  const badge = entryBadge(entry)
  return (
    <div
      className="relative rounded-sm border border-subtle hover:border-strong cursor-pointer overflow-hidden"
      style={{ width: '100%', height: ITEM_SIZE, flexShrink: 0 }}
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
        <span className="absolute bottom-0 right-0 bg-canvas/80 text-fg-primary text-[10px] px-1 rounded-tl">
          {badge}
        </span>
      )}
    </div>
  )
}

/** 进行中项：pending 灰占位 + ✕；running 脉冲边框，点击回到实时视图。 */
function LiveItem({ task, onSelect, onCancel }: { task: Task; onSelect: () => void; onCancel: () => void }) {
  const { t } = useTranslation()
  const running = task.status === 'running'
  return (
    <div
      className={`relative rounded-sm border overflow-hidden flex items-center justify-center ${running ? 'border-accent cursor-pointer' : 'border-subtle border-dashed'}`}
      style={{ width: '100%', height: ITEM_SIZE, flexShrink: 0, background: 'var(--bg-overlay)' }}
      onClick={running ? onSelect : undefined}
      title={`#${task.id} · ${running ? t('status.running') : t('status.queued')}`}
    >
      {running
        ? <span className="dot dot-running" style={{ transform: 'scale(1.4)' }} />
        : <span className="text-fg-secondary text-xs text-center leading-tight px-1">{t('status.queued')}</span>}
      {/* ✕ 放大 + 内缩 + 圆底，做成明确目标，避免和点选卡片本体误触。 */}
      <button
        onClick={(e) => { e.stopPropagation(); onCancel() }}
        className="absolute top-1 right-1 text-err bg-canvas/90 hover:bg-canvas rounded text-sm leading-none px-1.5 py-0.5 cursor-pointer border-0 font-bold"
        title={t('common.cancel')}
        aria-label={t('common.cancel')}
        data-testid={`timeline-cancel-${task.id}`}
      >✕</button>
      <span className="absolute bottom-0 left-0 bg-canvas/80 text-fg-tertiary text-[11px] px-1 rounded-tr">
        #{task.id}
      </span>
    </div>
  )
}
