import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import {
  api,
  type DuplicateGroup,
  type DuplicateItem,
  type DuplicateScanOptions,
  type DuplicateScanResult,
} from '../api/client'

export const DEFAULT_DUPLICATE_OPTIONS: DuplicateScanOptions = {
  match_scope: 'both',
  hash_size: 768,
  hash_workers: 4,
  tile_grids: [4, 6],
  structure_threshold: 6,
  variant_score: 72,
  aspect_tolerance: 0.045,
  min_close_tiles: 0.48,
  tile_median: 14,
  min_gray_close: 0.42,
}

interface Props {
  projectId: number
  result: DuplicateScanResult | null
  selected: Set<string>
  busy: boolean
  onSelect: (next: Set<string>) => void
  onPreview: (name: string) => void
}

export default function DuplicateReviewPanel({
  projectId,
  result,
  selected,
  busy,
  onSelect,
  onPreview,
}: Props) {
  const { t } = useTranslation()
  const suggested = useMemo(
    () =>
      result
        ? result.groups.flatMap((group) =>
            group.items.filter((item) => !item.keep).map((item) => item.name)
          )
        : [],
    [result]
  )
  // 二态：selected = 去除（黄），unselected = 留存（绿）。每张图独立选择，
  // 无 "每组至少留一张" 校验，也无 "设为保留 → 反选其他" 联动。
  const toggleName = (name: string) => {
    const next = new Set(selected)
    if (next.has(name)) next.delete(name)
    else next.add(name)
    onSelect(next)
  }
  return (
    <section className="flex flex-col flex-1 min-h-0 rounded-md border border-subtle bg-surface overflow-hidden">
      <div className="h-0.5 bg-warn" />
      <header className="flex flex-wrap items-center gap-2 px-2.5 py-1.5 border-b border-subtle text-sm">
        <h3 className="font-semibold">{t('duplicates.reviewTitle')}</h3>
        <span className="text-xs text-fg-tertiary">
          {result
            ? t('duplicates.summary', {
                groups: result.group_count,
                candidates: result.candidate_count,
                total: result.total_images,
              })
            : t('duplicates.empty')}
        </span>
        <span className="flex-1" />
        <button
          type="button"
          onClick={() => onSelect(new Set(suggested))}
          disabled={busy || suggested.length === 0}
          className="btn btn-secondary btn-sm"
        >
          {t('duplicates.selectSuggested')}
        </button>
        <button
          type="button"
          onClick={() => onSelect(new Set())}
          disabled={busy || selected.size === 0}
          className="btn btn-secondary btn-sm"
        >
          {t('common.deselect')}
        </button>
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto p-2">
        {!result ? (
          <div className="min-h-[180px] h-full flex flex-col items-center justify-center text-center px-6 py-10">
            <div className="text-sm font-medium text-fg-secondary">{t('duplicates.emptyTitle')}</div>
            <p className="text-sm text-fg-tertiary mt-1 max-w-[52ch]">{t('duplicates.empty')}</p>
          </div>
        ) : result.groups.length === 0 ? (
          <div className="min-h-[180px] h-full flex items-center justify-center text-sm text-fg-tertiary px-6 text-center">
            {t('duplicates.noGroups', { total: result.total_images })}
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {result.groups.map((group) => (
              <DuplicateGroupCard
                key={group.group_id}
                projectId={projectId}
                group={group}
                selected={selected}
                busy={busy}
                onToggle={toggleName}
                onPreview={onPreview}
              />
            ))}
          </div>
        )}
      </div>
    </section>
  )
}

function DuplicateGroupCard({
  projectId,
  group,
  selected,
  busy,
  onToggle,
  onPreview,
}: {
  projectId: number
  group: DuplicateGroup
  selected: Set<string>
  busy: boolean
  onToggle: (name: string) => void
  onPreview: (name: string) => void
}) {
  const { t } = useTranslation()
  return (
    <article className="rounded-md border border-subtle bg-sunken p-2">
      <div className="flex flex-wrap items-center gap-2 mb-2 text-xs">
        <span className="badge badge-neutral">#{group.group_id}</span>
        <span className="badge badge-neutral">{t('duplicates.groupCandidates', { n: group.items.length })}</span>
        {group.best && (
          <span className="badge badge-warn">
            {group.best.match_type} · {group.best.score}
          </span>
        )}
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(136px,1fr))] gap-1.5">
        {group.items.map((item) => (
          <DuplicateItemCell
            key={item.name}
            projectId={projectId}
            item={item}
            selected={selected.has(item.name)}
            suggestedKeep={item.keep}
            busy={busy}
            onToggle={() => onToggle(item.name)}
            onPreview={() => onPreview(item.name)}
          />
        ))}
      </div>
    </article>
  )
}

function DuplicateItemCell({
  projectId,
  item,
  selected,
  suggestedKeep,
  busy,
  onToggle,
  onPreview,
}: {
  projectId: number
  item: DuplicateItem
  selected: boolean
  suggestedKeep: boolean
  busy: boolean
  onToggle: () => void
  onPreview: () => void
}) {
  const { t } = useTranslation()
  const metrics = item.metrics
  return (
    <div
      className={
        'group relative rounded-md border overflow-hidden bg-surface ' +
        (selected ? 'border-warn ring-2 ring-warn-soft' : 'border-ok ring-1 ring-ok-soft')
      }
    >
      <button type="button" onClick={onPreview} className="block w-full aspect-square bg-sunken" title={item.name}>
        <img
          src={api.projectThumbUrl(projectId, item.name, 'download', 256)}
          alt={item.name}
          loading="lazy"
          decoding="async"
          className="w-full h-full object-cover"
        />
      </button>
      <div className="p-1.5 flex flex-col gap-1 text-[11px]">
        <div className="flex items-center gap-1 min-w-0 flex-wrap">
          <button
            type="button"
            onClick={onToggle}
            disabled={busy}
            className={`shrink-0 px-1.5 py-0.5 rounded-sm border text-[11px] font-medium ${
              selected
                ? 'bg-warn text-white border-warn'
                : 'bg-ok-soft text-ok border-ok'
            } disabled:opacity-60 disabled:cursor-not-allowed`}
            aria-label={`${selected ? t('duplicates.restoreCandidate') : t('duplicates.removeCandidate')} ${item.name}`}
          >
            {selected ? t('duplicates.selectedRemove') : t('duplicates.keep')}
          </button>
          {suggestedKeep && (
            <span className="badge badge-neutral shrink-0">{t('duplicates.suggestedBadge')}</span>
          )}
          <code className="mono truncate min-w-0">{item.name}</code>
        </div>
        <div className="text-fg-tertiary">
          {item.width}x{item.height} · {item.filesize_kb}KB
        </div>
        {metrics && (
          <div className="text-fg-tertiary truncate" title={metrics.note}>
            {metrics.match_type} · {metrics.score}
          </div>
        )}
      </div>
    </div>
  )
}
