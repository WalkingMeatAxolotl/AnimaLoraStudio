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
  options: DuplicateScanOptions
  result: DuplicateScanResult | null
  selected: Set<string>
  busy: boolean
  onOptionsChange: (next: DuplicateScanOptions) => void
  onScan: () => void
  onSelect: (next: Set<string>) => void
  onMove: () => void
  onDelete: () => void
  onPreview: (name: string) => void
}

export default function DuplicateReviewPanel({
  projectId,
  options,
  result,
  selected,
  busy,
  onOptionsChange,
  onScan,
  onSelect,
  onMove,
  onDelete,
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
  const patch = <K extends keyof DuplicateScanOptions>(key: K, value: DuplicateScanOptions[K]) => {
    onOptionsChange({ ...options, [key]: value })
  }
  const toggleName = (name: string) => {
    const next = new Set(selected)
    if (next.has(name)) next.delete(name)
    else next.add(name)
    onSelect(next)
  }
  return (
    <section className="rounded-md border border-subtle bg-surface overflow-hidden shrink-0">
      <div className="h-0.5 bg-warn" />
      <header className="flex flex-wrap items-center gap-2 px-2.5 py-1.5 border-b border-subtle text-sm">
        <h3 className="font-semibold">{t('duplicates.title')}</h3>
        <span className="text-xs text-fg-tertiary">
          {result
            ? t('duplicates.summary', {
                groups: result.group_count,
                candidates: result.candidate_count,
                total: result.total_images,
              })
            : t('duplicates.subtitle')}
        </span>
        <span className="flex-1" />
        <button
          onClick={() => onSelect(new Set(suggested))}
          disabled={busy || suggested.length === 0}
          className="btn btn-secondary btn-sm"
        >
          {t('duplicates.selectSuggested')}
        </button>
        <button
          onClick={() => onSelect(new Set())}
          disabled={busy || selected.size === 0}
          className="btn btn-secondary btn-sm"
        >
          {t('common.deselect')}
        </button>
        <button onClick={onMove} disabled={busy || selected.size === 0} className="btn btn-secondary btn-sm">
          {t('duplicates.moveBtn', { n: selected.size })}
        </button>
        <button onClick={onDelete} disabled={busy || selected.size === 0} className="btn btn-sm bg-err-soft text-err border-err">
          {t('duplicates.deleteBtn', { n: selected.size })}
        </button>
      </header>

      <div className="grid grid-cols-1 2xl:grid-cols-[360px,1fr] gap-3 p-2">
        <div className="flex flex-col gap-2 text-sm">
          <label className="flex flex-col gap-1">
            <span className="text-xs text-fg-tertiary">{t('duplicates.scope')}</span>
            <select
              className="input px-2 py-1 text-sm"
              value={options.match_scope}
              onChange={(e) => patch('match_scope', e.target.value as DuplicateScanOptions['match_scope'])}
              disabled={busy}
            >
              <option value="strict">{t('duplicates.scopeStrict')}</option>
              <option value="both">{t('duplicates.scopeBoth')}</option>
            </select>
          </label>
          <div className="grid grid-cols-2 md:grid-cols-4 2xl:grid-cols-2 gap-2">
            <NumOption label={t('duplicates.hashSize')} value={options.hash_size} min={0} max={2048} step={64} disabled={busy} onChange={(value) => patch('hash_size', value)} />
            <NumOption label={t('duplicates.workers')} value={options.hash_workers} min={1} max={32} step={1} disabled={busy} onChange={(value) => patch('hash_workers', value)} />
            <NumOption label={t('duplicates.structure')} value={options.structure_threshold} min={0} max={24} step={1} disabled={busy} onChange={(value) => patch('structure_threshold', value)} />
            <NumOption label={t('duplicates.variantScore')} value={options.variant_score} min={40} max={98} step={1} disabled={busy} onChange={(value) => patch('variant_score', value)} />
            <NumOption label={t('duplicates.aspect')} value={options.aspect_tolerance} min={0.005} max={0.2} step={0.005} disabled={busy} onChange={(value) => patch('aspect_tolerance', value)} />
            <NumOption label={t('duplicates.closeTiles')} value={options.min_close_tiles} min={0} max={1} step={0.01} disabled={busy} onChange={(value) => patch('min_close_tiles', value)} />
            <NumOption label={t('duplicates.tileMedian')} value={options.tile_median} min={0} max={40} step={1} disabled={busy} onChange={(value) => patch('tile_median', value)} />
            <NumOption label={t('duplicates.grayClose')} value={options.min_gray_close} min={0} max={1} step={0.01} disabled={busy} onChange={(value) => patch('min_gray_close', value)} />
          </div>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-fg-tertiary">{t('duplicates.tileGrids')}</span>
            <input
              className="input input-mono px-2 py-1 text-sm"
              value={options.tile_grids.join(',')}
              disabled={busy}
              onChange={(e) => {
                const grids = e.target.value
                  .split(',')
                  .map((part) => Number(part.trim()))
                  .filter((value) => Number.isFinite(value))
                patch('tile_grids', grids.length ? grids : options.tile_grids)
              }}
            />
          </label>
          <button onClick={onScan} disabled={busy} className="btn btn-primary btn-sm">
            {busy ? t('duplicates.scanning') : t('duplicates.scanBtn')}
          </button>
        </div>

        <div className="min-h-[160px] max-h-[34vh] overflow-y-auto pr-1">
          {!result ? (
            <p className="text-sm text-fg-tertiary py-2">{t('duplicates.empty')}</p>
          ) : result.groups.length === 0 ? (
            <p className="text-sm text-fg-tertiary py-2">
              {t('duplicates.noGroups', { total: result.total_images })}
            </p>
          ) : (
            <div className="flex flex-col gap-2">
              {result.groups.map((group) => (
                <DuplicateGroupCard
                  key={group.group_id}
                  projectId={projectId}
                  group={group}
                  selected={selected}
                  onToggle={toggleName}
                  onPreview={onPreview}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  )
}

function NumOption({
  label,
  value,
  min,
  max,
  step,
  disabled,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  disabled: boolean
  onChange: (value: number) => void
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs text-fg-tertiary">{label}</span>
      <input
        type="number"
        className="input input-mono px-2 py-1 text-sm"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  )
}

function DuplicateGroupCard({
  projectId,
  group,
  selected,
  onToggle,
  onPreview,
}: {
  projectId: number
  group: DuplicateGroup
  selected: Set<string>
  onToggle: (name: string) => void
  onPreview: (name: string) => void
}) {
  const { t } = useTranslation()
  return (
    <article className="rounded-md border border-subtle bg-sunken p-2">
      <div className="flex flex-wrap items-center gap-2 mb-2 text-xs">
        <span className="badge badge-neutral">#{group.group_id}</span>
        <span className="text-fg-secondary">
          {t('duplicates.keepSuggested')} <code className="mono">{group.keep}</code>
        </span>
        {group.best && (
          <span className="badge badge-warn">
            {group.best.match_type} · {group.best.score}
          </span>
        )}
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(112px,1fr))] gap-1.5">
        {group.items.map((item) => (
          <DuplicateItemCell
            key={item.name}
            projectId={projectId}
            item={item}
            selected={selected.has(item.name)}
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
  onToggle,
  onPreview,
}: {
  projectId: number
  item: DuplicateItem
  selected: boolean
  onToggle: () => void
  onPreview: () => void
}) {
  const { t } = useTranslation()
  const metrics = item.metrics
  return (
    <div
      className={
        'group relative rounded-md border overflow-hidden bg-surface ' +
        (item.keep ? 'border-ok' : selected ? 'border-warn ring-2 ring-warn-soft' : 'border-subtle')
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
        <div className="flex items-center gap-1 min-w-0">
          {item.keep ? (
            <span className="badge badge-ok shrink-0">{t('duplicates.keep')}</span>
          ) : (
            <button
              type="button"
              onClick={onToggle}
              className={`shrink-0 w-5 h-5 rounded-sm border text-[12px] font-bold ${
                selected ? 'bg-warn text-white border-warn' : 'bg-surface border-dim text-transparent'
              }`}
              aria-label={`${selected ? t('common.deselect') : t('common.select')} ${item.name}`}
            >
              ✓
            </button>
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
