import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import {
  api,
  type DuplicateScanResult,
} from '../../api/client'

interface Props {
  projectId: number
  versionId: number
  result: DuplicateScanResult | null
  selected: Set<string>
  busy: boolean
  onSelectNames: (names: string[]) => void
  onToggle: (name: string) => void
  onPreview: (name: string) => void
  className?: string
}

export function qualityPreviewNames(result: DuplicateScanResult | null): string[] {
  if (!result) return []
  return Array.from(new Set([
    ...result.blur_candidates.map((item) => item.name),
    ...result.crop_relations.flatMap((item) => [item.source, item.crop_candidate]),
  ]))
}

export default function QualityReviewPanel({
  projectId,
  versionId,
  result,
  selected,
  busy,
  onSelectNames,
  onToggle,
  onPreview,
  className = '',
}: Props) {
  const { t } = useTranslation()
  const blurCandidates = result?.blur_candidates ?? []
  const cropRelations = result?.crop_relations ?? []
  const blurNames = useMemo(
    () => Array.from(new Set((result?.blur_candidates ?? []).map((item) => item.name))),
    [result],
  )
  const cropCandidateNames = useMemo(
    () => Array.from(new Set((result?.crop_relations ?? []).map((item) => item.crop_candidate))),
    [result],
  )
  const cropSourceNames = useMemo(
    () => Array.from(new Set((result?.crop_relations ?? []).map((item) => item.source))),
    [result],
  )
  const cropBothNames = useMemo(
    () => Array.from(new Set((result?.crop_relations ?? []).flatMap((item) => [item.source, item.crop_candidate]))),
    [result],
  )
  const cropRelationKindLabel = (kind: string) => {
    if (kind === 'crop_upscaled') return t('duplicates.cropKindUpscaled')
    if (kind === 'crop_same_area') return t('duplicates.cropKindSameArea')
    return t('duplicates.cropKindSmaller')
  }
  const cropLargerLabel = (name: string) => (
    name === 'same_area'
      ? t('duplicates.cropLargerSameArea')
      : t('duplicates.cropLarger', { name })
  )
  const hasResult = !!result
  const hasQuality = blurCandidates.length > 0 || cropRelations.length > 0

  return (
    <section className={`flex flex-col min-h-0 rounded-md border border-subtle bg-surface overflow-hidden ${className}`}>
      <div className="h-0.5 bg-accent" />
      <header className="flex flex-wrap items-center gap-2 px-2.5 py-1.5 border-b border-subtle text-sm">
        <h3 className="font-semibold">{t('duplicates.qualityTitle')}</h3>
        <span className="text-xs text-fg-tertiary">
          {hasResult
            ? t('duplicates.qualitySummary', { blur: blurCandidates.length, crops: cropRelations.length })
            : t('duplicates.qualityEmpty')}
        </span>
        <span className="text-xs text-fg-tertiary min-w-full">
          {t('duplicates.qualityHint')}
        </span>
        <div className="flex flex-wrap items-center gap-1.5 min-w-full">
          <button
            type="button"
            disabled={busy || blurNames.length === 0}
            onClick={() => onSelectNames(blurNames)}
            className="btn btn-secondary btn-sm !py-0.5 text-[11px]"
          >
            {t('duplicates.selectBlur')}
          </button>
          <button
            type="button"
            disabled={busy || cropCandidateNames.length === 0}
            onClick={() => onSelectNames(cropCandidateNames)}
            className="btn btn-secondary btn-sm !py-0.5 text-[11px]"
          >
            {t('duplicates.selectCropCandidates')}
          </button>
          <button
            type="button"
            disabled={busy || cropSourceNames.length === 0}
            onClick={() => onSelectNames(cropSourceNames)}
            className="btn btn-secondary btn-sm !py-0.5 text-[11px]"
          >
            {t('duplicates.selectCropSources')}
          </button>
          <button
            type="button"
            disabled={busy || cropBothNames.length === 0}
            onClick={() => onSelectNames(cropBothNames)}
            className="btn btn-secondary btn-sm !py-0.5 text-[11px]"
          >
            {t('duplicates.selectCropBoth')}
          </button>
        </div>
      </header>
      <div className="flex-1 min-h-0 overflow-y-auto p-2">
        {!hasResult ? (
          <div className="min-h-[150px] flex flex-col items-center justify-center text-center px-6 py-8">
            <div className="text-sm font-medium text-fg-secondary">{t('duplicates.qualityEmptyTitle')}</div>
            <p className="text-sm text-fg-tertiary mt-1 max-w-[52ch]">{t('duplicates.qualityEmpty')}</p>
          </div>
        ) : !hasQuality ? (
          <div className="min-h-[150px] flex items-center justify-center text-sm text-fg-tertiary px-6 text-center">
            {t('duplicates.qualityNoCandidates')}
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            <QualitySection
              title={t('duplicates.blurTitle')}
              empty={t('duplicates.blurEmpty')}
              items={blurCandidates.map((item) => ({
                key: item.name,
                images: [{ name: item.name }],
                meta: `${item.width}x${item.height} · ${item.filesize_kb}KB`,
                score: t('duplicates.blurMetric', {
                  score: Math.round(item.blur_score),
                  local: Math.round(item.largest_blur_region_ratio * 100),
                }),
                note: item.reason,
              }))}
              projectId={projectId}
              versionId={versionId}
              selected={selected}
              busy={busy}
              onToggle={onToggle}
              onPreview={onPreview}
            />
            <QualitySection
              title={t('duplicates.cropTitle')}
              empty={t('duplicates.cropEmpty')}
              items={cropRelations.map((item, index) => ({
                key: `${item.source}:${item.crop_candidate}:${index}`,
                images: [
                  { name: item.source, label: t('duplicates.cropSource') },
                  { name: item.crop_candidate, label: t('duplicates.cropCandidate') },
                ],
                meta: `${item.source_width}x${item.source_height} -> ${item.crop_width}x${item.crop_height}`,
                score: t('duplicates.cropMetric', {
                  score: Math.round(item.score * 100),
                  area: Math.round(item.window_ratio * 100),
                }),
                note: [
                  cropRelationKindLabel(item.relation_kind),
                  cropLargerLabel(item.larger_image),
                  t('duplicates.cropAreaRatio', { ratio: item.area_ratio.toFixed(2) }),
                  `${item.source_window.x},${item.source_window.y},${item.source_window.width},${item.source_window.height}`,
                  item.note,
                ].join(' · '),
              }))}
              projectId={projectId}
              versionId={versionId}
              selected={selected}
              busy={busy}
              onToggle={onToggle}
              onPreview={onPreview}
            />
          </div>
        )}
      </div>
    </section>
  )
}

function QualitySection({
  title,
  empty,
  items,
  projectId,
  versionId,
  selected,
  busy,
  onToggle,
  onPreview,
}: {
  title: string
  empty: string
  items: Array<{ key: string; images: Array<{ name: string; label?: string }>; meta: string; score: string; note: string }>
  projectId: number
  versionId: number
  selected: Set<string>
  busy: boolean
  onToggle: (name: string) => void
  onPreview: (name: string) => void
}) {
  return (
    <div className="rounded-sm border border-subtle bg-sunken/40 overflow-hidden">
      <div className="px-2 py-1.5 border-b border-subtle text-xs font-medium text-fg-secondary">{title}</div>
      {items.length === 0 ? (
        <div className="px-2 py-3 text-xs text-fg-tertiary">{empty}</div>
      ) : (
        <div
          className="p-2 grid gap-2"
          style={{
            gridTemplateColumns: `repeat(auto-fill, minmax(${
              items.some((item) => item.images.length > 1) ? 340 : 200
            }px, 1fr))`,
          }}
        >
          {items.map((item) => (
            <article key={item.key} className="rounded-sm border border-subtle bg-surface p-1.5">
              <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${item.images.length}, minmax(0, 1fr))` }}>
                {item.images.map((image) => (
                  <QualityImageCell
                    key={image.name}
                    projectId={projectId}
                    versionId={versionId}
                    name={image.name}
                    label={image.label}
                    selected={selected.has(image.name)}
                    busy={busy}
                    onToggle={() => onToggle(image.name)}
                    onPreview={() => onPreview(image.name)}
                  />
                ))}
              </div>
              <div className="mt-1.5 flex flex-col gap-0.5 text-[11px]">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="badge badge-neutral shrink-0">{item.score}</span>
                  <span className="text-fg-tertiary truncate">{item.meta}</span>
                </div>
                <code className="mono text-fg-secondary truncate">{item.images.map((image) => image.name).join(' <-> ')}</code>
                <div className="text-fg-tertiary truncate" title={item.note}>{item.note}</div>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  )
}

function QualityImageCell({
  projectId,
  versionId,
  name,
  label,
  selected,
  busy,
  onToggle,
  onPreview,
}: {
  projectId: number
  versionId: number
  name: string
  label?: string
  selected: boolean
  busy: boolean
  onToggle: () => void
  onPreview: () => void
}) {
  const { t } = useTranslation()
  return (
    <div
      className={
        'rounded-sm border overflow-hidden bg-surface ' +
        (selected ? 'border-warn ring-2 ring-warn-soft' : 'border-subtle')
      }
    >
      <button
        type="button"
        disabled={busy}
        onClick={onPreview}
        className="block w-full aspect-square bg-sunken disabled:opacity-70"
        title={name}
      >
        {(() => {
          const i = name.lastIndexOf('/')
          const folder = i >= 0 ? name.slice(0, i) : ''
          const filename = i >= 0 ? name.slice(i + 1) : name
          return (
            <img
              src={api.versionThumbUrl(projectId, versionId, 'train', filename, folder, 256)}
              alt={name}
              loading="lazy"
              decoding="async"
              className="w-full h-full object-cover"
            />
          )
        })()}
      </button>
      <div className="px-1.5 py-1 flex items-center gap-1 min-w-0">
        <button
          type="button"
          onClick={onToggle}
          disabled={busy}
          className={`shrink-0 px-1.5 py-0.5 rounded-sm border text-[11px] font-medium ${
            selected
              ? 'bg-warn text-white border-warn'
              : 'bg-ok-soft text-ok border-ok'
          } disabled:opacity-60 disabled:cursor-not-allowed`}
          aria-label={`${selected ? t('duplicates.restoreCandidate') : t('duplicates.removeCandidate')} ${name}`}
        >
          {selected ? t('duplicates.selectedRemove') : t('duplicates.keep')}
        </button>
        {label && <span className="badge badge-neutral shrink-0">{label}</span>}
        <code className="mono truncate min-w-0 text-[11px]">{name}</code>
      </div>
    </div>
  )
}
