import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  api,
  type LoraCatalogItem,
  type LoraCatalogResponse,
  type LoraCatalogSource,
  type LoraEntry,
  type XYAxisType,
} from '../../../api/client'
import GenerateAttachedDrawer from './GenerateAttachedDrawer'
import NumberListInput from './NumberListInput'
import { axisLabel, splitAxisRaw, type XYAxisDraft } from './xy'

const AXES: XYAxisType[] = ['lora_ckpt', 'lora_scale', 'cfg_scale', 'steps']
const PAGE_SIZE = 100
const MAX_RANGE_VALUES = 1000

type SourceFilter = 'all' | 'project' | 'non_project'

function normalizePath(path: string): string {
  return path.replace(/\\/g, '/').toLocaleLowerCase()
}

function isAbsolutePath(path: string): boolean {
  return /^(?:[a-z]:[\\/]|\\\\|\/)/i.test(path)
}

function sourceName(source: LoraCatalogSource): string {
  if (source.source_type === 'project') return source.source_label
  const normalized = source.path.replace(/[\\/]+$/, '').replace(/\\/g, '/')
  return normalized.split('/').pop() || source.source_label
}

function itemName(item: LoraCatalogItem): string {
  return item.name.replace(/\.safetensors$/i, '')
}

function relativeDirectory(item: LoraCatalogItem): string {
  const parts = item.relative_path.replace(/\\/g, '/').split('/')
  return parts.length > 1 ? parts.slice(0, -1).join('/') : ''
}

function checkpointNumber(item: LoraCatalogItem): number {
  const match = `${item.relative_path} ${item.name}`.match(/(?:step|epoch)[-_ ]?(\d+)/i)
  return match ? Number(match[1]) : -1
}

/** 与历史 picker 的 canonical 顺序一致：final → step↓ → epoch↓ → other。 */
function compareCheckpoints(a: LoraCatalogItem, b: LoraCatalogItem): number {
  const kindRank: Record<LoraCatalogItem['kind'], number> = {
    final: 0,
    step: 1,
    epoch: 2,
    other: 3,
  }
  const rankDiff = kindRank[a.kind] - kindRank[b.kind]
  if (rankDiff !== 0) return rankDiff
  if (a.kind === 'step' || a.kind === 'epoch') {
    const numberDiff = checkpointNumber(b) - checkpointNumber(a)
    if (numberDiff !== 0) return numberDiff
  }
  return a.relative_path.localeCompare(b.relative_path, undefined, {
    numeric: true,
    sensitivity: 'base',
  })
}

function asLoraEntry(item: LoraCatalogItem): LoraEntry {
  return {
    path: item.path,
    scale: 1,
    project_id: item.project_id,
    version_id: item.version_id,
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function formatRangeNumber(value: number): string {
  return String(Number(value.toPrecision(12)))
}

function NumericAxisEditor({
  draft,
  onChange,
}: {
  draft: XYAxisDraft
  onChange: (draft: XYAxisDraft) => void
}) {
  const { t } = useTranslation()
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [step, setStep] = useState(draft.axis === 'steps' ? '1' : '0.1')
  const [error, setError] = useState('')

  const addRange = () => {
    const startValue = Number(start)
    const endValue = Number(end)
    const stepValue = Number(step)
    const allPresent = [start, end, step].every((value) => value.trim().length > 0)
    const allFinite = [startValue, endValue, stepValue].every(Number.isFinite)
    const wrongDirection = (endValue - startValue) * stepValue < 0
    const nonIntegerSteps = draft.axis === 'steps'
      && ![startValue, endValue, stepValue].every(Number.isInteger)
    if (!allPresent || !allFinite || stepValue === 0 || wrongDirection || nonIntegerSteps) {
      setError(t('generate.axisRangeInvalid'))
      return
    }

    const generated: string[] = []
    const ascending = stepValue > 0
    const tolerance = Math.max(Math.abs(startValue), Math.abs(endValue), 1) * Number.EPSILON * 8
    const withinRange = (value: number) => ascending
      ? value <= endValue + tolerance
      : value >= endValue - tolerance
    for (let index = 0; index < MAX_RANGE_VALUES; index += 1) {
      const value = startValue + index * stepValue
      if (!withinRange(value)) break
      const formatted = formatRangeNumber(value)
      if (generated[generated.length - 1] !== formatted) generated.push(formatted)
    }
    if (withinRange(startValue + MAX_RANGE_VALUES * stepValue)) {
      setError(t('generate.axisRangeTooLarge'))
      return
    }
    onChange({ ...draft, raw: generated.join(', ') })
    setError('')
  }

  return (
    <div className="flex flex-col gap-5">
      <div>
        <label className="caption block mb-1.5">{t('generate.axisDirectInput')}</label>
        <NumberListInput
          raw={draft.raw}
          onChange={(raw) => onChange({ ...draft, raw })}
          placeholder={draft.axis === 'steps' ? '20, 25, 30' : '0.5, 0.75, 1.0'}
        />
      </div>

      <div className="border-t border-subtle pt-4">
        <div className="caption mb-2">{t('generate.axisRange')}</div>
        <div className="grid grid-cols-3 gap-2">
          {[
            [t('generate.axisRangeStart'), start, setStart],
            [t('generate.axisRangeEnd'), end, setEnd],
            [t('generate.axisRangeStep'), step, setStep],
          ].map(([label, value, setter]) => (
            <label key={String(label)} className="min-w-0">
              <span className="text-2xs text-fg-tertiary block mb-1">{String(label)}</span>
              <input
                type="number"
                step="any"
                className="input input-mono w-full text-xs"
                value={String(value)}
                onChange={(event) => (setter as (value: string) => void)(event.target.value)}
              />
            </label>
          ))}
        </div>
        <button type="button" className="btn btn-secondary btn-sm mt-2" onClick={addRange}>
          {t('generate.axisRangeAdd')}
        </button>
        {error && <div className="text-xs text-err mt-2" role="alert">{error}</div>}
      </div>
    </div>
  )
}

function CheckpointBrowser({
  draft,
  onChange,
  fixedLoras,
  selectedSource,
  sourceResponse,
  response,
  query,
  sourceFilter,
  loadingSources,
  loadingItems,
  loadingMore,
  error,
  itemCache,
  manualOrderRevision,
  onSourceChange,
  onLoadMore,
  onRetry,
}: {
  draft: XYAxisDraft
  onChange: (draft: XYAxisDraft) => void
  fixedLoras: LoraEntry[]
  selectedSource: LoraCatalogSource | null
  sourceResponse: LoraCatalogResponse | null
  response: LoraCatalogResponse | null
  query: string
  sourceFilter: SourceFilter
  loadingSources: boolean
  loadingItems: boolean
  loadingMore: boolean
  error: string
  itemCache: Map<string, LoraCatalogItem>
  manualOrderRevision: number
  onSourceChange: (source: LoraCatalogSource | null) => void
  onLoadMore: () => void
  onRetry: () => void
}) {
  const { t } = useTranslation()
  const [manualOrder, setManualOrder] = useState(() => splitAxisRaw(draft.raw).length > 1)
  const paths = splitAxisRaw(draft.raw)
  const selected = new Set(paths.map(normalizePath))
  const fixed = new Set(fixedLoras.map((entry) => normalizePath(entry.path)))

  useEffect(() => {
    if (manualOrderRevision > 0) setManualOrder(true)
  }, [manualOrderRevision])

  const updatePaths = (next: string[]) => {
    let checkpointAnchor = draft.checkpointAnchor ?? null
    if (!checkpointAnchor || !next.some((path) => normalizePath(path) === normalizePath(checkpointAnchor!.path))) {
      const firstPath = next[0]
      const first = firstPath ? itemCache.get(normalizePath(firstPath)) : undefined
      checkpointAnchor = first
        ? asLoraEntry(first)
        : firstPath && isAbsolutePath(firstPath)
          ? { path: firstPath, scale: 1 }
          : null
    }
    onChange({ ...draft, raw: next.join(', '), checkpointAnchor })
  }

  const toggleItem = (item: LoraCatalogItem) => {
    const itemKey = normalizePath(item.path)
    if (fixed.has(itemKey)) return
    if (selected.has(itemKey)) {
      updatePaths(paths.filter((path) => normalizePath(path) !== itemKey))
      return
    }

    // Snapshot 只保存 basename；第一次重新选择时以真实 catalog 路径替换占位值。
    const reusablePaths = paths.every(isAbsolutePath) ? paths : []
    const next = [...reusablePaths, item.path]
    if (!manualOrder) {
      next.sort((left, right) => {
        const leftItem = itemCache.get(normalizePath(left))
        const rightItem = itemCache.get(normalizePath(right))
        if (leftItem && rightItem) return compareCheckpoints(leftItem, rightItem)
        if (leftItem) return -1
        if (rightItem) return 1
        return 0
      })
    }
    const checkpointAnchor = draft.checkpointAnchor && reusablePaths.some(
      (path) => normalizePath(path) === normalizePath(draft.checkpointAnchor!.path),
    )
      ? draft.checkpointAnchor
      : asLoraEntry(itemCache.get(normalizePath(next[0])) ?? item)
    onChange({ ...draft, raw: next.join(', '), checkpointAnchor })
  }

  const visibleSources = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase()
    return (sourceResponse?.sources ?? [])
      .filter((source) => source.source_type !== 'project' || source.item_count > 0 || Boolean(source.error))
      .filter((source) => sourceFilter === 'all'
        || (sourceFilter === 'project' ? source.source_type === 'project' : source.source_type !== 'project'))
      .filter((source) => !needle
        || `${sourceName(source)} ${source.source_label} ${source.path}`.toLocaleLowerCase().includes(needle))
      .sort((a, b) => sourceName(a).localeCompare(sourceName(b), undefined, { numeric: true, sensitivity: 'base' }))
  }, [query, sourceFilter, sourceResponse])

  if (error) {
    return (
      <div className="rounded-md border border-err bg-err-soft p-3 text-xs text-err" role="alert">
        <div>{t('generate.catalogLoadFailed')}: {error}</div>
        <button type="button" className="btn btn-ghost btn-sm mt-2" onClick={onRetry}>{t('common.retry')}</button>
      </div>
    )
  }

  if (!selectedSource) {
    if (loadingSources) return <div className="text-sm text-fg-tertiary p-8 text-center">{t('common.loading')}</div>
    if (visibleSources.length === 0) return <div className="text-sm text-fg-tertiary p-8 text-center">{t('generate.catalogSourcesEmpty')}</div>
    return (
      <div
        className="grid gap-2"
        style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))' }}
        data-testid="xy-axis-sources"
      >
        {visibleSources.map((source) => (
          <button
            type="button"
            key={source.source_id}
            className="rounded-md border border-subtle bg-sunken hover:bg-overlay p-3 text-left cursor-pointer transition-colors min-w-0"
            onClick={() => onSourceChange(source)}
            data-testid="xy-axis-source"
            title={source.path}
          >
            <span className="block text-sm font-medium text-fg-primary truncate">{sourceName(source)}</span>
            <span className="block text-2xs text-fg-tertiary mt-1 truncate">
              {source.source_type === 'project' ? t('generate.catalogProject') : source.path}
            </span>
            <span className={`block text-xs mt-2 ${source.error ? 'text-warn' : 'text-fg-secondary'}`}>
              {source.error || t('generate.catalogItemCount', { count: source.item_count })}
            </span>
          </button>
        ))}
      </div>
    )
  }

  if (loadingItems) return <div className="text-sm text-fg-tertiary p-8 text-center">{t('common.loading')}</div>
  if (!response || response.items.length === 0) return <div className="text-sm text-fg-tertiary p-8 text-center">{t('generate.catalogEmpty')}</div>

  return (
    <>
      <div className="flex flex-col gap-1">
        {response.items.map((item) => {
          const itemKey = normalizePath(item.path)
          const active = selected.has(itemKey)
          const blocked = fixed.has(itemKey)
          const secondary = item.source_type === 'project' ? item.version_label : relativeDirectory(item)
          return (
            <button
              key={item.path}
              type="button"
              disabled={blocked}
              data-testid="xy-axis-checkpoint"
              className={`w-full rounded-md border px-3 py-2 flex items-center gap-3 text-left transition-colors ${active ? 'border-selected bg-selected-soft' : 'border-subtle bg-sunken hover:bg-overlay'}`}
              style={{ opacity: blocked ? 0.55 : 1, cursor: blocked ? 'not-allowed' : 'pointer' }}
              onClick={() => toggleItem(item)}
              aria-pressed={active}
              aria-disabled={blocked}
              title={blocked ? t('generate.axisDuplicateLora') : item.path}
            >
              <span className={`w-4 shrink-0 text-center ${active ? 'text-ok' : 'text-fg-tertiary'}`}>{active ? '✓' : ''}</span>
              <span className="flex-1 min-w-0">
                <span className="block font-mono text-xs text-fg-primary truncate">{itemName(item)}</span>
                {secondary && <span className="block text-2xs text-fg-tertiary truncate">{secondary}</span>}
              </span>
            </button>
          )
        })}
      </div>
      {response.next_cursor != null && (
        <button type="button" className="btn btn-secondary w-full mt-2" disabled={loadingMore} onClick={onLoadMore}>
          {loadingMore ? t('common.loading') : t('generate.catalogLoadMore')}
        </button>
      )}
    </>
  )
}

export default function XYAxisEditorDrawer({
  open,
  label,
  draft,
  otherAxis,
  fixedLoras,
  manualOrderRevision = 0,
  onChange,
  onClose,
}: {
  open: boolean
  label: 'X' | 'Y'
  draft: XYAxisDraft
  otherAxis: XYAxisType | null
  fixedLoras: LoraEntry[]
  manualOrderRevision?: number
  onChange: (draft: XYAxisDraft) => void
  onClose: () => void
}) {
  const { t } = useTranslation()
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>('all')
  const [selectedSource, setSelectedSource] = useState<LoraCatalogSource | null>(null)
  const [sourceResponse, setSourceResponse] = useState<LoraCatalogResponse | null>(null)
  const [response, setResponse] = useState<LoraCatalogResponse | null>(null)
  const [loadingSources, setLoadingSources] = useState(false)
  const [loadingItems, setLoadingItems] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [sourceError, setSourceError] = useState('')
  const [itemError, setItemError] = useState('')
  const [reloadKey, setReloadKey] = useState(0)
  const itemCacheRef = useRef(new Map<string, LoraCatalogItem>())
  const itemGenerationRef = useRef(0)

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedQuery(query.trim()), 180)
    return () => window.clearTimeout(timer)
  }, [query])

  useEffect(() => {
    if (!open || draft.axis !== 'lora_ckpt') return
    let alive = true
    setLoadingSources(true)
    setSourceError('')
    void api.getLoraCatalog({ limit: 1, refresh: reloadKey > 0 })
      .then((next) => { if (alive) setSourceResponse(next) })
      .catch((error) => { if (alive) setSourceError(errorMessage(error)) })
      .finally(() => { if (alive) setLoadingSources(false) })
    return () => { alive = false }
  }, [draft.axis, open, reloadKey])

  useEffect(() => {
    const generation = ++itemGenerationRef.current
    setLoadingMore(false)
    if (!open || draft.axis !== 'lora_ckpt' || !selectedSource) {
      setResponse(null)
      setItemError('')
      return
    }
    setLoadingItems(true)
    setItemError('')
    void api.getLoraCatalog({
      source: selectedSource.source_id,
      q: debouncedQuery || undefined,
      sort: 'recommended',
      order: 'asc',
      limit: PAGE_SIZE,
      cursor: 0,
    })
      .then((next) => {
        if (generation !== itemGenerationRef.current) return
        for (const item of next.items) itemCacheRef.current.set(normalizePath(item.path), item)
        setResponse(next)
      })
      .catch((error) => {
        if (generation === itemGenerationRef.current) setItemError(errorMessage(error))
      })
      .finally(() => {
        if (generation === itemGenerationRef.current) setLoadingItems(false)
      })
  }, [debouncedQuery, draft.axis, open, reloadKey, selectedSource])

  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose, open])

  const loadMore = () => {
    if (!selectedSource || response?.next_cursor == null || loadingMore) return
    const generation = itemGenerationRef.current
    const cursor = response.next_cursor
    setLoadingMore(true)
    setItemError('')
    void api.getLoraCatalog({
      source: selectedSource.source_id,
      q: debouncedQuery || undefined,
      sort: 'recommended',
      order: 'asc',
      limit: PAGE_SIZE,
      cursor,
    })
      .then((next) => {
        if (generation !== itemGenerationRef.current) return
        for (const item of next.items) itemCacheRef.current.set(normalizePath(item.path), item)
        setResponse((previous) => previous ? { ...next, items: [...previous.items, ...next.items], cursor: 0 } : next)
      })
      .catch((error) => {
        if (generation === itemGenerationRef.current) setItemError(errorMessage(error))
      })
      .finally(() => {
        if (generation === itemGenerationRef.current) setLoadingMore(false)
      })
  }

  if (!open) return null

  const checkpointMode = draft.axis === 'lora_ckpt'
  const error = sourceError || itemError

  return (
    <GenerateAttachedDrawer
      id="xy-axis-editor-drawer"
      ariaLabel={`${t('generate.xyAxis', { label })} · ${axisLabel(draft.axis)}`}
      testId="xy-axis-editor-drawer"
    >
      <header className="p-3 border-b border-subtle flex flex-col gap-2 shrink-0">
        <div className="flex items-center gap-2">
          {checkpointMode && selectedSource && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => {
                setSelectedSource(null)
                setQuery('')
                setDebouncedQuery('')
              }}
              aria-label={t('generate.catalogBack')}
            >‹</button>
          )}
          <strong className="text-sm flex-1 truncate">
            {checkpointMode && selectedSource
              ? sourceName(selectedSource)
              : `${t('generate.xyAxis', { label })} · ${axisLabel(draft.axis)}`}
          </strong>
          {checkpointMode && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => setReloadKey((key) => key + 1)}
              disabled={loadingSources || loadingItems}
              title={t('common.refresh')}
              aria-label={t('common.refresh')}
            >
              <svg className={loadingSources || loadingItems ? 'animate-spin' : ''} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
                <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
                <path d="M16 16h5v5" />
              </svg>
            </button>
          )}
          <select
            className="input text-xs shrink-0"
            style={{ width: 132 }}
            aria-label={t('generate.axisType')}
            value={draft.axis}
            onChange={(event) => onChange({
              axis: event.target.value as XYAxisType,
              raw: '',
              loraIndex: null,
              checkpointAnchor: null,
            })}
          >
            {AXES.map((axis) => (
              <option key={axis} value={axis} disabled={axis === otherAxis}>{axisLabel(axis)}</option>
            ))}
          </select>
          <button
            type="button"
            className="btn btn-ghost btn-sm text-fg-tertiary px-1.5"
            onClick={onClose}
            title={t('common.close')}
            aria-label={t('common.close')}
          >×</button>
        </div>

        {checkpointMode && (
          <div className="flex items-center gap-2 flex-wrap">
            <input
              className="input text-xs flex-1"
              style={{ minWidth: 240 }}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={selectedSource ? t('generate.catalogSearchLoras') : t('generate.catalogSearchSources')}
              autoFocus
            />
            {!selectedSource && (
              <select
                className="input text-xs shrink-0"
                style={{ width: 132 }}
                value={sourceFilter}
                onChange={(event) => setSourceFilter(event.target.value as SourceFilter)}
                aria-label={t('generate.catalogFilter')}
              >
                <option value="all">{t('generate.catalogFilterAll')}</option>
                <option value="project">{t('generate.catalogFilterProjects')}</option>
                <option value="non_project">{t('generate.catalogFilterNonProjects')}</option>
              </select>
            )}
          </div>
        )}
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto p-3">
        {checkpointMode ? (
          <CheckpointBrowser
            draft={draft}
            onChange={onChange}
            fixedLoras={fixedLoras}
            selectedSource={selectedSource}
            sourceResponse={sourceResponse}
            response={response}
            query={query}
            sourceFilter={sourceFilter}
            loadingSources={loadingSources}
            loadingItems={loadingItems}
            loadingMore={loadingMore}
            error={error}
            itemCache={itemCacheRef.current}
            manualOrderRevision={manualOrderRevision}
            onSourceChange={(source) => {
              setSelectedSource(source)
              setQuery('')
              setDebouncedQuery('')
            }}
            onLoadMore={loadMore}
            onRetry={() => setReloadKey((key) => key + 1)}
          />
        ) : (
          <NumericAxisEditor key={draft.axis} draft={draft} onChange={onChange} />
        )}
      </div>
    </GenerateAttachedDrawer>
  )
}
