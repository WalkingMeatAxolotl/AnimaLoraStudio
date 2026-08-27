import { useEffect, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type LoraCatalogItem, type LoraCatalogResponse, type LoraCatalogSource, type LoraEntry, type XYAxisType } from '../../../api/client'
import { axisLabel, splitAxisRaw, type XYAxisDraft } from './xy'
import NumberListInput from './NumberListInput'

const AXES: XYAxisType[] = ['lora_ckpt', 'lora_scale', 'cfg_scale', 'steps']
const PAGE_SIZE = 100
const MAX_RANGE_VALUES = 1000

function normalizePath(path: string): string {
  return path.replace(/\\/g, '/').toLocaleLowerCase()
}

function isAbsolutePath(path: string): boolean {
  return /^(?:[a-z]:[\\/]|\\\\|\/)/i.test(path)
}

function sourceName(source: LoraCatalogSource): string {
  if (source.source_type === 'studio_models') return 'loras'
  return source.source_label
}

function itemName(item: LoraCatalogItem): string {
  return item.name.replace(/\.safetensors$/i, '')
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
      const formatted = formatRangeNumber(startValue + index * stepValue)
      if (!withinRange(startValue + index * stepValue)) break
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

function SourceCard({
  source,
  onSelect,
}: {
  source: LoraCatalogSource
  onSelect: () => void
}) {
  const { t } = useTranslation()
  return (
    <button
      type="button"
      data-testid="xy-axis-source"
      className="text-left rounded-lg border border-subtle bg-overlay p-3 hover:bg-hover transition-colors"
      onClick={onSelect}
      title={source.path}
    >
      <div className="font-medium text-sm text-fg-primary truncate">{sourceName(source)}</div>
      <div className="text-2xs text-fg-tertiary mt-1">
        {t('generate.catalogItemCount', { count: source.item_count })}
      </div>
      {source.error && <div className="text-2xs text-warn mt-1 truncate">{source.error}</div>}
    </button>
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
  onFilterChange,
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
  sourceFilter: 'all' | 'project' | 'non_project'
  loadingSources: boolean
  loadingItems: boolean
  loadingMore: boolean
  error: string
  itemCache: Map<string, LoraCatalogItem>
  onFilterChange: (value: 'all' | 'project' | 'non_project') => void
  onSourceChange: (source: LoraCatalogSource | null) => void
  onLoadMore: () => void
  onRetry: () => void
}) {
  const { t } = useTranslation()
  // A reopened editor must treat an existing multi-checkpoint sequence as
  // intentional; otherwise adding one more checkpoint would silently restore
  // catalog order and discard the persisted user order.
  const [manualOrder, setManualOrder] = useState(() => splitAxisRaw(draft.raw).length > 1)
  const paths = splitAxisRaw(draft.raw)
  const selected = new Set(paths.map(normalizePath))
  const fixed = new Set(fixedLoras.map((entry) => normalizePath(entry.path)))

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

    // Snapshot 为避免泄露绝对路径只保存 basename。用户第一次重新选择时，
    // 用真实 catalog 路径替换这些不可直接提交的占位值。
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

  const movePath = (index: number, offset: -1 | 1) => {
    const target = index + offset
    if (target < 0 || target >= paths.length) return
    const next = [...paths]
    ;[next[index], next[target]] = [next[target], next[index]]
    setManualOrder(true)
    updatePaths(next)
  }

  const visibleSources = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase()
    return (sourceResponse?.sources ?? [])
      .filter((source) => source.source_type !== 'project' || source.item_count > 0)
      .filter((source) => sourceFilter === 'all'
        || (sourceFilter === 'project' ? source.source_type === 'project' : source.source_type !== 'project'))
      .filter((source) => !needle
        || `${sourceName(source)} ${source.path}`.toLocaleLowerCase().includes(needle))
      .sort((a, b) => {
        if (a.source_type === 'project' && b.source_type !== 'project') return -1
        if (a.source_type !== 'project' && b.source_type === 'project') return 1
        return sourceName(a).localeCompare(sourceName(b), undefined, { numeric: true, sensitivity: 'base' })
      })
  }, [query, sourceFilter, sourceResponse])

  return (
    <div className="flex flex-col gap-4 min-h-0">
      {paths.length > 0 && (
        <div>
          <div className="caption mb-2">
            {t('generate.axisSelectedValues')} · {paths.length}
          </div>
          <div className="flex flex-col gap-1.5">
            {paths.map((path, index) => (
              <div key={`${path}-${index}`} className="flex items-center gap-2 rounded-md border border-subtle bg-overlay px-2.5 py-2">
                <span className="font-mono text-xs text-fg-primary flex-1 min-w-0 truncate" title={path}>
                  {path.split(/[\\/]/).pop()?.replace(/\.safetensors$/i, '') ?? path}
                </span>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  disabled={index === 0}
                  onClick={() => movePath(index, -1)}
                  title={t('generate.axisMoveUp')}
                  aria-label={t('generate.axisMoveUp')}
                >↑</button>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  disabled={index === paths.length - 1}
                  onClick={() => movePath(index, 1)}
                  title={t('generate.axisMoveDown')}
                  aria-label={t('generate.axisMoveDown')}
                >↓</button>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm text-err"
                  onClick={() => updatePaths(paths.filter((_, i) => i !== index))}
                  title={t('common.delete')}
                  aria-label={t('common.delete')}
                >×</button>
              </div>
            ))}
          </div>
        </div>
      )}

      {!selectedSource ? (
        <>
          <div className="flex items-center gap-2">
            <span className="caption flex-1">{t('generate.catalogChooseSource')}</span>
            <select
              className="input text-xs"
              value={sourceFilter}
              onChange={(event) => onFilterChange(event.target.value as 'all' | 'project' | 'non_project')}
              aria-label={t('generate.catalogFilter')}
            >
              <option value="all">{t('generate.catalogFilterAll')}</option>
              <option value="project">{t('generate.catalogFilterProjects')}</option>
              <option value="non_project">{t('generate.catalogFilterNonProjects')}</option>
            </select>
          </div>
          {loadingSources ? (
            <div className="text-sm text-fg-tertiary py-8 text-center">{t('common.loading')}</div>
          ) : visibleSources.length > 0 ? (
            <div className="grid grid-cols-2 gap-2 overflow-y-auto">
              {visibleSources.map((source) => (
                <SourceCard key={source.source_id} source={source} onSelect={() => onSourceChange(source)} />
              ))}
            </div>
          ) : (
            <div className="text-sm text-fg-tertiary py-8 text-center">{t('generate.catalogSourcesEmpty')}</div>
          )}
        </>
      ) : (
        <>
          <button type="button" className="btn btn-ghost btn-sm self-start" onClick={() => onSourceChange(null)}>
            ← {t('generate.catalogBack')}
          </button>
          {loadingItems ? (
            <div className="text-sm text-fg-tertiary py-8 text-center">{t('common.loading')}</div>
          ) : response && response.items.length > 0 ? (
            <>
              <div className="grid grid-cols-2 gap-2 overflow-y-auto">
                {response.items.map((item) => {
                  const itemKey = normalizePath(item.path)
                  const active = selected.has(itemKey)
                  const blocked = fixed.has(itemKey)
                  return (
                    <button
                      key={item.path}
                      type="button"
                      disabled={blocked}
                      data-testid="xy-axis-checkpoint"
                      className="text-left rounded-lg border p-3 transition-colors"
                      style={{
                        borderColor: active ? 'var(--accent)' : blocked ? 'var(--warn)' : 'var(--border-subtle)',
                        background: active ? 'var(--accent-soft)' : 'var(--bg-overlay)',
                        opacity: blocked ? 0.55 : 1,
                        cursor: blocked ? 'not-allowed' : 'pointer',
                      }}
                      onClick={() => toggleItem(item)}
                      aria-pressed={active}
                      aria-disabled={blocked}
                      title={blocked ? t('generate.axisDuplicateLora') : item.path}
                    >
                      <div className="font-mono text-xs text-fg-primary truncate">{itemName(item)}</div>
                      <div className="text-2xs text-fg-tertiary mt-1 truncate">
                        {item.version_label ?? item.relative_path}
                      </div>
                    </button>
                  )
                })}
              </div>
              {response.next_cursor != null && (
                <button type="button" className="btn btn-secondary btn-sm self-center" disabled={loadingMore} onClick={onLoadMore}>
                  {loadingMore ? t('common.loading') : t('generate.catalogLoadMore')}
                </button>
              )}
            </>
          ) : (
            <div className="text-sm text-fg-tertiary py-8 text-center">{t('generate.catalogEmpty')}</div>
          )}
        </>
      )}

      {error && (
        <div className="rounded-md border border-err bg-err-soft p-3 text-xs text-err" role="alert">
          <div>{t('generate.catalogLoadFailed')}: {error}</div>
          <button type="button" className="btn btn-ghost btn-sm mt-2" onClick={onRetry}>{t('common.retry')}</button>
        </div>
      )}
    </div>
  )
}

export default function XYAxisEditorDrawer({
  open,
  label,
  draft,
  otherAxis,
  fixedLoras,
  onChange,
  onClose,
}: {
  open: boolean
  label: 'X' | 'Y'
  draft: XYAxisDraft
  otherAxis: XYAxisType | null
  fixedLoras: LoraEntry[]
  onChange: (draft: XYAxisDraft) => void
  onClose: () => void
}) {
  const { t } = useTranslation()
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [sourceFilter, setSourceFilter] = useState<'all' | 'project' | 'non_project'>('all')
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
  const axisTypeRef = useRef<HTMLSelectElement>(null)
  const returnFocusRef = useRef<HTMLElement | null>(null)

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedQuery(query.trim()), 220)
    return () => window.clearTimeout(timer)
  }, [query])

  useEffect(() => {
    if (!open || draft.axis !== 'lora_ckpt') return
    let alive = true
    setLoadingSources(true)
    setSourceError('')
    api.getLoraCatalog({ limit: 1 })
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
    api.getLoraCatalog({
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
    returnFocusRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null
    axisTypeRef.current?.focus()
    return () => {
      returnFocusRef.current?.focus()
    }
  }, [open])

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
    api.getLoraCatalog({
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

  const trapFocus = (event: ReactKeyboardEvent<HTMLElement>) => {
    if (event.key !== 'Tab') return
    const focusable = Array.from(event.currentTarget.querySelectorAll<HTMLElement>(
      'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ))
    if (focusable.length === 0) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  if (!open) return null

  const checkpointMode = draft.axis === 'lora_ckpt'
  const error = sourceError || itemError

  return (
    <div className="fixed inset-0 z-40 flex justify-end" data-testid="xy-axis-editor-drawer">
      <button
        type="button"
        tabIndex={-1}
        aria-hidden="true"
        className="absolute inset-0 bg-black/45"
        onClick={onClose}
      />
      <aside
        className="relative z-10 h-full w-full border-l border-subtle bg-surface shadow-2xl flex flex-col"
        style={{ maxWidth: 720 }}
        role="dialog"
        aria-modal="true"
        aria-labelledby="xy-axis-editor-title"
        onKeyDown={trapFocus}
      >
        <header className="px-5 py-4 border-b border-subtle flex items-center gap-3">
          <div className="flex-1 min-w-0">
            <div className="text-xs text-fg-tertiary">{t('generate.xyAxis', { label })}</div>
            <h2 id="xy-axis-editor-title" className="m-0 text-lg font-semibold">{axisLabel(draft.axis)}</h2>
          </div>
          <select
            ref={axisTypeRef}
            className="input text-sm"
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
          <button type="button" className="btn btn-ghost" onClick={onClose} aria-label={t('common.close')}>×</button>
        </header>

        {checkpointMode && (
          <div className="px-5 py-3 border-b border-subtle">
            <input
              className="input w-full"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={selectedSource ? t('generate.catalogSearchLoras') : t('generate.catalogSearchSources')}
            />
          </div>
        )}

        <div className="flex-1 min-h-0 overflow-y-auto p-5">
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
              onFilterChange={setSourceFilter}
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
      </aside>
    </div>
  )
}
