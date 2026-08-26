import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  api,
  type LoraCatalogItem,
  type LoraCatalogResponse,
  type LoraCatalogSource,
  type LoraEntry,
} from '../../../api/client'
import { createLoraUiState, normalizeLoraPath, type LoraUiState } from './loraSelection'

const EMPTY_RESPONSE: LoraCatalogResponse = {
  items: [], sources: [], total: 0, cursor: 0, next_cursor: null,
  generated_at: 0, cached: false, cache_ttl_seconds: 0,
}

type SourceFilter = 'all' | 'project' | 'non_project'

function displayName(name: string): string {
  return name.replace(/\.safetensors$/i, '')
}

function relativeDirectory(item: LoraCatalogItem): string {
  const parts = item.relative_path.replace(/\\/g, '/').split('/')
  return parts.length > 1 ? parts.slice(0, -1).join('/') : ''
}

function sourceDisplayName(source: LoraCatalogSource): string {
  if (source.source_type === 'project') return source.source_label
  const normalized = source.path.replace(/[\\/]+$/, '').replace(/\\/g, '/')
  return normalized.split('/').pop() || source.source_label
}

export default function LoraCatalogDrawer({
  open,
  onClose,
  loras,
  ui,
  onChange,
}: {
  open: boolean
  onClose: () => void
  loras: LoraEntry[]
  ui: LoraUiState[]
  onChange: (loras: LoraEntry[], ui: LoraUiState[]) => void
}) {
  const { t } = useTranslation()
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [selectedSource, setSelectedSource] = useState<LoraCatalogSource | null>(null)
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>('all')
  const [versionId, setVersionId] = useState('')
  const [sources, setSources] = useState<LoraCatalogSource[]>([])
  const [response, setResponse] = useState<LoraCatalogResponse>(EMPTY_RESPONSE)
  const [loading, setLoading] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [refreshToken, setRefreshToken] = useState(0)
  const handledRefreshToken = useRef(0)

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedQuery(query.trim()), 180)
    return () => window.clearTimeout(timer)
  }, [query])

  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [open, onClose])

  const requestQuery = selectedSource ? debouncedQuery : ''

  useEffect(() => {
    if (!open) return
    let canceled = false
    setLoading(true)
    setError(null)
    const forceRefresh = refreshToken !== handledRefreshToken.current
    handledRefreshToken.current = refreshToken
    void api.getLoraCatalog(selectedSource ? {
      q: requestQuery,
      source: selectedSource.source_id,
      sort: 'name',
      order: 'asc',
      include_archived: false,
      limit: 500,
      refresh: forceRefresh,
    } : {
      include_archived: false,
      limit: 1,
      refresh: forceRefresh,
    }).then((result) => {
      if (canceled) return
      if (selectedSource) setResponse(result)
      else {
        setSources(result.sources)
        setResponse(EMPTY_RESPONSE)
      }
    }).catch((reason) => {
      if (!canceled) setError(String(reason))
    }).finally(() => {
      if (!canceled) setLoading(false)
    })
    return () => { canceled = true }
  }, [open, selectedSource, requestQuery, refreshToken])

  const visibleSources = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase()
    const filtered = sources.filter((source) => {
      const hasCatalogItems = source.source_type !== 'project' || source.item_count > 0 || Boolean(source.error)
      const matchesFilter = sourceFilter === 'all'
        || (sourceFilter === 'project' && source.source_type === 'project')
        || (sourceFilter === 'non_project' && source.source_type !== 'project')
      const matchesQuery = !needle
        || `${sourceDisplayName(source)}\n${source.source_label}\n${source.path}`.toLocaleLowerCase().includes(needle)
      return hasCatalogItems && matchesFilter && matchesQuery
    })
    return [...filtered].sort((a, b) => sourceDisplayName(a).localeCompare(
      sourceDisplayName(b), undefined, { sensitivity: 'base', numeric: true },
    ))
  }, [query, sourceFilter, sources])

  const versions = useMemo(() => {
    const found = new Map<number, string>()
    response.items.forEach((item) => {
      if (item.version_id != null) found.set(item.version_id, item.version_label || `#${item.version_id}`)
    })
    return [...found.entries()]
  }, [response.items])

  const visibleItems = useMemo(() => {
    if (!versionId) return response.items
    return response.items.filter((item) => String(item.version_id) === versionId)
  }, [response.items, versionId])

  const selectedByPath = useMemo(() => {
    const selected = new Map<string, { index: number; state: LoraUiState }>()
    loras.forEach((entry, index) => {
      if (entry.path) selected.set(normalizeLoraPath(entry.path), { index, state: ui[index] })
    })
    return selected
  }, [loras, ui])

  const enterSource = (source: LoraCatalogSource) => {
    setQuery('')
    setDebouncedQuery('')
    setVersionId('')
    setResponse(EMPTY_RESPONSE)
    setSelectedSource(source)
  }

  const leaveSource = () => {
    setQuery('')
    setDebouncedQuery('')
    setVersionId('')
    setSelectedSource(null)
  }

  const toggle = (item: LoraCatalogItem) => {
    const existing = selectedByPath.get(normalizeLoraPath(item.path))
    if (existing) {
      onChange(loras.filter((_, index) => index !== existing.index), ui.filter((_, index) => index !== existing.index))
      return
    }
    onChange([
      ...loras,
      {
        path: item.path,
        scale: 1,
        project_id: item.project_id,
        version_id: item.version_id,
      },
    ], [...ui, createLoraUiState(true)])
  }

  const loadMore = async () => {
    if (!selectedSource || response.next_cursor == null) return
    setLoadingMore(true)
    try {
      const next = await api.getLoraCatalog({
        q: debouncedQuery,
        source: selectedSource.source_id,
        sort: 'name',
        order: 'asc',
        include_archived: false,
        limit: 500,
        cursor: response.next_cursor,
      })
      setResponse((previous) => ({ ...next, items: [...previous.items, ...next.items] }))
    } catch (reason) {
      setError(String(reason))
    } finally {
      setLoadingMore(false)
    }
  }

  if (!open) return null

  return (
    <aside
      id="lora-catalog-drawer"
      aria-label={t('generate.loraCatalog')}
      className="absolute z-20 flex flex-col bg-surface border border-subtle border-l-0 shadow-xl"
      style={{
        left: 'calc(420px + 1.5rem)',
        top: '1.5rem',
        bottom: '1.5rem',
        width: 'min(860px, max(480px, calc(100% - 850px - 1.5rem)), calc(100% - 420px - 10.25rem))',
        borderRadius: '0 var(--r-lg) var(--r-lg) 0',
        overflow: 'hidden',
      }}
      data-testid="lora-catalog-drawer"
    >
      <header className="p-3 border-b border-subtle flex flex-col gap-2 shrink-0">
        <div className="flex items-center gap-2">
          {selectedSource && (
            <button type="button" className="btn btn-ghost btn-sm" onClick={leaveSource} aria-label={t('generate.catalogBack')}>
              ‹
            </button>
          )}
          <strong className="text-sm flex-1 truncate">
            {selectedSource ? sourceDisplayName(selectedSource) : t('generate.catalogChooseSource')}
          </strong>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() => setRefreshToken((token) => token + 1)}
            disabled={loading}
            title={t('common.refresh')}
            aria-label={t('common.refresh')}
          >
            <svg className={loading ? 'animate-spin' : ''} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
              <path d="M3 3v5h5" />
              <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
              <path d="M16 16h5v5" />
            </svg>
          </button>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {selectedSource?.source_type === 'project' && versions.length > 1 && (
            <select
              className="input text-xs shrink-0"
              style={{ width: 132 }}
              value={versionId}
              onChange={(event) => setVersionId(event.target.value)}
              aria-label={t('generate.catalogVersion')}
            >
              <option value="">{t('generate.catalogAllVersions')}</option>
              {versions.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
            </select>
          )}
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
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto p-3">
        {error && <div className="text-xs text-err bg-err-soft border border-err rounded-md p-2 mb-2" role="alert">{error}</div>}
        {loading ? (
          <div className="text-sm text-fg-tertiary p-8 text-center">{t('common.loading')}</div>
        ) : selectedSource ? (
          <>
            {selectedSource.error && (
              <div className="text-xs text-warn bg-warn-soft border border-subtle rounded-md p-2 mb-2">{selectedSource.error}</div>
            )}
            {visibleItems.length === 0 ? (
              <div className="text-sm text-fg-tertiary p-8 text-center">{t('generate.catalogEmpty')}</div>
            ) : (
              <div className="flex flex-col gap-1">
                {visibleItems.map((item) => {
                  const existing = selectedByPath.get(normalizeLoraPath(item.path))
                  const selected = Boolean(existing)
                  const secondary = item.source_type === 'project'
                    ? item.version_label
                    : relativeDirectory(item)
                  return (
                    <button
                      type="button"
                      key={item.path}
                      className={`w-full rounded-md border px-3 py-2 flex items-center gap-3 text-left cursor-pointer transition-colors ${selected ? 'border-selected bg-selected-soft' : 'border-subtle bg-sunken hover:bg-overlay'}`}
                      onClick={() => toggle(item)}
                      data-testid="lora-catalog-item"
                      aria-pressed={selected}
                    >
                      <span className={`w-4 shrink-0 text-center ${selected ? 'text-ok' : 'text-fg-tertiary'}`}>{selected ? '✓' : ''}</span>
                      <span className="flex-1 min-w-0">
                        <span className="block font-mono text-xs text-fg-primary truncate" title={displayName(item.name)}>{displayName(item.name)}</span>
                        {secondary && <span className="block text-2xs text-fg-tertiary truncate">{secondary}</span>}
                      </span>
                    </button>
                  )
                })}
              </div>
            )}
            {response.next_cursor != null && (
              <button type="button" className="btn btn-secondary w-full mt-2" onClick={() => void loadMore()} disabled={loadingMore}>
                {loadingMore ? t('common.loading') : t('generate.catalogLoadMore')}
              </button>
            )}
          </>
        ) : visibleSources.length === 0 ? (
          <div className="text-sm text-fg-tertiary p-8 text-center">{t('generate.catalogSourcesEmpty')}</div>
        ) : (
          <div
            className="grid gap-2"
            style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))' }}
            data-testid="lora-catalog-sources"
          >
            {visibleSources.map((source) => (
              <button
                type="button"
                key={source.source_id}
                className="rounded-md border border-subtle bg-sunken hover:bg-overlay p-3 text-left cursor-pointer transition-colors min-w-0"
                onClick={() => enterSource(source)}
                data-testid="lora-catalog-source"
              >
                <span className="block text-sm font-medium text-fg-primary truncate">{sourceDisplayName(source)}</span>
                <span className="block text-2xs text-fg-tertiary mt-1 truncate">
                  {source.source_type === 'project' ? t('generate.catalogProject') : source.path}
                </span>
                <span className={`block text-xs mt-2 ${source.error ? 'text-warn' : 'text-fg-secondary'}`}>
                  {source.error || t('generate.catalogItemCount', { count: source.item_count })}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>
    </aside>
  )
}
