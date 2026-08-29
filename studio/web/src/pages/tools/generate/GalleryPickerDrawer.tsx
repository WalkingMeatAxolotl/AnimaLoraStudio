import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type GalleryItem, type GalleryRating, type GallerySource, type GalleryTagger } from '../../../api/client'
import { useOptionalToast } from '../../../components/Toast'
import { useLocalStorageState } from '../../../lib/useLocalStorageState'

const SOURCE_KEY = 'studio:generate:gallery:source'
const TAGGER_KEY = 'studio:generate:gallery:tagger'

type SearchState = {
  items: GalleryItem[]
  page: number
  hasMore: boolean
}

const EMPTY_RESULT: SearchState = { items: [], page: 1, hasMore: false }

export default function GalleryPickerDrawer({
  onApplyPrompt,
  onClose,
}: {
  onApplyPrompt: (prompt: string) => void
  onClose: () => void
}) {
  const { t } = useTranslation()
  const { toast } = useOptionalToast()
  const [source, setSource] = useLocalStorageState<GallerySource>(SOURCE_KEY, 'danbooru')
  const [tagger, setTagger] = useLocalStorageState<GalleryTagger>(TAGGER_KEY, 'wd14')
  const [query, setQuery] = useState('')
  const [submittedQuery, setSubmittedQuery] = useState('')
  const [rating, setRating] = useState<GalleryRating>('general')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [page, setPage] = useState(1)
  const [refreshToken, setRefreshToken] = useState(0)
  const [result, setResult] = useState<SearchState>(EMPTY_RESULT)
  const [selected, setSelected] = useState<GalleryItem | null>(null)
  const [loading, setLoading] = useState(false)
  const [tagging, setTagging] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (dateFrom && dateTo && dateFrom > dateTo) {
      setError(t('generate.galleryDateRangeInvalid'))
      setResult(EMPTY_RESULT)
      return
    }
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    setSelected(null)
    void api.searchGallery({
      source,
      query: submittedQuery,
      rating,
      dateFrom: dateFrom || undefined,
      dateTo: dateTo || undefined,
      page,
    }, controller.signal).then((response) => {
      setResult({ items: response.items, page: response.page, hasMore: response.has_more })
    }).catch((reason: unknown) => {
      if (reason instanceof DOMException && reason.name === 'AbortError') return
      setResult(EMPTY_RESULT)
      setError(String(reason))
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })
    return () => controller.abort()
  }, [dateFrom, dateTo, page, rating, refreshToken, source, submittedQuery, t])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  const runSearch = () => {
    setPage(1)
    setSubmittedQuery(query.trim())
    setRefreshToken((value) => value + 1)
  }

  const updateSource = (next: GallerySource) => {
    setSource(next)
    setPage(1)
  }

  const updateRating = (next: GalleryRating) => {
    setRating(next)
    setPage(1)
  }

  const updateDateFrom = (next: string) => {
    setDateFrom(next)
    setPage(1)
  }

  const updateDateTo = (next: string) => {
    setDateTo(next)
    setPage(1)
  }

  const tagSelected = async () => {
    if (!selected || tagging) return
    setTagging(true)
    setError(null)
    try {
      const response = await api.tagGalleryImage({
        source: selected.source,
        post_id: selected.post_id,
        image_url: selected.image_url,
        tagger,
      })
      onApplyPrompt(response.prompt)
      toast(t('generate.galleryTagSuccess'), 'success')
    } catch (reason) {
      const message = String(reason)
      setError(message)
      toast(message, 'error')
    } finally {
      setTagging(false)
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden" data-testid="gallery-picker">
      <header className="flex shrink-0 flex-col gap-2 border-b border-subtle p-3">
        <div className="flex items-center gap-2">
          <select
            className="input min-w-0 flex-1 text-xs"
            value={source}
            disabled={tagging}
            onChange={(event) => updateSource(event.target.value as GallerySource)}
            aria-label={t('generate.gallerySource')}
            title={t('generate.galleryUsesGlobalSettings')}
          >
            <option value="danbooru">Danbooru</option>
            <option value="gelbooru">Gelbooru</option>
          </select>
          <select
            className="input min-w-0 flex-1 text-xs"
            value={tagger}
            disabled={tagging}
            onChange={(event) => setTagger(event.target.value as GalleryTagger)}
            aria-label={t('generate.galleryTagger')}
            title={t('generate.galleryUsesGlobalSettings')}
          >
            <option value="wd14">WD14</option>
            <option value="cltagger">CLTagger</option>
            <option value="llm">LLM</option>
          </select>
          <button
            type="button"
            className="btn btn-primary btn-sm shrink-0"
            disabled={!selected || tagging}
            onClick={() => void tagSelected()}
          >
            {tagging ? t('generate.galleryTagging') : t('generate.galleryTag')}
          </button>
          <button
            type="button"
            className="btn btn-ghost btn-sm shrink-0 px-1.5 text-fg-tertiary"
            onClick={onClose}
            title={t('generate.closeGalleryPicker')}
            aria-label={t('common.close')}
          >
            ×
          </button>
        </div>

        <form
          className="grid grid-cols-2 gap-2 xl:grid-cols-[minmax(150px,1fr)_auto_auto_auto_auto]"
          onSubmit={(event) => { event.preventDefault(); runSearch() }}
        >
          <input
            type="search"
            className="input col-span-2 min-w-0 text-xs xl:col-span-1"
            value={query}
            disabled={tagging}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={t('generate.gallerySearchPlaceholder')}
            aria-label={t('generate.gallerySearch')}
          />
          <select
            className="input min-w-0 text-xs"
            value={rating}
            disabled={tagging}
            onChange={(event) => updateRating(event.target.value as GalleryRating)}
            aria-label={t('generate.galleryRating')}
          >
            <option value="general">{t('generate.galleryRatingGeneral')}</option>
            <option value="sensitive">{t('generate.galleryRatingSensitive')}</option>
            <option value="questionable">{t('generate.galleryRatingQuestionable')}</option>
            <option value="explicit">{t('generate.galleryRatingExplicit')}</option>
          </select>
          <input
            type="date"
            className="input min-w-0 text-xs"
            value={dateFrom}
            disabled={tagging}
            max={dateTo || undefined}
            onChange={(event) => updateDateFrom(event.target.value)}
            aria-label={t('generate.galleryDateFrom')}
            title={t('generate.galleryDateFrom')}
          />
          <input
            type="date"
            className="input min-w-0 text-xs"
            value={dateTo}
            disabled={tagging}
            min={dateFrom || undefined}
            onChange={(event) => updateDateTo(event.target.value)}
            aria-label={t('generate.galleryDateTo')}
            title={t('generate.galleryDateTo')}
          />
          <button type="submit" className="btn btn-secondary btn-sm shrink-0" disabled={loading || tagging}>
            {t('generate.galleryRefresh')}
          </button>
        </form>
      </header>

      {error && (
        <div className="shrink-0 border-b border-subtle bg-err-soft px-3 py-2 text-2xs text-err" role="alert">
          {error}
        </div>
      )}

      <div className="relative min-h-0 flex-1 overflow-y-auto bg-sunken p-2" data-testid="gallery-image-list">
        {loading && (
          <div className="sticky top-0 z-10 mb-2 rounded bg-elevated/95 px-3 py-2 text-center text-2xs text-fg-tertiary shadow-sm">
            {t('common.loading')}
          </div>
        )}
        {!loading && !error && result.items.length === 0 && (
          <div className="grid h-full place-items-center px-4 text-center text-xs text-fg-tertiary">
            {t('generate.galleryEmpty')}
          </div>
        )}
        <div className="gallery-waterfall" aria-busy={loading}>
          {result.items.map((item) => {
            const active = selected?.source === item.source && selected.post_id === item.post_id
            return (
              <button
                type="button"
                key={`${item.source}:${item.post_id}`}
                className="gallery-waterfall-card relative mb-2 block w-full overflow-hidden rounded-md border bg-overlay p-0 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                style={{
                  borderColor: active ? 'var(--accent)' : 'var(--border-subtle)',
                  boxShadow: active ? '0 0 0 1px var(--accent)' : undefined,
                }}
                aria-pressed={active}
                disabled={loading || tagging}
                aria-label={t('generate.galleryImageAria', { id: item.post_id })}
                data-selected={active ? 'true' : 'false'}
                onClick={() => setSelected(active ? null : item)}
              >
                <div className="relative w-full bg-overlay" style={{ aspectRatio: `${item.width} / ${item.height}` }}>
                  <img
                    src={item.thumbnail_url}
                    alt={item.tags.slice(0, 5).join(', ') || t('generate.galleryImageAria', { id: item.post_id })}
                    loading="lazy"
                    width={item.width}
                    height={item.height}
                    className="absolute inset-0 h-full w-full object-cover"
                  />
                  {active && (
                    <span className="absolute inset-0 grid place-items-center bg-black/20" aria-hidden="true">
                      <span className="grid h-9 w-9 place-items-center rounded-full bg-accent text-lg font-bold text-white shadow-lg">✓</span>
                    </span>
                  )}
                </div>
                <span className="block truncate px-2 py-1 text-2xs text-fg-tertiary">#{item.post_id}</span>
              </button>
            )
          })}
        </div>
      </div>

      <footer className="flex shrink-0 items-center justify-between gap-2 border-t border-subtle px-3 py-2">
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          disabled={loading || tagging || page <= 1}
          onClick={() => setPage((value) => Math.max(1, value - 1))}
        >
          {t('generate.galleryPrevious')}
        </button>
        <span className="text-2xs text-fg-tertiary">{t('generate.galleryPage', { page: result.page || page })}</span>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          disabled={loading || tagging || !result.hasMore}
          onClick={() => setPage((value) => value + 1)}
        >
          {t('generate.galleryNext')}
        </button>
      </footer>
    </div>
  )
}
