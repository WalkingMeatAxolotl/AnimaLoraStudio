import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type GalleryItem, type GalleryRating, type GallerySource, type GalleryTagger } from '../../../api/client'
import { TagSuggestList } from '../../../components/tagSuggest/TagSuggestList'
import { useTagSuggest } from '../../../components/tagSuggest/useTagSuggest'
import { useOptionalToast } from '../../../components/Toast'
import { useLocalStorageState } from '../../../lib/useLocalStorageState'
import GenerateAttachedDrawer from './GenerateAttachedDrawer'

const SOURCE_KEY = 'studio:generate:gallery:source'
const TAGGER_KEY = 'studio:generate:gallery:tagger'
const QUERY_KEY = 'studio:generate:gallery:query'
const RATING_KEY = 'studio:generate:gallery:rating'
const DATE_FROM_KEY = 'studio:generate:gallery:dateFrom'
const DATE_TO_KEY = 'studio:generate:gallery:dateTo'
const PAGE_KEY = 'studio:generate:gallery:page'
const AUTO_GENERATE_KEY = 'studio:generate:gallery:autoGenerate'
const MAX_PAGE = 10_000
const SEARCH_SUGGESTIONS_ID = 'gallery-search-tag-suggestions'
const ALL_RATINGS: GalleryRating[] = ['general', 'sensitive', 'questionable', 'explicit']
const DEFAULT_RATINGS: GalleryRating[] = ['general']

type StoredRatings = GalleryRating | GalleryRating[]

function normalizeRatings(value: StoredRatings): GalleryRating[] {
  const selected = new Set(Array.isArray(value) ? value : [value])
  const normalized = ALL_RATINGS.filter((rating) => selected.has(rating))
  return normalized.length > 0 ? normalized : [...DEFAULT_RATINGS]
}

function sameRatings(left: GalleryRating[], right: GalleryRating[]): boolean {
  return left.length === right.length && left.every((rating, index) => rating === right[index])
}

type SearchState = {
  items: GalleryItem[]
  page: number
  hasMore: boolean
}

const EMPTY_RESULT: SearchState = { items: [], page: 1, hasMore: false }

function GalleryPickerDrawer({
  open = true,
  onApplyPrompt,
  onClose,
}: {
  open?: boolean
  onApplyPrompt: (prompt: string, autoGenerate: boolean) => void | Promise<void>
  onClose: () => void
}) {
  const { t } = useTranslation()
  const { toast } = useOptionalToast()
  const [source, setSource] = useLocalStorageState<GallerySource>(SOURCE_KEY, 'danbooru')
  const [tagger, setTagger] = useLocalStorageState<GalleryTagger>(TAGGER_KEY, 'wd14')
  const [query, setQuery] = useLocalStorageState(QUERY_KEY, '')
  const [storedRatings, setStoredRatings] = useLocalStorageState<StoredRatings>(RATING_KEY, DEFAULT_RATINGS)
  const ratings = useMemo(() => normalizeRatings(storedRatings), [storedRatings])
  const [dateFrom, setDateFrom] = useLocalStorageState(DATE_FROM_KEY, '')
  const [dateTo, setDateTo] = useLocalStorageState(DATE_TO_KEY, '')
  const [page, setPage] = useLocalStorageState(PAGE_KEY, 1)
  const [autoGenerate, setAutoGenerate] = useLocalStorageState(AUTO_GENERATE_KEY, false)
  const [submittedQuery, setSubmittedQuery] = useState(() => query.trim())
  const [pageInput, setPageInput] = useState(() => String(page))
  const [ratingDraft, setRatingDraft] = useState<GalleryRating[]>(() => [...ratings])
  const [ratingOpen, setRatingOpen] = useState(false)
  const [dateDraft, setDateDraft] = useState(() => ({ from: dateFrom, to: dateTo }))
  const [timeOpen, setTimeOpen] = useState(false)
  const searchInputRef = useRef<HTMLInputElement | null>(null)
  const ratingWrapRef = useRef<HTMLDivElement | null>(null)
  const timeWrapRef = useRef<HTMLDivElement | null>(null)
  const [refreshToken, setRefreshToken] = useState(0)
  const [result, setResult] = useState<SearchState>(EMPTY_RESULT)
  const [selected, setSelected] = useState<GalleryItem | null>(null)
  const [loading, setLoading] = useState(false)
  const [tagging, setTagging] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const searchSuggest = useTagSuggest({
    value: query,
    inputRef: searchInputRef,
    tokenMode: 'whitespace',
    disabled: tagging || !open,
    onPick: ({ suggestion, range }) => {
      const before = query.slice(0, range.start)
      const current = query.slice(range.start, range.end).trim()
      const modifier = current.match(/^[-~]/)?.[0] ?? ''
      const tag = `${modifier}${suggestion.tag.replace(/\s+/g, '_')}`
      const cleanAfter = query.slice(range.end).replace(/^\s+/, '')
      const next = `${before}${tag}${cleanAfter ? ` ${cleanAfter}` : ' '}`
      setQuery(next)
      const newCursor = before.length + tag.length + 1
      requestAnimationFrame(() => {
        const input = searchInputRef.current
        if (input) { input.focus(); input.setSelectionRange(newCursor, newCursor) }
      })
    },
  })

  useEffect(() => {
    if (!Array.isArray(storedRatings) || !sameRatings(storedRatings, ratings)) {
      try {
        localStorage.setItem(RATING_KEY, JSON.stringify(ratings))
      } catch { /* localStorage unavailable: keep the normalized in-memory value */ }
    }
  }, [ratings, storedRatings])

  useEffect(() => {
    if (!ratingOpen) setRatingDraft([...ratings])
  }, [ratingOpen, ratings])

  useEffect(() => {
    if (!timeOpen) setDateDraft({ from: dateFrom, to: dateTo })
  }, [dateFrom, dateTo, timeOpen])

  useEffect(() => {
    setPageInput(String(page))
  }, [page])

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
      ratings,
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
  }, [dateFrom, dateTo, page, ratings, refreshToken, source, submittedQuery, t])

  const commitRatingFilter = useCallback(() => {
    const next = normalizeRatings(ratingDraft)
    setRatingOpen(false)
    if (!sameRatings(next, ratings)) {
      setStoredRatings(next)
      setPage(1)
    }
  }, [ratingDraft, ratings, setPage, setStoredRatings])

  const commitTimeFilter = useCallback(() => {
    setTimeOpen(false)
    if (dateDraft.from && dateDraft.to && dateDraft.from > dateDraft.to) {
      setError(t('generate.galleryDateRangeInvalid'))
      return
    }
    if (dateDraft.from !== dateFrom || dateDraft.to !== dateTo) {
      setDateFrom(dateDraft.from)
      setDateTo(dateDraft.to)
      setPage(1)
    }
  }, [dateDraft, dateFrom, dateTo, setDateFrom, setDateTo, setPage, t])

  useEffect(() => {
    if (open) return
    setRatingOpen(false)
    setTimeOpen(false)
  }, [open])

  useEffect(() => {
    if (!open || (!ratingOpen && !timeOpen)) return
    const onMouseDown = (event: MouseEvent) => {
      const target = event.target as Node
      if (ratingOpen && !ratingWrapRef.current?.contains(target)) commitRatingFilter()
      if (timeOpen && !timeWrapRef.current?.contains(target)) commitTimeFilter()
    }
    document.addEventListener('mousedown', onMouseDown)
    return () => document.removeEventListener('mousedown', onMouseDown)
  }, [commitRatingFilter, commitTimeFilter, open, ratingOpen, timeOpen])

  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      if (ratingOpen) commitRatingFilter()
      else if (timeOpen) commitTimeFilter()
      else onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [commitRatingFilter, commitTimeFilter, onClose, open, ratingOpen, timeOpen])

  const runSearch = () => {
    setPage(1)
    setSubmittedQuery(query.trim())
    setRefreshToken((value) => value + 1)
  }

  const updateSource = (next: GallerySource) => {
    setSource(next)
    setPage(1)
  }

  const toggleRatingDraft = (rating: GalleryRating) => {
    setRatingDraft((current) => {
      if (current.includes(rating)) {
        return current.length === 1 ? current : current.filter((value) => value !== rating)
      }
      return ALL_RATINGS.filter((value) => value === rating || current.includes(value))
    })
  }

  const toggleRatingFilter = () => {
    if (ratingOpen) {
      commitRatingFilter()
      return
    }
    setRatingDraft([...ratings])
    setRatingOpen(true)
  }

  const toggleTimeFilter = () => {
    if (timeOpen) {
      commitTimeFilter()
      return
    }
    setDateDraft({ from: dateFrom, to: dateTo })
    setTimeOpen(true)
  }

  const jumpToPage = () => {
    const parsed = Number(pageInput)
    if (!Number.isInteger(parsed) || parsed < 1 || parsed > MAX_PAGE) {
      setPageInput(String(page))
      setError(t('generate.galleryPageInvalid', { max: MAX_PAGE }))
      return
    }
    setError(null)
    setPage(parsed)
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
      await onApplyPrompt(response.prompt, autoGenerate)
      toast(t('generate.galleryTagSuccess'), 'success')
    } catch (reason) {
      setError(String(reason))
    } finally {
      setTagging(false)
    }
  }

  const ratingOptions: Array<{ value: GalleryRating; label: string }> = [
    { value: 'general', label: t('generate.galleryRatingGeneral') },
    { value: 'sensitive', label: t('generate.galleryRatingSensitive') },
    { value: 'questionable', label: t('generate.galleryRatingQuestionable') },
    { value: 'explicit', label: t('generate.galleryRatingExplicit') },
  ]
  const timeFilterActive = Boolean(dateFrom || dateTo)

  return (
    <GenerateAttachedDrawer
      id="prompt-gallery-drawer"
      ariaLabel={t('generate.galleryTitle')}
      testId="prompt-gallery-drawer"
      open={open}
    >
      <div className="flex h-full min-h-0 flex-col overflow-hidden" data-testid="gallery-picker">
      <header className="relative z-20 flex shrink-0 flex-col gap-2 border-b border-subtle p-3">
        <div className="flex flex-wrap items-center gap-2">
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
            role="switch"
            aria-checked={autoGenerate}
            aria-label={t('generate.galleryAutoGenerate')}
            title={t('generate.galleryAutoGenerateHint')}
            className="btn btn-ghost btn-sm shrink-0 gap-1 px-1.5 text-2xs"
            disabled={tagging}
            onClick={() => setAutoGenerate((value) => !value)}
          >
            <span
              className="relative inline-flex h-4 w-7 shrink-0 rounded-full border border-subtle transition-colors"
              style={{ background: autoGenerate ? 'var(--accent)' : 'var(--bg-overlay)' }}
              aria-hidden="true"
            >
              <span
                className="absolute top-0.5 h-2.5 w-2.5 rounded-full bg-white shadow-sm transition-transform"
                style={{ transform: `translateX(${autoGenerate ? 13 : 2}px)` }}
              />
            </span>
            {t('generate.galleryAutoGenerate')}
          </button>
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
          className="grid grid-cols-[minmax(0,1fr)_auto_auto] gap-2 xl:grid-cols-[minmax(150px,1fr)_auto_auto_auto]"
          onSubmit={(event) => { event.preventDefault(); runSearch() }}
        >
          <div className="col-span-3 min-w-0 xl:col-span-1">
            <input
              ref={searchInputRef}
              type="search"
              className="input w-full min-w-0 text-xs"
              value={query}
              disabled={tagging}
              onChange={(event) => { setQuery(event.target.value); searchSuggest.notifyChange() }}
              onKeyDown={(event) => { searchSuggest.handleKeyDown(event) }}
              onKeyUp={() => searchSuggest.notifySelect()}
              onClick={() => searchSuggest.notifyClick()}
              onFocus={() => searchSuggest.notifyFocus()}
              onBlur={() => searchSuggest.notifyBlur()}
              placeholder={t('generate.gallerySearchPlaceholder')}
              aria-label={t('generate.gallerySearch')}
              aria-autocomplete="list"
              aria-expanded={searchSuggest.open && searchSuggest.suggestions.length > 0}
              aria-controls={searchSuggest.open && searchSuggest.suggestions.length > 0
                ? SEARCH_SUGGESTIONS_ID
                : undefined}
              aria-activedescendant={searchSuggest.open && searchSuggest.suggestions.length > 0
                ? `${SEARCH_SUGGESTIONS_ID}-option-${searchSuggest.activeIdx}`
                : undefined}
            />
            <TagSuggestList
              id={SEARCH_SUGGESTIONS_ID}
              open={searchSuggest.open}
              suggestions={searchSuggest.suggestions}
              activeIdx={searchSuggest.activeIdx}
              onPick={(suggestion) => searchSuggest.pickAt(searchSuggest.suggestions.indexOf(suggestion))}
              onHover={searchSuggest.setActiveIdx}
              inputRef={searchInputRef}
              cursor={searchSuggest.cursor}
              positionDeps={[query]}
            />
          </div>

          <div ref={ratingWrapRef} className="relative">
            <button
              type="button"
              className="btn btn-secondary btn-sm w-full shrink-0 justify-center gap-1.5"
              disabled={tagging}
              aria-label={t('generate.galleryRating')}
              aria-haspopup="dialog"
              aria-expanded={ratingOpen}
              onClick={toggleRatingFilter}
            >
              <span>{t('generate.galleryRating')}</span>
              <span className="rounded-full bg-overlay px-1.5 font-mono text-2xs text-fg-tertiary">
                {ratings.length}
              </span>
              <span aria-hidden="true">{ratingOpen ? '▴' : '▾'}</span>
            </button>
            {ratingOpen && (
              <div
                role="dialog"
                aria-label={t('generate.galleryRating')}
                className="absolute left-0 top-full z-40 mt-1 min-w-[210px] rounded-md border border-subtle bg-elevated shadow-xl"
              >
                <div className="flex flex-col py-1">
                  {ratingOptions.map((option) => {
                    const checked = ratingDraft.includes(option.value)
                    return (
                      <label
                        key={option.value}
                        className="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-xs hover:bg-overlay"
                      >
                        <input
                          type="checkbox"
                          checked={checked}
                          disabled={checked && ratingDraft.length === 1}
                          onChange={() => toggleRatingDraft(option.value)}
                        />
                        <span className="text-fg-secondary">{option.label}</span>
                      </label>
                    )
                  })}
                </div>
                <div className="flex items-center gap-2 border-t border-subtle px-2.5 py-1.5">
                  <span className="flex-1 text-2xs text-fg-tertiary">
                    {t('generate.galleryFilterApplyOnClose')}
                  </span>
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm px-2"
                    onClick={commitRatingFilter}
                  >
                    {t('generate.galleryFilterDone')}
                  </button>
                </div>
              </div>
            )}
          </div>

          <div ref={timeWrapRef} className="relative">
            <button
              type="button"
              className="btn btn-secondary btn-sm w-full shrink-0 justify-center gap-1.5"
              disabled={tagging}
              aria-label={timeFilterActive
                ? t('generate.galleryTimeFilterActive')
                : t('generate.galleryTimeFilter')}
              aria-haspopup="dialog"
              aria-expanded={timeOpen}
              data-active={timeFilterActive ? 'true' : 'false'}
              onClick={toggleTimeFilter}
            >
              <span>{t('generate.galleryTimeFilter')}</span>
              {timeFilterActive && <span className="dot dot-err shrink-0" aria-hidden="true" />}
              <span aria-hidden="true">{timeOpen ? '▴' : '▾'}</span>
            </button>
            {timeOpen && (
              <div
                role="dialog"
                aria-label={t('generate.galleryTimeFilter')}
                className="absolute right-0 top-full z-40 mt-1 w-[min(280px,calc(100vw-32px))] rounded-md border border-subtle bg-elevated shadow-xl"
              >
                <div className="grid gap-2 p-3">
                  <label className="grid gap-1 text-2xs text-fg-tertiary">
                    <span>{t('generate.galleryDateFrom')}</span>
                    <input
                      type="date"
                      className="input min-w-0 text-xs"
                      value={dateDraft.from}
                      disabled={tagging}
                      max={dateDraft.to || undefined}
                      onChange={(event) => setDateDraft((current) => ({ ...current, from: event.target.value }))}
                      aria-label={t('generate.galleryDateFrom')}
                    />
                  </label>
                  <label className="grid gap-1 text-2xs text-fg-tertiary">
                    <span>{t('generate.galleryDateTo')}</span>
                    <input
                      type="date"
                      className="input min-w-0 text-xs"
                      value={dateDraft.to}
                      disabled={tagging}
                      min={dateDraft.from || undefined}
                      onChange={(event) => setDateDraft((current) => ({ ...current, to: event.target.value }))}
                      aria-label={t('generate.galleryDateTo')}
                    />
                  </label>
                </div>
                <div className="flex items-center justify-end gap-2 border-t border-subtle px-2.5 py-1.5">
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm px-2"
                    onClick={() => setDateDraft({ from: '', to: '' })}
                  >
                    {t('generate.galleryFilterClear')}
                  </button>
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm px-2"
                    onClick={commitTimeFilter}
                  >
                    {t('generate.galleryFilterDone')}
                  </button>
                </div>
              </div>
            )}
          </div>

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
        <form
          className="flex min-w-0 items-center justify-center gap-1"
          onSubmit={(event) => { event.preventDefault(); jumpToPage() }}
        >
          <input
            type="number"
            className="input w-16 px-1.5 py-1 text-center text-xs"
            min={1}
            max={MAX_PAGE}
            step={1}
            value={pageInput}
            disabled={loading || tagging}
            onChange={(event) => setPageInput(event.target.value)}
            aria-label={t('generate.galleryPageInput')}
            title={t('generate.galleryPage', { page })}
          />
          <button
            type="submit"
            className="btn btn-ghost btn-sm px-2"
            disabled={loading || tagging}
          >
            {t('generate.galleryJump')}
          </button>
        </form>
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
    </GenerateAttachedDrawer>
  )
}

export default memo(
  GalleryPickerDrawer,
  (previous, next) => previous.open === false && next.open === false,
)
