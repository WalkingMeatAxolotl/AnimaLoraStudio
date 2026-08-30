import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type GalleryItem } from '../../../api/client'
import { ToastProvider } from '../../../components/Toast'
import { __setTagPrefsForTest } from '../../../tagDict/prefs'
import { __setStateForTest } from '../../../tagDict/store'
import GalleryPickerDrawer from './GalleryPickerDrawer'

const ITEM: GalleryItem = {
  source: 'danbooru',
  post_id: '42',
  width: 800,
  height: 1200,
  tags: ['1girl', 'blue_hair'],
  thumbnail_url: '/api/gallery/image?source=danbooru&post_id=42&url=x',
  image_url: 'https://cdn.donmai.us/sample.jpg',
}

function setup(onApplyPrompt = vi.fn()) {
  render(
    <ToastProvider>
      <GalleryPickerDrawer onApplyPrompt={onApplyPrompt} onClose={vi.fn()} />
    </ToastProvider>,
  )
  return onApplyPrompt
}

beforeEach(() => {
  window.localStorage.clear()
  __setTagPrefsForTest({ loaded: true, autocomplete: true })
  const entries = new Map<string, string[]>([
    ['long hair', ['长发']],
    ['long sleeves', ['长袖']],
  ])
  const tagKeys = Array.from(entries.keys())
  __setStateForTest({
    status: 'ready',
    entries,
    tagKeys,
    compactedKeys: tagKeys.map((tag) => tag.replace(/[\s_]/g, '')),
    reverse: [{ zh: '长发', tags: ['long hair'] }, { zh: '长袖', tags: ['long sleeves'] }],
    meta: null,
    error: null,
  })
  vi.spyOn(api, 'searchGallery').mockImplementation(async (params) => ({
    items: [ITEM], page: params.page, page_size: 30, has_more: true,
  }))
  vi.spyOn(api, 'tagGalleryImage').mockResolvedValue({ prompt: 'tagged, prompt' })
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('GalleryPickerDrawer', () => {
  it('loads a proportional waterfall, single-selects, and tags with global method', async () => {
    const user = userEvent.setup()
    const onApplyPrompt = setup()

    const card = await screen.findByRole('button', { name: '选择图片 #42' })
    expect(card).toHaveAttribute('aria-pressed', 'false')
    expect(card.querySelector('img')).toHaveAttribute('width', '800')
    expect(card.querySelector('img')).toHaveAttribute('height', '1200')
    expect(screen.getByRole('button', { name: '打标' })).toBeDisabled()

    await user.click(card)
    expect(card).toHaveAttribute('aria-pressed', 'true')
    expect(card).toHaveTextContent('✓')
    await user.selectOptions(screen.getByRole('combobox', { name: '打标方式' }), 'llm')
    const autoGenerate = screen.getByRole('switch', { name: '自动生成' })
    const tagButton = screen.getByRole('button', { name: '打标' })
    expect(autoGenerate.compareDocumentPosition(tagButton) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    await user.click(autoGenerate)
    await user.click(tagButton)

    await waitFor(() => expect(api.tagGalleryImage).toHaveBeenCalledWith({
      source: 'danbooru',
      post_id: '42',
      image_url: 'https://cdn.donmai.us/sample.jpg',
      tagger: 'llm',
    }))
    expect(onApplyPrompt).toHaveBeenCalledWith('tagged, prompt', true)
    expect(window.localStorage.getItem('studio:generate:gallery:tagger')).toBe('"llm"')
    expect(window.localStorage.getItem('studio:generate:gallery:autoGenerate')).toBe('true')
  })

  it('reuses tag autocomplete and inserts Booru tags without submitting early', async () => {
    const user = userEvent.setup()
    setup()
    await screen.findByRole('button', { name: '选择图片 #42' })
    const search = screen.getByRole('searchbox', { name: '搜索画廊' })
    const callsBeforeTyping = vi.mocked(api.searchGallery).mock.calls.length

    await user.type(search, 'long_h')
    const option = await screen.findByRole('option', { name: /long hair/ })
    expect(option).toHaveAttribute('id', 'gallery-search-tag-suggestions-option-0')
    expect(search).toHaveAttribute('aria-expanded', 'true')
    expect(search).toHaveAttribute('aria-activedescendant', option.id)
    expect(api.searchGallery).toHaveBeenCalledTimes(callsBeforeTyping)

    await user.keyboard('{Enter}')
    expect(search).toHaveValue('long_hair ')
    expect(search).toHaveAttribute('aria-expanded', 'false')
    expect(api.searchGallery).toHaveBeenCalledTimes(callsBeforeTyping)

    await user.keyboard('{Enter}')
    await waitFor(() => expect(api.searchGallery).toHaveBeenLastCalledWith(
      expect.objectContaining({ query: 'long_hair' }), expect.any(AbortSignal),
    ))
  })

  it('commits multi-rating and date drafts only when their popovers close', async () => {
    const user = userEvent.setup()
    setup()
    const card = await screen.findByRole('button', { name: '选择图片 #42' })
    await user.click(card)

    const ratingButton = screen.getByRole('button', { name: '分级过滤' })
    await user.click(ratingButton)
    const callsBeforeRatingDraft = vi.mocked(api.searchGallery).mock.calls.length
    await user.click(screen.getByRole('checkbox', { name: '敏感' }))
    expect(api.searchGallery).toHaveBeenCalledTimes(callsBeforeRatingDraft)
    await user.click(ratingButton)
    await waitFor(() => expect(api.searchGallery).toHaveBeenLastCalledWith(
      expect.objectContaining({ ratings: ['general', 'sensitive'], page: 1 }),
      expect.any(AbortSignal),
    ))

    const timeButton = screen.getByRole('button', { name: '时间过滤' })
    await user.click(timeButton)
    const callsBeforeDateDraft = vi.mocked(api.searchGallery).mock.calls.length
    await user.type(screen.getByLabelText('开始日期'), '2025-01-02')
    await user.type(screen.getByLabelText('结束日期'), '2025-02-03')
    expect(api.searchGallery).toHaveBeenCalledTimes(callsBeforeDateDraft)
    await user.click(screen.getByRole('button', { name: '完成' }))
    await waitFor(() => expect(api.searchGallery).toHaveBeenLastCalledWith(
      expect.objectContaining({ dateFrom: '2025-01-02', dateTo: '2025-02-03', page: 1 }),
      expect.any(AbortSignal),
    ))
    expect(timeButton).toHaveAttribute('data-active', 'true')
    expect(timeButton).not.toHaveTextContent('2025')

    await user.type(screen.getByRole('searchbox', { name: '搜索画廊' }), 'cat ears')
    await user.click(screen.getByRole('button', { name: '刷新' }))

    await waitFor(() => expect(api.searchGallery).toHaveBeenLastCalledWith({
      source: 'danbooru',
      query: 'cat ears',
      ratings: ['general', 'sensitive'],
      dateFrom: '2025-01-02',
      dateTo: '2025-02-03',
      page: 1,
    }, expect.any(AbortSignal)))
    expect(screen.getByRole('button', { name: '打标' })).toBeDisabled()

    await user.click(screen.getByRole('button', { name: '下一页' }))
    await waitFor(() => expect(api.searchGallery).toHaveBeenLastCalledWith(
      expect.objectContaining({ page: 2 }), expect.any(AbortSignal),
    ))

    const pageInput = screen.getByRole('spinbutton', { name: '页码' })
    await user.clear(pageInput)
    await user.type(pageInput, '37')
    await user.click(screen.getByRole('button', { name: '跳转' }))
    await waitFor(() => expect(api.searchGallery).toHaveBeenLastCalledWith(
      expect.objectContaining({ page: 37 }), expect.any(AbortSignal),
    ))
    expect(window.localStorage.getItem('studio:generate:gallery:query')).toBe('"cat ears"')
    expect(window.localStorage.getItem('studio:generate:gallery:rating')).toBe('["general","sensitive"]')
    expect(window.localStorage.getItem('studio:generate:gallery:dateFrom')).toBe('"2025-01-02"')
    expect(window.localStorage.getItem('studio:generate:gallery:dateTo')).toBe('"2025-02-03"')
    expect(window.localStorage.getItem('studio:generate:gallery:page')).toBe('37')
  })

  it('restores persisted filters and browsing page', async () => {
    window.localStorage.setItem('studio:generate:gallery:source', '"gelbooru"')
    window.localStorage.setItem('studio:generate:gallery:tagger', '"cltagger"')
    window.localStorage.setItem('studio:generate:gallery:query', '"fox ears"')
    window.localStorage.setItem('studio:generate:gallery:rating', '"questionable"')
    window.localStorage.setItem('studio:generate:gallery:dateFrom', '"2025-03-01"')
    window.localStorage.setItem('studio:generate:gallery:dateTo', '"2025-03-31"')
    window.localStorage.setItem('studio:generate:gallery:page', '8')
    window.localStorage.setItem('studio:generate:gallery:autoGenerate', 'true')

    setup()

    await waitFor(() => expect(api.searchGallery).toHaveBeenLastCalledWith({
      source: 'gelbooru',
      query: 'fox ears',
      ratings: ['questionable'],
      dateFrom: '2025-03-01',
      dateTo: '2025-03-31',
      page: 8,
    }, expect.any(AbortSignal)))
    expect(screen.getByRole('combobox', { name: '图片来源' })).toHaveValue('gelbooru')
    expect(screen.getByRole('combobox', { name: '打标方式' })).toHaveValue('cltagger')
    expect(screen.getByRole('searchbox', { name: '搜索画廊' })).toHaveValue('fox ears')
    expect(screen.getByRole('button', { name: '时间过滤（已启用）' })).toHaveAttribute('data-active', 'true')
    expect(screen.queryByLabelText('开始日期')).not.toBeInTheDocument()
    expect(screen.getByRole('spinbutton', { name: '页码' })).toHaveValue(8)
    expect(screen.getByRole('switch', { name: '自动生成' })).toHaveAttribute('aria-checked', 'true')

    await userEvent.setup().click(screen.getByRole('button', { name: '分级过滤' }))
    expect(screen.getByRole('checkbox', { name: '存疑' })).toBeChecked()
    await waitFor(() => {
      expect(window.localStorage.getItem('studio:generate:gallery:rating')).toBe('["questionable"]')
      expect(api.searchGallery).toHaveBeenCalledTimes(1)
    })
  })

  it('does not overwrite the prompt when tagging fails', async () => {
    const user = userEvent.setup()
    vi.mocked(api.tagGalleryImage).mockRejectedValueOnce(new Error('tag failed'))
    const onApplyPrompt = setup()

    await user.click(await screen.findByRole('button', { name: '选择图片 #42' }))
    await user.click(screen.getByRole('button', { name: '打标' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('tag failed')
    expect(onApplyPrompt).not.toHaveBeenCalled()
  })
})
