import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type GalleryItem } from '../../../api/client'
import { ToastProvider } from '../../../components/Toast'
import GalleryPickerDrawer from './GalleryPickerDrawer'

const ITEM: GalleryItem = {
  source: 'danbooru',
  post_id: '42',
  width: 800,
  height: 1200,
  tags: ['1girl', 'blue_hair'],
  thumbnail_url: '/api/gallery/image?source=danbooru&url=x',
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
  vi.spyOn(api, 'searchGallery').mockResolvedValue({
    items: [ITEM], page: 1, page_size: 30, has_more: true,
  })
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
    await user.click(screen.getByRole('button', { name: '打标' }))

    await waitFor(() => expect(api.tagGalleryImage).toHaveBeenCalledWith({
      source: 'danbooru',
      post_id: '42',
      image_url: 'https://cdn.donmai.us/sample.jpg',
      tagger: 'llm',
    }))
    expect(onApplyPrompt).toHaveBeenCalledWith('tagged, prompt')
  })

  it('maps rating/date/search/page controls and resets selection on navigation', async () => {
    const user = userEvent.setup()
    setup()
    const card = await screen.findByRole('button', { name: '选择图片 #42' })
    await user.click(card)

    await user.type(screen.getByRole('searchbox', { name: '搜索画廊' }), 'cat ears')
    await user.selectOptions(screen.getByRole('combobox', { name: '分级过滤' }), 'sensitive')
    await user.type(screen.getByLabelText('开始日期'), '2025-01-02')
    await user.type(screen.getByLabelText('结束日期'), '2025-02-03')
    await user.click(screen.getByRole('button', { name: '刷新' }))

    await waitFor(() => expect(api.searchGallery).toHaveBeenLastCalledWith({
      source: 'danbooru',
      query: 'cat ears',
      rating: 'sensitive',
      dateFrom: '2025-01-02',
      dateTo: '2025-02-03',
      page: 1,
    }, expect.any(AbortSignal)))
    expect(screen.getByRole('button', { name: '打标' })).toBeDisabled()

    await user.click(screen.getByRole('button', { name: '下一页' }))
    await waitFor(() => expect(api.searchGallery).toHaveBeenLastCalledWith(
      expect.objectContaining({ page: 2 }), expect.any(AbortSignal),
    ))
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
