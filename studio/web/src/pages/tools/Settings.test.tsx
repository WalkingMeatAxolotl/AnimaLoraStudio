import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ToastProvider } from '../../components/Toast'
import SettingsPage from './Settings'

const initialServerState = {
  gelbooru: {
    user_id: 'alice',
    api_key: '***', // 已保存，掩码
    save_tags: false,
    convert_to_png: true,
    remove_alpha_channel: false,
  },
  danbooru: { username: '', api_key: '' },
  download: { exclude_tags: [] },
  huggingface: { token: '' },
  joycaption: {
    base_url: 'http://localhost:8000/v1',
    model: 'fancyfeast/llama-joycaption-beta-one-hf-llava',
    prompt_template: 'Descriptive Caption',
  },
  wd14: {
    model_id: 'SmilingWolf/wd-eva02-large-tagger-v3',
    model_ids: [
      'SmilingWolf/wd-eva02-large-tagger-v3',
      'SmilingWolf/wd-vit-tagger-v3',
      'SmilingWolf/wd-vit-large-tagger-v3',
      'SmilingWolf/wd-v1-4-convnext-tagger-v2',
    ],
    local_dir: null,
    threshold_general: 0.35,
    threshold_character: 0.85,
    blacklist_tags: [],
  },
}

const fetchMock = vi.fn()

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockReset()
  fetchMock.mockImplementation((_url: string, init?: RequestInit) => {
    if (init?.method === 'PUT') {
      const body = JSON.parse(String(init.body)) as Record<
        string,
        Record<string, unknown>
      >
      const merged = JSON.parse(JSON.stringify(initialServerState))
      for (const k of Object.keys(body)) {
        Object.assign(merged[k], body[k])
      }
      return Promise.resolve(
        new Response(JSON.stringify(merged), { status: 200 })
      )
    }
    return Promise.resolve(
      new Response(JSON.stringify(initialServerState), { status: 200 })
    )
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

function renderPage() {
  return render(
    <ToastProvider>
      <SettingsPage />
    </ToastProvider>
  )
}

describe('SettingsPage (PP0)', () => {
  it('hydrates from /api/secrets and shows masked sensitive fields as placeholder', async () => {
    renderPage()
    await waitFor(() =>
      expect(screen.getByDisplayValue('alice')).toBeInTheDocument()
    )
    // api_key 是 password input，placeholder 提示「已保存」
    const placeholder = screen.getByPlaceholderText(/已保存/)
    expect(placeholder).toBeInTheDocument()
    expect((placeholder as HTMLInputElement).value).toBe('')
  })

  it('PUT /api/secrets only sends the changed leaves', async () => {
    const user = userEvent.setup()
    renderPage()
    const userInput = await screen.findByDisplayValue('alice')
    await user.clear(userInput)
    await user.type(userInput, 'bob')

    const saveBtn = screen.getByRole('button', { name: /保存/ })
    await user.click(saveBtn)

    await waitFor(() => {
      const putCall = fetchMock.mock.calls.find(
        ([, init]) => init?.method === 'PUT'
      )
      expect(putCall).toBeDefined()
      const body = JSON.parse(String(putCall![1].body))
      // 只有 user_id 被改动；api_key 仍是 *** ⇒ 不应该出现在 body 里
      expect(body).toEqual({ gelbooru: { user_id: 'bob' } })
    })
  })
})
