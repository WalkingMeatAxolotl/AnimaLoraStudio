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
  models: { root: null, selected_anima: 'preview3-base' },
}

const emptyModelsCatalog = {
  models_root: '/tmp/anima',
  anima_main: {
    id: 'anima_main',
    name: 'Anima 主模型',
    description: 'test',
    repo: 'circlestone-labs/Anima',
    variants: [],
    latest: 'preview3-base',
  },
  anima_vae: {
    id: 'anima_vae',
    name: 'VAE',
    description: 'test',
    repo: 'circlestone-labs/Anima',
    target_path: '/tmp/anima/vae/x.safetensors',
    exists: false,
    size: 0,
    mtime: 0,
  },
  qwen3: {
    id: 'qwen3',
    name: 'Qwen3',
    description: 'test',
    repo: 'Qwen/Qwen3-0.6B-Base',
    target_dir: '/tmp/anima/text_encoders',
    files: [],
  },
  t5_tokenizer: {
    id: 't5_tokenizer',
    name: 'T5',
    description: 'test',
    repo: 'google/t5-v1_1-xxl',
    target_dir: '/tmp/anima/t5_tokenizer',
    files: [],
  },
  downloads: {},
}

const fetchMock = vi.fn()

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockReset()
  fetchMock.mockImplementation((url: string, init?: RequestInit) => {
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
    if (typeof url === 'string' && url.includes('/api/models/catalog')) {
      return Promise.resolve(
        new Response(JSON.stringify(emptyModelsCatalog), { status: 200 })
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

    // 主表单 Save 按钮文案就是「保存」；Models 区块的「保存路径」按钮
    // 也含「保存」字样，正则匹配会撞 → 用精确名定位主按钮。
    const saveBtn = screen.getByRole('button', { name: '保存' })
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
