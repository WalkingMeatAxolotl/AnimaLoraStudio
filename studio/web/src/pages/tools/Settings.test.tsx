import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import { DialogProvider } from '../../components/Dialog'
import { ToastProvider } from '../../components/Toast'
import { SettingsDataProvider } from '../../lib/SettingsData'
import { SettingsDrawerProvider } from '../../lib/SettingsDrawer'
import SettingsPage from './Settings'

const initialServerState = {
  gelbooru: {
    user_id: 'alice',
    api_key: '***', // 已保存，掩码
    save_tags: false,
    convert_to_png: true,
    remove_alpha_channel: false,
  },
  danbooru: { username: '', api_key: '', account_type: 'free' },
  download: {
    exclude_tags: [],
    parallel_workers: 4,
    api_rate_per_sec: 2,
    cdn_rate_per_sec: 5,
  },
  reg: { default_excluded_tags: [] },
  huggingface: { token: '', endpoint: '' },
  wandb: {
    enabled: false,
    api_key: '',
    project: 'AnimaLoraStudio',
    entity: '',
    base_url: '',
    mode: 'online',
    log_samples: true,
    sample_max_side: 1216,
    sample_every_n_steps: 0,
  },
  llm_tagger: {
    current_preset: 'style_json',
    presets: [
      {
        id: 'style_json',
        label: '画风 LoRA JSON',
        builtin: true,
        base_url: '',
        api_key: '',
        model: '',
        model_ids: [],
        endpoint: 'chat_completions',
        messages: [
          {
            type: 'text',
            role: 'system',
            content: 'Return JSON captions for anime style LoRA training.',
          },
          { type: 'image', role: 'user', content: '' },
        ],
        output_format: 'json',
        temperature: 0.2,
        max_tokens: 700,
        max_side: 1280,
        jpeg_quality: 85,
        max_image_mb: 5,
        timeout: 60,
        max_retries: 3,
        concurrency: 1,
        requests_per_second: 0,
        max_requests_per_minute: 0,
      },
      {
        id: 'joycaption',
        label: 'JoyCaption（vLLM 本地）',
        builtin: true,
        base_url: 'http://localhost:8000/v1',
        api_key: '',
        model: 'fancyfeast/llama-joycaption-beta-one-hf-llava',
        model_ids: [],
        endpoint: 'chat_completions',
        messages: [
          { type: 'text', role: 'system', content: 'Descriptive Caption' },
          { type: 'image', role: 'user', content: '' },
        ],
        output_format: 'text',
        temperature: 0.6,
        max_tokens: 300,
        max_side: 1280,
        jpeg_quality: 85,
        max_image_mb: 5,
        timeout: 60,
        max_retries: 3,
        concurrency: 1,
        requests_per_second: 0,
        max_requests_per_minute: 0,
      },
    ],
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
    batch_size: 8,
  },
  cltagger: {
    model_id: 'cella110n/cl_tagger',
    model_path: 'cl_tagger_1_02/model.onnx',
    tag_mapping_path: 'cl_tagger_1_02/tag_mapping.json',
    local_dir: null,
    threshold_general: 0.35,
    threshold_character: 0.6,
    add_copyright_tag: true,
    add_meta_tag: false,
    add_model_tag: false,
    add_rating_tag: false,
    add_quality_tag: false,
    blacklist_tags: [],
    batch_size: 8,
  },
  models: { root: null, selected_anima: '1.0', selected_upscaler: '4x-AnimeSharp', auto_sync_paths: true },
  queue: { allow_gpu_during_train: false },
  download_source: 'huggingface',
  modelscope: { token: '' },
  generate: { preview_every_n_steps: 0, attention_backend: 'sdpa' },
  system: { update_channel: 'stable', show_dev_channel: false },
  proxy: { enabled: false, http_proxy: '', https_proxy: '', no_proxy: '' },
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
  wd14: {
    id: 'wd14',
    name: 'WD14',
    description: 'test',
    repo: 'SmilingWolf/*',
    current_model_id: 'SmilingWolf/wd-eva02-large-tagger-v3',
    variants: [],
  },
  cltagger: {
    id: 'cltagger',
    name: 'CLTagger',
    description: 'test',
    repo: 'cella110n/cl_tagger',
    target_dir: '/tmp/anima/cltagger',
    current_model_path: 'cl_tagger_1_02/model.onnx',
    current_tag_mapping_path: 'cl_tagger_1_02/tag_mapping.json',
    variants: [],
  },
  download_source_options: {
    training: { current: 'huggingface', available: ['huggingface', 'modelscope'] },
    wd14: { current: 'huggingface', available: ['huggingface', 'modelscope'] },
    upscaler: { current: 'huggingface', available: ['huggingface', 'modelscope'] },
    cltagger: { current: 'huggingface', available: ['huggingface'] },
    taeflux: { current: 'huggingface', available: ['huggingface'] },
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
    if (typeof url === 'string' && url.includes('/api/wd14/runtime')) {
      return Promise.resolve(
        new Response(
          JSON.stringify({
            installed: 'onnxruntime',
            version: '1.18.0',
            providers: ['CPUExecutionProvider'],
            cuda_available: false,
            cuda_detect: { available: false, driver_version: null, gpu_name: null },
          }),
          { status: 200 }
        )
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
    <MemoryRouter>
      <ToastProvider>
        <DialogProvider>
          <SettingsDataProvider>
            <SettingsDrawerProvider>
              <SettingsPage />
            </SettingsDrawerProvider>
          </SettingsDataProvider>
        </DialogProvider>
      </ToastProvider>
    </MemoryRouter>
  )
}

describe('SettingsPage (PP0)', () => {
  it('hydrates from /api/secrets and shows masked sensitive fields as placeholder', async () => {
    const user = userEvent.setup()
    renderPage()
    // gelbooru 凭证已挪到「密钥」tab
    await user.click(await screen.findByRole('button', { name: '密钥' }))
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
    await user.click(await screen.findByRole('button', { name: '密钥' }))
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

  it('credentials tab gathers all service tokens; old sections show a moved hint', async () => {
    const user = userEvent.setup()
    renderPage()

    await user.click(await screen.findByRole('button', { name: '密钥' }))
    // 下载 / 抓取类凭证聚到密钥 tab（WandB token 留在监控页跟其配置一起）
    for (const name of ['HuggingFace', 'ModelScope', 'Gelbooru', 'Danbooru']) {
      expect(screen.getByRole('heading', { name })).toBeInTheDocument()
    }
    expect(screen.queryByRole('heading', { name: 'Weights & Biases' })).not.toBeInTheDocument()
    // gelbooru user_id 现在在密钥 tab 编辑
    expect(screen.getByDisplayValue('alice')).toBeInTheDocument()

    // 原数据集 tab 的 gelbooru 不再有 user_id，只留「前往」指引
    await user.click(await screen.findByRole('button', { name: '数据集' }))
    expect(screen.queryByDisplayValue('alice')).not.toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: '前往' }).length).toBeGreaterThan(0)
  })

  it('per-item source dropdown writes download_sources immediately', async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByRole('button', { name: '打标' }))
    // WD14 卡的源 dropdown：本 tab 唯一带 ModelScope 选项的 select
    // （CLTagger 是固定 HF 单选，无 ModelScope 选项）。
    const msOption = await screen.findByRole('option', { name: /ModelScope/ })
    const select = msOption.closest('select') as HTMLSelectElement
    await user.selectOptions(select, 'modelscope')

    await waitFor(() => {
      const putCall = fetchMock.mock.calls.find(([url, init]) => {
        if (init?.method !== 'PUT' || !String(url).includes('/api/secrets')) return false
        try { return 'download_sources' in JSON.parse(String(init.body)) } catch { return false }
      })
      expect(putCall).toBeDefined()
      const body = JSON.parse(String(putCall![1].body))
      expect(body.download_sources).toEqual({ wd14: 'modelscope' })
    })
  })

  it('shows LLM request pool controls on the tagging settings tab', async () => {
    const user = userEvent.setup()
    renderPage()

    await user.click(await screen.findByRole('button', { name: '打标' }))
    await user.click(screen.getByText('高级参数'))

    expect(screen.getByText('Concurrency')).toBeInTheDocument()
    expect(screen.getByText('Requests/sec')).toBeInTheDocument()
    expect(screen.getByText('Max/min')).toBeInTheDocument()
    expect(screen.getAllByText('0 = no limit').length).toBeGreaterThanOrEqual(2)
  })
})
