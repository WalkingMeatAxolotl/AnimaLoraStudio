import { render, screen, waitFor, within } from '@testing-library/react'
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
  },
  danbooru: { username: '', api_key: '', account_type: 'free' },
  download: {
    exclude_tags: [],
    parallel_workers: 4,
    api_rate_per_sec: 2,
    cdn_rate_per_sec: 5,
    save_tags: false,
    convert_to_png: true,
    remove_alpha_channel: false,
  },
  reg: { default_excluded_tags: [] },
  huggingface: { token: '', endpoint: '' },
  wandb: {
    enabled: false,
    current_preset: 'default',
    presets: [
      {
        id: 'default',
        label: 'Default',
        api_key: '',
        project: 'AnimaLoraStudio',
        entity: '',
        base_url: '',
        mode: 'online',
        log_samples: true,
        sample_max_side: 1216,
        sample_every_n_steps: 0,
        upload_model: false,
        upload_model_policy: 'last',
        upload_state_manual: false,
        upload_state_manual_policy: 'last',
        upload_state_auto: false,
        upload_state_auto_policy: 'last',
      },
    ],
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
        assist_tagger: '',
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
        assist_tagger: '',
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
    threshold_general: 0.35,
    threshold_character: 0.85,
    blacklist_tags: [],
    batch_size: 8,
  },
  cltagger: {
    model_id: 'cella110n/cl_tagger',
    model_path: 'cl_tagger_1_02/model.onnx',
    tag_mapping_path: 'cl_tagger_1_02/tag_mapping.json',
    threshold_general: 0.35,
    threshold_character: 0.6,
    add_copyright_tag: true,
    add_artist_tag: false,
    add_meta_tag: false,
    add_model_tag: false,
    add_rating_tag: false,
    add_quality_tag: false,
    blacklist_tags: [],
    batch_size: 8,
  },
  models: { root: null, selected: { anima: '1.0', krea2: 'raw' }, selected_anima: '1.0', custom_anima_paths: [], selected_upscaler: '4x-AnimeSharp', auto_sync_paths: true },
  queue: { light_tasks_during_train: true },
  download_source: 'huggingface',
  modelscope: { token: '' },
  generate: {
    preview_every_n_steps: 0,
    attention_backend: 'sdpa',
    vae_precision: 'bf16',
    lora_merge_precision: 'fp32',
  },
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
    custom: [],
    selected: '1.0',
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
  krea2_main: {
    id: 'krea2_main',
    name: 'Krea 2 主模型',
    description: 'test',
    repo: 'krea/Krea-2-{Raw,Turbo}',
    variants: [
      {
        variant: 'raw', is_latest: true, repo: 'krea/Krea-2-Raw',
        purpose: 'training', size_estimate: 26_300_000_000,
        target_path: '/tmp/anima/diffusion_models/krea2-raw-bf16.safetensors',
        exists: false, size: 0, mtime: 0,
      },
      {
        variant: 'turbo', is_latest: false, repo: 'krea/Krea-2-Turbo',
        purpose: 'inference', size_estimate: 26_300_000_000,
        target_path: '/tmp/anima/diffusion_models/krea2-turbo-bf16.safetensors',
        exists: false, size: 0, mtime: 0,
      },
    ],
    custom: [
      {
        path: '/tmp/models/my-krea2.safetensors', name: 'my-krea2.safetensors',
        exists: true, size: 1024, mtime: 1,
      },
    ],
    selected: 'raw',
    latest: 'raw',
    license: 'Krea 2 Community License',
    license_url: 'https://huggingface.co/krea/Krea-2-Raw/blob/main/LICENSE.pdf',
  },
  krea2_text_encoder: {
    id: 'krea2_text_encoder',
    name: 'Krea 2 · Qwen3-VL-4B-Instruct',
    description: 'test',
    repo: 'Qwen/Qwen3-VL-4B-Instruct',
    target_dir: '/tmp/anima/text_encoders/Qwen_Qwen3-VL-4B-Instruct',
    files: [],
  },
  krea2_text_encoder_fp8: {
    id: 'krea2_text_encoder_fp8',
    name: 'Krea 2 · Qwen3-VL fp8',
    description: 'test',
    repo: 'Comfy-Org/Krea-2',
    target_dir: '/tmp/anima/text_encoders/qwen3vl-4b-fp8',
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
    variants: [
      {
        label: 'cl_tagger_1_02',
        model_id: 'cella110n/cl_tagger',
        model_path: 'cl_tagger_1_02/model.onnx',
        tag_mapping_path: 'cl_tagger_1_02/tag_mapping.json',
        is_current: true,
        exists: false,
        size: 0,
        files: [],
      },
      {
        label: 'cl_tagger_v2_v2_01a',
        model_id: 'cella110n/cl_tagger_v2',
        model_path: 'v2_01a/model.onnx',
        tag_mapping_path: 'v2_01a/model_vocabulary.json',
        is_current: false,
        exists: false,
        size: 0,
        files: [],
      },
    ],
  },
  download_source_options: {
    training: { current: 'huggingface', available: ['huggingface', 'modelscope'] },
    wd14: { current: 'huggingface', available: ['huggingface', 'modelscope'] },
    upscaler: { current: 'huggingface', available: ['huggingface', 'modelscope'] },
    cltagger: { current: 'huggingface', available: ['huggingface'] },
    taeflux: { current: 'huggingface', available: ['huggingface'] },
  },
  // 统一来源候选行（ModelSourceCard 消费；缺键会渲染 loading 文案）
  model_sources: {
    wd14: [],
    cltagger: [
      {
        kind: 'preset', candidate: null,
        value: 'cella110n/cl_tagger|cl_tagger_1_02/model.onnx|cl_tagger_1_02/tag_mapping.json',
        label: 'cl_tagger_1_02', description: '',
        download_id: 'cltagger', download_variant: 'cl_tagger_1_02',
        status_key: 'cltagger:cl_tagger_1_02', exists: false, size: 0,
        files: [], size_estimate: 0, is_current: true,
        removable: false, deletable: true,
        extra: {
          model_id: 'cella110n/cl_tagger',
          model_path: 'cl_tagger_1_02/model.onnx',
          tag_mapping_path: 'cl_tagger_1_02/tag_mapping.json',
        },
      },
      {
        kind: 'preset', candidate: null,
        value: 'cella110n/cl_tagger_v2|v2_01a/model.onnx|v2_01a/model_vocabulary.json',
        label: 'cl_tagger_v2_v2_01a', description: '',
        download_id: 'cltagger', download_variant: 'cl_tagger_v2_v2_01a',
        status_key: 'cltagger:cl_tagger_v2_v2_01a', exists: false, size: 0,
        files: [], size_estimate: 0, is_current: false,
        removable: false, deletable: true,
        extra: {
          model_id: 'cella110n/cl_tagger_v2',
          model_path: 'v2_01a/model.onnx',
          tag_mapping_path: 'v2_01a/model_vocabulary.json',
        },
      },
    ],
    eval_clip: [],
    eval_dino: [],
    eval_ccip: [],
    upscaler: [],
    anima: [],
    krea2: [
      {
        kind: 'preset', candidate: null, value: 'raw', label: 'raw',
        description: '', download_id: 'krea2_main', download_variant: 'raw',
        status_key: 'krea2_main:raw', exists: true, size: 1024, files: null,
        size_estimate: 0, is_current: true, removable: false, deletable: true,
        extra: { purpose: 'training' },
      },
      {
        kind: 'preset', candidate: null, value: 'turbo', label: 'turbo',
        description: '', download_id: 'krea2_main', download_variant: 'turbo',
        status_key: 'krea2_main:turbo', exists: true, size: 1024, files: null,
        size_estimate: 0, is_current: false, removable: false, deletable: true,
        extra: { purpose: 'inference' },
      },
      {
        kind: 'local',
        candidate: { kind: 'local', path: '/tmp/models/my-krea2.safetensors' },
        value: '/tmp/models/my-krea2.safetensors', label: 'my-krea2.safetensors',
        description: '', download_id: null, download_variant: null,
        status_key: null, exists: true, size: 1024, files: null,
        size_estimate: 0, is_current: false, removable: true, deletable: false,
        extra: {},
      },
    ],
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
    if (typeof url === 'string' && url.includes('/api/torch/status')) {
      return Promise.resolve(new Response(JSON.stringify({
        installed: true,
        version: '2.7.0+cu128',
        cuda_build: 'cu128',
        cuda_available: true,
        device_name: 'Test GPU',
        cuda_detect: { available: true, driver_version: '555.0', gpu_name: 'Test GPU' },
        recommended_cu_tag: 'cu128',
        is_cpu_with_gpu: false,
        is_cuda_build_unavailable: false,
      }), { status: 200 }))
    }
    if (typeof url === 'string' && url.includes('/api/flash-attention/status')) {
      return Promise.resolve(new Response(JSON.stringify({
        installed: false,
        version: null,
        env: {
          python_tag: 'cp313', cuda_tag: 'cu128', cuda_ver: '12.8',
          driver_cuda_ver: '12.8', torch_tag: 'torch2.7', torch_ver: '2.7.0',
          torch_cuda_build: 'cu128', platform: 'win_amd64',
        },
        candidates: [],
        fetch_error: null,
      }), { status: 200 }))
    }
    if (typeof url === 'string' && url.includes('/api/xformers/status')) {
      return Promise.resolve(new Response(JSON.stringify({
        installed: false, version: null,
      }), { status: 200 }))
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
    // instant-apply：文本框失焦即提交，无显式保存按钮
    await user.tab()

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

  it('changes LoRA merge precision independently from VAE precision', async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByRole('button', { name: '测试' }))

    const mergeLabel = await screen.findByText('LoRA merge 精度')
    const mergeRow = mergeLabel.closest('.grid') as HTMLElement
    const mergeSelect = mergeRow.querySelector('select') as HTMLSelectElement
    expect(mergeSelect.value).toBe('fp32')
    await user.selectOptions(mergeSelect, 'bf16')

    await waitFor(() => {
      const putCall = fetchMock.mock.calls.find(([url, init]) => {
        if (init?.method !== 'PUT' || !String(url).includes('/api/secrets')) return false
        try {
          return JSON.parse(String(init.body)).generate?.lora_merge_precision === 'bf16'
        } catch {
          return false
        }
      })
      expect(putCall).toBeDefined()
      const body = JSON.parse(String(putCall![1].body))
      expect(body).toEqual({ generate: { lora_merge_precision: 'bf16' } })
    })
  })

  it('credentials tab gathers all service tokens; old sections no longer hold them', async () => {
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

    // 原数据集 tab 的 gelbooru 不再有 user_id（凭证已挪走，无指引文案）
    await user.click(await screen.findByRole('button', { name: '数据集' }))
    expect(screen.queryByDisplayValue('alice')).not.toBeInTheDocument()
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

  it('splits Anima and Krea2 into separate model sections', async () => {
    const user = userEvent.setup()
    renderPage()

    await user.click(await screen.findByRole('button', { name: '训练' }))
    expect(await screen.findByRole('heading', { name: 'Anima 模型' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Krea2 模型' })).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: 'Krea 2 主模型' })).toBeInTheDocument()
    // TE variant 合并卡（bf16/fp8 两行 radio，标题走 i18n）
    expect(screen.getByText('Krea 2 · Qwen3-VL 文本编码器')).toBeInTheDocument()
    expect(screen.getByText('bf16')).toBeInTheDocument()
    expect(screen.getByText('fp8')).toBeInTheDocument()
    expect(screen.getByText('raw')).toBeInTheDocument()
    expect(screen.getByText('turbo')).toBeInTheDocument()
    expect(screen.getByText('my-krea2.safetensors')).toBeInTheDocument()
    const customRow = screen.getByText('my-krea2.safetensors').closest('li')
    expect(customRow).not.toBeNull()
    // 统一候选卡（D2）：local 行带「本地」徽标 + 状态 badge + × 移除（不删文件），
    // 永远没有删除文件按钮
    expect(within(customRow!).getByText('本地')).toBeInTheDocument()
    expect(customRow!.querySelector('.bg-ok-soft')).not.toBeNull()
    expect(within(customRow!).getByTitle('从列表移除（不删除文件）')).toBeInTheDocument()
    expect(within(customRow!).queryByText(/🗑/)).not.toBeInTheDocument()
    // 主模型 3（raw/turbo/custom）+ TE variant 卡 2（bf16/fp8）
    expect(screen.getAllByRole('radio')).toHaveLength(5)
    expect(screen.queryByText(/推荐工作流/)).not.toBeInTheDocument()
    // purpose 徽标（C10）：krea2 variant 行标注用途（raw=训练 / turbo=推理）
    const rawRow = screen.getByText('raw').closest('li')!
    const turboRow = screen.getByText('turbo').closest('li')!
    expect(within(rawRow).getByText('训练')).toBeInTheDocument()
    expect(within(turboRow).getByText('推理')).toBeInTheDocument()
  })

  it('picking a variant writes new-style selected.{family}（多模型 P4-5）', async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByRole('button', { name: '训练' }))
    await screen.findByRole('heading', { name: 'Krea 2 主模型' })
    // fixture：官方 variants 都未下载（radio 禁用），只有 custom 可点
    const customRow = screen.getByText('my-krea2.safetensors').closest('li')!
    await user.click(within(customRow).getByRole('radio'))

    await waitFor(() => {
      const putCall = fetchMock.mock.calls.find(
        ([url, init]) => String(url).includes('/api/secrets') && init?.method === 'PUT'
      )
      expect(putCall).toBeDefined()
      const body = JSON.parse(String(putCall![1].body))
      // 统一写 selected.{family} 新结构——不再有 anima 写 legacy 键的分叉
      expect(body).toEqual({
        models: { selected: { krea2: '/tmp/models/my-krea2.safetensors' } },
      })
    })
  })

  it('opens LLM preset editor modal with rate limit controls from the preset list', async () => {
    const user = userEvent.setup()
    renderPage()

    await user.click(await screen.findByRole('button', { name: '打标' }))
    // 预设列表：全局默认行高亮 + 每行「编辑」action
    const defaultRow = screen.getByText('画风 LoRA JSON').closest('li')
    expect(defaultRow).not.toBeNull()
    expect(within(defaultRow as HTMLElement).getByRole('radio')).toBeChecked()
    await user.click(within(defaultRow as HTMLElement).getByRole('button', { name: /编辑/ }))

    // 编辑 modal：名称字段（补上的预设改名）+ 采样限速字段都在
    const modal = await screen.findByTestId('llm-preset-editor-modal')
    expect(within(modal).getByText('预设名称')).toBeInTheDocument()
    expect(within(modal).getByText('并发数')).toBeInTheDocument()
    expect(within(modal).getByText('每秒请求数（0 = 不限）')).toBeInTheDocument()
    expect(within(modal).getByText('每分钟最大请求数（0 = 不限）')).toBeInTheDocument()
  })

  it('selecting another LLM preset as global default PUTs current_preset', async () => {
    const user = userEvent.setup()
    renderPage()

    await user.click(await screen.findByRole('button', { name: '打标' }))
    const joyRow = screen.getByText('JoyCaption（vLLM 本地）').closest('li')
    await user.click(within(joyRow as HTMLElement).getByRole('radio'))

    await waitFor(() => {
      const putCall = fetchMock.mock.calls.find(([url, init]) => {
        if (init?.method !== 'PUT' || !String(url).includes('/api/secrets')) return false
        try { return JSON.parse(String(init.body)).llm_tagger?.current_preset === 'joycaption' } catch { return false }
      })
      expect(putCall).toBeDefined()
    })
  })


  it('selecting CLTagger v2 updates model id and versioned file paths', async () => {
    const user = userEvent.setup()
    renderPage()

    await user.click(await screen.findByRole('button', { name: '打标' }))
    const v2Row = screen.getByText('cl_tagger_v2_v2_01a').closest('li')
    expect(v2Row).not.toBeNull()
    await user.click(within(v2Row as HTMLElement).getByRole('radio'))
    // instant-apply：选 variant 即时提交，无显式保存按钮

    await waitFor(() => {
      const putCall = fetchMock.mock.calls.find(
        ([, init]) => init?.method === 'PUT'
      )
      expect(putCall).toBeDefined()
      const body = JSON.parse(String(putCall![1].body))
      expect(body.cltagger).toMatchObject({
        model_id: 'cella110n/cl_tagger_v2',
        model_path: 'v2_01a/model.onnx',
        tag_mapping_path: 'v2_01a/model_vocabulary.json',
      })
    })
  })

})
