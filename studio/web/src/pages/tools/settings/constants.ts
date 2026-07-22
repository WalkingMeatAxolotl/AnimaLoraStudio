import type { TFunction } from 'i18next'
import {
  DEFAULT_WD14_MODELS,
  type LLMPreset,
  type Secrets,
  type WandBPreset,
} from '../../../api/client'

export const MASK = '***'

/** WandB 预设默认值（与后端 WandBPresetConfig 默认一致），新建/导入预设的底板。 */
export const DEFAULT_WANDB_PRESET: WandBPreset = {
  id: 'default',
  label: 'Default',
  api_key: '',
  project: 'AnimaLoraStudio',
  entity: '',
  base_url: '',
  mode: 'online',
  log_samples: true,
  sample_max_side: 512,
  sample_every_n_steps: 0,
  upload_model: false,
  upload_model_policy: 'last',
  upload_state_manual: false,
  upload_state_manual_policy: 'last',
  upload_state_auto: false,
  upload_state_auto_policy: 'last',
}

export type Section =
  | 'gelbooru'
  | 'danbooru'
  | 'download'
  | 'reg'
  | 'huggingface'
  | 'wandb'
  | 'modelscope'
  | 'eval_metrics'
  | 'llm_tagger'
  | 'wd14'
  | 'cltagger'
  | 'models'
  | 'queue'
  | 'generate'
  | 'proxy'


export type Tab = 'dataset' | 'tagging' | 'preprocess' | 'training' | 'monitor' | 'testing' | 'credentials' | 'appearance' | 'system'

// 外部页面通过 `?section=<id>` 跳转到 SettingsPage 的特定 section 时，用这个
// 反向映射决定要先切到哪个 tab。只列出能从外部链接到的 sections。
export const SECTION_TO_TAB: Record<string, Tab> = {
  'models': 'training',
  'krea2-models': 'training',
  'eval-metrics': 'monitor',
  'version': 'system',
  'service': 'system',
}

export const TAB_LIST: { id: Tab; labelKey: string }[] = [
  { id: 'credentials', labelKey: 'settings.tabCredentials' },
  { id: 'dataset', labelKey: 'settings.tabDataset' },
  { id: 'preprocess', labelKey: 'settings.tabPreprocess' },
  { id: 'tagging', labelKey: 'settings.tabTagging' },
  { id: 'training', labelKey: 'settings.tabTraining' },
  { id: 'monitor', labelKey: 'settings.tabMonitor' },
  { id: 'testing', labelKey: 'settings.tabGenerate' },
  { id: 'appearance', labelKey: 'settings.tabAppearance' },
  { id: 'system', labelKey: 'settings.tabSystem' },
]

// 每个 tab 的 section index — 用于右侧 sticky 导航。id 与各 section 的 DOM id
// 对应；label 在导航里直接显示。修改 section 顺序时记得同步这里。
export const TAB_SECTIONS: Record<Tab, { id: string; labelKey: string }[]> = {
  dataset: [
    { id: 'download-global', labelKey: 'settings.downloadGlobal' },
    { id: 'reg', labelKey: 'settings.reg.sectionTitle' },
    { id: 'proxy', labelKey: 'settings.proxy.sectionTitle' },
  ],
  preprocess: [
    { id: 'upscalers', labelKey: 'settings.upscalers' },
  ],
  tagging: [
    { id: 'llm-tagger', labelKey: 'settings.llmTagger' },
    { id: 'wd14', labelKey: 'settings.wd14' },
    { id: 'cltagger', labelKey: 'settings.clTagger' },
    { id: 'onnxruntime', labelKey: 'settings.onnxRuntime' },
    { id: 'tag-dictionary', labelKey: 'settings.tagDictionary.title' },
  ],
  training: [
    { id: 'queue', labelKey: 'settings.queueSchedule' },
    { id: 'training-params', labelKey: 'settings.trainingParams' },
    { id: 'pytorch', labelKey: 'settings.torch' },
    { id: 'flash-attn', labelKey: 'settings.flashAttn' },
    { id: 'xformers', labelKey: 'settings.xformers' },
    { id: 'models', labelKey: 'settings.animaModels' },
    { id: 'krea2-models', labelKey: 'settings.krea2Models' },
  ],
  monitor: [
    { id: 'eval-metrics', labelKey: 'settings.evalMetrics' },
    { id: 'wandb', labelKey: 'settings.wandb' },
  ],
  testing: [
    { id: 'idle-timeout', labelKey: 'settings.idleTimeout.title' },
    { id: 'preview', labelKey: 'settings.intermediatePreview' },
    { id: 'save-test-images', labelKey: 'settings.saveTestImages.title' },
  ],
  credentials: [
    { id: 'cred-huggingface', labelKey: 'settings.credHuggingface' },
    { id: 'cred-modelscope', labelKey: 'settings.credModelscope' },
    { id: 'cred-gelbooru', labelKey: 'settings.gelbooru' },
    { id: 'cred-danbooru', labelKey: 'settings.danbooru' },
  ],
  appearance: [
    { id: 'display', labelKey: 'settings.display' },
  ],
  system: [
    { id: 'version', labelKey: 'settings.version' },
    { id: 'storage', labelKey: 'settings.storage.sectionTitle' },
    { id: 'service', labelKey: 'settings.service' },
  ],
}

export const TAB_STORAGE_KEY = 'studio.settings.activeTab'

// fallback 预设：仅在 GET /api/secrets 失败时充当占位，真实 prompt 由后端 builtin
// json 文件提供。命中此 fallback 然后 PUT 回去不会破坏 builtin（后端 validator
// 会再补全 builtin defaults）。
export function _makeFallbackPreset(id: string, label: string, output_format: 'json' | 'text', extra: Partial<LLMPreset> = {}): LLMPreset {
  return {
    id,
    label,
    builtin: true,
    base_url: '',
    api_key: '',
    model: '',
    model_ids: [],
    endpoint: 'chat_completions',
    messages: [
      { type: 'text', role: 'system', content: '' },
      { type: 'image', role: 'user', content: '' },
    ],
    output_format,
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
    ...extra,
  }
}

export const DEFAULT_LLM_PRESETS: LLMPreset[] = [
  _makeFallbackPreset('style_json', '画风 LoRA JSON', 'json'),
  _makeFallbackPreset('general_json', '通用 LoRA JSON', 'json'),
  _makeFallbackPreset('txt_tags', 'TXT 标签列表', 'json'),
  _makeFallbackPreset('joycaption', 'JoyCaption（vLLM 本地）', 'text', {
    base_url: 'http://localhost:8000/v1',
    model: 'fancyfeast/llama-joycaption-beta-one-hf-llava',
    temperature: 0.6,
    max_tokens: 300,
  }),
]

export function getStoredTab(): Tab {
  try {
    const v = localStorage.getItem(TAB_STORAGE_KEY)
    if (
      v === 'dataset' || v === 'tagging' || v === 'preprocess' || v === 'training'
      || v === 'monitor' || v === 'testing' || v === 'credentials'
      || v === 'appearance' || v === 'system'
    ) return v
  } catch {
    /* ignore localStorage errors */
  }
  return 'dataset'
}

export const EMPTY: Secrets = {
  gelbooru: { user_id: '', api_key: '' },
  danbooru: { username: '', api_key: '', account_type: 'free' },
  download: {
    exclude_tags: [],
    parallel_workers: 4,
    api_rate_per_sec: 2,
    cdn_rate_per_sec: 5,
    save_tags: false,
    convert_to_png: true,
    remove_alpha_channel: true,
  },
  reg: { default_excluded_tags: [] },
  huggingface: { token: '', endpoint: '' },
  wandb: {
    enabled: false,
    current_preset: 'default',
    presets: [DEFAULT_WANDB_PRESET],
  },
  modelscope: { token: '' },
  eval_metrics: {
    clip_model_name: 'openai/clip-vit-base-patch32',
    dino_model_name: 'facebook/dinov2-small',
    ccip_model_name: 'ccip-caformer-24-randaug-pruned',
    enabled_metrics: ['clip_t', 'clip_i', 'dino_i'],
    eval_baseline_enabled: true,
  },
  download_source: 'huggingface',
  download_sources: {},
  llm_tagger: {
    current_preset: 'style_json',
    presets: [...DEFAULT_LLM_PRESETS],
  },
  wd14: {
    model_id: 'SmilingWolf/wd-eva02-large-tagger-v3',
    model_ids: [...DEFAULT_WD14_MODELS],
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
  generate: { preview_every_n_steps: 3, attention_backend: 'auto', vae_precision: 'bf16', lora_merge_precision: 'fp32', idle_timeout_minutes: 10, save_test_images: false, vram_policy: 'auto', ram_guard: true, blocks_to_swap: 0, task_timeout_minutes: 0 },
  system: { update_channel: 'stable', show_dev_channel: false },
  proxy: {
    enabled: false,
    http_proxy: '',
    https_proxy: '',
    no_proxy: '',
  }
}

export const textInputClass = 'w-full px-2 py-1 outline-none rounded-sm bg-sunken border border-subtle text-sm text-fg-primary focus:border-accent'

export const MODEL_DESCRIPTION_KEYS: Record<string, string> = {
  anima_main: 'settings.modelDescriptions.animaMain',
  anima_vae: 'settings.modelDescriptions.animaVae',
  qwen3: 'settings.modelDescriptions.qwen3',
  t5_tokenizer: 'settings.modelDescriptions.t5Tokenizer',
  krea2_main: 'settings.modelDescriptions.krea2Main',
  krea2_text_encoder: 'settings.modelDescriptions.krea2TextEncoder',
  krea2_text_encoder_fp8: 'settings.modelDescriptions.krea2TextEncoderFp8',
  wd14: 'settings.modelDescriptions.wd14',
  cltagger: 'settings.modelDescriptions.cltagger',
}

export const UPSCALER_DESCRIPTION_KEYS: Record<string, string> = {
  '4x-AnimeSharp': 'settings.upscalerDescriptions.animeSharp',
  'R-ESRGAN_4x+Anime6B': 'settings.upscalerDescriptions.realEsrganAnime6B',
  '4x_foolhardy_Remacri': 'settings.upscalerDescriptions.remacri',
  'ESRGAN_4x': 'settings.upscalerDescriptions.esrgan4x',
}

export function translatedCatalogText(keys: Record<string, string>, id: string, fallback: string | undefined, t: TFunction): string {
  const key = keys[id]
  return key ? t(key, { defaultValue: fallback ?? '' }) : (fallback ?? '')
}

// ── Models Section ─────────────────────────────────────────────────────────

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}
