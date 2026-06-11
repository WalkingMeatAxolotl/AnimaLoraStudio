import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from 'react'
import type { TFunction } from 'i18next'
import { Trans, useTranslation } from 'react-i18next'
import {
  api,
  DEFAULT_WD14_MODELS,
  type CLTaggerVariantInfo,
  type LLMPreset,
  type FlashAttnStatus,
  type XformersStatus,
  type ModelDownloadStatus,
  type ModelsCatalog,
  type Secrets,
  type SecretsPatch,
  type DevCommit,
  type DevCommitsResult,
  type PreflightResult,
  type ReleaseNotes,
  type SystemPrefsConfig,
  type SystemUpdateCheck,
  type SystemUpdateStatus,
  type SystemVersion,
  type TorchCuTag,
  type TorchStatus,
  type WandBConfig,
  type WD14Runtime,
} from '../../api/client'
import { useDialog } from '../../components/Dialog'
import {
  clearOnboardingDone,
  ONBOARDING_EVENTS,
} from '../../components/FirstRunOnboardingModal'
import { InfoButton } from '../../components/InfoButton'
import LLMTaggerWorkspace from '../../components/LLMTaggerWorkspace'
import { TagListInput } from '../../components/TagsInput'
import { useShowTagTranslation } from '../../tagDict/showToggle'
import { useTagDict, reloadDict } from '../../tagDict/store'
import PageHeader from '../../components/PageHeader'
import { useToast } from '../../components/Toast'
import { useSettingsData } from '../../lib/SettingsData'
import { useSettingsDrawer } from '../../lib/SettingsDrawer'
import {
  formatMasterStateText,
  formatDevStateText,
  shouldShowMasterUpdateButton,
  shouldShowSwitchToStableButton,
  isDevSwitchButtonDisabled,
} from '../../lib/versionPanel'
import { applyDensity, applyTheme, getStoredDensity, getStoredTheme, setStoredDensity, setStoredTheme, type Density, type Theme } from '../../lib/theme'
import i18n, { getStoredLangWithDefault, setStoredLang } from '../../i18n'

const MASK = '***'

type Section =
  | 'gelbooru'
  | 'danbooru'
  | 'download'
  | 'huggingface'
  | 'wandb'
  | 'modelscope'
  | 'llm_tagger'
  | 'wd14'
  | 'cltagger'
  | 'models'
  | 'queue'
  | 'generate'
  | 'proxy'

type Tab = 'dataset' | 'tagging' | 'preprocess' | 'training' | 'monitor' | 'testing' | 'appearance' | 'system'

// 外部页面通过 `?section=<id>` 跳转到 SettingsPage 的特定 section 时，用这个
// 反向映射决定要先切到哪个 tab。只列出能从外部链接到的 sections。
const SECTION_TO_TAB: Record<string, Tab> = {
  'models': 'training',
  'download-source': 'training',
  'version': 'system',
  'service': 'system',
}

const TAB_LIST: { id: Tab; labelKey: string }[] = [
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
const TAB_SECTIONS: Record<Tab, { id: string; labelKey: string }[]> = {
  dataset: [
    { id: 'gelbooru', labelKey: 'settings.gelbooru' },
    { id: 'danbooru', labelKey: 'settings.danbooru' },
    { id: 'download-global', labelKey: 'settings.downloadGlobal' },
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
    { id: 'download-source', labelKey: 'settings.modelSource' },
    { id: 'queue', labelKey: 'settings.queueSchedule' },
    { id: 'pytorch', labelKey: 'settings.torch' },
    { id: 'flash-attn', labelKey: 'settings.flashAttn' },
    { id: 'xformers', labelKey: 'settings.xformers' },
    { id: 'models', labelKey: 'settings.trainingModels' },
  ],
  monitor: [
    { id: 'wandb', labelKey: 'settings.wandb' },
  ],
  testing: [
    { id: 'idle-timeout', labelKey: 'settings.idleTimeout.title' },
    { id: 'preview', labelKey: 'settings.intermediatePreview' },
    { id: 'save-test-images', labelKey: 'settings.saveTestImages.title' },
  ],
  appearance: [
    { id: 'display', labelKey: 'settings.display' },
  ],
  system: [
    { id: 'onboarding', labelKey: 'settings.onboardingSection' },
    { id: 'version', labelKey: 'settings.version' },
    { id: 'service', labelKey: 'settings.service' },
  ],
}

const TAB_STORAGE_KEY = 'studio.settings.activeTab'

// fallback 预设：仅在 GET /api/secrets 失败时充当占位，真实 prompt 由后端 builtin
// json 文件提供。命中此 fallback 然后 PUT 回去不会破坏 builtin（后端 validator
// 会再补全 builtin defaults）。
function _makeFallbackPreset(id: string, label: string, output_format: 'json' | 'text', extra: Partial<LLMPreset> = {}): LLMPreset {
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

const DEFAULT_LLM_PRESETS: LLMPreset[] = [
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

function getStoredTab(): Tab {
  try {
    const v = localStorage.getItem(TAB_STORAGE_KEY)
    if (
      v === 'dataset' || v === 'tagging' || v === 'preprocess' || v === 'training'
      || v === 'monitor' || v === 'testing' || v === 'appearance'
      || v === 'system'
    ) return v
  } catch {
    /* ignore localStorage errors */
  }
  return 'dataset'
}

const EMPTY: Secrets = {
  gelbooru: {
    user_id: '',
    api_key: '',
    save_tags: false,
    convert_to_png: true,
    remove_alpha_channel: true,
  },
  danbooru: { username: '', api_key: '', account_type: 'free' },
  download: {
    exclude_tags: [],
    parallel_workers: 4,
    api_rate_per_sec: 2,
    cdn_rate_per_sec: 5,
  },
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
    upload_model: false,
    upload_model_policy: 'last',
    upload_state_manual: false,
    upload_state_manual_policy: 'last',
    upload_state_auto: false,
    upload_state_auto_policy: 'last',
  },
  modelscope: { token: '' },
  download_source: 'huggingface',
  llm_tagger: {
    current_preset: 'style_json',
    presets: [...DEFAULT_LLM_PRESETS],
  },
  wd14: {
    model_id: 'SmilingWolf/wd-eva02-large-tagger-v3',
    model_ids: [...DEFAULT_WD14_MODELS],
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
  generate: { preview_every_n_steps: 3, attention_backend: 'auto', idle_timeout_minutes: 10, save_test_images: false },
  system: { update_channel: 'stable', show_dev_channel: false },
  proxy: {
    enabled: false,
    http_proxy: '',
    https_proxy: '',
    no_proxy: '',
  }
}

const textInputClass = 'w-full px-2 py-1 outline-none rounded-sm bg-sunken border border-subtle text-sm text-fg-primary focus:border-accent'

const MODEL_DESCRIPTION_KEYS: Record<string, string> = {
  anima_main: 'settings.modelDescriptions.animaMain',
  anima_vae: 'settings.modelDescriptions.animaVae',
  qwen3: 'settings.modelDescriptions.qwen3',
  t5_tokenizer: 'settings.modelDescriptions.t5Tokenizer',
  wd14: 'settings.modelDescriptions.wd14',
  cltagger: 'settings.modelDescriptions.cltagger',
}

const UPSCALER_DESCRIPTION_KEYS: Record<string, string> = {
  '4x-AnimeSharp': 'settings.upscalerDescriptions.animeSharp',
  'R-ESRGAN_4x+Anime6B': 'settings.upscalerDescriptions.realEsrganAnime6B',
  '4x_foolhardy_Remacri': 'settings.upscalerDescriptions.remacri',
  'ESRGAN_4x': 'settings.upscalerDescriptions.esrgan4x',
}

function translatedCatalogText(keys: Record<string, string>, id: string, fallback: string | undefined, t: TFunction): string {
  const key = keys[id]
  return key ? t(key, { defaultValue: fallback ?? '' }) : (fallback ?? '')
}

export default function SettingsPage() {
  const { t } = useTranslation()
  // 共享数据层（SettingsDataProvider）：secrets / catalog / SSE / downloadBusy 都在根级常驻，
  // 本组件 mount/unmount（抽屉开关）不再触发重拉。`server` 别名保留是为了让下方
  // 大段表单代码改动最小。
  const {
    secrets: server,
    secretsError,
    setSecrets: setServer,
    catalog,
    catalogError,
    reloadCatalog,
    downloadBusy,
    startDownload,
  } = useSettingsData()
  const [draft, setDraft] = useState<Secrets>(EMPTY)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [tab, setTab] = useState<Tab>(getStoredTab)
  const [llmModelsBusy, setLlmModelsBusy] = useState(false)
  const [llmTestBusy, setLlmTestBusy] = useState(false)
  const { toast } = useToast()
  const { prompt } = useDialog()
  const drawer = useSettingsDrawer()
  // 右侧 section index 用：sticky nav 的 IntersectionObserver root + 滚动平移容器
  const scrollContainerRef = useRef<HTMLDivElement>(null)

  // 第一次拿到 secrets 时把 draft 同步过来；之后 server 变化（save 后）不再
  // 覆盖 draft，避免抹掉用户的未保存编辑（save 里会自己 setDraft(next)）。
  const draftInitRef = useRef(false)
  useEffect(() => {
    if (server && !draftInitRef.current) {
      setDraft(server)
      draftInitRef.current = true
    }
  }, [server])
  // 数据层 fetch secrets 失败时把错误透出到本组件 error 状态，复用底部错误条。
  useEffect(() => { if (secretsError) setError(secretsError) }, [secretsError])

  const switchTab = (next: Tab) => {
    setTab(next)
    try {
      localStorage.setItem(TAB_STORAGE_KEY, next)
    } catch {
      /* ignore localStorage errors */
    }
  }

  const dirty = useMemo(
    () => server !== null && JSON.stringify(server) !== JSON.stringify(draft),
    [server, draft]
  )

  // 抽屉关闭前用这个 ref 询问"是否 dirty"；ref 每次 render 刷新，
  // 注册的函数只挂载一次，避免 effect churn。
  const dirtyRef = useRef(false)
  dirtyRef.current = dirty
  useEffect(() => {
    drawer.registerDirtyGuard(() => dirtyRef.current)
    return () => drawer.registerDirtyGuard(null)
  }, [drawer])

  // 抽屉以 open({ section }) 打开时跳到对应 section（取代旧的 ?section= URL 参数）。
  // sectionRequest 带 nonce，相同 section 重复 open 也会触发 effect 重跑。
  const drawerSectionReq = drawer.sectionRequest
  useEffect(() => {
    if (!drawerSectionReq) return
    const section = drawerSectionReq.section
    const targetTab = SECTION_TO_TAB[section]
    if (targetTab) setTab(targetTab)
    const t1 = setTimeout(() => {
      const el = document.getElementById(section)
      el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 50)
    return () => clearTimeout(t1)
  }, [drawerSectionReq])

  const update = <S extends Section, K extends keyof Secrets[S]>(
    section: S,
    key: K,
    value: Secrets[S][K]
  ) => {
    setDraft((prev) => ({
      ...prev,
      [section]: { ...prev[section], [key]: value },
    }))
  }

  /** 更新 Secrets 顶层非对象字段（如 download_source）。 */
  const updateTop = <K extends keyof Secrets>(key: K, value: Secrets[K]) => {
    setDraft((prev) => ({ ...prev, [key]: value }))
  }

  const save = async () => {
    if (!server) return
    const patch = buildPatch(draft, server)
    setSaving(true)
    setError(null)
    try {
      const next = await api.updateSecrets(patch)
      setServer(next)
      setDraft(next)
      // 候选 model_ids 改了之后，catalog 里的 wd14 variants 需要刷新
      void reloadCatalog()
      toast(t('settings.saved'), 'success')
    } catch (e) {
      setError(String(e))
      toast(t('settings.saveFailed'), 'error')
    } finally {
      setSaving(false)
    }
  }

  // 找到当前 active preset；如果 current_preset 指向不存在的 id（理论上 validator
  // 已保底），fallback 到第一个，避免空 crash。
  const currentPreset: LLMPreset =
    draft.llm_tagger.presets.find((p) => p.id === draft.llm_tagger.current_preset)
    ?? draft.llm_tagger.presets[0]
    ?? DEFAULT_LLM_PRESETS[0]

  const serverCurrentPreset: LLMPreset | undefined =
    server?.llm_tagger.presets.find((p) => p.id === currentPreset.id)

  /** 改 active preset 的某个字段。 */
  const updatePreset = <K extends keyof LLMPreset>(field: K, value: LLMPreset[K]) => {
    const next = draft.llm_tagger.presets.map((p) =>
      p.id === currentPreset.id ? { ...p, [field]: value } : p
    )
    update('llm_tagger', 'presets', next)
  }

  const addPreset = () => {
    const used = new Set(draft.llm_tagger.presets.map((p) => p.id))
    let idx = 1
    let id = `preset_${idx}`
    while (used.has(id)) {
      idx += 1
      id = `preset_${idx}`
    }
    const next: LLMPreset = _makeFallbackPreset(id, t('settings.newPresetLabel', { n: idx }), 'json')
    next.builtin = false
    update('llm_tagger', 'presets', [...draft.llm_tagger.presets, next])
    update('llm_tagger', 'current_preset', id)
  }

  const deleteCurrentPreset = () => {
    if (currentPreset.builtin || draft.llm_tagger.presets.length <= 1) return
    const next = draft.llm_tagger.presets.filter((p) => p.id !== currentPreset.id)
    update('llm_tagger', 'presets', next)
    update('llm_tagger', 'current_preset', next[0]?.id ?? 'style_json')
  }

  const resetCurrentPresetToBuiltin = () => {
    // 删除当前 builtin preset，让 backend validator 在 PUT 后从 defaults 补回
    if (!currentPreset.builtin) return
    const next = draft.llm_tagger.presets.filter((p) => p.id !== currentPreset.id)
    update('llm_tagger', 'presets', next)
    // current_preset 不变；validator 会重建 preset
  }

  const saveAsNewPreset = async () => {
    const label = await prompt(t('settings.newPresetName'), {
      defaultValue: t('settings.presetCopy', { label: currentPreset.label }),
      placeholder: 'my-preset',
      validate: (v) => (v.trim() ? null : t('settings.nameRequired')),
    })
    if (!label) return
    const slug = label.toLowerCase().replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '') || 'preset'
    const used = new Set(draft.llm_tagger.presets.map((p) => p.id))
    let idx = 1
    let id = slug
    while (used.has(id)) {
      idx += 1
      id = `${slug}_${idx}`
    }
    const next: LLMPreset = {
      ...currentPreset,
      // deep-copy messages 避免共享引用
      messages: currentPreset.messages.map((m) => ({ ...m })),
      model_ids: [...currentPreset.model_ids],
      id,
      label,
      builtin: false,
    }
    update('llm_tagger', 'presets', [...draft.llm_tagger.presets, next])
    update('llm_tagger', 'current_preset', id)
  }

  const refreshLLMModels = async () => {
    if (!server) return
    setLlmModelsBusy(true)
    setError(null)
    try {
      let source = draft
      if (dirty) {
        const saved = await api.updateSecrets(buildPatch(draft, server))
        setServer(saved)
        setDraft(saved)
        source = saved
      }
      const sourcePreset = source.llm_tagger.presets.find((p) => p.id === currentPreset.id)
        ?? source.llm_tagger.presets[0]
      const result = await api.refreshLLMModels({
        preset_id: sourcePreset.id,
        base_url: sourcePreset.base_url,
        api_key: sourcePreset.api_key,
        timeout: sourcePreset.timeout,
      })
      setServer(result.secrets)
      setDraft(result.secrets)
      toast(t('settings.modelsLoaded', { n: result.items.length }), 'success')
    } catch (e) {
      setError(String(e))
      toast(t('settings.modelsLoadFailed', { error: String(e) }), 'error')
    } finally {
      setLlmModelsBusy(false)
    }
  }

  const testLLMConnection = async () => {
    setLlmTestBusy(true)
    setError(null)
    try {
      const result = await api.testLLMConnection({
        preset_id: currentPreset.id,
        base_url: currentPreset.base_url,
        api_key: currentPreset.api_key,
        model: currentPreset.model,
        endpoint: currentPreset.endpoint,
        timeout: currentPreset.timeout,
        max_tokens: Math.max(512, currentPreset.max_tokens),
        temperature: currentPreset.temperature,
      })
      // 把延迟 / HTTP 状态 / 错误预览拼进 toast，避免移除 ConnBar 后用户拿不到详情。
      const parts: string[] = [result.ok ? t('settings.llmTestOk') : t('settings.llmTestNotOk')]
      if (result.elapsed_ms > 0) parts.push(`${result.elapsed_ms} ms`)
      if (result.status_code !== null) parts.push(`HTTP ${result.status_code}`)
      if (!result.ok) {
        const detail = result.error || result.response_preview
        if (detail) parts.push(detail.slice(0, 120))
      }
      toast(parts.join(' · '), result.ok ? 'success' : 'error')
    } catch (e) {
      toast(t('settings.llmTestFailed', { error: String(e) }), 'error')
    } finally {
      setLlmTestBusy(false)
    }
  }

  if (error && !server) {
    return (
      <div className="text-err font-mono text-sm p-4 bg-err-soft rounded-md">
        {error}
      </div>
    )
  }

  // Tab nav 抽出来传给 PageHeader 的 tabs prop（取代旧的 subtitle 位置）。
  const tabNav = (
    <nav className="flex gap-1 -mb-4">
      {TAB_LIST.map((item) => (
        <button
          key={item.id}
          onClick={() => switchTab(item.id)}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            tab === item.id
              ? 'border-accent text-fg-primary'
              : 'border-transparent text-fg-tertiary hover:text-fg-secondary'
          }`}
        >
          {t(item.labelKey)}
        </button>
      ))}
    </nav>
  )

  return (
    <div className="flex flex-col h-full min-h-0">
      <PageHeader
        title={t('settings.title')}
        tabs={tabNav}
        sticky
        topRight={drawer.isOpen ? (
          <button
            onClick={() => void drawer.close()}
            title={t('settings.drawerClose')}
            aria-label={t('settings.drawerClose')}
            className="w-7 h-7 grid place-items-center text-fg-tertiary bg-transparent border-none rounded-sm cursor-pointer hover:bg-overlay hover:text-fg-primary transition-colors"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M6 6l12 12M18 6l-12 12" />
            </svg>
          </button>
        ) : undefined}
        actions={
          <button
            onClick={save}
            disabled={!dirty || saving}
            className={dirty ? 'btn btn-primary btn-sm' : 'btn btn-secondary btn-sm'}
          >
            {saving ? t('common.saving') : t('common.save')}
          </button>
        }
      />

      <div ref={scrollContainerRef} className="p-6 pb-12 flex-1 overflow-y-auto">
      <div className="grid gap-10 max-w-[1400px]" style={{ gridTemplateColumns: 'minmax(0,1fr) 200px' }}>
      <div className="flex flex-col gap-8 min-w-0">

      {error && (
        <div className="p-3 rounded-md bg-err-soft border border-err text-err text-sm font-mono">
          {error}
        </div>
      )}

      {tab === 'dataset' && (<>
      <SettingsSection id="gelbooru" title="Gelbooru">
        <SettingsField label="user_id">
          <SettingsInput
            type="text"
            value={draft.gelbooru.user_id}
            onChange={(v) => update('gelbooru', 'user_id', v)}
            autoComplete="off"
            data-lpignore="true"
            data-1p-ignore
            data-form-type="other"
            className={textInputClass}
          />
        </SettingsField>
        <SettingsField label="api_key">
          <SensitiveInput
            value={draft.gelbooru.api_key}
            serverValue={server?.gelbooru.api_key ?? ''}
            onChange={(v) => update('gelbooru', 'api_key', v)}
          />
        </SettingsField>
        <SettingsField label="save_tags">
          <Bool value={draft.gelbooru.save_tags} onChange={(v) => update('gelbooru', 'save_tags', v)} />
        </SettingsField>
        <SettingsField label="convert_to_png">
          <Bool value={draft.gelbooru.convert_to_png} onChange={(v) => update('gelbooru', 'convert_to_png', v)} />
        </SettingsField>
        <SettingsField label="remove_alpha_channel">
          <Bool value={draft.gelbooru.remove_alpha_channel} onChange={(v) => update('gelbooru', 'remove_alpha_channel', v)} />
        </SettingsField>
      </SettingsSection>

      <SettingsSection id="danbooru" title="Danbooru">
        <SettingsField label="username">
          <SettingsInput
            type="text"
            value={draft.danbooru.username}
            onChange={(v) => update('danbooru', 'username', v)}
            placeholder={t('settings.danbooruUsernamePlaceholder')}
            autoComplete="off"
            data-lpignore="true"
            data-1p-ignore
            data-form-type="other"
            className={textInputClass}
          />
        </SettingsField>
        <SettingsField label="api_key">
          <SensitiveInput
            value={draft.danbooru.api_key}
            serverValue={server?.danbooru.api_key ?? ''}
            onChange={(v) => update('danbooru', 'api_key', v)}
          />
        </SettingsField>
        <SettingsField label="account_type">
          <select
            value={draft.danbooru.account_type}
            onChange={(e) => update('danbooru', 'account_type', e.target.value as 'free' | 'gold' | 'platinum')}
            className={textInputClass}          >
            <option value="free">{t('settings.accountFree')}</option>
            <option value="gold">{t('settings.accountGold')}</option>
            <option value="platinum">{t('settings.accountPlatinum')}</option>
          </select>
        </SettingsField>
      </SettingsSection>

      <SettingsSection id="download-global" title={t('settings.downloadGlobal')}>
        <SettingsField
          label="exclude_tags"
          desc={t('settings.commaSeparated')}
          helpTooltip={<p><Trans i18nKey="settings.excludeTagsHelp" components={{ code: <code /> }} /></p>}
        >
          <SettingsInput
            type="text"
            value={draft.download.exclude_tags.join(', ')}
            onChange={(v) =>
              update('download', 'exclude_tags',
                v.split(',').map((t) => t.trim().replace(/^-+/, '')).filter(Boolean)
              )
            }
            placeholder={t('settings.excludeTagsPlaceholder')}
            className={textInputClass}
          />
        </SettingsField>

        <div className="grid grid-cols-3 gap-3 pt-2 border-t border-subtle">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-fg-secondary font-mono">parallel_workers</label>
            <SettingsInput
              type="number" min={1} max={16}
              value={draft.download.parallel_workers}
              onChange={(v) => update('download', 'parallel_workers', Math.max(1, Number(v) || 1))}
              className={`${textInputClass} max-w-24`}
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-fg-secondary font-mono">api_rate_per_sec</label>
            <SettingsInput
              type="number" step="0.5" min={0.5} max={10}
              value={draft.download.api_rate_per_sec}
              onChange={(v) => update('download', 'api_rate_per_sec', Math.max(0.5, Number(v) || 0.5))}
              className={`${textInputClass} max-w-24`}
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-fg-secondary font-mono">cdn_rate_per_sec</label>
            <SettingsInput
              type="number" step="1" min={1} max={20}
              value={draft.download.cdn_rate_per_sec}
              onChange={(v) => update('download', 'cdn_rate_per_sec', Math.max(1, Number(v) || 1))}
              className={`${textInputClass} max-w-24`}
            />
          </div>
        </div>
      </SettingsSection>

      <SettingsSection id="proxy" title={t('settings.proxy.sectionTitle')}>
        <SettingsField label={t('settings.proxy.enableLabel')}>
          <Bool
            value={draft.proxy.enabled}
            onChange={(v) => update('proxy', 'enabled', v)}
          />
          <p className="text-xs text-fg-tertiary mt-1">
            {t('settings.proxy.enableDesc')}
          </p>
        </SettingsField>

        <SettingsField
          label={t('settings.proxy.httpLabel')}
          desc={t('settings.proxy.httpDesc')}
        >
          <SettingsInput
            type="text"
            value={draft.proxy.http_proxy}
            onChange={(v) => update('proxy', 'http_proxy', v)}
            placeholder="http://127.0.0.1:7890"
            className={textInputClass}
            disabled={!draft.proxy.enabled}
          />
        </SettingsField>

        <SettingsField
          label={t('settings.proxy.httpsLabel')}
          desc={t('settings.proxy.httpsDesc')}
        >
          <SettingsInput
            type="text"
            value={draft.proxy.https_proxy}
            onChange={(v) => update('proxy', 'https_proxy', v)}
            placeholder="http://127.0.0.1:7890"
            className={textInputClass}
            disabled={!draft.proxy.enabled}
          />
        </SettingsField>

        <SettingsField
          label={t('settings.proxy.noProxyLabel')}
          desc={t('settings.proxy.noProxyDesc')}
        >
          <SettingsInput
            type="text"
            value={draft.proxy.no_proxy}
            onChange={(v) => update('proxy', 'no_proxy', v)}
            placeholder="localhost,127.0.0.1"
            className={textInputClass}
            disabled={!draft.proxy.enabled}
          />
        </SettingsField>

        <div className="text-xs text-fg-tertiary border-t border-subtle pt-3 mt-1">
          <p className="m-0">{t('settings.proxy.tipsTitle')}</p>
          <ul className="list-disc pl-4 m-0 mt-1 space-y-0.5">
            <li>{t('settings.proxy.tips1')}</li>
            <li>
              {t('settings.proxy.tips2')}
              <code className="text-fg-primary">http://user:pass@host:port</code>
            </li>
            <li>{t('settings.proxy.tips3')}</li>
          </ul>
        </div>
      </SettingsSection>
      </>)}

      {tab === 'tagging' && (<>
      {/* LLMTaggerWorkspace 自带 card；title 渲染在 card 内最顶部跟 WD14/CLTagger 视觉对齐。
       * 外层 div 只承担 id（给 section index 滚动定位用）+ scroll-mt-24 锚点偏移。 */}
      <div id="llm-tagger" className="scroll-mt-24">
        <LLMTaggerWorkspace
          title="LLM Tagger"
          currentPreset={currentPreset}
          serverCurrentPreset={serverCurrentPreset}
          presets={draft.llm_tagger.presets}
          currentPresetId={draft.llm_tagger.current_preset}
          onSelectPreset={(id) => update('llm_tagger', 'current_preset', id)}
          onUpdatePreset={updatePreset}
          onResetToBuiltin={resetCurrentPresetToBuiltin}
          onSaveAs={saveAsNewPreset}
          onAddPreset={addPreset}
          onDeletePreset={deleteCurrentPreset}
          llmModelsBusy={llmModelsBusy}
          llmTestBusy={llmTestBusy}
          onRefreshModels={() => void refreshLLMModels()}
          onTestConnection={() => void testLLMConnection()}
        />
      </div>

      <SettingsSection id="wd14" title="WD14">
        <WD14ModelCard
          catalog={catalog}
          busy={downloadBusy}
          start={startDownload}
          currentModelId={draft.wd14.model_id}
          onSelectModelId={(id) => update('wd14', 'model_id', id)}
          candidates={draft.wd14.model_ids}
          onCandidatesChange={(next) => update('wd14', 'model_ids', next)}
          t={t}
        />
        <SettingsField label="local_dir" desc={t('settings.blankAutoHfDownload')}>
          <SettingsInput
            type="text"
            value={draft.wd14.local_dir ?? ''}
            onChange={(v) => update('wd14', 'local_dir', v || null)}
            className={textInputClass}
          />
        </SettingsField>
        <div className="grid grid-cols-2 gap-3">
          <SettingsField label="threshold_general">
            <SettingsInput
              type="number" step="0.01" min={0} max={1}
              value={draft.wd14.threshold_general}
              onChange={(v) => update('wd14', 'threshold_general', Number(v))}
              className={`${textInputClass} max-w-32`}
            />
          </SettingsField>
          <SettingsField label="threshold_character">
            <SettingsInput
              type="number" step="0.01" min={0} max={1}
              value={draft.wd14.threshold_character}
              onChange={(v) => update('wd14', 'threshold_character', Number(v))}
              className={`${textInputClass} max-w-32`}
            />
          </SettingsField>
        </div>
        <SettingsField label="blacklist_tags" desc={t('settings.commaSeparated')}>
          <TagListInput
            value={draft.wd14.blacklist_tags}
            onChange={(tags) => update('wd14', 'blacklist_tags', tags)}
            className={textInputClass}
          />
        </SettingsField>
        <SettingsField label="batch_size" desc={t('settings.batchSizeHint')}>
          <SettingsInput
            type="number" min={1} max={64}
            value={draft.wd14.batch_size}
            onChange={(v) => update('wd14', 'batch_size', Math.max(1, Number(v) || 1))}
            className={`${textInputClass} max-w-24`}
          />
        </SettingsField>
      </SettingsSection>

      <SettingsSection id="cltagger" title="CLTagger">
        <CLTaggerModelCard
          catalog={catalog}
          busy={downloadBusy}
          start={startDownload}
          currentModelPath={draft.cltagger.model_path}
          currentTagMappingPath={draft.cltagger.tag_mapping_path}
          onSelectVariant={(v: CLTaggerVariantInfo) => {
            update('cltagger', 'model_path', v.model_path)
            update('cltagger', 'tag_mapping_path', v.tag_mapping_path)
          }}
          modelId={draft.cltagger.model_id}
          onModelIdChange={(id) => update('cltagger', 'model_id', id)}
          t={t}
        />
        <SettingsField label="local_dir" desc={t('settings.blankAutoHfDownload')}>
          <SettingsInput
            type="text"
            value={draft.cltagger.local_dir ?? ''}
            onChange={(v) => update('cltagger', 'local_dir', v || null)}
            className={textInputClass}
          />
        </SettingsField>
        <div className="grid grid-cols-2 gap-3">
          <SettingsField label="threshold_general">
            <SettingsInput
              type="number" step="0.01" min={0} max={1}
              value={draft.cltagger.threshold_general}
              onChange={(v) => update('cltagger', 'threshold_general', Number(v))}
              className={`${textInputClass} max-w-32`}
            />
          </SettingsField>
          <SettingsField label="threshold_character">
            <SettingsInput
              type="number" step="0.01" min={0} max={1}
              value={draft.cltagger.threshold_character}
              onChange={(v) => update('cltagger', 'threshold_character', Number(v))}
              className={`${textInputClass} max-w-32`}
            />
          </SettingsField>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <SettingsField label="add_copyright_tag">
            <Bool value={draft.cltagger.add_copyright_tag} onChange={(v) => update('cltagger', 'add_copyright_tag', v)} />
          </SettingsField>
          <SettingsField label="add_meta_tag">
            <Bool value={draft.cltagger.add_meta_tag} onChange={(v) => update('cltagger', 'add_meta_tag', v)} />
          </SettingsField>
          <SettingsField label="add_model_tag">
            <Bool value={draft.cltagger.add_model_tag} onChange={(v) => update('cltagger', 'add_model_tag', v)} />
          </SettingsField>
          <SettingsField label="add_rating_tag">
            <Bool value={draft.cltagger.add_rating_tag} onChange={(v) => update('cltagger', 'add_rating_tag', v)} />
          </SettingsField>
          <SettingsField label="add_quality_tag">
            <Bool value={draft.cltagger.add_quality_tag} onChange={(v) => update('cltagger', 'add_quality_tag', v)} />
          </SettingsField>
        </div>
        <SettingsField label="blacklist_tags" desc={t('settings.commaSeparated')}>
          <TagListInput
            value={draft.cltagger.blacklist_tags}
            onChange={(tags) => update('cltagger', 'blacklist_tags', tags)}
            className={textInputClass}
          />
        </SettingsField>
        <SettingsField label="batch_size" desc={t('settings.batchSizeHint')}>
          <SettingsInput
            type="number" min={1} max={64}
            value={draft.cltagger.batch_size}
            onChange={(v) => update('cltagger', 'batch_size', Math.max(1, Number(v) || 1))}
            className={`${textInputClass} max-w-24`}
          />
        </SettingsField>
      </SettingsSection>

      <ONNXRuntimeSection />
      <TagDictionarySection />
      </>)}

      {tab === 'training' && (<>
      <SettingsSection id="download-source" title={t('settings.modelSource')}>
        <SettingsField
          label={t('settings.downloadSource')}
          helpTooltip={
            <p>{t('settings.downloadSourceHelp')}</p>
          }
        >
          <DownloadSourceSelect
            value={draft.download_source}
            onChange={(v) => updateTop('download_source', v)}
          />
        </SettingsField>

        {/* 下方按当前下载源条件渲染对应凭证配置。HF/ModelScope token 都保留在
         * secrets 里（即便切换源也不丢失），只是 UI 一次只露面一份。 */}
        {draft.download_source === 'huggingface' ? (
          <>
            <SettingsField
              label="token"
              helpTooltip={
                <p>{t('settings.hfTokenHelp')}</p>
              }
            >
              <SensitiveInput
                value={draft.huggingface.token}
                serverValue={server?.huggingface.token ?? ''}
                onChange={(v) => update('huggingface', 'token', v)}
              />
            </SettingsField>
            <SettingsField
              label="endpoint"
              helpTooltip={<p>{t('settings.hfEndpointHelp')}</p>}
            >
              <HFEndpointSelect
                value={draft.huggingface.endpoint}
                onChange={(v) => update('huggingface', 'endpoint', v)}
              />
            </SettingsField>
          </>
        ) : (
          <SettingsField
            label="token"
            helpTooltip={
              <>
                <p>{t('settings.modelscopeTokenHelp')}</p>
                <p><Trans i18nKey="settings.modelscopeInstallHelp" components={{ code: <code /> }} /></p>
              </>
            }
          >
            <SensitiveInput
              value={draft.modelscope.token}
              serverValue={server?.modelscope.token ?? ''}
              onChange={(v) => update('modelscope', 'token', v)}
            />
          </SettingsField>
        )}
      </SettingsSection>

      <SettingsSection id="queue" title={t('settings.queueSchedule')}>
        <SettingsField label={t('settings.allowGpuDuringTrain')}>
          <div className="flex items-center gap-3">
            <Bool value={draft.queue.allow_gpu_during_train} onChange={(v) => update('queue', 'allow_gpu_during_train', v)} />
            <span className="text-xs text-warn">
              {t('settings.allowGpuDuringTrainHint')}
            </span>
          </div>
        </SettingsField>
      </SettingsSection>

      <PyTorchSection />

      <FlashAttentionSection />

      <XformersSection />

      <ModelsSection
        catalog={catalog}
        busy={downloadBusy}
        start={startDownload}
        reloadCatalog={reloadCatalog}
        catalogError={catalogError}
        t={t}
      />
      </>)}

      {tab === 'monitor' && (<>
      <SettingsSection id="wandb" title="Weights & Biases">
        <SettingsField label={t('settings.enableWandb')} desc={t('settings.enableWandbHint')}>
          <Bool value={draft.wandb.enabled} onChange={(v) => update('wandb', 'enabled', v)} />
        </SettingsField>
        <SettingsField label="api_key">
          <SensitiveInput
            value={draft.wandb.api_key}
            serverValue={server?.wandb.api_key ?? ''}
            onChange={(v) => update('wandb', 'api_key', v)}
          />
        </SettingsField>
        <SettingsField label="project">
          <SettingsInput
            type="text"
            value={draft.wandb.project}
            onChange={(v) => update('wandb', 'project', v)}
            placeholder="AnimaLoraStudio"
            className={textInputClass}
          />
        </SettingsField>
        <SettingsField label="entity" desc={t('settings.wandbEntityHint')}>
          <SettingsInput
            type="text"
            value={draft.wandb.entity}
            onChange={(v) => update('wandb', 'entity', v)}
            className={textInputClass}
          />
        </SettingsField>
        <SettingsField label="base_url" desc={t('settings.wandbBaseUrlHint')}>
          <SettingsInput
            type="text"
            value={draft.wandb.base_url}
            onChange={(v) => update('wandb', 'base_url', v)}
            placeholder="https://api.wandb.ai"
            className={textInputClass}
          />
        </SettingsField>
        <div className="grid grid-cols-2 gap-3">
          <SettingsField label="mode">
            <select
              value={draft.wandb.mode}
              onChange={(e) => update('wandb', 'mode', e.target.value as WandBConfig['mode'])}
              className={textInputClass}
            >
              <option value="online">online</option>
              <option value="offline">offline</option>
              <option value="disabled">disabled</option>
            </select>
          </SettingsField>
          <SettingsField
            label={t('settings.logSamples')}
            helpTooltip={
              <p><Trans i18nKey="settings.logSamplesHelp" components={{ code: <code /> }} /></p>
            }
          >
            <Bool value={draft.wandb.log_samples} onChange={(v) => update('wandb', 'log_samples', v)} />
          </SettingsField>
        </div>
        {draft.wandb.log_samples && (
          <div className="grid grid-cols-2 gap-3">
            <SettingsField
              label={t('settings.sampleMaxSide')}
              helpTooltip={<p>{t('settings.sampleMaxSideHelp')}</p>}
            >
              <SettingsInput
                type="number"
                min={64}
                step={64}
                value={draft.wandb.sample_max_side}
                onChange={(v) => update('wandb', 'sample_max_side', Math.max(64, parseInt(v) || 1216))}
                className={textInputClass}
              />
            </SettingsField>
            <SettingsField
              label={t('settings.sampleEveryNSteps')}
              helpTooltip={
                <p><Trans i18nKey="settings.sampleEveryNStepsHelp" components={{ code: <code /> }} /></p>
              }
            >
              <SettingsInput
                type="number"
                min={0}
                step={50}
                value={draft.wandb.sample_every_n_steps}
                onChange={(v) => update('wandb', 'sample_every_n_steps', Math.max(0, parseInt(v) || 0))}
                className={textInputClass}
              />
            </SettingsField>

            <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mt-4 mb-2">
              {t('settings.uploadArtifacts')}
            </h4>
            <SettingsField
              label={t('settings.uploadModel')}
              helpTooltip={<p>{t('settings.uploadModelHelp')}</p>}
            >
              <div className="flex items-center gap-3">
                <Bool value={draft.wandb.upload_model} onChange={(v) => update('wandb', 'upload_model', v)} />
                {draft.wandb.upload_model && (
                  <select
                    value={draft.wandb.upload_model_policy}
                    onChange={(e) => update('wandb', 'upload_model_policy', e.target.value as 'all' | 'last')}
                    className={textInputClass + ' w-auto'}
                  >
                    <option value="last">{t('settings.policyLast')}</option>
                    <option value="all">{t('settings.policyAll')}</option>
                  </select>
                )}
              </div>
            </SettingsField>
            <SettingsField
              label={t('settings.uploadStateManual')}
              helpTooltip={<p>{t('settings.uploadStateManualHelp')}</p>}
            >
              <div className="flex items-center gap-3">
                <Bool value={draft.wandb.upload_state_manual} onChange={(v) => update('wandb', 'upload_state_manual', v)} />
                {draft.wandb.upload_state_manual && (
                  <select
                    value={draft.wandb.upload_state_manual_policy}
                    onChange={(e) => update('wandb', 'upload_state_manual_policy', e.target.value as 'all' | 'last')}
                    className={textInputClass + ' w-auto'}
                  >
                    <option value="last">{t('settings.policyLast')}</option>
                    <option value="all">{t('settings.policyAll')}</option>
                  </select>
                )}
              </div>
            </SettingsField>
            <SettingsField
              label={t('settings.uploadStateAuto')}
              helpTooltip={<p>{t('settings.uploadStateAutoHelp')}</p>}
            >
              <div className="flex items-center gap-3">
                <Bool value={draft.wandb.upload_state_auto} onChange={(v) => update('wandb', 'upload_state_auto', v)} />
                {draft.wandb.upload_state_auto && (
                  <select
                    value={draft.wandb.upload_state_auto_policy}
                    onChange={(e) => update('wandb', 'upload_state_auto_policy', e.target.value as 'all' | 'last')}
                    className={textInputClass + ' w-auto'}
                  >
                    <option value="last">{t('settings.policyLast')}</option>
                    <option value="all">{t('settings.policyAll')}</option>
                  </select>
                )}
              </div>
            </SettingsField>
          </div>
        )}
      </SettingsSection>
      </>)}

      {tab === 'preprocess' && (
        <UpscalerSection
          catalog={catalog}
          busy={downloadBusy}
          start={startDownload}
          reloadCatalog={reloadCatalog}
          t={t}
        />
      )}

      {tab === 'testing' && (<>
        {/* Test generation uses the server Comfy-style runtime. Attention backend
            uses global auto-detect; advanced users may override
            secrets.generate.attention_backend (flash_attn / xformers / none).
            Only xformers is an exact ComfyUI KSampler parity target. */}
        <IdleTimeoutSection draft={draft} update={update} />
        <TaeFluxSection draft={draft} update={update} />
        <SaveTestImagesSection draft={draft} update={update} />
      </>)}

      {tab === 'appearance' && (
        <DisplaySection />
      )}

      {tab === 'system' && (
        <SystemSection />
      )}

    </div>

    <SectionIndex sections={TAB_SECTIONS[tab]} scrollContainer={scrollContainerRef} />
    </div>
    </div>
    </div>
  )
}

// ── Section / Field ────────────────────────────────────────────────────────

function SettingsSection({
  id, title, headerExtras, children,
}: {
  id?: string
  title: string
  headerExtras?: React.ReactNode  // 可选 slot：渲染在 h2 右侧（紧贴），给 ⓘ tooltip 之类用
  children: React.ReactNode
}) {
  const titleEl = <h2 className="text-sm font-semibold text-fg-primary">{title}</h2>
  return (
    <section id={id} className="rounded-md border border-subtle bg-surface p-4 flex flex-col gap-3 scroll-mt-24">
      {headerExtras ? (
        <div className="flex items-center gap-2 mb-0.5">
          {titleEl}
          {headerExtras}
        </div>
      ) : (
        <div className="mb-0.5">{titleEl}</div>
      )}
      {children}
    </section>
  )
}

/**
 * 右侧 sticky section 目录。基于 IntersectionObserver 在 scrollContainer 视口内
 * 跟踪当前可见 section，并提供点击平滑滚动。
 *
 * rootMargin 调整为顶部 -20%、底部 -70%：让"当前可见"判定集中在视口偏上区域，
 * 滚动时高亮跟随更自然（用户视线在 viewport 上 1/3 处）。
 */
function SectionIndex({
  sections,
  scrollContainer,
}: {
  sections: { id: string; labelKey: string }[]
  scrollContainer: RefObject<HTMLDivElement>
}) {
  const { t } = useTranslation()
  const [active, setActive] = useState<string>(sections[0]?.id ?? '')

  useEffect(() => {
    // 切换 tab 后重置 active 到第一条
    setActive(sections[0]?.id ?? '')
  }, [sections])

  useEffect(() => {
    const root = scrollContainer.current
    if (!root || sections.length === 0) return
    // jsdom（vitest 环境）没有 IntersectionObserver；非浏览器环境直接跳过。
    if (typeof IntersectionObserver === 'undefined') return
    const observers: IntersectionObserver[] = []
    // 收集 (id, top) 用来在 onIntersect 时挑当前最靠上的可见 section
    const visible = new Set<string>()
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) visible.add(e.target.id)
          else visible.delete(e.target.id)
        }
        // 按 sections 顺序取第一个可见的作为 active
        const next = sections.find((s) => visible.has(s.id))
        if (next) setActive(next.id)
      },
      { root, rootMargin: '-20% 0px -70% 0px', threshold: 0 },
    )
    sections.forEach((s) => {
      const el = document.getElementById(s.id)
      if (el) obs.observe(el)
    })
    observers.push(obs)
    return () => observers.forEach((o) => o.disconnect())
  }, [sections, scrollContainer])

  const onJump = (id: string) => {
    const el = document.getElementById(id)
    if (!el) return
    el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    setActive(id)
  }

  return (
    <aside className="hidden lg:block">
      <nav className="sticky top-4 flex flex-col gap-0.5">
        <div className="caption mb-2 px-2">{t('settings.pageIndex')}</div>
        {sections.map((s) => (
          <button
            key={s.id}
            onClick={() => onJump(s.id)}
            className={`text-left text-xs px-2 py-1.5 rounded-sm transition-colors border-l-2 ${
              active === s.id
                ? 'border-accent text-accent bg-accent-soft/40'
                : 'border-transparent text-fg-tertiary hover:text-fg-secondary hover:bg-overlay/40'
            }`}
          >
            {t(s.labelKey)}
          </button>
        ))}
      </nav>
    </aside>
  )
}

function SettingsField({ label, desc, helpTooltip, children }: {
  label: string
  desc?: string
  /** 可选 ⓘ tooltip slot，渲染在 label 旁边。中长说明（≥20 字 / 详细用法）
   *  适合放这里，避免 inline desc 把字段名行撑得过长。一般和 desc 二选一。 */
  helpTooltip?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="grid grid-cols-[240px_1fr] gap-3 items-start">
      <div className="flex flex-col gap-0.5 pt-1.5">
        <div className="flex items-center gap-2 min-w-0">
          <label className="text-xs text-fg-secondary font-mono leading-none">{label}</label>
          {helpTooltip && <InfoButton>{helpTooltip}</InfoButton>}
        </div>
        {desc && <p className="text-[10px] text-fg-tertiary m-0 leading-snug">{desc}</p>}
      </div>
      <div className="min-w-0">{children}</div>
    </div>
  )
}

function Bool({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <input
      type="checkbox"
      checked={value}
      onChange={(e) => onChange(e.target.checked)}
      className="w-4 h-4"
      style={{ accentColor: 'var(--accent)' }}
    />
  )
}

function SensitiveInput({ value, serverValue, onChange }: {
  value: string; serverValue: string; onChange: (v: string) => void
}) {
  const { t } = useTranslation()
  const [localValue, setLocalValue] = useState(value)

  useEffect(() => {
    setLocalValue(value)
  }, [value])

  const masked = localValue === MASK

  const handleBlur = () => {
    if (localValue !== value) {
      onChange(localValue)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.currentTarget.blur()
    }
  }

  return (
    <input
      type="password"
      value={masked ? '' : localValue}
      placeholder={serverValue === MASK ? t('settings.sensitiveSavedPlaceholder') : ''}
      onChange={(e) => setLocalValue(e.target.value || MASK)}
      onBlur={handleBlur}
      onKeyDown={handleKeyDown}
      autoComplete="new-password"
      data-lpignore="true"
      data-1p-ignore
      data-form-type="other"
      className={textInputClass}
    />
  )
}

interface SettingsInputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange'> {
  value: string | number
  onChange: (v: string) => void
}

function SettingsInput({ value, onChange, type = 'text', ...props }: SettingsInputProps) {
  const [localValue, setLocalValue] = useState(value)

  useEffect(() => {
    setLocalValue(value)
  }, [value])

  const handleBlur = () => {
    if (String(localValue) !== String(value)) {
      onChange(String(localValue))
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.currentTarget.blur()
    }
  }

  return (
    <input
      type={type}
      {...props}
      value={localValue}
      onChange={(e) => setLocalValue(e.target.value)}
      onBlur={handleBlur}
      onKeyDown={handleKeyDown}
    />
  )
}

// ── HFEndpointSelect ────────────────────────────────────────────────────────
//
// HF 模型下载 endpoint 选择器：preset + 自定义 URL 输入。
// 0.8.2 hotfix：hf-mirror.com preset 暂时隐藏（服务端 redirect 改动后所有
// huggingface_hub 版本均失败，详见 docs/todo/hf-mirror-recheck.md）。endpoint
// 字段本身仍接受任意 URL，用户可通过「自定义 URL」粘贴 hf-mirror / sjtug /
// 腾讯镜像 / 自建反代。复活后把 preset 加回来即可。

const HF_ENDPOINT_PRESETS: { value: string; label: string; hintKey: string }[] = [
  { value: '', label: 'huggingface.co', hintKey: 'settings.hfOfficialHint' },
  { value: '__custom__', label: 'Custom URL...', hintKey: 'settings.hfCustomHint' },
]

function HFEndpointSelect({ value, onChange }: {
  value: string; onChange: (v: string) => void
}) {
  const { t } = useTranslation()
  const isPreset = HF_ENDPOINT_PRESETS.some(p => p.value !== '__custom__' && p.value === value)
  const [mode, setMode] = useState<'preset' | 'custom'>(isPreset ? 'preset' : 'custom')
  const selectedPreset = isPreset
    ? value
    : (mode === 'custom' ? '__custom__' : '')

  return (
    <div className="flex flex-col gap-1.5">
      <select
        value={selectedPreset}
        onChange={(e) => {
          const v = e.target.value
          if (v === '__custom__') {
            setMode('custom')
            // 不清当前值，让用户在下方输入
          } else {
            setMode('preset')
            onChange(v)
          }
        }}
        className={`${textInputClass} max-w-md`}
      >
        {HF_ENDPOINT_PRESETS.map(p => (
          <option key={p.value} value={p.value}>
            {p.label}{p.hintKey ? ` — ${t(p.hintKey)}` : ''}
          </option>
        ))}
      </select>
      {mode === 'custom' && (
        <input
          type="text"
          value={value && !isPreset ? value : ''}
          placeholder="https://your-mirror.example.com"
          onChange={(e) => onChange(e.target.value.trim())}
          className={`${textInputClass} max-w-md`}
        />
      )}
    </div>
  )
}

// ── DownloadSourceSelect ────────────────────────────────────────────────────

function DownloadSourceSelect({ value, onChange }: {
  value: string; onChange: (v: string) => void
}) {
  const { t } = useTranslation()
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`${textInputClass} max-w-xs`}
    >
      <option value="huggingface">{t('settings.downloadSourceHuggingface')}</option>
      <option value="modelscope">{t('settings.downloadSourceModelscope')}</option>
    </select>
  )
}

// ── ModelIdsEditor ──────────────────────────────────────────────────────────

function ModelIdsEditor({ ids, currentId, onChange }: {
  ids: string[]; currentId: string; onChange: (next: string[]) => void
}) {
  const { t } = useTranslation()
  const [draft, setDraft] = useState('')
  const seen = new Set(ids)

  const add = () => {
    const v = draft.trim()
    if (!v) return
    if (seen.has(v)) { setDraft(''); return }
    onChange([...ids, v])
    setDraft('')
  }
  const remove = (m: string) => {
    if (m === currentId) return
    onChange(ids.filter((x) => x !== m))
  }

  return (
    <div className="flex flex-col gap-1.5">
      <ul className="flex flex-col gap-1 list-none m-0 p-0">
        {ids.map((m) => {
          const isCurrent = m === currentId
          return (
            <li key={m} className={`flex items-center gap-2 px-2 py-1 rounded-sm text-xs ${
              isCurrent ? 'border border-accent bg-accent-soft' : 'border border-subtle bg-sunken'
            }`}>
              <code className="font-mono text-fg-primary flex-1 min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">{m}</code>
              {isCurrent ? (
                <span className="text-xs text-accent">{t('settings.current')}</span>
              ) : (
                <button onClick={() => remove(m)} className="text-xs text-fg-tertiary hover:text-err bg-transparent border-none cursor-pointer transition-colors">×</button>
              )}
            </li>
          )
        })}
      </ul>
      <div className="flex gap-1.5">
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); add() } }}
          placeholder={t('settings.addHfModelId')}
          className={`${textInputClass} flex-1`}                            />
        <button onClick={add} disabled={!draft.trim() || seen.has(draft.trim())} className="btn btn-secondary btn-sm">{t('settings.add')}</button>
      </div>
    </div>
  )
}

// ── WD14 / CLTagger Model Cards（打标 tab 内嵌的模型管理器） ─────────────────

function WD14ModelCard({
  catalog, busy, start,
  currentModelId, onSelectModelId,
  candidates, onCandidatesChange, t,
}: {
  catalog: ModelsCatalog | null
  busy: Set<string>
  start: (model_id: string, variant?: string) => Promise<void>
  currentModelId: string
  onSelectModelId: (id: string) => void
  candidates: string[]
  onCandidatesChange: (next: string[]) => void
  t: TFunction
}) {
  const [advOpen, setAdvOpen] = useState(false)
  const wd14 = catalog?.wd14
  const wd14Description = translatedCatalogText(MODEL_DESCRIPTION_KEYS, 'wd14', wd14?.description, t)
  if (!wd14) {
    return <p className="text-fg-tertiary text-xs">{t('settings.loadingModelCatalog')}</p>
  }
  return (
    <ModelGroupCard
      title={t('settings.wd14CandidateTitle', { name: wd14.name })}
      helpTooltip={
        <p><Trans i18nKey="settings.wd14CandidateHelp" values={{ desc: wd14Description }} components={{ code: <code /> }} /></p>
      }
    >
      <ul className="list-none m-0 p-0 flex flex-col gap-1">
        {wd14.variants.map((v) => {
          const key = `wd14:${v.model_id}`
          const dl = catalog.downloads[key]
          const isSel = v.model_id === currentModelId
          return (
            <li key={v.model_id} className={`flex items-center gap-2 text-xs px-1.5 py-1 rounded-sm ${
              isSel ? 'bg-accent-soft border border-accent' : 'bg-transparent border border-transparent'
            }`}>
              <input type="radio" name="wd14_variant" checked={isSel}
                onChange={() => onSelectModelId(v.model_id)}
                className="shrink-0"
                style={{ accentColor: 'var(--accent)' }}
                title={t('settings.selectWd14ModelId')}
              />
              <code className="font-mono text-fg-primary flex-1 min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">{v.model_id}</code>
              <ModelStatusBadge
                exists={v.exists} size={v.size} status={dl?.status}
                fileCount={v.files.length}
                existsCount={v.files.filter((f) => f.exists).length}
              />
              <DownloadButton
                exists={v.exists} status={dl?.status} busy={busy.has(key)}
                onClick={() => void start('wd14', v.model_id)}
              />
            </li>
          )
        })}
      </ul>
      <button type="button" onClick={() => setAdvOpen(!advOpen)}
        className="btn btn-ghost btn-sm text-xs text-fg-tertiary self-start">
        {advOpen ? '▾' : '▸'} {t('settings.candidateEditor')}
      </button>
      {advOpen && (
        <ModelIdsEditor
          ids={candidates} currentId={currentModelId}
          onChange={onCandidatesChange}
        />
      )}
    </ModelGroupCard>
  )
}

function CLTaggerModelCard({
  catalog, busy, start,
  currentModelPath, currentTagMappingPath, onSelectVariant,
  modelId, onModelIdChange, t,
}: {
  catalog: ModelsCatalog | null
  busy: Set<string>
  start: (model_id: string, variant?: string) => Promise<void>
  currentModelPath: string
  currentTagMappingPath: string
  onSelectVariant: (v: CLTaggerVariantInfo) => void
  modelId: string
  onModelIdChange: (id: string) => void
  t: TFunction
}) {
  const [advOpen, setAdvOpen] = useState(false)
  const cl = catalog?.cltagger
  const clDescription = translatedCatalogText(MODEL_DESCRIPTION_KEYS, 'cltagger', cl?.description, t)
  if (!cl) {
    return <p className="text-fg-tertiary text-xs">{t('settings.loadingModelCatalog')}</p>
  }
  return (
    <ModelGroupCard
      title={t('settings.clTaggerVersionTitle', { name: cl.name })}
      helpTooltip={
        <p><Trans i18nKey="settings.repoHelp" values={{ desc: clDescription, repo: cl.repo }} components={{ code: <code /> }} /></p>
      }
    >
      <ul className="list-none m-0 p-0 flex flex-col gap-1">
        {cl.variants.map((v) => {
          const key = `cltagger:${v.label}`
          const dl = catalog.downloads[key]
          const isSel =
            v.model_path === currentModelPath &&
            v.tag_mapping_path === currentTagMappingPath
          return (
            <li key={v.label} className={`flex items-center gap-2 text-xs px-1.5 py-1 rounded-sm ${
              isSel ? 'bg-accent-soft border border-accent' : 'bg-transparent border border-transparent'
            }`}>
              <input type="radio" name="cltagger_variant" checked={isSel}
                onChange={() => onSelectVariant(v)}
                className="shrink-0"
                style={{ accentColor: 'var(--accent)' }}
                title={t('settings.selectClTaggerVersion')}
              />
              <code className="font-mono text-fg-primary flex-1 min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">{v.label}</code>
              <ModelStatusBadge
                exists={v.exists} size={v.size} status={dl?.status}
                fileCount={v.files.length}
                existsCount={v.files.filter((f) => f.exists).length}
              />
              <DownloadButton
                exists={v.exists} status={dl?.status} busy={busy.has(key)}
                onClick={() => void start('cltagger', v.label)}
              />
            </li>
          )
        })}
      </ul>
      <button type="button" onClick={() => setAdvOpen(!advOpen)}
        className="btn btn-ghost btn-sm text-xs text-fg-tertiary self-start">
        {advOpen ? '▾' : '▸'} {t('settings.customRepoAdvanced')}
      </button>
      {advOpen && (
        <SettingsField label="model_id">
          <input
            type="text"
            value={modelId}
            onChange={(e) => onModelIdChange(e.target.value)}
            className={textInputClass}
            placeholder="cella110n/cl_tagger"
          />
        </SettingsField>
      )}
    </ModelGroupCard>
  )
}

// 顶层非 object 字段（string / number / bool），直接比较后塞入 patch。
const TOP_LEVEL_SCALARS: (keyof Secrets)[] = ['download_source']

function buildPatch(draft: Secrets, server: Secrets): SecretsPatch {
  const out: Record<string, unknown> = {}
  for (const key of Object.keys(draft) as (keyof Secrets)[]) {
    if (TOP_LEVEL_SCALARS.includes(key)) {
      if (draft[key] !== server[key]) out[key] = draft[key]
      continue
    }
    const sub: Record<string, unknown> = {}
    const d = draft[key] as unknown as Record<string, unknown>
    const s = server[key] as unknown as Record<string, unknown>
    for (const k of Object.keys(d)) {
      const dv = d[k]
      const sv = s[k]
      if (dv === MASK) continue
      if (JSON.stringify(dv) !== JSON.stringify(sv)) sub[k] = dv
    }
    if (Object.keys(sub).length) out[key] = sub
  }
  return out as SecretsPatch
}

// ── Models Section ─────────────────────────────────────────────────────────

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}

function ModelsSection({ catalog, busy, start, reloadCatalog, catalogError, t }: {
  catalog: ModelsCatalog | null
  busy: Set<string>
  start: (model_id: string, variant?: string) => Promise<void>
  reloadCatalog: () => Promise<void>
  catalogError: string | null
  t: TFunction
}) {
  const { toast } = useToast()
  const [rootDraft, setRootDraft] = useState<string>('')
  const [serverRoot, setServerRoot] = useState<string | null>(null)
  const [savingRoot, setSavingRoot] = useState(false)
  const [selectedAnima, setSelectedAnima] = useState<string>('1.0')
  const [autoSyncPaths, setAutoSyncPaths] = useState<boolean>(true)
  const [savingAutoSync, setSavingAutoSync] = useState(false)
  const [secretsLoaded, setSecretsLoaded] = useState(false)

  // 一次性拉一份 secrets 取 models.root + selected_anima + auto_sync_paths
  // （这几项走独立 PUT，不进 SettingsPage 的全局 dirty 流程）。catalog 由父级注入。
  useEffect(() => {
    void api.getSecrets().then((sec) => {
      setServerRoot(sec.models?.root ?? null)
      setSelectedAnima(sec.models?.selected_anima ?? '1.0')
      setAutoSyncPaths(sec.models?.auto_sync_paths ?? true)
      setSecretsLoaded(true)
    }).catch(() => { setSecretsLoaded(true) })
  }, [])

  // secrets + catalog 都到位后，把输入框预填成「已保存值」或「实际默认绝对路径」。
  // 用 prev !== '' 当作"已初始化 / 用户已编辑"的标志，避免覆盖用户输入。
  useEffect(() => {
    if (!secretsLoaded || !catalog) return
    setRootDraft((prev) => (prev !== '' ? prev : (serverRoot ?? catalog.models_root ?? '')))
  }, [secretsLoaded, catalog, serverRoot])

  const pickAnima = async (variant: string) => {
    if (variant === selectedAnima) return
    setSelectedAnima(variant)
    try {
      await api.updateSecrets({ models: { selected_anima: variant } })
      toast(t('settings.mainModelSelected', { name: variant }), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
      void reloadCatalog()
    }
  }

  const saveRoot = async () => {
    const v = rootDraft.trim()
    setSavingRoot(true)
    try {
      await api.updateSecrets({ models: { root: v ? v : null } })
      toast(v ? t('settings.modelRootSaved', { path: v }) : t('settings.modelRootDefault'), 'success')
      setServerRoot(v ? v : null)
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setSavingRoot(false)
    }
  }

  const saveAutoSync = async (next: boolean) => {
    setSavingAutoSync(true)
    const prev = autoSyncPaths
    setAutoSyncPaths(next)
    try {
      await api.updateSecrets({ models: { auto_sync_paths: next } })
      toast(next ? t('settings.autoSyncPathsOn') : t('settings.autoSyncPathsOff'), 'success')
    } catch (e) {
      setAutoSyncPaths(prev)
      toast(String(e), 'error')
    } finally {
      setSavingAutoSync(false)
    }
  }

  const rootDirty = rootDraft.trim() !== (serverRoot ?? '')
  const error = catalogError

  return (
    <SettingsSection id="models" title={t('settings.trainingModelsOneClick')}>
      <SettingsField label={t('settings.modelsRoot')}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <input
            type="text"
            value={rootDraft}
            onChange={(e) => setRootDraft(e.target.value)}
            className={`${textInputClass} flex-1`}                                  />
          <button onClick={saveRoot} disabled={!rootDirty || savingRoot} className="btn btn-primary btn-sm"
            title={rootDirty ? t('settings.savePathConfig') : t('settings.notModified')}>
            {savingRoot ? t('common.saving') : t('settings.savePath')}
          </button>
          <button onClick={() => setRootDraft(serverRoot ?? (catalog?.models_root ?? ''))} disabled={!rootDirty || savingRoot}
            className="px-2 py-0.5 text-fg-tertiary bg-transparent border-none cursor-pointer rounded-sm"
            style={{ opacity: !rootDirty ? 0.3 : 1 }}
          >↻</button>
        </div>
      </SettingsField>

      <SettingsField
        label={t('settings.autoSyncPathsLabel')}
        helpTooltip={<p>{t('settings.autoSyncPathsHelp')}</p>}
      >
        <label className="flex items-center gap-2 pt-1.5">
          <input
            type="checkbox"
            checked={autoSyncPaths}
            onChange={(e) => void saveAutoSync(e.target.checked)}
            disabled={savingAutoSync}
            style={{ height: 16, width: 16 }}
          />
        </label>
      </SettingsField>

      {error && <div className="text-err text-xs font-mono">{error}</div>}
      {!catalog ? (
        <p className="text-fg-tertiary text-xs">{t('settings.loadingModelCatalog')}</p>
      ) : (
        <div className="flex flex-col gap-2">
          {/* Anima 主模型 */}
          <ModelGroupCard
            title={catalog.anima_main.name}
            helpTooltip={
              <>
                <p><Trans i18nKey="settings.repoHelp" values={{ desc: translatedCatalogText(MODEL_DESCRIPTION_KEYS, 'anima_main', catalog.anima_main.description, t), repo: catalog.anima_main.repo }} components={{ code: <code /> }} /></p>
                <p><Trans i18nKey="settings.defaultTransformerHelp" components={{ strong: <strong /> }} /></p>
              </>
            }
          >
            <ul className="list-none m-0 p-0 flex flex-col gap-1">
              {catalog.anima_main.variants.map((v) => {
                const key = `anima_main:${v.variant}`
                const dl = catalog.downloads[key]
                const isSel = v.variant === selectedAnima
                const canSelect = v.exists && dl?.status !== 'running'
                return (
                  <li key={v.variant} className={`flex items-center gap-2 text-xs px-1.5 py-1 rounded-sm ${
                    isSel ? 'bg-accent-soft border border-accent' : 'bg-transparent border border-transparent'
                  }`}>
                    <input type="radio" name="anima_variant" checked={isSel} disabled={!canSelect}
                      onChange={() => void pickAnima(v.variant)}
                      className="shrink-0"
                      style={{ accentColor: 'var(--accent)' }}
                      title={canSelect ? t('settings.selectDefaultMainModel') : v.exists ? t('settings.downloadInProgress') : t('settings.downloadRequiredFirst')}
                    />
                    <code className="font-mono text-fg-primary w-32 shrink-0">{v.variant}</code>
                    <ModelStatusBadge exists={v.exists} size={v.size} status={dl?.status} />
                    <span style={{ flex: 1 }} />
                    <DownloadButton exists={v.exists} status={dl?.status} busy={busy.has(key)} onClick={() => void start('anima_main', v.variant)} />
                  </li>
                )
              })}
            </ul>
          </ModelGroupCard>

          {/* VAE */}
          <ModelGroupCard title={catalog.anima_vae.name}>
            <div className="flex items-center gap-2 text-xs">
              <span className="text-fg-tertiary">{translatedCatalogText(MODEL_DESCRIPTION_KEYS, 'anima_vae', catalog.anima_vae.description, t)} · <code>{catalog.anima_vae.repo}</code></span>
              <span style={{ flex: 1 }} />
              <ModelStatusBadge exists={catalog.anima_vae.exists} size={catalog.anima_vae.size} status={catalog.downloads.anima_vae?.status} />
              <DownloadButton exists={catalog.anima_vae.exists} status={catalog.downloads.anima_vae?.status} busy={busy.has('anima_vae')} onClick={() => void start('anima_vae')} />
            </div>
          </ModelGroupCard>

          {/* Qwen3 + T5（CLTagger 已挪到「打标」tab） */}
          {(['qwen3', 't5_tokenizer'] as const).map((id) => {
            const m = catalog[id]
            const dl = catalog.downloads[id]
            const allExist = m.files.every((f) => f.exists)
            const totalSize = m.files.reduce((s, f) => s + f.size, 0)
            return (
              <ModelGroupCard key={id} title={m.name}>
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-fg-tertiary">{translatedCatalogText(MODEL_DESCRIPTION_KEYS, id, m.description, t)} · <code>{m.repo}</code></span>
                  <span style={{ flex: 1 }} />
                  <ModelStatusBadge exists={allExist} size={totalSize} status={dl?.status} fileCount={m.files.length} existsCount={m.files.filter((f) => f.exists).length} />
                  <DownloadButton exists={allExist} status={dl?.status} busy={busy.has(id)} onClick={() => void start(id)} />
                </div>
              </ModelGroupCard>
            )
          })}

          {/* 下载日志 */}
          {Object.values(catalog.downloads).filter((d) => d.status === 'running' || d.status === 'failed').length > 0 && (
            <details className="text-xs">
              <summary className="cursor-pointer text-fg-tertiary">
                {t('settings.downloadLogs', { n: Object.values(catalog.downloads).filter((d) => d.status === 'running' || d.status === 'failed').length })}
              </summary>
              <div className="mt-1 flex flex-col gap-2">
                {Object.values(catalog.downloads).map((d) => (
                  <div key={d.key} className="rounded-sm border border-subtle bg-sunken p-2">
                    <div className="flex items-center gap-2 mb-1">
                      <code className="font-mono text-fg-secondary">{d.key}</code>
                      <ModelStatusBadge exists={d.status === 'done'} size={0} status={d.status} />
                      {d.message && <span className="text-err overflow-hidden text-ellipsis whitespace-nowrap">{d.message}</span>}
                    </div>
                    <pre className="text-xs font-mono text-fg-tertiary max-h-32 overflow-auto whitespace-pre-wrap m-0">
                      {d.log_tail.join('\n') || t('settings.emptyLog')}
                    </pre>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
    </SettingsSection>
  )
}

function UpscalerSection({
  catalog, busy, start, reloadCatalog, t,
}: {
  catalog: ModelsCatalog | null
  busy: Set<string>
  start: (model_id: string, variant?: string) => Promise<void>
  reloadCatalog: () => Promise<void>
  t: TFunction
}) {
  const { toast } = useToast()
  const [customSource, setCustomSource] = useState<'hf' | 'ms'>('hf')
  const [customRepo, setCustomRepo] = useState('')
  const [customFile, setCustomFile] = useState('')
  const [customBusy, setCustomBusy] = useState(false)

  const pickUpscaler = async (label: string) => {
    try {
      await api.selectUpscaler(label)
      toast(t('settings.defaultUpscaler', { name: label }), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const submitCustom = async () => {
    const repo = customRepo.trim()
    const file = customFile.trim()
    if (!repo || !file) {
      toast(t('settings.repoAndFilenameRequired'), 'error')
      return
    }
    setCustomBusy(true)
    try {
      await api.startUpscalerCustomDownload({
        source: customSource, repo_id: repo, filename: file,
      })
      toast(t('settings.downloadStarted', { name: file }), 'success')
      setCustomRepo('')
      setCustomFile('')
      // SSE 推 model_download_changed 会刷 catalog；这里兜底
      setTimeout(() => void reloadCatalog(), 1500)
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setCustomBusy(false)
    }
  }

  const variants = catalog?.upscalers?.variants ?? []
  const current = catalog?.upscalers?.current ?? ''

  return (
    <SettingsSection id="upscalers" title={t('settings.upscalersPreprocess')}>
      {!catalog ? (
        <p className="text-fg-tertiary text-xs">{t('common.loading')}</p>
      ) : (
        <div className="flex flex-col gap-2">
          <ModelGroupCard
            title={t('settings.availableUpscalers')}
            helpTooltip={
              <>
                <p><Trans i18nKey="settings.upscalersHelpPath" values={{ path: catalog.upscalers?.target_dir }} components={{ code: <code /> }} /></p>
                <p>{t('settings.upscalersHelpDefault')}</p>
              </>
            }
          >
            <ul className="list-none m-0 p-0 flex flex-col gap-1">
              {variants.map((v) => {
                const key = v.kind === 'custom'
                  ? `upscaler:custom:${v.filename}`
                  : `upscaler:${v.label}`
                const dl = catalog.downloads[key]
                const isSel = v.label === current
                const canSelect = v.exists && dl?.status !== 'running'
                return (
                  <li key={v.label} className={`flex items-center gap-2 text-xs px-1.5 py-1 rounded-sm ${
                    isSel ? 'bg-accent-soft border border-accent' : 'bg-transparent border border-transparent'
                  }`}>
                    <input
                      type="radio"
                      name="selected_upscaler"
                      checked={isSel}
                      disabled={!canSelect}
                      onChange={() => void pickUpscaler(v.label)}
                      className="shrink-0"
                      style={{ accentColor: 'var(--accent)' }}
                      title={canSelect ? t('settings.selectDefaultPreprocess') : v.exists ? t('settings.downloadInProgress') : t('settings.notDownloaded')}
                    />
                    <div className="flex flex-col min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <code className="font-mono text-fg-primary truncate">{v.label}</code>
                        {v.kind === 'custom' && (
                          <span className="text-[10px] px-1 py-0 rounded-sm bg-sunken text-fg-tertiary">custom</span>
                        )}
                      </div>
                      <span className="text-fg-tertiary text-[11px] truncate">
                        {translatedCatalogText(UPSCALER_DESCRIPTION_KEYS, v.label, v.description, t)}
                        {v.hf_repo && <> · HF <code>{v.hf_repo}</code></>}
                        {v.ms_repo && <> · MS <code>{v.ms_repo}</code></>}
                        {v.size_mb != null && <> · ~{v.size_mb} MB</>}
                      </span>
                    </div>
                    <ModelStatusBadge exists={v.exists} size={v.size} status={dl?.status} />
                    {v.kind === 'preset' && (
                      <DownloadButton
                        exists={v.exists}
                        status={dl?.status}
                        busy={busy.has(`upscaler:${v.label}`)}
                        onClick={() => void start('upscaler', v.label)}
                      />
                    )}
                  </li>
                )
              })}
            </ul>
          </ModelGroupCard>

          <ModelGroupCard
            title={t('settings.customDownload')}
            helpTooltip={
              <>
                <p><Trans i18nKey="settings.customUpscalerHelpTypes" components={{ code: <code /> }} /></p>
                <p><Trans i18nKey="settings.customUpscalerHelpSources" components={{ code: <code /> }} /></p>
                <p>{t('settings.customUpscalerHelpEnable')}</p>
              </>
            }
          >
            <div className="flex flex-col gap-2 text-xs">
              <SettingsField label={t('settings.source')}>
                <select
                  value={customSource}
                  onChange={(e) => setCustomSource(e.target.value as 'hf' | 'ms')}
                  className="input text-xs"
                  style={{ width: 'auto' }}
                >
                  <option value="hf">HuggingFace</option>
                  <option value="ms">ModelScope</option>
                </select>
              </SettingsField>
              <SettingsField label={t('settings.repoId')}>
                <input
                  type="text"
                  value={customRepo}
                  onChange={(e) => setCustomRepo(e.target.value)}
                  placeholder={customSource === 'hf' ? 'Kim2091/UltraSharp' : 'libfishopen/upscaler'}
                  className={`${textInputClass} flex-1 font-mono`}
                />
              </SettingsField>
              <SettingsField label={t('common.filename')}>
                <input
                  type="text"
                  value={customFile}
                  onChange={(e) => setCustomFile(e.target.value)}
                  placeholder="4x-UltraSharp.pth"
                  className={`${textInputClass} flex-1 font-mono`}
                />
              </SettingsField>
              <div className="flex justify-end">
                <button
                  onClick={() => void submitCustom()}
                  disabled={customBusy || !customRepo.trim() || !customFile.trim()}
                  className="btn btn-primary btn-sm"
                >
                  {customBusy ? t('settings.downloadInProgress') : t('common.download')}
                </button>
              </div>
            </div>
          </ModelGroupCard>

          {/* 下载日志 */}
          {Object.values(catalog.downloads).filter((d) => d.key.startsWith('upscaler') && (d.status === 'running' || d.status === 'failed')).length > 0 && (
            <details className="text-xs">
              <summary className="cursor-pointer text-fg-tertiary">{t('settings.upscalerDownloadLogs')}</summary>
              <div className="mt-1 flex flex-col gap-2">
                {Object.values(catalog.downloads).filter((d) => d.key.startsWith('upscaler')).map((d) => (
                  <div key={d.key} className="rounded-sm border border-subtle bg-sunken p-2">
                    <div className="flex items-center gap-2 mb-1">
                      <code className="font-mono text-fg-secondary">{d.key}</code>
                      <ModelStatusBadge exists={d.status === 'done'} size={0} status={d.status} />
                      {d.message && <span className="text-err overflow-hidden text-ellipsis whitespace-nowrap">{d.message}</span>}
                    </div>
                    <pre className="text-xs font-mono text-fg-tertiary max-h-32 overflow-auto whitespace-pre-wrap m-0">
                      {d.log_tail.join('\n') || t('settings.emptyLog')}
                    </pre>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
    </SettingsSection>
  )
}

function ModelGroupCard({
  title, helpTooltip, children,
}: {
  title: string
  helpTooltip?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="rounded-sm border border-subtle bg-sunken p-2.5">
      <h4 className="text-xs font-semibold text-fg-primary mb-1.5 flex items-center gap-2">
        <span>{title}</span>
        {helpTooltip && <InfoButton>{helpTooltip}</InfoButton>}
      </h4>
      {children}
    </div>
  )
}

function ModelStatusBadge({ exists, size, status, fileCount, existsCount }: {
  exists: boolean; size: number; status?: ModelDownloadStatus['status']; fileCount?: number; existsCount?: number
}) {
  const { t } = useTranslation()
  if (status === 'running') {
    return <StatusLabel bg="bg-warn-soft" fg="text-warn" text={t('settings.downloadInProgress')} pulse />
  }
  if (status === 'failed') {
    return <StatusLabel bg="bg-err-soft" fg="text-err" text={t('status.failed')} />
  }
  if (exists) {
    return <StatusLabel bg="bg-ok-soft" fg="text-ok" text={`✓ ${fmtBytes(size)}${fileCount !== undefined ? ` (${existsCount}/${fileCount})` : ''}`} />
  }
  if (fileCount !== undefined && existsCount! > 0) {
    return <StatusLabel bg="bg-warn-soft" fg="text-warn" text={t('settings.partialFiles', { exists: existsCount, total: fileCount })} />
  }
  return <StatusLabel bg="bg-overlay" fg="text-fg-tertiary" text={t('settings.notDownloaded')} />
}

function StatusLabel({ bg, fg, text, pulse }: { bg: string; fg: string; text: string; pulse?: boolean }) {
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded-sm font-mono ${bg} ${fg}`}
      style={pulse ? { animation: 'pulse 1.5s infinite' } : undefined}
    >{text}</span>
  )
}

function DownloadButton({ exists, status, busy, onClick }: {
  exists: boolean; status?: ModelDownloadStatus['status']; busy: boolean; onClick: () => void
}) {
  const { t } = useTranslation()
  const running = status === 'running' || busy
  if (running) {
    return <button disabled className="btn btn-secondary btn-sm" style={{ opacity: 0.5 }}>...</button>
  }
  return (
    <button onClick={onClick} className={exists ? 'btn btn-secondary btn-sm' : 'btn btn-primary btn-sm'}
      title={exists ? t('settings.redownloadTitle') : t('common.download')}>
      {exists ? t('settings.redownload') : t('settings.downloadAction')}
    </button>
  )
}

// ── Tag 翻译词典：上传 / 恢复默认 / 全局 chip toggle ──────────────────────

function TagDictionarySection() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const dict = useTagDict()
  const [show, setShow] = useShowTagTranslation()
  const [busy, setBusy] = useState<null | 'reset' | 'upload'>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const meta = dict.meta
  const sourceLabel = meta?.kind === 'default'
    ? t('settings.tagDictionary.sourceDefault')
    : meta?.kind === 'user'
      ? t('settings.tagDictionary.sourceUser')
      : t('settings.tagDictionary.sourceUnknown')

  const downloadedAt = meta?.downloaded_at
    ? new Date(meta.downloaded_at * 1000).toLocaleString()
    : '—'

  const reset = async () => {
    if (busy) return
    setBusy('reset')
    try {
      await api.resetTagDictionary()
      await reloadDict()
      toast(t('settings.tagDictionary.resetOk'))
    } catch (err) {
      toast(`${t('settings.tagDictionary.resetFail')}: ${err instanceof Error ? err.message : String(err)}`)
    } finally { setBusy(null) }
  }

  const upload = async (file: File) => {
    setBusy('upload')
    try {
      await api.uploadTagDictionary(file)
      await reloadDict()
      toast(t('settings.tagDictionary.uploadOk', { name: file.name }))
    } catch (err) {
      toast(`${t('settings.tagDictionary.uploadFail')}: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setBusy(null)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  return (
    <SettingsSection id="tag-dictionary" title={t('settings.tagDictionary.title')}>
      <SettingsField label={t('settings.tagDictionary.statusLabel')}>
        {dict.status === 'loading' && (
          <span className="text-xs text-fg-tertiary">{t('settings.tagDictionary.loading')}</span>
        )}
        {dict.status === 'empty' && (
          <span className="text-xs text-warn">{t('settings.tagDictionary.empty')}</span>
        )}
        {dict.status === 'error' && (
          <span className="text-xs text-err">{dict.error ?? t('settings.tagDictionary.error')}</span>
        )}
        {dict.status === 'ready' && meta && (
          <div className="text-xs flex flex-col gap-0.5">
            <div>
              <span className="text-fg-tertiary">{sourceLabel}：</span>
              <code className="font-mono text-fg-primary">{meta.source_name}</code>
            </div>
            <div className="text-fg-tertiary">
              {t('settings.tagDictionary.entryCount', { n: meta.entry_count })} · {downloadedAt}
            </div>
          </div>
        )}
      </SettingsField>

      <SettingsField
        label={t('settings.tagDictionary.uploadLabel')}
        desc={t('settings.tagDictionary.uploadHint')}
      >
        <div className="flex gap-1.5 items-center flex-wrap">
          <input
            ref={fileRef}
            type="file"
            accept=".csv,.txt"
            disabled={busy !== null}
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) void upload(f)
            }}
            className="text-xs"
          />
          <button
            type="button"
            disabled={busy !== null}
            onClick={() => void reset()}
            className="btn btn-secondary btn-sm"
            title={t('settings.tagDictionary.resetHint')}
          >
            {busy === 'reset' ? t('common.downloading') : t('settings.tagDictionary.resetButton')}
          </button>
        </div>
      </SettingsField>

      <SettingsField
        label={t('settings.tagDictionary.showToggleLabel')}
        desc={t('settings.tagDictionary.showToggleHint')}
      >
        <Bool value={show} onChange={setShow} />
      </SettingsField>
    </SettingsSection>
  )
}

// ── ONNX Runtime Section（WD14 + CLTagger 共用 onnxruntime 包管理） ─────────

function ONNXRuntimeSection() {
  const { t } = useTranslation()
  const dialog = useDialog()
  const [rt, setRt] = useState<WD14Runtime | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<null | 'auto' | 'gpu' | 'cpu' | 'directml'>(null)
  const [reinstallOpen, setReinstallOpen] = useState(false)
  const { toast } = useToast()

  const refresh = useCallback(async () => {
    try {
      const r = await api.getWD14Runtime()
      setRt(r)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const install = async (target: 'auto' | 'gpu' | 'cpu' | 'directml') => {
    const confirmKey = target === 'auto'
      ? 'settings.confirmInstallOnnxAuto'
      : target === 'gpu'
        ? 'settings.confirmInstallOnnxGpu'
        : target === 'directml'
          ? 'settings.confirmInstallOnnxDirectml'
          : 'settings.confirmInstallOnnxCpu'
    const ok = await dialog.confirm(
      t(confirmKey),
      { tone: 'warn', okText: t('settings.startInstall') },
    )
    if (!ok) return
    setBusy(target)
    try {
      const result = await api.installWD14Runtime(target)
      setRt({
        installed: result.installed, version: result.version, providers: result.providers,
        cuda_available: result.cuda_available,
        directml_available: result.directml_available,
        platform: result.platform,
        restart_required: result.restart_required,
        cuda_load_error: result.cuda_load_error, preload: result.preload, cuda_detect: result.cuda_detect,
      })
      const newPkg = result.installed_pkg ?? result.installed ?? '?'
      const newVer = result.installed_version ?? result.version ?? '?'
      toast(t('settings.packageInstalledRestart', { pkg: newPkg, version: newVer }), 'success')
    } catch (e) {
      toast(t('settings.packageInstallFailed', { error: String(e) }), 'error')
    } finally {
      setBusy(null)
    }
  }

  const cuda = rt?.cuda_detect ?? { available: false, driver_version: null, gpu_name: null }
  const notInstalled = !!rt && rt.installed === null
  const gpuAccel = !!rt && (rt.cuda_available || rt.directml_available)
  const isWindows = !rt || rt.platform === 'win32'
  // mismatched: 装了某个包 + 有 GPU 但没用上任何加速 EP（CUDA 或 DirectML）。
  // 未装由 notInstalled 接管；已装且已经在用 DirectML 不算 mismatch。
  const mismatched = !!rt && rt.installed !== null && cuda.available && !gpuAccel
  // 默认状态正常时整体折叠；未装 / 有错 / mismatch / 需重启时自动展开
  const hasIssue = !!error || (rt && (
    notInstalled || !!rt.cuda_load_error || rt.restart_required || mismatched
  ))

  // summary 里显示一行简短状态，用户不展开就能扫到
  const epShort = !rt
    ? '?'
    : rt.cuda_available ? 'CUDA' : rt.directml_available ? 'DirectML' : 'CPU'
  const statusLabel = error
    ? `⚠ ${t('settings.statusLoadFailed')}`
    : !rt
      ? t('settings.loadingStatus')
      : notInstalled
        ? `⚠ ${t('settings.notInstalledShort')}`
        : rt.cuda_load_error
          ? `⚠ ${t('settings.cudaLoadFailed')}`
          : rt.restart_required
            ? `⚠ ${t('settings.restartStudioRequired')}`
            : mismatched
              ? `⚠ ${t('settings.gpuRunningCpuEp')}`
              : `${epShort} · ${rt.installed ?? '?'}`
  const statusOk = rt && !hasIssue

  return (
    <details id="onnxruntime" open={!!hasIssue} className="rounded-md border border-subtle bg-surface group scroll-mt-24">
      <summary className="cursor-pointer p-4 list-none flex items-center gap-2">
        <span className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3">▸</span>
        <h2 className="text-sm font-semibold text-fg-primary m-0">ONNX Runtime</h2>
        <span className="text-xs text-fg-tertiary">{t('settings.sharedByWd14ClTagger')}</span>
        <span className={`ml-auto text-xs font-mono ${statusOk ? 'text-ok' : 'text-warn'}`}>{statusLabel}</span>
      </summary>

      <div className="px-4 pb-4 flex flex-col gap-3">
        {error && <div className="text-err text-xs font-mono">{error}</div>}
        {!error && !rt && <div className="text-xs text-fg-tertiary">{t('settings.loadingRuntimeStatus')}</div>}
        {rt && (
          <>
            <div className="rounded-sm border border-subtle bg-sunken p-2 flex flex-col gap-1 text-xs">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-fg-tertiary shrink-0">runtime:</span>
                <code className="font-mono text-fg-primary">{rt.installed ?? t('settings.notInstalledParen')}{rt.version ? `==${rt.version}` : ''}</code>
                <StatusLabel bg={gpuAccel ? 'bg-ok-soft' : 'bg-warn-soft'} fg={gpuAccel ? 'text-ok' : 'text-warn'} text={rt.cuda_available ? 'CUDA' : rt.directml_available ? 'DirectML' : 'CPU only'} />
              </div>
              <div className="text-fg-tertiary">EP: <code className="text-fg-secondary font-mono">{(rt.providers ?? []).map((p) => p.replace('ExecutionProvider', '')).join(' / ') || '(none)'}</code></div>
              <div className="text-fg-tertiary">{t('settings.gpuDetect')}: <span className="text-fg-secondary">{cuda.available ? `${cuda.gpu_name ?? '?'} (driver ${cuda.driver_version ?? '?'})` : t('settings.noNvidiaGpu')}</span></div>
            </div>

            {rt.restart_required && (
              <div className="rounded-sm border border-err bg-err-soft px-2 py-1.5 text-err text-xs">
                <Trans i18nKey="settings.onnxRestartRequired" components={{ strong: <strong /> }} />
              </div>
            )}
            {!rt.restart_required && notInstalled && (
              <div className="rounded-sm border border-info bg-info-soft px-2 py-1.5 text-info text-xs">
                {cuda.available ? t('settings.onnxNotInstalledHintGpu') : t('settings.onnxNotInstalledHintCpu')}
              </div>
            )}
            {!rt.restart_required && mismatched && (
              <div className="rounded-sm border border-info bg-info-soft px-2 py-1.5 text-info text-xs">
                {t('settings.onnxCpuEpWarning')}
              </div>
            )}
            {rt.cuda_load_error && (
              <div className="rounded-sm border border-err bg-err-soft px-2 py-1.5 text-xs text-err">
                <div>{t('settings.cudaEpFailedCpu')}</div>
                <code className="block font-mono text-xs text-err break-all whitespace-pre-wrap mt-1">
                  {rt.cuda_load_error}
                </code>
              </div>
            )}

            <div className="flex gap-1.5 items-center flex-wrap">
              <button onClick={() => install('auto')} disabled={busy !== null} className="btn btn-primary btn-sm">
                {busy === 'auto' ? t('settings.installingPackage') : t('settings.autoDetectInstall')}
              </button>
              <button onClick={() => void refresh()} disabled={busy !== null} title={t('settings.refreshStatus')}
                className="px-2 py-0.5 text-fg-tertiary bg-transparent border-none cursor-pointer rounded-sm">↻</button>
              <button type="button" onClick={() => setReinstallOpen(!reinstallOpen)}
                className="btn btn-ghost btn-sm text-xs text-fg-tertiary ml-auto">
                {reinstallOpen ? '▾' : '▸'} {t('settings.forceReinstallAdvanced')}
              </button>
            </div>
            {reinstallOpen && (
              <div className="flex flex-col gap-2 pt-2 border-t border-subtle">
                <div className="flex gap-1.5 items-center flex-wrap">
                  <button
                    onClick={() => install('directml')}
                    disabled={busy !== null || !isWindows}
                    title={isWindows ? t('settings.directmlPackageHint') : t('settings.directmlWinOnlyHint')}
                    className="btn btn-secondary btn-sm"
                  >
                    {busy === 'directml' ? t('settings.installingPackage') : t('settings.reinstallDirectml')}
                  </button>
                  <button
                    onClick={() => install('gpu')}
                    disabled={busy !== null}
                    title={t('settings.cudaPackageHint')}
                    className="btn btn-secondary btn-sm"
                  >
                    {busy === 'gpu' ? t('settings.installingPackage') : t('settings.reinstallGpu')}
                  </button>
                  <button
                    onClick={() => install('cpu')}
                    disabled={busy !== null}
                    title={t('settings.cpuPackageHint')}
                    className="btn btn-secondary btn-sm"
                  >
                    {busy === 'cpu' ? t('settings.installingPackage') : t('settings.reinstallCpu')}
                  </button>
                </div>
                <span className="text-[10px] text-fg-tertiary">{t('settings.onnxForceHint')}</span>
              </div>
            )}
          </>
        )}
      </div>
    </details>
  )
}

// ── PyTorch Section（训练 tab）──────────────────────────────────────────────
//
// 已有 venv 用户的「一键修」入口。PR-4 启动期会 warn「检测到 GPU 但 torch 是
// CPU 版」并给 pip 命令；这里把命令 UI 化，普通用户不用进终端。
//
// 三种状态：
// - cuda_available=True               → ✓ 一切 OK（折叠默认；提供「换 CUDA 版本」高级选项）
// - is_cpu_with_gpu=True               → 红色误装提示 + 显著「重装为 CUDA」主按钮
// - is_cuda_build_unavailable=True     → 黄色驱动警告（pip 修不了，给文档链接）

function PyTorchSection() {
  const { t } = useTranslation()
  const dialog = useDialog()
  const [status, setStatus] = useState<TorchStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const { toast } = useToast()

  const refresh = useCallback(async () => {
    try {
      const s = await api.getTorchStatus()
      setStatus(s)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const reinstall = async (target: 'auto' | TorchCuTag) => {
    const tag = target === 'auto' ? status?.recommended_cu_tag ?? '?' : target
    // 注册 → 用户 Ctrl+C 重启 → launcher 进程跑 pip。Windows 上 torch.pyd 被
    // 当前 server 进程锁住，没法直接 replace；只能 defer 到 launcher。
    if (!(await dialog.confirm(
      t('settings.confirmRegisterTorch', { tag }),
      { tone: 'warn', okText: t('settings.registerRequest') },
    ))) return
    setBusy(true)
    try {
      const result = await api.reinstallTorch(target)
      // 后端已写 marker，server 进程没真装；提示用户去重启
      toast(result.message, 'success')
    } catch (e) {
      toast(t('settings.registerFailed', { error: String(e) }), 'error')
    } finally {
      setBusy(false)
    }
  }

  const hasIssue = !!error || (status && (status.is_cpu_with_gpu || status.is_cuda_build_unavailable || !status.installed))
  const statusOk = status?.cuda_available && !error
  const statusLabel = error
    ? t('settings.loadFailedShort')
    : !status
      ? t('settings.loadingStatus')
      : !status.installed
        ? t('settings.notInstalledShort')
        : status.is_cpu_with_gpu
          ? t('settings.cpuBuildMisinstalled')
          : !status.cuda_available && status.cuda_build !== 'cpu'
            ? t('settings.cudaUnavailableDriver')
            : status.cuda_available
              ? `CUDA ✓ ${status.cuda_build}`
              : `CPU ${status.cuda_build}`

  return (
    <details id="pytorch" open={!!hasIssue} className="rounded-md border border-subtle bg-surface group scroll-mt-24">
      <summary className="cursor-pointer p-4 list-none flex items-center gap-2">
        <span className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3">▸</span>
        <h2 className="text-sm font-semibold text-fg-primary m-0">PyTorch</h2>
        <span className="text-xs text-fg-tertiary">{t('settings.trainingCoreDependency')}</span>
        <span className={`ml-auto text-xs font-mono ${statusOk ? 'text-ok' : status?.is_cpu_with_gpu ? 'text-err' : 'text-warn'}`}>
          {statusLabel}
        </span>
      </summary>

      <div className="px-4 pb-4 flex flex-col gap-3">
        {error && <div className="text-err text-xs font-mono">{error}</div>}
        {!error && !status && <div className="text-xs text-fg-tertiary">{t('settings.loadingStatus')}</div>}

        {status && (<>
          {/* 当前状态卡 */}
          <div className="rounded-sm border border-subtle bg-sunken p-2 flex flex-col gap-1 text-xs">
            <div className="flex gap-4 flex-wrap">
              <span className="text-fg-tertiary">torch: <code className="text-fg-secondary font-mono">{status.version ?? t('settings.notInstalledParen')}</code></span>
              {status.cuda_build && (
                <span className="text-fg-tertiary">build: <code className="text-fg-secondary font-mono">{status.cuda_build}</code></span>
              )}
              {status.cuda_available && status.device_name && (
                <span className="text-fg-tertiary">GPU: <code className="text-fg-secondary font-mono">{status.device_name}</code></span>
              )}
            </div>
            <div className="flex gap-4 flex-wrap">
              <span className="text-fg-tertiary">
                {t('settings.driverLabel')}:{' '}
                <code className="text-fg-secondary font-mono">
                  {status.cuda_detect.driver_version ?? t('settings.notDetected')}
                </code>
              </span>
              {status.cuda_detect.gpu_name && !status.cuda_available && (
                <span className="text-fg-tertiary">
                  {t('settings.systemGpu')}:{' '}
                  <code className="text-fg-secondary font-mono">{status.cuda_detect.gpu_name}</code>
                </span>
              )}
            </div>
          </div>

          {/* 误装：CPU torch + 有 GPU */}
          {status.is_cpu_with_gpu && (
            <div className="rounded-sm border border-err bg-err-soft px-2 py-1.5 text-err text-xs">
              <Trans
                i18nKey="settings.torchCpuWithGpuWarning"
                values={{ tag: status.recommended_cu_tag }}
                components={{ code: <code className="font-mono" /> }}
              />
            </div>
          )}

          {/* CUDA build 但运行时不可用：驱动 / WSL 问题 */}
          {status.is_cuda_build_unavailable && (
            <div className="rounded-sm border border-warn bg-warn-soft px-2 py-1.5 text-warn text-xs">
              <Trans
                i18nKey="settings.torchCudaUnavailableWarning"
                components={{ code: <code className="font-mono" /> }}
              />
            </div>
          )}

          {/* 操作按钮 */}
          <div className="flex gap-1.5 items-center flex-wrap">
            <button
              onClick={() => void reinstall('auto')}
              disabled={busy || !status.cuda_detect.available}
              className={status.is_cpu_with_gpu ? 'btn btn-primary btn-sm' : 'btn btn-secondary btn-sm'}
              title={status.cuda_detect.available
                ? t('settings.autoSelect', { tag: status.recommended_cu_tag })
                : t('settings.noNvidiaDriverCannotCuda')}
            >
              {busy ? t('settings.installing') : status.is_cpu_with_gpu
                ? t('settings.reinstallCudaBuild', { tag: status.recommended_cu_tag })
                : t('settings.reinstallAuto', { tag: status.recommended_cu_tag })}
            </button>
            <button onClick={() => void refresh()} disabled={busy}
              className="px-2 py-0.5 text-fg-tertiary bg-transparent border-none cursor-pointer rounded-sm">↻</button>
            <button type="button" onClick={() => setAdvancedOpen(!advancedOpen)}
              className="btn btn-ghost btn-sm text-xs text-fg-tertiary ml-auto">
              {advancedOpen ? '▾' : '▸'} {t('settings.advancedManualCuda')}
            </button>
          </div>

          {/* 手动选版本 */}
          {advancedOpen && (
            <div className="flex flex-col gap-1.5 pt-2 border-t border-subtle text-xs">
              <p className="text-fg-tertiary m-0">
                {t('settings.manualCudaHint')}
              </p>
              <div className="flex gap-1.5 flex-wrap">
                {(['cu128', 'cu126', 'cu124', 'cu118', 'cpu'] as const).map((tag) => (
                  <button
                    key={tag}
                    onClick={() => void reinstall(tag)}
                    disabled={busy}
                    className={`btn btn-secondary btn-sm ${
                      status.cuda_build === tag ? 'border-accent' : ''
                    }`}
                    title={
                      tag === 'cpu'
                        ? t('settings.installCpuBuildHint')
                        : t('settings.installCudaBuildHint', { tag })
                    }
                  >
                    {tag}{status.cuda_build === tag ? ' ✓' : ''}
                  </button>
                ))}
              </div>
            </div>
          )}
        </>)}
      </div>
    </details>
  )
}

// ── Flash Attention Section（训练 tab）─────────────────────────────────────
//
// 训练加速的可选优化。装好 flash_attn 后启动期会自动 set_flash_attn_enabled(True)。
// 本组件给 UI 一键装 wheel 的能力，复用 PR-7a 的 service：状态 + GitHub 候选 + 安装。
//
// 设计要点：
// - install 是同步 pip（几分钟），用 confirm() + busy 状态防误触
// - Python ABI 不一致的 wheel（usable=false）灰显，但保留「强制安装」按钮（
//   极少数情况用户可能在 ABI 兼容子集里跑）
// - GitHub API 限流时 candidates=[] + fetch_error，给手动 URL 输入兜底

function FlashAttentionSection() {
  const { t } = useTranslation()
  const dialog = useDialog()
  const [status, setStatus] = useState<FlashAttnStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [candidatesOpen, setCandidatesOpen] = useState(false)
  const [manualUrl, setManualUrl] = useState('')
  const { toast } = useToast()

  const refresh = useCallback(async () => {
    try {
      const s = await api.getFlashAttnStatus()
      setStatus(s)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const install = async (url: string | null) => {
    const msg = url ? t('settings.confirmInstallFlashUrl') : t('settings.confirmInstallFlashAuto')
    if (!(await dialog.confirm(msg, { tone: 'warn', okText: t('settings.startInstall') }))) return
    setBusy(true)
    try {
      const result = await api.installFlashAttn(url)
      toast(t('settings.flashAttnInstalled', { version: result.version ?? '?' }), 'success')
      await refresh()
    } catch (e) {
      toast(t('settings.installFailed', { error: String(e) }), 'error')
    } finally {
      setBusy(false)
    }
  }

  const env = status?.env
  const candidates = status?.candidates ?? []
  const fetchError = status?.fetch_error ?? null
  const usable = candidates.filter((c) => c.usable)
  const bestCandidate = usable[0] ?? null
  const hasIssue = !!error || (status && !status.installed)
  const canAutoInstall = !!env?.torch_tag && !!env?.platform && usable.length > 0

  const statusLabel = error
    ? t('settings.loadFailedShort')
    : !status
      ? t('settings.loadingStatus')
      : status.installed
        ? t('settings.installedVersion', { version: status.version ?? '?' })
        : t('settings.notInstalledShort')
  const statusOk = status?.installed && !error

  return (
    <details id="flash-attn" open={!!hasIssue} className="rounded-md border border-subtle bg-surface group scroll-mt-24">
      <summary className="cursor-pointer p-4 list-none flex items-center gap-2">
        <span className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3">▸</span>
        <h2 className="text-sm font-semibold text-fg-primary m-0">Flash Attention</h2>
        <span className="text-xs text-fg-tertiary">{t('settings.trainingAccelerationOptional')}</span>
        <span className={`ml-auto text-xs font-mono ${statusOk ? 'text-ok' : 'text-warn'}`}>{statusLabel}</span>
      </summary>

      <div className="px-4 pb-4 flex flex-col gap-3">
        {error && <div className="text-err text-xs font-mono">{error}</div>}
        {!error && !status && <div className="text-xs text-fg-tertiary">{t('settings.loadingStatus')}</div>}

        {status && env && (<>
          {/* 环境信息 */}
          <div className="rounded-sm border border-subtle bg-sunken p-2 flex flex-col gap-1 text-xs">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-fg-tertiary shrink-0">flash_attn:</span>
              <code className="font-mono text-fg-primary">
                {status.installed ? `v${status.version ?? '?'}` : t('settings.notInstalledParen')}
              </code>
              {status.installed && <StatusLabel bg="bg-ok-soft" fg="text-ok" text={t('settings.installed')} />}
            </div>
            <div className="flex gap-4 flex-wrap">
              <span className="text-fg-tertiary">Python: <code className="text-fg-secondary font-mono">{env.python_tag}</code></span>
              <span className="text-fg-tertiary">CUDA: <code className="text-fg-secondary font-mono">{env.cuda_tag ?? t('settings.notDetected')}</code></span>
              <span className="text-fg-tertiary">PyTorch: <code className="text-fg-secondary font-mono">{env.torch_tag ?? t('settings.notDetected')}</code></span>
              <span className="text-fg-tertiary">{t('settings.platform')}: <code className="text-fg-secondary font-mono">{env.platform ?? t('settings.unsupported')}</code></span>
            </div>
          </div>

          {/* GitHub API 失败 */}
          {fetchError && (
            <div className="rounded-sm border border-err bg-err-soft px-2 py-1.5 text-err text-xs">
              {t('settings.githubApiFailed')}
              <code className="block mt-0.5 break-all">{fetchError}</code>
            </div>
          )}

          {/* 没匹配 wheel */}
          {!canAutoInstall && !fetchError && env.platform && env.torch_tag && (
            <div className="rounded-sm border border-warn bg-warn-soft px-2 py-1.5 text-warn text-xs">
              {t('settings.noWheelForPython', { python: env.python_tag })}
            </div>
          )}

          {/* 操作按钮 */}
          <div className="flex gap-1.5 items-center flex-wrap">
            <button
              onClick={() => void install(null)}
              disabled={busy || !canAutoInstall}
              className="btn btn-primary btn-sm"
              title={canAutoInstall
                ? t('settings.autoSelect', { tag: bestCandidate?.name ?? '' })
                : t('settings.noWheelManual')}
            >
              {busy ? t('settings.installing') : status.installed ? t('settings.reinstallAutoMatch') : t('settings.autoMatchInstall')}
            </button>
            <button onClick={() => void refresh()} disabled={busy}
              className="px-2 py-0.5 text-fg-tertiary bg-transparent border-none cursor-pointer rounded-sm">↻</button>
            <button type="button" onClick={() => setCandidatesOpen(!candidatesOpen)}
              className="btn btn-ghost btn-sm text-xs text-fg-tertiary ml-auto">
              {candidatesOpen ? '▾' : '▸'} {t('settings.candidateWheels', { n: usable.length })}
            </button>
          </div>

          {/* 候选列表 + 手动 URL */}
          {candidatesOpen && (
            <div className="flex flex-col gap-2 pt-2 border-t border-subtle">
              {candidates.length === 0 ? (
                <p className="text-xs text-fg-tertiary m-0">{t('settings.wheelQueryFailed')}</p>
              ) : (
                <ul className="list-none m-0 p-0 flex flex-col gap-1">
                  {candidates.map((c) => (
                    <li key={c.url} className={`flex items-start gap-2 text-xs px-2 py-1.5 rounded-sm border ${
                      c.usable ? 'border-subtle bg-sunken' : 'border-transparent bg-transparent opacity-50'
                    }`}>
                      <div className="flex flex-col gap-0.5 flex-1 min-w-0">
                        <code className="font-mono text-fg-primary text-[11px] break-all">{c.name}</code>
                        {c.notes.map((n, i) => (
                          <span key={i} className="text-warn text-[10px]">{n}</span>
                        ))}
                      </div>
                      <button
                        onClick={() => void install(c.url)}
                        disabled={busy}
                        className={c.usable ? 'btn btn-primary btn-sm shrink-0' : 'btn btn-secondary btn-sm shrink-0'}
                        title={c.usable ? t('settings.installWheel') : t('settings.wheelAbiIncompatible')}
                      >
                        {c.usable ? t('settings.installAction') : t('settings.forceInstall')}
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              <div className="flex flex-col gap-1 pt-1 border-t border-subtle">
                <p className="text-xs text-fg-tertiary m-0">{t('settings.manualUrl')}</p>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    value={manualUrl}
                    onChange={(e) => setManualUrl(e.target.value)}
                    placeholder="https://github.com/.../flash_attn-...whl"
                    className={`${textInputClass} flex-1`}
                  />
                  <button
                    onClick={() => { if (manualUrl.trim()) void install(manualUrl.trim()) }}
                    disabled={busy || !manualUrl.trim()}
                    className="btn btn-secondary btn-sm shrink-0"
                  >{t('settings.install')}</button>
                </div>
              </div>
            </div>
          )}
        </>)}
      </div>
    </details>
  )
}

// ── xformers Section（训练 tab）─────────────────────────────────────────────
//
// 简化版 attention 加速（替代 flash_attn 的另一选项）。xformers 走 PyPI 直装，
// 不需要 flash_attn 那种 GitHub 候选 wheel 列表。失败时给 stderr 让用户排错。

function XformersSection() {
  const { t } = useTranslation()
  const dialog = useDialog()
  const [status, setStatus] = useState<XformersStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const { toast } = useToast()

  const refresh = useCallback(async () => {
    try {
      const s = await api.getXformersStatus()
      setStatus(s)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const install = async () => {
    if (
      !(await dialog.confirm(
        t('settings.confirmInstallXformers'),
        { tone: 'warn', okText: t('settings.startInstall') },
      ))
    ) return
    setBusy(true)
    try {
      const r = await api.installXformers()
      toast(t('settings.xformersInstalled', { version: r.version ?? '?' }), 'success')
      await refresh()
    } catch (e) {
      toast(t('settings.installFailed', { error: String(e) }), 'error')
    } finally {
      setBusy(false)
    }
  }

  const statusLabel = error
    ? t('settings.loadFailedShort')
    : !status
      ? t('settings.loadingStatus')
      : status.installed
        ? t('settings.installedVersion', { version: status.version ?? '?' })
        : t('settings.notInstalledShort')
  const statusOk = status?.installed && !error
  const hasIssue = !!error

  return (
    <details id="xformers" open={!!hasIssue} className="rounded-md border border-subtle bg-surface group scroll-mt-24">
      <summary className="cursor-pointer p-4 list-none flex items-center gap-2">
        <span className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3">▸</span>
        <h2 className="text-sm font-semibold text-fg-primary m-0">xformers</h2>
        <span className="text-xs text-fg-tertiary">{t('settings.xformersSubtitle')}</span>
        <InfoButton>
          <p><Trans i18nKey="settings.xformersHelp1" components={{ strong: <strong />, code: <code /> }} /></p>
          <p>{t('settings.xformersHelp2')}</p>
          <p>{t('settings.xformersHelp3')}</p>
        </InfoButton>
        <span className={`ml-auto text-xs font-mono ${statusOk ? 'text-ok' : 'text-warn'}`}>{statusLabel}</span>
      </summary>

      <div className="px-4 pb-4 flex flex-col gap-3">
        {error && <div className="text-err text-xs font-mono">{error}</div>}
        {!error && !status && <div className="text-xs text-fg-tertiary">{t('settings.loadingStatus')}</div>}

        {status && (<>
          <div className="rounded-sm border border-subtle bg-sunken p-2 flex items-center gap-2 text-xs">
            <span className="text-fg-tertiary shrink-0">xformers:</span>
            <code className="font-mono text-fg-primary">
              {status.installed ? `v${status.version ?? '?'}` : t('settings.notInstalledParen')}
            </code>
            {status.installed && <StatusLabel bg="bg-ok-soft" fg="text-ok" text={t('settings.installed')} />}
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => void install()}
              disabled={busy}
              className="btn btn-primary btn-sm"
            >
              {busy
                ? t('settings.installing')
                : status.installed
                  ? t('settings.reinstallAutoMatchPlain')
                  : t('settings.installAutoMatchPlain')}
            </button>
            <button
              onClick={() => void refresh()}
              disabled={busy}
              className="btn btn-ghost btn-sm"
              title={t('settings.refreshStatus')}
            >↻</button>
          </div>
        </>)}
      </div>
    </details>
  )
}

// ── 中间步预览（节流） ────────────────────────────────────────────────────
//
// TAEFlux 模型 server 启动时后台下载（lifespan startup）；UI 只暴露用户必须
// 控制的「节流 N」一个输入，其他状态/下载/帮助文字全删（用户决策）。

function IdleTimeoutSection({
  draft, update,
}: {
  draft: Secrets
  update: <S extends Section, K extends keyof Secrets[S]>(
    section: S, key: K, value: Secrets[S][K],
  ) => void
}) {
  const { t } = useTranslation()
  const minutes = draft.generate.idle_timeout_minutes
  return (
    <SettingsSection id="idle-timeout" title={t('settings.idleTimeout.title')}>
      <SettingsField
        label={t('settings.idleTimeout.label')}
        desc={t('settings.idleTimeout.desc')}
        helpTooltip={<p>{t('settings.idleTimeout.help')}</p>}
      >
        <div className="flex items-center gap-2">
          <input
            type="number"
            min={0}
            max={240}
            value={minutes}
            onChange={(e) => update('generate', 'idle_timeout_minutes', Math.max(0, Number(e.target.value) || 0))}
            className="input"
            style={{ width: 80 }}
          />
          <span className="text-xs text-fg-tertiary">
            {minutes === 0
              ? t('settings.idleTimeout.offHint')
              : t('settings.idleTimeout.minutesSuffix')}
          </span>
        </div>
      </SettingsField>
    </SettingsSection>
  )
}


function TaeFluxSection({
  draft, update,
}: {
  draft: Secrets
  update: <S extends Section, K extends keyof Secrets[S]>(
    section: S, key: K, value: Secrets[S][K],
  ) => void
}) {
  const { t } = useTranslation()
  const n = draft.generate.preview_every_n_steps
  return (
    <SettingsSection id="preview" title={t('settings.intermediatePreview')}>
      <SettingsField
        label={t('settings.previewThrottle')}
        desc={t('settings.previewThrottleDesc')}
        helpTooltip={
          <p>{t('settings.taeFluxHelp')}</p>
        }
      >
        <input
          type="number"
          min={0}
          max={50}
          value={n}
          onChange={(e) => update('generate', 'preview_every_n_steps', Number(e.target.value) || 0)}
          className="input"
          style={{ width: 80 }}
        />
      </SettingsField>
    </SettingsSection>
  )
}


function SaveTestImagesSection({
  draft, update,
}: {
  draft: Secrets
  update: <S extends Section, K extends keyof Secrets[S]>(
    section: S, key: K, value: Secrets[S][K],
  ) => void
}) {
  const { t } = useTranslation()
  return (
    <SettingsSection id="save-test-images" title={t('settings.saveTestImages.title')}>
      <SettingsField
        label={t('settings.saveTestImages.label')}
        helpTooltip={t('settings.saveTestImages.tooltip')}
      >
        <Bool
          value={draft.generate.save_test_images}
          onChange={(v) => update('generate', 'save_test_images', v)}
        />
      </SettingsField>
    </SettingsSection>
  )
}


// ── Display Section ────────────────────────────────────────────────────────

function DisplaySection() {
  const { t } = useTranslation()
  const [theme, setTheme] = useState<Theme>(() => getStoredTheme())
  const [density, setDensity] = useState<Density>(() => getStoredDensity())
  const [lang, setLang] = useState<string>(() => getStoredLangWithDefault())

  const handleThemeChange = (t: Theme) => {
    setTheme(t)
    setStoredTheme(t)
    applyTheme(t)
  }

  const handleDensityChange = (d: Density) => {
    setDensity(d)
    setStoredDensity(d)
    applyDensity(d)
  }

  const handleLangChange = (newLang: string) => {
    setLang(newLang)
    setStoredLang(newLang)
    void i18n.changeLanguage(newLang)
  }

  const densityLabel = (d: Density): string => {
    if (d === 'tight') return t('settings.densityTight')
    if (d === 'loose') return t('settings.densityLoose')
    return t('settings.densityDefault')
  }

  return (
    <SettingsSection id="display" title={t('settings.display')}>
      <SettingsField label={t('settings.language')}>
        <div className="flex gap-1">
          {[
            { id: 'zh', label: t('settings.languageZh') },
            { id: 'en', label: t('settings.languageEn') },
          ].map((l) => (
            <button
              key={l.id}
              onClick={() => handleLangChange(l.id)}
              className={`btn btn-sm ${lang === l.id ? 'btn-primary' : 'btn-secondary'}`}
            >
              {l.label}
            </button>
          ))}
        </div>
      </SettingsField>

      <SettingsField label={t('settings.theme')}>
        <div className="flex gap-1">
          {(['light', 'dark'] as Theme[]).map((themeOption) => (
            <button
              key={themeOption}
              onClick={() => handleThemeChange(themeOption)}
              className={`btn btn-sm ${theme === themeOption ? 'btn-primary' : 'btn-secondary'}`}
            >
              {themeOption === 'light' ? t('settings.themeLight') : t('settings.themeDark')}
            </button>
          ))}
        </div>
      </SettingsField>

      <SettingsField
        label={t('settings.uiScale')}
        helpTooltip={
          <>
            <p><strong>{t('settings.densityTight')}</strong>：{t('settings.densityTightHelp')}</p>
            <p><strong>{t('settings.densityDefault')}</strong>：{t('settings.densityDefaultHelp')}</p>
            <p><strong>{t('settings.densityLoose')}</strong>：{t('settings.densityLooseHelp')}</p>
          </>
        }
      >
        <div className="flex gap-1">
          {(['tight', 'default', 'loose'] as Density[]).map((d) => (
            <button
              key={d}
              onClick={() => handleDensityChange(d)}
              className={`btn btn-sm ${density === d ? 'btn-primary' : 'btn-secondary'}`}
            >
              {densityLabel(d)}
            </button>
          ))}
        </div>
      </SettingsField>
    </SettingsSection>
  )
}

// ── System Section（系统 tab）─────────────────────────────────────────────
//
// PR-B 起拆成两个 sub-section：
//   - VersionSection：当前版本 / 检查更新 / 立即更新（master 通道）
//   - ServiceSection：重启 server
//
// 共用流程"触发后端退出 → 轮询 /api/health 等回来 → 刷新页面"由
// `pollHealthThenReload` 抽出。restart 超时 5 分钟，update 超时 10 分钟
// （要多跑 git pull + 可能 pip install / npm install 的时间）。
function SystemSection() {
  return (
    <>
      <OnboardingSection />
      <VersionSection />
      <ServiceSection />
    </>
  )
}

// ── 重新运行首次引导 Section ─────────────────────────────────────────────
//
// 清掉 localStorage 的 onboarding done 标记 + dispatch event,触发
// FirstRunOnboardingModal 显示。不重启服务、不动 secrets。
function OnboardingSection() {
  const { t } = useTranslation()
  const handleReopen = () => {
    clearOnboardingDone()
    window.dispatchEvent(new Event(ONBOARDING_EVENTS.open))
  }
  return (
    <SettingsSection id="onboarding" title={t('settings.onboardingSection')}>
      <SettingsField
        label={t('settings.onboardingReopenTitle')}
        helpTooltip={<p>{t('settings.onboardingReopenHelp')}</p>}
      >
        <button
          type="button"
          onClick={handleReopen}
          className="btn btn-secondary btn-sm self-start"
        >
          {t('settings.onboardingReopen')}
        </button>
      </SettingsField>
    </SettingsSection>
  )
}

// ── 公共：触发后端退出后轮询 health 并刷新 ─────────────────────────────
//
// 调用者负责在 await 之前已经成功触发了 server 退出（POST /restart 或 /update
// 已经 200 回来）。这里只管"等服务回来 + 刷页面 + 失败提示"。
type ToastFn = (msg: string, kind?: 'info' | 'success' | 'error') => void

async function pollHealthThenReload(
  toast: ToastFn,
  timeoutMs: number,
  label: string,
  onTimeout: () => void,
  t: TFunction,
): Promise<void> {
  const deadline = Date.now() + timeoutMs
  const pollInterval = 500
  // 间隔后开始轮询：给 server 时间真正退出，避免命中还没死的旧进程
  await new Promise((r) => setTimeout(r, 1500))
  while (Date.now() < deadline) {
    try {
      await api.health()
      toast(t('settings.operationCompletedReloading', { label }), 'success')
      setTimeout(() => window.location.reload(), 800)
      return
    } catch {
      // server 还没回来，继续轮询
    }
    await new Promise((r) => setTimeout(r, pollInterval))
  }
  const mins = Math.round(timeoutMs / 60_000)
  toast(t('settings.operationTimeout', { label, mins }), 'error')
  onTimeout()
}

// ── 版本 Section（ADR 0005 重设计 — 单视图 + 通道偏好）───────────────
//
// 产品模型：
// - 通道（channel）是**用户视图偏好**：你想订阅哪条更新轨道（稳定 / 开发）
// - 与 git 工作树状态**解耦**：切 toggle 不动 git；真正"切到 dev HEAD" /
//   "更新到 vX.Y.Z" 是单独按钮
// - 同屏只显示当前选中通道的卡片（不并排）—— 通道是互斥视图，并排会让
//   用户陷入"我究竟在哪里"的矛盾
// - 文案语言只有"版本号"+"状态"，绝不出现"commits"/"sha"等 git 词汇
//
// 数据：
// - version.installed_kind (stable / dev / custom) + installed_label：装了什么
// - check.state (up_to_date / update_available / ahead / detached)：相对所选
//   通道的状态
// - prefs.update_channel：用户偏好（"stable" / "dev"）
//
// 自动检查 + Topbar 红点仍然只看 master（ADR 0002 决策）。
function VersionSection() {
  const { t } = useTranslation()
  const { toast } = useToast()
  // chunk 4：dialog 模态被 inline preview 面板取代，VersionSection 不再用 dialog
  const [version, setVersion] = useState<SystemVersion | null>(null)
  const [check, setCheck] = useState<SystemUpdateCheck | null>(null)
  const [status, setStatus] = useState<SystemUpdateStatus | null>(null)
  const [prefs, setPrefs] = useState<SystemPrefsConfig | null>(null)
  const [devCheck, setDevCheck] = useState<SystemUpdateCheck | null>(null)
  // chunk 2 — 当前显示的 release notes（hasUpdate 时为 target tag，否则 current tag）
  const [releaseNotes, setReleaseNotes] = useState<ReleaseNotes | null>(null)
  // chunk 3 — dev 通道最近 commit 列表 + 选中状态（用户点 commit 准备切换）
  const [devCommits, setDevCommits] = useState<DevCommitsResult | null>(null)
  const [selectedSha, setSelectedSha] = useState<string | null>(null)
  // chunk 4 — 状态机 + preview / progress 数据。CardState / PendingTarget 类型
  // 在模块底部声明（同时给 MasterCardProps / DevCardProps 用，避免重复定义）。
  const [masterState, setMasterState] = useState<CardState>('idle')
  const [devState, setDevState] = useState<CardState>('idle')
  const [pendingTarget, setPendingTarget] = useState<PendingTarget | null>(null)
  const [preflight, setPreflight] = useState<PreflightResult | null>(null)
  const [preflightLoading, setPreflightLoading] = useState(false)
  const [checking, setChecking] = useState(false)
  const [checkingDev, setCheckingDev] = useState(false)
  const [busy, setBusy] = useState(false)
  const [logModal, setLogModal] = useState<{ open: boolean; content: string; loading: boolean }>(
    { open: false, content: '', loading: false },
  )
  // chunk 2 重做：release notes 详细内容 modal（含 detail markdown）
  const [detailModalOpen, setDetailModalOpen] = useState(false)
  // 0.8.1 hotfix — zip 安装用户首次 init git 仓库
  const [initing, setIniting] = useState(false)
  const [initError, setInitError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    void (async () => {
      const v = await api.getSystemVersion().catch(() => null)
      if (cancelled) return
      if (v) setVersion(v)
      // zip 模式下 check_update 会失败（git fetch 在没 .git/ 时报错），
      // 没必要发请求。等用户 init 完后再触发。
      if (v?.is_git_repo !== false) {
        void api.checkSystemUpdate('master').then((r) => { if (!cancelled) setCheck(r) }).catch(() => { /* silent */ })
      }
    })()
    void api.getSystemUpdateStatus().then(setStatus).catch(() => { /* silent */ })
    void api.getSecrets().then((s) => setPrefs(s.system)).catch(() => { /* silent */ })
    return () => { cancelled = true }
  }, [])

  // 0.8.1 hotfix — 触发 zip → git 自动 normalize。成功后刷一遍 version + check
  // 让 banner 消失、版本面板正常显示。bootstrap 不重启 server（只动 .git/），
  // 不需要 pollHealthThenReload。
  const handleInitGit = async () => {
    setIniting(true)
    setInitError(null)
    try {
      await api.initGitRepo()
      toast(t('settings.gitInitEnabled'), 'success')
      const v = await api.getSystemVersion().catch(() => null)
      if (v) setVersion(v)
      // init 后立刻拉一次 check，banner 消失 + 同屏显示「已是最新 / 有新版」
      void api.checkSystemUpdate('master', true).then(setCheck).catch(() => { /* silent */ })
    } catch (e) {
      const err = e as Error & { detail?: { message?: string } }
      const msg = err.detail?.message ?? err.message ?? String(e)
      setInitError(msg)
      toast(t('settings.gitInitFailed', { error: msg }), 'error')
    } finally {
      setIniting(false)
    }
  }

  // 选中 dev 通道时自动拉 dev_commits + dev check（用户不用先手动按 [抓取 dev]）。
  // 即便装的是 stable，用户切到 dev 通道偏好时也要能立刻看到 dev HEAD 信息。
  const channelPref: 'stable' | 'dev' = prefs?.update_channel ?? 'stable'
  const showDevView = channelPref === 'dev'
  // 两个 fetch 拆独立 effect：避免「commits 先 resolve 触发 re-render，
  // effect 用 devCommits !== null 早 return 跳过 check fetch」race
  // —— 实测会导致 devCheck 一直 null、"切到 dev HEAD" 按钮不知道
  // 该 disabled，UI 显示 enabled 但点了 no-op。
  useEffect(() => {
    if (!showDevView || devCommits !== null) return
    let cancelled = false
    void api.getDevCommits(10).then((r) => { if (!cancelled) setDevCommits(r) }).catch(() => { /* silent */ })
    return () => { cancelled = true }
  }, [showDevView, devCommits])
  useEffect(() => {
    if (!showDevView || devCheck !== null) return
    let cancelled = false
    void api.checkSystemUpdate('dev', true).then((r) => { if (!cancelled) setDevCheck(r) }).catch(() => { /* silent */ })
    return () => { cancelled = true }
  }, [showDevView, devCheck])

  const handleCheck = async () => {
    setChecking(true)
    try {
      const r = await api.checkSystemUpdate('master', true)
      setCheck(r)
      if (r.error) {
        toast(t('settings.checkFailed', { error: r.error }), 'error')
      } else if (r.state === 'update_available') {
        const target = r.latest_version ?? r.latest_tag ?? r.latest_commit.slice(0, 8)
        toast(t('settings.stableUpdateAvailable', { version: target }), 'info')
      } else if (r.state === 'ahead') {
        toast(t('settings.stableAhead'), 'info')
      } else {
        toast(t('settings.upToDateStable', { version: r.latest_version ? ` ${r.latest_version}` : '' }), 'success')
      }
    } catch (e) {
      toast(t('settings.checkUpdateFailed', { error: String(e) }), 'error')
    } finally {
      setChecking(false)
    }
  }

  // 公用的 422 / 其它错误分流（update 和 rollback 都用）
  const _formatActionError = (e: unknown, action: string): string => {
    const err = e as Error & { status?: number; detail?: { error?: string; tasks?: { name: string; id?: number }[] } }
    if (err.status === 422 && err.detail?.error === 'running_tasks_present') {
      const names = (err.detail.tasks ?? []).map((task) => task.name || `task#${task.id ?? '?'}`).join(', ')
      return t('settings.taskRunningCancelFirst', { names })
    }
    if (err.status === 422 && err.detail?.error === 'dirty_working_tree') {
      return t('settings.dirtyWorkingTree')
    }
    if (err.status === 409 && err.detail?.error === 'no_rollback_target') {
      return t('settings.noRollbackTarget')
    }
    return t('settings.triggerActionFailed', { action, error: err.message ?? String(e) })
  }

  // chunk 4 — inline preview/progress/failed 状态机：所有 "更新 / 切换 / 回滚"
  // 动作不再走 dialog 模态，而是把卡 body 替换成 preview 面板（含 release
  // notes / commit info + pre-flight 检查 + 取消/确认 按钮）。确认后切到
  // progress 状态显示 spinner，pollHealthThenReload 触发页面刷新。
  const enterPreview = (target: PendingTarget) => {
    setPendingTarget(target)
    setSelectedSha(null)
    setPreflight(null)
    setPreflightLoading(true)
    if (target.kind === 'master') setMasterState('preview')
    else setDevState('preview')
    void api.getPreflight(target.ref)
      .then((r) => setPreflight(r))
      .catch(() => setPreflight(null))
      .finally(() => setPreflightLoading(false))
  }

  const cancelPreview = () => {
    setMasterState('idle')
    setDevState('idle')
    setPendingTarget(null)
    setPreflight(null)
    setPreflightLoading(false)
  }

  const confirmPreview = async () => {
    if (!pendingTarget) return
    const t = pendingTarget
    if (t.kind === 'master') setMasterState('progress')
    else setDevState('progress')
    setBusy(true)
    try {
      await api.performSystemUpdate(t.ref)
    } catch (e) {
      toast(_formatActionError(e, t.kind === 'master' ? i18n.t('settings.actionUpdate') : i18n.t('settings.actionSwitch')), 'error')
      setBusy(false)
      if (t.kind === 'master') setMasterState('idle')
      else setDevState('idle')
      return
    }
    void pollHealthThenReload(
      toast,
      10 * 60_000,
      t.kind === 'master' ? i18n.t('settings.actionUpdate') : i18n.t('settings.actionSwitch'),
      () => {
        setBusy(false)
        if (t.kind === 'master') setMasterState('idle')
        else setDevState('idle')
      },
      i18n.t.bind(i18n),
    )
  }

  // 各 action 入口：构造 PendingTarget 后委托给 enterPreview。
  const handleUpdate = () => {
    if (!check?.has_update) return
    const label = check.latest_tag ?? check.latest_commit.slice(0, 8)
    enterPreview({ kind: 'master', ref: 'origin/master', label })
  }

  const handleSwitchToMaster = () => {
    const label = check?.latest_tag ?? check?.latest_commit?.slice(0, 8) ?? 'master'
    enterPreview({ kind: 'master', ref: 'origin/master', label })
  }

  const handleRollback = () => {
    if (!status?.rollback_target) return
    enterPreview({
      kind: 'master',
      ref: status.rollback_target,
      label: status.rollback_target.slice(0, 8),
    })
  }

  const handleViewLog = async () => {
    setLogModal({ open: true, content: '', loading: true })
    try {
      const r = await api.getSystemUpdateLog()
      setLogModal({ open: true, content: r.content || t('settings.emptyLog'), loading: false })
    } catch (e) {
      setLogModal({ open: true, content: t('settings.loadFailedWithError', { error: String(e) }), loading: false })
    }
  }

  // ADR 0005 — 通道偏好持久化到 secrets.json。**不触发任何 git 操作**，
  // 只是切换 UI 视图。乐观更新 + 失败回滚。
  const handleSwitchChannel = async (next: 'stable' | 'dev') => {
    const prev = prefs
    setPrefs((p) => p
      ? { ...p, update_channel: next }
      : { update_channel: next, show_dev_channel: next === 'dev' })
    if (next === 'stable') setDevCheck(null)  // 切回稳定版时清掉 dev 缓存
    try {
      // 同步写 show_dev_channel 字段，保留对老版本回滚兼容
      const updated = await api.updateSecrets({
        system: { update_channel: next, show_dev_channel: next === 'dev' },
      })
      setPrefs(updated.system)
    } catch (e) {
      setPrefs(prev)
      toast(t('settings.saveChannelFailed', { error: (e as Error).message ?? String(e) }), 'error')
    }
  }

  const handleCheckDev = async () => {
    setCheckingDev(true)
    try {
      // chunk 3：同时拉 update_check（HEAD 比对）和 dev_commits（commit 时间线）
      const [check, commits] = await Promise.all([
        api.checkSystemUpdate('dev', true),
        api.getDevCommits(10),
      ])
      setDevCheck(check)
      setDevCommits(commits)
      if (check.error) {
        toast(t('settings.devCheckFailed', { error: check.error }), 'error')
      } else if (commits.error && !commits.fetched) {
        toast(t('settings.devFetchPartialFailed', { error: commits.error }), 'error')
      } else if (check.state === 'update_available') {
        toast(t('settings.devHasNewCommits', { count: check.behind_count }), 'info')
      } else if (check.state === 'ahead') {
        toast(t('settings.devAhead'), 'info')
      } else {
        toast(t('settings.devUpToDate'), 'success')
      }
    } catch (e) {
      toast(t('settings.devCheckFailed', { error: String(e) }), 'error')
    } finally {
      setCheckingDev(false)
    }
  }

  // chunk 3 + chunk 4：选中 commit 后进 preview 面板（dev 卡）。
  const handleSwitchToCommit = (commit: DevCommit) => {
    enterPreview({
      kind: 'dev',
      ref: commit.sha,
      label: commit.short_sha,
      msg: commit.msg,
      author: commit.author,
    })
  }

  // "切到 dev (HEAD)" 当 master 用户初次切到 dev：进 preview 面板。
  const handleUpdateDev = () => {
    const headCommit = devCommits?.commits?.[0]
    if (!headCommit) {
      // 还没抓取过 dev，先 fetch 再 retry（避免空 ref）
      void handleCheckDev()
      return
    }
    enterPreview({
      kind: 'dev',
      ref: 'origin/dev',
      label: headCommit.short_sha,
      msg: headCommit.msg,
      author: headCommit.author,
    })
  }

  // 派生状态（ADR 0005）：installed_kind / state 取代 branch / has_update
  const installedIsDevHead = version?.installed_kind === 'dev'
  const masterHasUpdate = check?.state === 'update_available'
  const hasRollback = !!status?.rollback_target
  // 上次 update 失败 banner（aborted / failed / partial 时显示红色提示）
  const statusBadFailed = !!status && (status.status === 'failed' || status.status === 'aborted' || status.status === 'partial')

  // release notes 拉对应 tag：stable 通道有更新时展示目标版本，否则展示当前
  // 已装版本；dev 通道不展示 release notes（dev 是滚动的，没有版本号语义）
  const displayedTag = showDevView
    ? null
    : masterHasUpdate
      ? (check?.latest_version ?? check?.latest_tag ?? null)
      : (version?.stable_version ?? version?.tag ?? (version ? `v${version.version}` : null))
  useEffect(() => {
    if (!displayedTag) {
      setReleaseNotes(null)
      return
    }
    let cancelled = false
    void api.getReleaseNotes(displayedTag).then((r) => {
      if (!cancelled) setReleaseNotes(r)
    }).catch(() => {
      if (!cancelled) setReleaseNotes(null)
    })
    return () => { cancelled = true }
  }, [displayedTag])

  return (
    <SettingsSection
      id="version"
      title={t('settings.version')}
      headerExtras={
        <InfoButton>
          <ul>
            <li>{t('settings.versionInfoChannel')}</li>
            <li>{t('settings.versionInfoAutoCheck')}</li>
            <li>{t('settings.versionInfoUpdateImpl')}</li>
            <li>{t('settings.versionInfoPreflight')}</li>
          </ul>
        </InfoButton>
      }
    >
      {/* 0.8.1 hotfix — zip 安装用户首次启用自更新功能的 banner。
          version.is_git_repo=false 时显示；git 不可用 vs 可用分两种文案。
          init 成功后 setVersion 刷新，banner 自动消失。 */}
      {version && !version.is_git_repo && (
        <div className="vs-zip-banner">
          {!version.git_available ? (
            <>
              <div className="vs-zip-banner-title">{t('settings.gitNotDetected')}</div>
              <div className="vs-zip-banner-body">
                <Trans
                  i18nKey="settings.gitRequiredHelp"
                  components={{ a: <a href="https://git-scm.com/downloads" target="_blank" rel="noreferrer" /> }}
                />
              </div>
            </>
          ) : (
            <>
              <div className="vs-zip-banner-title">{t('settings.enableAutoUpdate')}</div>
              <div className="vs-zip-banner-body">
                <Trans
                  i18nKey="settings.zipInstallGitInitHelp"
                  values={{ version: `v${version.stable_version?.replace(/^v/, '') ?? version.version}` }}
                  components={{ b: <b /> }}
                />
              </div>
              <div className="vs-zip-banner-actions">
                <button
                  type="button"
                  className="btn btn-sm btn-primary"
                  onClick={() => void handleInitGit()}
                  disabled={initing}
                >
                  {initing ? t('settings.initializingGit') : t('settings.enableAutoUpdate')}
                </button>
                {initError && <span className="vs-zip-banner-error">{t('settings.failedWithError', { error: initError })}</span>}
              </div>
            </>
          )}
        </div>
      )}

      {/* 顶部：你装的是什么（一行事实状态，与通道偏好解耦） */}
      <div className="vs-installed-row">
        <span className="vs-installed-label">{t('settings.installedVersionLabel')}</span>
        <b className="vs-installed-value">{version?.installed_label ?? t('settings.loadingEllipsis')}</b>
        {version?.is_dirty && !version.installed_label.includes(t('settings.uncommittedChangesText')) && (
          <span className="vs-installed-warn">· {t('settings.localChanges')}</span>
        )}
      </div>

      {/* 通道偏好：radio toggle（不触发 git） */}
      <div className="vs-channel-toggle-row">
        <span className="vs-channel-toggle-label">{t('settings.updateChannel')}</span>
        <button
          type="button"
          role="radio"
          aria-checked={channelPref === 'stable'}
          className={`vs-channel-radio${channelPref === 'stable' ? ' on' : ''}`}
          onClick={() => { if (channelPref !== 'stable') void handleSwitchChannel('stable') }}
        >
          <span className="vs-channel-dot" />{t('settings.stable')}
        </button>
        <button
          type="button"
          role="radio"
          aria-checked={channelPref === 'dev'}
          className={`vs-channel-radio${channelPref === 'dev' ? ' on' : ''}`}
          onClick={() => { if (channelPref !== 'dev') void handleSwitchChannel('dev') }}
        >
          <span className="vs-channel-dot" />{t('settings.devBuild')}
        </button>
        <span className="vs-channel-hint">{t('settings.channelUiOnly')}</span>
      </div>

      <div className="vs-sec-card">
        <div className="vs-channels">
          {!showDevView ? (
            <MasterCard
              on={true}
              solo={true}
              version={version}
              check={check}
              status={status}
              hasUpdate={masterHasUpdate}
              hasRollback={hasRollback}
              statusBadFailed={statusBadFailed}
              releaseNotes={releaseNotes}
              onShowReleaseNotesDetail={() => setDetailModalOpen(true)}
              checking={checking}
              busy={busy}
              cardState={masterState}
              pendingTarget={pendingTarget}
              preflight={preflight}
              preflightLoading={preflightLoading}
              onCancelPreview={cancelPreview}
              onConfirmPreview={confirmPreview}
              onCheck={handleCheck}
              onUpdate={handleUpdate}
              onSwitchToMaster={handleSwitchToMaster}
              onRollback={handleRollback}
              onViewLog={handleViewLog}
            />
          ) : (
            <DevCard
              on={installedIsDevHead}
              check={devCheck}
              commits={devCommits}
              currentSha={version?.commit ?? ''}
              installedKind={version?.installed_kind}
              selectedSha={selectedSha}
              setSelectedSha={setSelectedSha}
              checking={checkingDev}
              busy={busy}
              cardState={devState}
              pendingTarget={pendingTarget}
              preflight={preflight}
              preflightLoading={preflightLoading}
              onCancelPreview={cancelPreview}
              onConfirmPreview={confirmPreview}
              onCheck={handleCheckDev}
              onSwitchToDev={handleUpdateDev}
              onSwitchToCommit={handleSwitchToCommit}
            />
          )}
        </div>
      </div>

      {logModal.open && (
        <UpdateLogModal
          loading={logModal.loading}
          content={logModal.content}
          onClose={() => setLogModal({ open: false, content: '', loading: false })}
        />
      )}

      {detailModalOpen && releaseNotes?.found && (
        <ReleaseNotesDetailModal
          notes={releaseNotes}
          onClose={() => setDetailModalOpen(false)}
        />
      )}
    </SettingsSection>
  )
}

// ── 子组件：图标 / Master 卡 / Dev 卡 ─────────────────────────────────
//
// 双卡布局拆成独立函数组件方便 chunk 2/3/4 各自扩展：
//   - chunk 2 把 release notes 填进 MasterCard.change-block
//   - chunk 3 给 DevCard 加 commits 列表 + 选中状态
//   - chunk 4 给两卡都加 preview / progress 状态机

const VERSION_ICON_PATHS: Record<string, React.ReactNode> = {
  refresh:  <><path d="M14 8a6 6 0 1 1-1.76-4.24" /><path d="M14 3v3.4h-3.4" /></>,
  log:      <><rect x="3" y="2.5" width="10" height="11" rx="1.5" /><path d="M5.5 5.5h5M5.5 8h5M5.5 10.5h3" /></>,
  rollback: <><path d="M3 8h7a3 3 0 1 1 0 6h-1" /><path d="M5.5 5.5L3 8l2.5 2.5" /></>,
  note:     <><path d="M4 3.5h6l2 2v7a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-8a1 1 0 0 1 1-1z" /><path d="M5.5 7h5M5.5 9.5h5M5.5 12h3" /></>,
  lock:     <><rect x="3.5" y="7" width="9" height="6.5" rx="1" /><path d="M5.5 7v-2a2.5 2.5 0 0 1 5 0v2" /></>,
}

function VersionIcon({ name }: { name: keyof typeof VERSION_ICON_PATHS | string }) {
  const path = VERSION_ICON_PATHS[name]
  if (!path) return null
  return (
    <svg width={12} height={12} viewBox="0 0 16 16" fill="none" stroke="currentColor"
      strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round">
      {path}
    </svg>
  )
}

type CardState = 'idle' | 'preview' | 'progress'
type PendingTarget = {
  kind: 'master' | 'dev'
  ref: string
  label: string
  msg?: string
  author?: string
}

type MasterCardProps = {
  on: boolean
  solo: boolean
  version: SystemVersion | null
  check: SystemUpdateCheck | null
  status: SystemUpdateStatus | null
  hasUpdate: boolean
  hasRollback: boolean
  statusBadFailed: boolean
  releaseNotes: ReleaseNotes | null
  onShowReleaseNotesDetail: () => void
  checking: boolean
  busy: boolean
  cardState: CardState
  pendingTarget: PendingTarget | null
  preflight: PreflightResult | null
  preflightLoading: boolean
  onCancelPreview: () => void
  onConfirmPreview: () => void
  onCheck: () => void
  onUpdate: () => void
  onSwitchToMaster: () => void
  onRollback: () => void
  onViewLog: () => void
}

// chunk 2 重做 — release_notes.yaml 派生 entries 渲染。每条 [kind 徽章] +
// summary，kind 颜色复用 vs-pill 体系。detail 通过 title hover 透出。
// 超 RN_MAX_ITEMS 折成 "+ 还有 N 项 · 详见 CHANGELOG.md"。
const RN_MAX_ITEMS = 5

// kind → vs-pill class（复用 master/dev/here/info 四套色）+ 中文 label
const KIND_PILL_CLASS: Record<string, string> = {
  added:      'vs-pill-stable',   // 绿（新东西，正面）
  improved:   'vs-pill-info',     // 蓝（优化）
  changed:    'vs-pill-info',     // 蓝（中性）
  fixed:      'vs-pill-dev',      // 橙（修 bug，提醒）
  removed:    'vs-pill-here',     // accent（强调，需注意）
  deprecated: 'vs-pill-here',     // accent
  security:   'vs-pill-here',     // accent
}

const KIND_LABEL_KEY: Record<string, string> = {
  added: 'kindAdded', changed: 'kindChanged', improved: 'kindImproved', fixed: 'kindFixed',
  removed: 'kindRemoved', deprecated: 'kindDeprecated', security: 'kindSecurity',
}

// chunk 4 — preview / progress 通用面板。channel 决定主按钮配色（master=primary
// orange / dev=warn yellow）。details 部分由 caller 决定渲染什么（master 用
// release notes，dev 用 commit msg / author）。
type PreviewPaneProps = {
  channel: 'master' | 'dev'
  fromLabel: string
  toLabel: string
  badge?: string
  details: React.ReactNode
  preflight: PreflightResult | null
  loading: boolean
  busy: boolean
  onCancel: () => void
  onConfirm: () => void
}

function PreviewPane(p: PreviewPaneProps) {
  const { t } = useTranslation()
  const confirmDisabled = !p.preflight || p.preflight.blocking || p.busy
  return (
    <div className="vs-preview-pane">
      <div className="vs-preview-head">
        <span className="vs-from">{p.fromLabel}</span>
        <span className="vs-arr">→</span>
        <span className="vs-to">{p.toLabel}</span>
        {p.badge && <span className="vs-badge">{p.badge}</span>}
      </div>

      {p.details}

      <div className="vs-preflight">
        <div className="vs-h">{t('settings.preflightCheck')}</div>
        {p.loading ? (
          <div className="vs-row">
            <span className="vs-glyph">·</span>
            <span>{t('settings.checkingEllipsis')}</span>
          </div>
        ) : p.preflight ? (
          p.preflight.checks.map((c, i) => (
            <div key={i} className={`vs-row ${c.level}`}>
              <span className="vs-glyph">
                {c.level === 'ok' ? '✓' : c.level === 'warn' ? '!' : '✗'}
              </span>
              <span>{c.label}</span>
            </div>
          ))
        ) : (
          <div className="vs-row err">
            <span className="vs-glyph">✗</span>
            <span>{t('settings.preflightFailedRetry')}</span>
          </div>
        )}
      </div>

      <div className="vs-chan-foot" style={{ borderTop: 0, paddingTop: 0 }}>
        <div className="vs-info">
          {t('settings.preflightInfo')}
        </div>
        <div className="vs-actions">
          <button onClick={p.onCancel} disabled={p.busy} className="btn btn-sm">
            {t('settings.cancel')}
          </button>
          <button
            onClick={p.onConfirm}
            disabled={confirmDisabled}
            className={`btn btn-sm ${p.channel === 'master' ? 'btn-primary' : 'btn-warn'}`}
          >
            {p.busy
              ? t('settings.processing')
              : t('settings.confirmActionTo', {
                action: p.channel === 'master' ? t('settings.actionUpdate') : t('settings.actionSwitch'),
                label: p.toLabel,
              })}
          </button>
        </div>
      </div>
    </div>
  )
}

function ProgressPane({ fromLabel, toLabel }: { fromLabel: string; toLabel: string }) {
  const { t } = useTranslation()
  return (
    <div className="vs-progress-pane">
      <div className="vs-preview-head">
        <span className="vs-from">{fromLabel}</span>
        <span className="vs-arr">→</span>
        <span className="vs-to">{toLabel}</span>
      </div>
      <div className="vs-progress-bar">
        <div className="vs-progress-fill" style={{ width: '100%' }} />
      </div>
      <div className="vs-progress-step">
        <span>{t('settings.progressStarted')}</span>
        <span>{t('settings.progressWaitReload')}</span>
      </div>
      <p style={{ color: 'var(--fg-tertiary)', fontSize: 11, lineHeight: 1.5, margin: 0 }}>
        {t('settings.progressDetail')}
      </p>
    </div>
  )
}

function MasterReleaseNotes({
  notes, onShowDetail,
}: { notes: ReleaseNotes | null; onShowDetail: () => void }) {
  const { t } = useTranslation()
  const entries = notes?.found ? notes.entries : []
  const total = entries.length
  if (total === 0) {
    return (
      <ul className="vs-change-list">
        <li>
          <span className="vs-glyph">▸</span>
          <span className="vs-txt">
            {notes && !notes.found
              ? <Trans i18nKey="settings.releaseNoEntry" components={{ code: <code /> }} />
              : <Trans i18nKey="settings.releaseSeeChangelog" components={{ code: <code /> }} />}
          </span>
        </li>
      </ul>
    )
  }
  const shown = entries.slice(0, RN_MAX_ITEMS)
  const overflow = total - shown.length
  // 任意 entry 有 detail → 即使全部顶层 entries 都显示，"详细内容" 入口仍有意义
  const anyDetail = entries.some((e) => !!e.detail)
  const showDetailLink = overflow > 0 || anyDetail
  return (
    <ul className="vs-change-list">
      {shown.map((e, i) => (
        <li key={i}>
          <span
            className={`vs-pill ${KIND_PILL_CLASS[e.kind] || 'vs-pill-info'}`}
            style={{ flexShrink: 0 }}
            title={e.detail ?? ''}
          >
            {t(`settings.${KIND_LABEL_KEY[e.kind] ?? ''}`, { defaultValue: e.kind })}
          </span>
          <span className="vs-txt">{e.summary}</span>
        </li>
      ))}
      {showDetailLink && (
        <li>
          <span className="vs-glyph">·</span>
          <span className="vs-txt" style={{ color: 'var(--fg-tertiary)' }}>
            {overflow > 0 && t('settings.moreItems', { count: overflow })}
            <button
              type="button"
              onClick={onShowDetail}
              className="vs-lnk"
              style={{ display: 'inline' }}
            >
              {t('settings.detailContent')}
            </button>
          </span>
        </li>
      )}
    </ul>
  )
}

function MasterCard(p: MasterCardProps) {
  const { t } = useTranslation()
  // 装的是 stable 时显示当前稳定版号，否则 ver-tag 区不显示 from（"你装的"
  // 顶部行已经表达了装了什么，避免 "v0.8.0 → v0.8.0" 这种因 __version__
  // 字符串与目标 tag 字面相同导致的伪箭头）
  const installedIsStable = p.version?.installed_kind === 'stable'
  const currentTag = installedIsStable
    ? (p.version?.stable_version ?? p.version?.tag ?? `v${p.version?.version ?? ''}`)
    : null
  // 远端最新稳定版（state=update_available 时显示）
  const targetTag = p.check?.latest_version ?? p.check?.latest_tag ?? ''
  const stateText = formatMasterStateText(p.check, t)
  const showUpdateButton = shouldShowMasterUpdateButton(p.check, p.version?.installed_kind)
  const showSwitchToStableButton = shouldShowSwitchToStableButton(p.check, p.version?.installed_kind)
  if (p.cardState === 'preview' && p.pendingTarget && p.pendingTarget.kind === 'master') {
    return (
      <div className="vs-chan">
        <div className="vs-chan-head">
          <div className="vs-lhs">
            <span className="vs-name">{t('settings.stableConfirmUpdate')}</span>
            <span className="vs-pill vs-pill-stable"><span className="vs-dot" />{t('settings.stable')}</span>
          </div>
          <button className="btn btn-sm btn-ghost" onClick={p.onCancelPreview} disabled={p.busy}>
            {t('settings.back')}
          </button>
        </div>
        <PreviewPane
          channel="master"
          fromLabel={currentTag ?? p.version?.installed_label ?? t('settings.currentShort')}
          toLabel={p.pendingTarget.label}
          details={
            <div className="vs-change-block">
              <div className="vs-h">{t('settings.targetUpdateContent', { label: p.pendingTarget.label })}</div>
              <MasterReleaseNotes notes={p.releaseNotes} onShowDetail={p.onShowReleaseNotesDetail} />
            </div>
          }
          preflight={p.preflight}
          loading={p.preflightLoading}
          busy={p.busy}
          onCancel={p.onCancelPreview}
          onConfirm={p.onConfirmPreview}
        />
      </div>
    )
  }
  if (p.cardState === 'progress' && p.pendingTarget && p.pendingTarget.kind === 'master') {
    return (
      <div className="vs-chan">
        <div className="vs-chan-head">
          <div className="vs-lhs">
            <span className="vs-name">{t('settings.stableUpdating')}</span>
            <span className="vs-pill vs-pill-stable"><span className="vs-dot" />{t('settings.stable')}</span>
          </div>
        </div>
        <ProgressPane fromLabel={currentTag ?? p.version?.installed_label ?? t('settings.currentShort')} toLabel={p.pendingTarget.label} />
      </div>
    )
  }
  const checkedAt = p.check?.checked_at
    ? new Date(p.check.checked_at * 1000).toLocaleString()
    : t('settings.notChecked')
  const releasedAt = p.version?.commit_time_iso
    ? new Date(p.version.commit_time_iso).toLocaleDateString()
    : null

  return (
    <div className="vs-chan">
      <div className="vs-chan-head">
        <div className="vs-lhs">
          <span className="vs-name">{t('settings.stable')}</span>
          <span className="vs-pill vs-pill-stable"><span className="vs-dot" />{t('settings.stable')}</span>
        </div>
        <div className={`vs-meta${p.hasUpdate ? ' attn' : ''}`}>{stateText}</div>
      </div>

      {p.statusBadFailed && p.status && (
        <div className="vs-fail-banner">
          <div className="vs-h">
            <span>
              {t('settings.lastUpdate')}
              {p.status.status === 'aborted' ? t('settings.statusAborted')
                : p.status.status === 'partial' ? t('settings.statusPartial')
                : t('settings.statusFailed')}
            </span>
            {!!p.status.finished_at && (
              <span className="vs-when">
                {new Date(p.status.finished_at * 1000).toLocaleString()}
              </span>
            )}
          </div>
          <div className="vs-d">
            {p.status.reason || t('settings.unknownReason')}
            {p.status.target && <> · target = <code>{p.status.target}</code></>}
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {p.hasUpdate && (
              <button className="btn btn-primary btn-sm" onClick={p.onUpdate} disabled={p.busy}>
                {t('settings.retryUpdateTo', { tag: targetTag })}
              </button>
            )}
            <button className="btn btn-sm" onClick={p.onViewLog} disabled={p.busy}>
              <VersionIcon name="log" />{t('settings.viewFullLog')}
            </button>
          </div>
        </div>
      )}

      <div className={`vs-chan-body ${p.solo ? 'solo' : 'split'}`}>
        <div className="vs-ver-block" style={{ flex: p.solo ? '0 0 220px' : 1 }}>
          <div className="vs-ver-tag">
            {/* 装的是 stable 且有新稳定版：显示 from → to 箭头
                装的是 stable 但已是最新：只显示当前版本号
                装的不是 stable（dev / custom）：只显示目标稳定版号（如有），
                  不再显示 "v0.8.0 → v0.8.0" 伪箭头 */}
            {installedIsStable && p.hasUpdate && targetTag && currentTag !== targetTag ? (
              <>
                <span className="vs-dim">{currentTag}</span>
                <span className="vs-arrow">→</span>
                <span className="vs-target">{targetTag}</span>
              </>
            ) : installedIsStable ? (
              currentTag
            ) : targetTag ? (
              <span className="vs-target">{targetTag}</span>
            ) : null}
          </div>
          <div className="vs-ver-meta">
            {releasedAt && <span>{t('settings.releasedAt')} <b>{releasedAt}</b></span>}
          </div>
          {!p.hasUpdate && p.solo && (
            <div className="vs-ver-tagline">{t('settings.topbarMasterOnly')}</div>
          )}
        </div>

        {p.solo && <div className="vs-v-rule" />}

        <div className="vs-change-block">
          <div className="vs-h">
            {p.hasUpdate ? t('settings.targetUpdateContent', { label: targetTag }) : t('settings.targetThisVersion', { label: targetTag || currentTag || '' })}
          </div>
          <MasterReleaseNotes notes={p.releaseNotes} onShowDetail={p.onShowReleaseNotesDetail} />
        </div>
      </div>

      <div className="vs-chan-foot">
        <div className="vs-info">
          {p.check?.error
            ? <span style={{ color: 'var(--err)' }}>{p.check.error}</span>
            : <span>{t('settings.lastCheck', { time: checkedAt })}</span>}
        </div>
        <div className="vs-actions">
          <button onClick={p.onCheck} disabled={p.checking || p.busy} className="btn btn-sm">
            <VersionIcon name="refresh" />{p.checking ? t('settings.checkingEllipsis') : t('settings.checkUpdates')}
          </button>
          {/* state=update_available 才显示更新按钮；up_to_date / ahead / detached 不显示
              （上面 vs-meta 已经把"已是最新 / 本地领先 / 当前不在历史上"说清楚了）*/}
          {showUpdateButton && (
            <button onClick={p.onUpdate} disabled={p.busy || p.checking} className="btn btn-sm btn-primary">
              {p.busy ? t('settings.updatingEllipsis') : t('settings.updateTo', { tag: targetTag })}
            </button>
          )}
          {/* 装在非稳定版（dev / custom）时显示"切到最新稳定版"按钮；
              与"更新到 X"按钮互斥（shouldShowMasterUpdateButton 内部已按
              installed_kind 排除了非 stable），避免同屏显示两个做同样事的按钮 */}
          {showSwitchToStableButton && (
            <button onClick={p.onSwitchToMaster} disabled={p.busy || p.checking} className="btn btn-sm btn-primary">
              {p.busy ? t('settings.switchingEllipsis') : t('settings.switchToStable', { tag: p.check?.latest_version })}
            </button>
          )}
        </div>
      </div>

      {p.hasRollback && p.status?.rollback_target && (() => {
        // rollback 显示优先 tag（"v0.6.0"），否则 sha 前 8 位
        const sha = p.status.rollback_target
        const tag = p.status.rollback_target_tag
        const label = tag || sha.slice(0, 8)
        return (
          // 回滚是潜在破坏性操作（reset --hard 丢失当前 commit 上的本地未
          // commit 改动 / GC 后 reflog 也可能消失），UI 默认折叠成小字提示
          // 让用户主动确认才展开按钮，降低误触概率。
          <details className="vs-rollback-collapse">
            <summary className="vs-rollback-summary">
              <span className="vs-caret">▸</span>
              {t('settings.rollbackAvailable', { label })}
            </summary>
            <div className="vs-rollback-inline-row">
              <div className="vs-lhs">
                <span className="vs-ico"><VersionIcon name="rollback" /></span>
                <span>{t('settings.previousVersion')}</span>
                <b>{label}</b>
                {tag && <span className="vs-when">{sha.slice(0, 8)}</span>}
              </div>
              <button onClick={p.onRollback} disabled={p.busy || p.checking} className="btn btn-sm">
                {t('settings.switchBackTo', { label })}
              </button>
            </div>
          </details>
        )
      })()}
    </div>
  )
}

type DevCardProps = {
  on: boolean
  check: SystemUpdateCheck | null
  commits: DevCommitsResult | null
  currentSha: string
  installedKind: 'stable' | 'dev' | 'custom' | 'zip' | undefined
  selectedSha: string | null
  setSelectedSha: (sha: string | null) => void
  checking: boolean
  busy: boolean
  cardState: CardState
  pendingTarget: PendingTarget | null
  preflight: PreflightResult | null
  preflightLoading: boolean
  onCancelPreview: () => void
  onConfirmPreview: () => void
  onCheck: () => void
  onSwitchToDev: () => void
  onSwitchToCommit: (commit: DevCommit) => void
}

function DevCard(p: DevCardProps) {
  const { t } = useTranslation()
  const commits = p.commits?.commits ?? []
  const head = commits[0]?.short_sha ?? p.check?.latest_commit?.slice(0, 8)
  const selectedCommit = p.selectedSha ? commits.find((c) => c.sha === p.selectedSha) ?? null : null
  const fetchError = p.commits?.error ?? p.check?.error
  const currentShortSha = p.currentSha ? p.currentSha.slice(0, 8) : t('settings.currentShort')
  const stateText = formatDevStateText(p.check, t)
  // installedKind 用作 check 还没 resolve 期间的 fallback：装 dev tip 时
  // 按钮 disabled，避免显示可点但点了 no-op
  const devSwitchDisabled = isDevSwitchButtonDisabled(p.check, p.installedKind)
  if (p.cardState === 'preview' && p.pendingTarget && p.pendingTarget.kind === 'dev') {
    const target = p.pendingTarget
    return (
      <div className="vs-chan">
        <div className="vs-chan-head">
          <div className="vs-lhs">
            <span className="vs-name">{t('settings.devConfirmSwitch')}</span>
            <span className="vs-pill vs-pill-dev"><span className="vs-dot" />{t('settings.devBuild')}</span>
          </div>
          <button className="btn btn-sm btn-ghost" onClick={p.onCancelPreview} disabled={p.busy}>
            {t('settings.back')}
          </button>
        </div>
        <PreviewPane
          channel="dev"
          fromLabel={currentShortSha}
          toLabel={target.label}
          details={
            <div className="vs-change-block">
              <div className="vs-h">{t('settings.targetThisCommit', { label: target.label })}</div>
              {target.msg ? (
                <>
                  <div style={{ fontSize: 13, color: 'var(--fg-primary)', marginTop: 4, lineHeight: 1.5 }}>
                    {target.msg}
                  </div>
                  {target.author && (
                    <div className="vs-ver-meta" style={{ marginTop: 6 }}>
                      <span>author <b>{target.author}</b></span>
                    </div>
                  )}
                </>
              ) : (
                <div style={{ fontSize: 13, color: 'var(--fg-tertiary)', marginTop: 4 }}>
                  {t('settings.switchToDevHeadDesc')}
                </div>
              )}
            </div>
          }
          preflight={p.preflight}
          loading={p.preflightLoading}
          busy={p.busy}
          onCancel={p.onCancelPreview}
          onConfirm={p.onConfirmPreview}
        />
      </div>
    )
  }
  if (p.cardState === 'progress' && p.pendingTarget && p.pendingTarget.kind === 'dev') {
    return (
      <div className="vs-chan">
        <div className="vs-chan-head">
          <div className="vs-lhs">
            <span className="vs-name">{t('settings.devSwitching')}</span>
            <span className="vs-pill vs-pill-dev"><span className="vs-dot" />{t('settings.devBuild')}</span>
          </div>
        </div>
        <ProgressPane fromLabel={currentShortSha} toLabel={p.pendingTarget.label} />
      </div>
    )
  }

  return (
    <div className="vs-chan">
      <div className="vs-chan-head">
        <div className="vs-lhs">
          <span className="vs-name">{t('settings.devBuild')}</span>
          <span className="vs-pill vs-pill-dev"><span className="vs-dot" />{t('settings.devBuild')}</span>
        </div>
        <div className={`vs-meta${p.check?.state === 'update_available' ? ' attn' : ''}`}>
          {fetchError && !head ? (
            <span style={{ color: 'var(--err)' }}>{fetchError}</span>
          ) : head ? (
            <>
              dev HEAD <b style={{ color: 'var(--fg-secondary)', fontWeight: 500 }}>{head}</b>
              {p.check && <>{' · '}{stateText}</>}
            </>
          ) : (
            <span>{t('settings.notFetched')}</span>
          )}
        </div>
      </div>

      <div className="vs-change-block" style={{ paddingTop: 4, paddingBottom: 4 }}>
        <div className="vs-h">{t('settings.recentCommits')}</div>
        {commits.length === 0 ? (
          <ul className="vs-change-list">
            <li>
              <span className="vs-glyph">·</span>
              <span className="vs-txt">
                {fetchError
                  ? <span style={{ color: 'var(--err)' }}>{fetchError}</span>
                  : t('settings.fetchDevHint')}
              </span>
            </li>
          </ul>
        ) : (
          <>
            <ul className="vs-commits">
              {commits.map((c, i) => {
                const isHead = i === 0
                const isCurrent = !!p.currentSha && c.sha === p.currentSha
                const isSelected = c.sha === p.selectedSha
                const clickable = !isCurrent
                // 行 class 同时跟 isHead / isCurrent / clickable / selected。
                // accent glyph 走 .current（"你在这里"）；HEAD 只在 pill 里
                // 用文字标记（不抢 glyph）。
                const classes = ['vs-commit']
                if (isHead) classes.push('head')
                if (isCurrent) classes.push('current')
                if (clickable) classes.push('clickable')
                if (isSelected) classes.push('selected')
                return (
                  <li
                    key={c.sha}
                    className={classes.join(' ')}
                    onClick={() => clickable && p.setSelectedSha(isSelected ? null : c.sha)}
                    title={c.msg}
                  >
                    <span className="vs-glyph" />
                    <span className="vs-msg">{c.msg}</span>
                    <span className="vs-sha">{c.short_sha}</span>
                    <span className="vs-pill-slot">
                      {isCurrent ? (
                        <span className="vs-head-pill">{t('settings.currentMarker')}</span>
                      ) : isHead ? (
                        <span className="vs-head-pill">HEAD</span>
                      ) : isSelected ? (
                        <span className="vs-switch-hint">{t('settings.selectedMarker')}</span>
                      ) : (
                        <span className="vs-switch-hint">{t('settings.switchHere')}</span>
                      )}
                    </span>
                  </li>
                )
              })}
            </ul>
            {p.commits && !p.commits.fetched && p.commits.error && (
              <p className="vs-d" style={{ color: 'var(--warn)', marginTop: 6 }}>
                {t('settings.fetchFailedCached', { error: p.commits.error })}
              </p>
            )}
          </>
        )}
      </div>

      {p.selectedSha && selectedCommit ? (
        // 选中确认条：仅 sha + 取消/确认 按钮。commit 信息上方 list 已可见，
        // 这里只是 action 收尾，info 段去掉避免长 message 挤换行。
        <div className="vs-selection-foot">
          <span className="vs-info" title={selectedCommit.msg}>
            <b>{selectedCommit.short_sha}</b>
          </span>
          <div className="vs-actions">
            <button onClick={() => p.setSelectedSha(null)} disabled={p.busy} className="btn btn-sm btn-ghost">
              {t('settings.cancel')}
            </button>
            <button
              onClick={() => p.onSwitchToCommit(selectedCommit)}
              disabled={p.busy || p.checking}
              className="btn btn-sm btn-warn"
            >
              {p.busy ? t('settings.switchingEllipsis') : t('settings.switchToCommit', { sha: selectedCommit.short_sha })}
            </button>
          </div>
        </div>
      ) : (
        <div className="vs-chan-foot">
          <div className="vs-info">
            {p.check?.checked_at
              ? <span>{t('settings.lastFetch', { time: new Date(p.check.checked_at * 1000).toLocaleString() })}</span>
              : <span style={{ color: 'var(--fg-tertiary)' }}>{t('settings.notFetched')}</span>}
          </div>
          <div className="vs-actions">
            <button onClick={p.onCheck} disabled={p.checking || p.busy} className="btn btn-sm">
              <VersionIcon name="refresh" />{p.checking ? t('settings.fetchingEllipsis') : t('settings.fetchDev')}
            </button>
            {/* 切按钮 disabled 条件改用 commit 比较（state=up_to_date），不再
                看 branch / installed_kind —— 因为 release 直后存在"装的是 stable
                但 commit 恰好等于 dev HEAD"的边界，此时切操作是 no-op */}
            {devSwitchDisabled ? (
              <button disabled className="btn btn-sm">{t('settings.alreadyAtDevHead')}</button>
            ) : commits.length > 0 ? (
              <button
                onClick={p.onSwitchToDev}
                disabled={p.busy || p.checking}
                className="btn btn-sm btn-warn"
              >
                {p.busy ? t('settings.switchingEllipsis') : t('settings.switchToDev', { head: head ? ` (${head})` : '' })}
              </button>
            ) : null}
          </div>
        </div>
      )}
    </div>
  )
}

// 简易的 modal：点遮罩 / 按 ESC 关闭，pre + 等宽字体显示日志。
// 没用 useDialog 是因为它返回的是命令式 confirm/prompt 接口，不适合长文本展示。
function UpdateLogModal({
  loading, content, onClose,
}: { loading: boolean; content: string; onClose: () => void }) {
  const { t } = useTranslation()
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={onClose}
    >
      <div
        className="bg-surface border border-subtle rounded-md shadow-lg max-w-4xl w-[92vw] max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-subtle px-4 py-2.5">
          <h3 className="text-sm font-semibold text-fg-primary">{t('settings.updateLogTitle')}</h3>
          <button
            onClick={onClose}
            className="text-fg-dim hover:text-fg-primary text-lg leading-none"
            aria-label={t('common.close')}
          >×</button>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <span className="text-fg-dim text-sm">{t('common.loading')}</span>
          ) : (
            <pre className="text-2xs font-mono text-fg-primary whitespace-pre-wrap break-words">
              {content}
            </pre>
          )}
        </div>
      </div>
    </div>
  )
}

// chunk 2 重做 — release notes 全量详细内容 modal。结构：
//   header: tag · date · block summary
//   body: 每条 entry 一块（kind 徽章 + summary + PR refs 链接 + detail 文本）
// detail 字段是 markdown 但这里不渲染 markdown 库（依赖最少），直接
// whitespace-pre-wrap 显示原文，code/`` /列表用户能读懂；未来想真渲染 markdown
// 再加 marked / react-markdown 依赖。
function ReleaseNotesDetailModal({
  notes, onClose,
}: { notes: ReleaseNotes; onClose: () => void }) {
  const { t } = useTranslation()
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={onClose}
    >
      <div
        className="bg-surface border border-subtle rounded-md shadow-lg max-w-3xl w-[92vw] max-h-[85vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-subtle px-5 py-3">
          <div className="flex flex-col gap-0.5">
            <h3 className="text-base font-semibold text-fg-primary font-mono">
              {notes.tag}
              {notes.date && <span className="text-fg-tertiary font-normal text-sm font-sans"> · {notes.date}</span>}
            </h3>
            {notes.summary && (
              <span className="text-xs text-fg-secondary">{notes.summary}</span>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-fg-dim hover:text-fg-primary text-xl leading-none px-1"
            aria-label={t('common.close')}
          >×</button>
        </div>
        <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4">
          {notes.entries.map((e, i) => (
            <div key={i} className="flex flex-col gap-1.5">
              <div className="flex items-start gap-2 flex-wrap">
                <span
                  className={`vs-pill ${KIND_PILL_CLASS[e.kind] || 'vs-pill-info'}`}
                  style={{ flexShrink: 0, marginTop: 2 }}
                >
                  {t(`settings.${KIND_LABEL_KEY[e.kind] ?? ''}`, { defaultValue: e.kind })}
                </span>
                <span className="text-sm text-fg-primary font-medium leading-snug">
                  {e.summary}
                </span>
              </div>
              {e.pr_refs.length > 0 && (
                <div className="flex flex-wrap gap-1.5 ml-1 mt-0.5">
                  {e.pr_refs.map((pr) => (
                    <a
                      key={pr}
                      href={`https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/pull/${pr}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-2xs font-mono text-fg-tertiary hover:text-accent underline-offset-2 hover:underline"
                    >
                      #{pr}
                    </a>
                  ))}
                </div>
              )}
              {e.detail && (
                <pre className="text-xs font-mono text-fg-secondary whitespace-pre-wrap break-words bg-sunken border border-subtle rounded p-3 mt-1 leading-relaxed">
                  {e.detail.trimEnd()}
                </pre>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── 服务 Section（重启 server）─────────────────────────────────────────
function ServiceSection() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const dialog = useDialog()
  const [busy, setBusy] = useState(false)

  const handleRestart = async () => {
    const ok = await dialog.confirm(
      t('settings.confirmRestartService'),
      { tone: 'warn', okText: t('settings.restart') },
    )
    if (!ok) return

    setBusy(true)
    try {
      await api.restartServer()
    } catch (e) {
      const err = e as Error & { status?: number; detail?: { error?: string; tasks?: { name: string; id?: number }[] } }
      if (err.status === 422 && err.detail?.error === 'running_tasks_present') {
        const names = (err.detail.tasks ?? []).map((task) => task.name || `task#${task.id ?? '?'}`).join(', ')
        toast(t('settings.taskRunningCancelFirst', { names }), 'error')
      } else {
        toast(t('settings.restartTriggerFailed', { error: err.message ?? String(e) }), 'error')
      }
      setBusy(false)
      return
    }

    void pollHealthThenReload(toast, 5 * 60_000, t('settings.restart'), () => setBusy(false), t)
  }

  return (
    <SettingsSection id="service" title={t('settings.service')}>
      <SettingsField
        label={t('settings.serviceRestartTitle')}
        helpTooltip={
          <>
            <p>{t('settings.serviceRestartHelp1')}</p>
            <p><Trans i18nKey="settings.serviceRestartHelp2" components={{ code: <code /> }} /></p>
          </>
        }
      >
        <button
          onClick={() => void handleRestart()}
          disabled={busy}
          className="btn btn-secondary btn-sm self-start"
        >
          {busy ? t('settings.restarting') : t('settings.restartServer')}
        </button>
      </SettingsField>
    </SettingsSection>
  )
}
