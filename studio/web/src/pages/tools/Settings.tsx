import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from 'react'
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
  type TorchCuTag,
  type TorchStatus,
  type WandBConfig,
  type WD14Runtime,
} from '../../api/client'
import LLMTaggerWorkspace from '../../components/LLMTaggerWorkspace'
import PageHeader from '../../components/PageHeader'
import { useToast } from '../../components/Toast'
import { useEventStream } from '../../lib/useEventStream'
import { applyDensity, applyTheme, getStoredDensity, getStoredTheme, setStoredDensity, setStoredTheme, type Density, type Theme } from '../../lib/theme'

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

type Tab = 'dataset' | 'tagging' | 'training' | 'monitor' | 'testing' | 'appearance'

const TAB_LIST: { id: Tab; label: string }[] = [
  { id: 'dataset', label: '数据集' },
  { id: 'tagging', label: '打标' },
  { id: 'training', label: '训练' },
  { id: 'monitor', label: '监控' },
  { id: 'testing', label: '测试' },
  { id: 'appearance', label: '页面' },
]

// 每个 tab 的 section index — 用于右侧 sticky 导航。id 与各 section 的 DOM id
// 对应；label 在导航里直接显示。修改 section 顺序时记得同步这里。
const TAB_SECTIONS: Record<Tab, { id: string; label: string }[]> = {
  dataset: [
    { id: 'gelbooru', label: 'Gelbooru' },
    { id: 'danbooru', label: 'Danbooru' },
    { id: 'download-global', label: '下载（全局）' },
  ],
  tagging: [
    { id: 'llm-tagger', label: 'LLM Tagger' },
    { id: 'wd14', label: 'WD14' },
    { id: 'cltagger', label: 'CLTagger' },
    { id: 'onnxruntime', label: 'ONNX Runtime' },
  ],
  training: [
    { id: 'download-source', label: '模型下载源' },
    { id: 'queue', label: '队列调度' },
    { id: 'pytorch', label: 'PyTorch' },
    { id: 'flash-attn', label: 'Flash Attention' },
    { id: 'xformers', label: 'xformers' },
    { id: 'models', label: '训练模型' },
  ],
  monitor: [
    { id: 'wandb', label: 'Weights & Biases' },
  ],
  testing: [
    { id: 'preview', label: '中间步预览' },
  ],
  appearance: [
    { id: 'display', label: '显示' },
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
      v === 'dataset' || v === 'tagging' || v === 'training'
      || v === 'monitor' || v === 'testing' || v === 'appearance'
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
  huggingface: { token: '', endpoint: 'https://hf-mirror.com' },
  wandb: {
    enabled: false,
    api_key: '',
    project: 'AnimaLoraStudio',
    entity: '',
    base_url: '',
    mode: 'online',
    log_samples: false,
    sample_max_side: 512,
    sample_every_n_steps: 0,
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
    add_rating_tag: false,
    add_model_tag: false,
    blacklist_tags: [],
    batch_size: 8,
  },
  models: { root: null, selected_anima: 'preview3-base' },
  queue: { allow_gpu_during_train: false },
  generate: { preview_every_n_steps: 3, attention_backend: 'auto' },
}

const textInputClass = 'w-full px-2 py-1 outline-none rounded-sm bg-sunken border border-subtle text-sm text-fg-primary focus:border-accent'

export default function SettingsPage() {
  const [server, setServer] = useState<Secrets | null>(null)
  const [draft, setDraft] = useState<Secrets>(EMPTY)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [tab, setTab] = useState<Tab>(getStoredTab)
  // Models catalog hoisted here so 打标 tab 的 WD14/CLTagger 卡片和 训练 tab
  // 的 ModelsSection 共用一份数据 + 一个 SSE 订阅。
  const [catalog, setCatalog] = useState<ModelsCatalog | null>(null)
  const [catalogError, setCatalogError] = useState<string | null>(null)
  const [downloadBusy, setDownloadBusy] = useState<Set<string>>(new Set())
  const [llmModelsBusy, setLlmModelsBusy] = useState(false)
  const [llmTestBusy, setLlmTestBusy] = useState(false)
  const { toast } = useToast()
  // 右侧 section index 用：sticky nav 的 IntersectionObserver root + 滚动平移容器
  const scrollContainerRef = useRef<HTMLDivElement>(null)

  const switchTab = (next: Tab) => {
    setTab(next)
    try {
      localStorage.setItem(TAB_STORAGE_KEY, next)
    } catch {
      /* ignore localStorage errors */
    }
  }

  const reloadCatalog = useCallback(async () => {
    try {
      const c = await api.getModelsCatalog()
      setCatalog(c)
      setCatalogError(null)
    } catch (e) {
      setCatalogError(String(e))
    }
  }, [])

  useEffect(() => { void reloadCatalog() }, [reloadCatalog])

  useEventStream((evt) => {
    if (evt.type === 'model_download_changed') { void reloadCatalog() }
  })

  const startDownload = useCallback(async (model_id: string, variant?: string) => {
    const key = variant ? `${model_id}:${variant}` : model_id
    setDownloadBusy((s) => new Set(s).add(key))
    try {
      await api.startModelDownload({ model_id, variant })
      toast(`开始下载 ${key}`, 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setDownloadBusy((s) => { const n = new Set(s); n.delete(key); return n })
    }
  }, [reloadCatalog, toast])

  useEffect(() => {
    api
      .getSecrets()
      .then((s) => {
        setServer(s)
        setDraft(s)
      })
      .catch((e) => setError(String(e)))
  }, [])

  const dirty = useMemo(
    () => server !== null && JSON.stringify(server) !== JSON.stringify(draft),
    [server, draft]
  )

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
      toast('已保存', 'success')
    } catch (e) {
      setError(String(e))
      toast('保存失败', 'error')
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
    const next: LLMPreset = _makeFallbackPreset(id, `新预设 ${idx}`, 'json')
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

  const saveAsNewPreset = () => {
    const label = window.prompt('新预设名称：', `${currentPreset.label} 副本`)
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
      toast(`已读取 ${result.items.length} 个模型`, 'success')
    } catch (e) {
      setError(String(e))
      toast(`读取模型列表失败：${e}`, 'error')
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
      const parts: string[] = [result.ok ? 'LLM 连接通过' : 'LLM 连接失败']
      if (result.elapsed_ms > 0) parts.push(`${result.elapsed_ms} ms`)
      if (result.status_code !== null) parts.push(`HTTP ${result.status_code}`)
      if (!result.ok) {
        const detail = result.error || result.response_preview
        if (detail) parts.push(detail.slice(0, 120))
      }
      toast(parts.join(' · '), result.ok ? 'success' : 'error')
    } catch (e) {
      toast(`LLM 连接测试失败：${String(e)}`, 'error')
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
      {TAB_LIST.map((t) => (
        <button
          key={t.id}
          onClick={() => switchTab(t.id)}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            tab === t.id
              ? 'border-accent text-fg-primary'
              : 'border-transparent text-fg-tertiary hover:text-fg-secondary'
          }`}
        >
          {t.label}
        </button>
      ))}
    </nav>
  )

  return (
    <div className="flex flex-col h-full min-h-0">
      <PageHeader
        title="设置"
        tabs={tabNav}
        sticky
        actions={
          <button
            onClick={save}
            disabled={!dirty || saving}
            className="btn btn-primary btn-sm"
          >
            {saving ? '保存中...' : '保存'}
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
          <input
            type="text"
            value={draft.gelbooru.user_id}
            onChange={(e) => update('gelbooru', 'user_id', e.target.value)}
            className={textInputClass}                                  />
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
          <input
            type="text"
            value={draft.danbooru.username}
            onChange={(e) => update('danbooru', 'username', e.target.value)}
            placeholder="必填 — danbooru 挂了 Cloudflare 后不再支持匿名"
            className={textInputClass}                                  />
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
            <option value="free">free（max 2 tag）</option>
            <option value="gold">gold（max 6 tag）</option>
            <option value="platinum">platinum（max 12 tag）</option>
          </select>
        </SettingsField>
      </SettingsSection>

      <SettingsSection id="download-global" title="下载（全局）">
        <SettingsField
          label="exclude_tags"
          desc="逗号分隔；搜索时自动追加 -tag，Gelbooru / Danbooru 同样生效"
        >
          <input
            type="text"
            value={draft.download.exclude_tags.join(', ')}
            onChange={(e) =>
              update('download', 'exclude_tags',
                e.target.value.split(',').map((t) => t.trim().replace(/^-+/, '')).filter(Boolean)
              )
            }
            placeholder="例：comic, monochrome, lowres"
            className={textInputClass}                                  />
        </SettingsField>

        <div className="grid grid-cols-3 gap-3 pt-2 border-t border-subtle">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-fg-secondary font-mono">parallel_workers</label>
            <input
              type="number" min={1} max={16}
              value={draft.download.parallel_workers}
              onChange={(e) => update('download', 'parallel_workers', Math.max(1, Number(e.target.value) || 1))}
              className={`${textInputClass} max-w-24`}                              />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-fg-secondary font-mono">api_rate_per_sec</label>
            <input
              type="number" step="0.5" min={0.5} max={10}
              value={draft.download.api_rate_per_sec}
              onChange={(e) => update('download', 'api_rate_per_sec', Math.max(0.5, Number(e.target.value) || 0.5))}
              className={`${textInputClass} max-w-24`}                              />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-fg-secondary font-mono">cdn_rate_per_sec</label>
            <input
              type="number" step="1" min={1} max={20}
              value={draft.download.cdn_rate_per_sec}
              onChange={(e) => update('download', 'cdn_rate_per_sec', Math.max(1, Number(e.target.value) || 1))}
              className={`${textInputClass} max-w-24`}                              />
          </div>
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
        />
        <SettingsField label="local_dir" desc="留空 = 自动 HF 下载">
          <input
            type="text"
            value={draft.wd14.local_dir ?? ''}
            onChange={(e) => update('wd14', 'local_dir', e.target.value || null)}
            className={textInputClass}                                  />
        </SettingsField>
        <div className="grid grid-cols-2 gap-3">
          <SettingsField label="threshold_general">
            <input
              type="number" step="0.01" min={0} max={1}
              value={draft.wd14.threshold_general}
              onChange={(e) => update('wd14', 'threshold_general', Number(e.target.value))}
              className={`${textInputClass} max-w-32`}                              />
          </SettingsField>
          <SettingsField label="threshold_character">
            <input
              type="number" step="0.01" min={0} max={1}
              value={draft.wd14.threshold_character}
              onChange={(e) => update('wd14', 'threshold_character', Number(e.target.value))}
              className={`${textInputClass} max-w-32`}                              />
          </SettingsField>
        </div>
        <SettingsField label="blacklist_tags" desc="逗号分隔">
          <input
            type="text"
            value={draft.wd14.blacklist_tags.join(', ')}
            onChange={(e) => update('wd14', 'blacklist_tags', e.target.value.split(',').map((t) => t.trim()).filter(Boolean))}
            className={textInputClass}                                  />
        </SettingsField>
        <SettingsField label="batch_size" desc="GPU 推理一批塞几张；CPU 自动降到 1">
          <input
            type="number" min={1} max={64}
            value={draft.wd14.batch_size}
            onChange={(e) => update('wd14', 'batch_size', Math.max(1, Number(e.target.value) || 1))}
            className={`${textInputClass} max-w-24`}                              />
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
        />
        <SettingsField label="local_dir" desc="留空 = 自动 HF 下载">
          <input
            type="text"
            value={draft.cltagger.local_dir ?? ''}
            onChange={(e) => update('cltagger', 'local_dir', e.target.value || null)}
            className={textInputClass}                                  />
        </SettingsField>
        <div className="grid grid-cols-2 gap-3">
          <SettingsField label="threshold_general">
            <input
              type="number" step="0.01" min={0} max={1}
              value={draft.cltagger.threshold_general}
              onChange={(e) => update('cltagger', 'threshold_general', Number(e.target.value))}
              className={`${textInputClass} max-w-32`}                                      />
          </SettingsField>
          <SettingsField label="threshold_character">
            <input
              type="number" step="0.01" min={0} max={1}
              value={draft.cltagger.threshold_character}
              onChange={(e) => update('cltagger', 'threshold_character', Number(e.target.value))}
              className={`${textInputClass} max-w-32`}                                      />
          </SettingsField>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <SettingsField label="add_rating_tag">
            <Bool value={draft.cltagger.add_rating_tag} onChange={(v) => update('cltagger', 'add_rating_tag', v)} />
          </SettingsField>
          <SettingsField label="add_model_tag">
            <Bool value={draft.cltagger.add_model_tag} onChange={(v) => update('cltagger', 'add_model_tag', v)} />
          </SettingsField>
        </div>
        <SettingsField label="blacklist_tags" desc="逗号分隔">
          <input
            type="text"
            value={draft.cltagger.blacklist_tags.join(', ')}
            onChange={(e) => update('cltagger', 'blacklist_tags', e.target.value.split(',').map((t) => t.trim()).filter(Boolean))}
            className={textInputClass}                                  />
        </SettingsField>
        <SettingsField label="batch_size" desc="GPU 推理一批塞几张；CPU 自动降到 1">
          <input
            type="number" min={1} max={64}
            value={draft.cltagger.batch_size}
            onChange={(e) => update('cltagger', 'batch_size', Math.max(1, Number(e.target.value) || 1))}
            className={`${textInputClass} max-w-24`}                              />
        </SettingsField>
      </SettingsSection>

      <ONNXRuntimeSection />
      </>)}

      {tab === 'training' && (<>
      <SettingsSection id="download-source" title="模型下载源">
        <SettingsField label="下载源" desc="魔搭（ModelScope）对国内用户更快；无映射的模型自动回退 HuggingFace">
          <DownloadSourceSelect
            value={draft.download_source}
            onChange={(v) => updateTop('download_source', v)}
          />
        </SettingsField>

        {/* 下方按当前下载源条件渲染对应凭证配置。HF/ModelScope token 都保留在
         * secrets 里（即便切换源也不丢失），只是 UI 一次只露面一份。 */}
        {draft.download_source === 'huggingface' ? (
          <>
            <SettingsField label="token" desc="用于 HF 私有 repo 鉴权；公开仓库（含 SmilingWolf WD14 / cella110n CLTagger）不用填">
              <SensitiveInput
                value={draft.huggingface.token}
                serverValue={server?.huggingface.token ?? ''}
                onChange={(v) => update('huggingface', 'token', v)}
              />
            </SettingsField>
            <SettingsField label="endpoint" desc="模型下载端点。国内推荐 hf-mirror，海外推荐官方源">
              <HFEndpointSelect
                value={draft.huggingface.endpoint}
                onChange={(v) => update('huggingface', 'endpoint', v)}
              />
            </SettingsField>
          </>
        ) : (
          <SettingsField label="token" desc="公开模型不用填；私有仓库或限速时需要。需先 pip install modelscope 安装命令行工具。">
            <SensitiveInput
              value={draft.modelscope.token}
              serverValue={server?.modelscope.token ?? ''}
              onChange={(v) => update('modelscope', 'token', v)}
            />
          </SettingsField>
        )}
      </SettingsSection>

      <SettingsSection id="queue" title="队列调度">
        <SettingsField label="允许 GPU 任务与训练并行">
          <div className="flex items-center gap-3">
            <Bool value={draft.queue.allow_gpu_during_train} onChange={(v) => update('queue', 'allow_gpu_during_train', v)} />
            <span className="text-xs text-warn">
              WD14 打标推理 onnxruntime-gpu 大约占 ~2 GB；确认训练之外的剩余显存够再打开，否则 OOM
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
      />
      </>)}

      {tab === 'monitor' && (<>
      <SettingsSection id="wandb" title="Weights & Biases">
        <SettingsField label="启用 WandB" desc="打开后所有训练任务都会写入 W&B；不再占用训练配置字段">
          <div className="flex items-center gap-3">
            <Bool value={draft.wandb.enabled} onChange={(v) => update('wandb', 'enabled', v)} />
            <span className="text-xs text-fg-tertiary">
              需要训练环境已安装 wandb 包
            </span>
          </div>
        </SettingsField>
        <SettingsField label="api_key">
          <SensitiveInput
            value={draft.wandb.api_key}
            serverValue={server?.wandb.api_key ?? ''}
            onChange={(v) => update('wandb', 'api_key', v)}
          />
        </SettingsField>
        <SettingsField label="project">
          <input
            type="text"
            value={draft.wandb.project}
            onChange={(e) => update('wandb', 'project', e.target.value)}
            placeholder="AnimaLoraStudio"
            className={textInputClass}
          />
        </SettingsField>
        <SettingsField label="entity" desc="可选；团队/用户名，留空使用 wandb 默认账号">
          <input
            type="text"
            value={draft.wandb.entity}
            onChange={(e) => update('wandb', 'entity', e.target.value)}
            className={textInputClass}
          />
        </SettingsField>
        <SettingsField label="base_url" desc="可选；私有 W&B/self-hosted 时填写">
          <input
            type="text"
            value={draft.wandb.base_url}
            onChange={(e) => update('wandb', 'base_url', e.target.value)}
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
          <SettingsField label="记录采样图" desc="开启后训练采样图会上传到 wandb.ai 公网；私有 IP / NSFW 数据集请保持关闭">
            <Bool value={draft.wandb.log_samples} onChange={(v) => update('wandb', 'log_samples', v)} />
          </SettingsField>
        </div>
        {draft.wandb.log_samples && (
          <div className="grid grid-cols-2 gap-3">
            <SettingsField label="采样图最长边" desc="上传前缩到此像素，原图常 2K+，512 已够 wandb 浏览">
              <input
                type="number"
                min={64}
                step={64}
                value={draft.wandb.sample_max_side}
                onChange={(e) => update('wandb', 'sample_max_side', Math.max(64, parseInt(e.target.value) || 512))}
                className={textInputClass}
              />
            </SettingsField>
            <SettingsField label="step 节流" desc="0 = 不额外节流；>0 时只在 global_step % N == 0 上传，避免长训练上 GB 级图">
              <input
                type="number"
                min={0}
                step={50}
                value={draft.wandb.sample_every_n_steps}
                onChange={(e) => update('wandb', 'sample_every_n_steps', Math.max(0, parseInt(e.target.value) || 0))}
                className={textInputClass}
              />
            </SettingsField>
          </div>
        )}
      </SettingsSection>
      </>)}

      {tab === 'testing' && (<>
        {/* attention 后端走全局 auto-detect，UI 不暴露切换；想强制覆盖
            的高级用户改 secrets.json 的 generate.attention_backend
            （flash_attn / xformers / none）。安装管理在『训练』tab。 */}
        <TaeFluxSection draft={draft} update={update} />
      </>)}

      {tab === 'appearance' && (
        <DisplaySection />
      )}
    </div>

    <SectionIndex sections={TAB_SECTIONS[tab]} scrollContainer={scrollContainerRef} />
    </div>
    </div>
    </div>
  )
}

// ── Section / Field ────────────────────────────────────────────────────────

function SettingsSection({ id, title, children }: { id?: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="rounded-md border border-subtle bg-surface p-4 flex flex-col gap-3 scroll-mt-24">
      <h2 className="text-sm font-semibold text-fg-primary mb-0.5">{title}</h2>
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
  sections: { id: string; label: string }[]
  scrollContainer: RefObject<HTMLDivElement>
}) {
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
        <div className="caption mb-2 px-2">本页索引</div>
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
            {s.label}
          </button>
        ))}
      </nav>
    </aside>
  )
}

function SettingsField({ label, desc, children }: {
  label: string
  desc?: string
  children: React.ReactNode
}) {
  return (
    <div className="grid grid-cols-[240px_1fr] gap-3 items-start">
      <div className="flex flex-col gap-0.5 pt-1.5">
        <label className="text-xs text-fg-secondary font-mono leading-none">{label}</label>
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
  const masked = value === MASK
  return (
    <input
      type="password"
      value={masked ? '' : value}
      placeholder={serverValue === MASK ? '已保存（不显示），输入新值才覆盖' : ''}
      onChange={(e) => onChange(e.target.value || MASK)}
      className={textInputClass}                />
  )
}

// ── HFEndpointSelect ────────────────────────────────────────────────────────
//
// HF 模型下载 endpoint 选择器：3 个 preset + 自定义 URL 输入。
// 默认 hf-mirror.com（项目主战场国内）；海外用户切到 huggingface.co；
// 也支持粘贴自定义 URL（自建反代 / sjtug / 腾讯镜像等）。

const HF_ENDPOINT_PRESETS: { value: string; label: string; hint: string }[] = [
  { value: 'https://hf-mirror.com', label: 'hf-mirror.com', hint: '国内推荐（社区维护反代）' },
  { value: '', label: 'huggingface.co', hint: '官方源；海外推荐' },
  { value: '__custom__', label: '自定义 URL...', hint: '粘贴自建反代 / sjtug / 腾讯等' },
]

function HFEndpointSelect({ value, onChange }: {
  value: string; onChange: (v: string) => void
}) {
  const isPreset = HF_ENDPOINT_PRESETS.some(p => p.value !== '__custom__' && p.value === value)
  const [mode, setMode] = useState<'preset' | 'custom'>(isPreset ? 'preset' : 'custom')
  const selectedPreset = isPreset
    ? value
    : (mode === 'custom' ? '__custom__' : 'https://hf-mirror.com')

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
            {p.label}{p.hint ? ` — ${p.hint}` : ''}
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
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`${textInputClass} max-w-xs`}
    >
      <option value="huggingface">HuggingFace（含 hf-mirror 等镜像）</option>
      <option value="modelscope">ModelScope（魔搭社区，国内直连）</option>
    </select>
  )
}

// ── ModelIdsEditor ──────────────────────────────────────────────────────────

function ModelIdsEditor({ ids, currentId, onChange }: {
  ids: string[]; currentId: string; onChange: (next: string[]) => void
}) {
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
                <span className="text-xs text-accent">当前</span>
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
          placeholder="添加 HuggingFace 模型 ID"
          className={`${textInputClass} flex-1`}                            />
        <button onClick={add} disabled={!draft.trim() || seen.has(draft.trim())} className="btn btn-secondary btn-sm">+ 添加</button>
      </div>
    </div>
  )
}

// ── WD14 / CLTagger Model Cards（打标 tab 内嵌的模型管理器） ─────────────────

function WD14ModelCard({
  catalog, busy, start,
  currentModelId, onSelectModelId,
  candidates, onCandidatesChange,
}: {
  catalog: ModelsCatalog | null
  busy: Set<string>
  start: (model_id: string, variant?: string) => Promise<void>
  currentModelId: string
  onSelectModelId: (id: string) => void
  candidates: string[]
  onCandidatesChange: (next: string[]) => void
}) {
  const [advOpen, setAdvOpen] = useState(false)
  const wd14 = catalog?.wd14
  if (!wd14) {
    return <p className="text-fg-tertiary text-xs">加载模型清单...</p>
  }
  return (
    <ModelGroupCard title={wd14.name + '（候选模型）'}>
      <p className="text-xs text-fg-tertiary m-0">
        {wd14.description} · 选中作为当前 model_id；下载缺的版本。
      </p>
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
                title="选作 WD14 当前 model_id"
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
        {advOpen ? '▾' : '▸'} 候选编辑（添加/删除自定义 model_id）
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
  modelId, onModelIdChange,
}: {
  catalog: ModelsCatalog | null
  busy: Set<string>
  start: (model_id: string, variant?: string) => Promise<void>
  currentModelPath: string
  currentTagMappingPath: string
  onSelectVariant: (v: CLTaggerVariantInfo) => void
  modelId: string
  onModelIdChange: (id: string) => void
}) {
  const [advOpen, setAdvOpen] = useState(false)
  const cl = catalog?.cltagger
  if (!cl) {
    return <p className="text-fg-tertiary text-xs">加载模型清单...</p>
  }
  return (
    <ModelGroupCard title={cl.name + '（版本）'}>
      <p className="text-xs text-fg-tertiary m-0">
        {cl.description} · <code>{cl.repo}</code>
      </p>
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
                title="选作 CLTagger 当前版本"
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
        {advOpen ? '▾' : '▸'} 自定义 repo（高级）
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

function ModelsSection({ catalog, busy, start, reloadCatalog, catalogError }: {
  catalog: ModelsCatalog | null
  busy: Set<string>
  start: (model_id: string, variant?: string) => Promise<void>
  reloadCatalog: () => Promise<void>
  catalogError: string | null
}) {
  const { toast } = useToast()
  const [rootDraft, setRootDraft] = useState<string>('')
  const [serverRoot, setServerRoot] = useState<string | null>(null)
  const [savingRoot, setSavingRoot] = useState(false)
  const [selectedAnima, setSelectedAnima] = useState<string>('preview3-base')

  // 一次性拉一份 secrets 取 models.root + selected_anima（这两项走独立 PUT，
  // 不进 SettingsPage 的全局 dirty 流程）。catalog 由父级注入。
  useEffect(() => {
    void api.getSecrets().then((sec) => {
      setServerRoot(sec.models?.root ?? null)
      setRootDraft(sec.models?.root ?? '')
      setSelectedAnima(sec.models?.selected_anima ?? 'preview3-base')
    }).catch(() => {})
  }, [])

  const pickAnima = async (variant: string) => {
    if (variant === selectedAnima) return
    setSelectedAnima(variant)
    try {
      await api.updateSecrets({ models: { selected_anima: variant } })
      toast(`默认主模型已切到 ${variant}`, 'success')
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
      toast(v ? `已保存模型根目录: ${v}` : '已恢复默认模型根目录', 'success')
      setServerRoot(v ? v : null)
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setSavingRoot(false)
    }
  }

  const rootDirty = rootDraft.trim() !== (serverRoot ?? '')
  const error = catalogError

  return (
    <SettingsSection id="models" title="训练模型（一键下载）">
      <SettingsField label="模型根目录 (models_root)">
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <input
            type="text"
            value={rootDraft}
            onChange={(e) => setRootDraft(e.target.value)}
            placeholder="留空 = 默认 REPO_ROOT/anima/"
            className={`${textInputClass} flex-1`}                                  />
          <button onClick={saveRoot} disabled={!rootDirty || savingRoot} className="btn btn-primary btn-sm"
            title={rootDirty ? '保存路径配置' : '未修改'}>
            {savingRoot ? '保存中...' : '保存路径'}
          </button>
          <button onClick={() => setRootDraft(serverRoot ?? '')} disabled={!rootDirty || savingRoot}
            className="px-2 py-0.5 text-fg-tertiary bg-transparent border-none cursor-pointer rounded-sm"
            style={{ opacity: !rootDirty ? 0.3 : 1 }}
          >↻</button>
        </div>
      </SettingsField>

      {error && <div className="text-err text-xs font-mono">{error}</div>}
      {!catalog ? (
        <p className="text-fg-tertiary text-xs">加载...</p>
      ) : (
        <div className="flex flex-col gap-2">
          {/* Anima 主模型 */}
          <ModelGroupCard title={catalog.anima_main.name}>
            <p className="text-xs text-fg-tertiary m-0">
              {catalog.anima_main.description} · <code>{catalog.anima_main.repo}</code>
              <br />选中的版本会作为<strong className="text-fg-primary">新建 version</strong>的默认 transformer。
            </p>
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
                      title={canSelect ? '选作默认主模型' : v.exists ? '下载中...' : '未下载，请先下载'}
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
              <span className="text-fg-tertiary">{catalog.anima_vae.description} · <code>{catalog.anima_vae.repo}</code></span>
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
                  <span className="text-fg-tertiary">{m.description} · <code>{m.repo}</code></span>
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
                下载日志 ({Object.values(catalog.downloads).filter((d) => d.status === 'running' || d.status === 'failed').length})
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
                      {d.log_tail.join('\n') || '(等待日志...)'}
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

function ModelGroupCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-sm border border-subtle bg-sunken p-2.5">
      <h4 className="text-xs font-semibold text-fg-primary mb-1.5">{title}</h4>
      {children}
    </div>
  )
}

function ModelStatusBadge({ exists, size, status, fileCount, existsCount }: {
  exists: boolean; size: number; status?: ModelDownloadStatus['status']; fileCount?: number; existsCount?: number
}) {
  if (status === 'running') {
    return <StatusLabel bg="bg-warn-soft" fg="text-warn" text="下载中..." pulse />
  }
  if (status === 'failed') {
    return <StatusLabel bg="bg-err-soft" fg="text-err" text="失败" />
  }
  if (exists) {
    return <StatusLabel bg="bg-ok-soft" fg="text-ok" text={`✓ ${fmtBytes(size)}${fileCount !== undefined ? ` (${existsCount}/${fileCount})` : ''}`} />
  }
  if (fileCount !== undefined && existsCount! > 0) {
    return <StatusLabel bg="bg-warn-soft" fg="text-warn" text={`部分 (${existsCount}/${fileCount})`} />
  }
  return <StatusLabel bg="bg-overlay" fg="text-fg-tertiary" text="未下载" />
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
  const running = status === 'running' || busy
  if (running) {
    return <button disabled className="btn btn-secondary btn-sm" style={{ opacity: 0.5 }}>...</button>
  }
  return (
    <button onClick={onClick} className={exists ? 'btn btn-secondary btn-sm' : 'btn btn-primary btn-sm'}
      title={exists ? '已下载，点击重新下载' : '下载'}>
      {exists ? '↻ 重下' : '⤓ 下载'}
    </button>
  )
}

// ── ONNX Runtime Section（WD14 + CLTagger 共用 onnxruntime 包管理） ─────────

function ONNXRuntimeSection() {
  const [rt, setRt] = useState<WD14Runtime | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<null | 'auto' | 'gpu' | 'cpu'>(null)
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

  const install = async (target: 'auto' | 'gpu' | 'cpu') => {
    const detail = target === 'auto' ? '将按 nvidia-smi 检测自动选 GPU/CPU 包'
      : target === 'gpu' ? '将卸载现有 onnxruntime 并安装 onnxruntime-gpu'
      : '将卸载现有 onnxruntime-gpu 并安装 onnxruntime（CPU）'
    if (!confirm(`${detail}。装包需要几分钟。\n\n注意：装完后必须重启 Studio 才能生效。继续？`)) return
    setBusy(target)
    try {
      const result = await api.installWD14Runtime(target)
      setRt({
        installed: result.installed, version: result.version, providers: result.providers,
        cuda_available: result.cuda_available, restart_required: result.restart_required,
        cuda_load_error: result.cuda_load_error, preload: result.preload, cuda_detect: result.cuda_detect,
      })
      const newPkg = result.installed_pkg ?? result.installed ?? '?'
      const newVer = result.installed_version ?? result.version ?? '?'
      toast(`已装 ${newPkg}==${newVer}，请重启 Studio 让 EP 生效`, 'success')
    } catch (e) {
      toast(`装包失败: ${e}`, 'error')
    } finally {
      setBusy(null)
    }
  }

  const cuda = rt?.cuda_detect ?? { available: false, driver_version: null, gpu_name: null }
  const mismatched = !!rt && cuda.available && !rt.cuda_available
  // 默认状态正常时整体折叠；有错 / mismatch / 需重启时自动展开
  const hasIssue = !!error || (rt && (
    !!rt.cuda_load_error || rt.restart_required || mismatched
  ))

  // summary 里显示一行简短状态，用户不展开就能扫到
  const statusLabel = error
    ? '⚠ 加载状态失败'
    : !rt
      ? '加载中...'
      : rt.cuda_load_error
        ? '⚠ CUDA 加载失败'
        : rt.restart_required
          ? '⚠ 需重启 Studio'
          : mismatched
            ? '⚠ GPU 但跑 CPU EP'
            : rt.cuda_available
              ? `CUDA · ${rt.installed ?? '?'}`
              : `CPU · ${rt.installed ?? '?'}`
  const statusOk = rt && !hasIssue

  return (
    <details id="onnxruntime" open={!!hasIssue} className="rounded-md border border-subtle bg-surface group scroll-mt-24">
      <summary className="cursor-pointer p-4 list-none flex items-center gap-2">
        <span className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3">▸</span>
        <h2 className="text-sm font-semibold text-fg-primary m-0">ONNX Runtime</h2>
        <span className="text-xs text-fg-tertiary">WD14 / CLTagger 共用</span>
        <span className={`ml-auto text-xs font-mono ${statusOk ? 'text-ok' : 'text-warn'}`}>{statusLabel}</span>
      </summary>

      <div className="px-4 pb-4 flex flex-col gap-3">
        {error && <div className="text-err text-xs font-mono">{error}</div>}
        {!error && !rt && <div className="text-xs text-fg-tertiary">加载 runtime 状态...</div>}
        {rt && (
          <>
            <div className="rounded-sm border border-subtle bg-sunken p-2 flex flex-col gap-1 text-xs">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-fg-tertiary shrink-0">runtime:</span>
                <code className="font-mono text-fg-primary">{rt.installed ?? '(未安装)'}{rt.version ? `==${rt.version}` : ''}</code>
                <StatusLabel bg={rt.cuda_available ? 'bg-ok-soft' : 'bg-warn-soft'} fg={rt.cuda_available ? 'text-ok' : 'text-warn'} text={rt.cuda_available ? 'CUDA' : 'CPU only'} />
              </div>
              <div className="text-fg-tertiary">EP: <code className="text-fg-secondary font-mono">{(rt.providers ?? []).map((p) => p.replace('ExecutionProvider', '')).join(' / ') || '(none)'}</code></div>
              <div className="text-fg-tertiary">GPU 检测: <span className="text-fg-secondary">{cuda.available ? `${cuda.gpu_name ?? '?'} (driver ${cuda.driver_version ?? '?'})` : '未检测到 NVIDIA GPU'}</span></div>
            </div>

            {rt.restart_required && (
              <div className="rounded-sm border border-err bg-err-soft px-2 py-1.5 text-err text-xs">
                已装新 onnxruntime 包，但当前进程仍在用旧的。<strong>请重启 Studio</strong> 让 EP 切换生效。
              </div>
            )}
            {!rt.restart_required && mismatched && (
              <div className="rounded-sm border border-info bg-info-soft px-2 py-1.5 text-info text-xs">
                检测到 NVIDIA GPU 但 onnxruntime 只有 CPU EP — WD14 / CLTagger 会跑得很慢。展开「强制重装」装 GPU 版本。
              </div>
            )}
            {rt.cuda_load_error && (
              <div className="rounded-sm border border-err bg-err-soft px-2 py-1.5 text-xs text-err">
                <div>CUDA EP 加载失败，已降级到 CPU。</div>
                <code className="block font-mono text-xs text-err break-all whitespace-pre-wrap mt-1">
                  {rt.cuda_load_error}
                </code>
              </div>
            )}

            <div className="flex gap-1.5 items-center flex-wrap">
              <button onClick={() => install('auto')} disabled={busy !== null} className="btn btn-primary btn-sm">
                {busy === 'auto' ? '装包中...' : '自动检测 + 装合适的包'}
              </button>
              <button onClick={() => void refresh()} disabled={busy !== null} title="刷新状态"
                className="px-2 py-0.5 text-fg-tertiary bg-transparent border-none cursor-pointer rounded-sm">↻</button>
              <button type="button" onClick={() => setReinstallOpen(!reinstallOpen)}
                className="btn btn-ghost btn-sm text-xs text-fg-tertiary ml-auto">
                {reinstallOpen ? '▾' : '▸'} 强制重装（高级）
              </button>
            </div>
            {reinstallOpen && (
              <div className="flex gap-1.5 items-center flex-wrap pt-2 border-t border-subtle">
                <button onClick={() => install('gpu')} disabled={busy !== null} className="btn btn-secondary btn-sm">{busy === 'gpu' ? '装包中...' : '重装为 GPU'}</button>
                <button onClick={() => install('cpu')} disabled={busy !== null} className="btn btn-secondary btn-sm">{busy === 'cpu' ? '装包中...' : '重装为 CPU'}</button>
                <span className="text-[10px] text-fg-tertiary">不知道选哪个就用上面"自动检测"。</span>
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
    const msg = `将注册 torch 重装请求（${tag} 版）。\n` +
      `提交后请 Ctrl+C 关闭 Studio 重新运行 studio.bat —— 启动时会装 torch（~3 GB，5-30 分钟）。继续？`
    if (!confirm(msg)) return
    setBusy(true)
    try {
      const result = await api.reinstallTorch(target)
      // 后端已写 marker，server 进程没真装；提示用户去重启
      toast(result.message, 'success')
    } catch (e) {
      toast(`注册失败: ${e}`, 'error')
    } finally {
      setBusy(false)
    }
  }

  const hasIssue = !!error || (status && (status.is_cpu_with_gpu || status.is_cuda_build_unavailable || !status.installed))
  const statusOk = status?.cuda_available && !error
  const statusLabel = error
    ? '加载失败'
    : !status
      ? '加载中...'
      : !status.installed
        ? '未安装'
        : status.is_cpu_with_gpu
          ? 'CPU 版（误装）'
          : !status.cuda_available && status.cuda_build !== 'cpu'
            ? 'CUDA 不可用（驱动？）'
            : status.cuda_available
              ? `CUDA ✓ ${status.cuda_build}`
              : `CPU ${status.cuda_build}`

  return (
    <details id="pytorch" open={!!hasIssue} className="rounded-md border border-subtle bg-surface group scroll-mt-24">
      <summary className="cursor-pointer p-4 list-none flex items-center gap-2">
        <span className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3">▸</span>
        <h2 className="text-sm font-semibold text-fg-primary m-0">PyTorch</h2>
        <span className="text-xs text-fg-tertiary">训练核心依赖</span>
        <span className={`ml-auto text-xs font-mono ${statusOk ? 'text-ok' : status?.is_cpu_with_gpu ? 'text-err' : 'text-warn'}`}>
          {statusLabel}
        </span>
      </summary>

      <div className="px-4 pb-4 flex flex-col gap-3">
        {error && <div className="text-err text-xs font-mono">{error}</div>}
        {!error && !status && <div className="text-xs text-fg-tertiary">加载状态...</div>}

        {status && (<>
          {/* 当前状态卡 */}
          <div className="rounded-sm border border-subtle bg-sunken p-2 flex flex-col gap-1 text-xs">
            <div className="flex gap-4 flex-wrap">
              <span className="text-fg-tertiary">torch: <code className="text-fg-secondary font-mono">{status.version ?? '（未安装）'}</code></span>
              {status.cuda_build && (
                <span className="text-fg-tertiary">build: <code className="text-fg-secondary font-mono">{status.cuda_build}</code></span>
              )}
              {status.cuda_available && status.device_name && (
                <span className="text-fg-tertiary">GPU: <code className="text-fg-secondary font-mono">{status.device_name}</code></span>
              )}
            </div>
            <div className="flex gap-4 flex-wrap">
              <span className="text-fg-tertiary">
                NVIDIA 驱动:{' '}
                <code className="text-fg-secondary font-mono">
                  {status.cuda_detect.driver_version ?? '未检测到'}
                </code>
              </span>
              {status.cuda_detect.gpu_name && !status.cuda_available && (
                <span className="text-fg-tertiary">
                  系统 GPU:{' '}
                  <code className="text-fg-secondary font-mono">{status.cuda_detect.gpu_name}</code>
                </span>
              )}
            </div>
          </div>

          {/* 误装：CPU torch + 有 GPU */}
          {status.is_cpu_with_gpu && (
            <div className="rounded-sm border border-err bg-err-soft px-2 py-1.5 text-err text-xs">
              检测到 NVIDIA GPU 但当前装的是 CPU 版 torch —— 训练 / 出图会跑在 CPU 上，单步常需数十秒。
              点下方按钮一键重装为 <code className="font-mono">{status.recommended_cu_tag}</code> 版。
            </div>
          )}

          {/* CUDA build 但运行时不可用：驱动 / WSL 问题 */}
          {status.is_cuda_build_unavailable && (
            <div className="rounded-sm border border-warn bg-warn-soft px-2 py-1.5 text-warn text-xs">
              torch 是 CUDA 版但 <code className="font-mono">torch.cuda.is_available()=False</code>。
              这通常不是 pip 能修的（驱动版本过低 / WSL 缺 CUDA 支持 / 容器无 GPU 直通）。
              请检查 NVIDIA 驱动与 CUDA Toolkit。
            </div>
          )}

          {/* 操作按钮 */}
          <div className="flex gap-1.5 items-center flex-wrap">
            <button
              onClick={() => void reinstall('auto')}
              disabled={busy || !status.cuda_detect.available}
              className={status.is_cpu_with_gpu ? 'btn btn-primary btn-sm' : 'btn btn-secondary btn-sm'}
              title={status.cuda_detect.available
                ? `自动选择：${status.recommended_cu_tag}`
                : '没检测到 NVIDIA 驱动，无法装 CUDA 版'}
            >
              {busy ? '安装中...' : status.is_cpu_with_gpu
                ? `⤓ 重装为 CUDA 版（${status.recommended_cu_tag}）`
                : `↻ 重装（auto: ${status.recommended_cu_tag}）`}
            </button>
            <button onClick={() => void refresh()} disabled={busy}
              className="px-2 py-0.5 text-fg-tertiary bg-transparent border-none cursor-pointer rounded-sm">↻</button>
            <button type="button" onClick={() => setAdvancedOpen(!advancedOpen)}
              className="btn btn-ghost btn-sm text-xs text-fg-tertiary ml-auto">
              {advancedOpen ? '▾' : '▸'} 高级（手动选 CUDA 版本）
            </button>
          </div>

          {/* 手动选版本 */}
          {advancedOpen && (
            <div className="flex flex-col gap-1.5 pt-2 border-t border-subtle text-xs">
              <p className="text-fg-tertiary m-0">
                按你的驱动 / 偏好选；通常用 auto 即可。
                选错版本会在 import 时报错，需重新选。
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
                        ? '装 CPU 版（极慢，仅诊断 / 无 GPU 用）'
                        : `装 ${tag} 版（PyTorch index: /whl/${tag}）`
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
    const msg = url
      ? '将 pip install 该 wheel，装包需要几分钟。\n装完后必须重启 Studio 才能生效。继续？'
      : '将自动从 GitHub Releases 选择最匹配的 flash_attn wheel 并安装。\n装包需要几分钟，装完后必须重启 Studio。继续？'
    if (!confirm(msg)) return
    setBusy(true)
    try {
      const result = await api.installFlashAttn(url)
      toast(`flash_attn==${result.version ?? '?'} 安装成功，请重启 Studio`, 'success')
      await refresh()
    } catch (e) {
      toast(`安装失败: ${e}`, 'error')
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
    ? '加载失败'
    : !status
      ? '加载中...'
      : status.installed
        ? `已安装 v${status.version ?? '?'}`
        : '未安装'
  const statusOk = status?.installed && !error

  return (
    <details id="flash-attn" open={!!hasIssue} className="rounded-md border border-subtle bg-surface group scroll-mt-24">
      <summary className="cursor-pointer p-4 list-none flex items-center gap-2">
        <span className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3">▸</span>
        <h2 className="text-sm font-semibold text-fg-primary m-0">Flash Attention</h2>
        <span className="text-xs text-fg-tertiary">训练加速（可选）</span>
        <span className={`ml-auto text-xs font-mono ${statusOk ? 'text-ok' : 'text-warn'}`}>{statusLabel}</span>
      </summary>

      <div className="px-4 pb-4 flex flex-col gap-3">
        {error && <div className="text-err text-xs font-mono">{error}</div>}
        {!error && !status && <div className="text-xs text-fg-tertiary">加载状态...</div>}

        {status && env && (<>
          {/* 环境信息 */}
          <div className="rounded-sm border border-subtle bg-sunken p-2 flex flex-col gap-1 text-xs">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-fg-tertiary shrink-0">flash_attn:</span>
              <code className="font-mono text-fg-primary">
                {status.installed ? `v${status.version ?? '?'}` : '（未安装）'}
              </code>
              {status.installed && <StatusLabel bg="bg-ok-soft" fg="text-ok" text="已安装" />}
            </div>
            <div className="flex gap-4 flex-wrap">
              <span className="text-fg-tertiary">Python: <code className="text-fg-secondary font-mono">{env.python_tag}</code></span>
              <span className="text-fg-tertiary">CUDA: <code className="text-fg-secondary font-mono">{env.cuda_tag ?? '未检测到'}</code></span>
              <span className="text-fg-tertiary">PyTorch: <code className="text-fg-secondary font-mono">{env.torch_tag ?? '未检测到'}</code></span>
              <span className="text-fg-tertiary">平台: <code className="text-fg-secondary font-mono">{env.platform ?? '不支持'}</code></span>
            </div>
          </div>

          {/* GitHub API 失败 */}
          {fetchError && (
            <div className="rounded-sm border border-err bg-err-soft px-2 py-1.5 text-err text-xs">
              GitHub API 请求失败（国内网络可能不稳定，请刷新重试）：
              <code className="block mt-0.5 break-all">{fetchError}</code>
            </div>
          )}

          {/* 没匹配 wheel */}
          {!canAutoInstall && !fetchError && env.platform && env.torch_tag && (
            <div className="rounded-sm border border-warn bg-warn-soft px-2 py-1.5 text-warn text-xs">
              未找到 {env.python_tag} 的预编译 wheel（当前 Python 版本可能尚无支持）。
              请在下方候选列表手动选择其他版本，或从 GitHub Releases 粘贴 URL。
            </div>
          )}

          {/* 操作按钮 */}
          <div className="flex gap-1.5 items-center flex-wrap">
            <button
              onClick={() => void install(null)}
              disabled={busy || !canAutoInstall}
              className="btn btn-primary btn-sm"
              title={canAutoInstall
                ? `自动选择：${bestCandidate?.name ?? ''}`
                : '无可用 wheel，请手动选择'}
            >
              {busy ? '安装中...' : status.installed ? '↻ 重装（自动匹配）' : '⤓ 自动匹配安装'}
            </button>
            <button onClick={() => void refresh()} disabled={busy}
              className="px-2 py-0.5 text-fg-tertiary bg-transparent border-none cursor-pointer rounded-sm">↻</button>
            <button type="button" onClick={() => setCandidatesOpen(!candidatesOpen)}
              className="btn btn-ghost btn-sm text-xs text-fg-tertiary ml-auto">
              {candidatesOpen ? '▾' : '▸'} 候选 wheel（{usable.length} 可用）
            </button>
          </div>

          {/* 候选列表 + 手动 URL */}
          {candidatesOpen && (
            <div className="flex flex-col gap-2 pt-2 border-t border-subtle">
              {candidates.length === 0 ? (
                <p className="text-xs text-fg-tertiary m-0">查询失败或无匹配（检查网络连接）</p>
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
                        title={c.usable ? '安装此 wheel' : 'Python ABI 不兼容，强制安装可能失败'}
                      >
                        {c.usable ? '⤓ 安装' : '强制安装'}
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              <div className="flex flex-col gap-1 pt-1 border-t border-subtle">
                <p className="text-xs text-fg-tertiary m-0">手动粘贴 URL：</p>
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
                  >安装</button>
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
    if (!confirm(
      '将 pip install xformers，按当前 torch+cu 选 PyTorch index URL。\n' +
      '装包几分钟到十几分钟，装完后必须重启 Studio。继续？'
    )) return
    setBusy(true)
    try {
      const r = await api.installXformers()
      toast(`xformers==${r.version ?? '?'} 安装成功，请重启 Studio`, 'success')
      await refresh()
    } catch (e) {
      toast(`安装失败: ${e}`, 'error')
    } finally {
      setBusy(false)
    }
  }

  const statusLabel = error
    ? '加载失败'
    : !status
      ? '加载中...'
      : status.installed
        ? `已安装 v${status.version ?? '?'}`
        : '未安装'
  const statusOk = status?.installed && !error
  const hasIssue = !!error

  return (
    <details id="xformers" open={!!hasIssue} className="rounded-md border border-subtle bg-surface group scroll-mt-24">
      <summary className="cursor-pointer p-4 list-none flex items-center gap-2">
        <span className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3">▸</span>
        <h2 className="text-sm font-semibold text-fg-primary m-0">xformers</h2>
        <span className="text-xs text-fg-tertiary">attention 加速（与 Flash Attention 二选一）</span>
        <span className={`ml-auto text-xs font-mono ${statusOk ? 'text-ok' : 'text-warn'}`}>{statusLabel}</span>
      </summary>

      <div className="px-4 pb-4 flex flex-col gap-3">
        {error && <div className="text-err text-xs font-mono">{error}</div>}
        {!error && !status && <div className="text-xs text-fg-tertiary">加载状态...</div>}

        {status && (<>
          <div className="rounded-sm border border-subtle bg-sunken p-2 flex items-center gap-2 text-xs">
            <span className="text-fg-tertiary shrink-0">xformers:</span>
            <code className="font-mono text-fg-primary">
              {status.installed ? `v${status.version ?? '?'}` : '（未安装）'}
            </code>
            {status.installed && <StatusLabel bg="bg-ok-soft" fg="text-ok" text="已安装" />}
          </div>

          <p className="text-2xs text-fg-tertiary m-0 leading-relaxed">
            xformers 与 Flash Attention <strong>互斥</strong>，每个训练/出图任务的{' '}
            <code className="font-mono">attention_backend</code>{' '}
            字段三选一（无 / xformers / flash_attn）。xformers 泛用性更广（支持
            sm_70+），flash_attn 性能更高（sm_80+ Ampere 起）。
          </p>

          <div className="flex gap-2">
            <button
              onClick={() => void install()}
              disabled={busy}
              className="btn btn-primary btn-sm"
            >
              {busy
                ? '安装中...'
                : status.installed
                  ? '重装（自动匹配）'
                  : '安装（自动匹配）'}
            </button>
            <button
              onClick={() => void refresh()}
              disabled={busy}
              className="btn btn-ghost btn-sm"
              title="刷新状态"
            >↻</button>
          </div>

          <p className="text-2xs text-fg-tertiary m-0">
            装失败多数是上游 PyPI / PyTorch index 没出对应 torch+cu 组合的 wheel
            （5090 / 新 GPU 常见）。失败时按钮 toast 会显示 stderr 末尾，可
            根据提示降 torch 版本或等上游覆盖。
          </p>
        </>)}
      </div>
    </details>
  )
}

// ── 中间步预览（节流） ────────────────────────────────────────────────────
//
// TAEFlux 模型 server 启动时后台下载（lifespan startup）；UI 只暴露用户必须
// 控制的「节流 N」一个输入，其他状态/下载/帮助文字全删（用户决策）。

function TaeFluxSection({
  draft, update,
}: {
  draft: Secrets
  update: <S extends Section, K extends keyof Secrets[S]>(
    section: S, key: K, value: Secrets[S][K],
  ) => void
}) {
  const n = draft.generate.preview_every_n_steps
  const enabled = n > 0
  return (
    <SettingsSection id="preview" title="中间步预览">
      <SettingsField
        label="节流（每 N 步推一次预览）"
        desc="0 = 关闭中间步预览；推荐 3-5。模型 server 启动时已后台下载，UI 不需要操作。"
      >
        <div className="flex items-center gap-2">
          <input
            type="number"
            min={0}
            max={50}
            value={n}
            onChange={(e) => update('generate', 'preview_every_n_steps', Number(e.target.value) || 0)}
            className="input"
            style={{ width: 80 }}
          />
          <span className="text-2xs text-fg-tertiary">
            {enabled ? `每 ${n} 步推一张 256px JPEG（~10KB/步）` : '不推预览（0）'}
          </span>
        </div>
      </SettingsField>
    </SettingsSection>
  )
}


// ── Display Section ────────────────────────────────────────────────────────

function DisplaySection() {
  const [theme, setTheme] = useState<Theme>(() => getStoredTheme())
  const [density, setDensity] = useState<Density>(() => getStoredDensity())

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

  const densityLabel = (d: Density): string => {
    if (d === 'tight') return '紧凑'
    if (d === 'loose') return '宽松'
    return '默认'
  }

  const densityDesc = (d: Density): string => {
    if (d === 'tight') return '字号更小，间距更紧，适合小屏或高信息密度'
    if (d === 'loose') return '字号更大，间距更宽，适合阅读舒适'
    return '标准字号与间距'
  }

  return (
    <SettingsSection id="display" title="显示">
      <SettingsField label="主题">
        <div className="flex gap-1">
          {(['light', 'dark'] as Theme[]).map((t) => (
            <button
              key={t}
              onClick={() => handleThemeChange(t)}
              className={`btn btn-sm ${theme === t ? 'btn-primary' : 'btn-secondary'}`}
            >
              {t === 'light' ? '☀ 日间' : '☾ 暗色'}
            </button>
          ))}
        </div>
      </SettingsField>

      <SettingsField label="界面缩放">
        <div className="flex flex-col gap-1.5">
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
          <p className="text-xs text-fg-tertiary m-0">
            {densityDesc(density)}
          </p>
        </div>
      </SettingsField>
    </SettingsSection>
  )
}
