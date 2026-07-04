import { useEffect, useRef, useState } from 'react'
import { Trans, useTranslation } from 'react-i18next'
import {
  api,
  type CLTaggerVariantInfo,
  type LLMPreset,
  type Secrets,
  type SecretsPatch,
  type WandBPreset,
} from '../../api/client'
import { useDialog } from '../../components/Dialog'
import LLMTaggerWorkspace from '../../components/LLMTaggerWorkspace'
import { TagListInput } from '../../components/TagsInput'
import PageHeader from '../../components/PageHeader'
import { useToast } from '../../components/Toast'
import { useSettingsData, type SaveStatus } from '../../lib/SettingsData'
import { useSettingsDrawer } from '../../lib/SettingsDrawer'
import {
  DEFAULT_LLM_PRESETS,
  DEFAULT_WANDB_PRESET,
  EMPTY,
  getStoredTab,
  _makeFallbackPreset,
  MASK,
  SECTION_TO_TAB,
  TAB_LIST,
  TAB_SECTIONS,
  TAB_STORAGE_KEY,
  textInputClass,
  type Section,
  type Tab,
} from './settings/constants'
import { Bool, SectionIndex, SensitiveInput, SettingsField, SettingsInput, SettingsSection } from './settings/fields'
import { CLTaggerModelCard, EvalMetricModelCard, HFEndpointSelect, SourceSelect, WD14ModelCard } from './settings/modelCards'
import {
  DisplaySection,
  FlashAttentionSection,
  IdleTimeoutSection,
  ModelsSection,
  ONNXRuntimeSection,
  PyTorchSection,
  SaveTestImagesSection,
  TaeFluxSection,
  TagDictionarySection,
  TrainingParamsSection,
  UpscalerSection,
  VaePrecisionSection,
  XformersSection,
} from './settings/sections'
import { SystemSection } from './settings/SystemSection'
import WandBWorkspace from './settings/WandBWorkspace'

// 全局保存状态指示（instant-apply 取代旧的顶部保存按钮）。idle 不渲染。
function SaveIndicator({ status }: { status: SaveStatus }) {
  const { t } = useTranslation()
  if (status.state === 'saving') {
    return <span className="text-xs text-fg-tertiary">{t('settings.saveStatus.saving')}</span>
  }
  if (status.state === 'saved') {
    const time = new Date(status.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    // key={status.at}：每次保存重挂载 → 重播 flash 动画，连续保存也能一眼看出"又存了"。
    return (
      <span key={status.at} className="settings-saved-flash text-xs inline-flex items-center gap-1">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
          <path d="M20 6L9 17l-5-5" />
        </svg>
        {t('settings.saveStatus.saved', { time })}
      </span>
    )
  }
  if (status.state === 'error') {
    return <span className="text-xs text-err">{t('settings.saveStatus.error')}</span>
  }
  return null
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
    commitSecrets,
    saveStatus,
    catalog,
    catalogError,
    reloadCatalog,
    downloadBusy,
    startDownload,
    setDownloadSource,
  } = useSettingsData()
  // instant-apply：不再有本地 draft，控件直接读 server（未加载时 EMPTY 占位），
  // 写一律走 commitSecrets 即时持久化。`draft` 别名保留是为了让下方大段表单
  // 代码改动最小。
  const draft = server ?? EMPTY
  const [error, setError] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>(getStoredTab)
  const [llmModelsBusy, setLlmModelsBusy] = useState(false)
  const [llmTestBusy, setLlmTestBusy] = useState(false)
  const { toast } = useToast()
  const { prompt, confirm } = useDialog()
  const drawer = useSettingsDrawer()
  // 右侧 section index 用：sticky nav 的 IntersectionObserver root + 滚动平移容器
  const scrollContainerRef = useRef<HTMLDivElement>(null)

  // 数据层 fetch secrets 失败时把错误透出到本组件 error 状态。
  useEffect(() => { if (secretsError) setError(secretsError) }, [secretsError])

  const switchTab = (next: Tab) => {
    setTab(next)
    try {
      localStorage.setItem(TAB_STORAGE_KEY, next)
    } catch {
      /* ignore localStorage errors */
    }
  }

  // instant-apply：所有改动即时落盘，抽屉关闭无需 dirty 守卫。
  useEffect(() => {
    drawer.registerDirtyGuard(null)
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
    // 敏感字段清空会回传 MASK 哨兵，语义是"未改"——跳过，保持现有契约。
    if (value === MASK) return
    commitSecrets({ [section]: { [key]: value } } as SecretsPatch)
  }


  const selectCLTaggerVariant = (variant: CLTaggerVariantInfo) => {
    commitSecrets({
      cltagger: {
        model_id: variant.model_id,
        model_path: variant.model_path,
        tag_mapping_path: variant.tag_mapping_path,
      },
    } as SecretsPatch)
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

  const deleteCurrentPreset = async () => {
    if (currentPreset.builtin || draft.llm_tagger.presets.length <= 1) return
    if (!(await confirm(t('settings.confirmDeletePreset', { label: currentPreset.label }), { tone: 'danger' }))) return
    const next = draft.llm_tagger.presets.filter((p) => p.id !== currentPreset.id)
    update('llm_tagger', 'presets', next)
    update('llm_tagger', 'current_preset', next[0]?.id ?? 'style_json')
  }

  const resetCurrentPresetToBuiltin = async () => {
    // 删除当前 builtin preset，让 backend validator 在 PUT 后从 defaults 补回
    if (!currentPreset.builtin) return
    if (!(await confirm(t('settings.confirmResetPreset', { label: currentPreset.label }), { tone: 'danger' }))) return
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

  // —— WandB 预设管理（0.18 预设化，复刻 llm_tagger 模式）——
  const currentWandbPreset: WandBPreset =
    draft.wandb.presets.find((p) => p.id === draft.wandb.current_preset)
    ?? draft.wandb.presets[0]
    ?? DEFAULT_WANDB_PRESET

  const updateWandbPreset = <K extends keyof WandBPreset>(field: K, value: WandBPreset[K]) => {
    if (value === MASK) return // SensitiveInput 未编辑回传 MASK = 未改
    const next = draft.wandb.presets.map((p) =>
      p.id === currentWandbPreset.id ? { ...p, [field]: value } : p
    )
    update('wandb', 'presets', next)
  }

  const wandbPresetIdFor = (label: string): string => {
    const slug = label.toLowerCase().replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '') || 'preset'
    const used = new Set(draft.wandb.presets.map((p) => p.id))
    let id = slug
    let idx = 1
    while (used.has(id)) {
      idx += 1
      id = `${slug}_${idx}`
    }
    return id
  }

  const addWandbPreset = async () => {
    const label = await prompt(t('settings.newPresetName'), {
      placeholder: 'team-b',
      validate: (v) => (v.trim() ? null : t('settings.nameRequired')),
    })
    if (!label) return
    const id = wandbPresetIdFor(label)
    const next: WandBPreset = { ...DEFAULT_WANDB_PRESET, id, label: label.trim() }
    update('wandb', 'presets', [...draft.wandb.presets, next])
    update('wandb', 'current_preset', id)
  }

  const duplicateWandbPreset = async () => {
    const label = await prompt(t('settings.newPresetName'), {
      defaultValue: t('settings.presetCopy', { label: currentWandbPreset.label }),
      validate: (v) => (v.trim() ? null : t('settings.nameRequired')),
    })
    if (!label) return
    const id = wandbPresetIdFor(label)
    // api_key 不复制：draft 里是 MASK 掩码，复制过去会被后端当"保持原值"哨兵，
    // 但新 preset 没有原值可保持 —— 语义混乱，明确留空让用户重填。
    const next: WandBPreset = { ...currentWandbPreset, api_key: '', id, label: label.trim() }
    update('wandb', 'presets', [...draft.wandb.presets, next])
    update('wandb', 'current_preset', id)
  }

  const deleteCurrentWandbPreset = async () => {
    if (draft.wandb.presets.length <= 1) return
    if (!(await confirm(t('settings.confirmDeletePreset', { label: currentWandbPreset.label }), { tone: 'danger' }))) return
    const next = draft.wandb.presets.filter((p) => p.id !== currentWandbPreset.id)
    update('wandb', 'presets', next)
    update('wandb', 'current_preset', next[0]?.id ?? 'default')
  }

  // 导出走后端端点：yaml 文件**包含真实 api_key**（draft 里只有掩码拿不到），
  // 用户显式动作，文件自行保管。
  const exportCurrentWandbPreset = () => {
    const a = document.createElement('a')
    a.href = api.wandbPresetExportUrl(currentWandbPreset.id)
    a.download = `wandb-preset-${currentWandbPreset.id}.yaml`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  // 导入走后端端点（yaml/json 上传，后端解析 + 校验 + 落盘 + 切换选中），
  // 用返回的权威 masked snapshot 回写 context。
  const importWandbPreset = async (file: File) => {
    try {
      const r = await api.importWandbPreset(file)
      setServer(r.secrets)
      toast(t('settings.wandbPresetImported', { label: r.label }), 'success')
    } catch (e) {
      toast(`${t('settings.wandbImportInvalid')}: ${e}`, 'error')
    }
  }

  const refreshLLMModels = async () => {
    if (!server) return
    setLlmModelsBusy(true)
    setError(null)
    try {
      // instant-apply：preset 字段已即时落盘，直接用 server 当前值刷新。
      const sourcePreset = server.llm_tagger.presets.find((p) => p.id === currentPreset.id)
        ?? server.llm_tagger.presets[0]
      const result = await api.refreshLLMModels({
        preset_id: sourcePreset.id,
        base_url: sourcePreset.base_url,
        api_key: sourcePreset.api_key,
        timeout: sourcePreset.timeout,
      })
      setServer(result.secrets)
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
        actions={<SaveIndicator status={saveStatus} />}
      />

      <div ref={scrollContainerRef} className="p-6 pb-12 flex-1 overflow-y-auto">
      <div className="grid gap-10 max-w-[1920px]" style={{ gridTemplateColumns: 'minmax(0,1fr) 200px' }}>
      <div className="flex flex-col gap-8 min-w-0">

      {error && (
        <div className="p-3 rounded-md bg-err-soft border border-err text-err text-sm font-mono">
          {error}
        </div>
      )}

      {tab === 'dataset' && (<>
      <SettingsSection id="download-global" title={t('settings.downloadGlobal')}>
        <SettingsField
          label={t('settings.fieldExcludeTags')}
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

        <div className="flex flex-col gap-2 pt-2 border-t border-subtle">
          <SettingsField label={t('settings.fieldParallelWorkers')}>
            <SettingsInput
              type="number" min={1} max={16}
              value={draft.download.parallel_workers}
              onChange={(v) => update('download', 'parallel_workers', Math.max(1, Number(v) || 1))}
              className={`${textInputClass} max-w-32`}
            />
          </SettingsField>
          <SettingsField label={t('settings.fieldApiRate')}>
            <SettingsInput
              type="number" step="0.5" min={0.5} max={10}
              value={draft.download.api_rate_per_sec}
              onChange={(v) => update('download', 'api_rate_per_sec', Math.max(0.5, Number(v) || 0.5))}
              className={`${textInputClass} max-w-32`}
            />
          </SettingsField>
          <SettingsField label={t('settings.fieldCdnRate')}>
            <SettingsInput
              type="number" step="1" min={1} max={20}
              value={draft.download.cdn_rate_per_sec}
              onChange={(v) => update('download', 'cdn_rate_per_sec', Math.max(1, Number(v) || 1))}
              className={`${textInputClass} max-w-32`}
            />
          </SettingsField>
        </div>

        <div className="flex flex-col gap-2 pt-2 border-t border-subtle">
          <SettingsField label={t('settings.fieldSaveTags')}>
            <Bool value={draft.download.save_tags} onChange={(v) => update('download', 'save_tags', v)} />
          </SettingsField>
          <SettingsField label={t('settings.fieldConvertPng')}>
            <Bool value={draft.download.convert_to_png} onChange={(v) => update('download', 'convert_to_png', v)} />
          </SettingsField>
          <SettingsField label={t('settings.fieldRemoveAlpha')}>
            <Bool value={draft.download.remove_alpha_channel} onChange={(v) => update('download', 'remove_alpha_channel', v)} />
          </SettingsField>
        </div>
      </SettingsSection>

      <SettingsSection id="reg" title={t('settings.reg.sectionTitle')}>
        <SettingsField
          label={t('settings.fieldDefaultExcludedTags')}
          desc={t('settings.commaSeparated')}
          helpTooltip={<p>{t('settings.reg.defaultExcludedHelp')}</p>}
        >
          <TagListInput
            value={draft.reg?.default_excluded_tags ?? []}
            onChange={(tags) => update('reg', 'default_excluded_tags', tags)}
            placeholder={t('settings.reg.defaultExcludedPlaceholder')}
            className={textInputClass}
            commitOnBlur
          />
        </SettingsField>
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
        <SourceSelect
          opt={catalog?.download_source_options?.wd14}
          onChange={(s) => void setDownloadSource('wd14', s)}
        />
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
        <div className="grid grid-cols-2 gap-3">
          <SettingsField label={t('settings.fieldThresholdGeneral')}>
            <SettingsInput
              type="number" step="0.01" min={0} max={1}
              value={draft.wd14.threshold_general}
              onChange={(v) => update('wd14', 'threshold_general', Number(v))}
              className={`${textInputClass} max-w-32`}
            />
          </SettingsField>
          <SettingsField label={t('settings.fieldThresholdCharacter')}>
            <SettingsInput
              type="number" step="0.01" min={0} max={1}
              value={draft.wd14.threshold_character}
              onChange={(v) => update('wd14', 'threshold_character', Number(v))}
              className={`${textInputClass} max-w-32`}
            />
          </SettingsField>
        </div>
        <SettingsField label={t('settings.fieldBlacklistTags')} desc={t('settings.commaSeparated')}>
          <TagListInput
            value={draft.wd14.blacklist_tags}
            onChange={(tags) => update('wd14', 'blacklist_tags', tags)}
            className={textInputClass}
            commitOnBlur
          />
        </SettingsField>
        <SettingsField label={t('settings.fieldBatchSize')} desc={t('settings.batchSizeHint')}>
          <SettingsInput
            type="number" min={1} max={64}
            value={draft.wd14.batch_size}
            onChange={(v) => update('wd14', 'batch_size', Math.max(1, Number(v) || 1))}
            className={`${textInputClass} max-w-32`}
          />
        </SettingsField>
      </SettingsSection>

      <SettingsSection id="cltagger" title="CLTagger">
        <SourceSelect
          opt={catalog?.download_source_options?.cltagger}
          onChange={(s) => void setDownloadSource('cltagger', s)}
        />
        <CLTaggerModelCard
          catalog={catalog}
          busy={downloadBusy}
          start={startDownload}
          currentModelPath={draft.cltagger.model_path}
          currentTagMappingPath={draft.cltagger.tag_mapping_path}
          onSelectVariant={selectCLTaggerVariant}
          modelId={draft.cltagger.model_id}
          onModelIdChange={(id) => update('cltagger', 'model_id', id)}
          t={t}
        />
        <div className="grid grid-cols-2 gap-3">
          <SettingsField label={t('settings.fieldThresholdGeneral')}>
            <SettingsInput
              type="number" step="0.01" min={0} max={1}
              value={draft.cltagger.threshold_general}
              onChange={(v) => update('cltagger', 'threshold_general', Number(v))}
              className={`${textInputClass} max-w-32`}
            />
          </SettingsField>
          <SettingsField label={t('settings.fieldThresholdCharacter')}>
            <SettingsInput
              type="number" step="0.01" min={0} max={1}
              value={draft.cltagger.threshold_character}
              onChange={(v) => update('cltagger', 'threshold_character', Number(v))}
              className={`${textInputClass} max-w-32`}
            />
          </SettingsField>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <SettingsField label={t('settings.fieldAddCopyright')}>
            <Bool value={draft.cltagger.add_copyright_tag} onChange={(v) => update('cltagger', 'add_copyright_tag', v)} />
          </SettingsField>
          <SettingsField label={t('settings.fieldAddMeta')}>
            <Bool value={draft.cltagger.add_meta_tag} onChange={(v) => update('cltagger', 'add_meta_tag', v)} />
          </SettingsField>
          <SettingsField label={t('settings.fieldAddModel')}>
            <Bool value={draft.cltagger.add_model_tag} onChange={(v) => update('cltagger', 'add_model_tag', v)} />
          </SettingsField>
          <SettingsField label={t('settings.fieldAddRating')}>
            <Bool value={draft.cltagger.add_rating_tag} onChange={(v) => update('cltagger', 'add_rating_tag', v)} />
          </SettingsField>
          <SettingsField label={t('settings.fieldAddQuality')}>
            <Bool value={draft.cltagger.add_quality_tag} onChange={(v) => update('cltagger', 'add_quality_tag', v)} />
          </SettingsField>
        </div>
        <SettingsField label={t('settings.fieldBlacklistTags')} desc={t('settings.commaSeparated')}>
          <TagListInput
            value={draft.cltagger.blacklist_tags}
            onChange={(tags) => update('cltagger', 'blacklist_tags', tags)}
            className={textInputClass}
            commitOnBlur
          />
        </SettingsField>
        <SettingsField label={t('settings.fieldBatchSize')} desc={t('settings.batchSizeHint')}>
          <SettingsInput
            type="number" min={1} max={64}
            value={draft.cltagger.batch_size}
            onChange={(v) => update('cltagger', 'batch_size', Math.max(1, Number(v) || 1))}
            className={`${textInputClass} max-w-32`}
          />
        </SettingsField>
      </SettingsSection>

      <ONNXRuntimeSection />
      <TagDictionarySection />
      </>)}

      {tab === 'training' && (<>
      <SettingsSection id="queue" title={t('settings.queueSchedule')}>
        <SettingsField
          label={t('settings.lightTasksDuringTrain')}
          helpTooltip={<p>{t('settings.lightTasksDuringTrainHelp')}</p>}
        >
          <Bool value={draft.queue.light_tasks_during_train} onChange={(v) => update('queue', 'light_tasks_during_train', v)} />
        </SettingsField>
      </SettingsSection>

      <TrainingParamsSection />

      <PyTorchSection />

      <FlashAttentionSection />

      <XformersSection />

      <ModelsSection
        catalog={catalog}
        busy={downloadBusy}
        start={startDownload}
        setSource={setDownloadSource}
        reloadCatalog={reloadCatalog}
        catalogError={catalogError}
        t={t}
      />
      </>)}

      {tab === 'monitor' && (<>
      <SettingsSection id="eval-metrics" title={t('settings.evalMetrics')}>
        <p className="text-xs text-fg-tertiary">{t('settings.evalMetricModelsHint')}</p>
        <SourceSelect
          opt={catalog?.download_source_options?.eval}
          onChange={(s) => void setDownloadSource('eval', s)}
        />
        <EvalMetricModelCard
          catalog={catalog} busy={downloadBusy} start={startDownload}
          kind="clip" dlId="eval_clip"
          titleKey="settings.evalClipModel" helpKey="settings.evalClipModelHelp"
          modelId={draft.eval_metrics.clip_model_name}
          onModelIdChange={(id) => update('eval_metrics', 'clip_model_name', id)}
          t={t}
        />
        <EvalMetricModelCard
          catalog={catalog} busy={downloadBusy} start={startDownload}
          kind="dino" dlId="eval_dino"
          titleKey="settings.evalDinoModel" helpKey="settings.evalDinoModelHelp"
          modelId={draft.eval_metrics.dino_model_name}
          onModelIdChange={(id) => update('eval_metrics', 'dino_model_name', id)}
          t={t}
        />
        <EvalMetricModelCard
          catalog={catalog} busy={downloadBusy} start={startDownload}
          kind="ccip" dlId="eval_ccip"
          titleKey="settings.evalCcipModel" helpKey="settings.evalCcipModelHelp"
          modelId={draft.eval_metrics.ccip_model_name}
          onModelIdChange={(id) => update('eval_metrics', 'ccip_model_name', id)}
          t={t}
        />
        <SettingsField
          label={t('settings.evalMetricsEnabled')}
          helpTooltip={<p>{t('settings.evalMetricsEnabledHelp')}</p>}
        >
          <div className="flex flex-col gap-1.5">
            {(catalog?.eval_metric_catalog ?? []).map((m) => {
              const on = (draft.eval_metrics.enabled_metrics ?? []).includes(m.key)
              return (
                <label key={m.key} className="flex items-start gap-2 text-xs cursor-pointer">
                  <input
                    type="checkbox"
                    checked={on}
                    onChange={() => {
                      const cur = draft.eval_metrics.enabled_metrics ?? []
                      update('eval_metrics', 'enabled_metrics',
                        on ? cur.filter((k) => k !== m.key) : [...cur, m.key])
                    }}
                    className="mt-0.5"
                  />
                  <span className="min-w-0">
                    <span className="font-mono text-fg-primary">{m.label}</span>
                    <span className="text-fg-tertiary"> — {m.desc}</span>
                    {m.note ? <span className="text-warn"> · {m.note}</span> : null}
                  </span>
                </label>
              )
            })}
          </div>
        </SettingsField>
        <SettingsField
          label={t('settings.evalBaseline')}
          helpTooltip={<p>{t('settings.evalBaselineHelp')}</p>}
        >
          <label className="flex items-center gap-2 text-xs cursor-pointer">
            <input
              type="checkbox"
              checked={draft.eval_metrics.eval_baseline_enabled ?? true}
              onChange={(e) => update('eval_metrics', 'eval_baseline_enabled', e.target.checked)}
            />
            <span className="text-fg-secondary">{t('settings.evalBaselineToggle')}</span>
          </label>
        </SettingsField>
      </SettingsSection>

      {/* WandBWorkspace 自带 card（同 LLMTaggerWorkspace 骨架：header 标题 +
          enabled 总开关、PresetBar 预设条、单 section 主体）；外层 div 只承担
          id（section index 滚动定位）+ 锚点偏移，与 llm-tagger 的嵌法一致。 */}
      <div id="wandb" className="scroll-mt-24">
        <WandBWorkspace
          title="Weights & Biases"
          config={draft.wandb}
          serverPresets={server?.wandb.presets ?? []}
          currentPreset={currentWandbPreset}
          onToggleEnabled={(v) => update('wandb', 'enabled', v)}
          onSelectPreset={(id) => update('wandb', 'current_preset', id)}
          onUpdatePreset={updateWandbPreset}
          onAddPreset={() => void addWandbPreset()}
          onSaveAs={() => void duplicateWandbPreset()}
          onDeletePreset={() => void deleteCurrentWandbPreset()}
          onExport={exportCurrentWandbPreset}
          onImportFile={(f) => void importWandbPreset(f)}
        />
      </div>
      </>)}

      {tab === 'preprocess' && (
        <UpscalerSection
          catalog={catalog}
          busy={downloadBusy}
          start={startDownload}
          setSource={setDownloadSource}
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
        <VaePrecisionSection draft={draft} update={update} />
        <TaeFluxSection draft={draft} update={update} />
        <SaveTestImagesSection draft={draft} update={update} />
      </>)}

      {tab === 'credentials' && (<>
        <p className="text-xs text-fg-tertiary">{t('settings.credentialsIntro')}</p>
        <SettingsSection id="cred-huggingface" title="HuggingFace">
          <SettingsField label={t('settings.fieldToken')} helpTooltip={<p>{t('settings.hfTokenHelp')}</p>}>
            <SensitiveInput
              value={draft.huggingface.token}
              serverValue={server?.huggingface.token ?? ''}
              onChange={(v) => update('huggingface', 'token', v)}
            />
          </SettingsField>
          <SettingsField label={t('settings.fieldEndpoint')} helpTooltip={<p>{t('settings.hfEndpointHelp')}</p>}>
            <HFEndpointSelect
              value={draft.huggingface.endpoint}
              onChange={(v) => update('huggingface', 'endpoint', v)}
            />
          </SettingsField>
        </SettingsSection>

        <SettingsSection id="cred-modelscope" title="ModelScope">
          <SettingsField
            label={t('settings.fieldToken')}
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
        </SettingsSection>

        <SettingsSection id="cred-gelbooru" title="Gelbooru">
          <SettingsField label={t('settings.fieldUserId')}>
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
          <SettingsField label={t('settings.fieldApiKey')}>
            <SensitiveInput
              value={draft.gelbooru.api_key}
              serverValue={server?.gelbooru.api_key ?? ''}
              onChange={(v) => update('gelbooru', 'api_key', v)}
            />
          </SettingsField>
        </SettingsSection>

        <SettingsSection id="cred-danbooru" title="Danbooru">
          <SettingsField label={t('settings.fieldUsername')}>
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
          <SettingsField label={t('settings.fieldApiKey')}>
            <SensitiveInput
              value={draft.danbooru.api_key}
              serverValue={server?.danbooru.api_key ?? ''}
              onChange={(v) => update('danbooru', 'api_key', v)}
            />
          </SettingsField>
          <SettingsField label={t('settings.fieldAccountType')} desc={t('settings.danbooruAccountTypeHint')}>
            <select
              value={draft.danbooru.account_type}
              onChange={(e) => update('danbooru', 'account_type', e.target.value as 'free' | 'gold' | 'platinum')}
              className={textInputClass}
            >
              <option value="free">{t('settings.accountFree')}</option>
              <option value="gold">{t('settings.accountGold')}</option>
              <option value="platinum">{t('settings.accountPlatinum')}</option>
            </select>
          </SettingsField>
        </SettingsSection>
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
