import type { TFunction } from 'i18next'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useOutletContext } from 'react-router-dom'
import {
  api,
  type Job,
  type CLTaggerConfig,
  type LLMMessage,
  type LLMPreset,
  type LLMTaggerConfig,
  type ProjectDetail,
  type TaggerName,
  type TaggerStatus,
  type Version,
  type WD14Config,
} from '../../../api/client'
import { InfoButton } from '../../../components/InfoButton'
import LLMMessagesEditor from '../../../components/LLMMessagesEditor'
import { TagListInput } from '../../../components/TagsInput'
import StepShell from '../../../components/StepShell'
import { useToast } from '../../../components/Toast'
import { CLTaggerModelCard, SourceSelect, WD14ModelCard } from '../../tools/settings/modelCards'
import { useSettingsData } from '../../../lib/SettingsData'
import { useSettingsDrawer } from '../../../lib/SettingsDrawer'
import { useEventStream } from '../../../lib/useEventStream'
import { useLatestJobReplay } from '../../../lib/useLatestJobReplay'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

type Wd14Form = {
  threshold_general: number
  threshold_character: number
  model_id: string
  blacklist_tags: string[]
}

type CLTaggerForm = {
  threshold_general: number
  threshold_character: number
  model_id: string
  model_path: string
  tag_mapping_path: string
  add_copyright_tag: boolean
  add_artist_tag: boolean
  add_meta_tag: boolean
  add_model_tag: boolean
  add_rating_tag: boolean
  add_quality_tag: boolean
  blacklist_tags: string[]
}

type LLMTaggerForm = {
  preset_id: string
  base_url: string
  model: string
  endpoint: LLMPreset['endpoint']
  messages: LLMMessage[]
  output_format: LLMPreset['output_format']
  assist_tagger: string
  temperature: number
  max_tokens: number
  timeout: number
  max_retries: number
  concurrency: number
  requests_per_second: number
  max_requests_per_minute: number
  max_side: number
  jpeg_quality: number
  max_image_mb: number
}

function fromConfig(cfg: WD14Config): Wd14Form {
  return {
    threshold_general: cfg.threshold_general,
    threshold_character: cfg.threshold_character,
    model_id: cfg.model_id,
    blacklist_tags: cfg.blacklist_tags,
  }
}

function fromCLTaggerConfig(cfg: CLTaggerConfig): CLTaggerForm {
  return {
    threshold_general: cfg.threshold_general,
    threshold_character: cfg.threshold_character,
    model_id: cfg.model_id,
    model_path: cfg.model_path,
    tag_mapping_path: cfg.tag_mapping_path,
    add_copyright_tag: cfg.add_copyright_tag,
    add_artist_tag: cfg.add_artist_tag,
    add_meta_tag: cfg.add_meta_tag,
    add_model_tag: cfg.add_model_tag,
    add_rating_tag: cfg.add_rating_tag,
    add_quality_tag: cfg.add_quality_tag,
    blacklist_tags: cfg.blacklist_tags,
  }
}

function activePresetOf(cfg: LLMTaggerConfig): LLMPreset | null {
  return cfg.presets.find((p) => p.id === cfg.current_preset) ?? cfg.presets[0] ?? null
}

function fromLLMPreset(p: LLMPreset): LLMTaggerForm {
  return {
    preset_id: p.id,
    base_url: p.base_url,
    model: p.model,
    endpoint: p.endpoint,
    messages: p.messages.map((m) => ({ ...m })),
    output_format: p.output_format,
    assist_tagger: p.assist_tagger,
    temperature: p.temperature,
    max_tokens: p.max_tokens,
    timeout: p.timeout,
    max_retries: p.max_retries,
    concurrency: p.concurrency,
    requests_per_second: p.requests_per_second,
    max_requests_per_minute: p.max_requests_per_minute,
    max_side: p.max_side,
    jpeg_quality: p.jpeg_quality,
    max_image_mb: p.max_image_mb,
  }
}

export default function TaggingPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const settingsDrawer = useSettingsDrawer()
  const {
    catalog, downloadBusy, startDownload, setDownloadSource,
    secrets, commitSecrets, reloadCatalog,
  } = useSettingsData()

  const [tagger, setTagger] = useState<TaggerName>('wd14')
  const [taggerStatus, setTaggerStatus] = useState<TaggerStatus | null>(null)
  // 落盘格式跟着产物走（LLM json preset → .json，其余 → .txt），不再由请求指定。
  const [onExisting, setOnExisting] = useState<'overwrite' | 'skip' | 'append'>('overwrite')
  // 触发词：初值从 activeVersion 取（持久化在 version 表）；启动打标时一并提交，
  // 后端会同步落库 + 传给 worker prepend 到每张 caption。
  const [triggerWord, setTriggerWord] = useState<string>('')
  // 打标范围：'all'（默认 train + validation）/ 'validation' / 某个 train 文件夹名。
  // folders 给 dropdown 列 train 子文件夹选项（从 curation 拿）。
  const [scope, setScope] = useState<string>('all')
  const [folders, setFolders] = useState<string[]>([])

  const [wd14Defaults, setWd14Defaults] = useState<WD14Config | null>(null)
  const [wd14Form, setWd14Form] = useState<Wd14Form | null>(null)
  const [cltaggerDefaults, setCltaggerDefaults] = useState<CLTaggerConfig | null>(null)
  const [cltaggerForm, setCltaggerForm] = useState<CLTaggerForm | null>(null)
  const [llmDefaults, setLlmDefaults] = useState<LLMTaggerConfig | null>(null)
  const [llmForm, setLlmForm] = useState<LLMTaggerForm | null>(null)

  const vid = activeVersion?.id ?? null

  const {
    item: job,
    logs,
    setItem: setJob,
    setLogs,
    itemIdRef: jobIdRef,
    refresh: refreshLatestTagJob,
  } = useLatestJobReplay<Job>(vid, (v) =>
    api.getLatestVersionJob(project.id, v, 'tag').then((r) => ({ item: r.job, log: r.log })),
  )

  useEffect(() => {
    void api
      .getSecrets()
      .then((s) => {
        setWd14Defaults(s.wd14)
        setWd14Form(fromConfig(s.wd14))
        setCltaggerDefaults(s.cltagger)
        setCltaggerForm(fromCLTaggerConfig(s.cltagger))
        setLlmDefaults(s.llm_tagger)
        const active = activePresetOf(s.llm_tagger)
        if (active) setLlmForm(fromLLMPreset(active))
      })
      .catch((e) => toast(t('tag.loadDefaultsFailed', { error: e }), 'error'))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    setTaggerStatus(null)
    void api
      .checkTagger(tagger)
      .then(setTaggerStatus)
      .catch((e) =>
        setTaggerStatus({ name: tagger, ok: false, msg: String(e), requires_service: false })
      )
  }, [tagger])

  // 刷新 / 进入页面时回放最近一次打标 job：锁回 id + 回放历史日志。
  useEffect(() => {
    void refreshLatestTagJob()
  }, [refreshLatestTagJob])

  // version 切换时同步 triggerWord 初值（持久化字段，避免回到 "" 让用户以为没保存）
  useEffect(() => {
    setTriggerWord(activeVersion?.trigger_word ?? '')
  }, [activeVersion?.id, activeVersion?.trigger_word])

  // 打标范围 dropdown 的 train 文件夹选项：拿当前版本的 curation folders。
  // 切版本时 scope 复位 'all'（旧版本的文件夹名在新版本可能不存在）。
  useEffect(() => {
    setScope('all')
    if (vid == null) { setFolders([]); return }
    void api
      .getCuration(project.id, vid)
      .then((v) => setFolders(v.folders))
      .catch(() => setFolders([]))
  }, [project.id, vid])

  useEventStream((evt) => {
    const jid = jobIdRef.current
    if (evt.type === 'job_log_appended' && jid && evt.job_id === jid) {
      setLogs((prev) => [...prev, String(evt.text ?? '')])
    } else if (evt.type === 'job_state_changed' && jid && evt.job_id === jid) {
      void api.getJob(jid).then(setJob).catch(() => {})
      if (evt.status === 'done' || evt.status === 'failed') {
        void reload()
      }
    }
  }, { onOpen: () => void refreshLatestTagJob() })

  if (!activeVersion) {
    return <p className="text-fg-tertiary p-6">{t('tag.noVersion')}</p>
  }

  const isLive = job?.status === 'running' || job?.status === 'pending'

  // 各打标器的简介：挪进「打标器」下拉旁的问号 tooltip（不再占右栏一整块）。
  const taggerDesc =
    tagger === 'wd14' ? t('tag.wd14Desc')
      : tagger === 'cltagger' ? t('tag.cltaggerDesc')
        : tagger === 'llm' ? t('tag.llmDesc')
          : ''

  // ── 右栏打标状态面板数据（对齐正则集页 RegStatusPanel）──────────────────
  // stats 由 outlet reload 刷新（打标 job done 时 reload → 拿新的 tagged 数）。
  const stats = activeVersion.stats
  const totalImages = stats?.train_image_count ?? 0
  const taggedImages = stats?.tagged_image_count ?? 0
  // 此轮需要打标：overwrite/append 全量、skip 只未打标。数据限制——后端只有整训练
  // 集已打标数：scope=all 精确；选文件夹能拿总数但拿不到已打标数（skip 退回文件夹
  // 总数）；validation 无计数（显示 —）。随 scope / onExisting 实时变。
  const scopeTotal =
    scope === 'all' ? totalImages
      : scope === 'validation' ? null
        : (stats?.train_folders.find((f) => f.name === scope)?.image_count ?? null)
  const thisRoundNeed =
    scopeTotal == null ? null
      : onExisting === 'skip'
        ? (scope === 'all' ? Math.max(0, totalImages - taggedImages) : scopeTotal)
        : scopeTotal
  // 打标模型（wd14 / cltagger 具体 model id；llm 无本地模型 → null → 该行跳过）
  const modelLabel =
    tagger === 'wd14' ? (wd14Form?.model_id || null)
      : tagger === 'cltagger' ? (cltaggerForm?.model_id || null)
        : null
  // LLM 预设 label（非 llm → null → 该行跳过）
  const presetLabel =
    tagger === 'llm' && llmForm && llmDefaults
      ? (llmDefaults.presets.find((p) => p.id === llmForm.preset_id)?.label ?? llmForm.preset_id)
      : null

  const buildWd14Overrides = (): Record<string, unknown> | undefined => {
    if (!wd14Form || !wd14Defaults) return undefined
    const out: Record<string, unknown> = {}
    if (wd14Form.threshold_general !== wd14Defaults.threshold_general)
      out.threshold_general = wd14Form.threshold_general
    if (wd14Form.threshold_character !== wd14Defaults.threshold_character)
      out.threshold_character = wd14Form.threshold_character
    if (wd14Form.model_id !== wd14Defaults.model_id) out.model_id = wd14Form.model_id
    if (JSON.stringify(wd14Form.blacklist_tags) !== JSON.stringify(wd14Defaults.blacklist_tags))
      out.blacklist_tags = wd14Form.blacklist_tags
    return Object.keys(out).length ? out : undefined
  }

  const buildCLTaggerOverrides = (): Record<string, unknown> | undefined => {
    if (!cltaggerForm || !cltaggerDefaults) return undefined
    const out: Record<string, unknown> = {}
    if (cltaggerForm.threshold_general !== cltaggerDefaults.threshold_general)
      out.threshold_general = cltaggerForm.threshold_general
    if (cltaggerForm.threshold_character !== cltaggerDefaults.threshold_character)
      out.threshold_character = cltaggerForm.threshold_character
    if (cltaggerForm.model_id !== cltaggerDefaults.model_id) out.model_id = cltaggerForm.model_id
    if (cltaggerForm.model_path !== cltaggerDefaults.model_path) out.model_path = cltaggerForm.model_path
    if (cltaggerForm.tag_mapping_path !== cltaggerDefaults.tag_mapping_path)
      out.tag_mapping_path = cltaggerForm.tag_mapping_path
    if (cltaggerForm.add_copyright_tag !== cltaggerDefaults.add_copyright_tag)
      out.add_copyright_tag = cltaggerForm.add_copyright_tag
    if (cltaggerForm.add_artist_tag !== cltaggerDefaults.add_artist_tag)
      out.add_artist_tag = cltaggerForm.add_artist_tag
    if (cltaggerForm.add_meta_tag !== cltaggerDefaults.add_meta_tag)
      out.add_meta_tag = cltaggerForm.add_meta_tag
    if (cltaggerForm.add_model_tag !== cltaggerDefaults.add_model_tag)
      out.add_model_tag = cltaggerForm.add_model_tag
    if (cltaggerForm.add_rating_tag !== cltaggerDefaults.add_rating_tag)
      out.add_rating_tag = cltaggerForm.add_rating_tag
    if (cltaggerForm.add_quality_tag !== cltaggerDefaults.add_quality_tag)
      out.add_quality_tag = cltaggerForm.add_quality_tag
    if (JSON.stringify(cltaggerForm.blacklist_tags) !== JSON.stringify(cltaggerDefaults.blacklist_tags))
      out.blacklist_tags = cltaggerForm.blacklist_tags
    return Object.keys(out).length ? out : undefined
  }

  const buildLLMOverrides = (): Record<string, unknown> | undefined => {
    if (!llmForm || !llmDefaults) return undefined
    const active = llmDefaults.presets.find((p) => p.id === llmForm.preset_id) ?? llmDefaults.presets[0]
    if (!active) return undefined
    const out: Record<string, unknown> = {}
    if (llmForm.preset_id !== llmDefaults.current_preset) out.current_preset = llmForm.preset_id
    const fields: ReadonlyArray<Exclude<keyof LLMTaggerForm, 'preset_id'>> = [
      'base_url', 'model', 'endpoint', 'messages', 'output_format', 'assist_tagger',
      'temperature', 'max_tokens', 'timeout', 'max_retries',
      'concurrency', 'requests_per_second', 'max_requests_per_minute',
      'max_side', 'jpeg_quality', 'max_image_mb',
    ]
    for (const key of fields) {
      const value = llmForm[key]
      const base = active[key]
      if (JSON.stringify(value) !== JSON.stringify(base)) out[key] = value
    }
    return Object.keys(out).length ? out : undefined
  }

  const startTagging = async () => {
    if (!taggerStatus?.ok) {
      toast(t('tag.taggerUnavailable', { tagger, msg: taggerStatus?.msg ?? '' }), 'error')
      return
    }
    try {
      const wd14_overrides = tagger === 'wd14' ? buildWd14Overrides() : undefined
      const cltagger_overrides = tagger === 'cltagger' ? buildCLTaggerOverrides() : undefined
      const llm_overrides = tagger === 'llm' ? buildLLMOverrides() : undefined
      const overrides = wd14_overrides ?? cltagger_overrides ?? llm_overrides
      const trigger = triggerWord.trim()
      const j = await api.startTag(project.id, activeVersion.id, {
        tagger, on_existing: onExisting,
        scope,
        wd14_overrides, cltagger_overrides, llm_overrides,
        // 传 trigger 永远，让 server 决定是否落库（与现有值比较），空串显式清空
        trigger_word: trigger,
      })
      setJob(j)
      setLogs([])
      const note = overrides ? t('tag.taggingEnqueuedOverrides', { n: Object.keys(overrides).length }) : ''
      toast(t('tag.taggingEnqueued', { id: j.id }) + note, 'success')
      // 触发词改了 → 让父级 reload version 状态，下次重渲染拿新的 trigger_word
      if (trigger !== (activeVersion.trigger_word ?? '')) {
        void reload()
      }
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  return (
    <StepShell
      idx={3}
      title={t('steps.tag.title')}
      subtitle={t('steps.tag.subtitle')}
      logSources={[
        job && {
          key: 'tag',
          label: t('logDrawer.tag'),
          status: job.status,
          lines: logs,
          startedAt: job.started_at,
          finishedAt: job.finished_at,
          onCancel: () => {
            void api
              .cancelJob(job.id)
              .then(() => toast(t('tag.cancelToast'), 'success'))
              .catch((e) => toast(String(e), 'error'))
          },
        },
      ]}
      actions={
        /* 样式对齐项目页「新建项目」（btn-primary btn-sm + icon + 文字） */
        <button
          onClick={startTagging}
          disabled={isLive || !taggerStatus?.ok}
          className="btn btn-primary btn-sm"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z" />
          </svg>
          <span>
            {isLive ? t('tag.taggingBtn') : taggerStatus === null ? t('tag.checkingBtn') : t('tag.startBtn')}
          </span>
        </button>
      }
    >
    <div className="flex flex-col h-full gap-3">

      <div className="grid gap-3 flex-1 min-h-0" style={{ gridTemplateColumns: '1.5fr 1fr' }}>

        {/* 左栏：参数区整体滚动；任务日志走 StepShell 的统一抽屉（issue #251） */}
        <div className="flex flex-col gap-3 min-h-0 min-w-0 overflow-y-auto">
          <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5 shrink-0 text-sm">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4">
              <TagField
                label={t('tag.fieldTagger')}
                helpTooltip={taggerDesc}
                help={
                  <span className="inline-flex items-center gap-2 flex-wrap">
                    <span
                      className={
                        taggerStatus
                          ? taggerStatus.ok ? 'badge badge-ok' : 'badge badge-err'
                          : 'badge badge-neutral'
                      }
                      title={taggerStatus?.msg ?? t('tag.checkingBtn')}
                    >
                      {taggerStatus
                        ? taggerStatus.ok
                          ? `${t('tag.statusReady')} ${taggerStatus.msg}`
                          : `${t('tag.statusUnavail')} ${taggerStatus.msg}`
                        : t('tag.statusChecking')}
                    </span>
                    {taggerStatus && !taggerStatus.ok && taggerStatus.msg.includes('未安装 onnxruntime') && (
                      <button
                        type="button"
                        onClick={() => settingsDrawer.open({ section: 'onnxruntime' })}
                        className="text-accent underline bg-transparent border-none p-0 cursor-pointer"
                      >
                        {t('tag.goInstallOnnx')}
                      </button>
                    )}
                    {taggerStatus && !taggerStatus.ok && taggerStatus.msg.includes('需下载模型') && (
                      <button
                        type="button"
                        onClick={() => settingsDrawer.open({ section: tagger === 'cltagger' ? 'cltagger' : 'wd14' })}
                        className="text-accent underline bg-transparent border-none p-0 cursor-pointer"
                      >
                        {t('tag.goDownload')}
                      </button>
                    )}
                  </span>
                }
              >
                <select
                  value={tagger}
                  onChange={(e) => setTagger(e.target.value as TaggerName)}
                  className="input" style={fieldInputStyle}
                >
                  <option value="wd14">WD14（本地 ONNX）</option>
                  <option value="cltagger">CLTagger（本地 ONNX）</option>
                  <option value="llm">LLM（OpenAI compatible，含 JoyCaption preset）</option>
                </select>
              </TagField>

              <TagField label={t('tag.scope')} helpTooltip={t('tag.scopeHint')}>
                <select
                  value={scope}
                  onChange={(e) => setScope(e.target.value)}
                  disabled={isLive}
                  className="input" style={fieldInputStyle}
                >
                  <option value="all">{t('tag.scopeAll')}</option>
                  {folders.map((f) => (
                    <option key={f} value={f}>{f}</option>
                  ))}
                  <option value="validation">{t('tag.scopeValidation')}</option>
                </select>
              </TagField>

              <TagField label={t('tag.onExisting')} helpTooltip={t('tag.onExistingHint')}>
                <select
                  value={onExisting}
                  onChange={(e) => setOnExisting(e.target.value as 'overwrite' | 'skip' | 'append')}
                  disabled={isLive}
                  className="input" style={fieldInputStyle}
                >
                  <option value="overwrite">{t('tag.onExistingOverwrite')}</option>
                  <option value="skip">{t('tag.onExistingSkip')}</option>
                  <option value="append">{t('tag.onExistingAppend')}</option>
                </select>
              </TagField>

              <TagField label={t('tag.triggerWord')} helpTooltip={t('tag.triggerWordHint')}>
                <input
                  type="text"
                  value={triggerWord}
                  onChange={(e) => setTriggerWord(e.target.value)}
                  placeholder={t('tag.triggerWordPlaceholder')}
                  disabled={isLive}
                  className="input input-mono"
                  style={fieldCtlStyle(triggerWord.trim() !== (activeVersion.trigger_word ?? ''))}
                />
              </TagField>
            </div>
          </section>

          {tagger === 'wd14' && (
            <Wd14Panel
              form={wd14Form}
              defaults={wd14Defaults}
              onChange={setWd14Form}
              disabled={isLive}
              downloadCenter={
                wd14Form && (
                  <div className="flex flex-col gap-3">
                    <SourceSelect
                      opt={catalog?.download_source_options?.wd14}
                      onChange={(s) => void setDownloadSource('wd14', s)}
                    />
                    <WD14ModelCard
                      catalog={catalog}
                      busy={downloadBusy}
                      start={startDownload}
                      currentModelId={wd14Form.model_id}
                      onSelectModelId={(id) => setWd14Form({ ...wd14Form, model_id: id })}
                      candidates={secrets?.wd14.model_ids ?? wd14Defaults?.model_ids ?? []}
                      onCandidatesChange={(next) => {
                        commitSecrets({ wd14: { model_ids: next } })
                        void reloadCatalog()
                      }}
                      t={t}
                    />
                  </div>
                )
              }
            />
          )}

          {tagger === 'cltagger' && (
            <CLTaggerPanel
              form={cltaggerForm}
              defaults={cltaggerDefaults}
              onChange={setCltaggerForm}
              disabled={isLive}
              downloadCenter={
                cltaggerForm && (
                  <div className="flex flex-col gap-3">
                    <SourceSelect
                      opt={catalog?.download_source_options?.cltagger}
                      onChange={(s) => void setDownloadSource('cltagger', s)}
                    />
                    <CLTaggerModelCard
                      catalog={catalog}
                      busy={downloadBusy}
                      start={startDownload}
                      currentModelPath={cltaggerForm.model_path}
                      currentTagMappingPath={cltaggerForm.tag_mapping_path}
                      modelId={cltaggerForm.model_id}
                      onSelectVariant={(v) =>
                        setCltaggerForm({
                          ...cltaggerForm,
                          model_id: v.model_id,
                          model_path: v.model_path,
                          tag_mapping_path: v.tag_mapping_path,
                        })
                      }
                      onModelIdChange={(id) => setCltaggerForm({ ...cltaggerForm, model_id: id })}
                      t={t}
                    />
                  </div>
                )
              }
            />
          )}

          {tagger === 'llm' && (
            <LLMTaggerPanel
              form={llmForm}
              defaults={llmDefaults}
              onChange={setLlmForm}
              disabled={isLive}
            />
          )}

        </div>

        {/* 右栏：当前打标状态（进度 + 本次配置摘要 + 此轮需要打标） */}
        <TagStatusPanel
          tagger={tagger}
          totalImages={totalImages}
          taggedImages={taggedImages}
          modelLabel={modelLabel}
          presetLabel={presetLabel}
          triggerWord={triggerWord}
          latestTaggedAt={job?.finished_at ?? null}
          thisRoundNeed={thisRoundNeed}
          isLive={isLive}
        />
      </div>
    </div>
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// WD14 紧凑参数行
// ---------------------------------------------------------------------------

function Wd14Panel({
  form, defaults, onChange, disabled, downloadCenter,
}: {
  form: Wd14Form | null
  defaults: WD14Config | null
  onChange: (f: Wd14Form) => void
  disabled: boolean
  /** 下载中心（下载源 + 模型卡）；替代原 model_id 下拉，放高级参数里。 */
  downloadCenter?: React.ReactNode
}) {
  const { t } = useTranslation()
  if (!form || !defaults) {
    return (
      <section className="rounded-md border border-subtle bg-surface px-3 py-2 text-xs text-fg-tertiary shrink-0">
        {t('tag.wd14Loading')}
      </section>
    )
  }

  const dirty =
    form.threshold_general !== defaults.threshold_general ||
    form.threshold_character !== defaults.threshold_character ||
    form.model_id !== defaults.model_id ||
    JSON.stringify(form.blacklist_tags) !== JSON.stringify(defaults.blacklist_tags)

  const restore = () => onChange(fromConfig(defaults))

  return (
    <>
      <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5 flex flex-col gap-2 shrink-0 text-sm">
        <PanelHeader dirty={dirty} onRestore={restore} disabled={disabled} />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4">
          <TagFieldNumber label={t('settings.fieldThresholdGeneral')} value={form.threshold_general} base={defaults.threshold_general} min={0} max={1} step={0.01} disabled={disabled} onChange={(v) => onChange({ ...form, threshold_general: v })} />
          <TagFieldNumber label={t('settings.fieldThresholdCharacter')} value={form.threshold_character} base={defaults.threshold_character} min={0} max={1} step={0.01} disabled={disabled} onChange={(v) => onChange({ ...form, threshold_character: v })} />
        </div>
      </section>

      <AdvancedSection>
        <div className="flex flex-col gap-3">
          {downloadCenter}
          <TagField label={t('settings.fieldBlacklistTags')}>
            <TagListInput
              value={form.blacklist_tags}
              placeholder={t('tag.blacklistPlaceholder1')}
              disabled={disabled}
              onChange={(tags) => onChange({ ...form, blacklist_tags: tags })}
              className="input input-mono"
              style={fieldCtlStyle(JSON.stringify(form.blacklist_tags) !== JSON.stringify(defaults.blacklist_tags))}
            />
          </TagField>
        </div>
      </AdvancedSection>
    </>
  )
}

// 面板公共 header：小圆点 + 统一标题「tagger 参数」+ dirty 徽章 / 还原。
function PanelHeader({ dirty, onRestore, disabled, subtitle }: {
  dirty: boolean; onRestore: () => void; disabled: boolean; subtitle?: string
}) {
  const { t } = useTranslation()
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <PanelDot />
      <span className="caption">{t('tag.taggerParams')}</span>
      {subtitle && <span className="text-xs text-fg-tertiary">{subtitle}</span>}
      <span className="flex-1" />
      {dirty && (
        <>
          <span className="badge badge-warn">{t('tag.modified')}</span>
          <button onClick={onRestore} disabled={disabled} className="btn btn-ghost btn-sm" title={t('tag.restore')}>
            {t('tag.restore')}
          </button>
        </>
      )}
    </div>
  )
}

// 高级参数：独立折叠卡（对齐训练配置页 SchemaForm 分组 section），默认收起。
function AdvancedSection({ children }: { children: React.ReactNode }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  return (
    <section className="rounded-md border border-subtle bg-surface shrink-0 text-sm">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 text-sm font-semibold text-fg-primary bg-transparent border-none cursor-pointer"
      >
        <span>{t('tag.advanced')}</span>
        <span className="text-fg-tertiary text-xs">{open ? '▾' : '▸'}</span>
      </button>
      {open && <div className="px-3.5 pb-2.5">{children}</div>}
    </section>
  )
}

function CLTaggerPanel({
  form, defaults, onChange, disabled, downloadCenter,
}: {
  form: CLTaggerForm | null
  defaults: CLTaggerConfig | null
  onChange: (f: CLTaggerForm) => void
  disabled: boolean
  /** 下载中心（下载源 + 模型卡，含变体选择）；替代原模型下拉，放高级参数里。 */
  downloadCenter?: React.ReactNode
}) {
  const { t } = useTranslation()
  if (!form || !defaults) {
    return (
      <section className="rounded-md border border-subtle bg-surface px-3 py-2 text-xs text-fg-tertiary shrink-0">
        {t('tag.cltaggerLoading')}
      </section>
    )
  }

  const dirty =
    form.threshold_general !== defaults.threshold_general ||
    form.threshold_character !== defaults.threshold_character ||
    form.model_id !== defaults.model_id ||
    form.model_path !== defaults.model_path ||
    form.tag_mapping_path !== defaults.tag_mapping_path ||
    form.add_copyright_tag !== defaults.add_copyright_tag ||
    form.add_artist_tag !== defaults.add_artist_tag ||
    form.add_meta_tag !== defaults.add_meta_tag ||
    form.add_model_tag !== defaults.add_model_tag ||
    form.add_rating_tag !== defaults.add_rating_tag ||
    form.add_quality_tag !== defaults.add_quality_tag ||
    JSON.stringify(form.blacklist_tags) !== JSON.stringify(defaults.blacklist_tags)

  const restore = () => onChange(fromCLTaggerConfig(defaults))

  return (
    <>
      <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5 flex flex-col gap-2 shrink-0 text-sm">
        <PanelHeader dirty={dirty} onRestore={restore} disabled={disabled} />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4">
          <TagFieldNumber label={t('settings.fieldThresholdGeneral')} value={form.threshold_general} base={defaults.threshold_general} min={0} max={1} step={0.01} disabled={disabled} onChange={(v) => onChange({ ...form, threshold_general: v })} />
          <TagFieldNumber label={t('settings.fieldThresholdCharacter')} value={form.threshold_character} base={defaults.threshold_character} min={0} max={1} step={0.01} disabled={disabled} onChange={(v) => onChange({ ...form, threshold_character: v })} />
        </div>
      </section>

      <AdvancedSection>
        <div className="flex flex-col gap-3">
          {downloadCenter}
          <TagField label={t('tag.cltaggerExtraTags')}>
            <div className="flex items-center gap-4 flex-wrap py-0.5">
              <TagFieldCheckbox label="copyright" checked={form.add_copyright_tag} disabled={disabled} onChange={(v) => onChange({ ...form, add_copyright_tag: v })} />
              <TagFieldCheckbox label="artist" checked={form.add_artist_tag} disabled={disabled} onChange={(v) => onChange({ ...form, add_artist_tag: v })} />
              <TagFieldCheckbox label="meta" checked={form.add_meta_tag} disabled={disabled} onChange={(v) => onChange({ ...form, add_meta_tag: v })} />
              <TagFieldCheckbox label="model" checked={form.add_model_tag} disabled={disabled} onChange={(v) => onChange({ ...form, add_model_tag: v })} />
              <TagFieldCheckbox label="rating" checked={form.add_rating_tag} disabled={disabled} onChange={(v) => onChange({ ...form, add_rating_tag: v })} />
              <TagFieldCheckbox label="quality" checked={form.add_quality_tag} disabled={disabled} onChange={(v) => onChange({ ...form, add_quality_tag: v })} />
            </div>
          </TagField>
          <TagField label={t('settings.fieldBlacklistTags')}>
            <TagListInput
              value={form.blacklist_tags}
              placeholder={t('tag.blacklistPlaceholder2')}
              disabled={disabled}
              onChange={(tags) => onChange({ ...form, blacklist_tags: tags })}
              className="input input-mono"
              style={fieldCtlStyle(JSON.stringify(form.blacklist_tags) !== JSON.stringify(defaults.blacklist_tags))}
            />
          </TagField>
        </div>
      </AdvancedSection>
    </>
  )
}

function LLMTaggerPanel({
  form, defaults, onChange, disabled,
}: {
  form: LLMTaggerForm | null
  defaults: LLMTaggerConfig | null
  onChange: (f: LLMTaggerForm) => void
  disabled: boolean
}) {
  const { t } = useTranslation()
  if (!form || !defaults) {
    return (
      <section className="rounded-md border border-subtle bg-surface px-3 py-2 text-xs text-fg-tertiary shrink-0">
        {t('tag.llmLoading')}
      </section>
    )
  }

  const activePreset = defaults.presets.find((p) => p.id === form.preset_id) ?? defaults.presets[0]
  if (!activePreset) {
    return (
      <section className="rounded-md border border-subtle bg-surface px-3 py-2 text-xs text-err shrink-0">
        {t('tag.llmNoPreset')}
      </section>
    )
  }

  const dirty =
    form.preset_id !== defaults.current_preset ||
    JSON.stringify(form) !== JSON.stringify(fromLLMPreset(activePreset))

  // 与设置页 LLM 工作区同款警示：开了 assist 但提示词（含本次临时编辑）没有
  // {{tags}} 占位符时，预打标不会生效。
  const assistNeedsTags =
    !!form.assist_tagger &&
    !form.messages.some((m) => m.type === 'text' && m.content.includes('{{tags}}'))

  const restore = () => {
    const original = activePresetOf(defaults)
    if (original) onChange(fromLLMPreset(original))
  }

  const switchPreset = (id: string) => {
    const next = defaults.presets.find((p) => p.id === id)
    if (next) onChange(fromLLMPreset(next))
  }

  return (
    <>
    <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5 flex flex-col gap-2 shrink-0 text-sm">
      <PanelHeader dirty={dirty} onRestore={restore} disabled={disabled} subtitle={t('tag.llmSubtitle')} />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4">
        <TagFieldSelect
          label={t('tag.fieldPreset')}
          value={form.preset_id}
          disabled={disabled}
          onChange={switchPreset}
          modified={form.preset_id !== defaults.current_preset}
          className="md:col-span-2"
        >
          {defaults.presets.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}{p.builtin ? t('tag.builtin') : ''}
            </option>
          ))}
        </TagFieldSelect>

        <TagFieldInput label={t('settings.fieldBaseUrl')} value={form.base_url} placeholder="http://localhost:8000/v1" disabled={disabled} onChange={(v) => onChange({ ...form, base_url: v })} modified={form.base_url !== activePreset.base_url} />
        {activePreset.model_ids.length > 0 ? (
          <TagFieldSelect
            label={t('tag.fieldModel')}
            value={form.model}
            disabled={disabled}
            onChange={(v) => onChange({ ...form, model: v })}
            modified={form.model !== activePreset.model}
          >
            {!activePreset.model_ids.includes(form.model) && form.model && (
              <option value={form.model}>{form.model}</option>
            )}
            {activePreset.model_ids.map((m) => <option key={m} value={m}>{m}</option>)}
          </TagFieldSelect>
        ) : (
          <TagFieldInput label={t('tag.fieldModel')} value={form.model} placeholder={t('tag.modelPlaceholder')} disabled={disabled} onChange={(v) => onChange({ ...form, model: v })} modified={form.model !== activePreset.model} />
        )}

        <TagFieldSelect
          label={t('tag.fieldEndpoint')}
          value={form.endpoint}
          disabled={disabled}
          onChange={(v) => onChange({ ...form, endpoint: v as LLMPreset['endpoint'] })}
          modified={form.endpoint !== activePreset.endpoint}
        >
          <option value="chat_completions">Chat Completions</option>
          <option value="responses">Responses</option>
        </TagFieldSelect>

        <TagFieldSelect
          label={t('tag.fieldOutputFormat')}
          value={form.output_format}
          disabled={disabled}
          onChange={(v) => onChange({ ...form, output_format: v as LLMPreset['output_format'] })}
          modified={form.output_format !== activePreset.output_format}
        >
          <option value="json">JSON</option>
          <option value="text">Text</option>
        </TagFieldSelect>

        <TagFieldSelect
          label={t('llmWorkspace.assistTagger')}
          value={form.assist_tagger}
          disabled={disabled}
          onChange={(v) => onChange({ ...form, assist_tagger: v })}
          modified={form.assist_tagger !== activePreset.assist_tagger}
          className="md:col-span-2"
          helpTooltip={t('llmWorkspace.assistTaggerHelp').split('%TAGS%').join('{{tags}}')}
          help={
            assistNeedsTags && (
              <span className="text-warn">
                {t('llmWorkspace.assistNeedsTags').split('%TAGS%').join('{{tags}}')}
              </span>
            )
          }
        >
          <option value="">Off</option>
          <option value="wd14">WD14</option>
          <option value="cltagger">CLTagger</option>
        </TagFieldSelect>

        <TagFieldNumber label={t('tag.fieldTemperature')} value={form.temperature} base={activePreset.temperature} min={0} max={2} step={0.05} disabled={disabled} onChange={(v) => onChange({ ...form, temperature: v })} />
        <TagFieldNumber label={t('tag.fieldMaxTokens')} value={form.max_tokens} base={activePreset.max_tokens} min={64} max={4096} disabled={disabled} onChange={(v) => onChange({ ...form, max_tokens: Math.round(v) })} />
        <TagFieldNumber label={t('tag.fieldConcurrency')} value={form.concurrency} base={activePreset.concurrency} min={1} max={8} disabled={disabled} onChange={(v) => onChange({ ...form, concurrency: Math.round(v) })} />
      </div>
    </section>

    <AdvancedSection>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4">
        <TagField label={t('tag.fieldMessages')} className="md:col-span-2">
          <div className="flex flex-col gap-1.5">
            {form.endpoint === 'responses' && (
              <div className="text-xs text-warn">{t('tag.responsesWarning')}</div>
            )}
            <LLMMessagesEditor
              messages={form.messages}
              onChange={(msgs) => onChange({ ...form, messages: msgs })}
              disabled={disabled}
            />
          </div>
        </TagField>
        <TagFieldNumber label={t('tag.fieldTimeout')} value={form.timeout} base={activePreset.timeout} min={5} max={600} disabled={disabled} onChange={(v) => onChange({ ...form, timeout: Math.round(v) })} />
        <TagFieldNumber label={t('tag.fieldMaxRetries')} value={form.max_retries} base={activePreset.max_retries} min={1} max={10} disabled={disabled} onChange={(v) => onChange({ ...form, max_retries: Math.round(v) })} />
        <TagFieldNumber label={t('tag.fieldRequestsPerSecond')} value={form.requests_per_second} base={activePreset.requests_per_second} min={0} max={60} step={0.1} disabled={disabled} onChange={(v) => onChange({ ...form, requests_per_second: v })} />
        <TagFieldNumber label={t('tag.fieldMaxRequestsPerMinute')} value={form.max_requests_per_minute} base={activePreset.max_requests_per_minute} min={0} max={3600} disabled={disabled} onChange={(v) => onChange({ ...form, max_requests_per_minute: Math.round(v) })} />
        <TagFieldNumber label={t('tag.fieldMaxSide')} value={form.max_side} base={activePreset.max_side} min={64} max={4096} disabled={disabled} onChange={(v) => onChange({ ...form, max_side: Math.round(v) })} />
        <TagFieldNumber label={t('tag.fieldJpegQuality')} value={form.jpeg_quality} base={activePreset.jpeg_quality} min={1} max={100} disabled={disabled} onChange={(v) => onChange({ ...form, jpeg_quality: Math.round(v) })} />
        <TagFieldNumber label={t('tag.fieldMaxImageMb')} value={form.max_image_mb} base={activePreset.max_image_mb} min={0.1} max={25} step={0.1} disabled={disabled} onChange={(v) => onChange({ ...form, max_image_mb: v })} />
      </div>
    </AdvancedSection>
    </>
  )
}

// ---------------------------------------------------------------------------
// 面板字段块：对齐训练配置页 components/Field.tsx 的视觉（label 上 / 控件全宽
// 在下 / 说明文字在控件下方），叠加打标页特有的 modified 语义（本次任务覆盖值
// ≠ 预填值 → 橙色边框 + title 提示原值）。
// ---------------------------------------------------------------------------

// 与 components/Field.tsx 的 inputStyle 同款（更紧凑；背景用 canvas）。
const fieldInputStyle: React.CSSProperties = {
  width: '100%', padding: '5px 10px',
  background: 'var(--bg-canvas)', border: '1px solid var(--border-default)',
  borderRadius: 'var(--r-sm)', fontSize: 'var(--t-sm)',
  color: 'var(--fg-primary)',
}

function fieldCtlStyle(modified?: boolean): React.CSSProperties {
  return modified ? { ...fieldInputStyle, borderColor: 'var(--warn)' } : fieldInputStyle
}

function TagField({ label, labelExtra, helpTooltip, help, className = '', children }: {
  label: string
  /** label 行内后缀（对齐训练配置页 label 旁小字徽章 / 链接，如「全局设置」跳转）。 */
  labelExtra?: React.ReactNode
  /** 静态说明放 ⓘ tooltip（对齐全局设置页 SettingsField.helpTooltip）。 */
  helpTooltip?: React.ReactNode
  /** 控件下方常驻内容：留给警示 / 状态徽章等必须一直可见的信息。 */
  help?: React.ReactNode
  className?: string
  children: React.ReactNode
}) {
  return (
    <div className={`py-1.5 ${className}`}>
      <div className="flex items-center gap-2 text-sm font-medium text-fg-secondary mb-1">
        <span>{label}</span>
        {helpTooltip && <InfoButton>{helpTooltip}</InfoButton>}
        {labelExtra && (
          <span className="text-[11px] font-normal">{labelExtra}</span>
        )}
      </div>
      {children}
      {help && <div className="text-xs text-fg-tertiary mt-1">{help}</div>}
    </div>
  )
}

function TagFieldInput({ label, value, placeholder, disabled, onChange, modified, help, className }: {
  label: string; value: string; placeholder?: string; disabled: boolean
  onChange: (v: string) => void; modified?: boolean; help?: React.ReactNode; className?: string
}) {
  return (
    <TagField label={label} help={help} className={className}>
      <input
        type="text" value={value} placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="input input-mono" style={fieldCtlStyle(modified)}
      />
    </TagField>
  )
}

function TagFieldNumber({ label, value, base, min, max, step = 1, disabled, onChange, help }: {
  label: string; value: number; base: number; min: number; max: number; step?: number
  disabled: boolean; onChange: (v: number) => void; help?: React.ReactNode
}) {
  const { t } = useTranslation()
  const modified = value !== base
  return (
    <TagField label={label} help={help}>
      <input
        type="number" min={min} max={max} step={step} value={value}
        onChange={(e) => { const n = Number(e.target.value); if (!Number.isNaN(n)) onChange(Math.max(min, Math.min(max, n))) }}
        disabled={disabled}
        className="input input-mono" style={fieldCtlStyle(modified)}
        title={modified ? `${t('tag.modified')} · ${base}` : undefined}
      />
    </TagField>
  )
}

function TagFieldSelect({ label, value, disabled, onChange, modified, helpTooltip, help, className, title, children }: {
  label: string; value: string; disabled: boolean
  onChange: (v: string) => void; modified?: boolean
  helpTooltip?: React.ReactNode; help?: React.ReactNode
  className?: string; title?: string; children: React.ReactNode
}) {
  return (
    <TagField label={label} helpTooltip={helpTooltip} help={help} className={className}>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="input input-mono" style={fieldCtlStyle(modified)}
        title={title}
      >
        {children}
      </select>
    </TagField>
  )
}

function TagFieldCheckbox({ label, checked, disabled, onChange }: {
  label: string; checked: boolean; disabled: boolean; onChange: (v: boolean) => void
}) {
  return (
    <label className={`flex items-center gap-1.5 text-sm text-fg-secondary ${disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}`}>
      <input
        type="checkbox" checked={checked} disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        style={{ height: 16, width: 16, borderRadius: 'var(--r-sm)' }}
      />
      {label}
    </label>
  )
}

function PanelDot() {
  return <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
}

// ---------------------------------------------------------------------------
// 右侧打标状态面板（对齐正则集页 RegStatusPanel）：当前版本打标进度 + 本次打标
// 配置摘要 + 此轮需要打标估算。下载中心已挪进各 tagger 的「高级参数」里。
// ---------------------------------------------------------------------------

function TagStatusPanel({
  tagger, totalImages, taggedImages,
  modelLabel, presetLabel, triggerWord,
  latestTaggedAt, thisRoundNeed, isLive,
}: {
  tagger: string
  totalImages: number
  taggedImages: number
  modelLabel: string | null
  presetLabel: string | null
  triggerWord: string
  latestTaggedAt: number | null
  thisRoundNeed: number | null
  isLive: boolean
}) {
  const { t } = useTranslation()
  const methodLabel = tagger === 'wd14' ? 'WD14' : tagger === 'cltagger' ? 'CLTagger' : 'LLM'
  const trigger = triggerWord.trim()
  return (
    <div className="flex flex-col gap-3 min-w-0 overflow-y-auto">
      <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5 flex flex-col gap-2.5">
        <div className="flex items-center gap-1.5 mb-0.5">
          <PanelDot />
          <span className="caption">{t('tag.statusPanelTitle')}</span>
        </div>
        <div className="flex flex-col gap-2">
          <TagStatusRow label={t('tag.statusImages')}>
            <span className="font-mono">
              <span className="text-ok">{taggedImages}</span>
              <span className="text-fg-tertiary text-2xs font-normal ml-1">
                / {totalImages} {t('tag.nImagesShort')}
              </span>
            </span>
          </TagStatusRow>
          <TagStatusRow label={t('tag.statusMethod')}>
            <span className="font-mono">{methodLabel}</span>
          </TagStatusRow>
          {modelLabel && (
            <TagStatusRow label={t('tag.statusModel')}>
              <span className="font-mono break-all">{modelLabel}</span>
            </TagStatusRow>
          )}
          {presetLabel && (
            <TagStatusRow label={t('tag.statusPreset')}>
              <span className="font-mono break-all">{presetLabel}</span>
            </TagStatusRow>
          )}
          {trigger && (
            <TagStatusRow label={t('tag.statusTrigger')}>
              <span className="font-mono break-all">{trigger}</span>
            </TagStatusRow>
          )}
          <TagStatusRow label={t('tag.statusLatest')}>
            <span className="text-fg-secondary text-sm">
              {latestTaggedAt ? formatAgo(latestTaggedAt, t) : '—'}
            </span>
          </TagStatusRow>
        </div>
        <div className="pt-2.5 border-t border-subtle">
          <TagStatusRow label={t('tag.statusThisRound')}>
            <span className="font-mono">
              {thisRoundNeed == null ? (
                <span className="text-fg-tertiary">—</span>
              ) : (
                <>
                  <span className="text-accent">{thisRoundNeed}</span>
                  <span className="text-fg-tertiary text-2xs font-normal ml-1">
                    {t('tag.nImagesShort')}
                  </span>
                </>
              )}
            </span>
          </TagStatusRow>
        </div>
      </section>

      {isLive && (
        <div className="rounded-md border border-subtle bg-surface px-3 py-2.5 text-center">
          <div className="badge badge-warn">{t('tag.taggingBadge')}</div>
        </div>
      )}
    </div>
  )
}

// 状态行：label 左 / value 右对齐（字号对齐任务详情页 OverviewTab）。
function TagStatusRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-sm text-fg-tertiary font-normal shrink-0">{label}</span>
      <span className="text-sm text-fg-primary text-right min-w-0 break-words">{children}</span>
    </div>
  )
}

// 相对时间（对齐正则集页 formatAgo）。
function formatAgo(unix: number, t: TFunction): string {
  const now = Date.now() / 1000
  const dt = now - unix
  if (dt < 60) return t('tag.agoJustNow')
  if (dt < 3600) return t('tag.agoMinutes', { n: Math.floor(dt / 60) })
  if (dt < 86400) return t('tag.agoHours', { n: Math.floor(dt / 3600) })
  return t('tag.agoDays', { n: Math.floor(dt / 86400) })
}
