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
import LLMMessagesEditor from '../../../components/LLMMessagesEditor'
import TagsInput from '../../../components/TagsInput'
import { SettingsField } from '../../tools/settings/fields'
import StepShell from '../../../components/StepShell'
import { useToast } from '../../../components/Toast'
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
  const [advOpen, setAdvOpen] = useState(false)

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
          <section className="rounded-md border border-subtle bg-surface px-3 py-2 flex flex-wrap items-center gap-2 shrink-0 text-sm">
            <span className="text-fg-tertiary">tagger</span>
            <select
              value={tagger}
              onChange={(e) => setTagger(e.target.value as TaggerName)}
              className="input text-sm"
              style={{ padding: '3px 8px' }}
            >
              <option value="wd14">WD14（本地 ONNX）</option>
              <option value="cltagger">CLTagger（本地 ONNX）</option>
              <option value="llm">LLM（OpenAI compatible，含 JoyCaption preset）</option>
            </select>
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
                className="text-xs text-accent underline bg-transparent border-none p-0 cursor-pointer"
                title={t('tag.goInstallOnnx')}
              >
                {t('tag.goInstallOnnx')}
              </button>
            )}
            {taggerStatus && !taggerStatus.ok && taggerStatus.msg.includes('需下载模型') && (
              <button
                type="button"
                onClick={() => settingsDrawer.open({ section: tagger === 'cltagger' ? 'cltagger' : 'wd14' })}
                className="text-xs text-accent underline bg-transparent border-none p-0 cursor-pointer"
                title={t('tag.goDownload')}
              >
                {t('tag.goDownload')}
              </button>
            )}

            <span className="text-dim">|</span>
            <span className="text-fg-tertiary" title={t('tag.scopeHint')}>{t('tag.scope')}</span>
            <select
              value={scope}
              onChange={(e) => setScope(e.target.value)}
              disabled={isLive}
              className="input text-sm"
              style={{ padding: '3px 8px' }}
              title={t('tag.scopeHint')}
            >
              <option value="all">{t('tag.scopeAll')}</option>
              {folders.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
              <option value="validation">{t('tag.scopeValidation')}</option>
            </select>

            <span className="text-dim">|</span>
            <span className="text-fg-tertiary" title={t('tag.onExistingHint')}>
              {t('tag.onExisting')}
            </span>
            <select
              value={onExisting}
              onChange={(e) => setOnExisting(e.target.value as 'overwrite' | 'skip' | 'append')}
              disabled={isLive}
              className="input text-sm"
              style={{ padding: '3px 8px' }}
              title={t('tag.onExistingHint')}
            >
              <option value="overwrite">{t('tag.onExistingOverwrite')}</option>
              <option value="skip">{t('tag.onExistingSkip')}</option>
              <option value="append">{t('tag.onExistingAppend')}</option>
            </select>

            <span className="text-dim">|</span>
            <span className="text-fg-tertiary" title={t('tag.triggerWordHint')}>
              {t('tag.triggerWord')}
            </span>
            <input
              type="text"
              value={triggerWord}
              onChange={(e) => setTriggerWord(e.target.value)}
              placeholder={t('tag.triggerWordPlaceholder')}
              disabled={isLive}
              className={`input input-mono text-sm ${
                triggerWord.trim() !== (activeVersion.trigger_word ?? '') ? 'border-warn' : ''
              }`}
              style={{ padding: '3px 8px', width: 180 }}
              title={t('tag.triggerWordHint')}
            />

            <span className="flex-1" />
          </section>

          {tagger === 'wd14' && (
            <Wd14Panel
              form={wd14Form}
              defaults={wd14Defaults}
              onChange={setWd14Form}
              advOpen={advOpen}
              setAdvOpen={setAdvOpen}
              disabled={isLive}
            />
          )}

          {tagger === 'cltagger' && (
            <CLTaggerPanel
              form={cltaggerForm}
              defaults={cltaggerDefaults}
              onChange={setCltaggerForm}
              advOpen={advOpen}
              setAdvOpen={setAdvOpen}
              disabled={isLive}
            />
          )}

          {tagger === 'llm' && (
            <LLMTaggerPanel
              form={llmForm}
              defaults={llmDefaults}
              onChange={setLlmForm}
              advOpen={advOpen}
              setAdvOpen={setAdvOpen}
              disabled={isLive}
            />
          )}

        </div>

        {/* 右栏：预览面板 */}
        <TagPreviewPanel
          tagger={tagger}
          taggerStatus={taggerStatus}
          isLive={isLive}
          taggerOk={taggerStatus?.ok ?? false}
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
  form, defaults, onChange, advOpen, setAdvOpen, disabled,
}: {
  form: Wd14Form | null
  defaults: WD14Config | null
  onChange: (f: Wd14Form) => void
  advOpen: boolean
  setAdvOpen: (b: boolean) => void
  disabled: boolean
}) {
  const { t } = useTranslation()
  const settingsDrawer = useSettingsDrawer()
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
    <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5 flex flex-col gap-2 shrink-0 text-sm">
      <div className="flex items-center gap-2 flex-wrap">
        <PanelDot />
        <span className="caption">{t('tag.wd14Params')}</span>
        <span className="text-xs text-fg-tertiary">
          {t('tag.prefilledFrom')}{' '}
          <button
            type="button"
            onClick={() => settingsDrawer.open({ section: 'wd14' })}
            className="bg-transparent border-none p-0 text-accent cursor-pointer"
            title={t('tag.globalSettings')}
          >
            {t('tag.globalSettings')}
          </button>
          {' · '}{t('tag.effectiveThisRun')}
        </span>
        <span className="flex-1" />
        {dirty && (
          <>
            <span className="badge badge-warn">{t('tag.modified')}</span>
            <button onClick={restore} disabled={disabled} className="btn btn-ghost btn-sm" title={t('tag.restore')}>
              {t('tag.restore')}
            </button>
          </>
        )}
      </div>

      <div className="flex items-center gap-3 flex-wrap">
        <ThresholdInput label="general" value={form.threshold_general} base={defaults.threshold_general} disabled={disabled} onChange={(v) => onChange({ ...form, threshold_general: v })} />
        <ThresholdInput label="character" value={form.threshold_character} base={defaults.threshold_character} disabled={disabled} onChange={(v) => onChange({ ...form, threshold_character: v })} />
        <button type="button" onClick={() => setAdvOpen(!advOpen)} className="btn btn-ghost btn-sm text-xs text-fg-tertiary">
          {advOpen ? '▾' : '▸'} {t('tag.advanced')}
        </button>
      </div>

      {advOpen && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1">
          <LabeledModelSelect
            label="model_id"
            value={form.model_id}
            options={defaults.model_ids}
            disabled={disabled}
            onChange={(v) => onChange({ ...form, model_id: v })}
            modified={form.model_id !== defaults.model_id}
          />
          <TagsInput
            className="md:col-span-2"
            label={t('tag.blacklistLabel')}
            value={form.blacklist_tags}
            placeholder={t('tag.blacklistPlaceholder1')}
            disabled={disabled}
            onChange={(tags) => onChange({ ...form, blacklist_tags: tags })}
            modified={JSON.stringify(form.blacklist_tags) !== JSON.stringify(defaults.blacklist_tags)}
          />
        </div>
      )}
    </section>
  )
}

function CLTaggerPanel({
  form, defaults, onChange, advOpen, setAdvOpen, disabled,
}: {
  form: CLTaggerForm | null
  defaults: CLTaggerConfig | null
  onChange: (f: CLTaggerForm) => void
  advOpen: boolean
  setAdvOpen: (b: boolean) => void
  disabled: boolean
}) {
  const { t } = useTranslation()
  const settingsDrawer = useSettingsDrawer()
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
    form.add_meta_tag !== defaults.add_meta_tag ||
    form.add_model_tag !== defaults.add_model_tag ||
    form.add_rating_tag !== defaults.add_rating_tag ||
    form.add_quality_tag !== defaults.add_quality_tag ||
    JSON.stringify(form.blacklist_tags) !== JSON.stringify(defaults.blacklist_tags)

  const restore = () => onChange(fromCLTaggerConfig(defaults))

  return (
    <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5 flex flex-col gap-2 shrink-0 text-sm">
      <div className="flex items-center gap-2 flex-wrap">
        <PanelDot />
        <span className="caption">{t('tag.cltaggerParams')}</span>
        <span className="text-xs text-fg-tertiary">
          {t('tag.prefilledFrom')}{' '}
          <button
            type="button"
            onClick={() => settingsDrawer.open({ section: 'cltagger' })}
            className="bg-transparent border-none p-0 text-accent cursor-pointer"
            title={t('tag.globalSettings')}
          >
            {t('tag.globalSettings')}
          </button>
          {' · '}{t('tag.effectiveThisRun')}
        </span>
        <span className="flex-1" />
        {dirty && (
          <>
            <span className="badge badge-warn">{t('tag.modified')}</span>
            <button onClick={restore} disabled={disabled} className="btn btn-ghost btn-sm" title={t('tag.restore')}>
              {t('tag.restore')}
            </button>
          </>
        )}
      </div>

      <div className="flex items-center gap-3 flex-wrap">
        <ThresholdInput label="general" value={form.threshold_general} base={defaults.threshold_general} disabled={disabled} onChange={(v) => onChange({ ...form, threshold_general: v })} />
        <ThresholdInput label="character" value={form.threshold_character} base={defaults.threshold_character} disabled={disabled} onChange={(v) => onChange({ ...form, threshold_character: v })} />
        <label className="flex items-center gap-1.5 text-xs text-fg-tertiary">
          <input type="checkbox" checked={form.add_copyright_tag} disabled={disabled} onChange={(e) => onChange({ ...form, add_copyright_tag: e.target.checked })} />
          copyright
        </label>
        <label className="flex items-center gap-1.5 text-xs text-fg-tertiary">
          <input type="checkbox" checked={form.add_meta_tag} disabled={disabled} onChange={(e) => onChange({ ...form, add_meta_tag: e.target.checked })} />
          meta
        </label>
        <label className="flex items-center gap-1.5 text-xs text-fg-tertiary">
          <input type="checkbox" checked={form.add_model_tag} disabled={disabled} onChange={(e) => onChange({ ...form, add_model_tag: e.target.checked })} />
          model
        </label>
        <label className="flex items-center gap-1.5 text-xs text-fg-tertiary">
          <input type="checkbox" checked={form.add_rating_tag} disabled={disabled} onChange={(e) => onChange({ ...form, add_rating_tag: e.target.checked })} />
          rating
        </label>
        <label className="flex items-center gap-1.5 text-xs text-fg-tertiary">
          <input type="checkbox" checked={form.add_quality_tag} disabled={disabled} onChange={(e) => onChange({ ...form, add_quality_tag: e.target.checked })} />
          quality
        </label>
        <button type="button" onClick={() => setAdvOpen(!advOpen)} className="btn btn-ghost btn-sm text-xs text-fg-tertiary">
          {advOpen ? '▾' : '▸'} {t('tag.advanced')}
        </button>
      </div>

      {advOpen && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1">
          <LabeledInput label="model_id" value={form.model_id} disabled={disabled} onChange={(v) => onChange({ ...form, model_id: v })} modified={form.model_id !== defaults.model_id} />
          <LabeledInput label="model_path" value={form.model_path} disabled={disabled} onChange={(v) => onChange({ ...form, model_path: v })} modified={form.model_path !== defaults.model_path} />
          <LabeledInput label="tag_mapping_path" value={form.tag_mapping_path} disabled={disabled} onChange={(v) => onChange({ ...form, tag_mapping_path: v })} modified={form.tag_mapping_path !== defaults.tag_mapping_path} />
          <TagsInput
            className="md:col-span-2"
            label={t('tag.blacklistLabel')}
            value={form.blacklist_tags}
            placeholder={t('tag.blacklistPlaceholder2')}
            disabled={disabled}
            onChange={(tags) => onChange({ ...form, blacklist_tags: tags })}
            modified={JSON.stringify(form.blacklist_tags) !== JSON.stringify(defaults.blacklist_tags)}
          />
        </div>
      )}
    </section>
  )
}

function LLMTaggerPanel({
  form, defaults, onChange, advOpen, setAdvOpen, disabled,
}: {
  form: LLMTaggerForm | null
  defaults: LLMTaggerConfig | null
  onChange: (f: LLMTaggerForm) => void
  advOpen: boolean
  setAdvOpen: (b: boolean) => void
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
    <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5 flex flex-col gap-2 shrink-0 text-sm">
      <div className="flex items-center gap-2 flex-wrap">
        <PanelDot />
        <span className="caption">{t('tag.llmParams')}</span>
        <span className="text-xs text-fg-tertiary">{t('tag.llmSubtitle')}</span>
        <span className="flex-1" />
        {dirty && (
          <>
            <span className="badge badge-warn">{t('tag.modified')}</span>
            <button onClick={restore} disabled={disabled} className="btn btn-ghost btn-sm" title={t('tag.restore')}>
              {t('tag.restore')}
            </button>
          </>
        )}
      </div>

      <SettingsField label="preset">
        <select
          value={form.preset_id}
          onChange={(e) => switchPreset(e.target.value)}
          disabled={disabled}
          className={llmCtl(form.preset_id !== defaults.current_preset)}
        >
          {defaults.presets.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}{p.builtin ? t('tag.builtin') : ''}
            </option>
          ))}
        </select>
      </SettingsField>

      <SettingsField label="base_url">
        <input
          type="text"
          value={form.base_url}
          placeholder="http://localhost:8000/v1"
          disabled={disabled}
          onChange={(e) => onChange({ ...form, base_url: e.target.value })}
          className={llmCtl(form.base_url !== activePreset.base_url)}
        />
      </SettingsField>

      <SettingsField label="model">
        {activePreset.model_ids.length > 0 ? (
          <select
            value={form.model}
            onChange={(e) => onChange({ ...form, model: e.target.value })}
            disabled={disabled}
            className={llmCtl(form.model !== activePreset.model)}
          >
            {!activePreset.model_ids.includes(form.model) && form.model && (
              <option value={form.model}>{form.model}</option>
            )}
            {activePreset.model_ids.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        ) : (
          <input
            type="text"
            value={form.model}
            placeholder={t('tag.modelPlaceholder')}
            disabled={disabled}
            onChange={(e) => onChange({ ...form, model: e.target.value })}
            className={llmCtl(form.model !== activePreset.model)}
          />
        )}
      </SettingsField>

      <SettingsField label="endpoint">
        <select
          value={form.endpoint}
          onChange={(e) => onChange({ ...form, endpoint: e.target.value as LLMPreset['endpoint'] })}
          disabled={disabled}
          className={llmCtl(form.endpoint !== activePreset.endpoint)}
        >
          <option value="chat_completions">Chat Completions</option>
          <option value="responses">Responses</option>
        </select>
      </SettingsField>

      <SettingsField label="output_format">
        <select
          value={form.output_format}
          onChange={(e) => onChange({ ...form, output_format: e.target.value as LLMPreset['output_format'] })}
          disabled={disabled}
          className={llmCtl(form.output_format !== activePreset.output_format)}
        >
          <option value="json">JSON</option>
          <option value="text">Text</option>
        </select>
      </SettingsField>

      <SettingsField
        label="assist_tagger"
        helpTooltip={t('llmWorkspace.assistTaggerHelp').split('%TAGS%').join('{{tags}}')}
      >
        <select
          value={form.assist_tagger}
          onChange={(e) => onChange({ ...form, assist_tagger: e.target.value })}
          disabled={disabled}
          className={llmCtl(form.assist_tagger !== activePreset.assist_tagger)}
        >
          <option value="">Off</option>
          <option value="wd14">WD14</option>
          <option value="cltagger">CLTagger</option>
        </select>
        {assistNeedsTags && (
          <p className="text-warn text-[11px] leading-snug mt-1 mb-0">
            {t('llmWorkspace.assistNeedsTags').split('%TAGS%').join('{{tags}}')}
          </p>
        )}
      </SettingsField>

      <LLMNumberField label="temperature" value={form.temperature} base={activePreset.temperature} step={0.05} min={0} max={2} disabled={disabled} onChange={(v) => onChange({ ...form, temperature: v })} />
      <LLMNumberField label="max_tokens" value={form.max_tokens} base={activePreset.max_tokens} min={64} max={4096} disabled={disabled} onChange={(v) => onChange({ ...form, max_tokens: Math.round(v) })} />
      <LLMNumberField label="concurrency" value={form.concurrency} base={activePreset.concurrency} min={1} max={8} disabled={disabled} onChange={(v) => onChange({ ...form, concurrency: Math.round(v) })} />

      <button type="button" onClick={() => setAdvOpen(!advOpen)} className="btn btn-ghost btn-sm text-xs text-fg-tertiary self-start">
        {advOpen ? '▾' : '▸'} {t('tag.advanced')}
      </button>

      {advOpen && (
        <>
          <SettingsField label="messages">
            <div className="flex flex-col gap-1.5">
              {form.endpoint === 'responses' && (
                <div className="text-[10px] text-warn">{t('tag.responsesWarning')}</div>
              )}
              <LLMMessagesEditor
                messages={form.messages}
                onChange={(msgs) => onChange({ ...form, messages: msgs })}
                disabled={disabled}
              />
            </div>
          </SettingsField>
          <LLMNumberField label="timeout" value={form.timeout} base={activePreset.timeout} min={5} max={600} disabled={disabled} onChange={(v) => onChange({ ...form, timeout: Math.round(v) })} />
          <LLMNumberField label="max_retries" value={form.max_retries} base={activePreset.max_retries} min={1} max={10} disabled={disabled} onChange={(v) => onChange({ ...form, max_retries: Math.round(v) })} />
          <LLMNumberField label="requests_per_second" value={form.requests_per_second} base={activePreset.requests_per_second} min={0} max={60} step={0.1} disabled={disabled} onChange={(v) => onChange({ ...form, requests_per_second: v })} />
          <LLMNumberField label="max_requests_per_minute" value={form.max_requests_per_minute} base={activePreset.max_requests_per_minute} min={0} max={3600} disabled={disabled} onChange={(v) => onChange({ ...form, max_requests_per_minute: Math.round(v) })} />
          <LLMNumberField label="max_side" value={form.max_side} base={activePreset.max_side} min={64} max={4096} disabled={disabled} onChange={(v) => onChange({ ...form, max_side: Math.round(v) })} />
          <LLMNumberField label="jpeg_quality" value={form.jpeg_quality} base={activePreset.jpeg_quality} min={1} max={100} disabled={disabled} onChange={(v) => onChange({ ...form, jpeg_quality: Math.round(v) })} />
          <LLMNumberField label="max_image_mb" value={form.max_image_mb} base={activePreset.max_image_mb} min={0.1} max={25} step={0.1} disabled={disabled} onChange={(v) => onChange({ ...form, max_image_mb: v })} />
        </>
      )}
    </section>
  )
}

// 对齐全局设置页 textInputClass 的控件样式；modified 时整体切换边框色，
// 避免 border-subtle / border-warn 两个颜色类共存时优先级不确定。
function llmCtl(modified: boolean): string {
  return `w-full px-2 py-1 outline-none rounded-sm bg-sunken border text-sm text-fg-primary font-mono focus:border-accent disabled:opacity-60 ${modified ? 'border-warn' : 'border-subtle'}`
}

function LLMNumberField({ label, value, base, min, max, step = 1, disabled, onChange }: {
  label: string; value: number; base: number; min: number; max: number; step?: number
  disabled: boolean; onChange: (v: number) => void
}) {
  return (
    <SettingsField label={label}>
      <input
        type="number" min={min} max={max} step={step} value={value}
        onChange={(e) => { const n = Number(e.target.value); if (!Number.isNaN(n)) onChange(Math.max(min, Math.min(max, n))) }}
        disabled={disabled}
        className={`${llmCtl(value !== base)} max-w-32`}
      />
    </SettingsField>
  )
}

function ThresholdInput({ label, value, base, disabled, onChange }: {
  label: string; value: number; base: number; disabled: boolean; onChange: (v: number) => void
}) {
  const { t } = useTranslation()
  const modified = value !== base
  return (
    <label className="flex items-center gap-1.5">
      <span className="text-fg-tertiary font-mono text-xs">{label}</span>
      <input
        type="number" min={0} max={1} step={0.01} value={value}
        onChange={(e) => { const n = Number(e.target.value); if (!Number.isNaN(n)) onChange(Math.max(0, Math.min(1, n))) }}
        disabled={disabled}
        className={`input input-mono ${modified ? 'border-warn' : ''}`}
        style={{ width: 72 }}
        title={modified ? `${t('tag.globalSettings')}: ${base}` : undefined}
      />
    </label>
  )
}

function LabeledInput({ label, value, placeholder, disabled, onChange, modified, className = '' }: {
  label: string; value: string; placeholder?: string; disabled: boolean
  onChange: (v: string) => void; modified?: boolean; className?: string
}) {
  return (
    <label className={'grid grid-cols-[140px_1fr] items-center gap-2 ' + className}>
      <span className="text-fg-tertiary font-mono text-xs">{label}</span>
      <input
        type="text" value={value} placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className={`input input-mono ${modified ? 'border-warn' : ''}`}
      />
    </label>
  )
}

function LabeledModelSelect({ label, value, options, disabled, onChange, modified }: {
  label: string; value: string; options: string[]; disabled: boolean
  onChange: (v: string) => void; modified?: boolean
}) {
  const { t } = useTranslation()
  const settingsDrawer = useSettingsDrawer()
  const opts = options.includes(value) ? options : [value, ...options]
  return (
    <label className="grid grid-cols-[140px_1fr] items-center gap-2">
      <span className="text-fg-tertiary font-mono text-xs">{label}</span>
      <div className="flex items-center gap-1.5 min-w-0">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className={`input input-mono min-w-0 flex-1 ${modified ? 'border-warn' : ''}`}
        >
          {opts.map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
        <button
          type="button"
          onClick={() => settingsDrawer.open()}
          className="text-xs text-fg-tertiary shrink-0 bg-transparent border-none p-0 cursor-pointer"
          title={t('tag.globalSettings')}
        >
          +
        </button>
      </div>
    </label>
  )
}

function PanelDot() {
  return <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
}

// ---------------------------------------------------------------------------
// 右侧预览面板
// ---------------------------------------------------------------------------

function TagPreviewPanel({
  tagger, taggerStatus, isLive, taggerOk,
}: {
  tagger: string
  taggerStatus: { ok: boolean; msg: string } | null
  isLive: boolean
  taggerOk: boolean
}) {
  const { t } = useTranslation()

  const taggerDesc = tagger === 'wd14'
    ? t('tag.wd14Desc')
    : tagger === 'cltagger'
      ? t('tag.cltaggerDesc')
      : tagger === 'llm'
        ? t('tag.llmDesc')
        : tagger

  return (
    <div className="flex flex-col gap-3 min-w-0">
      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <div className="flex items-center gap-1.5 mb-2">
          <span className={`inline-block w-1.5 h-1.5 rounded-full shrink-0 ${taggerOk ? 'bg-ok' : 'bg-err'}`} />
          <span className="caption">{t('tag.statusTitle')}</span>
        </div>
        <div className="text-xs text-fg-secondary">
          <div className={`font-mono font-medium ${taggerOk ? 'text-ok' : 'text-err'}`}>
            {tagger} {taggerStatus
              ? (taggerOk ? t('tag.statusReady') : t('tag.statusUnavail'))
              : t('tag.statusChecking')}
          </div>
          {!taggerOk && taggerStatus && (
            <div className="mt-1 text-fg-tertiary break-all">{taggerStatus.msg}</div>
          )}
        </div>
      </div>

      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <div className="flex items-center gap-1.5 mb-2">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
          <span className="caption">{t('tag.descTitle')}</span>
        </div>
        <div className="text-xs text-fg-secondary leading-relaxed">{taggerDesc}</div>
      </div>

      {isLive && (
        <div className="rounded-md border border-subtle bg-surface px-3 py-2.5 text-center">
          <div className="badge badge-warn">{t('tag.taggingBadge')}</div>
        </div>
      )}
    </div>
  )
}
