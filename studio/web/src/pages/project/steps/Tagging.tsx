import { useEffect, useRef, useState } from 'react'
import { Link, useOutletContext } from 'react-router-dom'
import {
  api,
  type Job,
  type CLTaggerConfig,
  type LLMTaggerConfig,
  type ProjectDetail,
  type TaggerName,
  type TaggerStatus,
  type Version,
  type WD14Config,
} from '../../../api/client'
import JobProgress from '../../../components/JobProgress'
import StepShell from '../../../components/StepShell'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

/**
 * WD14 本次任务的参数表单。`null` 占位含义：还没拉到 settings 的全局值。
 * 拉到之后用全局值填充，让用户在打标页直接微调；不会写回 settings。
 */
type Wd14Form = {
  threshold_general: number
  threshold_character: number
  model_id: string
  local_dir: string
  blacklist_tags: string[]
}

type CLTaggerForm = {
  threshold_general: number
  threshold_character: number
  model_id: string
  model_path: string
  tag_mapping_path: string
  local_dir: string
  add_rating_tag: boolean
  add_model_tag: boolean
  blacklist_tags: string[]
}

type LLMTaggerForm = {
  base_url: string
  model: string
  endpoint: LLMTaggerConfig['endpoint']
  prompt_preset: string
  custom_prompt: string
  // 仅 prompt_preset === 'custom' 时生效；写入 llm_overrides._output_format。
  // 非 custom 时由所选 preset 的 output_format 决定（后端读 LLMTaggerConfig.prompt_presets）。
  custom_output_format: 'json' | 'text'
  temperature: number
  max_tokens: number
  timeout: number
  max_retries: number
  max_side: number
  jpeg_quality: number
  max_image_mb: number
}

function fromConfig(cfg: WD14Config): Wd14Form {
  return {
    threshold_general: cfg.threshold_general,
    threshold_character: cfg.threshold_character,
    model_id: cfg.model_id,
    local_dir: cfg.local_dir ?? '',
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
    local_dir: cfg.local_dir ?? '',
    add_rating_tag: cfg.add_rating_tag,
    add_model_tag: cfg.add_model_tag,
    blacklist_tags: cfg.blacklist_tags,
  }
}

function fromLLMTaggerConfig(cfg: LLMTaggerConfig): LLMTaggerForm {
  return {
    base_url: cfg.base_url,
    model: cfg.model,
    endpoint: cfg.endpoint,
    prompt_preset: cfg.prompt_preset,
    custom_prompt: cfg.custom_prompt,
    custom_output_format: 'json',
    temperature: cfg.temperature,
    max_tokens: cfg.max_tokens,
    timeout: cfg.timeout,
    max_retries: cfg.max_retries,
    max_side: cfg.max_side,
    jpeg_quality: cfg.jpeg_quality,
    max_image_mb: cfg.max_image_mb,
  }
}

export default function TaggingPage() {
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()

  const [tagger, setTagger] = useState<TaggerName>('wd14')
  const [taggerStatus, setTaggerStatus] = useState<TaggerStatus | null>(null)
  const [outputFormat, setOutputFormat] = useState<'txt' | 'json'>('txt')

  const [wd14Defaults, setWd14Defaults] = useState<WD14Config | null>(null)
  const [wd14Form, setWd14Form] = useState<Wd14Form | null>(null)
  const [cltaggerDefaults, setCltaggerDefaults] = useState<CLTaggerConfig | null>(null)
  const [cltaggerForm, setCltaggerForm] = useState<CLTaggerForm | null>(null)
  const [llmDefaults, setLlmDefaults] = useState<LLMTaggerConfig | null>(null)
  const [llmForm, setLlmForm] = useState<LLMTaggerForm | null>(null)
  const [advOpen, setAdvOpen] = useState(false)

  const [job, setJob] = useState<Job | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const jobIdRef = useRef<number | null>(null)
  jobIdRef.current = job?.id ?? null

  // 拉一次 settings 的 wd14 默认值；用作预填 + 「还原全局」的基准。
  useEffect(() => {
    void api
      .getSecrets()
      .then((s) => {
        setWd14Defaults(s.wd14)
        setWd14Form(fromConfig(s.wd14))
        setCltaggerDefaults(s.cltagger)
        setCltaggerForm(fromCLTaggerConfig(s.cltagger))
        setLlmDefaults(s.llm_tagger)
        setLlmForm(fromLLMTaggerConfig(s.llm_tagger))
      })
      .catch((e) => toast(`读取 tagger 默认配置失败：${e}`, 'error'))
    // toast 函数引用稳定；只在 mount 时跑一次
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    setTaggerStatus(null)
    void api
      .checkTagger(tagger)
      .then(setTaggerStatus)
      .catch((e) =>
        setTaggerStatus({
          name: tagger,
          ok: false,
          msg: String(e),
          requires_service: false,
        })
      )
  }, [tagger])

  // 页面刷新 / 进入时回放最近一次 tag job：锁回 jid + 回放历史日志，让 SSE 接力
  const vid = activeVersion?.id ?? null
  useEffect(() => {
    if (!vid) return
    void api
      .getLatestVersionJob(project.id, vid, 'tag')
      .then((r) => {
        if (!r.job) return
        setJob(r.job)
        setLogs(r.log ? r.log.split('\n') : [])
      })
      .catch(() => {})
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
  })

  if (!activeVersion) {
    return <p className="text-fg-tertiary p-6">请先选择 / 创建一个版本</p>
  }

  const isLive = job?.status === 'running' || job?.status === 'pending'

  // 仅当 form 与 settings 默认不同的字段进 overrides；空 dict 不发。
  const buildWd14Overrides = (): Record<string, unknown> | undefined => {
    if (!wd14Form || !wd14Defaults) return undefined
    const out: Record<string, unknown> = {}
    if (wd14Form.threshold_general !== wd14Defaults.threshold_general)
      out.threshold_general = wd14Form.threshold_general
    if (wd14Form.threshold_character !== wd14Defaults.threshold_character)
      out.threshold_character = wd14Form.threshold_character
    if (wd14Form.model_id !== wd14Defaults.model_id)
      out.model_id = wd14Form.model_id
    const localDirChanged =
      (wd14Form.local_dir || null) !== (wd14Defaults.local_dir ?? null)
    if (localDirChanged) out.local_dir = wd14Form.local_dir || null
    if (
      JSON.stringify(wd14Form.blacklist_tags) !==
      JSON.stringify(wd14Defaults.blacklist_tags)
    )
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
    if (cltaggerForm.model_id !== cltaggerDefaults.model_id)
      out.model_id = cltaggerForm.model_id
    if (cltaggerForm.model_path !== cltaggerDefaults.model_path)
      out.model_path = cltaggerForm.model_path
    if (cltaggerForm.tag_mapping_path !== cltaggerDefaults.tag_mapping_path)
      out.tag_mapping_path = cltaggerForm.tag_mapping_path
    const localDirChanged =
      (cltaggerForm.local_dir || null) !== (cltaggerDefaults.local_dir ?? null)
    if (localDirChanged) out.local_dir = cltaggerForm.local_dir || null
    if (cltaggerForm.add_rating_tag !== cltaggerDefaults.add_rating_tag)
      out.add_rating_tag = cltaggerForm.add_rating_tag
    if (cltaggerForm.add_model_tag !== cltaggerDefaults.add_model_tag)
      out.add_model_tag = cltaggerForm.add_model_tag
    if (
      JSON.stringify(cltaggerForm.blacklist_tags) !==
      JSON.stringify(cltaggerDefaults.blacklist_tags)
    )
      out.blacklist_tags = cltaggerForm.blacklist_tags
    return Object.keys(out).length ? out : undefined
  }

  const buildLLMOverrides = (): Record<string, unknown> | undefined => {
    if (!llmForm || !llmDefaults) return undefined
    const out: Record<string, unknown> = {}
    for (const key of Object.keys(llmForm) as Array<keyof LLMTaggerForm>) {
      // custom_output_format 不是 LLMTaggerConfig 字段，单独写到 _output_format 里
      if (key === 'custom_output_format') continue
      const value = llmForm[key]
      const base = (llmDefaults as unknown as Record<string, unknown>)[key]
      if (value === '***') continue
      if (JSON.stringify(value) !== JSON.stringify(base)) out[key] = value
    }
    if (llmForm.prompt_preset === 'custom') {
      out._output_format = llmForm.custom_output_format
    }
    return Object.keys(out).length ? out : undefined
  }

  const startTagging = async () => {
    if (!taggerStatus?.ok) {
      toast(`${tagger} 不可用：${taggerStatus?.msg ?? '未知'}`, 'error')
      return
    }
    try {
      const wd14_overrides = tagger === 'wd14' ? buildWd14Overrides() : undefined
      const cltagger_overrides = tagger === 'cltagger' ? buildCLTaggerOverrides() : undefined
      const llm_overrides = tagger === 'llm' ? buildLLMOverrides() : undefined
      const overrides = wd14_overrides ?? cltagger_overrides ?? llm_overrides
      const j = await api.startTag(project.id, activeVersion.id, {
        tagger,
        output_format: outputFormat,
        wd14_overrides,
        cltagger_overrides,
        llm_overrides,
      })
      setJob(j)
      setLogs([])
      const note = overrides
        ? `（含 ${Object.keys(overrides).length} 项覆盖）`
        : ''
      toast(`已入队 #${j.id}${note}`, 'success')
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  return (
    <StepShell
      idx={3}
      title="自动打标"
      subtitle="WD14 / CLTagger 本地推理，或远程 LLM 视觉打标"
      actions={
        <button
          onClick={startTagging}
          disabled={isLive || !taggerStatus?.ok}
          className="btn btn-primary"
        >
          {isLive
            ? '打标中…'
            : taggerStatus === null
              ? '检查中…'
              : '开始打标全部'}
        </button>
      }
    >
    <div className="flex flex-col h-full gap-3">

      {/* 主体两栏：左（tagger 控制 + 模型卡片 + 参数） / 右（预览面板） */}
      <div className="grid gap-3 flex-1 min-h-0" style={{ gridTemplateColumns: '1.5fr 1fr' }}>

        {/* 左栏 */}
        <div className="flex flex-col gap-3 min-h-0 min-w-0 overflow-y-auto">

          {/* tagger / format 控制栏 */}
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
              <option value="llm">LLM（OpenAI compatible）</option>
              <option value="joycaption">JoyCaption（远程 vLLM）</option>
            </select>
            <span
              className={
                taggerStatus
                  ? taggerStatus.ok ? 'badge badge-ok' : 'badge badge-err'
                  : 'badge badge-neutral'
              }
              title={taggerStatus?.msg ?? '检查中...'}
            >
              {taggerStatus
                ? taggerStatus.ok ? `✓ ${taggerStatus.msg}` : `✗ ${taggerStatus.msg}`
                : '检查中...'}
            </span>
            {taggerStatus && !taggerStatus.ok && taggerStatus.msg.includes('需下载模型') && (
              <Link
                to="/tools/settings"
                className="text-xs text-accent underline"
                title="去设置页下载模型"
              >
                去下载 →
              </Link>
            )}

            <span className="text-dim">|</span>
            <span className="text-fg-tertiary">format</span>
            <select
              value={outputFormat}
              onChange={(e) => setOutputFormat(e.target.value as 'txt' | 'json')}
              className="input text-sm"
              style={{ padding: '3px 8px' }}
            >
              <option value="txt">.txt</option>
              <option value="json">.json</option>
            </select>

            <span className="flex-1" />
          </section>

          {/* WD14 本次参数；预填充全局 settings，不写回 */}
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

          {job && (
            <JobProgress
              job={job}
              logs={logs}
              onCancel={async () => {
                try {
                  await api.cancelJob(job.id)
                  toast('已取消', 'success')
                } catch (e) {
                  toast(String(e), 'error')
                }
              }}
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
  form,
  defaults,
  onChange,
  advOpen,
  setAdvOpen,
  disabled,
}: {
  form: Wd14Form | null
  defaults: WD14Config | null
  onChange: (f: Wd14Form) => void
  advOpen: boolean
  setAdvOpen: (b: boolean) => void
  disabled: boolean
}) {
  if (!form || !defaults) {
    return (
      <section className="rounded-md border border-subtle bg-surface px-3 py-2 text-xs text-fg-tertiary shrink-0">
        加载 wd14 默认参数...
      </section>
    )
  }

  const dirty =
    form.threshold_general !== defaults.threshold_general ||
    form.threshold_character !== defaults.threshold_character ||
    form.model_id !== defaults.model_id ||
    (form.local_dir || null) !== (defaults.local_dir ?? null) ||
    JSON.stringify(form.blacklist_tags) !==
      JSON.stringify(defaults.blacklist_tags)

  const restore = () => onChange(fromConfig(defaults))

  return (
    <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5 flex flex-col gap-2 shrink-0 text-sm">
      <div className="flex items-center gap-2 flex-wrap">
        <PanelDot />
        <span className="caption">WD14 参数</span>
        <span className="text-xs text-fg-tertiary">
          预填{' '}
          <Link to="/tools/settings" className="text-accent" title="去设置页编辑全局默认">
            全局设置
          </Link>{' '}
          · 本次有效，不写回
        </span>
        <span className="flex-1" />
        {dirty && (
          <>
            <span className="badge badge-warn">已改</span>
            <button
              onClick={restore}
              disabled={disabled}
              className="btn btn-ghost btn-sm"
              title="还原为全局设置"
            >
              ↻ 还原
            </button>
          </>
        )}
      </div>

      <div className="flex items-center gap-3 flex-wrap">
        <ThresholdInput
          label="general"
          value={form.threshold_general}
          base={defaults.threshold_general}
          disabled={disabled}
          onChange={(v) => onChange({ ...form, threshold_general: v })}
        />
        <ThresholdInput
          label="character"
          value={form.threshold_character}
          base={defaults.threshold_character}
          disabled={disabled}
          onChange={(v) => onChange({ ...form, threshold_character: v })}
        />
        <button
          type="button"
          onClick={() => setAdvOpen(!advOpen)}
          className="btn btn-ghost btn-sm text-xs text-fg-tertiary"
        >
          {advOpen ? '▾' : '▸'} 高级
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
          <LabeledInput
            label="local_dir"
            value={form.local_dir}
            placeholder="留空 = 自动 HF 下载"
            disabled={disabled}
            onChange={(v) => onChange({ ...form, local_dir: v })}
            modified={
              (form.local_dir || null) !== (defaults.local_dir ?? null)
            }
          />
          <LabeledInput
            className="md:col-span-2"
            label="blacklist_tags（逗号分隔）"
            value={form.blacklist_tags.join(', ')}
            placeholder="如 monochrome, comic"
            disabled={disabled}
            onChange={(v) =>
              onChange({
                ...form,
                blacklist_tags: v
                  .split(',')
                  .map((t) => t.trim())
                  .filter(Boolean),
              })
            }
            modified={
              JSON.stringify(form.blacklist_tags) !==
              JSON.stringify(defaults.blacklist_tags)
            }
          />
        </div>
      )}
    </section>
  )
}

function CLTaggerPanel({
  form,
  defaults,
  onChange,
  advOpen,
  setAdvOpen,
  disabled,
}: {
  form: CLTaggerForm | null
  defaults: CLTaggerConfig | null
  onChange: (f: CLTaggerForm) => void
  advOpen: boolean
  setAdvOpen: (b: boolean) => void
  disabled: boolean
}) {
  if (!form || !defaults) {
    return (
      <section className="rounded-md border border-subtle bg-surface px-3 py-2 text-xs text-fg-tertiary shrink-0">
        加载 CLTagger 默认参数...
      </section>
    )
  }

  const dirty =
    form.threshold_general !== defaults.threshold_general ||
    form.threshold_character !== defaults.threshold_character ||
    form.model_id !== defaults.model_id ||
    form.model_path !== defaults.model_path ||
    form.tag_mapping_path !== defaults.tag_mapping_path ||
    (form.local_dir || null) !== (defaults.local_dir ?? null) ||
    form.add_rating_tag !== defaults.add_rating_tag ||
    form.add_model_tag !== defaults.add_model_tag ||
    JSON.stringify(form.blacklist_tags) !==
      JSON.stringify(defaults.blacklist_tags)

  const restore = () => onChange(fromCLTaggerConfig(defaults))

  return (
    <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5 flex flex-col gap-2 shrink-0 text-sm">
      <div className="flex items-center gap-2 flex-wrap">
        <PanelDot />
        <span className="caption">CLTagger 参数</span>
        <span className="text-xs text-fg-tertiary">
          预填{' '}
          <Link to="/tools/settings" className="text-accent" title="去设置页编辑全局默认">
            全局设置
          </Link>{' '}
          · 本次有效，不写回
        </span>
        <span className="flex-1" />
        {dirty && (
          <>
            <span className="badge badge-warn">已改</span>
            <button
              onClick={restore}
              disabled={disabled}
              className="btn btn-ghost btn-sm"
              title="还原为全局设置"
            >
              ↻ 还原
            </button>
          </>
        )}
      </div>

      <div className="flex items-center gap-3 flex-wrap">
        <ThresholdInput
          label="general"
          value={form.threshold_general}
          base={defaults.threshold_general}
          disabled={disabled}
          onChange={(v) => onChange({ ...form, threshold_general: v })}
        />
        <ThresholdInput
          label="character"
          value={form.threshold_character}
          base={defaults.threshold_character}
          disabled={disabled}
          onChange={(v) => onChange({ ...form, threshold_character: v })}
        />
        <label className="flex items-center gap-1.5 text-xs text-fg-tertiary">
          <input
            type="checkbox"
            checked={form.add_rating_tag}
            disabled={disabled}
            onChange={(e) => onChange({ ...form, add_rating_tag: e.target.checked })}
          />
          rating
        </label>
        <label className="flex items-center gap-1.5 text-xs text-fg-tertiary">
          <input
            type="checkbox"
            checked={form.add_model_tag}
            disabled={disabled}
            onChange={(e) => onChange({ ...form, add_model_tag: e.target.checked })}
          />
          model
        </label>
        <button
          type="button"
          onClick={() => setAdvOpen(!advOpen)}
          className="btn btn-ghost btn-sm text-xs text-fg-tertiary"
        >
          {advOpen ? '▾' : '▸'} 高级
        </button>
      </div>

      {advOpen && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1">
          <LabeledInput
            label="model_id"
            value={form.model_id}
            disabled={disabled}
            onChange={(v) => onChange({ ...form, model_id: v })}
            modified={form.model_id !== defaults.model_id}
          />
          <LabeledInput
            label="local_dir"
            value={form.local_dir}
            placeholder="留空 = 自动 HF 下载"
            disabled={disabled}
            onChange={(v) => onChange({ ...form, local_dir: v })}
            modified={(form.local_dir || null) !== (defaults.local_dir ?? null)}
          />
          <LabeledInput
            label="model_path"
            value={form.model_path}
            disabled={disabled}
            onChange={(v) => onChange({ ...form, model_path: v })}
            modified={form.model_path !== defaults.model_path}
          />
          <LabeledInput
            label="tag_mapping_path"
            value={form.tag_mapping_path}
            disabled={disabled}
            onChange={(v) => onChange({ ...form, tag_mapping_path: v })}
            modified={form.tag_mapping_path !== defaults.tag_mapping_path}
          />
          <LabeledInput
            className="md:col-span-2"
            label="blacklist_tags（逗号分隔）"
            value={form.blacklist_tags.join(', ')}
            placeholder="如 low quality, signature"
            disabled={disabled}
            onChange={(v) =>
              onChange({
                ...form,
                blacklist_tags: v
                  .split(',')
                  .map((t) => t.trim())
                  .filter(Boolean),
              })
            }
            modified={
              JSON.stringify(form.blacklist_tags) !==
              JSON.stringify(defaults.blacklist_tags)
            }
          />
        </div>
      )}
    </section>
  )
}

function LLMTaggerPanel({
  form,
  defaults,
  onChange,
  advOpen,
  setAdvOpen,
  disabled,
}: {
  form: LLMTaggerForm | null
  defaults: LLMTaggerConfig | null
  onChange: (f: LLMTaggerForm) => void
  advOpen: boolean
  setAdvOpen: (b: boolean) => void
  disabled: boolean
}) {
  if (!form || !defaults) {
    return (
      <section className="rounded-md border border-subtle bg-surface px-3 py-2 text-xs text-fg-tertiary shrink-0">
        加载 LLM 默认参数...
      </section>
    )
  }

  const dirty = JSON.stringify(form) !== JSON.stringify(fromLLMTaggerConfig(defaults))
  const restore = () => onChange(fromLLMTaggerConfig(defaults))

  return (
    <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5 flex flex-col gap-2 shrink-0 text-sm">
      <div className="flex items-center gap-2 flex-wrap">
        <PanelDot />
        <span className="caption">LLM 参数</span>
        <span className="text-xs text-fg-tertiary">
          OpenAI compatible · JSON 解析后可写 .json 或 .txt
        </span>
        <span className="flex-1" />
        {dirty && (
          <>
            <span className="badge badge-warn">已改</span>
            <button
              onClick={restore}
              disabled={disabled}
              className="btn btn-ghost btn-sm"
              title="还原为全局设置"
            >
              ↻ 还原
            </button>
          </>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        <LabeledInput
          label="base_url"
          value={form.base_url}
          placeholder="http://localhost:8000/v1"
          disabled={disabled}
          onChange={(v) => onChange({ ...form, base_url: v })}
          modified={form.base_url !== defaults.base_url}
        />
        {defaults.model_ids.length > 0 ? (
          <label className="grid grid-cols-[140px_1fr] items-center gap-2">
            <span className="text-fg-tertiary font-mono text-xs">model</span>
            <select
              value={form.model}
              onChange={(e) => onChange({ ...form, model: e.target.value })}
              disabled={disabled}
              className={`input input-mono ${form.model !== defaults.model ? 'border-warn' : ''}`}
            >
              {!defaults.model_ids.includes(form.model) && form.model && (
                <option value={form.model}>{form.model}</option>
              )}
              {defaults.model_ids.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </label>
        ) : (
          <LabeledInput
            label="model"
            value={form.model}
            placeholder="模型名"
            disabled={disabled}
            onChange={(v) => onChange({ ...form, model: v })}
            modified={form.model !== defaults.model}
          />
        )}
        <label className="grid grid-cols-[140px_1fr] items-center gap-2">
          <span className="text-fg-tertiary font-mono text-xs">endpoint</span>
          <select
            value={form.endpoint}
            onChange={(e) => onChange({ ...form, endpoint: e.target.value as LLMTaggerConfig['endpoint'] })}
            disabled={disabled}
            className={`input input-mono ${form.endpoint !== defaults.endpoint ? 'border-warn' : ''}`}
          >
            <option value="chat_completions">Chat Completions</option>
            <option value="responses">Responses</option>
          </select>
        </label>
        <label className="grid grid-cols-[140px_1fr] items-center gap-2">
          <span className="text-fg-tertiary font-mono text-xs">prompt_preset</span>
          <select
            value={form.prompt_preset}
            onChange={(e) => onChange({ ...form, prompt_preset: e.target.value })}
            disabled={disabled}
            className={`input input-mono ${form.prompt_preset !== defaults.prompt_preset ? 'border-warn' : ''}`}
          >
            {defaults.prompt_presets.map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
            <option value="custom">custom（旧版单独提示词）</option>
          </select>
        </label>
      </div>

      {form.prompt_preset === 'custom' && (
        <>
          <label className="grid grid-cols-[140px_1fr] items-center gap-2">
            <span className="text-fg-tertiary font-mono text-xs">output_format</span>
            <select
              value={form.custom_output_format}
              onChange={(e) => onChange({ ...form, custom_output_format: e.target.value as 'json' | 'text' })}
              disabled={disabled}
              className="input input-mono"
            >
              <option value="json">JSON caption（按字段标准化后写入 caption）</option>
              <option value="text">Text caption（自然语言原文写入 .txt / .json.tags）</option>
            </select>
          </label>
          <label className="grid grid-cols-[140px_1fr] items-start gap-2">
            <span className="text-fg-tertiary font-mono text-xs pt-1">custom_prompt</span>
            <textarea
              value={form.custom_prompt}
              onChange={(e) => onChange({ ...form, custom_prompt: e.target.value })}
              disabled={disabled}
              className={`input input-mono min-h-28 ${form.custom_prompt !== defaults.custom_prompt ? 'border-warn' : ''}`}
            />
          </label>
        </>
      )}

      <div className="flex items-center gap-3 flex-wrap">
        <LLMNumberInput label="temperature" value={form.temperature} base={defaults.temperature} step={0.05} min={0} max={2} disabled={disabled} onChange={(v) => onChange({ ...form, temperature: v })} />
        <LLMNumberInput label="max_tokens" value={form.max_tokens} base={defaults.max_tokens} step={1} min={64} max={4096} disabled={disabled} onChange={(v) => onChange({ ...form, max_tokens: Math.round(v) })} />
        <button
          type="button"
          onClick={() => setAdvOpen(!advOpen)}
          className="btn btn-ghost btn-sm text-xs text-fg-tertiary"
        >
          {advOpen ? '▾' : '▸'} 高级
        </button>
      </div>

      {advOpen && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1">
          <LLMLabeledNumber label="timeout" value={form.timeout} base={defaults.timeout} min={5} max={600} disabled={disabled} onChange={(v) => onChange({ ...form, timeout: Math.round(v) })} />
          <LLMLabeledNumber label="max_retries" value={form.max_retries} base={defaults.max_retries} min={1} max={10} disabled={disabled} onChange={(v) => onChange({ ...form, max_retries: Math.round(v) })} />
          <LLMLabeledNumber label="max_side" value={form.max_side} base={defaults.max_side} min={64} max={4096} disabled={disabled} onChange={(v) => onChange({ ...form, max_side: Math.round(v) })} />
          <LLMLabeledNumber label="jpeg_quality" value={form.jpeg_quality} base={defaults.jpeg_quality} min={1} max={100} disabled={disabled} onChange={(v) => onChange({ ...form, jpeg_quality: Math.round(v) })} />
          <LLMLabeledNumber label="max_image_mb" value={form.max_image_mb} base={defaults.max_image_mb} min={0.1} max={25} step={0.1} disabled={disabled} onChange={(v) => onChange({ ...form, max_image_mb: v })} />
        </div>
      )}
    </section>
  )
}

function LLMNumberInput({
  label,
  value,
  base,
  min,
  max,
  step,
  disabled,
  onChange,
}: {
  label: string
  value: number
  base: number
  min: number
  max: number
  step: number
  disabled: boolean
  onChange: (v: number) => void
}) {
  return (
    <label className="flex items-center gap-1.5">
      <span className="text-fg-tertiary font-mono text-xs">{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => {
          const n = Number(e.target.value)
          if (!Number.isNaN(n)) onChange(Math.max(min, Math.min(max, n)))
        }}
        disabled={disabled}
        className={`input input-mono ${value !== base ? 'border-warn' : ''}`}
        style={{ width: 88 }}
      />
    </label>
  )
}

function LLMLabeledNumber({
  label,
  value,
  base,
  min,
  max,
  step = 1,
  disabled,
  onChange,
}: {
  label: string
  value: number
  base: number
  min: number
  max: number
  step?: number
  disabled: boolean
  onChange: (v: number) => void
}) {
  return (
    <label className="grid grid-cols-[140px_1fr] items-center gap-2">
      <span className="text-fg-tertiary font-mono text-xs">{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => {
          const n = Number(e.target.value)
          if (!Number.isNaN(n)) onChange(Math.max(min, Math.min(max, n)))
        }}
        disabled={disabled}
        className={`input input-mono ${value !== base ? 'border-warn' : ''}`}
      />
    </label>
  )
}

function ThresholdInput({
  label,
  value,
  base,
  disabled,
  onChange,
}: {
  label: string
  value: number
  base: number
  disabled: boolean
  onChange: (v: number) => void
}) {
  const modified = value !== base
  return (
    <label className="flex items-center gap-1.5">
      <span className="text-fg-tertiary font-mono text-xs">{label}</span>
      <input
        type="number"
        min={0}
        max={1}
        step={0.01}
        value={value}
        onChange={(e) => {
          const n = Number(e.target.value)
          if (!Number.isNaN(n)) onChange(Math.max(0, Math.min(1, n)))
        }}
        disabled={disabled}
        className={`input input-mono ${modified ? 'border-warn' : ''}`}
        style={{ width: 72 }}
        title={modified ? `全局 ${base}` : undefined}
      />
    </label>
  )
}

function LabeledInput({
  label,
  value,
  placeholder,
  disabled,
  onChange,
  modified,
  className = '',
}: {
  label: string
  value: string
  placeholder?: string
  disabled: boolean
  onChange: (v: string) => void
  modified?: boolean
  className?: string
}) {
  return (
    <label className={'grid grid-cols-[140px_1fr] items-center gap-2 ' + className}>
      <span className="text-fg-tertiary font-mono text-xs">{label}</span>
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className={`input input-mono ${modified ? 'border-warn' : ''}`}
      />
    </label>
  )
}

function LabeledModelSelect({
  label,
  value,
  options,
  disabled,
  onChange,
  modified,
}: {
  label: string
  value: string
  options: string[]
  disabled: boolean
  onChange: (v: string) => void
  modified?: boolean
}) {
  // 当前选中的 model_id 万一不在 options 里（设置同步前的边界），仍显示它，
  // 避免 dropdown 视觉上回退到 options[0]。
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
        <Link
          to="/tools/settings"
          className="text-xs text-fg-tertiary shrink-0"
          title="去设置编辑候选模型列表"
        >
          + 候选
        </Link>
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
  tagger,
  taggerStatus,
  isLive,
  taggerOk,
}: {
  tagger: string
  taggerStatus: { ok: boolean; msg: string } | null
  isLive: boolean
  taggerOk: boolean
}) {
  return (
    <div className="flex flex-col gap-3 min-w-0">
      {/* 状态卡片 */}
      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <div className="flex items-center gap-1.5 mb-2">
          <span className={`inline-block w-1.5 h-1.5 rounded-full shrink-0 ${taggerOk ? 'bg-ok' : 'bg-err'}`} />
          <span className="caption">状态</span>
        </div>
        <div className="text-xs text-fg-secondary">
          <div className={`font-mono font-medium ${taggerOk ? 'text-ok' : 'text-err'}`}>
            {tagger} {taggerStatus ? (taggerOk ? '✓ 就绪' : '✗ 不可用') : '… 检查中'}
          </div>
          {!taggerOk && taggerStatus && (
            <div className="mt-1 text-fg-tertiary break-all">
              {taggerStatus.msg}
            </div>
          )}
        </div>
      </div>

      {/* 说明卡片 */}
      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <div className="flex items-center gap-1.5 mb-2">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
          <span className="caption">说明</span>
        </div>
        <div className="text-xs text-fg-secondary leading-relaxed">
          {tagger === 'wd14'
            ? 'WD14 ONNX 本地推理，无需网络'
            : tagger === 'cltagger'
              ? 'CLTagger ONNX 本地推理，支持角色阈值'
              : tagger === 'llm'
                ? 'OpenAI compatible 视觉 LLM，支持 Responses / Chat Completions'
                : 'JoyCaption 远程 vLLM，自然语言描述'}
        </div>
      </div>

      {/* 进度提示 */}
      {isLive && (
        <div className="rounded-md border border-subtle bg-surface px-3 py-2.5 text-center">
          <div className="badge badge-warn">打标中</div>
        </div>
      )}
    </div>
  )
}
