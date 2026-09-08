import type { TFunction } from 'i18next'
import { useEffect, useId, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useOutletContext } from 'react-router-dom'
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
import Alert from '../../../components/Alert'
import Badge from '../../../components/Badge'
import Button from '../../../components/Button'
import Card from '../../../components/Card'
import { Checkbox, Input, Select, controlClassName } from '../../../components/FormControl'
import { InfoButton } from '../../../components/InfoButton'
import LLMPresetEditorModal, { llmPresetLabel } from '../../../components/LLMPresetEditorModal'
import Modal from '../../../components/Modal'
import ProgressBar from '../../../components/ProgressBar'
import { TagListInput } from '../../../components/TagsInput'
import StepShell from '../../../components/StepShell'
import { useToast } from '../../../components/Toast'
import { ModelSourceCard, SourceSelect } from '../../tools/settings/modelCards'
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

// form 与全局默认不同的字段挑出来作 check override（判据与 buildXXXOverrides
// 一致，但只取 keys 列出的影响可用性的字段）。form / defaults 未加载 → 不覆盖。
export function availabilityOverrides<T extends object>(
  form: T | null, defaults: T | null, keys: (keyof T)[],
): Partial<T> | undefined {
  if (!form || !defaults) return undefined
  const out: Partial<T> = {}
  for (const k of keys) if (form[k] !== defaults[k]) out[k] = form[k]
  return Object.keys(out).length ? out : undefined
}

export default function TaggingPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const settingsDrawer = useSettingsDrawer()
  const { catalog, setDownloadSource, secrets } = useSettingsData()

  const [tagger, setTagger] = useState<TaggerName>('wd14')
  const [taggerStatus, setTaggerStatus] = useState<TaggerStatus | null>(null)
  // 落盘格式跟着产物走（LLM json preset → .json，其余 → .txt），不再由请求指定。
  const [onExisting, setOnExisting] = useState<'overwrite' | 'skip' | 'append'>('skip')
  // 触发词：初值从 activeVersion 取（持久化在 version 表）；启动打标时一并提交，
  // 后端会同步落库 + 传给 worker prepend 到每张 caption。
  const [triggerWord, setTriggerWord] = useState<string>('')
  // 打标范围：'all'（默认 train + validation）/ 'validation' / 某个 train 文件夹名。
  // folders 给 dropdown 列 train 子文件夹选项（从 curation 拿）。
  const [scope, setScope] = useState<string>('all')
  const [folders, setFolders] = useState<string[]>([])
  const [folderLoadError, setFolderLoadError] = useState<string | null>(null)
  const [overwriteConfirmOpen, setOverwriteConfirmOpen] = useState(false)

  // defaults 直接从 SettingsData 的 live secrets 派生：设置抽屉 instant-apply
  // 的改动关上抽屉立即回流本页（不再持有 mount 时的一次性 getSecrets 快照）。
  const wd14Defaults: WD14Config | null = secrets?.wd14 ?? null
  const cltaggerDefaults: CLTaggerConfig | null = secrets?.cltagger ?? null
  const llmDefaults: LLMTaggerConfig | null = secrets?.llm_tagger ?? null

  // form 是"本次运行覆盖"草稿（不落盘）。defaults 更新时：未改动（与上一份
  // defaults 相等）→ 跟随刷新；改过 → 保留用户输入。
  const [wd14Form, setWd14Form] = useState<Wd14Form | null>(null)
  const [cltaggerForm, setCltaggerForm] = useState<CLTaggerForm | null>(null)
  // LLM 瘦身后本次覆盖只剩预设选择；编辑字段一律走 LLMPresetEditorModal。
  const [llmPresetId, setLlmPresetId] = useState<string | null>(null)
  const llmPresetTouchedRef = useRef(false)
  const [llmEditorOpen, setLlmEditorOpen] = useState(false)

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

  // secrets 回流时同步 form：初始化 + 未改动跟随（比较基准 = 上一份 defaults）。
  const prevTaggerDefaultsRef = useRef<{ wd14?: WD14Config; cltagger?: CLTaggerConfig }>({})
  useEffect(() => {
    if (!secrets) return
    const prev = prevTaggerDefaultsRef.current
    setWd14Form((f) =>
      !f || (prev.wd14 && JSON.stringify(f) === JSON.stringify(fromConfig(prev.wd14)))
        ? fromConfig(secrets.wd14)
        : f)
    setCltaggerForm((f) =>
      !f || (prev.cltagger && JSON.stringify(f) === JSON.stringify(fromCLTaggerConfig(prev.cltagger)))
        ? fromCLTaggerConfig(secrets.cltagger)
        : f)
    prevTaggerDefaultsRef.current = { wd14: secrets.wd14, cltagger: secrets.cltagger }
  }, [secrets])

  // LLM 预设选择：用户没手动切过就跟随全局默认；切过则保留（预设被删时回退默认）。
  useEffect(() => {
    if (!llmDefaults) return
    setLlmPresetId((id) => {
      if (!llmPresetTouchedRef.current || id === null) return llmDefaults.current_preset
      return llmDefaults.presets.some((p) => p.id === id) ? id : llmDefaults.current_preset
    })
  }, [llmDefaults])

  // 可用性检查跟着「本次打标实际生效的配置」走（issue #477）：模型版本 / 预设
  // 选择属于本次覆盖，不带上会按全局默认误报「需下载 / 未配置」并锁死开始按钮。
  // 只挑影响可用性的字段——阈值 / 附加标签类别不参与，避免输入过程反复请求。
  const availabilityKey = JSON.stringify([
    tagger,
    tagger === 'wd14'
      ? availabilityOverrides(wd14Form, wd14Defaults, ['model_id'])
      : tagger === 'cltagger'
        ? availabilityOverrides(cltaggerForm, cltaggerDefaults, [
            'model_id', 'model_path', 'tag_mapping_path',
          ])
        : tagger === 'llm' && llmPresetId && llmDefaults && llmPresetId !== llmDefaults.current_preset
          ? { current_preset: llmPresetId }
          : undefined,
  ])
  // secrets / catalog 进依赖：预设编辑器与设置抽屉保存（instant-apply 回流）、
  // 模型下载完成（model_download_changed → reloadCatalog）都要触发重查。
  useEffect(() => {
    const [name, overrides] = JSON.parse(availabilityKey) as [
      TaggerName,
      Record<string, unknown> | null,
    ]
    let stale = false
    setTaggerStatus(null)
    void api
      .checkTagger(name, overrides ?? undefined)
      .then((s) => { if (!stale) setTaggerStatus(s) })
      .catch((e) => {
        if (!stale) setTaggerStatus({ name, ok: false, msg: String(e), requires_service: false })
      })
    return () => { stale = true }
  }, [availabilityKey, secrets, catalog])

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
    setFolderLoadError(null)
    if (vid == null) { setFolders([]); return }
    void api
      .getCuration(project.id, vid)
      .then((v) => setFolders(v.folders))
      .catch((error) => {
        setFolders([])
        setFolderLoadError(String(error))
      })
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
  const valTotal = stats?.validation_image_count ?? 0
  const valTagged = stats?.validation_tagged_count ?? 0
  // 本次运行影响：all / validation 可精确计算；单训练文件夹缺少已打标数，
  // skip 时必须显示“启动后扫描”而不是把文件夹总数伪装成精确工作量。
  const selectedFolderTotal = scope === 'all' || scope === 'validation'
    ? null
    : (stats?.train_folders.find((folder) => folder.name === scope)?.image_count ?? 0)
  const runEstimate = scope === 'all'
    ? (onExisting === 'skip'
        ? Math.max(0, totalImages - taggedImages) + Math.max(0, valTotal - valTagged)
        : totalImages + valTotal)
    : scope === 'validation'
      ? (onExisting === 'skip' ? Math.max(0, valTotal - valTagged) : valTotal)
      : onExisting === 'skip'
        ? null
        : selectedFolderTotal
  const scopeLabel = scope === 'all'
    ? t('tag.scopeAll')
    : scope === 'validation'
      ? t('tag.scopeValidation')
      : scope
  const existingPolicyLabel = onExisting === 'skip'
    ? t('tag.onExistingSkip')
    : onExisting === 'append'
      ? t('tag.onExistingAppend')
      : t('tag.onExistingOverwrite')
  // 单文件夹没有已打标计数：若训练集全局已有 caption，则覆盖风险未知但存在，
  // 仍要求确认；若全局为 0，则可证明无需确认。
  const overwriteExistingCount = scope === 'all'
    ? taggedImages + valTagged
    : scope === 'validation'
      ? valTagged
      : taggedImages === 0 ? 0 : null
  const overwriteNeedsConfirm = onExisting === 'overwrite' && overwriteExistingCount !== 0

  // 历史卡只显示 job ledger 能证明的事实，不把今天的全局默认冒充成上次配置。
  const lastParams = jobParams(job)
  const lastTagger = typeof lastParams.tagger === 'string' ? lastParams.tagger : null
  const lastMethodLabel = lastTagger ? taggerLabel(lastTagger, t) : null
  const lastModelLabel =
    lastTagger === 'wd14'
      ? ((lastParams.wd14_overrides as { model_id?: string } | undefined)?.model_id ?? null)
      : lastTagger === 'cltagger'
        ? ((lastParams.cltagger_overrides as { model_id?: string } | undefined)?.model_id ?? null)
        : null
  const lastPresetId = lastTagger === 'llm'
    ? ((lastParams.llm_overrides as { current_preset?: string } | undefined)?.current_preset ?? null)
    : null
  const lastPresetLabel = lastPresetId
    ? (llmDefaults?.presets.find((preset) => preset.id === lastPresetId)?.label ?? lastPresetId)
    : null
  const lastTrigger = typeof lastParams.trigger_word === 'string'
    ? lastParams.trigger_word.trim()
    : ''

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

  // LLM 本次覆盖只剩预设选择：选了非全局默认的预设才发 current_preset override。
  const buildLLMOverrides = (): Record<string, unknown> | undefined => {
    if (!llmPresetId || !llmDefaults) return undefined
    if (llmPresetId === llmDefaults.current_preset) return undefined
    return { current_preset: llmPresetId }
  }

  const enqueueTagging = async () => {
    if (!taggerStatus?.ok) {
      toast(t('tag.taggerUnavailable', { tagger: taggerLabel(tagger, t), msg: taggerStatus?.msg ?? '' }), 'error')
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
      setOverwriteConfirmOpen(false)
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

  const requestStartTagging = () => {
    if (overwriteNeedsConfirm) {
      setOverwriteConfirmOpen(true)
      return
    }
    void enqueueTagging()
  }

  const unavailableMessage = taggerStatus && !taggerStatus.ok ? taggerStatus.msg : ''
  const needsOnnxRecovery = /onnx\s*runtime|onnxruntime/i.test(unavailableMessage)
  const needsModelRecovery = /需下载模型|download.*model|model.*(?:missing|not found)/i.test(unavailableMessage)

  return (
    <StepShell
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
        <Button
          variant="primary"
          size="sm"
          onClick={requestStartTagging}
          disabled={!taggerStatus?.ok || isLive}
          loading={isLive || taggerStatus === null}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M8 5v14l11-7z" />
          </svg>
          <span>
            {isLive ? t('tag.taggingBtn') : taggerStatus === null ? t('tag.checkingBtn') : t('tag.startBtn')}
          </span>
        </Button>
      }
    >
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div
        data-tagging-workspace
        className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-y-auto xl:grid-cols-[minmax(0,3fr)_minmax(20rem,2fr)] xl:overflow-hidden"
      >
        {/* 紧凑桌面由 workspace 统一滚动；宽桌面两栏各自滚动。 */}
        <div className="flex min-h-0 min-w-0 flex-col gap-3 xl:overflow-y-auto">
          {isLive && job && (
            <Alert
              tone="info"
              size="sm"
              title={t('tag.currentTaskTitle', {
                tagger: lastMethodLabel ?? taggerLabel(tagger, t),
                id: job.id,
              })}
            >
              {t('tag.nextRunDraftHint')}
            </Alert>
          )}
          <Card
            as="section"
            padding="md"
            className="shrink-0 text-sm"
            aria-labelledby="tag-run-settings-title"
          >
            <h2 id="tag-run-settings-title" className="type-panel-title mb-related">
              {t(isLive ? 'tag.nextRunSettingsTitle' : 'tag.runSettingsTitle')}
            </h2>
            <div className="grid grid-cols-1 gap-x-4 md:grid-cols-2">
              <TagField
                htmlFor="tagging-tagger"
                label={t('tag.fieldTagger')}
                helpTooltip={taggerDesc}
                help={
                  <span className="inline-flex min-w-0 items-center gap-2 flex-wrap">
                    <Badge
                      tone={taggerStatus ? (taggerStatus.ok ? 'success' : 'danger') : 'neutral'}
                      size="sm"
                      title={taggerStatus?.msg ?? t('tag.checkingBtn')}
                    >
                      {taggerStatus
                        ? taggerStatus.ok ? t('tag.statusReady') : t('tag.statusUnavail')
                        : t('tag.statusChecking')}
                    </Badge>
                    {taggerStatus?.msg && (
                      <span className="min-w-0 truncate" title={taggerStatus.msg}>{taggerStatus.msg}</span>
                    )}
                    {taggerStatus && !taggerStatus.ok && needsOnnxRecovery && (
                      <Button
                        variant="ghost"
                        size="xs"
                        onClick={() => settingsDrawer.open({ section: 'onnxruntime' })}
                      >
                        {t('tag.goInstallOnnx')}
                      </Button>
                    )}
                    {taggerStatus && !taggerStatus.ok && needsModelRecovery && (
                      <Button
                        variant="ghost"
                        size="xs"
                        onClick={() => settingsDrawer.open({ section: tagger === 'cltagger' ? 'cltagger' : 'wd14' })}
                      >
                        {t('tag.goDownload')}
                      </Button>
                    )}
                  </span>
                }
              >
                <Select
                  id="tagging-tagger"
                  value={tagger}
                  onChange={(event) => setTagger(event.target.value as TaggerName)}
                  controlSize="sm"
                  surface="canvas"
                >
                  <option value="wd14">{t('tag.taggerWd14')}</option>
                  <option value="cltagger">{t('tag.taggerCltagger')}</option>
                  <option value="llm">{t('tag.taggerLlm')}</option>
                </Select>
              </TagField>

              <TagField htmlFor="tagging-scope" label={t('tag.scope')} helpTooltip={t('tag.scopeHint')}>
                <Select
                  id="tagging-scope"
                  value={scope}
                  onChange={(event) => setScope(event.target.value)}
                  controlSize="sm"
                  surface="canvas"
                >
                  <option value="all">{t('tag.scopeAll')}</option>
                  {folders.map((folder) => (
                    <option key={folder} value={folder}>{folder}</option>
                  ))}
                  <option value="validation">{t('tag.scopeValidation')}</option>
                </Select>
              </TagField>

              <TagField htmlFor="tagging-existing" label={t('tag.onExisting')} helpTooltip={t('tag.onExistingHint')}>
                <Select
                  id="tagging-existing"
                  value={onExisting}
                  onChange={(event) => setOnExisting(event.target.value as 'overwrite' | 'skip' | 'append')}
                  controlSize="sm"
                  surface="canvas"
                >
                  <option value="skip">{t('tag.onExistingSkip')}</option>
                  <option value="overwrite">{t('tag.onExistingOverwrite')}</option>
                  <option value="append">{t('tag.onExistingAppend')}</option>
                </Select>
              </TagField>

              <TagField htmlFor="tagging-trigger" label={t('tag.triggerWord')} helpTooltip={t('tag.triggerWordHint')}>
                <Input
                  id="tagging-trigger"
                  type="text"
                  value={triggerWord}
                  onChange={(event) => setTriggerWord(event.target.value)}
                  placeholder={t('tag.triggerWordPlaceholder')}
                  controlSize="sm"
                  surface="canvas"
                  mono
                  className={triggerWord.trim() !== (activeVersion.trigger_word ?? '') ? 'border-warn' : ''}
                />
              </TagField>
            </div>
            {folderLoadError && (
              <Alert tone="warning" size="sm" className="mt-related" title={t('tag.scopeLoadFailed')}>
                {folderLoadError}
              </Alert>
            )}
          </Card>

          {tagger === 'wd14' && (
            <Wd14Panel
              form={wd14Form}
              defaults={wd14Defaults}
              onChange={setWd14Form}
              disabled={false}
              downloadCenter={
                wd14Form && (
                  <div className="flex flex-col gap-3">
                    <SourceSelect
                      opt={catalog?.download_source_options?.wd14}
                      onChange={(s) => void setDownloadSource('wd14', s)}
                    />
                    <ModelSourceCard
                      domain="wd14"
                      title={t('settings.wd14CandidateTitle', { name: catalog?.wd14?.name ?? 'WD14' })}
                      catalog={catalog}
                      currentValue={wd14Form.model_id}
                      onSelect={(id) => setWd14Form({ ...wd14Form, model_id: id })}
                      addDownload={{}}
                      addLocal={{ dirOnly: true }}
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
              disabled={false}
              downloadCenter={
                cltaggerForm && (
                  <div className="flex flex-col gap-3">
                    <SourceSelect
                      opt={catalog?.download_source_options?.cltagger}
                      onChange={(s) => void setDownloadSource('cltagger', s)}
                    />
                    <ModelSourceCard
                      domain="cltagger"
                      title={t('settings.clTaggerVersionTitle', { name: catalog?.cltagger?.name ?? 'CLTagger' })}
                      catalog={catalog}
                      currentValue={`${cltaggerForm.model_id}|${cltaggerForm.model_path}|${cltaggerForm.tag_mapping_path}`}
                      onSelect={(_, row) =>
                        setCltaggerForm({
                          ...cltaggerForm,
                          model_id: row.extra.model_id ?? '',
                          model_path: row.extra.model_path ?? '',
                          tag_mapping_path: row.extra.tag_mapping_path ?? '',
                        })
                      }
                      addDownload={{ repoPlaceholder: 'cella110n/cl_tagger' }}
                      addLocal={{ secondFileKey: 'tag_mapping_path' }}
                      t={t}
                    />
                  </div>
                )
              }
            />
          )}

          {tagger === 'llm' && (
            <LLMTaggerPanel
              presetId={llmPresetId}
              defaults={llmDefaults}
              onSelect={(id) => {
                llmPresetTouchedRef.current = true
                setLlmPresetId(id)
              }}
              onEdit={() => setLlmEditorOpen(true)}
              disabled={false}
            />
          )}

        </div>

        <TagStatusPanel
          currentTagger={taggerLabel(tagger, t)}
          scopeLabel={scopeLabel}
          existingPolicyLabel={existingPolicyLabel}
          runEstimate={runEstimate}
          currentTriggerWord={triggerWord.trim()}
          overwriteWarning={overwriteNeedsConfirm}
          totalImages={totalImages}
          taggedImages={taggedImages}
          methodLabel={lastMethodLabel}
          modelLabel={lastModelLabel}
          presetLabel={lastPresetLabel}
          lastTriggerWord={lastTrigger}
          latestTaggedAt={job?.finished_at ?? null}
          validationTotal={valTotal}
          validationTagged={valTagged}
          isLive={isLive}
        />
      </div>
    </div>

    {llmEditorOpen && llmPresetId && (
      <LLMPresetEditorModal
        presetId={llmPresetId}
        onClose={() => setLlmEditorOpen(false)}
      />
    )}

    {overwriteConfirmOpen && (
      <Modal
        role="alertdialog"
        size="sm"
        title={t('tag.overwriteConfirmTitle')}
        description={t('tag.overwriteConfirmDescription')}
        onClose={() => setOverwriteConfirmOpen(false)}
        footer={
          <div className="flex justify-end gap-related">
            <Button variant="secondary" onClick={() => setOverwriteConfirmOpen(false)}>
              {t('common.cancel')}
            </Button>
            <Button variant="danger" onClick={() => void enqueueTagging()}>
              {t('tag.overwriteConfirmAction')}
            </Button>
          </div>
        }
      >
        <Alert tone="warning" title={t('tag.overwriteWarningTitle')}>
          {overwriteExistingCount == null
            ? t('tag.overwriteWarningUnknown', { scope: scopeLabel })
            : t('tag.overwriteWarningKnown', { scope: scopeLabel, n: overwriteExistingCount })}
        </Alert>
      </Modal>
    )}
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
      <Card as="section" padding="sm" className="shrink-0 text-xs text-fg-tertiary">
        {t('tag.wd14Loading')}
      </Card>
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
      <Card as="section" padding="md" className="flex shrink-0 flex-col gap-2 text-sm">
        <PanelHeader dirty={dirty} onRestore={restore} disabled={disabled} />
        <div className="grid grid-cols-1 gap-x-4 md:grid-cols-2">
          <TagFieldNumber label={t('settings.fieldThresholdGeneral')} value={form.threshold_general} base={defaults.threshold_general} min={0} max={1} step={0.01} disabled={disabled} onChange={(v) => onChange({ ...form, threshold_general: v })} />
          <TagFieldNumber label={t('settings.fieldThresholdCharacter')} value={form.threshold_character} base={defaults.threshold_character} min={0} max={1} step={0.01} disabled={disabled} onChange={(v) => onChange({ ...form, threshold_character: v })} />
        </div>
      </Card>

      <AdvancedSection>
        <div className="flex flex-col gap-3">
          {downloadCenter}
          <TagField label={t('settings.fieldBlacklistTags')}>
            <TagListInput
              ariaLabel={t('settings.fieldBlacklistTags')}
              value={form.blacklist_tags}
              placeholder={t('tag.blacklistPlaceholder1')}
              disabled={disabled}
              onChange={(tags) => onChange({ ...form, blacklist_tags: tags })}
              className={controlClassName({
                size: 'sm', surface: 'canvas', mono: true,
                className: JSON.stringify(form.blacklist_tags) !== JSON.stringify(defaults.blacklist_tags)
                  ? 'border-warn' : '',
              })}
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
      <h2 className="type-panel-title">{t('tag.taggerParams')}</h2>
      {subtitle && <span className="text-xs text-fg-tertiary">{subtitle}</span>}
      <span className="flex-1" />
      {dirty && (
        <>
          <Badge tone="warning" size="sm">{t('tag.modified')}</Badge>
          <Button variant="ghost" size="xs" onClick={onRestore} disabled={disabled} title={t('tag.restore')}>
            {t('tag.restore')}
          </Button>
        </>
      )}
    </div>
  )
}

// 高级参数：独立折叠卡（对齐训练配置页 SchemaForm 分组 section），默认收起。
function AdvancedSection({ children }: { children: React.ReactNode }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const contentId = useId()
  return (
    <Card as="section" padding="none" className="shrink-0 text-sm">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-controls={contentId}
        className="w-full justify-between rounded-none px-3.5 py-2.5"
      >
        <span>{t('tag.advanced')}</span>
        <span className="text-xs text-fg-tertiary" aria-hidden="true">{open ? '▾' : '▸'}</span>
      </Button>
      <div id={contentId} hidden={!open} className="px-3.5 pb-2.5">
        {open ? children : null}
      </div>
    </Card>
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
      <Card as="section" padding="sm" className="shrink-0 text-xs text-fg-tertiary">
        {t('tag.cltaggerLoading')}
      </Card>
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
      <Card as="section" padding="md" className="flex shrink-0 flex-col gap-2 text-sm">
        <PanelHeader dirty={dirty} onRestore={restore} disabled={disabled} />
        <div className="grid grid-cols-1 gap-x-4 md:grid-cols-2">
          <TagFieldNumber label={t('settings.fieldThresholdGeneral')} value={form.threshold_general} base={defaults.threshold_general} min={0} max={1} step={0.01} disabled={disabled} onChange={(v) => onChange({ ...form, threshold_general: v })} />
          <TagFieldNumber label={t('settings.fieldThresholdCharacter')} value={form.threshold_character} base={defaults.threshold_character} min={0} max={1} step={0.01} disabled={disabled} onChange={(v) => onChange({ ...form, threshold_character: v })} />
        </div>
      </Card>

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
              ariaLabel={t('settings.fieldBlacklistTags')}
              value={form.blacklist_tags}
              placeholder={t('tag.blacklistPlaceholder2')}
              disabled={disabled}
              onChange={(tags) => onChange({ ...form, blacklist_tags: tags })}
              className={controlClassName({
                size: 'sm', surface: 'canvas', mono: true,
                className: JSON.stringify(form.blacklist_tags) !== JSON.stringify(defaults.blacklist_tags)
                  ? 'border-warn' : '',
              })}
            />
          </TagField>
        </div>
      </AdvancedSection>
    </>
  )
}

// LLM 面板（瘦身版）：只保留本次运行的预设选择 + 预设编辑 modal 入口。
// 字段级 per-run 覆盖已移除——要调参数就编辑预设本体（全局唯一编辑器）。
function LLMTaggerPanel({
  presetId, defaults, onSelect, onEdit, disabled,
}: {
  presetId: string | null
  defaults: LLMTaggerConfig | null
  onSelect: (id: string) => void
  onEdit: () => void
  disabled: boolean
}) {
  const { t } = useTranslation()
  if (!defaults || !presetId) {
    return (
      <Card as="section" padding="sm" className="shrink-0 text-xs text-fg-tertiary">
        {t('tag.llmLoading')}
      </Card>
    )
  }

  const active = defaults.presets.find((p) => p.id === presetId) ?? defaults.presets[0]
  if (!active) {
    return (
      <Card as="section" padding="sm" className="shrink-0 text-xs text-err">
        {t('tag.llmNoPreset')}
      </Card>
    )
  }

  const overridden = presetId !== defaults.current_preset
  // 预设本体的健康提示：开了 assist 但提示词没有 {{tags}} 占位符时预打标不会生效。
  const assistNeedsTags =
    !!active.assist_tagger &&
    !active.messages.some((m) => m.type === 'text' && m.content.includes('{{tags}}'))

  return (
    <Card as="section" padding="md" className="flex shrink-0 flex-col gap-2 text-sm">
      {/* header 与 WD14/CLTagger 面板同款 PanelHeader；还原 = 切回全局默认预设 */}
      <PanelHeader
        dirty={overridden}
        onRestore={() => onSelect(defaults.current_preset)}
        disabled={disabled}
      />

      <div className="grid grid-cols-1 gap-x-4">
        <TagFieldSelect
          label={t('tag.fieldPreset')}
          labelExtra={
            <Button variant="ghost" size="xs" onClick={onEdit}>
              {t('tag.llmEditPreset')}
            </Button>
          }
          value={active.id}
          disabled={disabled}
          onChange={onSelect}
          modified={overridden}
          helpTooltip={t('tag.llmPresetScopeHint')}
          help={
            <>
              {/* 选中预设的关键配置摘要——细节和编辑都在预设编辑器里 */}
              <span className="font-mono">
                {active.model || t('tag.llmNoModel')}
                {' · '}{active.output_format === 'json' ? t('llmPreset.jsonCaption') : t('llmPreset.textCaption')}
                {' · temp '}{active.temperature}
                {' · ×'}{active.concurrency}
                {active.assist_tagger ? ` · +${active.assist_tagger}` : ''}
              </span>
              {assistNeedsTags && (
                <div className="text-warn mt-0.5">
                  {t('llmPreset.assistNeedsTags').split('%TAGS%').join('{{tags}}')}
                </div>
              )}
            </>
          }
        >
          {defaults.presets.map((p) => (
            <option key={p.id} value={p.id}>
              {llmPresetLabel(p, t)}{p.builtin ? t('tag.builtin') : ''}
            </option>
          ))}
        </TagFieldSelect>
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// 面板字段块：统一使用共享 FormControl，并让可见 label 与控件建立程序化关联。
// ---------------------------------------------------------------------------

function TagField({ htmlFor, label, labelExtra, helpTooltip, help, className = '', children }: {
  htmlFor?: string
  label: string
  labelExtra?: React.ReactNode
  helpTooltip?: React.ReactNode
  help?: React.ReactNode
  className?: string
  children: React.ReactNode
}) {
  return (
    <div className={`py-1.5 ${className}`}>
      <div className="mb-1 flex items-center gap-2 type-field-label">
        {htmlFor ? <label htmlFor={htmlFor}>{label}</label> : <span>{label}</span>}
        {helpTooltip && <InfoButton>{helpTooltip}</InfoButton>}
        {labelExtra && <span className="text-[11px] font-normal">{labelExtra}</span>}
      </div>
      {children}
      {help && <div className="mt-1 type-field-help">{help}</div>}
    </div>
  )
}

function TagFieldNumber({ label, value, base, min, max, step = 1, disabled, onChange, help }: {
  label: string; value: number; base: number; min: number; max: number; step?: number
  disabled: boolean; onChange: (v: number) => void; help?: React.ReactNode
}) {
  const { t } = useTranslation()
  const inputId = useId()
  const modified = value !== base
  return (
    <TagField htmlFor={inputId} label={label} help={help}>
      <Input
        id={inputId}
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => {
          const next = Number(event.target.value)
          if (!Number.isNaN(next)) onChange(Math.max(min, Math.min(max, next)))
        }}
        disabled={disabled}
        controlSize="sm"
        surface="canvas"
        mono
        className={modified ? 'border-warn' : ''}
        title={modified ? `${t('tag.modified')} · ${base}` : undefined}
      />
    </TagField>
  )
}

function TagFieldSelect({ label, labelExtra, value, disabled, onChange, modified, helpTooltip, help, className, title, children }: {
  label: string; labelExtra?: React.ReactNode; value: string; disabled: boolean
  onChange: (v: string) => void; modified?: boolean
  helpTooltip?: React.ReactNode; help?: React.ReactNode
  className?: string; title?: string; children: React.ReactNode
}) {
  const selectId = useId()
  return (
    <TagField
      htmlFor={selectId}
      label={label}
      labelExtra={labelExtra}
      helpTooltip={helpTooltip}
      help={help}
      className={className}
    >
      <Select
        id={selectId}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
        controlSize="sm"
        surface="canvas"
        mono
        className={modified ? 'border-warn' : ''}
        title={title}
      >
        {children}
      </Select>
    </TagField>
  )
}

function TagFieldCheckbox({ label, checked, disabled, onChange }: {
  label: string; checked: boolean; disabled: boolean; onChange: (v: boolean) => void
}) {
  return (
    <label className={`flex items-center gap-1.5 text-sm text-fg-secondary ${disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}`}>
      <Checkbox
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
        controlSize="sm"
      />
      {label}
    </label>
  )
}

function PanelDot() {
  return <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
}

// ---------------------------------------------------------------------------
// 右栏：运行中把不可变任务与下一轮草稿分开；数据状态只承载持久化事实。
// ---------------------------------------------------------------------------

function TagStatusPanel({
  currentTagger, scopeLabel, existingPolicyLabel, runEstimate, currentTriggerWord,
  overwriteWarning,
  totalImages, taggedImages,
  methodLabel, modelLabel, presetLabel, lastTriggerWord, latestTaggedAt,
  validationTotal, validationTagged, isLive,
}: {
  currentTagger: string
  scopeLabel: string
  existingPolicyLabel: string
  runEstimate: number | null
  currentTriggerWord: string
  overwriteWarning: boolean
  totalImages: number
  taggedImages: number
  methodLabel: string | null
  modelLabel: string | null
  presetLabel: string | null
  lastTriggerWord: string
  latestTaggedAt: number | null
  validationTotal: number
  validationTagged: number
  isLive: boolean
}) {
  const { t } = useTranslation()
  const hasHistory = methodLabel != null || latestTaggedAt != null

  return (
    <aside
      aria-label={t('tag.statusTitle')}
      className="flex min-w-0 flex-col gap-3 xl:min-h-0 xl:overflow-y-auto"
    >
      <Card as="section" padding="md" aria-labelledby="tag-run-plan-title">
        <h2 id="tag-run-plan-title" className="type-panel-title mb-section">
          {t(isLive ? 'tag.nextRunPlanTitle' : 'tag.runPlanTitle')}
        </h2>
        <div className="flex flex-col gap-2">
          <TagStatusRow label={t('tag.fieldTagger')}>
            <span className="font-mono">{currentTagger}</span>
          </TagStatusRow>
          <TagStatusRow label={t('tag.scope')}>
            <span>{scopeLabel}</span>
          </TagStatusRow>
          <TagStatusRow label={t('tag.onExisting')}>
            <span>{existingPolicyLabel}</span>
          </TagStatusRow>
          <TagStatusRow label={t('tag.statusThisRound')}>
            {runEstimate == null ? (
              <span className="text-fg-secondary">{t('tag.scanAfterStart')}</span>
            ) : (
              <span className="font-mono text-accent">
                {runEstimate} <span className="text-2xs font-normal text-fg-tertiary">{t('tag.nImagesShort')}</span>
              </span>
            )}
          </TagStatusRow>
          {currentTriggerWord && (
            <TagStatusRow label={t('tag.statusTrigger')}>
              <span className="break-all font-mono">{currentTriggerWord}</span>
            </TagStatusRow>
          )}
        </div>
        {overwriteWarning && (
          <Alert tone="warning" size="sm" className="mt-section">
            {t('tag.overwriteInlineWarning')}
          </Alert>
        )}
      </Card>

      <Card as="section" padding="md" aria-labelledby="tag-data-status-title">
        <h2 id="tag-data-status-title" className="type-panel-title mb-section">{t('tag.dataStatusTitle')}</h2>
        <div className="flex flex-col gap-section">
          <TagCoverage label={t('tag.trainingCoverage')} tagged={taggedImages} total={totalImages} />
          {validationTotal > 0 && (
            <TagCoverage label={t('tag.validationCoverage')} tagged={validationTagged} total={validationTotal} />
          )}
          <div className="border-t border-subtle pt-section">
            <h3 className="type-field-label mb-related">{t('tag.lastRunTitle')}</h3>
            {hasHistory ? (
              <div className="flex flex-col gap-2">
                {methodLabel && (
                  <TagStatusRow label={t('tag.statusMethod')}>
                    <span className="font-mono">{methodLabel}</span>
                  </TagStatusRow>
                )}
                {modelLabel && (
                  <TagStatusRow label={t('tag.statusModel')}>
                    <span className="break-all font-mono">{modelLabel}</span>
                  </TagStatusRow>
                )}
                {presetLabel && (
                  <TagStatusRow label={t('tag.statusPreset')}>
                    <span className="break-all font-mono">{presetLabel}</span>
                  </TagStatusRow>
                )}
                {lastTriggerWord && (
                  <TagStatusRow label={t('tag.statusTrigger')}>
                    <span className="break-all font-mono">{lastTriggerWord}</span>
                  </TagStatusRow>
                )}
                <TagStatusRow label={t('tag.statusLatest')}>
                  <span className="text-sm text-fg-secondary">
                    {latestTaggedAt ? formatAgo(latestTaggedAt, t) : '—'}
                  </span>
                </TagStatusRow>
              </div>
            ) : (
              <p className="type-field-help">{t('tag.noPreviousRun')}</p>
            )}
          </div>
        </div>
      </Card>
    </aside>
  )
}

function TagCoverage({ label, tagged, total }: { label: string; tagged: number; total: number }) {
  const { t } = useTranslation()
  const valueText = t('tag.coverageValue', { tagged, total })
  return (
    <div className="flex flex-col gap-related">
      <TagStatusRow label={label}>
        <span className="font-mono">
          <span className="text-ok">{tagged}</span>
          <span className="ml-1 text-2xs font-normal text-fg-tertiary">
            / {total} {t('tag.nImagesShort')}
          </span>
        </span>
      </TagStatusRow>
      <ProgressBar
        label={label}
        value={tagged}
        max={Math.max(1, total)}
        valueText={valueText}
        size="xs"
        tone={total > 0 && tagged >= total ? 'success' : 'accent'}
      />
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

function taggerLabel(name: string | null, t: TFunction): string {
  if (name === 'wd14') return t('tag.taggerWd14Short')
  if (name === 'cltagger') return t('tag.taggerCltaggerShort')
  if (name === 'llm') return t('tag.taggerLlmShort')
  return name ?? ''
}

// 最近一次 tag job 的参数：优先 params_decoded，退回解析 params 原始 JSON。
function jobParams(job: Job | null): Record<string, unknown> {
  if (!job) return {}
  if (job.params_decoded && typeof job.params_decoded === 'object') return job.params_decoded
  try {
    return job.params ? (JSON.parse(job.params) as Record<string, unknown>) : {}
  } catch {
    return {}
  }
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
