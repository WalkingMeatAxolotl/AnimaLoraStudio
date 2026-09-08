import type { TFunction } from 'i18next'
import { useCallback, useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { useTranslation } from 'react-i18next'
import { useOutletContext } from 'react-router-dom'
import {
  api,
  type Job,
  type ProjectDetail,
  type RegAiRequest,
  type RegBuildRequest,
  type RegStatus,
  type RegTagCount,
  type Task,
  type Version,
} from '../../../api/client'
import ActionGroup from '../../../components/ActionGroup'
import Alert from '../../../components/Alert'
import BaseModelSelect from '../../../components/BaseModelSelect'
import Button from '../../../components/Button'
import Card from '../../../components/Card'
import EmptyState from '../../../components/EmptyState'
import { Checkbox, Input, Select, Textarea } from '../../../components/FormControl'
import { InfoButton } from '../../../components/InfoButton'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import StepShell from '../../../components/StepShell'
import { Tabs } from '../../../components/SelectionGroup'
import { TranslatedTag } from '../../../components/tagDisplay/TranslatedTag'
import { TagSuggestList } from '../../../components/tagSuggest/TagSuggestList'
import { useTagSuggest } from '../../../components/tagSuggest/useTagSuggest'
import { useDialog } from '../../../components/Dialog'
import { useToast } from '../../../components/Toast'
import { compareImagePath } from '../../../lib/imageSort'
import { useEventStream } from '../../../lib/useEventStream'
import { useLatestJobReplay } from '../../../lib/useLatestJobReplay'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

interface AdvancedParams {
  skip_similar: boolean
  aspect_ratio_filter_enabled: boolean
  min_aspect_ratio: number
  max_aspect_ratio: number
  postprocess_method: 'smart' | 'stretch' | 'crop'
  postprocess_max_crop_ratio: number
}

// batch_size 不暴露 — 多 train 子文件夹（5_concept / 1_general 等）共用同一 batch
// 概念在 UI 上意义不大，保持源脚本默认 5。
const ADVANCED_DEFAULTS: AdvancedParams = {
  skip_similar: true,
  aspect_ratio_filter_enabled: false,
  min_aspect_ratio: 0.5,
  max_aspect_ratio: 2.0,
  postprocess_method: 'smart',
  postprocess_max_crop_ratio: 0.1,
}

// 排除 tag 归一到 booru 形态（小写 + 下划线）：train top-tag、自定义输入、全局默认
// 三个来源都过这个，保证 excluded 集里形态一致（与后端 booru 搜索语法对齐）。
const normalizeRegTag = (raw: string): string =>
  raw.trim().toLowerCase().replace(/\s+/g, '_')

export default function RegularizationPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()

  const [reg, setReg] = useState<RegStatus | null>(null)
  const [regLoading, setRegLoading] = useState(true)
  const [regError, setRegError] = useState<string | null>(null)
  const [trainTags, setTrainTags] = useState<RegTagCount[]>([])
  const [trainTagsLoading, setTrainTagsLoading] = useState(true)
  const [trainTagsError, setTrainTagsError] = useState<string | null>(null)
  // excluded 既包含 train top-tag 上点掉的，也包含「自定义排除」输入框加的（这部分
  // 在 train top-tag 列表里查不到）。后端不存这份选择，切页面回来需要按
  // (project, version) 在 localStorage 恢复，不然用户加的自定义 tag 看着就丢了。
  const [excluded, setExcluded] = useState<Set<string>>(new Set())
  const [autoTag, setAutoTag] = useState(true)
  // A3 — reg 自动打标的 tagger 选择。UI 暴露 wd14 / cltagger；后端 422 校验同。
  const [autoTagKind, setAutoTagKind] = useState<'wd14' | 'cltagger'>('wd14')
  // A4 v2 — build 模式 + 自动去重，默认增量 + 开。模式取代了原来的「开始 / 补足」
  // 两按钮（去掉补足，统一一个「开始生成」按钮按 mode 跑）。
  const [mode, setMode] = useState<'full' | 'incremental'>('incremental')
  const [autoDedup, setAutoDedup] = useState(true)
  // B1（PR-2）— 构建模式 + 目标数（仅 flat 模式生效）。默认 flat，target 留空 = train 总数。
  const [buildMode, setBuildMode] = useState<'mirror' | 'flat'>('flat')
  const [targetCount, setTargetCount] = useState<string>('')  // input value (string for blank → null)
  const [apiSource, setApiSource] = useState<'gelbooru' | 'danbooru'>('gelbooru')
  const [advanced, setAdvanced] = useState<AdvancedParams>(ADVANCED_DEFAULTS)

  const vid = activeVersion?.id ?? null

  // booru reg_build job：最近一次任务 + 日志回放（进页面 / SSE 重连时 hydrate）
  const {
    item: job,
    logs,
    setItem: setJob,
    setLogs,
    itemIdRef: jobIdRef,
    refresh: refreshLatestRegBuild,
  } = useLatestJobReplay<Job>(vid, (v) =>
    api.getLatestVersionJob(project.id, v, 'reg_build').then((r) => ({ item: r.job, log: r.log })),
  )

  // 生成配置与历史任务合并为单一「生成」阶段；顶部来源选择决定表单与主操作语义。
  const [activeTab, setActiveTab] = useState<'generate' | 'images'>('generate')
  // 来源默认 AI 先验（#8 决策 2026-05-30）：对齐 DreamBooth 原论文 neutral prior。
  // Booru 路径保留作"省时间"备选（不烧 GPU、更快出图）。
  const [source, setSource] = useState<'booru' | 'ai'>('ai')

  // 先验生成 — base 模型对每张 train 图反向出对照图，无 LoRA 参数（DreamBooth prior preservation）。
  // excluded tag 复用主组件 `excluded` Set，与 booru tab 双向同步。
  const [aiNeg, setAiNeg] = useState(
    'worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, bad anatomy, bad hands, bad feet'
  )
  const [aiWidth, setAiWidth] = useState(1024)
  const [aiHeight, setAiHeight] = useState(1024)
  const [aiSteps, setAiSteps] = useState(25)
  const [aiCfg, setAiCfg] = useState(4.0)
  const [aiSeed, setAiSeed] = useState(0)
  const [aiIncremental, setAiIncremental] = useState(true)
  // 本次先验生成临时选用的底模（null = 跟随设置页该族 selected）。
  const [aiBaseModel, setAiBaseModel] = useState<string | null>(null)
  // 先验生成的模型族跟随 version 训练配置（服务端权威解析；这里只为
  // BaseModelSelect 列对族的底模）。无 config / 读失败 → anima。
  const [aiFamily, setAiFamily] = useState<'anima' | 'krea2'>('anima')
  useEffect(() => {
    if (!vid) return
    let alive = true
    api.getVersionConfig(project.id, vid)
      .then((r) => {
        if (!alive) return
        const fam = r.config?.model_family === 'krea2' ? 'krea2' : 'anima'
        setAiFamily(fam)
        setAiBaseModel(null)  // 族变 → 清临时底模覆盖（variant key 是族内值）
      })
      .catch(() => {})
    return () => { alive = false }
  }, [project.id, vid])
  const [aiBusy, setAiBusy] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  // AI 先验 task：同上；hydrate 时顺带把 aiBusy 同步到 task 真实状态
  const {
    item: aiTask,
    logs: aiLogs,
    setItem: setAiTask,
    setLogs: setAiLogs,
    itemIdRef: aiTaskIdRef,
    refresh: refreshLatestRegPrior,
  } = useLatestJobReplay<Task>(
    vid,
    (v) => api.getLatestRegPriorTask(project.id, v).then((r) => ({ item: r.task, log: r.log })),
    (task) => setAiBusy(task ? task.status === 'running' || task.status === 'pending' : false),
  )

  // 预览 modal
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)
  const [previewCaption, setPreviewCaption] = useState<string>('')

  const refreshReg = useCallback(async () => {
    if (!vid) return
    setRegLoading(true)
    try {
      const s = await api.getRegStatus(project.id, vid)
      // 在源头排序：预览 modal 的 onPick(idx) 直接索引 reg.files，
      // 缩略图网格也从 reg.files 顺序派生，两者必须同序。
      setReg({ ...s, files: [...s.files].sort(compareImagePath) })
      setRegError(null)
    } catch (e) {
      setRegError(String(e))
    } finally {
      setRegLoading(false)
    }
  }, [project.id, vid])

  const refreshTrainTags = useCallback(async () => {
    if (!vid) return
    setTrainTagsLoading(true)
    try {
      const items = await api.previewRegTags(project.id, vid, 30)
      setTrainTags(items)
      setTrainTagsError(null)
    } catch (e) {
      setTrainTagsError(String(e))
    } finally {
      setTrainTagsLoading(false)
    }
  }, [project.id, vid])

  useEffect(() => {
    setReg(null)
    setRegError(null)
    setTrainTags([])
    setTrainTagsError(null)
    void refreshReg()
    void refreshTrainTags()
  }, [vid, refreshReg, refreshTrainTags])

  // 全局默认排除（Settings → 正则集）。null = 还没拉到 secrets；拉到前既不 seed 也
  // 不写盘，避免初始空值把某个 build 的本地记录覆盖成空、让种子永远不触发。
  const [defaultExcluded, setDefaultExcluded] = useState<string[] | null>(null)
  useEffect(() => {
    let alive = true
    api.getSecrets()
      .then((s) => { if (alive) setDefaultExcluded(s.reg?.default_excluded_tags ?? []) })
      .catch(() => { if (alive) setDefaultExcluded([]) })
    return () => { alive = false }
  }, [])

  // 把 excluded 持久化到 localStorage（按 project + version 隔离），切页面回来也在。
  // 进页面 / 切 version 时：有本地记录就恢复；没有则用全局默认排除做种子（归一到
  // booru 形态，与 train top-tag / 自定义输入一致）。之后随 setExcluded 自动保存。
  const excludedStorageKey = vid
    ? `studio.reg.excluded.${project.id}.${vid}`
    : null
  const hydratedKeyRef = useRef<string | null>(null)
  useEffect(() => {
    if (!excludedStorageKey || defaultExcluded === null) return
    try {
      const raw = localStorage.getItem(excludedStorageKey)
      if (!raw) {
        // 没有这个 build 的本地选择 → 用全局默认排除做初始值
        setExcluded(new Set(defaultExcluded.map(normalizeRegTag).filter(Boolean)))
      } else {
        const arr = JSON.parse(raw)
        setExcluded(Array.isArray(arr)
          ? new Set(arr.filter((x): x is string => typeof x === 'string'))
          : new Set())
      }
    } catch {
      setExcluded(new Set())
    }
    hydratedKeyRef.current = excludedStorageKey
  }, [excludedStorageKey, defaultExcluded])
  useEffect(() => {
    // 必须等 seed 跑完（hydratedKeyRef 命中当前 key）才允许写盘，否则 secrets 还没
    // 拉到时的初始空值会先落盘，把「无本地记录」变成「有空记录」，种子就被吃掉了。
    if (!excludedStorageKey || hydratedKeyRef.current !== excludedStorageKey) return
    try {
      localStorage.setItem(excludedStorageKey, JSON.stringify(Array.from(excluded)))
    } catch { /* quota / privacy mode：丢就丢，不打扰用户 */ }
  }, [excludedStorageKey, excluded])

  // 刷新 / 进入页面时回放最近一次生成任务：锁回 id + 回放历史日志。
  useEffect(() => {
    void refreshLatestRegBuild()
    void refreshLatestRegPrior()
  }, [refreshLatestRegBuild, refreshLatestRegPrior])

  const refreshLiveLogs = useCallback(() => {
    void refreshLatestRegBuild()
    void refreshLatestRegPrior()
  }, [refreshLatestRegBuild, refreshLatestRegPrior])

  useEventStream((evt) => {
    const jid = jobIdRef.current
    const tid = aiTaskIdRef.current
    if (evt.type === 'job_log_appended' && jid && evt.job_id === jid) {
      setLogs((prev) => [...prev, String(evt.text ?? '')])
    } else if (evt.type === 'job_state_changed' && jid && evt.job_id === jid) {
      void api.getJob(jid).then(setJob).catch(() => {})
      if (evt.status === 'done' || evt.status === 'failed' || evt.status === 'canceled') {
        void refreshReg()
        void reload()
        if (evt.status === 'done') setActiveTab('images')
      }
    } else if (evt.type === 'task_log_appended' && tid && evt.task_id === tid) {
      setAiLogs((prev) => [...prev, String(evt.text ?? '')])
    } else if (evt.type === 'task_state_changed' && tid && evt.task_id === tid) {
      void api.getRegPriorTask(project.id, vid!, tid).then((t) => {
        setAiTask(t)
        if (t.status === 'done' || t.status === 'failed' || t.status === 'canceled') {
          setAiBusy(false)
          void refreshReg()
          void reload()
          if (t.status === 'done') setActiveTab('images')
        }
      }).catch(() => {})
    }
  }, { onOpen: refreshLiveLogs })

  const trainImageCount = activeVersion?.stats?.train_image_count ?? 0
  // 任意一种生成跑着都视为 live —— 防止 booru / AI 并发同时写 reg/。
  const isTaskLive = job?.status === 'running' || job?.status === 'pending' || aiBusy
  const isLive = isTaskLive || submitting

  // B1（PR-2）— 现有 reg 集结构推断：meta.build_mode 优先（新 meta 写入），
  // 否则看 reg.files 路径前缀（仅 1_data/ → flat；含 N_xxx 多种 → mirror）。
  // 空集 → null（mode 可自由切换）。
  const existingMode = useMemo<'mirror' | 'flat' | null>(() => {
    if (!reg || !reg.exists || reg.image_count === 0) return null
    if (reg.meta?.build_mode === 'mirror' || reg.meta?.build_mode === 'flat') {
      return reg.meta.build_mode
    }
    const prefixes = new Set<string>()
    for (const rel of reg.files) {
      const idx = rel.indexOf('/')
      prefixes.add(idx >= 0 ? rel.slice(0, idx) : '')
    }
    if (prefixes.size === 1 && prefixes.has('1_data')) return 'flat'
    return 'mirror'
  }, [reg])
  // 增量模式必须沿用现有目录结构；全量模式会先清空，因此可选择新结构。
  const modeLocked = existingMode !== null && mode === 'incremental'

  // 「此轮需要生成」：按当前来源 + 模式 + 目标数推算点「开始生成」会处理多少张。
  // 与后端一致：覆盖(full)=清空重建 → need=目标数；增量(incremental)=补足到目标
  // → need=max(0, 目标−现有)。目标数：AI 先验=train 图数；Booru mirror=train 图数、
  // flat=目标数（留空=train 图数）。随左侧设置实时变。
  const thisRound = useMemo(() => {
    const current = reg?.image_count ?? 0
    let target: number
    let incremental: boolean
    if (source === 'ai') {
      target = trainImageCount
      incremental = aiIncremental
    } else {
      target = buildMode === 'mirror'
        ? trainImageCount
        : targetCount.trim() === ''
          ? trainImageCount
          : Math.max(0, Number(targetCount) || 0)
      incremental = mode === 'incremental'
    }
    const need = incremental ? Math.max(0, target - current) : target
    return { need, incremental }
  }, [source, aiIncremental, buildMode, targetCount, mode, trainImageCount, reg?.image_count])

  const activeRun = useMemo(() => {
    if (aiTask && (aiTask.status === 'running' || aiTask.status === 'pending')) {
      return { id: aiTask.id, sourceLabel: t('reg.sourceAi'), status: aiTask.status }
    }
    if (job && (job.status === 'running' || job.status === 'pending')) {
      return { id: job.id, sourceLabel: t('reg.sourceBooru'), status: job.status }
    }
    return null
  }, [aiTask, job, t])
  const draftSourceLabel = source === 'ai' ? t('reg.sourceAi') : t('reg.sourceBooru')
  const draftRangeLabel = thisRound.incremental
    ? t('reg.thisRoundIncremental')
    : t('reg.thisRoundFull')

  // 现有 reg 集存在时，把 buildMode 自动对齐它（避免切到 version 看到错的初始值）。
  // 用户点 disabled 的下拉看到 tooltip 提示「先清空」。
  useEffect(() => {
    if (existingMode && existingMode !== buildMode) setBuildMode(existingMode)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [existingMode])

  const toggleTag = (tag: string) => {
    setExcluded((prev) => {
      const next = new Set(prev)
      if (next.has(tag)) next.delete(tag)
      else next.add(tag)
      return next
    })
  }

  const handleAiGenerate = async () => {
    if (!vid) return
    if (trainImageCount <= 0) {
      toast(t('reg.noTrainForAi'), 'error')
      return
    }
    setSubmitting(true)
    setAiBusy(true)
    setAiTask(null)
    setAiLogs([])
    try {
      const body: RegAiRequest = {
        excluded_tags: Array.from(excluded),
        base_model: aiBaseModel ?? undefined,
        negative_prompt: aiNeg,
        width: aiWidth,
        height: aiHeight,
        steps: aiSteps,
        cfg_scale: aiCfg,
        seed: aiSeed,
        incremental: aiIncremental,
      }
      const task = await api.enqueueRegPrior(project.id, vid, body)
      setAiTask(task)
      toast(t('reg.aiEnqueued', { id: task.id }), 'success')
    } catch (e) {
      toast(String(e), 'error')
      setAiBusy(false)
    } finally {
      setSubmitting(false)
    }
  }

  const startBuild = async () => {
    if (!vid) return
    if (trainImageCount <= 0) {
      toast(t('reg.noTrainForBuild'), 'error')
      return
    }
    const incremental = mode === 'incremental'
    const parsedTarget = targetCount.trim() === '' ? null : Number(targetCount)
    const body: RegBuildRequest = {
      excluded_tags: Array.from(excluded),
      auto_tag: autoTag,
      auto_tag_kind: autoTagKind,
      api_source: apiSource,
      incremental,
      auto_dedup: autoDedup,
      build_mode: buildMode,
      target_count: parsedTarget,
      ...advanced,
    }
    setSubmitting(true)
    try {
      const j = await api.startRegBuild(project.id, vid, body)
      setJob(j)
      setLogs([])
      toast(t('reg.enqueued', { id: j.id }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setSubmitting(false)
    }
  }

  const startGeneration = async () => {
    const replacesExisting = source === 'ai' ? !aiIncremental : mode === 'full'
    if (replacesExisting && reg?.exists) {
      const sourceLabel = source === 'ai' ? t('reg.sourceAi') : t('reg.sourceBooru')
      const ok = await confirm(
        t('reg.confirmFullBuild', {
          current: reg.image_count,
          target: thisRound.need,
          source: sourceLabel,
        }),
        { tone: 'danger', okText: t('reg.confirmFullBuildOk') },
      )
      if (!ok) return
    }
    if (source === 'ai') await handleAiGenerate()
    else await startBuild()
  }

  const onDelete = async () => {
    if (!vid) return
    if (!(await confirm(t('reg.confirmDelete'), { tone: 'danger', okText: t('reg.deleteOkText') }))) return
    try {
      await api.deleteReg(project.id, vid)
      toast(t('reg.deleted'), 'success')
      setReg(null)
      await refreshReg()
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  // 预览：点击缩略图 → 加载该图 caption → 打开 modal
  const openPreview = useCallback(
    async (idx: number) => {
      if (!reg || !vid) return
      const path = reg.files[idx]
      setPreviewIdx(idx)
      setPreviewCaption(t('reg.captionLoading'))
      try {
        const r = await api.getRegCaption(project.id, vid, path)
        setPreviewCaption(r.tags.length ? r.tags.join(', ') : t('reg.captionEmpty'))
      } catch (e) {
        setPreviewCaption(t('reg.captionFailed', { error: String(e) }))
      }
    },
    [reg, vid, project.id, t]
  )

  if (!activeVersion || !vid) {
    return <p className="text-fg-tertiary p-6">{t('reg.noVersion')}</p>
  }

  return (
    <StepShell
      title={t('steps.reg.title')}
      subtitle={t('steps.reg.subtitle')}
      logSources={[
        job && {
          key: 'reg_build',
          label: t('logDrawer.regBuild'),
          status: job.status,
          lines: logs,
          startedAt: job.started_at,
          finishedAt: job.finished_at,
          onCancel: () => {
            void api
              .cancelJob(job.id)
              .then(() => toast(t('reg.cancelToast'), 'success'))
              .catch((e) => toast(String(e), 'error'))
          },
        },
        aiTask && {
          key: 'reg_ai',
          label: t('logDrawer.regPrior'),
          status: aiTask.status,
          lines: aiLogs,
          startedAt: aiTask.started_at,
          finishedAt: aiTask.finished_at,
          onCancel: () => {
            void api
              .cancelTask(aiTask.id)
              .then(() => toast(t('reg.cancelToast'), 'success'))
              .catch((e) => toast(String(e), 'error'))
          },
        },
      ]}
      actions={
        <ActionGroup
          aria-label={t('reg.pageActionsLabel')}
          secondary={reg?.exists && (
            <Button
              variant="danger"
              size="sm"
              onClick={() => void onDelete()}
              disabled={isLive}
              title={t('reg.deleteBtn')}
            >
              {t('reg.deleteBtn')}
            </Button>
          )}
          primary={(
            <Button
              variant="primary"
              size="sm"
              onClick={() => void startGeneration()}
              disabled={isLive || trainImageCount <= 0}
              loading={submitting}
            >
              {!submitting && (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                  <path d="M8 5v14l11-7z" />
                </svg>
              )}
              <span>
                {isTaskLive
                  ? t('reg.generatingBtn')
                  : source === 'ai'
                    ? t('reg.aiGenerateBtn')
                    : t('reg.startBuildBtn')}
              </span>
            </Button>
          )}
        />
      }
      belowHeader={(
        <Tabs
          items={[
            {
              value: 'generate',
              label: t('reg.tabGenerate'),
              controls: 'reg-panel-generate',
            },
            {
              value: 'images',
              label: reg?.image_count
                ? t('reg.tabImagesCount', { n: reg.image_count })
                : t('reg.tabImages'),
              controls: 'reg-panel-images',
            },
          ]}
          value={activeTab}
          onChange={setActiveTab}
          ariaLabel={t('reg.tabsLabel')}
          idPrefix="reg-tabs"
          className="w-full px-page"
        />
      )}
    >
    <div className="flex flex-col h-full gap-3 min-h-0">
      {activeTab === 'generate' ? (
        <div
          id="reg-panel-generate"
          role="tabpanel"
          aria-labelledby="reg-tabs-generate"
          className="grid flex-1 min-h-0 gap-3 overflow-y-auto xl:grid-cols-[minmax(0,3fr)_minmax(20rem,2fr)] xl:overflow-hidden"
        >
          <div className="flex min-w-0 flex-col xl:min-h-0 xl:overflow-y-auto">
            {activeRun && (
              <Alert
                tone="info"
                size="sm"
                title={t('reg.currentTaskTitle', {
                  source: activeRun.sourceLabel,
                  id: activeRun.id,
                })}
                className="mb-3"
              >
                {t('reg.nextRunDraftHint')}
              </Alert>
            )}

            <SourceSegmented source={source} onChange={setSource} />

            {trainTagsError && (
              <Alert
                tone="warning"
                size="sm"
                title={t('reg.trainTagsLoadFailedTitle')}
                action={(
                  <Button variant="secondary" size="sm" onClick={() => void refreshTrainTags()}>
                    {t('common.retry')}
                  </Button>
                )}
                className="mb-3"
              >
                {t('reg.trainTagsLoadFailedHint')}
              </Alert>
            )}

            {source === 'ai' ? (
              <AiForm
                trainTags={trainTags}
                trainTagsLoading={trainTagsLoading}
                trainTagsFailed={Boolean(trainTagsError)}
                excluded={excluded}
                onToggleExcluded={toggleTag}
                neg={aiNeg} onNegChange={setAiNeg}
                width={aiWidth} onWidthChange={setAiWidth}
                height={aiHeight} onHeightChange={setAiHeight}
                steps={aiSteps} onStepsChange={setAiSteps}
                cfg={aiCfg} onCfgChange={setAiCfg}
                seed={aiSeed} onSeedChange={setAiSeed}
                baseModel={aiBaseModel} onBaseModelChange={setAiBaseModel}
                family={aiFamily}
                incremental={aiIncremental}
                onIncrementalChange={setAiIncremental}
              />
            ) : (
              <BooruForm
                trainTags={trainTags}
                trainTagsLoading={trainTagsLoading}
                trainTagsFailed={Boolean(trainTagsError)}
                trainImageCount={trainImageCount}
                excluded={excluded}
                onToggleExcluded={toggleTag}
                apiSource={apiSource} onApiSourceChange={setApiSource}
                buildMode={buildMode} onBuildModeChange={setBuildMode}
                modeLocked={modeLocked}
                existingMode={existingMode}
                targetCount={targetCount} onTargetCountChange={setTargetCount}
                mode={mode} onModeChange={setMode}
                autoTag={autoTag} onAutoTagChange={setAutoTag}
                autoTagKind={autoTagKind} onAutoTagKindChange={setAutoTagKind}
                autoDedup={autoDedup} onAutoDedupChange={setAutoDedup}
                advanced={advanced} onAdvancedChange={setAdvanced}
              />
            )}
          </div>

          <aside className="flex min-w-0 flex-col gap-3 xl:min-h-0 xl:overflow-y-auto">
            <RegPlanPanel
              source={source}
              sourceLabel={draftSourceLabel}
              rangeLabel={draftRangeLabel}
              buildMode={buildMode}
              target={thisRound.need}
              excludedCount={excluded.size}
              isNextRun={Boolean(activeRun)}
            />
            <RegStatusPanel
              reg={reg}
              loading={regLoading}
              error={regError}
              autoTagKind={autoTagKind}
              onRetry={() => void refreshReg()}
            />
          </aside>
        </div>
      ) : (
        <div
          id="reg-panel-images"
          role="tabpanel"
          aria-labelledby="reg-tabs-images"
          className="flex flex-1 min-h-0 flex-col gap-3"
        >
          {reg && reg.image_count > 0 ? (
            <>
              {regError && (
                <Alert
                  tone="warning"
                  size="sm"
                  title={t('reg.refreshErrorTitle')}
                  action={(
                    <Button variant="secondary" size="sm" onClick={() => void refreshReg()}>
                      {t('common.retry')}
                    </Button>
                  )}
                >
                  {regError}
                </Alert>
              )}
              <RegPreview
                pid={project.id}
                vid={vid}
                reg={reg}
                isLive={isLive}
                onPick={(idx) => void openPreview(idx)}
                onDeleted={() => {
                  void refreshReg()
                  void reload()
                }}
              />
            </>
          ) : regError && !reg ? (
            <Alert
              tone="danger"
              title={t('reg.loadErrorTitle')}
              action={(
                <Button variant="secondary" size="sm" onClick={() => void refreshReg()}>
                  {t('common.retry')}
                </Button>
              )}
              className="w-full self-start"
              role="alert"
            >
              {regError}
            </Alert>
          ) : (
            <EmptyState
              className="flex-1"
              title={t('reg.emptyRegTitle')}
              description={regLoading ? t('common.loading') : t('reg.emptyRegHint')}
              action={!regLoading && (
                <Button variant="primary" size="sm" onClick={() => setActiveTab('generate')}>
                  {t('reg.goToGenerate')}
                </Button>
              )}
            />
          )}
        </div>
      )}

      {previewIdx !== null && reg && reg.files[previewIdx] && (
        <ImagePreviewModal
          src={regOrigUrl(project.id, vid, reg.files[previewIdx])}
          caption={previewCaption}
          index={previewIdx}
          total={reg.files.length}
          hasPrev={previewIdx > 0}
          hasNext={previewIdx < reg.files.length - 1}
          onClose={() => setPreviewIdx(null)}
          onPrev={() =>
            previewIdx > 0 ? void openPreview(previewIdx - 1) : undefined
          }
          onNext={() =>
            previewIdx < reg.files.length - 1
              ? void openPreview(previewIdx + 1)
              : undefined
          }
        />
      )}
    </div>
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// 子组件 — task-first Regularization 工作台（见 phase4b-regularization brief）
// ---------------------------------------------------------------------------

function RegPlanPanel({
  source,
  sourceLabel,
  rangeLabel,
  buildMode,
  target,
  excludedCount,
  isNextRun,
}: {
  source: 'ai' | 'booru'
  sourceLabel: string
  rangeLabel: string
  buildMode: 'mirror' | 'flat'
  target: number
  excludedCount: number
  isNextRun: boolean
}) {
  const { t } = useTranslation()
  return (
    <Card as="section" radius="compact" padding="sm" aria-labelledby="reg-plan-title">
      <PanelHeading
        id="reg-plan-title"
        title={t(isNextRun ? 'reg.nextRunPlanTitle' : 'reg.runPlanTitle')}
      />
      <div className="flex flex-col gap-2">
        <StatusRow label={t('reg.statusCellSource')}>
          <span className="font-mono">{sourceLabel}</span>
        </StatusRow>
        <StatusRow label={t('reg.modeLabel')}>
          <span>{rangeLabel}</span>
        </StatusRow>
        {source === 'booru' && (
          <StatusRow label={t('reg.buildModeLabel')}>
            <span>{t(buildMode === 'mirror' ? 'reg.buildModeMirror' : 'reg.buildModeFlat')}</span>
          </StatusRow>
        )}
        <StatusRow label={t('reg.statusCellThisRound')}>
          <span className="font-mono text-accent">
            {t('reg.nImages', { n: target })}
          </span>
        </StatusRow>
        <StatusRow label={t('reg.excludePlanLabel')}>
          <span className="font-mono">{excludedCount}</span>
        </StatusRow>
      </div>
    </Card>
  )
}

function RegStatusPanel({
  reg,
  loading,
  error,
  autoTagKind,
  onRetry,
}: {
  reg: RegStatus | null
  loading: boolean
  error: string | null
  autoTagKind: string
  onRetry: () => void
}) {
  const { t } = useTranslation()
  const header = <PanelHeading id="reg-status-title" title={t('reg.statusPanelTitle')} />

  if (!reg && error) {
    return (
      <Card as="section" radius="compact" padding="sm" aria-labelledby="reg-status-title">
        {header}
        <Alert
          tone="danger"
          size="sm"
          title={t('reg.loadErrorTitle')}
          action={(
            <Button variant="secondary" size="sm" onClick={onRetry}>
              {t('common.retry')}
            </Button>
          )}
          role="alert"
        >
          {error}
        </Alert>
      </Card>
    )
  }

  if (!reg) {
    return (
      <Card
        as="section"
        radius="compact"
        padding="sm"
        aria-labelledby="reg-status-title"
        aria-busy={loading || undefined}
      >
        {header}
        <p className="m-0 text-xs text-fg-tertiary">{t('reg.statusLoading')}</p>
      </Card>
    )
  }

  const m = reg.meta
  const sourceLabel = m
    ? m.generation_method === 'ai_base'
      ? t('reg.statusAiGen')
      : m.api_source
    : '—'
  const taggerLabel = m
    ? m.auto_tagged
      ? (m.auto_tag_kind ?? autoTagKind ?? 'wd14')
      : null
    : null

  return (
    <Card
      as="section"
      radius="compact"
      padding="sm"
      aria-labelledby="reg-status-title"
      aria-busy={loading || undefined}
    >
      {header}
      {error && (
        <Alert
          tone="warning"
          size="sm"
          title={t('reg.refreshErrorTitle')}
          action={(
            <Button variant="secondary" size="sm" onClick={onRetry}>
              {t('common.retry')}
            </Button>
          )}
          className="mb-3"
        >
          {error}
        </Alert>
      )}
      {!reg.exists ? (
        <p className="m-0 text-xs text-fg-tertiary">{t('reg.statusNotExist')}</p>
      ) : (
        <div className="flex flex-col gap-2">
          <StatusRow label={t('reg.statusCellSet')}>
            <span className="font-mono">
              <span className="text-ok">{reg.image_count}</span>
              {m && (
                <span className="ml-1 text-2xs font-normal text-fg-tertiary">
                  / {m.target_count} {t('reg.nImagesShort')}
                </span>
              )}
            </span>
          </StatusRow>
          <StatusRow label={t('reg.statusCellSource')}>
            <span className="font-mono">{sourceLabel}</span>
          </StatusRow>
          <StatusRow label={t('reg.statusCellTagger')}>
            <span className="font-mono">
              {taggerLabel
                ? <span className="text-ok">✓ {taggerLabel}</span>
                : <span className="text-fg-tertiary">{t('reg.statusTaggerOff')}</span>}
            </span>
          </StatusRow>
          <StatusRow label={t('reg.statusCellLatest')}>
            <span className="text-sm text-fg-primary">
              {m ? formatAgo(m.generated_at, t) : '—'}
            </span>
          </StatusRow>
        </div>
      )}
    </Card>
  )
}

function PanelHeading({ id, title }: { id: string; title: string }) {
  return (
    <div className="mb-2.5 flex items-center gap-1.5">
      <span className="inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
      <h2 id={id} className="m-0 type-panel-title">{title}</h2>
    </div>
  )
}

// 状态面板信息行：label 左 / value 右对齐。字号对齐任务详情页 OverviewTab
// （label text-sm fg-tertiary，value text-sm fg-primary），不用 text-2xs。
function StatusRow({
  label, children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-sm text-fg-tertiary font-normal shrink-0">
        {label}
      </span>
      <span className="text-sm text-fg-primary text-right min-w-0 break-words">
        {children}
      </span>
    </div>
  )
}

function handleRadioGroupKeyDown(event: KeyboardEvent<HTMLDivElement>) {
  if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return
  const options = Array.from(
    event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="radio"]:not(:disabled)'),
  )
  if (options.length === 0) return
  const current = options.indexOf(document.activeElement as HTMLButtonElement)
  let next = current
  if (event.key === 'Home') next = 0
  else if (event.key === 'End') next = options.length - 1
  else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
    next = (current - 1 + options.length) % options.length
  } else {
    next = (current + 1) % options.length
  }
  event.preventDefault()
  options[next].focus()
  options[next].click()
}

export function SourceSegmented({
  source, onChange,
}: {
  source: 'ai' | 'booru'
  onChange: (s: 'ai' | 'booru') => void
}) {
  const { t } = useTranslation()
  return (
    <Card as="section" radius="compact" padding="sm" className="mb-3.5">
      <h2 className="mb-2 type-panel-title">{t('reg.sourceLabel')}</h2>
      <div
        className="flex flex-wrap items-center gap-2"
        role="radiogroup"
        aria-label={t('reg.sourcePickerLabel')}
        onKeyDown={handleRadioGroupKeyDown}
      >
        <SourceRadio
          on={source === 'ai'}
          onClick={() => onChange('ai')}
          label={t('reg.sourceAi')}
          sub={t('reg.sourceAiSub')}
        />
        <SourceRadio
          on={source === 'booru'}
          onClick={() => onChange('booru')}
          label={t('reg.sourceBooru')}
          sub={t('reg.sourceBooruSub')}
        />
      </div>
      <p className="mb-0 mt-2 text-xs leading-relaxed text-fg-tertiary">
        {source === 'ai' ? t('reg.sourceAiHint') : t('reg.sourceBooruHint')}
      </p>
    </Card>
  )
}

function SourceRadio({
  on, onClick, label, sub,
}: {
  on: boolean
  onClick: () => void
  label: string
  sub: string
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={on}
      tabIndex={on ? 0 : -1}
      data-state={on ? 'active' : 'inactive'}
      onClick={onClick}
      className={`pill-radio pill-radio-content${on ? ' on' : ''}`}
    >
      <span className="pill-radio-dot" aria-hidden="true" />
      <span>{label}</span>
      <span className="text-2xs opacity-70">· {sub}</span>
    </button>
  )
}

// 分组卡（grp）：标题 + 标签 + 可选折叠
function GrpCard({
  title, tag, meta, collapsible, defaultOpen, children,
}: {
  title: string
  tag?: string
  meta?: React.ReactNode
  collapsible?: boolean
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(!collapsible || defaultOpen !== false)
  const contentId = useId()
  const heading = (
    <>
      <span className="type-panel-title">{title}</span>
      {tag && (
        <span className="badge badge-info">{tag}</span>
      )}
      {meta && <span className="text-xs text-fg-tertiary">{meta}</span>}
      {collapsible && (
        <span className="ml-auto inline-flex items-center gap-2 text-xs text-fg-tertiary">
          <span>{open ? t('reg.grpCollapse') : t('reg.grpExpand')}</span>
          <span
            aria-hidden="true"
            className="inline-block transition-transform"
            style={{ transform: open ? 'rotate(90deg)' : undefined }}
          >
            ›
          </span>
        </span>
      )}
    </>
  )

  return (
    <Card radius="compact" className="mb-3.5 overflow-hidden">
      {collapsible ? (
        <button
          type="button"
          className="flex w-full items-center gap-2.5 bg-transparent px-4 py-3 text-left text-fg-primary hover:bg-overlay"
          aria-expanded={open}
          aria-controls={contentId}
          onClick={() => setOpen((value) => !value)}
        >
          {heading}
        </button>
      ) : (
        <div className="flex items-center gap-2.5 px-4 py-3">{heading}</div>
      )}
      {open && (
        <div id={contentId} className="border-t border-subtle px-4 pb-4 pt-1">
          {children}
        </div>
      )}
    </Card>
  )
}

// 与训练配置页 components/Field.tsx 的 inputStyle 同款（紧凑；canvas 背景）。
const fieldInputStyle: React.CSSProperties = {
  width: '100%', padding: '5px 10px',
  background: 'var(--bg-canvas)', border: '1px solid var(--border-default)',
  borderRadius: 'var(--r-sm)', fontSize: 'var(--t-sm)',
  color: 'var(--fg-primary)',
}

// 单字段封装：对齐训练配置页 / 打标页 Field 语言 — 静态说明放 label 旁 ⓘ tooltip
// （helpTooltip，不占控件下方空间）；控件下方只留必须常驻的动态提示（hint 如锁定
// 警示 / locked）。
function Field({
  label, htmlFor, helpTooltip, hint, locked, children,
}: {
  label: React.ReactNode
  htmlFor?: string
  /** 静态说明 → label 旁 ⓘ 点开弹层（对齐打标页 TagField.helpTooltip）。 */
  helpTooltip?: React.ReactNode
  /** 控件下方常驻：留给动态状态 / 锁定警示等必须一直可见的信息。 */
  hint?: React.ReactNode
  locked?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="py-1.5">
      <div className="flex items-center gap-2 text-sm font-medium text-fg-secondary mb-1">
        {htmlFor
          ? <label htmlFor={htmlFor}>{label}</label>
          : <span>{label}</span>}
        {helpTooltip && <InfoButton>{helpTooltip}</InfoButton>}
      </div>
      {children}
      {hint && <div className="text-xs text-fg-tertiary mt-1">{hint}</div>}
      {locked && (
        <div className="font-mono text-2xs text-fg-tertiary mt-1">
          {locked}
        </div>
      )}
    </div>
  )
}

// AI 表单 — grp 卡：出图（常用）/ 排除 tag / 采样（进阶）
function AiForm({
  trainTags, trainTagsLoading, trainTagsFailed,
  excluded, onToggleExcluded,
  neg, onNegChange,
  width, onWidthChange,
  height, onHeightChange,
  steps, onStepsChange,
  cfg, onCfgChange,
  seed, onSeedChange,
  baseModel, onBaseModelChange, family,
  incremental, onIncrementalChange,
}: {
  trainTags: RegTagCount[]
  trainTagsLoading: boolean
  trainTagsFailed: boolean
  excluded: Set<string>
  onToggleExcluded: (tag: string) => void
  neg: string
  onNegChange: (v: string) => void
  width: number; onWidthChange: (v: number) => void
  height: number; onHeightChange: (v: number) => void
  steps: number; onStepsChange: (v: number) => void
  cfg: number; onCfgChange: (v: number) => void
  seed: number; onSeedChange: (v: number) => void
  baseModel: string | null; onBaseModelChange: (v: string) => void
  /** version 训练配置声明的模型族（底模列表按族列） */
  family: 'anima' | 'krea2'
  incremental: boolean; onIncrementalChange: (v: boolean) => void
}) {
  const { t } = useTranslation()
  return (
    <>
      {/* 正则集设置：只留「模式」（覆盖 / 增量） */}
      <GrpCard title={t('reg.grpRegSettings')}>
        <Field
          label={t('reg.modeLabel')}
          htmlFor="reg-ai-mode"
          helpTooltip={t('reg.modeHintAi')}
        >
          <Select
            id="reg-ai-mode"
            controlSize="sm"
            surface="canvas"
            value={incremental ? 'incremental' : 'full'}
            onChange={(e) => onIncrementalChange(e.target.value === 'incremental')}
          >
            <option value="incremental">{t('reg.modeIncrementalAi')}</option>
            <option value="full">{t('reg.modeFullAi')}</option>
          </Select>
        </Field>
      </GrpCard>

      {/* 出图设置：负面 + 宽高 + 采样参数（原「采样进阶」改名，吸收负面/宽高）。
          放「排除 train tag」上面，默认折叠。 */}
      <GrpCard title={t('reg.grpImageSettings')} collapsible defaultOpen={false}>
        <Field label={t('reg.negPrompt')} htmlFor="reg-ai-negative">
          <Textarea
            id="reg-ai-negative"
            controlSize="sm"
            surface="canvas"
            className="font-mono"
            rows={3}
            value={neg}
            onChange={(e) => onNegChange(e.target.value)}
          />
        </Field>
        <div className="grid grid-cols-2 gap-3.5">
          <Field label={t('reg.widthLabel')} htmlFor="reg-ai-width">
            <UnitInput
              id="reg-ai-width"
              value={width}
              onChange={onWidthChange}
              unit="px"
              min={256}
              max={4096}
              step={64}
            />
          </Field>
          <Field label={t('reg.heightLabel')} htmlFor="reg-ai-height">
            <UnitInput
              id="reg-ai-height"
              value={height}
              onChange={onHeightChange}
              unit="px"
              min={256}
              max={4096}
              step={64}
            />
          </Field>
        </div>
        <div className="grid grid-cols-3 gap-3.5">
          <Field label={t('reg.stepsLabel')} htmlFor="reg-ai-steps">
            <Input
              id="reg-ai-steps"
              type="number"
              controlSize="sm"
              surface="canvas"
              className="font-mono"
              value={steps}
              onChange={(e) => onStepsChange(Number(e.target.value) || 0)}
              min={1} max={150}
            />
          </Field>
          <Field label="CFG Scale" htmlFor="reg-ai-cfg">
            <Input
              id="reg-ai-cfg"
              type="number"
              controlSize="sm"
              surface="canvas"
              className="font-mono"
              value={cfg}
              onChange={(e) => onCfgChange(Number(e.target.value) || 0)}
              min={0} max={20} step={0.5}
            />
          </Field>
          <Field
            label={t('reg.seedLabel')}
            htmlFor="reg-ai-seed"
            helpTooltip={t('reg.seedHintRandom')}
          >
            <Input
              id="reg-ai-seed"
              type="number"
              controlSize="sm"
              surface="canvas"
              className="font-mono"
              value={seed}
              onChange={(e) => onSeedChange(Number(e.target.value) || 0)}
              min={0}
            />
          </Field>
        </div>
        <Field
          label={t('reg.baseModelLabel')}
          helpTooltip={t('reg.baseModelHint')}
        >
          <BaseModelSelect
            value={baseModel}
            onChange={onBaseModelChange}
            family={family}
            className="select input"
            style={fieldInputStyle}
            ariaLabel={t('reg.baseModelLabel')}
          />
        </Field>
      </GrpCard>

      <ExcludeTags
        trainTags={trainTags}
        loading={trainTagsLoading}
        failed={trainTagsFailed}
        excluded={excluded}
        onToggle={onToggleExcluded}
      />
    </>
  )
}

// Booru 表单 — grp 卡：抓取（常用）/ 排除 tag / 进阶
function BooruForm({
  trainTags, trainTagsLoading, trainTagsFailed, trainImageCount,
  excluded, onToggleExcluded,
  apiSource, onApiSourceChange,
  buildMode, onBuildModeChange, modeLocked, existingMode,
  targetCount, onTargetCountChange,
  mode, onModeChange,
  autoTag, onAutoTagChange,
  autoTagKind, onAutoTagKindChange,
  autoDedup, onAutoDedupChange,
  advanced, onAdvancedChange,
}: {
  trainTags: RegTagCount[]
  trainTagsLoading: boolean
  trainTagsFailed: boolean
  trainImageCount: number
  excluded: Set<string>
  onToggleExcluded: (tag: string) => void
  apiSource: 'gelbooru' | 'danbooru'
  onApiSourceChange: (v: 'gelbooru' | 'danbooru') => void
  buildMode: 'mirror' | 'flat'
  onBuildModeChange: (v: 'mirror' | 'flat') => void
  modeLocked: boolean
  existingMode: 'mirror' | 'flat' | null
  targetCount: string
  onTargetCountChange: (v: string) => void
  mode: 'full' | 'incremental'
  onModeChange: (v: 'full' | 'incremental') => void
  autoTag: boolean
  onAutoTagChange: (v: boolean) => void
  autoTagKind: 'wd14' | 'cltagger'
  onAutoTagKindChange: (v: 'wd14' | 'cltagger') => void
  autoDedup: boolean
  onAutoDedupChange: (v: boolean) => void
  advanced: AdvancedParams
  onAdvancedChange: (v: AdvancedParams) => void
}) {
  const { t } = useTranslation()
  const mirror = buildMode === 'mirror'
  return (
    <>
      {/* 正则集设置：来源 / 构建模式 / 目标数 / 增量模式（去掉「常用」标签，
          自动打标移到「正则集处理」卡） */}
      <GrpCard title={t('reg.grpRegSettings')}>
        <div className="grid grid-cols-2 gap-3.5">
          <Field label={t('reg.source')} htmlFor="reg-booru-source">
            <Select
              id="reg-booru-source"
              controlSize="sm"
              surface="canvas"
              value={apiSource}
              onChange={(e) => onApiSourceChange(e.target.value as 'gelbooru' | 'danbooru')}
            >
              <option value="gelbooru">Gelbooru</option>
              <option value="danbooru">Danbooru</option>
            </Select>
          </Field>
          <Field
            label={t('reg.buildModeLabel')}
            htmlFor="reg-booru-structure"
            hint={modeLocked ? t('reg.buildModeLocked', { mode: existingMode }) : undefined}
          >
            <Select
              id="reg-booru-structure"
              controlSize="sm"
              surface="canvas"
              value={buildMode}
              onChange={(e) => onBuildModeChange(e.target.value as 'mirror' | 'flat')}
              disabled={modeLocked}
            >
              <option value="flat">{t('reg.buildModeFlat')}</option>
              <option value="mirror">{t('reg.buildModeMirror')}</option>
            </Select>
          </Field>
        </div>
        <div className="grid grid-cols-2 gap-3.5">
          <Field
            label={t('reg.targetCount')}
            htmlFor="reg-booru-target"
            helpTooltip={t('reg.targetCountHint')}
            locked={mirror ? t('reg.targetMirrorLocked', { n: trainImageCount }) : undefined}
          >
            <Input
              id="reg-booru-target"
              type="number"
              controlSize="sm"
              surface="canvas"
              className="font-mono"
              value={mirror ? String(trainImageCount) : targetCount}
              onChange={(e) => onTargetCountChange(e.target.value)}
              placeholder={String(trainImageCount)}
              disabled={mirror}
              min={1}
            />
          </Field>
          <Field
            label={t('reg.modeLabel')}
            htmlFor="reg-booru-mode"
            helpTooltip={t('reg.modeHintBooru')}
          >
            <Select
              id="reg-booru-mode"
              controlSize="sm"
              surface="canvas"
              value={mode}
              onChange={(e) => onModeChange(e.target.value as 'full' | 'incremental')}
            >
              <option value="incremental">{t('reg.modeIncrementalBooru')}</option>
              <option value="full">{t('reg.modeFullBooru')}</option>
            </Select>
          </Field>
        </div>
      </GrpCard>

      {/* 正则集处理：自动打标 + 打标模型（勾选后才显示）+ 长宽比 / 裁剪 / 去重。
          默认折叠，放「排除 train tag」前面。 */}
      <GrpCard title={t('reg.grpRegProcessing')} collapsible defaultOpen={false}>
        <CheckRow
          checked={autoTag}
          onChange={onAutoTagChange}
          label={t('reg.autoTagLabel')}
        />
        {autoTag && (
          <Field label={t('reg.autoTagKindLabel')} htmlFor="reg-booru-tagger">
            <Select
              id="reg-booru-tagger"
              controlSize="sm"
              surface="canvas"
              value={autoTagKind}
              onChange={(e) => onAutoTagKindChange(e.target.value as 'wd14' | 'cltagger')}
            >
              <option value="wd14">WD14</option>
              <option value="cltagger">CLTagger</option>
            </Select>
          </Field>
        )}
        <AdvancedFields
          value={advanced}
          onChange={onAdvancedChange}
          autoDedup={autoDedup}
          onAutoDedupChange={onAutoDedupChange}
        />
      </GrpCard>

      <ExcludeTags
        trainTags={trainTags}
        loading={trainTagsLoading}
        failed={trainTagsFailed}
        excluded={excluded}
        onToggle={onToggleExcluded}
        modeHint={t('reg.excludeHintBooru')}
      />
    </>
  )
}

// 数字输入 + 单位后缀（px）
function UnitInput({
  id, value, onChange, unit, min, max, step,
}: {
  id: string
  value: number
  onChange: (v: number) => void
  unit: string
  min?: number; max?: number; step?: number
}) {
  return (
    <div className="relative">
      <Input
        id={id}
        type="number"
        controlSize="sm"
        surface="canvas"
        className="font-mono pr-9"
        value={value}
        onChange={(e) => onChange(Number(e.target.value) || 0)}
        min={min} max={max} step={step}
      />
      <span className="absolute right-3 top-1/2 -translate-y-1/2 font-mono text-2xs text-fg-tertiary pointer-events-none">
        {unit}
      </span>
    </div>
  )
}

// checkbox 块：checkbox 左 + label，说明放 label 旁 ⓘ tooltip（helpTooltip）。
// ⓘ 放在 <label> 之外（同排 sibling），点它只开弹层、不会误触发勾选。
function CheckRow({
  checked, onChange, label, helpTooltip,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label: string
  helpTooltip?: React.ReactNode
}) {
  return (
    <div className="flex items-center gap-2 py-1.5">
      <label className="flex items-center gap-3 cursor-pointer select-none">
        <Checkbox
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="cursor-pointer"
        />
        <span className="text-sm text-fg-primary">{label}</span>
      </label>
      {helpTooltip && <InfoButton>{helpTooltip}</InfoButton>}
    </div>
  )
}

// 排除 tag — train 高频 tag 一栏列出 + 自定义排除一栏
export function ExcludeTags({
  trainTags, loading, failed = false, excluded, onToggle, modeHint,
}: {
  trainTags: RegTagCount[]
  loading: boolean
  failed?: boolean
  excluded: Set<string>
  onToggle: (tag: string) => void
  modeHint?: string
}) {
  const { t } = useTranslation()
  const [draft, setDraft] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const suggest = useTagSuggest({
    value: draft,
    inputRef,
    wholeAsToken: true,
    // 选中候选时把整段 draft 替换；用户再按 Enter 走 addCustom 走 normalize → 落 booru 形态
    onPick: ({ suggestion }) => { setDraft(suggestion.tag) },
  })
  const trainTagSet = useMemo(
    () => new Set(trainTags.map((t) => t.tag)),
    [trainTags]
  )
  const customTags = useMemo(
    () => Array.from(excluded).filter((t) => !trainTagSet.has(t)).sort(),
    [excluded, trainTagSet]
  )
  const excludedCount = excluded.size
  const addCustom = () => {
    const items = draft
      .split(/[,，\n]+/)
      .map(normalizeRegTag)
      .filter(Boolean)
    if (items.length === 0) return
    for (const tag of items) {
      if (!excluded.has(tag)) onToggle(tag)
    }
    setDraft('')
  }

  return (
    <GrpCard
      title={t('reg.excludeTitle')}
      meta={
        <>
          {t('reg.excludeMetaPrefix')}{' '}
          <b style={{ color: 'var(--accent)' }}>{excludedCount}</b>
          {modeHint && <span className="ml-1">· {modeHint}</span>}
        </>
      }
      collapsible
      defaultOpen
    >
      {loading ? (
        <p className="m-0 text-xs text-fg-tertiary" aria-live="polite">
          {t('common.loading')}
        </p>
      ) : failed ? (
        <p className="m-0 text-xs text-fg-tertiary">{t('reg.excludeTagsUnavailable')}</p>
      ) : trainTags.length > 0 ? (
        <div className="flex flex-wrap gap-1.5">
          {trainTags.map((info) => {
            const on = excluded.has(info.tag)
            return (
              <button
                key={info.tag}
                type="button"
                aria-pressed={on}
                onClick={() => onToggle(info.tag)}
                className={
                  'inline-flex items-center gap-1.5 h-6 max-w-full overflow-hidden whitespace-nowrap px-2.5 rounded-md font-mono text-xs cursor-pointer transition-colors border ' +
                  (on
                    ? 'text-accent-hover'
                    : 'bg-sunken text-fg-secondary hover:text-fg-primary')
                }
                style={
                  on
                    ? { background: 'var(--accent-soft)', borderColor: 'rgba(237,107,58,0.42)' }
                    : { borderColor: 'var(--border-default)' }
                }
                title={on ? t('reg.excludeUnclick') : t('reg.excludeClick')}
              >
                <span className={`shrink-0 ${on ? 'text-accent' : 'text-fg-tertiary'}`}>
                  {on ? '✕' : '+'}
                </span>
                <span
                  className="min-w-0 truncate text-left"
                  title={info.tag.replace(/_/g, ' ')}
                >
                  <TranslatedTag tag={info.tag.replace(/_/g, ' ')} />
                </span>
                <span className="shrink-0 text-fg-disabled text-2xs">×{info.count}</span>
              </button>
            )
          })}
        </div>
      ) : (
        <p className="m-0 text-xs text-fg-tertiary">{t('reg.excludeNoTags')}</p>
      )}
      <div className="font-mono text-2xs uppercase tracking-wider text-fg-tertiary mt-4 mb-2 flex items-center gap-2">
        <span>{t('reg.tierCustom')}</span>
        <span className="flex-1 h-px bg-subtle" />
      </div>
      {customTags.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-2.5">
          {customTags.map((tag) => (
            <span
              key={tag}
              className="inline-flex items-center gap-1.5 h-6 max-w-full overflow-hidden whitespace-nowrap px-2.5 rounded-md font-mono text-xs border"
              style={{
                background: 'var(--warn-soft)',
                borderColor: 'rgba(224,162,58,0.4)',
                color: 'var(--warn)',
              }}
            >
              <span
                className="min-w-0 truncate text-left"
                title={tag.replace(/_/g, ' ')}
              >
                <TranslatedTag tag={tag.replace(/_/g, ' ')} />
              </span>
              <button
                type="button"
                onClick={() => onToggle(tag)}
                className="shrink-0 bg-transparent border-none cursor-pointer p-0 text-warn opacity-80"
                aria-label={t('reg.excludeCustomRemoveAria', { tag })}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
      <div className="flex gap-2 mt-3">
        <div className="relative flex-1">
          <label htmlFor="reg-custom-exclude" className="sr-only">
            {t('reg.excludeCustomTitle')}
          </label>
          <Input
            id="reg-custom-exclude"
            ref={inputRef}
            controlSize="sm"
            surface="canvas"
            className="w-full font-mono"
            value={draft}
            onChange={(e) => { setDraft(e.target.value); suggest.notifyChange() }}
            onKeyDown={(e) => {
              if (suggest.handleKeyDown(e)) return
              if (e.key === 'Enter') {
                e.preventDefault()
                addCustom()
              }
            }}
            onClick={() => suggest.notifyClick()}
            onFocus={() => suggest.notifyFocus()}
            onBlur={() => suggest.notifyBlur()}
            placeholder={t('reg.excludePlaceholder')}
          />
          <TagSuggestList
            open={suggest.open}
            suggestions={suggest.suggestions}
            activeIdx={suggest.activeIdx}
            onPick={(s) => suggest.pickAt(suggest.suggestions.indexOf(s))}
            onHover={suggest.setActiveIdx}
            inputRef={inputRef}
            cursor={suggest.cursor}
            positionDeps={[draft]}
          />
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={addCustom}
          disabled={!draft.trim()}
        >
          {t('reg.excludeAdd')}
        </Button>
      </div>
    </GrpCard>
  )
}

// Booru 进阶里的 PP5.5 后处理 + 长宽比 — 设计稿没强调但功能保留
function AdvancedFields({
  value, onChange, autoDedup, onAutoDedupChange,
}: {
  value: AdvancedParams
  onChange: (v: AdvancedParams) => void
  autoDedup: boolean
  onAutoDedupChange: (v: boolean) => void
}) {
  const { t } = useTranslation()
  const set = <K extends keyof AdvancedParams>(k: K, v: AdvancedParams[K]) =>
    onChange({ ...value, [k]: v })
  return (
    <>
      {/* 长宽比过滤：「启用」勾选与标题合并成单个「长宽比过滤」CheckRow */}
      <div>
        <CheckRow
          checked={value.aspect_ratio_filter_enabled}
          onChange={(v) => set('aspect_ratio_filter_enabled', v)}
          label={t('reg.aspectFilter')}
          helpTooltip={t('reg.aspectFilterHint')}
        />
        {value.aspect_ratio_filter_enabled && (
          <div className="grid grid-cols-2 gap-3.5 mt-1 mb-1.5">
            <Input
              aria-label={t('reg.minAspectLabel')}
              type="number"
              controlSize="sm"
              surface="canvas"
              className="font-mono"
              min={0.1} max={1} step={0.05}
              value={value.min_aspect_ratio}
              onChange={(e) =>
                set('min_aspect_ratio', Math.max(0.1, Math.min(1, Number(e.target.value) || 0.5)))
              }
            />
            <Input
              aria-label={t('reg.maxAspectLabel')}
              type="number"
              controlSize="sm"
              surface="canvas"
              className="font-mono"
              min={1} max={10} step={0.1}
              value={value.max_aspect_ratio}
              onChange={(e) =>
                set('max_aspect_ratio', Math.max(1, Math.min(10, Number(e.target.value) || 2)))
              }
            />
          </div>
        )}
      </div>
      {/* 聚类裁剪算法（原「后处理」） */}
      <Field label={t('reg.postprocess')} htmlFor="reg-postprocess-method">
        <div className="grid grid-cols-2 gap-3.5">
          <Select
            id="reg-postprocess-method"
            controlSize="sm"
            surface="canvas"
            value={value.postprocess_method}
            onChange={(e) => set('postprocess_method', e.target.value as 'smart' | 'stretch' | 'crop')}
          >
            <option value="smart">{t('reg.postprocessSmart')}</option>
            <option value="stretch">{t('reg.postprocessStretch')}</option>
            <option value="crop">{t('reg.postprocessCrop')}</option>
          </Select>
          <Input
            aria-label={t('reg.maxCropLabel')}
            type="number"
            controlSize="sm"
            surface="canvas"
            className="font-mono"
            min={0.05} max={0.5} step={0.05}
            value={value.postprocess_max_crop_ratio}
            onChange={(e) =>
              set('postprocess_max_crop_ratio', Math.max(0.05, Math.min(0.5, Number(e.target.value) || 0.1)))
            }
            title={t('reg.maxCropTitle')}
          />
        </div>
      </Field>
      {/* 去重：拉取时跳跃去重（skip_similar）→ 拉取后算法去重（autoDedup），相邻并列 */}
      <CheckRow
        checked={value.skip_similar}
        onChange={(v) => set('skip_similar', v)}
        label={t('reg.skipSimilarLabel')}
        helpTooltip={t('reg.skipSimilarTitle')}
      />
      <CheckRow
        checked={autoDedup}
        onChange={onAutoDedupChange}
        label={t('reg.autoDedupLabel')}
        helpTooltip={t('reg.autoDedupSub')}
      />
    </>
  )
}

function RegPreview({
  pid,
  vid,
  reg,
  isLive,
  onPick,
  onDeleted,
}: {
  pid: number
  vid: number
  reg: RegStatus
  isLive: boolean
  onPick: (idx: number) => void
  onDeleted: () => void
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { confirm } = useDialog()
  // reg.files 是相对 reg/ 的路径（含子文件夹镜像 train，例如 "5_concept/2001.png"）
  const allItems = useMemo(
    () =>
      reg.files.map((rel) => {
        const idx = rel.lastIndexOf('/')
        const folder = idx >= 0 ? rel.slice(0, idx) : ''
        const name = idx >= 0 ? rel.slice(idx + 1) : rel
        return {
          name: rel,
          folder,
          thumbUrl: api.versionThumbUrl(pid, vid, 'reg', name, folder),
        }
      }),
    [reg.files, pid, vid]
  )
  // A1 — 按子文件夹分 tab。"" 视作根（reg/ 直接子文件，无子目录的老 build 才有）。
  // 排序：保留出现顺序（builder 按 train 子文件夹排），但根放最末（通常空）。
  const folders = useMemo(() => {
    const seen = new Set<string>()
    const order: string[] = []
    for (const it of allItems) {
      if (!seen.has(it.folder)) {
        seen.add(it.folder)
        order.push(it.folder)
      }
    }
    order.sort((a, b) => {
      if (a === '' && b !== '') return 1
      if (b === '' && a !== '') return -1
      return a.localeCompare(b)
    })
    return order
  }, [allItems])
  const folderCounts = useMemo(() => {
    const m = new Map<string, number>()
    for (const it of allItems) m.set(it.folder, (m.get(it.folder) ?? 0) + 1)
    return m
  }, [allItems])
  // null = 全部；否则限定到该 folder
  const [activeFolder, setActiveFolder] = useState<string | null>(null)
  const items = useMemo(
    () =>
      activeFolder === null
        ? allItems
        : allItems.filter((it) => it.folder === activeFolder),
    [allItems, activeFolder]
  )
  const names = useMemo(() => items.map((it) => it.name), [items])
  // indexByName 用 allItems 的全局索引：onPick 走的是主组件 reg.files 的下标
  const allIndexByName = useMemo(() => {
    const m = new Map<string, number>()
    allItems.forEach((it, i) => m.set(it.name, i))
    return m
  }, [allItems])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [anchor, setAnchor] = useState<string | null>(null)
  // 切 tab：清空选择 + anchor。多选只在当前 tab 范围内生效。
  useEffect(() => {
    setSelected(new Set())
    setAnchor(null)
  }, [activeFolder])
  // reg.files 变化（删除完 refreshReg 后）：把已不存在的 name 从 selected 清掉
  useEffect(() => {
    const fileSet = new Set(allItems.map((it) => it.name))
    setSelected((prev) => {
      let changed = false
      const next = new Set<string>()
      for (const n of prev) {
        if (fileSet.has(n)) next.add(n)
        else changed = true
      }
      return changed ? next : prev
    })
  }, [allItems])

  const openByName = (name: string) => {
    const i = allIndexByName.get(name)
    if (i !== undefined) onPick(i)
  }

  const onDelete = async () => {
    if (selected.size === 0) return
    const ok = await confirm(
      t('reg.confirmDeleteFiles', { n: selected.size }),
      { tone: 'danger', okText: t('reg.deleteOkText') }
    )
    if (!ok) return
    try {
      const r = await api.deleteRegFiles(pid, vid, Array.from(selected))
      toast(t('reg.deleteFilesDone', { n: r.count }), 'success')
      setSelected(new Set())
      setAnchor(null)
      onDeleted()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  // A4 — 自动去重：用默认参数扫，把每组里的"推荐删除"项直接删，没 review panel。
  // reg 集 quality bar 比 train 低，不需要逐组人工选保留。
  const [dedupBusy, setDedupBusy] = useState(false)
  const onDedup = async () => {
    if (dedupBusy || isLive) return
    const ok = await confirm(t('reg.confirmDedup'), {
      tone: 'danger', okText: t('reg.dedupOkText'),
    })
    if (!ok) return
    setDedupBusy(true)
    try {
      const r = await api.dedupPurgeReg(pid, vid)
      toast(
        t('reg.dedupDone', {
          scanned: r.scanned, groups: r.groups, deleted: r.count,
        }),
        'success'
      )
      setSelected(new Set())
      setAnchor(null)
      onDeleted()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setDedupBusy(false)
    }
  }

  return (
    <Card as="section" radius="compact" padding="sm" className="flex flex-1 min-h-0 flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2 border-b border-subtle pb-2">
        <div
          className="flex min-w-0 flex-wrap items-center gap-1"
          role="radiogroup"
          aria-label={t('reg.folderFilterLabel')}
          onKeyDown={handleRadioGroupKeyDown}
        >
          <RegFolderTab
            label={t('reg.folderAll')}
            count={allItems.length}
            active={activeFolder === null}
            onClick={() => setActiveFolder(null)}
          />
          {folders.map((folder) => (
            <RegFolderTab
              key={folder || '__root__'}
              label={folder || t('reg.folderRoot')}
              count={folderCounts.get(folder) ?? 0}
              active={activeFolder === folder}
              onClick={() => setActiveFolder(folder)}
            />
          ))}
        </div>
        <span className="min-w-2 flex-1" aria-hidden="true" />
        <ActionGroup
          status={selected.size > 0 && (
            <span className="pr-1 text-xs text-accent">
              {t('reg.regPreviewSelected', { n: selected.size })}
            </span>
          )}
          secondary={(
            <Button
              variant="secondary"
              size="sm"
              onClick={() => void onDedup()}
              disabled={dedupBusy || isLive}
              loading={dedupBusy}
              title={t('reg.dedupTitle')}
            >
              {t('reg.dedupBtn')}
            </Button>
          )}
          primary={(
            <Button
              variant="danger"
              size="sm"
              onClick={() => void onDelete()}
              disabled={selected.size === 0 || isLive || dedupBusy}
              title={t('reg.deleteFilesTitle')}
            >
              {t('reg.deleteFilesBtn', { n: selected.size })}
            </Button>
          )}
        />
      </div>

      <div className="flex-1 min-h-0 flex flex-col">
        <p className="text-2xs text-fg-tertiary px-1 pb-1 m-0">
          {t('reg.regPreviewTitle', { n: items.length })}
        </p>
        <ImageGrid
          items={items}
          selected={selected}
          onSelect={(name, e) => {
            const r = applySelection(selected, name, e, names, anchor)
            setSelected(r.next)
            setAnchor(r.anchor)
          }}
          onActivate={openByName}
          onPreview={openByName}
          clickMode="activate"
          ariaLabel={t('reg.imageGridLabel')}
          className="flex-1 min-h-0"
        />
      </div>
    </Card>
  )
}

function RegFolderTab({
  label, count, active, onClick,
}: {
  label: string
  count: number
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      tabIndex={active ? 0 : -1}
      data-state={active ? 'active' : 'inactive'}
      onClick={onClick}
      className={[
        'rounded-full px-2 py-0.5 text-xs font-medium transition-colors',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-1',
        active
          ? 'bg-accent text-white'
          : 'bg-overlay text-fg-secondary hover:bg-accent-soft',
      ].join(' ')}
    >
      <span className="font-mono">{label}</span>
      <span className="ml-1 opacity-70">{count}</span>
    </button>
  )
}

function regOrigUrl(pid: number, vid: number, rel: string): string {
  const idx = rel.lastIndexOf('/')
  const folder = idx >= 0 ? rel.slice(0, idx) : ''
  const name = idx >= 0 ? rel.slice(idx + 1) : rel
  // 768px 预览（与 PP3 alt-hover 同尺寸）
  return api.versionThumbUrl(pid, vid, 'reg', name, folder, 768)
}

function formatAgo(unix: number, t: TFunction): string {
  const now = Date.now() / 1000
  const dt = now - unix
  if (dt < 60) return t('reg.agoJustNow')
  if (dt < 3600) return t('reg.agoMinutes', { n: Math.floor(dt / 60) })
  if (dt < 86400) return t('reg.agoHours', { n: Math.floor(dt / 3600) })
  return t('reg.agoDays', { n: Math.floor(dt / 86400) })
}
