import type { TFunction } from 'i18next'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
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
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import JobProgress from '../../../components/JobProgress'
import StepShell from '../../../components/StepShell'
import { useDialog } from '../../../components/Dialog'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'

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

export default function RegularizationPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()

  const [reg, setReg] = useState<RegStatus | null>(null)
  const [trainTags, setTrainTags] = useState<RegTagCount[]>([])
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

  const [job, setJob] = useState<Job | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const jobIdRef = useRef<number | null>(null)
  jobIdRef.current = job?.id ?? null

  // B2（PR-2）：「设置 & 日志」+「先验生成」合并成单 tab「生成」；顶部 source picker
  // 决定渲染 Booru 配置面板还是 AI 配置面板。「开始生成」按钮按 source 调对应 endpoint。
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
  const [aiIncremental, setAiIncremental] = useState(false)
  const [aiTask, setAiTask] = useState<Task | null>(null)
  const [aiLogs, setAiLogs] = useState<string[]>([])
  const [aiBusy, setAiBusy] = useState(false)
  const aiTaskIdRef = useRef<number | null>(null)
  aiTaskIdRef.current = aiTask?.id ?? null

  // 预览 modal
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)
  const [previewCaption, setPreviewCaption] = useState<string>('')

  const vid = activeVersion?.id ?? null

  const refreshReg = useCallback(async () => {
    if (!vid) return
    try {
      const s = await api.getRegStatus(project.id, vid)
      setReg(s)
    } catch (e) {
      toast(t('reg.loadFailed', { error: String(e) }), 'error')
    }
  }, [project.id, vid, t, toast])

  const refreshTrainTags = useCallback(async () => {
    if (!vid) return
    try {
      const items = await api.previewRegTags(project.id, vid, 30)
      setTrainTags(items)
    } catch {
      setTrainTags([])
    }
  }, [project.id, vid])

  useEffect(() => {
    void refreshReg()
    void refreshTrainTags()
  }, [refreshReg, refreshTrainTags])

  // 把 excluded 持久化到 localStorage（按 project + version 隔离），切页面回来也在。
  // 切 version / 进入页面时先 seed 一次；之后随 setExcluded 变化自动保存。
  const excludedStorageKey = vid
    ? `studio.reg.excluded.${project.id}.${vid}`
    : null
  useEffect(() => {
    if (!excludedStorageKey) return
    try {
      const raw = localStorage.getItem(excludedStorageKey)
      if (!raw) {
        setExcluded(new Set())
        return
      }
      const arr = JSON.parse(raw)
      if (Array.isArray(arr)) {
        setExcluded(new Set(arr.filter((x): x is string => typeof x === 'string')))
      } else {
        setExcluded(new Set())
      }
    } catch {
      setExcluded(new Set())
    }
  }, [excludedStorageKey])
  useEffect(() => {
    if (!excludedStorageKey) return
    try {
      localStorage.setItem(excludedStorageKey, JSON.stringify(Array.from(excluded)))
    } catch { /* quota / privacy mode：丢就丢，不打扰用户 */ }
  }, [excludedStorageKey, excluded])

  // 刷新 / 进入页面时回放最近一次 reg_build job：锁回 jid + 回放历史日志
  useEffect(() => {
    if (!vid) return
    void api
      .getLatestVersionJob(project.id, vid, 'reg_build')
      .then((r) => {
        if (!r.job) return
        setJob(r.job)
        setLogs(r.log ? r.log.split('\n') : [])
      })
      .catch(() => {})
  }, [project.id, vid])

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
          if (t.status === 'done') setActiveTab('images')
        }
      }).catch(() => {})
    }
  })

  const trainImageCount = activeVersion?.stats?.train_image_count ?? 0
  // 任意一种生成跑着都视为 live —— 防止 booru / AI 并发同时写 reg/。
  const isLive = job?.status === 'running' || job?.status === 'pending' || aiBusy

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
  // mode 跟现有结构不一致时禁用切换（incremental 沿用会撞结构；用户必须先清空）
  const modeLocked = existingMode !== null && existingMode !== buildMode

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
    setAiBusy(true)
    setAiTask(null)
    setAiLogs([])
    try {
      const body: RegAiRequest = {
        excluded_tags: Array.from(excluded),
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
    try {
      const j = await api.startRegBuild(project.id, vid, body)
      setJob(j)
      setLogs([])
      toast(t('reg.enqueued', { id: j.id }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const onDelete = async () => {
    if (!vid) return
    if (!(await confirm(t('reg.confirmDelete'), { tone: 'danger', okText: t('reg.deleteOkText') }))) return
    try {
      await api.deleteReg(project.id, vid)
      toast(t('reg.deleted'), 'success')
      setReg(null)
      void refreshReg()
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
      idx={5}
      title={t('steps.reg.title')}
      subtitle={t('steps.reg.subtitle')}
      actions={
        <button
          onClick={() => {
            if (source === 'ai') void handleAiGenerate()
            else void startBuild()
          }}
          disabled={isLive || trainImageCount <= 0}
          className="btn btn-primary"
        >
          {isLive
            ? t('reg.generatingBtn')
            : source === 'ai'
              ? t('reg.aiGenerateBtn')
              : t('reg.startBuildBtn')}
        </button>
      }
    >
    <div className="flex flex-col h-full gap-3 min-h-0">

      {/* restyle: 状态条 — 4 cell info bar + 右端清空 */}
      <StatusStrip
        reg={reg}
        onDelete={onDelete}
        disabled={isLive}
        autoTagKind={autoTagKind}
      />

      {/* restyle: tab — 生成 / 图片 */}
      <div className="flex items-center gap-1 border-b border-subtle shrink-0">
        <RegTab
          active={activeTab === 'generate'}
          onClick={() => setActiveTab('generate')}
          label={t('reg.tabGenerate')}
          live={isLive}
        />
        <RegTab
          active={activeTab === 'images'}
          onClick={() => setActiveTab('images')}
          label={t('reg.tabImages')}
          count={reg && reg.image_count > 0 ? reg.image_count : undefined}
        />
      </div>

      {/* tab 内容（占满剩余高度，全宽） */}
      {activeTab === 'generate' ? (
        <div className="flex-1 min-h-0 overflow-y-auto">
          <div className="max-w-[1380px] py-2">

            {/* 来源 segmented control + hint */}
            <SourceSegmented
              source={source}
              onChange={setSource}
            />

            {/* AI / Booru 表单 */}
            {source === 'ai' ? (
              <AiForm
                trainTags={trainTags}
                excluded={excluded}
                onToggleExcluded={toggleTag}
                neg={aiNeg} onNegChange={setAiNeg}
                width={aiWidth} onWidthChange={setAiWidth}
                height={aiHeight} onHeightChange={setAiHeight}
                steps={aiSteps} onStepsChange={setAiSteps}
                cfg={aiCfg} onCfgChange={setAiCfg}
                seed={aiSeed} onSeedChange={setAiSeed}
                incremental={aiIncremental}
                onIncrementalChange={setAiIncremental}
              />
            ) : (
              <BooruForm
                trainTags={trainTags}
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

            {/* 运行日志：全宽置底 */}
            {(job || aiTask) && (
              <LogPanel
                job={job}
                jobLogs={logs}
                onCancelJob={async () => {
                  if (!job) return
                  try {
                    await api.cancelJob(job.id)
                    toast(t('reg.cancelToast'), 'success')
                  } catch (e) {
                    toast(String(e), 'error')
                  }
                }}
                aiTask={aiTask}
                aiLogs={aiLogs}
              />
            )}

          </div>
        </div>
      ) : (
        reg && reg.image_count > 0 ? (
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
        ) : (
          <section
            className="flex-1 flex flex-col items-center justify-center gap-1.5 text-center rounded-md border border-subtle bg-surface text-fg-tertiary"
            style={{ minHeight: 0 }}
          >
            <div className="text-sm text-fg-secondary font-medium">
              {t('reg.emptyRegTitle')}
            </div>
            <div className="text-2xs">
              {t('reg.emptyRegHint')}
            </div>
          </section>
        )
      )}

      {previewIdx !== null && reg && reg.files[previewIdx] && (
        <ImagePreviewModal
          src={regOrigUrl(project.id, vid, reg.files[previewIdx])}
          caption={previewCaption}
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
// 子组件 — restyle（按设计稿 `tmp/reg-restyle-design/.../正则集 restyle.html`）
// ---------------------------------------------------------------------------

// 状态条：4 cells 信息 + grow + 右端「清空」。锚定设计稿 `.status`。
function StatusStrip({
  reg,
  onDelete,
  disabled,
  autoTagKind,
}: {
  reg: RegStatus | null
  onDelete: () => void
  disabled: boolean
  autoTagKind: string
}) {
  const { t } = useTranslation()
  if (!reg) {
    return (
      <section className="rounded-md border border-subtle bg-surface px-3 py-2 text-xs text-fg-tertiary shrink-0">
        {t('reg.statusLoading')}
      </section>
    )
  }
  if (!reg.exists) {
    return (
      <section className="rounded-md border border-subtle bg-surface px-3 py-2 text-xs text-fg-tertiary shrink-0">
        {t('reg.statusNotExist')}
      </section>
    )
  }
  const m = reg.meta
  const failedCount = m?.failed_tags.length ?? 0
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
    <section className="rounded-lg border border-subtle bg-surface flex items-stretch overflow-hidden shrink-0">
      <StatusCell label={t('reg.statusCellSet')}>
        <span className="font-mono">
          <span className="text-ok">{reg.image_count}</span>
          {m && (
            <span className="text-fg-tertiary text-2xs font-normal ml-1">
              / {m.target_count} {t('reg.nImagesShort')}
            </span>
          )}
        </span>
      </StatusCell>
      <StatusCell label={t('reg.statusCellSourceTag')}>
        <span className="font-mono">
          {sourceLabel}
          {taggerLabel && (
            <span className="text-ok text-2xs font-normal ml-1.5">
              ✓ {taggerLabel}
            </span>
          )}
        </span>
      </StatusCell>
      <StatusCell label={t('reg.statusCellInvalidTags')}>
        <span className="font-mono">
          {failedCount > 0 ? (
            <>
              <span className="text-warn">{failedCount}</span>
              <span className="text-fg-tertiary text-2xs font-normal ml-1">
                {t('reg.statusCellSkipped')}
              </span>
            </>
          ) : (
            <span className="text-fg-tertiary">0</span>
          )}
        </span>
      </StatusCell>
      <StatusCell label={t('reg.statusCellLatest')}>
        <span className="font-normal text-fg-secondary text-sm">
          {m ? formatAgo(m.generated_at, t) : '—'}
        </span>
      </StatusCell>
      <div className="flex-1 border-r border-subtle" />
      <div className="flex items-center gap-2 px-3 border-l border-subtle">
        <button
          onClick={onDelete}
          disabled={disabled}
          className="btn btn-sm bg-transparent text-err border border-err-soft hover:bg-err-soft"
        >
          {t('reg.deleteBtn')}
        </button>
      </div>
    </section>
  )
}

function StatusCell({
  label, children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="px-4 py-2.5 border-r border-subtle flex flex-col gap-0.5">
      <span className="font-mono text-2xs uppercase tracking-wider text-fg-tertiary">
        {label}
      </span>
      <span className="text-sm text-fg-primary font-medium">
        {children}
      </span>
    </div>
  )
}

// Tab：设计稿 `.tab`，border-bottom 下划线（accent 色）。
function RegTab({
  active, onClick, label, count, live,
}: {
  active: boolean
  onClick: () => void
  label: string
  count?: number
  live?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{ marginBottom: -1 }}
      className={
        'inline-flex items-center gap-1.5 px-4 pt-2.5 pb-3 text-sm font-medium border-b-2 bg-transparent cursor-pointer transition-colors ' +
        (active
          ? 'text-fg-primary border-accent'
          : 'text-fg-tertiary border-transparent hover:text-fg-primary')
      }
    >
      <span>{label}</span>
      {count !== undefined && count > 0 && (
        <span className="font-mono text-2xs px-1.5 py-px rounded-full text-fg-tertiary bg-overlay">
          {count}
        </span>
      )}
      {live && (
        <span className="font-mono text-2xs px-1.5 py-px rounded-full text-warn bg-warn-soft">
          live
        </span>
      )}
    </button>
  )
}

// 来源 segmented control + 下方常驻 hint。
function SourceSegmented({
  source, onChange,
}: {
  source: 'ai' | 'booru'
  onChange: (s: 'ai' | 'booru') => void
}) {
  const { t } = useTranslation()
  return (
    <div className="mb-4">
      <div
        className="inline-flex p-0.5 gap-0.5 rounded-md border border-dim bg-sunken"
      >
        <SourceSegBtn
          active={source === 'ai'}
          onClick={() => onChange('ai')}
          label={t('reg.sourceAi')}
          sub={t('reg.sourceAiSub')}
        />
        <SourceSegBtn
          active={source === 'booru'}
          onClick={() => onChange('booru')}
          label={t('reg.sourceBooru')}
          sub={t('reg.sourceBooruSub')}
        />
      </div>
      <p className="mt-2 text-xs text-fg-tertiary leading-relaxed max-w-[720px]">
        <span className="mr-1.5">{source === 'ai' ? '◈' : '⚡'}</span>
        {source === 'ai' ? t('reg.sourceAiHint') : t('reg.sourceBooruHint')}
      </p>
    </div>
  )
}

function SourceSegBtn({
  active, onClick, label, sub,
}: {
  active: boolean
  onClick: () => void
  label: string
  sub: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        'inline-flex items-center gap-1.5 px-4 py-1.5 rounded text-sm font-medium border-0 cursor-pointer whitespace-nowrap transition-colors ' +
        (active
          ? 'bg-accent text-white'
          : 'bg-transparent text-fg-secondary hover:text-fg-primary')
      }
    >
      <span>{label}</span>
      <span
        className={
          'text-2xs font-normal ' +
          (active ? 'opacity-80' : 'text-fg-tertiary')
        }
      >
        · {sub}
      </span>
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
  return (
    <div className="rounded-lg border border-subtle bg-surface mb-3.5 overflow-hidden">
      <div
        onClick={collapsible ? () => setOpen((v) => !v) : undefined}
        className={
          'flex items-center gap-2.5 px-4 py-3 ' +
          (collapsible ? 'cursor-pointer' : '')
        }
      >
        <span className="text-sm font-semibold text-fg-primary">{title}</span>
        {tag && (
          <span
            className="font-mono text-2xs uppercase tracking-wider rounded-full px-2 py-0.5 border"
            style={{
              color: 'var(--accent)',
              background: 'var(--accent-soft)',
              borderColor: 'rgba(237,107,58,0.42)',
            }}
          >
            {tag}
          </span>
        )}
        {meta && (
          <span className="text-xs text-fg-tertiary">{meta}</span>
        )}
        {collapsible && (
          <span className="ml-auto inline-flex items-center gap-2 text-xs text-fg-tertiary">
            <span className="font-mono">
              {open ? t('reg.grpCollapse') : t('reg.grpExpand')}
            </span>
            <span
              className="inline-block transition-transform"
              style={{ transform: open ? 'rotate(90deg)' : undefined }}
            >
              ›
            </span>
          </span>
        )}
      </div>
      {open && (
        <div className="border-t border-subtle px-4 pt-1 pb-4">
          {children}
        </div>
      )}
    </div>
  )
}

// 单字段封装：label + control。
function Field({
  label, hint, locked, children,
}: {
  label: React.ReactNode
  hint?: React.ReactNode
  locked?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="pt-4">
      <label className="block text-xs text-fg-secondary mb-1.5 font-medium">
        {label}
        {hint && (
          <span className="ml-1.5 text-fg-tertiary font-normal text-2xs">
            {hint}
          </span>
        )}
      </label>
      {children}
      {locked && (
        <div className="font-mono text-2xs text-fg-tertiary mt-1.5">
          {locked}
        </div>
      )}
    </div>
  )
}

// AI 表单 — grp 卡：出图（常用）/ 排除 tag / 采样（进阶）
function AiForm({
  trainTags,
  excluded, onToggleExcluded,
  neg, onNegChange,
  width, onWidthChange,
  height, onHeightChange,
  steps, onStepsChange,
  cfg, onCfgChange,
  seed, onSeedChange,
  incremental, onIncrementalChange,
}: {
  trainTags: RegTagCount[]
  excluded: Set<string>
  onToggleExcluded: (tag: string) => void
  neg: string
  onNegChange: (v: string) => void
  width: number; onWidthChange: (v: number) => void
  height: number; onHeightChange: (v: number) => void
  steps: number; onStepsChange: (v: number) => void
  cfg: number; onCfgChange: (v: number) => void
  seed: number; onSeedChange: (v: number) => void
  incremental: boolean; onIncrementalChange: (v: boolean) => void
}) {
  const { t } = useTranslation()
  return (
    <>
      <GrpCard title={t('reg.grpAiGen')} tag={t('reg.grpTagCommon')}>
        <Field label={t('reg.negPrompt')}>
          <textarea
            className="input font-mono text-sm"
            style={{ minHeight: 78, lineHeight: 1.6, padding: '10px 12px' }}
            rows={3}
            value={neg}
            onChange={(e) => onNegChange(e.target.value)}
          />
        </Field>
        <div className="grid grid-cols-2 gap-3.5">
          <Field label={t('reg.widthLabel')}>
            <UnitInput
              value={width}
              onChange={onWidthChange}
              unit="px"
              min={256}
              max={4096}
              step={64}
            />
          </Field>
          <Field label={t('reg.heightLabel')}>
            <UnitInput
              value={height}
              onChange={onHeightChange}
              unit="px"
              min={256}
              max={4096}
              step={64}
            />
          </Field>
        </div>
        <Field
          label={t('reg.modeLabel')}
          hint={t('reg.modeHintAi')}
        >
          <select
            className="select input"
            value={incremental ? 'incremental' : 'full'}
            onChange={(e) => onIncrementalChange(e.target.value === 'incremental')}
          >
            <option value="incremental">{t('reg.modeIncrementalAi')}</option>
            <option value="full">{t('reg.modeFullAi')}</option>
          </select>
        </Field>
      </GrpCard>

      <ExcludeTags
        trainTags={trainTags}
        excluded={excluded}
        onToggle={onToggleExcluded}
      />

      <GrpCard
        title={t('reg.grpSampling')}
        meta={t('reg.grpSamplingMeta')}
        collapsible
        defaultOpen={false}
      >
        <div className="grid grid-cols-3 gap-3.5">
          <Field label={t('reg.stepsLabel')}>
            <input
              type="number"
              className="input font-mono"
              value={steps}
              onChange={(e) => onStepsChange(Number(e.target.value) || 0)}
              min={1} max={150}
            />
          </Field>
          <Field label="CFG Scale">
            <input
              type="number"
              className="input font-mono"
              value={cfg}
              onChange={(e) => onCfgChange(Number(e.target.value) || 0)}
              min={0} max={20} step={0.5}
            />
          </Field>
          <Field
            label={t('reg.seedLabel')}
            hint={t('reg.seedHintRandom')}
          >
            <input
              type="number"
              className="input font-mono"
              value={seed}
              onChange={(e) => onSeedChange(Number(e.target.value) || 0)}
              min={0}
            />
          </Field>
        </div>
      </GrpCard>
    </>
  )
}

// Booru 表单 — grp 卡：抓取（常用）/ 排除 tag / 进阶
function BooruForm({
  trainTags, trainImageCount,
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
      <GrpCard title={t('reg.grpBooruScrape')} tag={t('reg.grpTagCommon')}>
        <div className="grid grid-cols-2 gap-3.5">
          <Field label={t('reg.source')}>
            <select
              className="select input"
              value={apiSource}
              onChange={(e) => onApiSourceChange(e.target.value as 'gelbooru' | 'danbooru')}
            >
              <option value="gelbooru">Gelbooru</option>
              <option value="danbooru">Danbooru</option>
            </select>
          </Field>
          <Field
            label={t('reg.buildModeLabel')}
            hint={modeLocked ? t('reg.buildModeLocked', { mode: existingMode }) : undefined}
          >
            <select
              className="select input"
              value={buildMode}
              onChange={(e) => onBuildModeChange(e.target.value as 'mirror' | 'flat')}
              disabled={modeLocked}
            >
              <option value="flat">{t('reg.buildModeFlat')}</option>
              <option value="mirror">{t('reg.buildModeMirror')}</option>
            </select>
          </Field>
        </div>
        <div className="grid grid-cols-2 gap-3.5">
          <Field
            label={t('reg.targetCount')}
            hint={t('reg.targetCountHint')}
            locked={mirror ? t('reg.targetMirrorLocked', { n: trainImageCount }) : undefined}
          >
            <input
              type="number"
              className="input font-mono"
              value={mirror ? String(trainImageCount) : targetCount}
              onChange={(e) => onTargetCountChange(e.target.value)}
              placeholder={String(trainImageCount)}
              disabled={mirror}
              min={1}
            />
          </Field>
          <Field
            label={t('reg.modeLabel')}
            hint={t('reg.modeHintBooru')}
          >
            <select
              className="select input"
              value={mode}
              onChange={(e) => onModeChange(e.target.value as 'full' | 'incremental')}
            >
              <option value="incremental">{t('reg.modeIncrementalBooru')}</option>
              <option value="full">{t('reg.modeFullBooru')}</option>
            </select>
          </Field>
        </div>
        <Field label="">
          <CheckRow
            checked={autoTag}
            onChange={onAutoTagChange}
            label={t('reg.autoTagLabel')}
          />
        </Field>
      </GrpCard>

      <ExcludeTags
        trainTags={trainTags}
        excluded={excluded}
        onToggle={onToggleExcluded}
        modeHint={t('reg.excludeHintBooru')}
      />

      <GrpCard
        title={t('reg.grpAdvanced')}
        meta={t('reg.grpAdvancedMetaBooru')}
        collapsible
        defaultOpen={false}
      >
        <Field
          label={t('reg.autoTagKindLabel')}
          hint={!autoTag ? t('reg.autoTagKindDisabled') : undefined}
        >
          <select
            className="select input"
            value={autoTagKind}
            onChange={(e) => onAutoTagKindChange(e.target.value as 'wd14' | 'cltagger')}
            disabled={!autoTag}
          >
            <option value="wd14">WD14</option>
            <option value="cltagger">CLTagger</option>
          </select>
        </Field>
        <Field label="">
          <CheckRow
            checked={autoDedup}
            onChange={onAutoDedupChange}
            label={t('reg.autoDedupLabel')}
            sub={t('reg.autoDedupSub')}
          />
        </Field>
        <AdvancedFields value={advanced} onChange={onAdvancedChange} />
      </GrpCard>
    </>
  )
}

// 数字输入 + 单位后缀（px）
function UnitInput({
  value, onChange, unit, min, max, step,
}: {
  value: number
  onChange: (v: number) => void
  unit: string
  min?: number; max?: number; step?: number
}) {
  return (
    <div className="relative">
      <input
        type="number"
        className="input font-mono"
        style={{ paddingRight: 36 }}
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

// checkbox 行
function CheckRow({
  checked, onChange, label, sub,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label: string
  sub?: string
}) {
  return (
    <label className="inline-flex items-center gap-2.5 cursor-pointer text-sm text-fg-secondary select-none">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="w-4 h-4 accent-accent cursor-pointer"
      />
      <span>{label}</span>
      {sub && <span className="text-2xs text-fg-tertiary">{sub}</span>}
    </label>
  )
}

// 排除 tag — train 高频 tag 一栏列出 + 自定义排除一栏
function ExcludeTags({
  trainTags, excluded, onToggle, modeHint,
}: {
  trainTags: RegTagCount[]
  excluded: Set<string>
  onToggle: (tag: string) => void
  modeHint?: string
}) {
  const { t } = useTranslation()
  const [draft, setDraft] = useState('')
  const trainTagSet = useMemo(
    () => new Set(trainTags.map((t) => t.tag)),
    [trainTags]
  )
  const customTags = useMemo(
    () => Array.from(excluded).filter((t) => !trainTagSet.has(t)).sort(),
    [excluded, trainTagSet]
  )
  const excludedCount = excluded.size
  const normalize = (raw: string): string =>
    raw.trim().toLowerCase().replace(/\s+/g, '_')
  const addCustom = () => {
    const items = draft
      .split(/[,，\n]+/)
      .map(normalize)
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
      {trainTags.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {trainTags.map((info) => {
            const on = excluded.has(info.tag)
            return (
              <button
                key={info.tag}
                onClick={() => onToggle(info.tag)}
                className={
                  'inline-flex items-center gap-1.5 h-6 px-2.5 rounded-md font-mono text-xs cursor-pointer transition-colors border ' +
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
                <span className={on ? 'text-accent' : 'text-fg-tertiary'}>
                  {on ? '✕' : '+'}
                </span>
                {info.tag.replace(/_/g, ' ')}
                <span className="text-fg-disabled text-2xs">×{info.count}</span>
              </button>
            )
          })}
        </div>
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
              className="inline-flex items-center gap-1.5 h-6 px-2.5 rounded-md font-mono text-xs border"
              style={{
                background: 'var(--warn-soft)',
                borderColor: 'rgba(224,162,58,0.4)',
                color: 'var(--warn)',
              }}
            >
              {tag.replace(/_/g, ' ')}
              <button
                onClick={() => onToggle(tag)}
                className="bg-transparent border-none cursor-pointer p-0 text-warn opacity-80"
                aria-label={t('reg.excludeCustomRemoveAria', { tag })}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
      <div className="flex gap-2 mt-3">
        <input
          className="input font-mono flex-1 text-sm"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              addCustom()
            }
          }}
          placeholder={t('reg.excludePlaceholder')}
        />
        <button
          onClick={addCustom}
          disabled={!draft.trim()}
          className="btn btn-secondary btn-sm"
        >
          {t('reg.excludeAdd')}
        </button>
      </div>
    </GrpCard>
  )
}

// Booru 进阶里的 PP5.5 后处理 + 长宽比 — 设计稿没强调但功能保留
function AdvancedFields({
  value, onChange,
}: {
  value: AdvancedParams
  onChange: (v: AdvancedParams) => void
}) {
  const { t } = useTranslation()
  const set = <K extends keyof AdvancedParams>(k: K, v: AdvancedParams[K]) =>
    onChange({ ...value, [k]: v })
  return (
    <>
      <Field label={t('reg.aspectFilter')}>
        <CheckRow
          checked={value.aspect_ratio_filter_enabled}
          onChange={(v) => set('aspect_ratio_filter_enabled', v)}
          label={t('reg.aspectFilterEnable')}
          sub={t('reg.aspectFilterHint')}
        />
        {value.aspect_ratio_filter_enabled && (
          <div className="grid grid-cols-2 gap-3.5 mt-2">
            <input
              type="number" className="input font-mono"
              min={0.1} max={1} step={0.05}
              value={value.min_aspect_ratio}
              onChange={(e) =>
                set('min_aspect_ratio', Math.max(0.1, Math.min(1, Number(e.target.value) || 0.5)))
              }
            />
            <input
              type="number" className="input font-mono"
              min={1} max={10} step={0.1}
              value={value.max_aspect_ratio}
              onChange={(e) =>
                set('max_aspect_ratio', Math.max(1, Math.min(10, Number(e.target.value) || 2)))
              }
            />
          </div>
        )}
      </Field>
      <Field label={t('reg.postprocess')}>
        <div className="grid grid-cols-2 gap-3.5">
          <select
            className="select input"
            value={value.postprocess_method}
            onChange={(e) => set('postprocess_method', e.target.value as 'smart' | 'stretch' | 'crop')}
          >
            <option value="smart">{t('reg.postprocessSmart')}</option>
            <option value="stretch">{t('reg.postprocessStretch')}</option>
            <option value="crop">{t('reg.postprocessCrop')}</option>
          </select>
          <input
            type="number" className="input font-mono"
            min={0.05} max={0.5} step={0.05}
            value={value.postprocess_max_crop_ratio}
            onChange={(e) =>
              set('postprocess_max_crop_ratio', Math.max(0.05, Math.min(0.5, Number(e.target.value) || 0.1)))
            }
            title={t('reg.maxCropTitle')}
          />
        </div>
      </Field>
      <Field label={t('reg.selectImages')}>
        <CheckRow
          checked={value.skip_similar}
          onChange={(v) => set('skip_similar', v)}
          label="skip_similar"
          sub={t('reg.skipSimilarTitle')}
        />
      </Field>
    </>
  )
}

// 运行日志全宽置底
function LogPanel({
  job, jobLogs, onCancelJob, aiTask, aiLogs,
}: {
  job: Job | null
  jobLogs: string[]
  onCancelJob: () => Promise<void>
  aiTask: Task | null
  aiLogs: string[]
}) {
  const { t } = useTranslation()
  // 优先显示进行中的 job；否则展示 aiTask
  if (job) {
    return (
      <div className="mt-4">
        <JobProgress
          job={job}
          logs={jobLogs}
          onCancel={onCancelJob}
        />
      </div>
    )
  }
  if (aiTask) {
    return (
      <div className="mt-4 rounded-lg border border-subtle bg-surface overflow-hidden">
        <div className="flex items-center gap-2.5 px-4 py-2.5 border-b border-subtle text-sm font-semibold">
          <span>{t('reg.aiLogTitle')}</span>
          <AiTaskBadge task={aiTask} />
        </div>
        {aiLogs.length > 0 && (
          <pre className="m-0 px-4 py-3.5 bg-sunken font-mono text-xs leading-relaxed text-fg-tertiary max-h-72 overflow-auto whitespace-pre-wrap break-words">
            {aiLogs.join('\n')}
          </pre>
        )}
      </div>
    )
  }
  return null
}

function AiTaskBadge({ task }: { task: Task }) {
  const { t } = useTranslation()
  const label =
    task.status === 'done' ? t('reg.aiStatusDone') :
    task.status === 'running' ? t('reg.aiStatusRunning') :
    task.status === 'failed' ? t('reg.aiStatusFailed') :
    task.status === 'pending' ? t('reg.aiStatusPending') :
    task.status === 'canceled' ? t('reg.aiStatusCanceled') : task.status
  const cls =
    task.status === 'done' ? 'bg-ok-soft text-ok' :
    task.status === 'running' ? 'bg-info-soft text-info' :
    task.status === 'failed' ? 'bg-err-soft text-err' : 'bg-overlay text-fg-tertiary'
  return (
    <span className={`font-mono text-2xs px-2 py-0.5 rounded-full ${cls}`}>
      {label} · #{task.id}
    </span>
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
    <section className="rounded-md border border-subtle bg-surface p-2 flex-1 min-h-0 flex flex-col gap-2">
      {/* tab 条（pill chip 风格，对齐 TagEdit）+ 选中数 + 删除按钮 */}
      <div className="flex items-center gap-1 flex-wrap pb-1.5 border-b border-subtle">
        <RegFolderTab
          label={t('reg.folderAll')}
          count={allItems.length}
          active={activeFolder === null}
          onClick={() => setActiveFolder(null)}
        />
        {folders.map((f) => (
          <RegFolderTab
            key={f || '__root__'}
            label={f || t('reg.folderRoot')}
            count={folderCounts.get(f) ?? 0}
            active={activeFolder === f}
            onClick={() => setActiveFolder(f)}
          />
        ))}
        <span className="flex-1" />
        {selected.size > 0 && (
          <span className="text-2xs text-accent pr-2">
            {t('reg.regPreviewSelected', { n: selected.size })}
          </span>
        )}
        <button
          onClick={() => void onDedup()}
          disabled={dedupBusy || isLive}
          className="btn btn-sm"
          title={t('reg.dedupTitle')}
        >
          {dedupBusy ? t('reg.dedupRunning') : t('reg.dedupBtn')}
        </button>
        <button
          onClick={() => void onDelete()}
          disabled={selected.size === 0 || isLive || dedupBusy}
          className="btn btn-sm bg-err-soft text-err border-err"
          title={t('reg.deleteFilesTitle')}
        >
          {t('reg.deleteFilesBtn', { n: selected.size })}
        </button>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto">
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
          ariaLabel="reg-preview"
        />
      </div>
    </section>
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
  // 跟 TagEdit / Preprocess 同款 pill chip 风格（rounded-full + bg-accent 主色填充）。
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        'px-2 py-0.5 rounded-full text-xs font-medium transition-colors ' +
        (active
          ? 'bg-accent text-white'
          : 'bg-overlay text-fg-secondary hover:bg-accent-soft')
      }
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
