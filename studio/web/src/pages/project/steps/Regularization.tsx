import type { TFunction } from 'i18next'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Trans, useTranslation } from 'react-i18next'
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
  const [advancedOpen, setAdvancedOpen] = useState(false)

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

      {/* 顶部常驻 StatusBar：reg 集快照（图片数 / target / 来源 / auto-tag /
          时间 / 补足 / 清空）—— 不进 tab，一直可见 */}
      <RegStatusBar
        reg={reg}
        onDelete={onDelete}
        disabled={isLive}
      />

      {/* tab 条（B2：合并到「生成 / 图片」两个） */}
      <div className="flex items-center gap-1 border-b border-subtle shrink-0">
        <TabButton
          active={activeTab === 'generate'}
          onClick={() => setActiveTab('generate')}
          label={t('reg.tabGenerate')}
          badge={isLive ? 'live' : undefined}
        />
        <TabButton
          active={activeTab === 'images'}
          onClick={() => setActiveTab('images')}
          label={t('reg.tabImages')}
          badge={reg && reg.image_count > 0 ? String(reg.image_count) : undefined}
        />
      </div>

      {/* tab 内容（占满剩余高度，全宽） */}
      {activeTab === 'generate' ? (
        <div className="flex flex-col gap-3 min-h-0 flex-1 overflow-y-auto">
          {/* 来源 picker（B2，PR-2）：默认 AI 先验，可切 Booru */}
          <section className="rounded-md border border-subtle bg-surface px-3.5 py-2 flex items-center gap-2 text-xs shrink-0">
            <span className="text-fg-tertiary">{t('reg.sourcePickerLabel')}</span>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="radio"
                name="reg-source"
                value="ai"
                checked={source === 'ai'}
                onChange={() => setSource('ai')}
              />
              <span className="text-fg-secondary">{t('reg.sourceAi')}</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="radio"
                name="reg-source"
                value="booru"
                checked={source === 'booru'}
                onChange={() => setSource('booru')}
              />
              <span className="text-fg-secondary">{t('reg.sourceBooru')}</span>
            </label>
            <span className="flex-1" />
            <span className="text-2xs text-fg-tertiary">
              {source === 'ai' ? t('reg.sourceAiHint') : t('reg.sourceBooruHint')}
            </span>
          </section>

          {source === 'ai' ? (
            <AiGenPanel
              trainTags={trainTags}
              excluded={excluded} onToggle={toggleTag}
              neg={aiNeg} onNegChange={setAiNeg}
              width={aiWidth} onWidthChange={setAiWidth}
              height={aiHeight} onHeightChange={setAiHeight}
              steps={aiSteps} onStepsChange={setAiSteps}
              cfg={aiCfg} onCfgChange={setAiCfg}
              seed={aiSeed} onSeedChange={setAiSeed}
              incremental={aiIncremental} onIncrementalChange={setAiIncremental}
              task={aiTask}
              trainImageCount={trainImageCount}
            />
          ) : (
          <>
          <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5 flex flex-col gap-2.5 shrink-0">
            <div className="flex flex-wrap items-center gap-2.5 text-xs">
              <span className="text-fg-tertiary">{t('reg.source')}</span>
              <select
                value={apiSource}
                onChange={(e) => setApiSource(e.target.value as 'gelbooru' | 'danbooru')}
                className="input px-2 py-0.5 text-sm"
              >
                <option value="gelbooru">Gelbooru</option>
                <option value="danbooru">Danbooru</option>
              </select>
              <span className="text-fg-tertiary">|</span>
              <span className="text-fg-tertiary">{t('reg.buildModeLabel')}</span>
              <select
                value={buildMode}
                onChange={(e) => setBuildMode(e.target.value as 'mirror' | 'flat')}
                disabled={modeLocked}
                className="input px-2 py-0.5 text-sm"
                title={modeLocked ? t('reg.buildModeLocked', { mode: existingMode }) : t('reg.buildModeTitle')}
              >
                <option value="flat">{t('reg.buildModeFlat')}</option>
                <option value="mirror">{t('reg.buildModeMirror')}</option>
              </select>
              {buildMode === 'flat' ? (
                <label
                  className="flex items-center gap-1"
                  title={t('reg.targetCountTitle', { fallback: trainImageCount })}
                >
                  <span className="text-fg-tertiary">{t('reg.targetCount')}</span>
                  <input
                    type="number"
                    min={1}
                    value={targetCount}
                    onChange={(e) => setTargetCount(e.target.value)}
                    placeholder={String(trainImageCount)}
                    className="input input-mono px-1 py-px"
                    style={{ width: 70 }}
                  />
                </label>
              ) : (
                <span className="text-fg-tertiary">
                  {t('reg.targetCount')}{' '}
                  <span className="font-mono text-fg-primary font-medium">{trainImageCount}</span>
                  <span className="text-fg-tertiary">{t('reg.mirrorTrain')}</span>
                </span>
              )}
              <span className="text-fg-tertiary">|</span>
              <label className="flex items-center gap-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoTag}
                  onChange={(e) => setAutoTag(e.target.checked)}
                />
                <span className="text-fg-secondary">{t('reg.autoTagLabel')}</span>
              </label>
              <select
                value={autoTagKind}
                onChange={(e) => setAutoTagKind(e.target.value as 'wd14' | 'cltagger')}
                disabled={!autoTag}
                className="input px-2 py-0.5 text-sm"
                title={t('reg.autoTagKindTitle')}
              >
                <option value="wd14">WD14</option>
                <option value="cltagger">CLTagger</option>
              </select>
              <span className="text-fg-tertiary">|</span>
              <label
                className="flex items-center gap-1 cursor-pointer"
                title={t('reg.autoDedupTitle')}
              >
                <input
                  type="checkbox"
                  checked={autoDedup}
                  onChange={(e) => setAutoDedup(e.target.checked)}
                />
                <span className="text-fg-secondary">{t('reg.autoDedupLabel')}</span>
              </label>
              <span className="text-fg-tertiary">|</span>
              <span className="text-fg-tertiary">{t('reg.modeLabel')}</span>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as 'full' | 'incremental')}
                className="input px-2 py-0.5 text-sm"
                title={t('reg.modeTitle')}
              >
                <option value="incremental">{t('reg.modeIncremental')}</option>
                <option value="full">{t('reg.modeFull')}</option>
              </select>
              <button
                onClick={() => setAdvancedOpen((v) => !v)}
                className="text-fg-tertiary bg-transparent border-none cursor-pointer text-xs"
              >
                {advancedOpen ? t('reg.advancedOpen') : t('reg.advancedClosed')}
              </button>
              <span className="flex-1" />
            </div>

            {advancedOpen && (
              <AdvancedPanel value={advanced} onChange={setAdvanced} />
            )}

            <ExcludeTagsPicker
              trainTags={trainTags}
              excluded={excluded}
              onToggle={toggleTag}
            />
          </section>

          {job && (
            <JobProgress
              job={job}
              logs={logs}
              onCancel={async () => {
                try {
                  await api.cancelJob(job.id)
                  toast(t('reg.cancelToast'), 'success')
                } catch (e) {
                  toast(String(e), 'error')
                }
              }}
            />
          )}
            </>
          )}
          {aiTask && (
            <AiTaskLogSection task={aiTask} logs={aiLogs} />
          )}
        </div>
      ) : (
        // 图片 tab：占满剩余高度
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
          <section style={{
            borderRadius: 'var(--r-md)', border: '1px solid var(--border-subtle)',
            background: 'var(--bg-surface)', flex: 1, display: 'flex',
            flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            minHeight: 0, color: 'var(--fg-tertiary)', fontSize: 'var(--t-sm)',
            textAlign: 'center', gap: 6,
          }}>
            <div style={{ fontSize: 'var(--t-md)', color: 'var(--fg-secondary)', fontWeight: 500 }}>
              {t('reg.emptyRegTitle')}
            </div>
            <div style={{ fontSize: 'var(--t-xs)' }}>
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
// 子组件
// ---------------------------------------------------------------------------

function TabButton({
  active, onClick, label, badge,
}: {
  active: boolean
  onClick: () => void
  label: string
  badge?: string
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '8px 14px',
        background: 'transparent',
        border: 'none',
        borderBottom: `2px solid ${active ? 'var(--accent)' : 'transparent'}`,
        color: active ? 'var(--fg-primary)' : 'var(--fg-secondary)',
        fontSize: 'var(--t-sm)',
        fontWeight: active ? 600 : 500,
        cursor: 'pointer',
        marginBottom: -1,
        display: 'inline-flex', alignItems: 'center', gap: 6,
        transition: 'color 100ms ease, border-color 100ms ease',
      }}
      onMouseEnter={(e) => { if (!active) (e.currentTarget as HTMLElement).style.color = 'var(--fg-primary)' }}
      onMouseLeave={(e) => { if (!active) (e.currentTarget as HTMLElement).style.color = 'var(--fg-secondary)' }}
    >
      {label}
      {badge && (
        <span style={{
          fontSize: 'var(--t-2xs)',
          padding: '1px 6px',
          borderRadius: 'var(--r-sm)',
          background: badge === 'live' ? 'var(--warn-soft)' : 'var(--bg-sunken)',
          color: badge === 'live' ? 'var(--warn)' : 'var(--fg-tertiary)',
          fontFamily: badge === 'live' ? 'var(--font-sans)' : 'var(--font-mono)',
          fontWeight: 500,
        }}>
          {badge === 'live' ? 'live' : badge}
        </span>
      )}
    </button>
  )
}

function RegStatusBar({
  reg,
  onDelete,
  disabled,
}: {
  reg: RegStatus | null
  onDelete: () => void
  disabled: boolean
}) {
  const { t } = useTranslation()
  if (!reg) {
    return (
      <section className="rounded-sm border border-subtle bg-surface px-2.5 py-1.5 text-xs text-fg-tertiary shrink-0">
        {t('reg.statusLoading')}
      </section>
    )
  }
  if (!reg.exists) {
    return (
      <section className="rounded-sm border border-subtle bg-surface px-2.5 py-1.5 text-xs text-fg-tertiary shrink-0">
        {t('reg.statusNotExist')}
      </section>
    )
  }
  const m = reg.meta
  const ago = m ? formatAgo(m.generated_at, t) : '?'
  return (
    <section className="rounded-sm border border-subtle bg-surface px-3 py-2 flex flex-col gap-1 shrink-0">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-fg-secondary">
          {t('reg.statusExists')}
          <span className="font-mono text-ok font-medium">{t('reg.nImages', { n: reg.image_count })}</span>
        </span>
        {m && (
          <>
            <span className="text-fg-tertiary">·</span>
            <span className="text-fg-tertiary">
              target {m.actual_count}/{m.target_count}
            </span>
            <span className="text-fg-tertiary">·</span>
            <span className="text-fg-tertiary">
              {m.generation_method === 'ai_base' ? t('reg.statusAiGen') : m.api_source}
            </span>
            <span className="text-fg-tertiary">·</span>
            <span className="text-fg-tertiary">
              auto-tag:{' '}
              <span className={m.auto_tagged ? 'text-ok' : 'text-fg-tertiary'}>
                {m.auto_tagged ? '✓' : '×'}
              </span>
            </span>
            <span className="text-fg-tertiary">·</span>
            <span className="text-fg-tertiary">{ago}</span>
            {m.failed_tags.length > 0 && (
              <span
                className="text-warn"
                title={t('reg.failedTagsTitle', { tags: m.failed_tags.map((x) => x.replace(/_/g, ' ')).join(', ') })}
              >
                {t('reg.failedTags', { n: m.failed_tags.length })}
              </span>
            )}
            {m.incremental_runs > 0 && (
              <span className="text-fg-tertiary" title={t('reg.topUpCountTitle')}>
                {t('reg.topUpCount', { n: m.incremental_runs })}
              </span>
            )}
          </>
        )}
        <span className="flex-1" />
        <button
          onClick={onDelete}
          disabled={disabled}
          className="btn btn-sm bg-err-soft text-err border-err"
        >
          {t('reg.deleteBtn')}
        </button>
      </div>

      {m && (
        <div className="flex flex-wrap items-center gap-2 text-2xs text-fg-tertiary">
          <span>{t('reg.clusteringLabel')}</span>
          {m.postprocess_clusters !== null ? (
            <span
              className="text-fg-secondary"
              title={t('reg.clusteringValueTitle', {
                method: m.postprocess_method,
                ratio: m.postprocess_max_crop_ratio,
              })}
            >
              {t('reg.clusteringValue', {
                n: m.postprocess_clusters,
                method: m.postprocess_method,
                ratio: m.postprocess_max_crop_ratio,
              })}
            </span>
          ) : (
            <span
              className="text-fg-tertiary"
              title={t('reg.clusteringNoneTitle')}
            >
              {t('reg.clusteringNone')}
            </span>
          )}
        </div>
      )}
    </section>
  )
}

function AdvancedPanel({
  value,
  onChange,
}: {
  value: AdvancedParams
  onChange: (v: AdvancedParams) => void
}) {
  const { t } = useTranslation()
  const set = <K extends keyof AdvancedParams>(k: K, v: AdvancedParams[K]) =>
    onChange({ ...value, [k]: v })
  return (
    <div className="rounded-sm border border-subtle bg-sunken px-3.5 py-2.5 flex flex-col gap-2.5 text-xs">
      <p className="text-2xs text-fg-tertiary m-0">
        {t('reg.advancedDefaults')}
      </p>

      <Group label={t('reg.selectImages')}>
        <label className="flex items-center gap-1 cursor-pointer">
          <input
            type="checkbox"
            checked={value.skip_similar}
            onChange={(e) => set('skip_similar', e.target.checked)}
          />
          <span
            className="text-fg-secondary"
            title={t('reg.skipSimilarTitle')}
          >
            skip_similar
          </span>
        </label>
      </Group>

      <Group label={t('reg.aspectFilter')}>
        <label className="flex items-center gap-1 cursor-pointer">
          <input
            type="checkbox"
            checked={value.aspect_ratio_filter_enabled}
            onChange={(e) => set('aspect_ratio_filter_enabled', e.target.checked)}
          />
          <span className="text-fg-secondary">{t('reg.aspectFilterEnable')}</span>
        </label>
        {value.aspect_ratio_filter_enabled && (
          <>
            <label className="flex items-center gap-1">
              <span className="text-fg-tertiary">min</span>
              <input
                type="number"
                min={0.1}
                max={1}
                step={0.05}
                value={value.min_aspect_ratio}
                onChange={(e) =>
                  set(
                    'min_aspect_ratio',
                    Math.max(0.1, Math.min(1, Number(e.target.value) || 0.5))
                  )
                }
                className="input input-mono px-1 py-px"
                style={{ width: 64 }}
              />
            </label>
            <label className="flex items-center gap-1">
              <span className="text-fg-tertiary">max</span>
              <input
                type="number"
                min={1}
                max={10}
                step={0.1}
                value={value.max_aspect_ratio}
                onChange={(e) =>
                  set(
                    'max_aspect_ratio',
                    Math.max(1, Math.min(10, Number(e.target.value) || 2))
                  )
                }
                className="input input-mono px-1 py-px"
                style={{ width: 64 }}
              />
            </label>
            <span className="text-2xs text-fg-tertiary">
              {t('reg.aspectFilterHint')}
            </span>
          </>
        )}
      </Group>

      <Group label={t('reg.postprocess')}>
        <span className="text-fg-tertiary">{t('reg.postprocessMethod')}</span>
        <select
          value={value.postprocess_method}
          onChange={(e) =>
            set('postprocess_method', e.target.value as 'smart' | 'stretch' | 'crop')
          }
          className="input px-1.5 py-px text-xs"
        >
          <option value="smart">{t('reg.postprocessSmart')}</option>
          <option value="stretch">{t('reg.postprocessStretch')}</option>
          <option value="crop">{t('reg.postprocessCrop')}</option>
        </select>
        <label className="flex items-center gap-1">
          <span className="text-fg-tertiary">max_crop</span>
          <input
            type="number"
            min={0.05}
            max={0.5}
            step={0.05}
            value={value.postprocess_max_crop_ratio}
            onChange={(e) =>
              set(
                'postprocess_max_crop_ratio',
                Math.max(0.05, Math.min(0.5, Number(e.target.value) || 0.1))
              )
            }
            className="input input-mono"
            style={{ width: 64, padding: '2px 4px' }}
            title={t('reg.maxCropTitle')}
          />
        </label>
      </Group>
    </div>
  )
}

function Group({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="flex items-baseline gap-2.5">
      <span className="text-2xs text-fg-tertiary w-[72px] shrink-0 uppercase tracking-wider">
        {label}
      </span>
      <div className="flex flex-wrap items-center gap-2.5 flex-1">{children}</div>
    </div>
  )
}

// ── AiGenPanel ──────────────────────────────────────────────────────────────
//
// 先验生成（DreamBooth prior preservation）：base 模型对每张 train 图的 tag
// 反向出对照图作正则集。**无 LoRA UI** —— LoRA 加进来反而把要保留的 prior
// 给覆盖了。

function AiNumField({
  label, value, onChange, min, max, step,
}: {
  label: string; value: number
  onChange: (v: number) => void
  min?: number; max?: number; step?: number
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="caption">{label}</label>
      <input
        type="number" className="input"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}

function AiTaskStatus({ task }: { task: Task | null }) {
  const { t } = useTranslation()
  if (!task) return null
  const label =
    task.status === 'done' ? t('reg.aiStatusDone') :
    task.status === 'running' ? t('reg.aiStatusRunning') :
    task.status === 'failed' ? t('reg.aiStatusFailed') :
    task.status === 'pending' ? t('reg.aiStatusPending') :
    task.status === 'canceled' ? t('reg.aiStatusCanceled') : task.status
  const cls =
    task.status === 'done' ? 'badge badge-ok' :
    task.status === 'running' ? 'badge badge-info' :
    task.status === 'failed' ? 'badge badge-err' : 'badge'
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className={cls}>{label}</span>
      <span className="caption font-mono">#{task.id}</span>
      {task.status === 'done' && (
        <span className="text-xs text-fg-tertiary">{t('reg.aiWritten')}</span>
      )}
      {task.status === 'failed' && task.error_msg && (
        <span className="text-xs text-err">{task.error_msg}</span>
      )}
    </div>
  )
}

function AiTaskLogSection({ task, logs }: { task: Task; logs: string[] }) {
  const { t } = useTranslation()
  const logEndRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs.length])
  return (
    <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5 flex flex-col gap-2 shrink-0">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-fg-secondary">{t('reg.aiLogTitle')}</span>
        <AiTaskStatus task={task} />
      </div>
      {logs.length > 0 && (
        <pre className="m-0 p-2.5 bg-sunken rounded-sm font-mono text-2xs text-fg-secondary leading-relaxed whitespace-pre-wrap break-words max-h-48 overflow-auto">
          {logs.join('\n')}
          <div ref={logEndRef} />
        </pre>
      )}
    </section>
  )
}

function AiGenPanel({
  trainTags,
  excluded, onToggle,
  neg, onNegChange,
  width, onWidthChange,
  height, onHeightChange,
  steps, onStepsChange,
  cfg, onCfgChange,
  seed, onSeedChange,
  incremental, onIncrementalChange,
  task,
  trainImageCount,
}: {
  trainTags: RegTagCount[]
  excluded: Set<string>; onToggle: (tag: string) => void
  neg: string; onNegChange: (v: string) => void
  width: number; onWidthChange: (v: number) => void
  height: number; onHeightChange: (v: number) => void
  steps: number; onStepsChange: (v: number) => void
  cfg: number; onCfgChange: (v: number) => void
  seed: number; onSeedChange: (v: number) => void
  incremental: boolean; onIncrementalChange: (v: boolean) => void
  task: Task | null
  trainImageCount: number
}) {
  const { t } = useTranslation()
  return (
    <div className="flex flex-col gap-3 min-h-0 flex-1 overflow-y-auto">
      <section className="rounded-md border border-subtle bg-surface px-3.5 py-3 flex flex-col gap-3 shrink-0 text-sm">
        <p className="text-2xs text-fg-tertiary m-0 leading-relaxed">
          <Trans
            i18nKey="reg.aiDescription"
            values={{ n: trainImageCount }}
            components={{
              count: <span className="font-mono text-fg-primary" />,
              path: <span className="font-mono" />,
            }}
          />
        </p>

        <ExcludeTagsPicker trainTags={trainTags} excluded={excluded} onToggle={onToggle} />

        <div className="flex flex-col gap-1">
          <label className="caption">{t('reg.negPrompt')}</label>
          <textarea
            className="input font-mono text-sm resize-y"
            rows={2}
            value={neg}
            onChange={(e) => onNegChange(e.target.value)}
          />
        </div>

        <div className="flex gap-2">
          <AiNumField label={t('reg.widthLabel')} value={width} onChange={onWidthChange} min={256} max={4096} step={64} />
          <AiNumField label={t('reg.heightLabel')} value={height} onChange={onHeightChange} min={256} max={4096} step={64} />
        </div>
        <div className="flex gap-2">
          <AiNumField label={t('reg.stepsLabel')} value={steps} onChange={onStepsChange} min={1} max={150} />
          <AiNumField label="CFG Scale" value={cfg} onChange={onCfgChange} min={0} max={20} step={0.5} />
        </div>
        <AiNumField label={t('reg.seedLabel')} value={seed} onChange={onSeedChange} min={0} />

        <label className="flex items-center gap-1.5 cursor-pointer text-xs">
          <input
            type="checkbox"
            checked={incremental}
            onChange={(e) => onIncrementalChange(e.target.checked)}
          />
          <span className="text-fg-secondary">{t('reg.incrementalLabel')}</span>
        </label>

        <AiTaskStatus task={task} />
      </section>
    </div>
  )
}

function ExcludeTagsPicker({
  trainTags,
  excluded,
  onToggle,
}: {
  trainTags: RegTagCount[]
  excluded: Set<string>
  onToggle: (tag: string) => void
}) {
  const { t } = useTranslation()
  const [draft, setDraft] = useState('')
  const trainTagSet = useMemo(
    () => new Set(trainTags.map((t) => t.tag)),
    [trainTags]
  )
  // 自定义 = excluded 里那些不在 train top tag 列表里的（含画师等 train 没出现的 tag）
  const customTags = useMemo(
    () => Array.from(excluded).filter((t) => !trainTagSet.has(t)).sort(),
    [excluded, trainTagSet]
  )

  // 与后端 `_normalize_tags` 对齐：小写、空白→下划线、去重。
  const normalize = (raw: string): string =>
    raw.trim().toLowerCase().replace(/\s+/g, '_')

  const addCustom = () => {
    // 支持一次粘多个：逗号 / 空格 / 换行分隔
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

  const showTrainList = trainTags.length > 0

  return (
    <div className="space-y-2">
      {showTrainList ? (
        <div>
          <p className="text-2xs text-fg-tertiary m-0 mb-1">
            {t('reg.excludeTrainTitle')}
          </p>
          <div className="flex flex-wrap gap-1">
            {trainTags.map((tagInfo) => {
              const on = excluded.has(tagInfo.tag)
              return (
                <button
                  key={tagInfo.tag}
                  onClick={() => onToggle(tagInfo.tag)}
                  className={`px-2 py-0.5 rounded-sm border text-2xs font-mono cursor-pointer transition-colors duration-150 ${
                    on ? 'border-warn bg-warn-soft text-warn' : 'border-dim bg-sunken text-fg-secondary'
                  }`}
                  title={on ? t('reg.excludeUnclick') : t('reg.excludeClick')}
                >
                  {on ? '✕' : '+'} {tagInfo.tag.replace(/_/g, ' ')}{' '}
                  <span className="opacity-50">×{tagInfo.count}</span>
                </button>
              )
            })}
          </div>
        </div>
      ) : (
        <p className="text-xs text-fg-tertiary m-0">
          {t('reg.excludeNoTags')}
        </p>
      )}

      <div>
        <p className="text-2xs text-fg-tertiary m-0 mb-1">
          {t('reg.excludeCustomTitle')}
        </p>
        <div className="flex items-center gap-1.5">
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                addCustom()
              }
            }}
            placeholder={t('reg.excludePlaceholder')}
            className="input flex-1 text-xs"
          />
          <button
            onClick={addCustom}
            disabled={!draft.trim()}
            className="btn btn-secondary btn-sm"
          >
            {t('reg.excludeAdd')}
          </button>
        </div>
        {customTags.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1.5">
            {customTags.map((tag) => (
              <span
                key={tag}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-sm border border-warn bg-warn-soft text-warn text-2xs font-mono"
                title={t('reg.excludeCustomRemoveTitle')}
              >
                {tag.replace(/_/g, ' ')}
                <button
                  onClick={() => onToggle(tag)}
                  className="text-warn opacity-70 cursor-pointer bg-transparent border-none p-0 text-xs"
                  aria-label={t('reg.excludeCustomRemoveAria', { tag })}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
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
