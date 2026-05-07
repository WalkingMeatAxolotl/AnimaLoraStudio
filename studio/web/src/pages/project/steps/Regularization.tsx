import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useOutletContext } from 'react-router-dom'
import {
  api,
  type Job,
  type LoraEntry,
  type ProjectDetail,
  type RegAiRequest,
  type RegBuildRequest,
  type RegStatus,
  type RegTagCount,
  type Task,
  type Version,
} from '../../../api/client'
import ImageGrid from '../../../components/ImageGrid'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import JobProgress from '../../../components/JobProgress'
import PathPicker from '../../../components/PathPicker'
import StepShell from '../../../components/StepShell'
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
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()

  const [reg, setReg] = useState<RegStatus | null>(null)
  const [trainTags, setTrainTags] = useState<RegTagCount[]>([])
  // excluded 既包含 train top-tag 上点掉的，也包含「自定义排除」输入框加的（这部分
  // 在 train top-tag 列表里查不到）。后端不存这份选择，切页面回来需要按
  // (project, version) 在 localStorage 恢复，不然用户加的自定义 tag 看着就丢了。
  const [excluded, setExcluded] = useState<Set<string>>(new Set())
  const [autoTag, setAutoTag] = useState(true)
  const [apiSource, setApiSource] = useState<'gelbooru' | 'danbooru'>('gelbooru')
  const [advanced, setAdvanced] = useState<AdvancedParams>(ADVANCED_DEFAULTS)
  const [advancedOpen, setAdvancedOpen] = useState(false)

  const [job, setJob] = useState<Job | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const jobIdRef = useRef<number | null>(null)
  jobIdRef.current = job?.id ?? null

  // Tab：设置&日志 / 图片预览 / 模型生成。job done 时自动切到图片，让用户看成果。
  const [activeTab, setActiveTab] = useState<'config' | 'images' | 'ai'>('config')

  // AI 生成参数（排除 tag 复用主组件的 excluded 状态）
  const [aiNeg, setAiNeg] = useState('worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, bad anatomy')
  const [aiWidth, setAiWidth] = useState(1024)
  const [aiHeight, setAiHeight] = useState(1024)
  const [aiSteps, setAiSteps] = useState(25)
  const [aiCfg, setAiCfg] = useState(4.0)
  const [aiSeed, setAiSeed] = useState(0)
  const [aiLoras, setAiLoras] = useState<LoraEntry[]>([])
  const [aiIncremental, setAiIncremental] = useState(false)
  const [aiTask, setAiTask] = useState<Task | null>(null)
  const [aiBusy, setAiBusy] = useState(false)
  const [aiPickerIdx, setAiPickerIdx] = useState<number | null>(null)

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
      toast(`加载 reg 状态失败: ${e}`, 'error')
    }
  }, [project.id, vid, toast])

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
    if (evt.type === 'job_log_appended' && jid && evt.job_id === jid) {
      setLogs((prev) => [...prev, String(evt.text ?? '')])
    } else if (evt.type === 'job_state_changed' && jid && evt.job_id === jid) {
      void api.getJob(jid).then(setJob).catch(() => {})
      if (evt.status === 'done' || evt.status === 'failed' || evt.status === 'canceled') {
        void refreshReg()
        void reload()
        // job 成功完成 → 自动切到图片 tab，让用户看成果
        if (evt.status === 'done') setActiveTab('images')
      }
    }
  })

  // 轮询 AI 正则生成任务状态
  const aiTaskId = aiTask?.id
  useEffect(() => {
    if (aiTaskId == null) return
    let active = true
    const poll = async () => {
      if (!active) return
      try {
        const t = await api.getRegAiTask(project.id, vid!, aiTaskId)
        if (!active) return
        setAiTask(t)
        if (['done', 'failed', 'canceled'].includes(t.status)) {
          setAiBusy(false)
          void refreshReg()
          if (t.status === 'done') setActiveTab('images')
        }
      } catch { /* ignore */ }
    }
    void poll()
    const interval = setInterval(poll, 2000)
    return () => { active = false; clearInterval(interval) }
  }, [aiTaskId, project.id, vid, refreshReg])

  const handleAiGenerate = async () => {
    if (!vid) return
    setAiBusy(true)
    setAiTask(null)
    try {
      const body: RegAiRequest = {
        excluded_tags: Array.from(excluded),
        negative_prompt: aiNeg,
        width: aiWidth,
        height: aiHeight,
        steps: aiSteps,
        cfg_scale: aiCfg,
        seed: aiSeed,
        lora_configs: aiLoras.filter((l) => l.path.trim()),
        incremental: aiIncremental,
      }
      const task = await api.enqueueRegAi(project.id, vid, body)
      setAiTask(task)
      toast(`AI 正则生成任务 #${task.id} 已入队`, 'success')
    } catch (e) {
      toast(String(e), 'error')
      setAiBusy(false)
    }
  }

  const trainImageCount = activeVersion?.stats?.train_image_count ?? 0
  const isLive = job?.status === 'running' || job?.status === 'pending'

  const toggleTag = (tag: string) => {
    setExcluded((prev) => {
      const next = new Set(prev)
      if (next.has(tag)) next.delete(tag)
      else next.add(tag)
      return next
    })
  }

  const startBuild = async (incremental = false) => {
    if (!vid) return
    if (trainImageCount <= 0) {
      toast('train 还没有图片，先去 ① 整理 / ② 下载', 'error')
      return
    }
    const body: RegBuildRequest = {
      excluded_tags: Array.from(excluded),
      auto_tag: autoTag,
      api_source: apiSource,
      incremental,
      ...advanced,
    }
    try {
      const j = await api.startRegBuild(project.id, vid, body)
      setJob(j)
      setLogs([])
      toast(incremental ? `已入队补足 #${j.id}` : `已入队 #${j.id}`, 'success')
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const onDelete = async () => {
    if (!vid) return
    if (!confirm('删除当前 reg 集？这是不可恢复的（meta + 所有图片都会清掉）。')) return
    try {
      await api.deleteReg(project.id, vid)
      toast('已删除', 'success')
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
      setPreviewCaption('加载中...')
      try {
        const r = await api.getRegCaption(project.id, vid, path)
        setPreviewCaption(r.tags.length ? r.tags.join(', ') : '(无 caption)')
      } catch (e) {
        setPreviewCaption(`加载失败: ${e}`)
      }
    },
    [reg, vid, project.id]
  )

  if (!activeVersion || !vid) {
    return <p className="text-fg-tertiary p-6">请先选择 / 创建一个版本</p>
  }

  return (
    <StepShell
      idx={5}
      title="正则集"
      subtitle="基于 train tag 拉正则图，镜像结构到 reg/"
      actions={
        activeTab !== 'ai' ? (
          <button
            onClick={() => void startBuild(false)}
            disabled={isLive || trainImageCount <= 0}
            className="btn btn-primary"
          >
            {isLive ? '生成中…' : '开始生成'}
          </button>
        ) : (
          <button
            onClick={() => void handleAiGenerate()}
            disabled={aiBusy}
            className="btn btn-primary"
          >
            {aiBusy ? '生成中…' : '模型生成'}
          </button>
        )
      }
    >
    <div className="flex flex-col h-full gap-3 min-h-0">

      {/* 顶部常驻 StatusBar：reg 集快照（图片数 / target / 来源 / auto-tag /
          时间 / 补足 / 清空）—— 不进 tab，一直可见 */}
      <RegStatusBar
        reg={reg}
        onDelete={onDelete}
        onTopUp={() => void startBuild(true)}
        disabled={isLive}
      />

      {/* tab 条 */}
      <div className="flex items-center gap-1 border-b border-subtle shrink-0">
        <TabButton
          active={activeTab === 'config'}
          onClick={() => setActiveTab('config')}
          label="设置 & 日志"
          badge={isLive ? 'live' : undefined}
        />
        <TabButton
          active={activeTab === 'images'}
          onClick={() => setActiveTab('images')}
          label="图片"
          badge={reg && reg.image_count > 0 ? String(reg.image_count) : undefined}
        />
        <TabButton
          active={activeTab === 'ai'}
          onClick={() => setActiveTab('ai')}
          label="模型生成"
          badge={aiBusy ? 'live' : aiTask?.status === 'done' ? '✓' : undefined}
        />
      </div>

      {/* tab 内容（占满剩余高度，全宽） */}
      {activeTab === 'ai' ? (
        <AiGenPanel
          trainTags={trainTags}
          excluded={excluded} onToggle={toggleTag}
          neg={aiNeg} onNegChange={setAiNeg}
          width={aiWidth} onWidthChange={setAiWidth}
          height={aiHeight} onHeightChange={setAiHeight}
          steps={aiSteps} onStepsChange={setAiSteps}
          cfg={aiCfg} onCfgChange={setAiCfg}
          seed={aiSeed} onSeedChange={setAiSeed}
          loras={aiLoras} onLorasChange={setAiLoras}
          incremental={aiIncremental} onIncrementalChange={setAiIncremental}
          pickerIdx={aiPickerIdx} onPickerIdxChange={setAiPickerIdx}
          task={aiTask}
          trainImageCount={trainImageCount}
        />
      ) : activeTab === 'config' ? (
        <div className="flex flex-col gap-3 min-h-0 flex-1 overflow-y-auto">
          <section className="rounded-md border border-subtle bg-surface px-3.5 py-2.5 flex flex-col gap-2.5 shrink-0">
            <div className="flex flex-wrap items-center gap-2.5 text-xs">
              <span className="text-fg-tertiary">来源</span>
              <select
                value={apiSource}
                onChange={(e) => setApiSource(e.target.value as 'gelbooru' | 'danbooru')}
                className="input px-2 py-0.5 text-sm"
              >
                <option value="gelbooru">Gelbooru</option>
                <option value="danbooru">Danbooru</option>
              </select>
              <span className="text-fg-tertiary">|</span>
              <span className="text-fg-tertiary">
                目标数量{' '}
                <span className="font-mono text-fg-primary font-medium">{trainImageCount}</span>
                <span className="text-fg-tertiary">（镜像 train）</span>
              </span>
              <span className="text-fg-tertiary">|</span>
              <label className="flex items-center gap-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoTag}
                  onChange={(e) => setAutoTag(e.target.checked)}
                />
                <span className="text-fg-secondary">拉完后自动 WD14 打标</span>
              </label>
              <button
                onClick={() => setAdvancedOpen((v) => !v)}
                className="text-fg-tertiary bg-transparent border-none cursor-pointer text-xs"
              >
                {advancedOpen ? '⌃ 进阶' : '⌄ 进阶'}
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
                  toast('已取消', 'success')
                } catch (e) {
                  toast(String(e), 'error')
                }
              }}
            />
          )}
        </div>
      ) : (
        // 图片 tab：占满剩余高度
        reg && reg.image_count > 0 ? (
          <RegPreview
            pid={project.id}
            vid={vid}
            reg={reg}
            onPick={(idx) => void openPreview(idx)}
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
              还没有 reg 集
            </div>
            <div style={{ fontSize: 'var(--t-xs)' }}>
              在「设置 &amp; 日志」配置后点击右上角「开始生成」
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
  onTopUp,
  disabled,
}: {
  reg: RegStatus | null
  onDelete: () => void
  onTopUp: () => void
  disabled: boolean
}) {
  if (!reg) {
    return (
      <section className="rounded-sm border border-subtle bg-surface px-2.5 py-1.5 text-xs text-fg-tertiary shrink-0">
        加载中...
      </section>
    )
  }
  if (!reg.exists) {
    return (
      <section className="rounded-sm border border-subtle bg-surface px-2.5 py-1.5 text-xs text-fg-tertiary shrink-0">
        当前版本 reg 集：<span className="text-fg-tertiary">不存在</span>
      </section>
    )
  }
  const m = reg.meta
  const ago = m ? formatAgo(m.generated_at) : '?'
  const shortfall = m ? m.target_count - m.actual_count : 0
  const canTopUp = m !== null && shortfall > 0
  return (
    <section className="rounded-sm border border-subtle bg-surface px-3 py-2 flex flex-col gap-1 shrink-0">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-fg-secondary">
          reg 集存在：
          <span className="font-mono text-ok font-medium">{reg.image_count} 张</span>
        </span>
        {m && (
          <>
            <span className="text-fg-tertiary">·</span>
            <span className="text-fg-tertiary">
              target {m.actual_count}/{m.target_count}
            </span>
            <span className="text-fg-tertiary">·</span>
            <span className="text-fg-tertiary">{m.api_source}</span>
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
                title={`搜索失败的 tag: ${m.failed_tags.join(', ')}`}
              >
                · {m.failed_tags.length} 失败 tag
              </span>
            )}
            {m.incremental_runs > 0 && (
              <span className="text-fg-tertiary" title="补足跑过的次数">
                · 补足 ×{m.incremental_runs}
              </span>
            )}
          </>
        )}
        <span className="flex-1" />
        {canTopUp && (
          <button
            onClick={onTopUp}
            disabled={disabled}
            className="btn btn-sm text-accent bg-accent-soft border-accent"
            title={`保留已下 ${m!.actual_count} 张，补足 ${shortfall} 张`}
          >
            补足 +{shortfall}
          </button>
        )}
        <button
          onClick={onDelete}
          disabled={disabled}
          className="btn btn-sm bg-err-soft text-err border-err"
        >
          清空
        </button>
      </div>

      {m && (
        <div className="flex flex-wrap items-center gap-2 text-2xs text-fg-tertiary">
          <span>分辨率聚类：</span>
          {m.postprocess_clusters !== null ? (
            <span
              className="text-fg-secondary"
              title={`方法 ${m.postprocess_method}，max_crop ${m.postprocess_max_crop_ratio}`}
            >
              {m.postprocess_clusters} 类（{m.postprocess_method},{' '}
              max_crop {m.postprocess_max_crop_ratio}）
            </span>
          ) : (
            <span
              className="text-fg-tertiary"
              title="分辨率差异过大或未启用 — 训练靠 bucketing 处理"
            >
              未聚类
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
  const set = <K extends keyof AdvancedParams>(k: K, v: AdvancedParams[K]) =>
    onChange({ ...value, [k]: v })
  return (
    <div className="rounded-sm border border-subtle bg-sunken px-3.5 py-2.5 flex flex-col gap-2.5 text-xs">
      <p className="text-2xs text-fg-tertiary m-0">
        保持默认即可
      </p>

      {/* 选图 */}
      <Group label="选图">
        <label className="flex items-center gap-1 cursor-pointer">
          <input
            type="checkbox"
            checked={value.skip_similar}
            onChange={(e) => set('skip_similar', e.target.checked)}
          />
          <span
            className="text-fg-secondary"
            title="候选只取偶数索引，避免相邻相似图（默认 ✓）"
          >
            skip_similar
          </span>
        </label>
      </Group>

      {/* 长宽比过滤 */}
      <Group label="长宽比过滤">
        <label className="flex items-center gap-1 cursor-pointer">
          <input
            type="checkbox"
            checked={value.aspect_ratio_filter_enabled}
            onChange={(e) => set('aspect_ratio_filter_enabled', e.target.checked)}
          />
          <span className="text-fg-secondary">启用</span>
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
              过滤极端长宽比图（例：0.5–2.0 = 1:2 到 2:1）
            </span>
          </>
        )}
      </Group>

      {/* 后处理 */}
      <Group label="后处理">
        <span className="text-fg-tertiary">方法</span>
        <select
          value={value.postprocess_method}
          onChange={(e) =>
            set('postprocess_method', e.target.value as 'smart' | 'stretch' | 'crop')
          }
          className="input px-1.5 py-px text-xs"
        >
          <option value="smart">smart（缩放+居中裁，推荐）</option>
          <option value="stretch">stretch（拉伸，可能变形）</option>
          <option value="crop">crop（先裁后缩）</option>
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
            title="单聚类内最大允许裁剪比例（默认 0.1 = 10%）"
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

function ExcludeTagsPicker({
  trainTags,
  excluded,
  onToggle,
}: {
  trainTags: RegTagCount[]
  excluded: Set<string>
  onToggle: (tag: string) => void
}) {
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
            排除 train top tag：
          </p>
          <div className="flex flex-wrap gap-1">
            {trainTags.map((t) => {
              const on = excluded.has(t.tag)
              return (
                <button
                  key={t.tag}
                  onClick={() => onToggle(t.tag)}
                  className={`px-2 py-0.5 rounded-sm border text-2xs font-mono cursor-pointer transition-colors duration-150 ${
                    on ? 'border-warn bg-warn-soft text-warn' : 'border-dim bg-sunken text-fg-secondary'
                  }`}
                  title={on ? '点击取消排除' : '点击加入排除'}
                >
                  {on ? '✕' : '+'} {t.tag}{' '}
                  <span className="opacity-50">×{t.count}</span>
                </button>
              )
            })}
          </div>
        </div>
      ) : (
        <p className="text-xs text-fg-tertiary m-0">
          train 还没有 tag 分布。也可以仅靠下方「自定义排除」继续。
        </p>
      )}

      <div>
        <p className="text-2xs text-fg-tertiary m-0 mb-1">
          自定义排除：
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
            placeholder="输入 tag，回车添加"
            className="input flex-1 text-xs"
          />
          <button
            onClick={addCustom}
            disabled={!draft.trim()}
            className="btn btn-secondary btn-sm"
          >
            + 添加
          </button>
        </div>
        {customTags.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1.5">
            {customTags.map((t) => (
              <span
                key={t}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-sm border border-warn bg-warn-soft text-warn text-2xs font-mono"
                title="自定义排除（点 × 移除）"
              >
                {t}
                <button
                  onClick={() => onToggle(t)}
                  className="text-warn opacity-70 cursor-pointer bg-transparent border-none p-0 text-xs"
                  aria-label={`移除 ${t}`}
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
  onPick,
}: {
  pid: number
  vid: number
  reg: RegStatus
  onPick: (idx: number) => void
}) {
  // reg.files 是相对 reg/ 的路径（含子文件夹镜像 train，例如 "5_concept/2001.png"）
  const items = useMemo(
    () =>
      reg.files.map((rel) => {
        const idx = rel.lastIndexOf('/')
        const folder = idx >= 0 ? rel.slice(0, idx) : ''
        const name = idx >= 0 ? rel.slice(idx + 1) : rel
        return {
          name: rel,
          thumbUrl: api.versionThumbUrl(pid, vid, 'reg', name, folder),
        }
      }),
    [reg.files, pid, vid]
  )
  const indexByName = useMemo(() => {
    const m = new Map<string, number>()
    items.forEach((it, i) => m.set(it.name, i))
    return m
  }, [items])
  return (
    <section className="rounded-md border border-subtle bg-surface p-2 flex-1 min-h-0 overflow-y-auto">
      <p className="text-2xs text-fg-tertiary px-1 pb-1 m-0">
        reg/（共 {reg.image_count} 张）— 点击查看大图 + caption
      </p>
      <ImageGrid
        items={items}
        selected={new Set()}
        onSelect={(name) => {
          const i = indexByName.get(name)
          if (i !== undefined) onPick(i)
        }}
        ariaLabel="reg-preview"
      />
    </section>
  )
}

// ── AiGenPanel ──────────────────────────────────────────────────────────────

function AiNumField({ label, value, onChange, min, max, step }: {
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
  if (!task) return null
  const label =
    task.status === 'done' ? '已完成' :
    task.status === 'running' ? '生成中' :
    task.status === 'failed' ? '失败' :
    task.status === 'pending' ? '排队中' :
    task.status === 'canceled' ? '已取消' : task.status
  const cls =
    task.status === 'done' ? 'badge badge-ok' :
    task.status === 'running' ? 'badge badge-info' :
    task.status === 'failed' ? 'badge badge-err' : 'badge'
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className={cls}>{label}</span>
      <span className="caption font-mono">#{task.id}</span>
      {task.status === 'done' && (
        <span className="text-xs text-fg-tertiary">已写入 reg 对应子目录</span>
      )}
      {task.status === 'failed' && task.error_msg && (
        <span className="text-xs text-err">{task.error_msg}</span>
      )}
    </div>
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
  loras, onLorasChange,
  incremental, onIncrementalChange,
  pickerIdx, onPickerIdxChange,
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
  loras: LoraEntry[]; onLorasChange: (v: LoraEntry[]) => void
  incremental: boolean; onIncrementalChange: (v: boolean) => void
  pickerIdx: number | null; onPickerIdxChange: (v: number | null) => void
  task: Task | null
  trainImageCount: number
}) {
  const addLora = () => onLorasChange([...loras, { path: '', scale: 1.0 }])
  const delLora = (i: number) => onLorasChange(loras.filter((_, idx) => idx !== i))
  const setLoraPath = (i: number, path: string) =>
    onLorasChange(loras.map((l, idx) => idx === i ? { ...l, path } : l))
  const setLoraScale = (i: number, scale: number) =>
    onLorasChange(loras.map((l, idx) => idx === i ? { ...l, scale } : l))

  return (
    <div className="flex flex-col gap-3 min-h-0 flex-1 overflow-y-auto">
      <section className="rounded-md border border-subtle bg-surface px-3.5 py-3 flex flex-col gap-3 shrink-0 text-sm">
        <p className="text-2xs text-fg-tertiary m-0">
          逐图生成：为 train 每张图（共{' '}
          <span className="font-mono text-fg-primary">{trainImageCount}</span>
          {' '}张）的 tag 生成对应正则图，写入 <span className="font-mono">reg/{'{subfolder}'}/</span>。
          排除 tag 与左侧「Booru」共用，修改后两边同步生效。
        </p>

        {/* 排除 tag（复用 ExcludeTagsPicker） */}
        <ExcludeTagsPicker trainTags={trainTags} excluded={excluded} onToggle={onToggle} />

        {/* 负面提示词 */}
        <div className="flex flex-col gap-1">
          <label className="caption">负面提示词</label>
          <textarea
            className="input font-mono text-sm resize-y"
            rows={2}
            value={neg}
            onChange={(e) => onNegChange(e.target.value)}
          />
        </div>

        {/* 数值参数 */}
        <div className="flex gap-2">
          <AiNumField label="宽度" value={width} onChange={onWidthChange} min={256} max={4096} step={64} />
          <AiNumField label="高度" value={height} onChange={onHeightChange} min={256} max={4096} step={64} />
        </div>
        <div className="flex gap-2">
          <AiNumField label="步数" value={steps} onChange={onStepsChange} min={1} max={150} />
          <AiNumField label="CFG Scale" value={cfg} onChange={onCfgChange} min={0} max={20} step={0.5} />
        </div>
        <AiNumField label="种子（0=随机）" value={seed} onChange={onSeedChange} min={0} />

        {/* 补足模式 */}
        <label className="flex items-center gap-1.5 cursor-pointer text-xs">
          <input
            type="checkbox"
            checked={incremental}
            onChange={(e) => onIncrementalChange(e.target.checked)}
          />
          <span className="text-fg-secondary">补足模式（跳过 reg 目录中已有对应文件的图）</span>
        </label>

        {/* LoRA */}
        <div className="flex flex-col gap-2">
          <label className="caption">LoRA（可选）</label>
          {loras.map((l, i) => (
            <div key={i} className="flex gap-1.5 items-center">
              <div className="flex-1 flex gap-1 items-center bg-sunken border border-dim rounded-md px-2 py-1.5">
                <span className="text-xs text-fg-tertiary shrink-0 w-4 text-center font-mono">{i + 1}</span>
                <input
                  type="text"
                  className="input input-mono flex-1 border-0 bg-transparent p-0 text-xs"
                  style={{ outline: 'none', boxShadow: 'none' }}
                  placeholder="LoRA 路径…"
                  value={l.path}
                  onChange={(e) => setLoraPath(i, e.target.value)}
                />
                <button
                  onClick={() => onPickerIdxChange(i)}
                  className="btn btn-ghost btn-sm text-xs shrink-0 px-1.5"
                  title="浏览文件"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
                  </svg>
                </button>
              </div>
              <div className="flex items-center gap-1 shrink-0">
                <span className="text-xs text-fg-tertiary">×</span>
                <input
                  type="number"
                  className="input text-center text-sm"
                  style={{ width: 60, padding: '5px 6px' }}
                  min={0} max={2} step={0.05}
                  value={l.scale}
                  onChange={(e) => setLoraScale(i, Number(e.target.value))}
                  title="权重倍率"
                />
              </div>
              <button onClick={() => delLora(i)} className="btn btn-ghost btn-sm text-fg-tertiary hover:text-err shrink-0 px-1.5">×</button>
            </div>
          ))}
          <button onClick={addLora} className="btn btn-ghost btn-sm self-start text-xs text-fg-tertiary">
            + 添加 LoRA
          </button>
        </div>

        {/* 任务状态 */}
        <AiTaskStatus task={task} />
      </section>

      {pickerIdx !== null && (
        <PathPicker
          dirOnly={false}
          onPick={(p) => { setLoraPath(pickerIdx, p); onPickerIdxChange(null) }}
          onClose={() => onPickerIdxChange(null)}
        />
      )}
    </div>
  )
}

function regOrigUrl(pid: number, vid: number, rel: string): string {
  const idx = rel.lastIndexOf('/')
  const folder = idx >= 0 ? rel.slice(0, idx) : ''
  const name = idx >= 0 ? rel.slice(idx + 1) : rel
  // 768px 预览（与 PP3 alt-hover 同尺寸）
  return api.versionThumbUrl(pid, vid, 'reg', name, folder, 768)
}

function formatAgo(unix: number): string {
  const now = Date.now() / 1000
  const dt = now - unix
  if (dt < 60) return '刚刚'
  if (dt < 3600) return `${Math.floor(dt / 60)} 分钟前`
  if (dt < 86400) return `${Math.floor(dt / 3600)} 小时前`
  return `${Math.floor(dt / 86400)} 天前`
}
