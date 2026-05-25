import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  api,
  type GenerateRequest,
  type LoraEntry,
  type Task,
  type XYMatrixSpec,
} from '../../api/client'
import PageHeader from '../../components/PageHeader'
import { useToast } from '../../components/Toast'
import { useEventStream } from '../../lib/useEventStream'
import { useMonitorProgress } from '../../lib/useMonitorProgress'
import { useLocalStorageState } from '../../lib/useLocalStorageState'
import AspectChips, { aspectFromDimensions, type AspectName } from './generate/AspectChips'
import DaemonControls from './generate/DaemonControls'
import DaemonLogDrawer from './generate/DaemonLogDrawer'
import GenerateProgressBar, { type GenerateProgress } from './generate/GenerateProgress'
import NumField from './generate/NumField'
import PreviewCompare from './generate/PreviewCompare'
import PreviewHistoryRail from './generate/PreviewHistoryRail'
import PromptFromDatasetPicker, { type DatasetPick } from './generate/PromptFromDatasetPicker'
import { makeThumbnail, useGenerateHistory, type HistoryEntry } from './generate/useGenerateHistory'
import PreviewXYGrid from './generate/PreviewXYGrid'
import PromptList from './generate/PromptList'
import SampleGallery from './generate/SampleGallery'
import SidebarLoras from './generate/SidebarLoras'
import SidebarXYAxes from './generate/SidebarXYAxes'
import StatusBadge from './generate/StatusBadge'
import ViewModeTabs, { type ViewMode } from './generate/ViewModeTabs'
import { DEFAULT_NEG } from './generate/types'
import { useProjectLoras } from './generate/useProjectLoras'
import { cellCount, draftToSpec, parseAxisValues, type XYAxisDraft } from './generate/xy'

const GENERATE_PREFS_KEY = 'studio:generate:params:v1'

const DEFAULT_GENERATE_PREFS = {
  mode: 'single' as ViewMode,
  prompts: ['newest, safe, 1girl, masterpiece, best quality'],
  negPrompt: DEFAULT_NEG,
  aspect: '1:1' as AspectName,
  width: 1024,
  height: 1024,
  steps: 25,
  cfgScale: 4.0,
  count: 1,
  seed: 0,
  loras: [] as LoraEntry[],
  xDraft: { axis: 'steps', raw: '20, 25, 30', loraIndex: null } as XYAxisDraft,
  yDraft: null as XYAxisDraft | null,
  datasetPick: null as DatasetPick | null,
}

export default function GeneratePage() {
  const { t } = useTranslation()
  const { toast } = useToast()

  const [prefs, setPrefs] = useLocalStorageState(GENERATE_PREFS_KEY, DEFAULT_GENERATE_PREFS)
  const { mode, prompts, negPrompt, aspect, width, height, steps, cfgScale, count, seed, loras, xDraft, yDraft, datasetPick } = prefs
  const setMode = (mode: ViewMode) => setPrefs((p) => ({ ...p, mode }))
  const setPrompts = (prompts: string[]) => setPrefs((p) => ({ ...p, prompts }))
  const setNegPrompt = (negPrompt: string) => setPrefs((p) => ({ ...p, negPrompt }))
  const setAspect = (aspect: AspectName) => setPrefs((p) => ({ ...p, aspect }))
  const setWidth = (width: number) => setPrefs((p) => ({ ...p, width }))
  const setHeight = (height: number) => setPrefs((p) => ({ ...p, height }))
  const setSteps = (steps: number) => setPrefs((p) => ({ ...p, steps }))
  const setCfgScale = (cfgScale: number) => setPrefs((p) => ({ ...p, cfgScale }))
  const setCount = (count: number) => setPrefs((p) => ({ ...p, count }))
  const setSeed = (seed: number) => setPrefs((p) => ({ ...p, seed }))
  const setLoras = (loras: LoraEntry[]) => setPrefs((p) => ({ ...p, loras }))

  // LoRA 预填 via URL query (?lora=<path>&projectId=N&versionId=N)
  // Overview StatusBanner "在测试中加载" CTA 跳进来时，把 LoRA 直接塞入 loras。
  // 用 history.replaceState 清掉 query 避免刷新时重复触发。
  useEffect(() => {
    const sp = new URLSearchParams(window.location.search)
    const lora = sp.get('lora')
    if (!lora) return
    const projectId = sp.get('projectId')
    const versionId = sp.get('versionId')
    setPrefs((p) => {
      if (p.loras.some((l) => l.path === lora)) return p
      return {
        ...p,
        loras: [...p.loras, {
          path: lora,
          scale: 1.0,
          project_id: projectId ? Number(projectId) : null,
          version_id: versionId ? Number(versionId) : null,
        }],
      }
    })
    const url = new URL(window.location.href)
    url.searchParams.delete('lora')
    url.searchParams.delete('projectId')
    url.searchParams.delete('versionId')
    window.history.replaceState({}, '', url.toString())
  }, [])
  // commit C: attention backend 已从 Generate 页移到 Settings；server 端
  // enqueue_generate 会自动从 secrets.generate.attention_backend 注入。

  const setXDraft = (xDraft: XYAxisDraft) => setPrefs((p) => ({ ...p, xDraft }))
  const setYDraft = (yDraft: XYAxisDraft | null) => setPrefs((p) => ({ ...p, yDraft }))
  const setDatasetPick = (datasetPick: DatasetPick | null) => setPrefs((p) => ({ ...p, datasetPick }))

  // 双图对比：选中的 2 个 sample 索引（从 PreviewXYGrid cell click 收集）
  const [selectedIndices, setSelectedIndices] = useState<number[]>([])

  // submitting：HTTP 入队中（短暂窗口，currentTask 还没回来）
  // busy 派生自 currentTask.status，避免靠 setBusy(false) 清状态卡 UI——
  // 之前用 useState 时遇过 SSE 漏事件 / race 后 busy=true 卡住，按钮 disabled
  // 没法重试也没法取消（status=failed 时 cancelable=false）
  const [submitting, setSubmitting] = useState(false)
  const [currentTask, setCurrentTask] = useState<Task | null>(null)
  // monitor 走 useMonitorProgress hook (PR #37 增量协议)：currentTask 变 →
  // hook 自动重拉快照 + 订阅 SSE delta 合并；本组件只用 samples 字段，其余
  // 字段在这页生成场景下不需要。
  const { state: monitorState } = useMonitorProgress(currentTask?.id ?? null)
  // commit 14：中间步预览（仅 single 模式有意义；XY/对比 cell 多预览意义小）
  const [previewStep, setPreviewStep] = useState<{ step: number; total: number; dataUrl: string } | null>(null)
  // 生成进度（image_started + preview_step 聚合）
  const [progress, setProgress] = useState<GenerateProgress>({
    batchIdx: null, batchTotal: null, currentStep: null, totalSteps: null,
  })
  const [datasetPickerOpen, setDatasetPickerOpen] = useState(false)
  const [logOpen, setLogOpen] = useState(false)
  // commit 16：图片历史栏。点击历史项 → 主预览替换为该项封面
  const history = useGenerateHistory()
  const [historyOverride, setHistoryOverride] = useState<HistoryEntry | null>(null)
  const taskIdRef = useRef<number | null>(null)
  taskIdRef.current = currentTask?.id ?? null
  const lastSnapshotRef = useRef<{ taskId: number; mode: ViewMode } | null>(null)

  // 切到 single 时清掉 XY 选择（与 XY 结果绑定，单图模式无意义）
  useEffect(() => {
    if (mode === 'single') setSelectedIndices([])
  }, [mode])

  // 选 2 张 → 自动切到 compare；toggle 已选项；满 2 时新点替换最旧
  const handleCellClick = (idx: number) => {
    setSelectedIndices((prev) => {
      if (prev.includes(idx)) return prev.filter((i) => i !== idx)
      if (prev.length >= 2) return [prev[1], idx]
      const next = [...prev, idx]
      // 选 2 张自动进入 xy 内部的 compare sub-view（不切顶部 mode）
      // 当前 mode 已经是 'xy'（cell click 仅 xy mode 触发），无需 setMode
      return next
    })
  }

  // xy mode 内部 selectedIndices=2 时切 compare sub-view
  const showCompareView = mode === 'xy' && selectedIndices.length === 2

  const projectLoras = useProjectLoras()
  // 用 useMemo 稳定引用：monitorState 不变时 samples 引用不变，避免下方
  // useEffect 把 samples 当依赖触发不必要的重跑
  const samples = useMemo(() => monitorState?.samples ?? [], [monitorState])

  // XY mode 时，按钮显示「生成 N×M=K 张」
  const xyCellCount = useMemo(() => {
    if (mode !== 'xy') return 0
    try {
      const xLen = parseAxisValues(xDraft.axis, xDraft.raw).length
      const yLen = yDraft ? parseAxisValues(yDraft.axis, yDraft.raw).length : null
      return cellCount(xLen, yLen)
    } catch {
      return 0
    }
  }, [mode, xDraft, yDraft])

  // SSE：task_state_changed 触发 task refresh；monitor_state_updated 推 sample 列表。
  useEventStream((evt) => {
    const tid = taskIdRef.current
    if (tid == null) return
    if (evt.type === 'task_state_changed' && evt.task_id === tid) {
      void api.getGenerateTask(tid).then((t) => {
        setCurrentTask(t)
        if (t.status === 'done' || t.status === 'failed' || t.status === 'canceled') {
          // busy 已是派生自 status，无需 setBusy；只清进度防残留
          setProgress({ batchIdx: null, batchTotal: null, currentStep: null, totalSteps: null })
        }
      }).catch(() => { /* task 已清也走这里 */ })
    } else if (
      evt.type === 'generate_preview_step'
      && String(evt.task_id) === String(tid)
    ) {
      const step = Number(evt.step) || 0
      const total = Number(evt.total) || 0
      // 进度永远更新
      setProgress((p) => ({ ...p, currentStep: step, totalSteps: total }))
      // image_b64 是可选的（settings 没开预览时无）
      if (typeof evt.image_b64 === 'string') {
        setPreviewStep({
          step, total,
          dataUrl: `data:image/jpeg;base64,${evt.image_b64}`,
        })
      }
    } else if (
      evt.type === 'generate_image_started'
      && String(evt.task_id) === String(tid)
    ) {
      // 新 batch 开始 → 重置 step 进度，更新 batch 计数
      setProgress({
        batchIdx: typeof evt.batch_idx === 'number' ? evt.batch_idx : null,
        batchTotal: typeof evt.batch_total === 'number' ? evt.batch_total : null,
        currentStep: 0,
        totalSteps: typeof evt.total_steps === 'number' ? evt.total_steps : null,
      })
    }
  })

  // task 切换 / 完成 / 切 mode 时清掉中间预览（最终图覆盖）
  useEffect(() => {
    setPreviewStep(null)
  }, [currentTask?.id, mode, samples.length])

  // 切 task / 切 mode 时清掉历史回看 override（让主预览跟着走当前 task）
  useEffect(() => {
    setHistoryOverride(null)
  }, [currentTask?.id, mode])

  // task done + 有样本 → 入库历史。lastSnapshotRef 防同 task 多次触发
  // 之前 dedup 还比 mode → 用户切 mode 时同 task 反复入库（"历史克隆"bug）。
  // 修：只 dedup taskId；entry.mode 记当时生成时的 mode，不被切 mode 影响。
  useEffect(() => {
    if (!currentTask || currentTask.status !== 'done') return
    if (samples.length === 0) return
    const snap = lastSnapshotRef.current
    if (snap?.taskId === currentTask.id) return
    lastSnapshotRef.current = { taskId: currentTask.id, mode }
    const taskId = currentTask.id
    // 选封面 sample
    let coverIdx = 0
    // XY：找 (xi=0, yi=0) 那张；找不到 fallback 0
    if (mode === 'xy') {
      const found = samples.findIndex(
        (s) => s.xy && s.xy.xi === 0 && s.xy.yi === 0
      )
      if (found >= 0) coverIdx = found
    }
    const cover = samples[coverIdx]
    if (!cover) return
    const filename = (cover.path.split(/[\\/]/).pop() ?? '')
    if (!filename) return
    // badge
    let badge = ''
    if (mode === 'xy') {
      const xs = new Set(samples.map((s) => s.xy?.xi).filter((x) => x !== undefined))
      const ys = new Set(samples.map((s) => s.xy?.yi).filter((x) => x !== undefined))
      badge = `XY ${xs.size}×${ys.size || 1}`
    }
    const filenames = samples
      .map((s) => s.path.split(/[\\/]/).pop() ?? '')
      .filter(Boolean)
    // commit: xy 历史回看用 PreviewXYGrid 重建网格 → 入库时收集 axis + sample 元数据
    let xyMeta: import('./generate/useGenerateHistory').HistoryXYMeta | undefined
    if (mode === 'xy') {
      const xValues = xDraft.raw.split(',').map((s) => s.trim()).filter(Boolean)
      const yValues = yDraft
        ? yDraft.raw.split(',').map((s) => s.trim()).filter(Boolean)
        : [null as string | null]
      const xySamples = samples
        .filter((s): s is typeof s & { xy: NonNullable<typeof s.xy> } => s.xy != null)
        .map((s) => ({
          path: s.path,
          xy: {
            xi: s.xy.xi, yi: s.xy.yi,
            xv: s.xy.xv ?? '', yv: s.xy.yv ?? null,
          },
        }))
      xyMeta = {
        xAxis: xDraft.axis, yAxis: yDraft?.axis ?? null,
        xValues, yValues, samples: xySamples,
      }
    }
    void makeThumbnail(api.generateSampleUrl(taskId, filename), 256)
      .then((dataUrl) => history.add({
        mode,
        taskId,
        thumbnailDataUrl: dataUrl,
        filenames,
        badge: badge || undefined,
        xy: xyMeta,
      }))
      .catch(() => { /* thumbnail 失败 — 不入库（避免无封面 entry） */ })
  }, [currentTask, samples, mode, selectedIndices, history, xDraft, yDraft])

  const handleHistorySelect = (entry: HistoryEntry) => {
    setHistoryOverride(entry)
  }

  const handleGenerate = async () => {
    const datasetSuffix = datasetPick && datasetPick.tags.length > 0
      ? datasetPick.tags.join(', ')
      : ''
    if (!prompts.some((p) => p.trim()) && !datasetSuffix) {
      toast(t('generate.promptOrDatasetRequired'), 'error')
      return
    }

    let xy_matrix: XYMatrixSpec | null = null
    if (mode === 'xy') {
      // schema 强制 prompts 单条 + count=1
      if (prompts.filter((p) => p.trim()).length > 1) {
        toast(t('generate.xySinglePromptOnly'), 'error')
        return
      }
      const filteredLoras = loras.filter((l) => l.path.trim())
      try {
        xy_matrix = {
          x: draftToSpec(xDraft, filteredLoras),
          y: yDraft ? draftToSpec(yDraft, filteredLoras) : null,
        }
      } catch (e) {
        toast(typeof e === 'string' ? e : String(e), 'error')
        return
      }
    }

    setSubmitting(true)
    setCurrentTask(null)
    // monitorState 由 useMonitorProgress hook 自动随 currentTask 切 null → 清空
    setSelectedIndices([])  // 新一轮生成 — 旧选择已失效
    setProgress({ batchIdx: null, batchTotal: null, currentStep: null, totalSteps: null })
    try {
      // 拼接顺序：手写正向在前，dataset tags 在后（与产品约定一致）
      const baseTrimmed = prompts.map((p) => p.trim()).filter((p) => p)
      const mergedPrompts = datasetSuffix
        ? (baseTrimmed.length > 0
            ? baseTrimmed.map((p) => `${p}, ${datasetSuffix}`)
            : [datasetSuffix])
        : baseTrimmed
      const body: GenerateRequest = {
        prompts: mergedPrompts,
        negative_prompt: negPrompt,
        width, height, steps,
        count: mode === 'xy' ? 1 : count,
        seed,
        cfg_scale: cfgScale,
        lora_configs: loras.filter((l) => l.path.trim()),
        // attention_backend 不带：server 自动从 secrets.generate.attention_backend 读
        xy_matrix,
      }
      const task = await api.enqueueGenerate(body)
      // 立即同步 ref，避免 supervisor 在 enqueue 返回 → setCurrentTask 渲染
      // 之间已经处理完任务并发了 task_state_changed 事件（config 缺失这种
      // 早期失败会马上发 SSE，handler 拿 taskIdRef 还是 null → 漏事件）
      taskIdRef.current = task.id
      setCurrentTask(task)
      toast(t('generate.taskEnqueued', { id: task.id }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setSubmitting(false)
    }
  }

  const handleCancel = async () => {
    if (!currentTask) return
    try {
      await api.cancelTask(currentTask.id)
      toast(t('generate.cancelRequested', { id: currentTask.id }), 'info')
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const cancelable = currentTask
    && (currentTask.status === 'pending' || currentTask.status === 'running')

  // busy 派生：HTTP 入队中 OR 任务还在 pending/running。terminal status
  //（done/failed/canceled）一律 busy=false，让 button 立刻可点重试
  const busy: boolean = submitting || Boolean(cancelable)

  const generateLabel = busy
    ? t('generate.generating')
    : mode === 'xy' && xyCellCount > 0
      ? t('generate.startGenerateCount', { n: xyCellCount })
      : t('generate.startGenerate')

  return (
    <div className="fade-in flex flex-col" style={{ height: '100%', overflow: 'hidden' }}>
      <PageHeader
        title={t('generate.title')}
        subtitle={t('generate.subtitle')}
        actions={<DaemonControls onToggleLog={() => setLogOpen((v) => !v)} />}
      />

      {/* 三列各自独立滚动，整页固定高度 = viewport */}
      <div className="p-6 flex gap-4 items-stretch flex-wrap xl:flex-nowrap flex-1 min-h-0">

          {/* 左：sidebar — 上半部分独立 scroll，Generate bar 固定底部始终可见 */}
          <div className="flex flex-col gap-4 w-full xl:w-[420px] shrink-0 self-stretch min-h-0">
          <div className="flex flex-col gap-4 flex-1 min-h-0 overflow-y-auto pr-2">

            {/* mode=single：独立 LoRA 卡片；mode=xy：LoRA 选择合并到 XY 卡片顶部 */}
            {mode === 'single' && (
              <div className="card" style={{ padding: 18 }}>
                <div className="flex items-baseline justify-between mb-3">
                  <h3 className="m-0 text-md font-semibold">LoRA</h3>
                  <span className="text-xs text-fg-tertiary">{t('generate.loraHint')}</span>
                </div>
                <SidebarLoras loras={loras} onChange={setLoras} projectLoras={projectLoras} />
              </div>
            )}

            {mode === 'xy' && (
              <SidebarXYAxes
                xDraft={xDraft}
                yDraft={yDraft}
                onXChange={setXDraft}
                onYChange={setYDraft}
                loras={loras}
                onLorasChange={setLoras}
                projectLoras={projectLoras}
              />
            )}

            <div className="card" style={{ padding: 18 }}>
              <div className="flex items-baseline justify-between mb-3">
                <h3 className="m-0 text-md font-semibold">{t('generate.prompts')}</h3>
                {!datasetPickerOpen && (
                  <button
                    onClick={() => setDatasetPickerOpen(true)}
                    className="btn btn-ghost text-xs text-fg-tertiary"
                    title={t('generate.pickFromDatasetTitle')}
                  >
                    {t('generate.pickFromDataset')}
                  </button>
                )}
              </div>
              {datasetPickerOpen && (
                <div className="mb-3">
                  <PromptFromDatasetPicker
                    value={datasetPick}
                    onChange={setDatasetPick}
                    onClose={() => {
                      setDatasetPick(null)
                      setDatasetPickerOpen(false)
                    }}
                  />
                </div>
              )}
              <label className="caption block mb-1">{t('generate.positive')}</label>
              <PromptList prompts={prompts} onChange={setPrompts} />
              <label className="caption block mb-1 mt-3">{t('generate.negative')}</label>
              <textarea
                className="input w-full font-mono text-xs resize-y"
                rows={5}
                value={negPrompt}
                onChange={(e) => setNegPrompt(e.target.value)}
              />
            </div>

            <div className="card" style={{ padding: 18 }}>
              <h3 className="m-0 text-md font-semibold mb-3">{t('generate.samplingParams')}</h3>
              <div className="flex flex-col gap-3">
                <div>
                  <label className="caption block mb-1.5">{t('generate.aspect')}</label>
                  <AspectChips
                    aspect={aspect}
                    onPick={(a, w, h) => {
                      setAspect(a)
                      if (w && h) { setWidth(w); setHeight(h) }
                    }}
                  />
                </div>
                <div className="flex gap-2 items-end">
                  <NumField label={t('generate.width')} value={width} onChange={(v) => { setWidth(v); setAspect(aspectFromDimensions(v, height)) }} min={256} max={4096} step={64} />
                  <NumField label={t('generate.height')} value={height} onChange={(v) => { setHeight(v); setAspect(aspectFromDimensions(width, v)) }} min={256} max={4096} step={64} />
                  <button
                    type="button"
                    onClick={() => {
                      const newW = height, newH = width
                      setWidth(newW); setHeight(newH)
                      setAspect(aspectFromDimensions(newW, newH))
                    }}
                    title={t('generate.swapSizeTitle')}
                    className="font-mono inline-flex items-center gap-1.5 shrink-0"
                    style={{
                      border: '1px solid var(--border-subtle)',
                      background: 'var(--bg-sunken)',
                      borderRadius: 'var(--r-md)',
                      padding: '7px 10px',
                      fontSize: 12,
                      color: 'var(--fg-secondary)',
                      cursor: 'pointer',
                      height: 32,
                    }}
                  >
                    <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round">
                      <path d="M16 3l4 4-4 4"/>
                      <path d="M20 7H4"/>
                      <path d="M8 21l-4-4 4-4"/>
                      <path d="M4 17h16"/>
                    </svg>
                    Swap
                  </button>
                </div>
                <div className="flex gap-2">
                  <NumField label={t('generate.steps')} value={steps} onChange={setSteps} min={1} max={150} />
                  <NumField label="CFG" value={cfgScale} onChange={setCfgScale} min={0} max={20} step={0.5} />
                  {mode !== 'xy' && (
                    <NumField label={t('generate.perPrompt')} value={count} onChange={setCount} min={1} max={32} />
                  )}
                </div>
                <NumField
                  label={t('generate.seed')}
                  value={seed}
                  onChange={setSeed}
                  min={0}
                />
                <div className="text-2xs text-fg-tertiary font-mono" style={{ marginTop: -4 }}>
                  {t('generate.seedHint')}
                </div>
              </div>
            </div>

          </div>
            {/* Generate bar：固定 sidebar 底部（在 scroll 区外），橙色大按钮 + 右侧 meta */}
            <div
              className="flex items-center gap-3 shrink-0"
              style={{
                padding: '10px 12px',
                borderRadius: 'var(--r-lg)',
                border: '1px solid var(--border-subtle)',
                background: 'var(--bg-elevated)',
                marginRight: 8, // 跟内层 pr-2 对齐，按钮区不被 scrollbar 占地
              }}
            >
              <button
                className="btn btn-primary flex-1"
                style={{ padding: 12, fontWeight: 600, justifyContent: 'center' }}
                onClick={handleGenerate}
                disabled={busy}
              >
                {generateLabel}
              </button>
              {cancelable && (
                <button className="btn btn-ghost" onClick={handleCancel} title={t('generate.cancelCurrentTitle')}>
                  {t('common.cancel')}
                </button>
              )}
              {!cancelable && (
                <div className="font-mono text-xs text-fg-tertiary text-right" style={{ lineHeight: 1.3 }}>
                  <div>{width}×{height}</div>
                  <div>{busy ? t('generate.generating') : t('generate.sharedGpu')}</div>
                </div>
              )}
            </div>
          </div>

          {/* 中：结果独立 scroll，card flex-1 占满列高 */}
          <div className="flex-1 min-w-0 flex flex-col overflow-y-auto self-stretch">
            <div className="card flex-1 flex flex-col" style={{ padding: 18, minHeight: 0 }}>
              <div className="flex items-center justify-between gap-2 mb-4 flex-wrap">
                <div className="flex items-center gap-2">
                  <span className="text-md font-semibold">{t('generate.results')}</span>
                  {currentTask && (
                    <>
                      <span className="caption">#{currentTask.id}</span>
                      <StatusBadge status={currentTask.status} />
                    </>
                  )}
                  {currentTask?.error_msg && (
                    <span className="text-xs text-err ml-1">{currentTask.error_msg}</span>
                  )}
                </div>
                <ViewModeTabs mode={mode} onModeChange={setMode} />
              </div>

              <GenerateProgressBar busy={busy} progress={progress} />

              {historyOverride ? (
                <div className="flex-1 min-h-0 flex flex-col gap-2">
                  {historyOverride.mode === 'xy' && historyOverride.xy ? (
                    /* xy 历史回看：用 PreviewXYGrid 重建（带轴标签 + 双击全屏） */
                    <PreviewXYGrid
                      samples={historyOverride.xy.samples.map((s) => ({
                        path: s.path,
                        xy: {
                          xi: s.xy.xi, yi: s.xy.yi,
                          xv: s.xy.xv as never, yv: s.xy.yv as never,
                        },
                      }))}
                      taskId={historyOverride.taskId}
                      xDraft={{
                        axis: historyOverride.xy.xAxis as never,
                        raw: historyOverride.xy.xValues.join(', '),
                        loraIndex: null,
                      }}
                      yDraft={historyOverride.xy.yAxis ? {
                        axis: historyOverride.xy.yAxis as never,
                        raw: (historyOverride.xy.yValues as string[]).filter(Boolean).join(', '),
                        loraIndex: null,
                      } : null}
                      onCellClick={undefined /* 历史回看不允许选 cell 进 compare */}
                      selectedIndices={[]}
                    />
                  ) : historyOverride.mode === 'xy' && historyOverride.filenames.length > 1 ? (
                    /* legacy: 旧 entry 没 xy meta，回退到 grid auto-fit 平铺 */
                    <div className="flex-1 min-h-0 overflow-auto">
                      <div
                        style={{
                          display: 'grid',
                          gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
                          gap: 2,
                        }}
                      >
                        {historyOverride.filenames.map((fn) => {
                          const url = api.generateSampleUrl(historyOverride.taskId, fn)
                          return (
                            <a
                              key={fn} href={url} target="_blank" rel="noreferrer"
                              className="block bg-sunken rounded-sm overflow-hidden"
                            >
                              <img
                                src={url}
                                onError={(e) => {
                                  (e.currentTarget as HTMLImageElement).src = historyOverride.thumbnailDataUrl
                                  ;(e.currentTarget as HTMLImageElement).title = t('generate.originalReleasedCoverOnly')
                                }}
                                alt={fn}
                                className="block w-full h-auto"
                                loading="lazy"
                              />
                            </a>
                          )
                        })}
                      </div>
                    </div>
                  ) : (
                    /* 单图回看（single / compare 历史 / 单张 xy） */
                    <a
                      className="flex-1 min-h-0 flex items-center justify-center w-full"
                      href={api.generateSampleUrl(historyOverride.taskId, historyOverride.filenames[0] ?? '')}
                      target="_blank"
                      rel="noreferrer"
                    >
                      <img
                        key={historyOverride.id}
                        src={api.generateSampleUrl(historyOverride.taskId, historyOverride.filenames[0] ?? '')}
                        onError={(e) => {
                          (e.currentTarget as HTMLImageElement).src = historyOverride.thumbnailDataUrl
                          ;(e.currentTarget as HTMLImageElement).title = t('generate.originalReleasedThumbOnly')
                        }}
                        alt={`history #${historyOverride.taskId}`}
                        className="rounded-md object-contain"
                        style={{ maxWidth: '100%', maxHeight: '100%' }}
                      />
                    </a>
                  )}
                  <div className="text-xs text-fg-tertiary shrink-0">
                    {t('generate.historyTask', { id: historyOverride.taskId })}
                    {historyOverride.badge ? ` · ${historyOverride.badge}` : ''}
                    {historyOverride.mode === 'xy' && historyOverride.filenames.length > 1
                      ? ` · ${t('generate.imageCount', { n: historyOverride.filenames.length })}` : ''}
                    <button
                      className="btn btn-ghost text-xs ml-2"
                      style={{ padding: '2px 8px' }}
                      onClick={() => setHistoryOverride(null)}
                    >
                      {t('generate.backToCurrent')}
                    </button>
                  </div>
                </div>
              ) : !currentTask ? (
                <div className="flex-1 grid place-items-center rounded-md border border-subtle bg-sunken text-fg-tertiary text-sm">
                  {t('generate.emptyHint')}
                </div>
              ) : mode === 'xy' && showCompareView ? (
                /* xy 内部 sub-view：选 2 张时切到 compare（不切顶部 mode） */
                <PreviewCompare
                  samples={samples}
                  taskId={currentTask.id}
                  selectedIndices={selectedIndices as [number, number]}
                  xDraft={xDraft}
                  yDraft={yDraft}
                  onBack={() => setSelectedIndices([])}
                />
              ) : mode === 'xy' ? (
                <PreviewXYGrid
                  samples={samples}
                  taskId={currentTask.id}
                  xDraft={xDraft}
                  yDraft={yDraft}
                  onCellClick={handleCellClick}
                  selectedIndices={selectedIndices}
                />
              ) : samples.length === 0 && previewStep ? (
                <div className="flex-1 min-h-0 flex flex-col items-center gap-2">
                  <div className="flex-1 min-h-0 w-full flex items-center justify-center">
                    <img
                      src={previewStep.dataUrl}
                      alt={`step ${previewStep.step}/${previewStep.total}`}
                      className="rounded-md object-contain"
                      style={{ maxWidth: '100%', maxHeight: '100%' }}
                    />
                  </div>
                  <div className="text-xs text-fg-tertiary shrink-0">
                    {t('generate.previewStep', { step: previewStep.step, total: previewStep.total })}
                  </div>
                </div>
              ) : samples.length === 0 ? (
                <div className="flex-1 grid place-items-center rounded-md border border-subtle bg-sunken text-fg-tertiary text-sm">
                  {busy ? t('generate.waitingImages') : t('generate.finishedNoImages')}
                </div>
              ) : (
                <SampleGallery samples={samples} taskId={currentTask.id} />
              )}
            </div>
          </div>

          {/* 右：图片历史栏（commit 16，按当前 mode 分桶） */}
          <PreviewHistoryRail
            entries={history.entries}
            mode={mode}
            onSelect={handleHistorySelect}
            onRemove={(id) => { void history.remove(id) }}
            onClear={() => { void history.clearByMode(mode) }}
            onPruneStale={history.pruneStale}
          />
      </div>

      {/* daemon log 抽屉（fixed 定位 + translateY，隐藏时完全不可见，不占 layout） */}
      <DaemonLogDrawer open={logOpen} onClose={() => setLogOpen(false)} />
    </div>
  )
}
