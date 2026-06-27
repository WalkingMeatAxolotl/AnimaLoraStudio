import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  api,
  type GenerateRequest,
  type LoraEntry,
  type Task,
  type XYMatrixSpec,
} from '../../api/client'
import BaseModelSelect from '../../components/BaseModelSelect'
import PageHeader from '../../components/PageHeader'
import { useToast } from '../../components/Toast'
import { schemaEnumLabel } from '../../lib/schema'
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
import {
  PARAMS_SNAPSHOT_VERSION, applySnapshot, loraBasename,
  transformAxisRawForSnapshot,
  type GenerateParamsSnapshot, type SnapshotLora,
} from './generate/paramsSnapshot'
import { saveSingleSamples, saveXYMatrix } from './generate/saveTestImages'
import { useGenerateHistory } from './generate/useGenerateHistory'
import {
  entryImageUrl,
  type HistoryEntry,
} from './generate/entryAdapter'
import PreviewXYGrid from './generate/PreviewXYGrid'
import PromptList from './generate/PromptList'
import NegPromptInput from './generate/NegPromptInput'
import SampleGallery from './generate/SampleGallery'
import SidebarLoras from './generate/SidebarLoras'
import SidebarSectionTabs, { type SidebarTab } from './generate/SidebarSectionTabs'
import SidebarXYAxes from './generate/SidebarXYAxes'
import StatusBadge from './generate/StatusBadge'
import ViewModeTabs, { type ViewMode } from './generate/ViewModeTabs'
import {
  DEFAULT_NEG, DEFAULT_SAMPLER, DEFAULT_SCHEDULER,
  SAMPLER_OPTIONS, SCHEDULER_OPTIONS,
  type SamplerName, type SchedulerName,
} from './generate/types'
import { useProjectLoras } from './generate/useProjectLoras'
import { buildXYMatrix, cellCount, parseAxisValues, type XYAxisDraft } from './generate/xy'

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
  samplerName: DEFAULT_SAMPLER as SamplerName,
  scheduler: DEFAULT_SCHEDULER as SchedulerName,
  count: 1,
  seed: 0,
  // single / xy 的 LoRA 列表完全独立（用户决策 2026-05-29）：切 mode 互不影响。
  // compare 是 xy 的子视图，跟 xy 共用 xyLoras。
  singleLoras: [] as LoraEntry[],
  xyLoras: [] as LoraEntry[],
  xDraft: { axis: 'steps', raw: '20, 25, 30', loraIndex: null } as XYAxisDraft,
  yDraft: null as XYAxisDraft | null,
  datasetPick: null as DatasetPick | null,
}

type GeneratePrefs = typeof DEFAULT_GENERATE_PREFS

/** 归一化 / 迁移持久化 prefs（readPersisted 不 merge default，必须自己补齐）：
 *  - 老版本只有共享 `loras`（single/xy 共用，正是被修的 bug）→ 拆成
 *    singleLoras/xyLoras 各复制一份，迁移不丢任何已选 LoRA；迁移后两边独立。
 *  - 补齐缺失字段（老 shape / 跨版本新增字段）。
 *  - clamp xDraft/yDraft.loraIndex 到 xyLoras 合法范围（xy 轴 loraIndex 指向
 *    xyLoras；越界会让 submit 抛 axisLoraMissing）。
 */
function normalizePrefs(p: GeneratePrefs): GeneratePrefs {
  const anyP = p as Partial<GeneratePrefs> & { loras?: LoraEntry[] }
  const legacy = Array.isArray(anyP.loras) ? anyP.loras : []
  const singleLoras = Array.isArray(anyP.singleLoras) ? anyP.singleLoras : legacy
  const xyLoras = Array.isArray(anyP.xyLoras) ? anyP.xyLoras : legacy
  const clampIdx = (d: XYAxisDraft | null): XYAxisDraft | null => {
    if (!d || d.loraIndex == null || d.loraIndex < xyLoras.length) return d
    return { ...d, loraIndex: xyLoras.length > 0 ? 0 : null }
  }
  const { loras: _legacy, ...rest } = anyP
  return {
    ...DEFAULT_GENERATE_PREFS,
    ...rest,
    singleLoras,
    xyLoras,
    xDraft: clampIdx(rest.xDraft ?? DEFAULT_GENERATE_PREFS.xDraft) ?? DEFAULT_GENERATE_PREFS.xDraft,
    yDraft: clampIdx(rest.yDraft ?? null),
  }
}

export default function GeneratePage() {
  const { t } = useTranslation()
  const { toast } = useToast()

  const [rawPrefs, setRawPrefs] = useLocalStorageState(GENERATE_PREFS_KEY, DEFAULT_GENERATE_PREFS)
  const prefs = useMemo(() => normalizePrefs(rawPrefs), [rawPrefs])
  // 所有 setPrefs 更新都先把 prev 归一化（迁移老 shape + clamp），保证 updater
  // 收到的永远是新 shape（含 singleLoras/xyLoras，无遗留 loras）。
  const setPrefs = useCallback(
    (next: GeneratePrefs | ((p: GeneratePrefs) => GeneratePrefs)) =>
      setRawPrefs((prev) => {
        const norm = normalizePrefs(prev)
        return typeof next === 'function' ? next(norm) : next
      }),
    [setRawPrefs],
  )
  // 一次性把老 shape（共享 loras）迁移落库，避免 storage 长期残留遗留字段；
  // 之后读到的就是干净的 singleLoras/xyLoras 双桶 shape。
  useEffect(() => {
    const raw = rawPrefs as Partial<GeneratePrefs> & { loras?: unknown }
    if ('loras' in raw || !('singleLoras' in raw) || !('xyLoras' in raw)) {
      setRawPrefs(normalizePrefs(rawPrefs))
    }
    // 仅 mount 跑一次：迁移是幂等的，rawPrefs 后续变化不需要重跑
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const { mode, prompts, negPrompt, aspect, width, height, steps, cfgScale, samplerName, scheduler, count, seed, xDraft, yDraft, datasetPick } = prefs
  // LoRA 列表按 mode 完全独立：single 用 singleLoras，xy（含 compare 子视图）用
  // xyLoras。读写都按当前 mode 路由，切 mode 互不影响。
  const loras = mode === 'single' ? prefs.singleLoras : prefs.xyLoras
  const setLoras = (loras: LoraEntry[]) =>
    setPrefs((p) => (p.mode === 'single' ? { ...p, singleLoras: loras } : { ...p, xyLoras: loras }))
  const setMode = (mode: ViewMode) => setPrefs((p) => ({ ...p, mode }))
  const setPrompts = (prompts: string[]) => setPrefs((p) => ({ ...p, prompts }))
  const setNegPrompt = (negPrompt: string) => setPrefs((p) => ({ ...p, negPrompt }))
  const setAspect = (aspect: AspectName) => setPrefs((p) => ({ ...p, aspect }))
  const setWidth = (width: number) => setPrefs((p) => ({ ...p, width }))
  const setHeight = (height: number) => setPrefs((p) => ({ ...p, height }))
  const setSteps = (steps: number) => setPrefs((p) => ({ ...p, steps }))
  const setCfgScale = (cfgScale: number) => setPrefs((p) => ({ ...p, cfgScale }))
  const setSamplerName = (samplerName: SamplerName) => setPrefs((p) => ({ ...p, samplerName }))
  const setScheduler = (scheduler: SchedulerName) => setPrefs((p) => ({ ...p, scheduler }))
  const setCount = (count: number) => setPrefs((p) => ({ ...p, count }))
  const setSeed = (seed: number) => setPrefs((p) => ({ ...p, seed }))

  // LoRA 预填 via URL query (?lora=<path>&projectId=N&versionId=N)
  // Overview StatusBanner "在测试中加载" CTA 跳进来时，URL 是显式 "测这条 LoRA"
  // 意图 = 测这一条 → 落到 single 模式的列表（replace 成 [urlLora]）并切到 single；
  // xy 列表独立、不受影响（xy 轴 loraIndex 已由 normalizePrefs clamp 到 xyLoras）。
  // 用 history.replaceState 清掉 query 避免刷新时重复触发。
  useEffect(() => {
    const sp = new URLSearchParams(window.location.search)
    const lora = sp.get('lora')
    if (!lora) return
    const projectId = sp.get('projectId')
    const versionId = sp.get('versionId')
    setPrefs((p) => {
      const newLoras: LoraEntry[] = [{
        path: lora,
        scale: 1.0,
        project_id: projectId ? Number(projectId) : null,
        version_id: versionId ? Number(versionId) : null,
      }]
      return { ...p, mode: 'single', singleLoras: newLoras }
    })
    const url = new URL(window.location.href)
    url.searchParams.delete('lora')
    url.searchParams.delete('projectId')
    url.searchParams.delete('versionId')
    window.history.replaceState({}, '', url.toString())
  }, [setPrefs])
  // Test generation omits attention_backend here; the server applies the
  // Comfy-style runtime and reads the configured generate backend there.

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
  // #1：本次生成「运行态」定格。dispatch 成功时把当时的 XY 轴 + 完整参数
  // 快照冻进这里；活动结果网格 / 双图对比 / 完成入库三处都读它，而不是读
  // live xDraft/yDraft/prefs。这样任务开始后改 sidebar（取消某个 XY LoRA、
  // 调参数等）只影响「下次生成」，不会串改已出 / 在出的右侧结果与入库元数据。
  // taskId 不匹配当前任务时回退 live（如刷新后中途回看的边缘场景，无回归）。
  const [run, setRun] = useState<{
    taskId: number
    xDraft: XYAxisDraft
    yDraft: XYAxisDraft | null
    snapshot: GenerateParamsSnapshot
  } | null>(null)
  // 本次出图临时选用的底模（null = 跟随设置页 selected_anima）。不进 prefs
  // 持久化：每次进页面都回到「设置页默认底模」，符合「默认用设置里的」。
  const [baseModel, setBaseModel] = useState<string | null>(null)
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
  // 左侧配置区当前分页（LoRA/XY · 提示词 · 配置）。跨 session 记忆用户停留的页。
  const [sidebarTab, setSidebarTab] = useLocalStorageState<SidebarTab>('studio:generate:sidebarTab', 'lora')
  const [logOpen, setLogOpen] = useState(false)
  // 训练 / reg-ai / 打标等 GPU 任务在跑时，禁用生成防 VRAM 竞争（driver 抢
  // 3D / Copy engine 触发图像渲染卡顿，甚至训练进程 OOM）。listQueue 默认
  // 不含 generate 任务自身，所以自己生成时不会自锁。
  const [activeBlockingTask, setActiveBlockingTask] = useState<Task | null>(null)
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

  // #1：活动结果网格用「dispatch 时定格的轴」(run) 而非 live xDraft/yDraft。
  // run 对应当前任务时取冻结值（任务开始后改 sidebar 不串改右侧）；否则回退 live。
  const frozenRun = run && currentTask && run.taskId === currentTask.id ? run : null
  const gridXDraft = frozenRun ? frozenRun.xDraft : xDraft
  const gridYDraft = frozenRun ? frozenRun.yDraft : yDraft

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

  const refreshBlockingTask = useCallback(async () => {
    try {
      const running = await api.listQueue('running')
      setActiveBlockingTask(running.length > 0 ? running[0] : null)
    } catch {
      // 拉队列失败时不阻塞生成 — bug 修保守，宁愿放过也别误锁。
    }
  }, [])

  useEffect(() => {
    void refreshBlockingTask()
  }, [refreshBlockingTask])

  // SSE：task_state_changed 触发 task refresh；monitor_state_updated 推 sample 列表。
  useEventStream((evt) => {
    if (evt.type === 'task_state_changed') void refreshBlockingTask()
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
    // #1：入库元数据（落盘 / cache 的 xyMeta + 参数快照）用本次运行态定格 run，
    // 而非 live prefs —— 任务开始后改 sidebar 不污染已生成结果的入库轴与参数。
    // run 不匹配当前任务时回退 live（刷新后中途完成的边缘场景）。
    const runFrozen = run && run.taskId === currentTask.id ? run : null
    const metaXDraft = runFrozen ? runFrozen.xDraft : xDraft
    const metaYDraft = runFrozen ? runFrozen.yDraft : yDraft
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
    // badge 字段不再存 entry（adapter.entryBadge 计算）
    const filenames = samples
      .map((s) => s.path.split(/[\\/]/).pop() ?? '')
      .filter(Boolean)
    // commit: xy 历史回看用 PreviewXYGrid 重建网格 → 入库时收集 axis + sample 元数据
    let xyMeta: import('./generate/useGenerateHistory').HistoryXYMeta | undefined
    if (mode === 'xy') {
      const xValues = metaXDraft.raw.split(',').map((s) => s.trim()).filter(Boolean)
      const yValues = metaYDraft
        ? metaYDraft.raw.split(',').map((s) => s.trim()).filter(Boolean)
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
        xAxis: metaXDraft.axis, yAxis: metaYDraft?.axis ?? null,
        xValues, yValues, samples: xySamples,
      }
    }
    // 参数快照（落盘 PNG metadata + cache entry 共用，回填用）。
    // #1：优先用 dispatch 时定格的完整快照（runFrozen.snapshot）——任务开始后
    // 改任意参数都不污染落盘 / cache 元数据。run 缺失时（边缘）回退按 live 现构造，
    // xy_draft 用上面定格的轴 metaXDraft/metaYDraft。
    // LoRA 只存 name + ids（不存 path 避免泄露 / 跨机器死链）；回填时通过
    // projectLoras 用 ids → path resolve。
    const params: GenerateParamsSnapshot = runFrozen?.snapshot ?? {
      schema_version: PARAMS_SNAPSHOT_VERSION,
      mode,
      prompts,
      negative_prompt: negPrompt,
      width, height, steps,
      cfg_scale: cfgScale,
      sampler_name: samplerName,
      scheduler,
      count, seed,
      base_model: baseModel,
      loras: loras.map((l) => ({
        name: loraBasename(l.path),
        scale: l.scale,
        project_id: l.project_id ?? null,
        version_id: l.version_id ?? null,
      })),
      xy_draft: mode === 'xy'
        ? {
            x: transformAxisRawForSnapshot(metaXDraft),
            y: metaYDraft ? transformAxisRawForSnapshot(metaYDraft) : null,
          }
        : null,
      dataset_pick: datasetPick,
    }
    // 决策 #5 二元模式：开关开 = 落盘 + refresh disk-history（DiskEntry 由
    // server 给）；开关关 = server 已自动入加密 cache，前端 refreshCache
    // 拉新 index 即可（不再前端构造 CacheEntry）。compare 不入历史（保留现状）。
    if (mode !== 'single' && mode !== 'xy') return
    void (async () => {
      const sec = await api.getSecrets().catch(() => null)
      const saveToDisk = !!sec?.generate?.save_test_images
      if (saveToDisk) {
        // 持久路径：落盘 + 重拉 disk-history（DiskEntry 由 server 端 disk-history
        // 接口构造，含 sha1 id + thumb url + 已 URL-encoded image url）
        if (mode === 'single') {
          await saveSingleSamples(taskId, filenames, params)
        } else if (xyMeta) {
          await saveXYMatrix({
            samples: xyMeta.samples.map((s) => ({ path: s.path, xy: { xi: s.xy.xi, yi: s.xy.yi } })),
            taskId,
            xAxis: xyMeta.xAxis as Parameters<typeof saveXYMatrix>[0]['xAxis'],
            yAxis: xyMeta.yAxis as Parameters<typeof saveXYMatrix>[0]['yAxis'],
            xValues: xyMeta.xValues,
            yValues: xyMeta.yValues,
          }, params)
        }
        await history.refresh()
      } else {
        // 临时路径：server 端 image_done 已写入加密 disk cache（含 snapshot +
        // xy 元数据），这里只拉新 index
        await history.refreshCache()
      }
    })()
  }, [currentTask, samples, mode, selectedIndices, history, run, xDraft, yDraft,
      prompts, negPrompt, width, height, steps, cfgScale, samplerName, scheduler, count, seed, baseModel, loras, datasetPick])

  const handleHistorySelect = (entry: HistoryEntry) => {
    setHistoryOverride(entry)
    // applySnapshot 统一所有"应用快照"入口（决策 #8 / Step 3）；老 entry 缺
    // params 会走 catch 兜底（snap.loras 等访问报错 → 不回填，仅切图）
    let applied
    try {
      applied = applySnapshot(entry.params, projectLoras)
    } catch {
      return
    }
    if (applied.unresolvedLoraCount > 0) {
      toast(t('generate.historyLorasMissing', { n: applied.unresolvedLoraCount }), 'info')
    }
    // datasetPick 非空 → 自动展开 picker 让用户看到选中行 + tags 文本（picker
    // 是 closed by default，不展开的话 prompts[0] 经常是 ""（用户全靠 dataset
    // tags 当 prompt 的常见场景），UI 表面看就像"啥都没回填"）。fallback 路径
    // 已经把 tags 灌到 prompts[0] + datasetPick=null，所以这里只看 applied 即可。
    if (applied.datasetPick) {
      setDatasetPickerOpen(true)
    }
    // 底模不在 prefs 里（独立 ephemeral state）→ 单独回填。
    setBaseModel(applied.baseModel)
    setPrefs((prev) => {
      const base: GeneratePrefs = {
        ...prev,
        mode: applied.mode,
        prompts: applied.prompts.length > 0 ? applied.prompts : prev.prompts,
        negPrompt: applied.negPrompt,
        width: applied.width,
        height: applied.height,
        aspect: aspectFromDimensions(applied.width, applied.height),
        steps: applied.steps,
        cfgScale: applied.cfgScale,
        samplerName: applied.samplerName,
        scheduler: applied.scheduler,
        count: applied.count,
        seed: applied.seed,
        datasetPick: applied.datasetPick,
      }
      if (applied.mode === 'single') {
        return { ...base, singleLoras: applied.loras }
      }
      return {
        ...base,
        xyLoras: applied.loras,
        xDraft: applied.xDraft ?? prev.xDraft,
        yDraft: applied.yDraft ?? null,
      }
    })
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
    // single：base LoRA = singleLoras 全发。xy：只发被轴引用的 anchor（见
    // buildXYMatrix —— xyLoras 会沉积 picker 切项目/版本/删轴遗留的孤儿 anchor，
    // 整桶发出去会让孤儿叠到每个 cell，正是反复出现的「混进没选过的 LoRA」根因）。
    let loraConfigs: LoraEntry[] = loras.filter((l) => l.path.trim())
    if (mode === 'xy') {
      // schema 强制 prompts 单条 + count=1
      if (prompts.filter((p) => p.trim()).length > 1) {
        toast(t('generate.xySinglePromptOnly'), 'error')
        return
      }
      try {
        const built = buildXYMatrix(xDraft, yDraft, loras)
        xy_matrix = built.xy_matrix
        loraConfigs = built.loraConfigs
      } catch (e) {
        toast(typeof e === 'string' ? e : String(e), 'error')
        return
      }
    }

    setSubmitting(true)
    setCurrentTask(null)
    setRun(null)  // #1：清掉上一次的运行态定格，本次 enqueue 成功后重新冻结
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
      // 跟 dispatch 一起送 snapshot 给 server：image_done 时塞进加密 cache
      // payload header（save=false）+ list_index 时返还回填用。落盘 save=true
      // 分支仍用各自 saveSingleSamples/saveXYMatrix 自己构造；两边字段对齐。
      const snapshotLoras: SnapshotLora[] = loras.map((l) => ({
        name: loraBasename(l.path),
        scale: l.scale,
        project_id: l.project_id ?? null,
        version_id: l.version_id ?? null,
      }))
      const dispatchSnapshot: GenerateParamsSnapshot = {
        schema_version: PARAMS_SNAPSHOT_VERSION,
        mode,
        prompts,
        negative_prompt: negPrompt,
        width, height, steps,
        cfg_scale: cfgScale,
        sampler_name: samplerName,
        scheduler,
        count: mode === 'xy' ? 1 : count,
        seed,
        base_model: baseModel,
        loras: snapshotLoras,
        xy_draft: mode === 'xy'
          ? {
              x: transformAxisRawForSnapshot(xDraft),
              y: yDraft ? transformAxisRawForSnapshot(yDraft) : null,
            }
          : null,
        dataset_pick: datasetPick,
      }
      const body: GenerateRequest = {
        prompts: mergedPrompts,
        base_model: baseModel ?? undefined,
        negative_prompt: negPrompt,
        width, height, steps,
        count: mode === 'xy' ? 1 : count,
        seed,
        cfg_scale: cfgScale,
        sampler_name: samplerName,
        scheduler,
        lora_configs: loraConfigs,
        // attention_backend 不带：server 端套 Comfy-style runtime 并读取 generate backend。
        xy_matrix,
        params_snapshot: dispatchSnapshot as unknown as Record<string, unknown>,
      }
      const task = await api.enqueueGenerate(body)
      // 立即同步 ref，避免 supervisor 在 enqueue 返回 → setCurrentTask 渲染
      // 之间已经处理完任务并发了 task_state_changed 事件（config 缺失这种
      // 早期失败会马上发 SSE，handler 拿 taskIdRef 还是 null → 漏事件）
      taskIdRef.current = task.id
      setCurrentTask(task)
      // #1：定格本次运行态。xDraft/yDraft 是 {axis,raw,loraIndex} 纯原始对象，
      // 浅拷贝即可隔离后续 sidebar 编辑；snapshot 复用上面已构造的 dispatchSnapshot。
      setRun({
        taskId: task.id,
        xDraft: { ...xDraft },
        yDraft: yDraft ? { ...yDraft } : null,
        snapshot: dispatchSnapshot,
      })
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

          {/* 左：sidebar — 单卡片包裹；内容区独立 scroll，底部 footer 固定 tab + 生成按钮 */}
          <div className="card flex flex-col w-full xl:w-[420px] shrink-0 self-stretch min-h-0 overflow-hidden">
            {/* 内容区：三个 section 都常驻 DOM、用 display 切换（不卸载）—— 切 tab 不重渲不闪烁。
                scrollbar-gutter: stable both-edges —— 两侧都常驻预留滚动条槽（槽在 padding 外侧、
                靠 border），所以左右 18px 内边距恒对称，且滚动条出现时只占右槽、不挤压/不位移内容。 */}
            <div
              className="flex flex-col flex-1 min-h-0 overflow-y-auto"
              style={{ padding: 18, scrollbarGutter: 'stable both-edges' }}
            >

            {/* tab=lora：mode=single → LoRA 选择；mode=xy → XY 轴（顶部合并 LoRA 选择） */}
            <div style={{ display: sidebarTab === 'lora' ? undefined : 'none' }}>
              {mode === 'single' ? (
                <>
                  <div className="flex items-baseline justify-between mb-3">
                    <h3 className="m-0 text-md font-semibold">LoRA</h3>
                    <span className="text-xs text-fg-tertiary">{t('generate.loraHint')}</span>
                  </div>
                  <SidebarLoras
                    loras={loras}
                    onChange={setLoras}
                    projectLoras={projectLoras}
                  />
                </>
              ) : (
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
            </div>

            {/* tab=prompts */}
            <div style={{ display: sidebarTab === 'prompts' ? undefined : 'none' }}>
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
              <NegPromptInput value={negPrompt} onChange={setNegPrompt} />
            </div>

            {/* tab=config */}
            <div style={{ display: sidebarTab === 'config' ? undefined : 'none' }}>
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
                <div className="flex gap-2">
                  <div className="flex-1 min-w-0">
                    <label className="caption block mb-1">{t('generate.sampler')}</label>
                    <select
                      className="input text-xs w-full"
                      value={samplerName}
                      onChange={(e) => setSamplerName(e.target.value as SamplerName)}
                      aria-label={t('generate.sampler')}
                    >
                      {/* 文案与训练配置页共用 schema.enums.* 映射，两边保持一致 */}
                      {SAMPLER_OPTIONS.map((s) => (
                        <option key={s} value={s}>{schemaEnumLabel('sample_sampler_name', s, t)}</option>
                      ))}
                    </select>
                  </div>
                  <div className="flex-1 min-w-0">
                    <label className="caption block mb-1">{t('generate.scheduler')}</label>
                    <select
                      className="input text-xs w-full"
                      value={scheduler}
                      onChange={(e) => setScheduler(e.target.value as SchedulerName)}
                      aria-label={t('generate.scheduler')}
                    >
                      {SCHEDULER_OPTIONS.map((s) => (
                        <option key={s} value={s}>{schemaEnumLabel('sample_scheduler', s, t)}</option>
                      ))}
                    </select>
                  </div>
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
                <div>
                  <label className="caption block mb-1">{t('generate.baseModel')}</label>
                  <BaseModelSelect
                    value={baseModel}
                    onChange={setBaseModel}
                    className="input text-xs w-full"
                    ariaLabel={t('generate.baseModel')}
                  />
                  <div className="text-2xs text-fg-tertiary font-mono mt-1">
                    {t('generate.baseModelHint')}
                  </div>
                </div>
              </div>
            </div>

            </div>

            {/* footer：分页 tab（segmented）+「开始生成」同处一个 footer、跟内容区共卡片，
                border-top 分隔。tab 选中态用 sunken 轨道而非橙色，跟下方生成按钮区分开。 */}
            <div
              className="shrink-0 flex flex-col gap-2.5"
              style={{ borderTop: '1px solid var(--border-subtle)', padding: 12 }}
            >
              <SidebarSectionTabs tab={sidebarTab} onTabChange={setSidebarTab} mode={mode} />
              <div className="flex items-center gap-3">
                <button
                  className="btn btn-primary flex-1"
                  style={{ padding: 12, fontWeight: 600, justifyContent: 'center' }}
                  onClick={handleGenerate}
                  disabled={busy || activeBlockingTask !== null}
                  title={
                    activeBlockingTask
                      ? t('generate.blockedByActiveTask', { id: activeBlockingTask.id })
                      : undefined
                  }
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
                    <div>
                      {busy
                        ? t('generate.generating')
                        : activeBlockingTask
                          ? t('generate.blockedByActiveTaskHint', { id: activeBlockingTask.id })
                          : t('generate.sharedGpu')}
                    </div>
                  </div>
                )}
              </div>
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
                  {historyOverride.mode === 'xy' && historyOverride.xyMeta ? (
                    /* XY 回看 (cache / disk 共用)：per-cell 信息齐 → PreviewXYGrid
                       cache 时 taskId 是真 task id（GridCell fallback 走 cache URL）；
                       disk 时 server 已给 imageUrl，taskId 走 -1 sentinel（不会被用到）。
                       disk 时多传 compositeUrl → 导出 PNG 走文件下载，不再 re-compose */
                    <PreviewXYGrid
                      samples={historyOverride.xyMeta.samples.map((s) => ({
                        path: s.path,
                        xy: {
                          xi: s.xy.xi, yi: s.xy.yi,
                          xv: s.xy.xv as never, yv: s.xy.yv as never,
                        },
                        imageUrl: s.imageUrl,
                      }))}
                      taskId={historyOverride.source === 'cache' ? historyOverride.taskId : -1}
                      xDraft={{
                        axis: historyOverride.xyMeta.xAxis as never,
                        raw: historyOverride.xyMeta.xValues.join(', '),
                        loraIndex: null,
                      }}
                      yDraft={historyOverride.xyMeta.yAxis ? {
                        axis: historyOverride.xyMeta.yAxis as never,
                        raw: (historyOverride.xyMeta.yValues as string[]).filter(Boolean).join(', '),
                        loraIndex: null,
                      } : null}
                      onCellClick={undefined /* 历史回看不允许选 cell 进 compare */}
                      selectedIndices={[]}
                      compositeUrl={historyOverride.source === 'disk' ? historyOverride.imageUrl : undefined}
                    />
                  ) : (
                    /* DiskEntry single / legacy XY（无 xyMeta） / CacheEntry single → 单图视图 */
                    <a
                      className="flex-1 min-h-0 flex items-center justify-center w-full"
                      href={entryImageUrl(historyOverride, 0)}
                      target="_blank"
                      rel="noreferrer"
                    >
                      <img
                        key={historyOverride.id}
                        src={entryImageUrl(historyOverride, 0)}
                        onError={(e) => {
                          (e.currentTarget as HTMLImageElement).title = t('generate.originalReleasedThumbOnly')
                        }}
                        alt=""
                        className="rounded-md object-contain"
                        style={{ maxWidth: '100%', maxHeight: '100%' }}
                      />
                    </a>
                  )}
                  <div className="text-xs text-fg-tertiary shrink-0">
                    {historyOverride.source === 'disk'
                      ? (historyOverride.folder ?? (historyOverride.filename ?? '').replace(/\.png$/i, ''))
                      : t('generate.historyTask', { id: historyOverride.taskId })}
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
                  xDraft={gridXDraft}
                  yDraft={gridYDraft}
                  onBack={() => setSelectedIndices([])}
                />
              ) : mode === 'xy' ? (
                <PreviewXYGrid
                  samples={samples}
                  taskId={currentTask.id}
                  xDraft={gridXDraft}
                  yDraft={gridYDraft}
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

          {/* 右：图片历史栏（按当前 mode 分桶） */}
          <PreviewHistoryRail
            entries={history.entries}
            mode={mode}
            onSelect={handleHistorySelect}
            onRefresh={history.refresh}
            loading={history.loading}
          />
      </div>

      {/* daemon log 抽屉（fixed 定位 + translateY，隐藏时完全不可见，不占 layout） */}
      <DaemonLogDrawer open={logOpen} onClose={() => setLogOpen(false)} />
    </div>
  )
}
