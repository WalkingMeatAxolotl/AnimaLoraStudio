import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  api,
  TERMINAL_TASK_STATUSES,
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
import PreviewHistoryRail, { type TimelineItem } from './generate/PreviewHistoryRail'
import PromptFromDatasetPicker, { type DatasetPick } from './generate/PromptFromDatasetPicker'
import {
  PARAMS_SNAPSHOT_VERSION, applySnapshot, loraBasename, resolveLoraFromCkpts,
  transformAxisRawForSnapshot,
  type GenerateParamsSnapshot, type SnapshotLora,
} from './generate/paramsSnapshot'
import { saveSingleSamples, saveXYMatrix } from './generate/saveTestImages'
import { useGenerateHistory } from './generate/useGenerateHistory'
import {
  entryImageUrl,
  entryTaskId,
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
import { useLoraCatalog } from './generate/useLoraCatalog'
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
  // 0.17 P-I：currentTask = **显示目标**（daemon 正在跑 / 最近一张），不再是「最后
  // 提交」。提交只入队，显示跟着 running 走（refreshLiveGenerates）。
  const [currentTask, setCurrentTask] = useState<Task | null>(null)
  // 0.17 P-I：本会话提交的 generate 里 running + pending（含自己），驱动「排队中 N 张」
  // 列表 + running 检测。来自 listQueueLive(undefined,'generate')。
  const [liveGenerates, setLiveGenerates] = useState<Task[]>([])
  const prevGenIdsRef = useRef<Set<number>>(new Set())
  // #1：每条 task 的「运行态」定格（XY 轴 + 完整参数快照），dispatch 时存。活动结果
  // 网格 / 双图对比 / 入库读它而非 live prefs，任务开始后改 sidebar 不串改已出结果。
  // 0.17 P-I：单值 → 按 taskId 存 Map，多任务各取各的。
  const runsRef = useRef<Map<number, {
    xDraft: XYAxisDraft
    yDraft: XYAxisDraft | null
    snapshot: GenerateParamsSnapshot
  }>>(new Map())
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
  // 0.17 P-I：useGenerateHistory 每渲染返回新对象（refresh/refreshCache 非 memoized）。
  // 用 ref 取最新，让 ingestGenerateTask/refreshLiveGenerates deps 稳定，避免 mount
  // effect 因它们 identity 每渲染变而无限重跑（fetch 风暴）。
  const historyRef = useRef(history)
  historyRef.current = history
  const [historyOverride, setHistoryOverride] = useState<HistoryEntry | null>(null)
  const taskIdRef = useRef<number | null>(null)
  taskIdRef.current = currentTask?.id ?? null
  const currentTaskRef = useRef<Task | null>(null)
  currentTaskRef.current = currentTask
  // 0.17 P-I：已入库的 taskId（去重，替代旧 lastSnapshotRef）。
  const ingestedRef = useRef<Set<number>>(new Set())

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

  const catalog = useLoraCatalog()
  // 用 useMemo 稳定引用：monitorState 不变时 samples 引用不变，避免下方
  // useEffect 把 samples 当依赖触发不必要的重跑
  const samples = useMemo(() => monitorState?.samples ?? [], [monitorState])
  const samplesRef = useRef(samples)
  samplesRef.current = samples

  // #1：活动结果网格用「dispatch 时定格的轴」而非 live xDraft/yDraft。
  // 显示任务有定格 run（runsRef）时取冻结值（任务开始后改 sidebar 不串改右侧）；否则
  // 回退 live。runsRef 是 ref，但 currentTask 变会 re-render → 这里随之重算，够 reactive。
  const frozenRun = currentTask ? runsRef.current.get(currentTask.id) ?? null : null
  const gridXDraft = frozenRun ? frozenRun.xDraft : xDraft
  const gridYDraft = frozenRun ? frozenRun.yDraft : yDraft

  // 0.17 P-I：统一出图时间线 = live 队列(pending/running) ∪ done 历史(cache/disk 扫盘)，
  // 按 taskId 去重（running→done 过渡窗口）。live 恒在最上（最新提交），done 往下。喂右栏。
  // 未来换后端 D 端点只改这一处派生（前端其余不动）。
  const timelineItems = useMemo<TimelineItem[]>(() => {
    const doneIds = new Set(
      history.entries.map(entryTaskId).filter((x): x is number => x != null),
    )
    const done: TimelineItem[] = [...history.entries]
      .sort((a, b) => b.createdAt - a.createdAt)
      .map((entry) => ({ kind: 'done', entry }))
    const live: TimelineItem[] = [...liveGenerates]
      .filter((task) => !doneIds.has(task.id))
      .sort((a, b) => b.created_at - a.created_at)
      .map((task) => ({
        kind: 'live',
        task,
        mode: runsRef.current.get(task.id)?.snapshot.mode ?? 'single',
      }))
    return [...live, ...done]
  }, [liveGenerates, history.entries])

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

  // 0.17 P-I：入库某条 generate。**每条 done 时各入各的，跟「当前显示哪张」解耦**
  // （多任务下 currentTask 跟着 running 走，不会在每条 done 停留）。
  // temp（默认 save_test_images=off）：server 在 image_done 已把图 + 参数写进加密 cache
  //   → 只 refreshCache 拉新 index。
  // disk（on）：用该 task 的定格 run（runsRef）+ samples 落盘。samplesOverride：显示
  //   任务已有 live samples 时直接传，省一次 getMonitorState。
  const ingestGenerateTask = useCallback(async (taskId: number, samplesOverride?: typeof samples) => {
    if (ingestedRef.current.has(taskId)) return
    const sec = await api.getSecrets().catch(() => null)
    const saveToDisk = !!sec?.generate?.save_test_images
    if (!saveToDisk) {
      ingestedRef.current.add(taskId)
      await historyRef.current.refreshCache()
      return
    }
    const runSnap = runsRef.current.get(taskId)
    const snapMode = runSnap?.snapshot.mode
    if (snapMode !== 'single' && snapMode !== 'xy') return  // compare / 缺 run → 无法重建，不标记（留后重试）
    let s = samplesOverride ?? []
    if (s.length === 0) {
      const st = await api.getMonitorState(taskId).catch(() => null)
      s = (st?.samples as typeof samples | undefined) ?? []
    }
    if (s.length === 0) return
    ingestedRef.current.add(taskId)
    const params = runSnap!.snapshot
    const filenames = s.map((x) => x.path.split(/[\\/]/).pop() ?? '').filter(Boolean)
    if (snapMode === 'single') {
      await saveSingleSamples(taskId, filenames, params)
    } else {
      const xd = runSnap!.xDraft
      const yd = runSnap!.yDraft
      const xValues = xd.raw.split(',').map((v) => v.trim()).filter(Boolean)
      const yValues = yd ? yd.raw.split(',').map((v) => v.trim()).filter(Boolean) : [null as string | null]
      const xySamples = s
        .filter((x): x is typeof x & { xy: NonNullable<typeof x.xy> } => x.xy != null)
        .map((x) => ({ path: x.path, xy: { xi: x.xy.xi, yi: x.xy.yi } }))
      await saveXYMatrix({
        samples: xySamples,
        taskId,
        xAxis: xd.axis as Parameters<typeof saveXYMatrix>[0]['xAxis'],
        yAxis: (yd?.axis ?? null) as Parameters<typeof saveXYMatrix>[0]['yAxis'],
        xValues,
        yValues,
      }, params)
    }
    await historyRef.current.refresh()
  }, [])

  // 0.17 P-I：拉本类型 running+pending generate（listQueueLive 的 type 参数），驱动排队
  // 列表 + 显示跟 running 走 + 对刚离开列表（done/failed/canceled）的每条各自入库。
  const refreshLiveGenerates = useCallback(async () => {
    let items: Task[]
    try { items = await api.listQueueLive(undefined, 'generate') } catch { return }
    setLiveGenerates(items)
    const newIds = new Set(items.map((t) => t.id))
    // finished = 上次在 live、这次不在 = 刚跑完/取消。
    const finished = [...prevGenIdsRef.current].filter((id) => !newIds.has(id))
    prevGenIdsRef.current = newIds
    const cur = currentTaskRef.current
    const running = items.find((t) => t.status === 'running') ?? null
    if (running) {
      // 显示跟着正在跑的那张走
      if (!cur || cur.id !== running.id) setCurrentTask(running)
    } else if (cur && finished.includes(cur.id)) {
      // 无 running 且当前显示那张刚跑完 → 拉终态定格状态徽章（图 samples 已在盘/cache）
      void api.getGenerateTask(cur.id).then(setCurrentTask).catch(() => {})
    }
    // 每条刚完成的各自入库（显示那张用 live samples，省一次 getMonitorState）
    for (const id of finished) {
      void ingestGenerateTask(id, id === cur?.id ? samplesRef.current : undefined)
    }
  }, [ingestGenerateTask])

  useEffect(() => {
    void refreshBlockingTask()
    void refreshLiveGenerates()
  }, [refreshBlockingTask, refreshLiveGenerates])

  // SSE：task_state_changed 触发 task refresh；monitor_state_updated 推 sample 列表。
  useEventStream((evt) => {
    if (evt.type === 'task_state_changed') {
      void refreshBlockingTask()
      // 0.17 P-I：显示态 + 排队列表 + 逐条入库统一由 refreshLiveGenerates 推进。
      void refreshLiveGenerates()
    }
    const tid = taskIdRef.current
    if (tid == null) return
    if (evt.type === 'task_state_changed' && evt.task_id === tid) {
      // currentTask 的推进交给 refreshLiveGenerates；这里只在显示任务终态时清进度。
      if (evt.status === 'done' || evt.status === 'failed' || evt.status === 'canceled') {
        setProgress({ batchIdx: null, batchTotal: null, currentStep: null, totalSteps: null })
      }
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

  // 0.17 P-I：**不再**随 currentTask.id 变自动清 override。多任务下 currentTask 跟着
  // running 自动走，若在此清 override 会把用户正回看的 done 项踢回实时视图。改为只在
  // 用户显式操作时清：点 running 时间线项（rail onSelect）→ 清；或切 mode（下面）→ 清。
  // 切 mode 时只清「属于别的 mode」的 override：手动切 mode 仍清（rail 按 mode 分桶，
  // override.mode 恒等于旧 mode ≠ 新 mode → 清）；但 ?task= 深链到异 mode 的 task 时
  // handleHistorySelect 会把 mode 对齐到 entry.mode，此时 override.mode===新 mode → 保留。
  useEffect(() => {
    setHistoryOverride((cur) => (cur && cur.mode !== mode ? null : cur))
  }, [mode])


  const handleHistorySelect = (entry: HistoryEntry) => {
    setHistoryOverride(entry)  // 先切图（同步），sidebar 回填随 ckpts 解析异步补上
    // applySnapshot 统一所有"应用快照"入口（决策 #8 / Step 3）；现在 async：
    // LoRA 解析按需拉对应版本 ckpts（懒级联），不依赖 mount 全量列表。老 entry
    // 缺 params 会走 catch 兜底（snap.loras 等访问报错 → 不回填，仅切图）。
    void (async () => {
    let applied
    try {
      const projects = await catalog.loadProjects()
      const projIds = new Set(projects.map((p) => p.id))
      applied = await applySnapshot(
        entry.params,
        async (snap) => {
          if (snap.project_id == null || snap.version_id == null) {
            return resolveLoraFromCkpts(snap, [])
          }
          const ckpts = await catalog
            .fetchCkpts(snap.project_id, snap.version_id)
            .catch(() => [])
          return resolveLoraFromCkpts(snap, ckpts)
        },
        (pid) => projIds.has(pid),
      )
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
    })()
  }

  // 0.17 P-H 深链回看：队列详情「查看出图结果」→ /tools/generate?task=<id>。Task 不带
  // mode/params，只有出图历史条目自带 → 等历史加载后按 task_id 命中条目，走现成的
  // historyOverride 回看路径（handleHistorySelect 会对齐 mode + 回填 sidebar）。
  const deepLinkTaskId = useMemo(() => {
    const v = new URLSearchParams(window.location.search).get('task')
    const n = v ? Number(v) : NaN
    return Number.isFinite(n) ? n : null
  }, [])
  const deepLinkConsumedRef = useRef(false)
  useEffect(() => {
    if (deepLinkTaskId == null || deepLinkConsumedRef.current || history.loading) return
    deepLinkConsumedRef.current = true
    // 清 query 避免刷新重触发（同 ?lora= 范式）
    const url = new URL(window.location.href)
    url.searchParams.delete('task')
    window.history.replaceState({}, '', url.toString())
    const entry = history.entries.find((e) => entryTaskId(e) === deepLinkTaskId)
    if (entry) handleHistorySelect(entry)
    // 图源（cache 同 session 未淘汰 / disk save 开着）都没了 = 物理上回看不了，兜底提示。
    else toast(t('generate.taskResultUnavailable', { id: deepLinkTaskId }), 'info')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deepLinkTaskId, history.loading, history.entries])

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

    // 0.17 P-I：提交只入队，**不清空/不劫持显示**——显示跟着正在跑的那张走，新提交的
    // 排到队尾（daemon 逐个跑）。旧的 setCurrentTask(null)/setRun(null)/清 selection/progress
    // 会打断正在出图那张，已移除。
    setSubmitting(true)
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
      const baseSnapshot: GenerateParamsSnapshot = {
        schema_version: PARAMS_SNAPSHOT_VERSION,
        mode,
        prompts,
        negative_prompt: negPrompt,
        width, height, steps,
        cfg_scale: cfgScale,
        sampler_name: samplerName,
        scheduler,
        count: 1,  // 0.17 P-I：每个 task 出 1 张；batch 拆成多 task（下面循环）
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
      // 0.17 P-I：count 现在 = **batch size**（每次入队的 task 数）。single 拆成 batch 个
      // task（各出 1 张、seed 递增区分）→ 在右栏时间线逐个排队；xy 一次一个矩阵（batch 忽略）。
      const batch = mode === 'xy' ? 1 : Math.max(1, count)
      let firstId: number | null = null
      for (let i = 0; i < batch; i++) {
        const taskSeed = seed + i
        const snap: GenerateParamsSnapshot = { ...baseSnapshot, seed: taskSeed }
        const body: GenerateRequest = {
          prompts: mergedPrompts,
          base_model: baseModel ?? undefined,
          negative_prompt: negPrompt,
          width, height, steps,
          count: 1,
          seed: taskSeed,
          cfg_scale: cfgScale,
          sampler_name: samplerName,
          scheduler,
          lora_configs: loraConfigs,
          // attention_backend 不带：server 端套 Comfy-style runtime 并读取 generate backend。
          xy_matrix,
          params_snapshot: snap as unknown as Record<string, unknown>,
        }
        const task = await api.enqueueGenerate(body)
        // #1 + P-I：每 task 的运行态定格存进 Map（xDraft/yDraft 纯原始对象浅拷贝隔离后续
        // 编辑；snapshot 各带自己的 seed）。显示/入库各按 taskId 取。
        runsRef.current.set(task.id, {
          xDraft: { ...xDraft }, yDraft: yDraft ? { ...yDraft } : null, snapshot: snap,
        })
        if (firstId === null) {
          firstId = task.id
          // 首次生成（当前无显示）乐观置为第一个 task，立刻看到「排队/开始」而非空屏。
          if (!currentTaskRef.current || TERMINAL_TASK_STATUSES.includes(currentTaskRef.current.status)) {
            setCurrentTask(task)
          }
        }
      }
      void refreshLiveGenerates()
      toast(
        batch > 1
          ? t('generate.batchEnqueued', { n: batch })
          : t('generate.taskEnqueued', { id: firstId ?? 0 }),
        'success',
      )
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

  // 0.17 P-I：取消某条排队中的 generate（时间线 live 项单条 ✕）。
  const cancelQueued = async (id: number) => {
    try {
      await api.cancelTask(id)
      toast(t('generate.cancelRequested', { id }), 'info')
      void refreshLiveGenerates()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  // 0.17 P-I：清空队列——取消所有等待中（pending）的 generate（不动正在跑的那张）。
  const pendingGenerateIds = useMemo(
    () => liveGenerates.filter((t) => t.status === 'pending').map((t) => t.id),
    [liveGenerates],
  )
  const clearQueue = async () => {
    if (pendingGenerateIds.length === 0) return
    await Promise.allSettled(pendingGenerateIds.map((id) => api.cancelTask(id)))
    toast(t('generate.queueCleared', { n: pendingGenerateIds.length }), 'info')
    void refreshLiveGenerates()
  }

  const cancelable = currentTask
    && (currentTask.status === 'pending' || currentTask.status === 'running')

  // busy 派生：HTTP 入队中 OR 任务还在 pending/running。terminal status
  //（done/failed/canceled）一律 busy=false，让 button 立刻可点重试
  const busy: boolean = submitting || Boolean(cancelable)

  // 0.17 P-I：按钮现在正在出图时也可点（提交新任务入队），所以 label 只在本次入队
  // HTTP 窗口（submitting）显示「生成中」，其余显示动作 label。
  const generateLabel = submitting
    ? t('generate.generating')
    : mode === 'xy' && xyCellCount > 0
      ? t('generate.startGenerateCount', { n: xyCellCount })
      : t('generate.startGenerate')

  return (
    <div className="fade-in flex flex-col" style={{ height: '100%', overflow: 'hidden' }}>
      <PageHeader
        title={t('generate.title')}
        subtitle={t('generate.subtitle')}
        actions={
          <div className="flex items-center gap-2">
            {/* 0.17 P-I：取消（当前显示 task）+ 清空队列（所有 pending）始终在位，不可用时
                disabled，放「清理显存」（DaemonControls）左边。 */}
            <button
              className="btn btn-ghost text-warn border-warn"
              onClick={handleCancel}
              disabled={!cancelable}
              title={t('generate.cancelCurrentTitle')}
            >
              {t('common.cancel')}
            </button>
            <button
              className="btn btn-ghost"
              onClick={() => void clearQueue()}
              disabled={pendingGenerateIds.length === 0}
              title={t('generate.clearQueueTitle')}
              data-testid="generate-clear-queue"
            >
              {pendingGenerateIds.length > 0
                ? t('generate.clearQueue', { n: pendingGenerateIds.length })
                : t('generate.clearQueueEmpty')}
            </button>
            <DaemonControls onToggleLog={() => setLogOpen((v) => !v)} />
          </div>
        }
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
                    catalog={catalog}
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
                  catalog={catalog}
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
                  {/* 0.17 P-I：count 移到「开始生成」旁改为 batch size（每次入队 task 数）。 */}
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
                  disabled={submitting || activeBlockingTask !== null}
                  title={
                    activeBlockingTask
                      ? t('generate.blockedByActiveTask', { id: activeBlockingTask.id })
                      : undefined
                  }
                >
                  {generateLabel}
                </button>
                {/* 0.17 P-I：batch size（每次入队 task 数），固定宽不抖动、无 label，hover
                    显示「批次数量」。取消已移右上。xy 一次一个矩阵、不适用。 */}
                {mode !== 'xy' && (
                  <input
                    type="number"
                    className="input shrink-0"
                    style={{ width: 64, textAlign: 'center' }}
                    min={1} max={32}
                    value={count}
                    onChange={(e) => setCount(Number(e.target.value))}
                    title={t('generate.batchSizeTitle')}
                    aria-label={t('generate.batchSizeTitle')}
                  />
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

              {/* 0.17 P-I：排队中的项已并入右栏时间线（live 占位 + 取消），不再单列。 */}
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
                    {/* 0.17 P-I：删「返回当前」——统一时间线后回到实时点右栏 running 项即可。 */}
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

          {/* 右：出图时间线（live 队列 + done 历史，按当前 mode 分桶） */}
          <PreviewHistoryRail
            items={timelineItems}
            mode={mode}
            onSelect={(it) => {
              if (it.kind === 'done') handleHistorySelect(it.entry)
              // running 项：清 override 回到实时视图（currentTask 已跟着 running 走）。
              else if (it.task.status === 'running') setHistoryOverride(null)
              // pending 项：无内容，不选中（只可取消）。
            }}
            onCancel={cancelQueued}
            onRefresh={history.refresh}
            loading={history.loading}
          />
      </div>

      {/* daemon log 抽屉（fixed 定位 + translateY，隐藏时完全不可见，不占 layout） */}
      <DaemonLogDrawer open={logOpen} onClose={() => setLogOpen(false)} />
    </div>
  )
}
