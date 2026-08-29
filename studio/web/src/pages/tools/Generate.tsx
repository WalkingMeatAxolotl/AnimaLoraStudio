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
import BaseModelSelect, { useBaseModelOptions, useKrea2TeOptions } from '../../components/BaseModelSelect'
import PageHeader from '../../components/PageHeader'
import { useToast } from '../../components/Toast'
import { schemaEnumLabel } from '../../lib/schema'
import { useEventStream } from '../../lib/useEventStream'
import { useMonitorProgress } from '../../lib/useMonitorProgress'
import { useLocalStorageState } from '../../lib/useLocalStorageState'
import AspectChips, { aspectFromDimensions, type AspectName } from './generate/AspectChips'
import DaemonControls from './generate/DaemonControls'
import DaemonLogDrawer from './generate/DaemonLogDrawer'
import GenerateAttachedDrawer from './generate/GenerateAttachedDrawer'
import GenerateProgressBar, { type GenerateProgress, type GeneratePhase } from './generate/GenerateProgress'
import NumField from './generate/NumField'
import PreviewCompare from './generate/PreviewCompare'
import ZoomableImage from '../../components/ZoomableImage'
import PreviewHistoryRail, { type TimelineItem } from './generate/PreviewHistoryRail'
import PromptFromDatasetPicker, { type DatasetPick } from './generate/PromptFromDatasetPicker'
import {
  PARAMS_SNAPSHOT_VERSION, applySnapshot, loraBasename, resolveLoraFromCkpts,
  transformAxisRawForSnapshot,
  type GenerateParamsSnapshot, type SnapshotLora,
} from './generate/paramsSnapshot'
import { composeXYMatrix } from './generate/exportXY'
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
import LoraCatalogDrawer from './generate/LoraCatalogDrawer'
import {
  createLoraUiState,
  enabledLoras,
  normalizeLoraUi,
  type LoraUiState,
} from './generate/loraSelection'
import SidebarSectionTabs, { type SidebarTab } from './generate/SidebarSectionTabs'
import { SidebarToolIcon, ToolbarAction } from './generate/SidebarToolbar'
import SidebarXYAxes, { XYAxisToolbar } from './generate/SidebarXYAxes'
import XYAxisEditorDrawer from './generate/XYAxisEditorDrawer'
import StatusBadge from './generate/StatusBadge'
import ViewModeTabs, { type ViewMode } from './generate/ViewModeTabs'
import {
  DEFAULT_NEG, DEFAULT_SAMPLER, DEFAULT_SCHEDULER,
  DISTILLED_GENERATE_DEFAULTS, FAMILY_GENERATE_DEFAULTS,
  SAMPLER_OPTIONS_BY_FAMILY, SCHEDULER_OPTIONS_BY_FAMILY,
  type GenerateFamily, type SamplerName, type SchedulerName,
} from './generate/types'
import { useLoraCatalog } from './generate/useLoraCatalog'
import {
  axisText, axisView, buildXYMatrix, cellCount,
  type XYAxisDraft,
} from './generate/xy'

const GENERATE_PREFS_KEY = 'studio:generate:params:v1'

const DEFAULT_GENERATE_PREFS = {
  mode: 'single' as ViewMode,
  modelFamily: 'anima' as GenerateFamily,
  prompts: ['newest, safe, 1girl, masterpiece, best quality'],
  negPrompt: DEFAULT_NEG,
  aspect: '1:1' as AspectName,
  width: 1024,
  height: 1024,
  steps: 25,
  cfgScale: 4.0,
  samplerName: DEFAULT_SAMPLER as SamplerName,
  scheduler: DEFAULT_SCHEDULER as SchedulerName,
  seed: 0,
  // single / xy 的 LoRA 列表完全独立（用户决策 2026-05-29）：切 mode 互不影响。
  // compare 是 xy 的子视图，跟 xy 共用固定 LoRA 与轴配置。
  singleLoras: [] as LoraEntry[],
  singleLoraUi: [] as LoraUiState[],
  xyLoras: [] as LoraEntry[], // legacy bucket; migrated to xyFixedLoras
  xyFixedLoras: [] as LoraEntry[],
  xyFixedLoraUi: [] as LoraUiState[],
  xDraft: { axis: 'steps', raw: '20, 25, 30', loraIndex: null } as XYAxisDraft,
  yDraft: null as XYAxisDraft | null,
  datasetPick: null as DatasetPick | null,
  // caption 来源身份与实际生成文本分离：前者用于高亮/图片定位，后者可手工编辑。
  datasetPrompt: '',
  // 底模 / TE 的显式覆盖也持久化（用户反馈：切页面被重置回全局默认太烦）。
  // null = 跟随设置页 selected / selected_te（仍是默认行为）。
  baseModel: null as string | null,
  textEncoder: null as 'bf16' | 'fp8' | null,
}

type GeneratePrefs = typeof DEFAULT_GENERATE_PREFS

/** 识别官方 variant key 与常见 custom 文件名中的 FP8 标记。
 * 这是性能提示，不参与后端执行判定；daemon 仍以实际模型层类型为准。 */
function isFp8BaseModel(value: string | null): boolean {
  if (!value) return false
  const name = value.split(/[\\/]/).pop() ?? value
  return /(?:^|[-_.])fp8(?:[-_.]|$)/i.test(name)
}

/** 归一化 / 迁移持久化 prefs（readPersisted 不 merge default，必须自己补齐）：
 *  - 老版本只有共享 `loras`（single/xy 共用，正是被修的 bug）→ 拆成
 *    singleLoras/xyLoras 各复制一份，迁移不丢任何已选 LoRA；迁移后两边独立。
 *  - 补齐缺失字段（老 shape / 跨版本新增字段）。
 *  - 迁移旧 checkpoint 轴的 loraIndex 为 checkpointAnchor；非 checkpoint 轴清除陈旧索引。
 */
function normalizePrefs(p: GeneratePrefs): GeneratePrefs {
  const anyP = p as Partial<GeneratePrefs> & { loras?: LoraEntry[]; count?: number }
  const legacy = Array.isArray(anyP.loras) ? anyP.loras : []
  const singleLoras = Array.isArray(anyP.singleLoras) ? anyP.singleLoras : legacy
  const singleLoraUi = normalizeLoraUi(singleLoras, anyP.singleLoraUi)
  const xyLoras = Array.isArray(anyP.xyLoras) ? anyP.xyLoras : legacy
  // `xyLoras` is the legacy axis-anchor bucket, not the new fixed-LoRA tab.  Do
  // not promote it to fixed LoRAs: old picker leftovers must remain invisible
  // unless an axis still references them.  Only the explicit new bucket is a
  // fixed selection.
  const xyFixedLoras = Array.isArray(anyP.xyFixedLoras) ? anyP.xyFixedLoras : []
  const xyFixedLoraUi = normalizeLoraUi(xyFixedLoras, anyP.xyFixedLoraUi)
  const migrateDraft = (draft: XYAxisDraft | null): XYAxisDraft | null => {
    if (!draft) return null
    if (draft.axis !== 'lora_ckpt') {
      return { ...draft, loraIndex: null, checkpointAnchor: null }
    }
    if (draft.checkpointAnchor?.path.trim()) {
      return { ...draft, loraIndex: null }
    }
    if (draft.loraIndex == null || !Number.isInteger(draft.loraIndex) || xyLoras.length === 0) {
      return { ...draft, loraIndex: null, checkpointAnchor: null }
    }
    const index = Math.max(0, Math.min(draft.loraIndex, xyLoras.length - 1))
    return {
      ...draft,
      loraIndex: index,
      checkpointAnchor: xyLoras[index] ?? null,
    }
  }
  const { loras: _legacy, count: _count, ...rest } = anyP  // count 已改瞬态，丢弃老持久值
  const datasetPrompt = typeof anyP.datasetPrompt === 'string'
    ? anyP.datasetPrompt
    : (anyP.datasetPick?.tags ?? []).join(', ')
  const merged = {
    ...DEFAULT_GENERATE_PREFS,
    ...rest,
    datasetPrompt,
    singleLoras,
    singleLoraUi,
    xyLoras,
    xyFixedLoras,
    xyFixedLoraUi,
    xDraft: migrateDraft(rest.xDraft ?? DEFAULT_GENERATE_PREFS.xDraft) ?? DEFAULT_GENERATE_PREFS.xDraft,
    yDraft: migrateDraft(rest.yDraft ?? null),
  }
  // 族与 sampler 一致性（多模型 P4-4）：老 prefs 无 modelFamily / 持久化的
  // sampler 与当前族白名单不符时（越族值后端 422），落回族默认（首项）。
  const family: GenerateFamily =
    merged.modelFamily === 'krea2' ? 'krea2' : 'anima'
  const samplers = SAMPLER_OPTIONS_BY_FAMILY[family] as readonly string[]
  const schedulers = SCHEDULER_OPTIONS_BY_FAMILY[family] as readonly string[]
  return {
    ...merged,
    modelFamily: family,
    samplerName: (samplers.includes(merged.samplerName)
      ? merged.samplerName : samplers[0]) as SamplerName,
    scheduler: (schedulers.includes(merged.scheduler)
      ? merged.scheduler : schedulers[0]) as SchedulerName,
    textEncoder: (merged.textEncoder === 'bf16' || merged.textEncoder === 'fp8')
      ? merged.textEncoder : null,
  }
}

function virtualYDraftFor(xDraft: XYAxisDraft, steps: number): XYAxisDraft {
  return xDraft.axis === 'lora_scale'
    ? { axis: 'steps', raw: String(steps), loraIndex: null, checkpointAnchor: null }
    : { axis: 'lora_scale', raw: '1.0', loraIndex: null, checkpointAnchor: null }
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
    if ('loras' in raw || !('singleLoras' in raw) || !('singleLoraUi' in raw) || !('xyLoras' in raw) || !('xyFixedLoras' in raw) || !('xyFixedLoraUi' in raw) || typeof raw.datasetPrompt !== 'string') {
      setRawPrefs(normalizePrefs(rawPrefs))
    }
    // 仅 mount 跑一次：迁移是幂等的，rawPrefs 后续变化不需要重跑
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const { mode, modelFamily, prompts, negPrompt, aspect, width, height, steps, cfgScale, samplerName, scheduler, seed, xDraft, yDraft, datasetPick, datasetPrompt } = prefs
  // single 与 XY 的固定 LoRA 完全隔离；旧 xyLoras 仅在 normalizePrefs 中迁移，新的编辑器不再使用它。
  const loras = mode === 'single' ? prefs.singleLoras : prefs.xyFixedLoras
  const loraUi = mode === 'single' ? prefs.singleLoraUi : prefs.xyFixedLoraUi
  const setSelection = (nextLoras: LoraEntry[], nextUi: LoraUiState[]) =>
    setPrefs((p) => (p.mode === 'single'
      ? { ...p, singleLoras: nextLoras, singleLoraUi: nextUi }
      : { ...p, xyFixedLoras: nextLoras, xyFixedLoraUi: nextUi }))
  const setMode = (mode: ViewMode) => {
    setPrefs((p) => ({ ...p, mode }))
    setSidebarTab(mode === 'xy' ? 'xy' : 'lora')
  }
  const setPrompts = (prompts: string[]) => setPrefs((p) => ({ ...p, prompts }))
  const setNegPrompt = (negPrompt: string) => setPrefs((p) => ({ ...p, negPrompt }))
  const setAspect = (aspect: AspectName) => setPrefs((p) => ({ ...p, aspect }))
  const setWidth = (width: number) => setPrefs((p) => ({ ...p, width }))
  const setHeight = (height: number) => setPrefs((p) => ({ ...p, height }))
  const setSteps = (steps: number) => setPrefs((p) => ({ ...p, steps }))
  const setCfgScale = (cfgScale: number) => setPrefs((p) => ({ ...p, cfgScale }))
  const setSamplerName = (samplerName: SamplerName) => setPrefs((p) => ({ ...p, samplerName }))
  const setScheduler = (scheduler: SchedulerName) => setPrefs((p) => ({ ...p, scheduler }))
  const setSeed = (seed: number) => setPrefs((p) => ({ ...p, seed }))
  /** 切模型族：sampler/scheduler/steps/cfg 落回目标族默认（越族值后端 422），
   *  底模临时覆盖清空（variant key 是族内值）。 */
  const setModelFamily = (family: GenerateFamily) => {
    setBaseModel(null)
    setTextEncoder(null)
    setPrefs((p) => ({
      ...p,
      modelFamily: family,
      samplerName: SAMPLER_OPTIONS_BY_FAMILY[family][0] as SamplerName,
      scheduler: SCHEDULER_OPTIONS_BY_FAMILY[family][0] as SchedulerName,
      steps: FAMILY_GENERATE_DEFAULTS[family].steps,
      cfgScale: FAMILY_GENERATE_DEFAULTS[family].cfgScale,
    }))
  }
  // 0.17 P-I：batch size（每次入队 task 数）是**瞬态** UI 值——不进 prefs、不持久化、
  // 不随点历史图回填（用户用 2 就一直 2）；刷新页面重置回 1。
  const [batchSize, setBatchSize] = useState(1)

  // LoRA 预填 via URL query (?lora=<path>&projectId=N&versionId=N)
  // Overview StatusBanner "在测试中加载" CTA 跳进来时，URL 是显式 "测这条 LoRA"
  // 意图 = 测这一条 → 落到 single 模式的列表（replace 成 [urlLora]）并切到 single；
  // xy 列表独立、不受影响（旧 checkpoint 轴索引由 normalizePrefs 迁移为 anchor）。
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
      return {
        ...p,
        mode: 'single',
        singleLoras: newLoras,
        singleLoraUi: [createLoraUiState(true)],
      }
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
  const virtualYDraft = useMemo(
    () => virtualYDraftFor(xDraft, steps),
    [steps, xDraft],
  )
  const visibleYDraft = yDraft ?? virtualYDraft
  const setDatasetPrompt = (datasetPrompt: string) => setPrefs((p) => ({ ...p, datasetPrompt }))
  const setDatasetPick = (datasetPick: DatasetPick | null) => setPrefs((p) => ({
    ...p,
    datasetPick,
    datasetPrompt: datasetPick ? datasetPick.tags.join(', ') : '',
  }))

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
  // 本次出图选用的底模 / TE（null = 跟随设置页 selected / selected_te）。
  // 显式覆盖持久化在 prefs（用户反馈：瞬态设计切页面即被重置太烦）。
  const baseModel = prefs.baseModel
  const setBaseModel = (v: string | null) => setPrefs((p) => ({ ...p, baseModel: v }))
  const textEncoder = prefs.textEncoder
  const setTextEncoder = (v: 'bf16' | 'fp8' | null) =>
    setPrefs((p) => ({ ...p, textEncoder: v }))
  const teOptions = useKrea2TeOptions()
  const effectiveTe = textEncoder ?? teOptions.selected
  // 当前族的底模选项（含 purpose 元数据）——选中蒸馏推理 variant（Krea2
  // Turbo）时应用 8 步 / 无 CFG 的默认参数（可再改，A1 不加限制）
  const { options: baseModelOptions, defaultValue: defaultBaseModel } = useBaseModelOptions(modelFamily)
  const fp8BaseModel = modelFamily === 'krea2' && isFp8BaseModel(baseModel ?? defaultBaseModel)
  const onBaseModelChange = (v: string) => {
    setBaseModel(v)
    const picked = baseModelOptions.find((o) => o.value === v)
    if (picked?.purpose === 'inference') {
      setPrefs((p) => ({
        ...p,
        steps: DISTILLED_GENERATE_DEFAULTS.steps,
        cfgScale: DISTILLED_GENERATE_DEFAULTS.cfgScale,
      }))
    }
  }
  // monitor 走 useMonitorProgress hook (PR #37 增量协议)：currentTask 变 →
  // hook 自动重拉快照 + 订阅 SSE delta 合并；本组件只用 samples 字段，其余
  // 字段在这页生成场景下不需要。
  const { state: monitorState } = useMonitorProgress(currentTask?.id ?? null)
  // commit 14：中间步预览（仅 single 模式有意义；XY/对比 cell 多预览意义小）
  const [previewStep, setPreviewStep] = useState<{ step: number; total: number; dataUrl: string } | null>(null)
  // 生成进度（image_started + preview_step 聚合）
  const [progress, setProgress] = useState<GenerateProgress>({
    phase: null, batchIdx: null, batchTotal: null, currentStep: null, totalSteps: null,
  })
  const [datasetPickerOpen, setDatasetPickerOpen] = useState(false)
  // 左侧配置区当前分页（LoRA/XY · 提示词 · 配置）。跨 session 记忆用户停留的页。
  const [sidebarTab, setSidebarTab] = useLocalStorageState<SidebarTab>(
    'studio:generate:sidebarTab:v2',
    mode === 'xy' ? 'xy' : 'lora',
  )
  const [catalogDrawerOpen, setCatalogDrawerOpen] = useState(false)
  const [activeAxis, setActiveAxis] = useState<'X' | 'Y'>('X')
  const [axisDrawerOpen, setAxisDrawerOpen] = useState(false)
  const [axisOrderRevision, setAxisOrderRevision] = useState({ X: 0, Y: 0 })
  const prevModeRef = useRef(mode)
  useEffect(() => {
    const prevMode = prevModeRef.current
    prevModeRef.current = mode
    if (mode === 'xy' && prevMode !== 'xy') {
      setSidebarTab('xy')
      return
    }
    if (mode !== 'xy' && sidebarTab === 'xy') {
      setSidebarTab('lora')
      return
    }
    if (sidebarTab !== 'lora') setCatalogDrawerOpen(false)
    if (sidebarTab !== 'prompts') setDatasetPickerOpen(false)
    if (mode !== 'xy' || sidebarTab !== 'xy') setAxisDrawerOpen(false)
  }, [mode, sidebarTab, setSidebarTab])
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

  const refreshBlockingTask = useCallback(async () => {
    try {
      const running = await api.listQueue('running')
      setActiveBlockingTask(running.length > 0 ? running[0] : null)
    } catch {
      // 拉队列失败时不阻塞生成 — bug 修保守，宁愿放过也别误锁。
    }
  }, [])

  // task done 后收尾。写路径已全部在 server 端闭环（daemon image_done →
  // generate_storage 落盘/记账），前端只剩两件事：
  //   1. XY + save 开 → 用定格 run 现拼 composite POST 补传（决策 1：盘上仍要
  //      有大图给外站上传；server 排 storage executor，天然等所有 cell 落完）。
  //   2. refresh timeline 拉新行。
  const ingestGenerateTask = useCallback(async (taskId: number, samplesOverride?: typeof samples) => {
    if (ingestedRef.current.has(taskId)) return
    ingestedRef.current.add(taskId)
    try {
      const runSnap = runsRef.current.get(taskId)
      if (runSnap?.snapshot.mode === 'xy') {
        const sec = await api.getSecrets().catch(() => null)
        if (sec?.generate?.save_test_images) {
          let s = samplesOverride ?? []
          if (s.length === 0) {
            const st = await api.getMonitorState(taskId).catch(() => null)
            s = (st?.samples as typeof samples | undefined) ?? []
          }
          const xySamples = s
            .filter((x): x is typeof x & { xy: NonNullable<typeof x.xy> } => x.xy != null)
            .map((x) => ({ path: x.path, xy: { xi: x.xy.xi, yi: x.xy.yi } }))
          if (xySamples.length > 0) {
            const xv = axisView(runSnap.xDraft)
            const yv = runSnap.yDraft ? axisView(runSnap.yDraft) : null
            const blob = await composeXYMatrix({
              samples: xySamples,
              taskId,
              xLabels: xv.values.map((v) => axisText(xv, v)),
              yLabels: yv ? yv.values.map((v) => axisText(yv, v)) : null,
            })
            const fd = new FormData()
            fd.append('image', blob, 'xy plot.png')
            await fetch(`/api/generate/${taskId}/xy-composite`, { method: 'POST', body: fd })
          }
        }
      }
    } catch {
      // composite 补传 best-effort：失败不影响时间线（cells 已由 server 落盘），
      // 用户仍可从回看条目导出现拼下载。
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

  // generate_images_updated 去抖：XY 25 格逐格落盘会连发事件，300ms 合并成一次拉取。
  const imagesUpdatedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // SSE：task_state_changed 触发 task refresh；monitor_state_updated 推 sample 列表。
  useEventStream((evt) => {
    if (evt.type === 'task_state_changed') {
      void refreshBlockingTask()
      // 0.17 P-I：显示态 + 排队列表 + 逐条入库统一由 refreshLiveGenerates 推进。
      void refreshLiveGenerates()
    }
    if (evt.type === 'generate_images_updated') {
      // 落盘 executor 异步完成（首次 hash 大模型时可达分钟级）→ 增量刷时间线，
      // 「已释放」占位自动变缩略图。
      if (imagesUpdatedTimerRef.current) clearTimeout(imagesUpdatedTimerRef.current)
      imagesUpdatedTimerRef.current = setTimeout(() => {
        imagesUpdatedTimerRef.current = null
        void historyRef.current.refresh()
      }, 300)
    }
    const tid = taskIdRef.current
    if (tid == null) return
    if (evt.type === 'task_state_changed' && evt.task_id === tid) {
      // currentTask 的推进交给 refreshLiveGenerates；这里只在显示任务终态时清进度。
      if (evt.status === 'done' || evt.status === 'failed' || evt.status === 'canceled') {
        setProgress({ phase: null, batchIdx: null, batchTotal: null, currentStep: null, totalSteps: null })
      }
    } else if (
      evt.type === 'generate_phase'
      && String(evt.task_id) === String(tid)
    ) {
      // 阶段推进（load/clip/sample/vae）→ 进度条覆盖非采样阶段
      const name = typeof evt.name === 'string' ? (evt.name as GeneratePhase) : null
      setProgress((p) => ({ ...p, phase: name }))
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
      // 新 batch 开始 → 重置 step 进度，更新 batch 计数（phase 由后续 generate_phase 驱动）
      setProgress({
        phase: null,
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
    // 缺 params → 不回填，仅切图（entry.released 时也常见：图没了参数还在的
    // 反例——参数也没了的老行）。
    if (!entry.params) return
    void (async () => {
    let applied
    try {
      const projects = await catalog.loadProjects()
      const projIds = new Set(projects.map((p) => p.id))
      applied = await applySnapshot(
        entry.params!,
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
    // 底模不在 prefs 里（独立 ephemeral state）→ 单独回填。
    setBaseModel(applied.baseModel)
    const restoredXDraft = applied.xDraft && applied.xDraft.axis === 'lora_ckpt'
      ? { ...applied.xDraft, checkpointAnchor: applied.xDraft.loraIndex != null ? applied.loras[applied.xDraft.loraIndex] ?? null : null }
      : applied.xDraft
    const restoredYDraft = applied.yDraft && applied.yDraft.axis === 'lora_ckpt'
      ? { ...applied.yDraft, checkpointAnchor: applied.yDraft.loraIndex != null ? applied.loras[applied.yDraft.loraIndex] ?? null : null }
      : applied.yDraft
    const axisIndices = new Set(
      [restoredXDraft, restoredYDraft]
        .filter((draft): draft is NonNullable<typeof restoredXDraft> => Boolean(draft))
        .filter((draft) => draft.axis === 'lora_ckpt')
        .map((draft) => draft.loraIndex)
        .filter((index): index is number => Number.isInteger(index)),
    )
    setPrefs((prev) => {
      const base: GeneratePrefs = {
        ...prev,
        mode: applied.mode,
        modelFamily: applied.modelFamily,
        prompts: applied.prompts.length > 0 ? applied.prompts : prev.prompts,
        negPrompt: applied.negPrompt,
        width: applied.width,
        height: applied.height,
        aspect: aspectFromDimensions(applied.width, applied.height),
        steps: applied.steps,
        cfgScale: applied.cfgScale,
        samplerName: applied.samplerName,
        scheduler: applied.scheduler,
        seed: applied.seed,
        datasetPick: applied.datasetPick,
        datasetPrompt: applied.datasetPrompt,
        // 0.17 P-I：batch size 是瞬态值，点历史图**不回填**（用户设的值保持不变）。
      }
      if (applied.mode === 'single') {
        return {
          ...base,
          singleLoras: applied.loras,
          singleLoraUi: applied.loras.map(() => createLoraUiState(true)),
        }
      }
      return {
        ...base,
        xyFixedLoras: applied.loras.filter((_, index) => !axisIndices.has(index)),
        xyFixedLoraUi: applied.loras
          .filter((_, index) => !axisIndices.has(index))
          .map(() => createLoraUiState(true)),
        // Keep the old bucket for readers of v1 snapshots, but the new UI uses
        // xyFixedLoras plus the axis-owned checkpoint binding below.
        xyLoras: applied.loras,
        xDraft: restoredXDraft ?? prev.xDraft,
        yDraft: restoredYDraft ?? null,
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
    const datasetSuffix = datasetPrompt.trim()
    if (!prompts.some((p) => p.trim()) && !datasetSuffix) {
      toast(t('generate.promptOrDatasetRequired'), 'error')
      return
    }

    let xy_matrix: XYMatrixSpec | null = null
    let snapshotXDraft = xDraft
    let snapshotYDraft = yDraft
    // single：base LoRA = singleLoras 全发。xy：只发被轴引用的 anchor（见
    // buildXYMatrix —— xyLoras 会沉积 picker 切项目/版本/删轴遗留的孤儿 anchor，
    // 整桶发出去会让孤儿叠到每个 cell，正是反复出现的「混进没选过的 LoRA」根因）。
    let loraConfigs: LoraEntry[] = mode === 'single'
      ? enabledLoras(prefs.singleLoras, prefs.singleLoraUi)
      : []
    if (mode === 'xy') {
      // schema 强制 prompts 单条 + count=1
      if (prompts.filter((p) => p.trim()).length > 1) {
        toast(t('generate.xySinglePromptOnly'), 'error')
        return
      }
      try {
        const built = buildXYMatrix(
          xDraft, yDraft, [], prefs.xyFixedLoras, prefs.xyFixedLoraUi,
        )
        xy_matrix = built.xy_matrix
        loraConfigs = built.loraConfigs
        // Snapshots must store the wire lora_index generated by buildXYMatrix,
        // not the editor-only checkpointAnchor identity. History restore and
        // per-cell PNG metadata both address snapshot.loras by this index.
        snapshotXDraft = {
          ...xDraft,
          loraIndex: built.xy_matrix.x.lora_index ?? null,
        }
        snapshotYDraft = yDraft
          ? { ...yDraft, loraIndex: built.xy_matrix.y?.lora_index ?? null }
          : null
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
      // snapshot 记录实际送进 daemon 的 LoRA，而不是 sidebar 原始桶。XY 构建会
      // 丢弃未被轴引用的孤儿 anchor；继续从 loras 取会让 PNG 声称用了实际未
      // 加载的资源，也会把错误 hash 交给 Civitai。
      const snapshotLoras: SnapshotLora[] = loraConfigs.map((l) => ({
        name: loraBasename(l.path),
        scale: l.scale,
        project_id: l.project_id ?? null,
        version_id: l.version_id ?? null,
      }))
      const baseSnapshot: GenerateParamsSnapshot = {
        schema_version: PARAMS_SNAPSHOT_VERSION,
        mode,
        model_family: modelFamily,
        prompts,
        negative_prompt: negPrompt,
        width, height, steps,
        cfg_scale: cfgScale,
        sampler_name: samplerName,
        scheduler,
        count: 1,  // 0.17 P-I：每个 task 出 1 张；batch 拆成多 task（下面循环）
        seed,
        base_model: baseModel,
        text_encoder: modelFamily === 'krea2' ? effectiveTe : undefined,
        loras: snapshotLoras,
        xy_draft: mode === 'xy'
          ? {
              x: transformAxisRawForSnapshot(snapshotXDraft),
              y: snapshotYDraft ? transformAxisRawForSnapshot(snapshotYDraft) : null,
            }
          : null,
        dataset_pick: datasetPick,
        dataset_prompt: datasetPrompt,
      }
      // 0.17 P-I：count 现在 = **batch size**（每次入队的 task 数）。single 拆成 batch 个
      // task（各出 1 张、seed 递增区分）→ 在右栏时间线逐个排队；xy 一次一个矩阵（batch 忽略）。
      const batch = mode === 'xy' ? 1 : Math.max(1, batchSize)
      let firstId: number | null = null
      for (let i = 0; i < batch; i++) {
        const taskSeed = seed + i
        const snap: GenerateParamsSnapshot = { ...baseSnapshot, seed: taskSeed }
        const body: GenerateRequest = {
          prompts: mergedPrompts,
          model_family: modelFamily,
          base_model: baseModel ?? undefined,
          text_encoder: modelFamily === 'krea2' ? effectiveTe : undefined,
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
          xDraft: { ...snapshotXDraft },
          yDraft: snapshotYDraft ? { ...snapshotYDraft } : null,
          snapshot: snap,
        })
        if (firstId === null) {
          firstId = task.id
          // 点「开始生成」= 明确要看这次出图 → 回到实时视图：清掉正在回看的历史
          // override（否则结果区停留在老图，看不到新入队/正在跑的这次，XY 尤甚 ——
          // 出图慢，用户常停在回看态点生成）。P-I 删掉了「currentTask.id 变自动清
          // override」的 effect（多任务下会把回看中的 done 项踢回实时），这里改成只在
          // 用户显式提交时清，兼顾两者。
          setHistoryOverride(null)
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
  const xyImageCount = cellCount(
    axisView(xDraft).values.length,
    yDraft ? axisView(yDraft).values.length : null,
  )
  const generateLabel = submitting
    ? t('generate.generating')
    : mode === 'xy'
      ? t('generate.generateImageCount', { count: xyImageCount })
      : t('generate.startGenerate')

  const attachedDrawerOpen = (
    (catalogDrawerOpen && sidebarTab === 'lora')
    || (datasetPickerOpen && sidebarTab === 'prompts')
    || (axisDrawerOpen && mode === 'xy' && sidebarTab === 'xy')
  )

  return (
    <div className="fade-in flex flex-col" style={{ height: '100%', overflow: 'hidden' }}>
      <PageHeader
        title={t('generate.title')}
        subtitle={t('generate.subtitle')}
        actions={
          <div className="flex items-center gap-2">
            {currentTask && (
              <>
                <span className="caption">#{currentTask.id}</span>
                <StatusBadge status={currentTask.status} />
              </>
            )}
            {currentTask?.error_msg && (
              <span className="text-xs text-err max-w-[240px] truncate" title={currentTask.error_msg}>
                {currentTask.error_msg}
              </span>
            )}
            {/* 0.17 P-I：取消（当前显示 task）+ 清空队列（所有 pending）始终在位，不可用时
                disabled，放「清理显存」（DaemonControls）左边。 */}
            <button
              className="btn btn-ghost"
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

      {/* 三列各自独立滚动，整页固定高度 = viewport。relative：进度条 absolute 叠在顶部
          p-6 既有 gap 上、不占布局，出现/消失不推动内容（防页面抖动）。 */}
      <div className="relative p-6 flex gap-4 items-stretch flex-wrap xl:flex-nowrap flex-1 min-h-0">
        {/* 出图进度条：全宽细线（浏览器加载条式）+ 小相位文字，绝对定位叠在 header 与内容间
            的既有 gap 上；覆盖 load/clip/sample/vae 全阶段，切历史图也照常显示当前进度。 */}
        {(busy || progress.currentStep != null || progress.phase != null) && (
          <div className="absolute top-0 inset-x-0 z-10 pointer-events-none">
            <GenerateProgressBar busy={busy} progress={progress} />
          </div>
        )}

          {/* 左：sidebar — 单卡片包裹；内容区独立 scroll，底部 footer 固定 tab + 生成按钮 */}
          <div
            className={`card generate-sidebar relative z-30 flex flex-col w-full xl:w-[420px] shrink-0 self-stretch min-h-0 overflow-hidden ${attachedDrawerOpen ? 'generate-sidebar--drawer-open' : ''}`}
          >
            <div
              className="shrink-0"
              style={{ padding: 12, borderBottom: '1px solid var(--border-subtle)' }}
            >
              <ViewModeTabs mode={mode} onModeChange={setMode} />
            </div>
            {/* 内容区：各 section 常驻 DOM，用 display 切换以保留状态。每个 section
                自己管理滚动；XY 额外把局部 X/Y tab 固定在内容面板底部，使主 footer
                的分区 tab 和生成按钮不因二级导航出现而移动。 */}
            <div className="flex-1 min-h-0 overflow-hidden">

            {/* XY：顶部是展开编辑器的上下文操作，中间滚动内容，底部是局部二级 tab。 */}
            <div
              id="generate-sidebar-panel-xy"
              role="tabpanel"
              aria-labelledby="generate-sidebar-tab-xy"
              hidden={sidebarTab !== 'xy' || mode !== 'xy'}
              className="h-full min-h-0 flex-col"
              style={{ display: sidebarTab === 'xy' && mode === 'xy' ? 'flex' : 'none' }}
            >
              <div
                className="flex-1 min-h-0 overflow-y-auto"
                style={{ padding: 18, scrollbarGutter: 'stable both-edges' }}
              >
                <div
                  className="sticky z-10 flex justify-end bg-surface pb-3"
                  style={{ top: -18, marginTop: -18, paddingTop: 18 }}
                >
                  <ToolbarAction
                    label={axisDrawerOpen
                      ? t('generate.collapseCatalog')
                      : t('generate.editAxis', { label: activeAxis })}
                    icon={<SidebarToolIcon name={axisDrawerOpen ? 'collapse' : 'edit'} />}
                    onClick={() => {
                      const opening = !axisDrawerOpen
                      setAxisDrawerOpen(opening)
                      if (opening) {
                        setCatalogDrawerOpen(false)
                        setDatasetPickerOpen(false)
                      }
                    }}
                    aria-expanded={axisDrawerOpen}
                    aria-controls="xy-axis-editor-drawer"
                  />
                </div>
                <SidebarXYAxes
                  xDraft={xDraft}
                  yDraft={visibleYDraft}
                  yEnabled={yDraft !== null}
                  activeAxis={activeAxis}
                  onAxisChange={(axis, next) => axis === 'X' ? setXDraft(next) : setYDraft(next)}
                  onManualReorder={(axis) => setAxisOrderRevision((revision) => ({
                    ...revision,
                    [axis]: revision[axis] + 1,
                  }))}
                  fp8BaseModel={fp8BaseModel}
                />
              </div>
              <div
                className="shrink-0 bg-surface"
                data-testid="xy-axis-secondary-tabs"
                style={{ borderTop: '1px solid var(--border-subtle)', padding: '10px 12px' }}
              >
                <XYAxisToolbar
                  xDraft={xDraft}
                  yDraft={visibleYDraft}
                  activeAxis={activeAxis}
                  onSelectAxis={setActiveAxis}
                  onSwap={() => {
                    setPrefs((p) => ({
                      ...p,
                      xDraft: p.yDraft ?? virtualYDraftFor(p.xDraft, p.steps),
                      yDraft: p.xDraft,
                    }))
                    setAxisOrderRevision((revision) => ({
                      X: revision.Y + 1,
                      Y: revision.X + 1,
                    }))
                  }}
                />
              </div>
            </div>

            <div
              id="generate-sidebar-panel-lora"
              role="tabpanel"
              aria-labelledby="generate-sidebar-tab-lora"
              hidden={sidebarTab !== 'lora'}
              className="h-full overflow-y-auto"
              style={{
                display: sidebarTab === 'lora' ? undefined : 'none',
                padding: 18,
                scrollbarGutter: 'stable both-edges',
              }}
            >
              <div
                className="sticky z-10 flex justify-end bg-surface pb-3"
                style={{ top: -18, marginTop: -18, paddingTop: 18 }}
              >
                <ToolbarAction
                  label={catalogDrawerOpen ? t('generate.collapseCatalog') : t('generate.expandCatalog')}
                  icon={<SidebarToolIcon name={catalogDrawerOpen ? 'collapse' : 'plus'} />}
                  onClick={() => {
                    const opening = !catalogDrawerOpen
                    setCatalogDrawerOpen(opening)
                    if (opening) {
                      setAxisDrawerOpen(false)
                      setDatasetPickerOpen(false)
                    }
                  }}
                  aria-expanded={catalogDrawerOpen}
                  aria-controls="lora-catalog-drawer"
                />
              </div>
              <SidebarLoras loras={loras} ui={loraUi} onChange={setSelection} />
            </div>

            {/* tab=prompts */}
            <div
              id="generate-sidebar-panel-prompts"
              role="tabpanel"
              aria-labelledby="generate-sidebar-tab-prompts"
              hidden={sidebarTab !== 'prompts'}
              className="h-full overflow-y-auto"
              style={{
                display: sidebarTab === 'prompts' ? undefined : 'none',
                padding: 18,
                scrollbarGutter: 'stable both-edges',
              }}
            >
              <div
                className="sticky z-10 flex justify-end bg-surface pb-3"
                style={{ top: -18, marginTop: -18, paddingTop: 18 }}
              >
                <ToolbarAction
                  label={datasetPickerOpen ? t('generate.collapseCatalog') : t('generate.pickFromDataset')}
                  icon={<SidebarToolIcon name={datasetPickerOpen ? 'collapse' : 'dataset'} />}
                  onClick={() => {
                    const opening = !datasetPickerOpen
                    setDatasetPickerOpen(opening)
                    if (opening) {
                      setCatalogDrawerOpen(false)
                      setAxisDrawerOpen(false)
                    }
                  }}
                  aria-expanded={datasetPickerOpen}
                  aria-controls="prompt-dataset-drawer"
                />
              </div>
              <label className="caption block mb-1">{t('generate.positive')}</label>
              <PromptList prompts={prompts} onChange={setPrompts} modelFamily={modelFamily} />
              <label className="caption block mb-1 mt-3">{t('generate.negative')}</label>
              <NegPromptInput value={negPrompt} onChange={setNegPrompt} modelFamily={modelFamily} />
              <div className="mt-3 flex items-center justify-between gap-2">
                <label className="caption block">{t('generate.datasetPromptLabel')}</label>
                {(datasetPick || datasetPrompt) && (
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm text-2xs text-fg-tertiary"
                    title={t('generate.clearDatasetPromptTitle')}
                    onClick={() => setDatasetPick(null)}
                  >
                    {t('generate.clearDatasetPick')}
                  </button>
                )}
              </div>
              <PromptList
                prompts={[datasetPrompt]}
                onChange={(next) => setDatasetPrompt(next[0] ?? '')}
                modelFamily={modelFamily}
                placeholder={t('generate.datasetPromptPlaceholder')}
                ariaLabel={t('generate.datasetPromptAria')}
              />
            </div>

            {/* tab=config */}
            <div
              id="generate-sidebar-panel-config"
              role="tabpanel"
              aria-labelledby="generate-sidebar-tab-config"
              hidden={sidebarTab !== 'config'}
              className="h-full overflow-y-auto"
              style={{
                display: sidebarTab === 'config' ? undefined : 'none',
                padding: 18,
                scrollbarGutter: 'stable both-edges',
              }}
            >
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
                      {/* 文案与训练配置页共用 schema.enums.* 映射；选项按族白名单 */}
                      {SAMPLER_OPTIONS_BY_FAMILY[modelFamily].map((s) => (
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
                      {SCHEDULER_OPTIONS_BY_FAMILY[modelFamily].map((s) => (
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
                  <label className="caption block mb-1">{t('generate.modelFamily')}</label>
                  <select
                    className="input text-xs w-full"
                    value={modelFamily}
                    onChange={(e) => setModelFamily(e.target.value as GenerateFamily)}
                    aria-label={t('generate.modelFamily')}
                  >
                    <option value="anima">{schemaEnumLabel('model_family', 'anima', t)}</option>
                    <option value="krea2">{schemaEnumLabel('model_family', 'krea2', t)}</option>
                  </select>
                </div>
                <div>
                  <label className="caption block mb-1">{t('generate.baseModel')}</label>
                  <BaseModelSelect
                    value={baseModel}
                    onChange={onBaseModelChange}
                    family={modelFamily}
                    className="input text-xs w-full"
                    ariaLabel={t('generate.baseModel')}
                  />
                  <div className="text-2xs text-fg-tertiary font-mono mt-1">
                    {t('generate.baseModelHint')}
                  </div>
                </div>
                {modelFamily === 'krea2' && (
                  <div>
                    <label className="caption block mb-1">{t('generate.textEncoder')}</label>
                    <select
                      className="input text-xs w-full"
                      value={effectiveTe}
                      onChange={(e) => setTextEncoder(e.target.value as 'bf16' | 'fp8')}
                      aria-label={t('generate.textEncoder')}
                    >
                      <option value="bf16">{t('generate.textEncoderBf16')}</option>
                      <option value="fp8" disabled={!teOptions.fp8Ready}>
                        {teOptions.fp8Ready
                          ? t('generate.textEncoderFp8')
                          : t('generate.textEncoderFp8NotDownloaded')}
                      </option>
                    </select>
                  </div>
                )}
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
              {/* items-stretch：batch 框跟「开始生成」按钮等高（按钮 padding:12 定高度）。 */}
              <div className="flex items-stretch gap-3">
                {/* R-5：GPU 任务运行时不再硬禁用——后端准入（R-1）保证互斥，
                    提交只是入队排队（锚点 §4-5）。按钮 title 提示会排队。 */}
                <button
                  className="btn btn-primary flex-1"
                  style={{ padding: 12, fontWeight: 600, justifyContent: 'center' }}
                  onClick={handleGenerate}
                  disabled={submitting}
                  title={
                    activeBlockingTask
                      ? t('generate.queuedBehindActiveTask', { id: activeBlockingTask.id })
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
                    value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                    title={t('generate.batchSizeTitle')}
                    aria-label={t('generate.batchSizeTitle')}
                  />
                )}
              </div>
            </div>
          </div>

          <LoraCatalogDrawer
            open={catalogDrawerOpen && sidebarTab === 'lora'}
            onClose={() => setCatalogDrawerOpen(false)}
            loras={loras}
            ui={loraUi}
            onChange={setSelection}
          />
          {datasetPickerOpen && sidebarTab === 'prompts' && (
            <GenerateAttachedDrawer
              id="prompt-dataset-drawer"
              ariaLabel={t('generate.datasetPromptTitle')}
              testId="prompt-dataset-drawer"
            >
              <PromptFromDatasetPicker
                variant="drawer"
                value={datasetPick}
                onChange={setDatasetPick}
                onClose={() => setDatasetPickerOpen(false)}
              />
            </GenerateAttachedDrawer>
          )}
          <XYAxisEditorDrawer
            open={axisDrawerOpen && mode === 'xy' && sidebarTab === 'xy'}
            label={activeAxis}
            draft={activeAxis === 'Y' ? visibleYDraft : xDraft}
            otherAxis={activeAxis === 'X' ? yDraft?.axis ?? null : xDraft.axis}
            fixedLoras={enabledLoras(prefs.xyFixedLoras, prefs.xyFixedLoraUi)}
            manualOrderRevision={axisOrderRevision[activeAxis]}
            onChange={(next) => activeAxis === 'Y' ? setYDraft(next) : setXDraft(next)}
            onClose={() => setAxisDrawerOpen(false)}
          />

          {/* 中：card flex-1 占满列高。overflow-hidden（非 auto）——内容本就 fit（预览区
              flex-1 min-h-0，XY 网格自带滚动），auto 会因一点点溢出触发幻影滚动条、吃掉
              10px 宽把 card 挤窄 → 结果卡与右栏之间凭空多出 10px margin（#2 根因）。 */}
          <div className="flex-1 min-w-0 flex flex-col overflow-hidden self-stretch">
            <div className="card flex-1 flex flex-col overflow-hidden" style={{ padding: 0, minHeight: 0 }}>
              {/* 进度条已上移到页面 header 下（全宽细线），不再在结果卡内。 */}
              {historyOverride ? (
                <div className="flex-1 min-h-0 flex flex-col gap-2">
                  {historyOverride.released || historyOverride.images.length === 0 ? (
                    /* 已释放：图不可取（temp 会话结束 / 文件手删），参数已回填。 */
                    <div className="flex-1 grid place-items-center bg-sunken text-fg-tertiary text-sm">
                      {t('generate.releasedHint')}
                    </div>
                  ) : historyOverride.mode === 'xy' && historyOverride.xyMeta ? (
                    /* XY 回看：per-cell 信息齐 → PreviewXYGrid。imageUrl server
                       已拼好（disk / temp 统一）；taskId 供 GridCell fallback
                       （sample 端点带落盘 fallback，两边都能通）。compositeUrl
                       盘上有 composite 时给 → 导出走文件下载，不再 re-compose */
                    <PreviewXYGrid
                      samples={historyOverride.xyMeta.samples.map((s) => ({
                        path: s.path,
                        xy: {
                          xi: s.xy.xi, yi: s.xy.yi,
                          xv: s.xy.xv as never, yv: s.xy.yv as never,
                        },
                        imageUrl: s.imageUrl,
                      }))}
                      taskId={historyOverride.taskId}
                      xAxis={axisView({
                        axis: historyOverride.xyMeta.xAxis as never,
                        raw: historyOverride.xyMeta.xValues.join(', '),
                        loraIndex: null,
                      })}
                      yAxis={historyOverride.xyMeta.yAxis ? axisView({
                        axis: historyOverride.xyMeta.yAxis as never,
                        raw: (historyOverride.xyMeta.yValues as string[]).filter(Boolean).join(', '),
                        loraIndex: null,
                      }) : null}
                      onCellClick={undefined /* 历史回看不允许选 cell 进 compare */}
                      selectedIndices={[]}
                      compositeUrl={historyOverride.compositeUrl}
                    />
                  ) : (
                    /* single / legacy XY（无 xyMeta）→ 单图视图 */
                    <div className="flex-1 min-h-0 w-full">
                      <ZoomableImage
                        key={historyOverride.id}
                        src={entryImageUrl(historyOverride, 0)}
                        alt=""
                      />
                    </div>
                  )}
                  {/* XY 网格保留 folder 作批次标识；单图 filename 与
                      ZoomableImage readout 重复，不再显示。 */}
                  {historyOverride.xyMeta && historyOverride.xyFolder && (
                    <div className="text-xs text-fg-tertiary shrink-0 px-3 pb-2">
                      {historyOverride.xyFolder}
                    </div>
                  )}
                </div>
              ) : !currentTask ? (
                <div className="flex-1 grid place-items-center bg-sunken text-fg-tertiary text-sm">
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
                  xAxis={axisView(gridXDraft)}
                  yAxis={gridYDraft ? axisView(gridYDraft) : null}
                  onCellClick={handleCellClick}
                  selectedIndices={selectedIndices}
                />
              ) : samples.length === 0 && previewStep ? (
                <div className="flex-1 min-h-0 flex flex-col items-center gap-2">
                  <div className="flex-1 min-h-0 w-full flex items-center justify-center">
                    {/* 中间步预览是低分辨率 latent2rgb 图（模糊但能看出大致图）：铺满结果区
                        —— width/height:100% + object-contain 会放大小图并保持比例；旧的
                        maxWidth/maxHeight 只限上限，小图不放大 → 显示成中间一小块。 */}
                    <img
                      src={previewStep.dataUrl}
                      alt={`step ${previewStep.step}/${previewStep.total}`}
                      className="rounded-md"
                      style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    />
                  </div>
                  <div className="text-xs text-fg-tertiary shrink-0">
                    {t('generate.previewStep', { step: previewStep.step, total: previewStep.total })}
                  </div>
                </div>
              ) : samples.length === 0 ? (
                <div className="flex-1 grid place-items-center bg-sunken text-fg-tertiary text-sm">
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
          />
      </div>

      {/* daemon log 抽屉（fixed 定位 + translateY，隐藏时完全不可见，不占 layout） */}
      <DaemonLogDrawer open={logOpen} onClose={() => setLogOpen(false)} />
    </div>
  )
}
