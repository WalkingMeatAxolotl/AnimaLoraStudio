import { useEffect, useMemo, useRef, useState } from 'react'
import {
  api,
  type GenerateRequest,
  type LoraEntry,
  type MonitorState,
  type Task,
  type XYMatrixSpec,
} from '../../api/client'
import PageHeader from '../../components/PageHeader'
import { useToast } from '../../components/Toast'
import { useEventStream } from '../../lib/useEventStream'
import AspectChips, { aspectFromDimensions, type AspectName } from './generate/AspectChips'
import DaemonControls from './generate/DaemonControls'
import NumField from './generate/NumField'
import PreviewCompare from './generate/PreviewCompare'
import PreviewHistoryRail from './generate/PreviewHistoryRail'
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

export default function GeneratePage() {
  const { toast } = useToast()

  const [mode, setMode] = useState<ViewMode>('single')
  const [prompts, setPrompts] = useState<string[]>(['newest, safe, 1girl, masterpiece, best quality'])
  const [negPrompt, setNegPrompt] = useState(DEFAULT_NEG)
  const [aspect, setAspect] = useState<AspectName>('1:1')
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [steps, setSteps] = useState(25)
  const [cfgScale, setCfgScale] = useState(4.0)
  const [count, setCount] = useState(1)
  const [seed, setSeed] = useState(0)
  const [loras, setLoras] = useState<LoraEntry[]>([])
  // commit C: attention backend 已从 Generate 页移到 Settings；server 端
  // enqueue_generate 会自动从 secrets.generate.attention_backend 注入。

  // XY 模式 state（mode='single' 时不参与 enqueue）
  const [xDraft, setXDraft] = useState<XYAxisDraft>({ axis: 'steps', raw: '20, 25, 30', loraIndex: null })
  const [yDraft, setYDraft] = useState<XYAxisDraft | null>(null)

  // 双图对比：选中的 2 个 sample 索引（从 PreviewXYGrid cell click 收集）
  const [selectedIndices, setSelectedIndices] = useState<number[]>([])

  const [busy, setBusy] = useState(false)
  const [currentTask, setCurrentTask] = useState<Task | null>(null)
  const [monitorState, setMonitorState] = useState<MonitorState | null>(null)
  // commit 14：中间步预览（仅 single 模式有意义；XY/对比 cell 多预览意义小）
  const [previewStep, setPreviewStep] = useState<{ step: number; total: number; dataUrl: string } | null>(null)
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
      if (next.length === 2) setMode('compare')
      return next
    })
  }

  const compareEnabled =
    (mode === 'xy' && selectedIndices.length === 2) || mode === 'compare'

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
          setBusy(false)
        }
      }).catch(() => { /* task 已清也走这里 */ })
    } else if (
      evt.type === 'monitor_state_updated'
      && String(evt.task_id) === String(tid)
      && evt.state
    ) {
      setMonitorState(evt.state as MonitorState)
    } else if (
      evt.type === 'generate_preview_step'
      && String(evt.task_id) === String(tid)
      && typeof evt.image_b64 === 'string'
    ) {
      // commit 14：中间步预览。data URL 直接覆盖主图 src 直到下一步或最终图来
      setPreviewStep({
        step: Number(evt.step) || 0,
        total: Number(evt.total) || 0,
        dataUrl: `data:image/jpeg;base64,${evt.image_b64}`,
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
  useEffect(() => {
    if (!currentTask || currentTask.status !== 'done') return
    if (samples.length === 0) return
    const snap = lastSnapshotRef.current
    if (snap && snap.taskId === currentTask.id && snap.mode === mode) return
    lastSnapshotRef.current = { taskId: currentTask.id, mode }
    const taskId = currentTask.id
    // 选封面 sample
    let coverIdx = 0
    if (mode === 'compare' && selectedIndices.length === 2) {
      coverIdx = selectedIndices[0]
    }
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
    } else if (mode === 'compare') {
      badge = '2×'
    }
    const filenames = samples
      .map((s) => s.path.split(/[\\/]/).pop() ?? '')
      .filter(Boolean)
    void makeThumbnail(api.generateSampleUrl(taskId, filename), 256)
      .then((dataUrl) => history.add({
        mode,
        taskId,
        thumbnailDataUrl: dataUrl,
        filenames,
        badge: badge || undefined,
      }))
      .catch(() => { /* thumbnail 失败 — 不入库（避免无封面 entry） */ })
  }, [currentTask, samples, mode, selectedIndices, history])

  const handleHistorySelect = (entry: HistoryEntry) => {
    setHistoryOverride(entry)
  }

  const handleGenerate = async () => {
    if (!prompts.some((p) => p.trim())) {
      toast('请输入至少一条提示词', 'error')
      return
    }

    let xy_matrix: XYMatrixSpec | null = null
    if (mode === 'xy') {
      // schema 强制 prompts 单条 + count=1
      if (prompts.filter((p) => p.trim()).length > 1) {
        toast('XY 模式只支持单条 prompt（多 prompt 与 XY 互斥）', 'error')
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

    setBusy(true)
    setCurrentTask(null)
    setMonitorState(null)
    setSelectedIndices([])  // 新一轮生成 — 旧选择已失效
    try {
      const body: GenerateRequest = {
        prompts: prompts.filter((p) => p.trim()),
        negative_prompt: negPrompt,
        width, height, steps,
        count: mode === 'xy' ? 1 : count,
        seed,
        cfg_scale: cfgScale,
        lora_configs: loras.filter((l) => l.path.trim()),
        // attention_backend 不带：server 自动从 secrets.generate.attention_backend 读
        xy_matrix,
      }
      const t = await api.enqueueGenerate(body)
      setCurrentTask(t)
      toast(`测试任务 #${t.id} 已入队`, 'success')
    } catch (e) {
      toast(String(e), 'error')
      setBusy(false)
    }
  }

  const handleCancel = async () => {
    if (!currentTask) return
    try {
      await api.cancelTask(currentTask.id)
      toast(`已请求取消 #${currentTask.id}`, 'info')
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const cancelable = currentTask
    && (currentTask.status === 'pending' || currentTask.status === 'running')

  const generateLabel = busy
    ? '生成中…'
    : mode === 'xy' && xyCellCount > 0
      ? `开始生成 · ${xyCellCount} 张`
      : '开始生成'

  return (
    <div className="fade-in flex flex-col" style={{ minHeight: '100vh' }}>
      <PageHeader
        eyebrow="工具"
        title="测试"
        subtitle="独立推理 · 单图 / XY 矩阵 / 双图对比（出图不保存，关页面即丢）"
        actions={<DaemonControls />}
      />

      <div className="p-6 flex gap-4 items-stretch flex-wrap xl:flex-nowrap flex-1 min-h-0">

          {/* 左：sidebar — 对齐 Test 重设计.html v1：LoRA → 提示词 → 参数 → Generate bar */}
          <div className="flex flex-col gap-4 w-full xl:w-[420px] shrink-0">

            {/* mode=single：独立 LoRA 卡片；mode=xy：LoRA 选择合并到 XY 卡片顶部 */}
            {mode === 'single' && (
              <div className="card" style={{ padding: 18 }}>
                <div className="flex items-baseline justify-between mb-3">
                  <h3 className="m-0 text-md font-semibold">LoRA</h3>
                  <span className="text-xs text-fg-tertiary">从项目挑版本，权重单滑块调</span>
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
              <h3 className="m-0 text-md font-semibold mb-3">提示词</h3>
              <label className="caption block mb-1">正向</label>
              <PromptList prompts={prompts} onChange={setPrompts} />
              <label className="caption block mb-1 mt-3">负向</label>
              <textarea
                className="input w-full font-mono text-xs resize-y"
                rows={5}
                value={negPrompt}
                onChange={(e) => setNegPrompt(e.target.value)}
              />
            </div>

            <div className="card" style={{ padding: 18 }}>
              <h3 className="m-0 text-md font-semibold mb-3">采样参数</h3>
              <div className="flex flex-col gap-3">
                <div>
                  <label className="caption block mb-1.5">画幅</label>
                  <AspectChips
                    aspect={aspect}
                    onPick={(a, w, h) => {
                      setAspect(a)
                      if (w && h) { setWidth(w); setHeight(h) }
                    }}
                  />
                </div>
                <div className="flex gap-2 items-end">
                  <NumField label="宽" value={width} onChange={(v) => { setWidth(v); setAspect(aspectFromDimensions(v, height)) }} min={256} max={4096} step={64} />
                  <NumField label="高" value={height} onChange={(v) => { setHeight(v); setAspect(aspectFromDimensions(width, v)) }} min={256} max={4096} step={64} />
                  <button
                    type="button"
                    onClick={() => {
                      const newW = height, newH = width
                      setWidth(newW); setHeight(newH)
                      setAspect(aspectFromDimensions(newW, newH))
                    }}
                    title="交换宽高（横版 ↔ 竖版）"
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
                  <NumField label="步数" value={steps} onChange={setSteps} min={1} max={150} />
                  <NumField label="CFG" value={cfgScale} onChange={setCfgScale} min={0} max={20} step={0.5} />
                  {mode !== 'xy' && (
                    <NumField label="每 prompt" value={count} onChange={setCount} min={1} max={32} />
                  )}
                </div>
                <NumField
                  label="种子（0 = 随机）"
                  value={seed}
                  onChange={setSeed}
                  min={0}
                />
                <div className="text-2xs text-fg-tertiary font-mono" style={{ marginTop: -4 }}>
                  填具体数字 → 固定可复现；0 → 每次随机
                </div>
              </div>
            </div>

            {/* Generate bar：对齐 design 的 .gen-bar — 橙色大按钮 + 右侧 meta */}
            <div
              className="flex items-center gap-3"
              style={{
                padding: '10px 12px',
                borderRadius: 'var(--r-lg)',
                border: '1px solid var(--border-subtle)',
                background: 'var(--bg-elevated)',
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
                <button className="btn btn-ghost" onClick={handleCancel} title="取消当前任务">
                  取消
                </button>
              )}
              {!cancelable && (
                <div className="font-mono text-xs text-fg-tertiary text-right" style={{ lineHeight: 1.3 }}>
                  <div>{width}×{height}</div>
                  <div>{busy ? '生成中…' : '与训练共享 GPU'}</div>
                </div>
              )}
            </div>
          </div>

          {/* 中：结果（高度填满 viewport 剩余） */}
          <div className="flex-1 min-w-0 flex flex-col">
            <div className="card flex-1 flex flex-col" style={{ padding: 18 }}>
              <div className="flex items-center justify-between gap-2 mb-4 flex-wrap">
                <div className="flex items-center gap-2">
                  <span className="text-md font-semibold">生成结果</span>
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
                <ViewModeTabs mode={mode} onModeChange={setMode} compareEnabled={compareEnabled} />
              </div>

              {historyOverride ? (
                <div className="flex flex-col items-center gap-2">
                  <img
                    key={historyOverride.id}
                    src={api.generateSampleUrl(historyOverride.taskId, historyOverride.filenames[0] ?? '')}
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).src = historyOverride.thumbnailDataUrl
                      ;(e.currentTarget as HTMLImageElement).title = '原图已释放，仅缩略图可见'
                    }}
                    alt={`history #${historyOverride.taskId}`}
                    style={{ maxWidth: '100%', maxHeight: 600, borderRadius: 6 }}
                  />
                  <div className="text-xs text-fg-tertiary">
                    历史 #{historyOverride.taskId}
                    {historyOverride.badge ? ` · ${historyOverride.badge}` : ''}
                    <button
                      className="btn btn-ghost text-xs ml-2"
                      style={{ padding: '2px 8px' }}
                      onClick={() => setHistoryOverride(null)}
                    >
                      返回当前
                    </button>
                  </div>
                </div>
              ) : !currentTask ? (
                <div className="flex-1 grid place-items-center rounded-md border border-subtle bg-sunken text-fg-tertiary text-sm">
                  填写参数后点击「开始生成」
                </div>
              ) : mode === 'compare' && selectedIndices.length === 2 ? (
                <PreviewCompare
                  samples={samples}
                  taskId={currentTask.id}
                  selectedIndices={selectedIndices as [number, number]}
                  xDraft={xDraft}
                  yDraft={yDraft}
                  onBack={() => setMode('xy')}
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
                <div className="flex flex-col items-center gap-2">
                  <img
                    src={previewStep.dataUrl}
                    alt={`step ${previewStep.step}/${previewStep.total}`}
                    style={{ maxWidth: '100%', maxHeight: 600, borderRadius: 6 }}
                  />
                  <div className="text-xs text-fg-tertiary">
                    采样 {previewStep.step} / {previewStep.total}（中间步预览）
                  </div>
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
          />
      </div>
    </div>
  )
}
