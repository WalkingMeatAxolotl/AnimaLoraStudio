import { useEffect, useMemo, useRef, useState } from 'react'
import {
  api,
  type AttentionBackend,
  type GenerateRequest,
  type LoraEntry,
  type MonitorState,
  type Task,
  type XYMatrixSpec,
} from '../../api/client'
import PageHeader from '../../components/PageHeader'
import { useToast } from '../../components/Toast'
import { useEventStream } from '../../lib/useEventStream'
import NumField from './generate/NumField'
import PreviewCompare from './generate/PreviewCompare'
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
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [steps, setSteps] = useState(25)
  const [cfgScale, setCfgScale] = useState(4.0)
  const [count, setCount] = useState(1)
  const [seed, setSeed] = useState(0)
  const [loras, setLoras] = useState<LoraEntry[]>([])
  const [attentionBackend, setAttentionBackend] = useState<AttentionBackend>('flash_attn')

  // XY 模式 state（mode='single' 时不参与 enqueue）
  const [xDraft, setXDraft] = useState<XYAxisDraft>({ axis: 'steps', raw: '20, 25, 30', loraIndex: null })
  const [yDraft, setYDraft] = useState<XYAxisDraft | null>(null)

  // 双图对比：选中的 2 个 sample 索引（从 PreviewXYGrid cell click 收集）
  const [selectedIndices, setSelectedIndices] = useState<number[]>([])

  const [busy, setBusy] = useState(false)
  const [currentTask, setCurrentTask] = useState<Task | null>(null)
  const [monitorState, setMonitorState] = useState<MonitorState | null>(null)
  const taskIdRef = useRef<number | null>(null)
  taskIdRef.current = currentTask?.id ?? null

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
  const samples = monitorState?.samples ?? []

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
    }
  })

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
        attention_backend: attentionBackend,
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
    <div className="fade-in">
      <PageHeader eyebrow="工具" title="测试" subtitle="独立推理 · 单图 / XY 矩阵 / 双图对比（出图不保存，关页面即丢）" />

      <div className="p-6 flex flex-col gap-4">

        {/* ── 提示词（全宽） ── */}
        <div className="card" style={{ padding: 18 }}>
          <div className="text-md font-semibold mb-3">正向提示词</div>
          <PromptList prompts={prompts} onChange={setPrompts} />
          <div className="mt-4">
            <label className="caption block mb-1.5">负面提示词</label>
            <textarea
              className="input w-full font-mono text-sm resize-y"
              rows={3}
              value={negPrompt}
              onChange={(e) => setNegPrompt(e.target.value)}
            />
          </div>
        </div>

        {/* ── 主体：参数 + 结果 ── */}
        <div className="flex gap-4 items-start flex-wrap xl:flex-nowrap">

          {/* 左：参数 */}
          <div className="flex flex-col gap-4 w-full xl:w-[340px] shrink-0">

            <div className="card" style={{ padding: 18 }}>
              <div className="text-md font-semibold mb-3">生成参数</div>
              <div className="flex flex-col gap-3">
                <div className="flex gap-2">
                  <NumField label="宽度" value={width} onChange={setWidth} min={256} max={4096} step={64} />
                  <NumField label="高度" value={height} onChange={setHeight} min={256} max={4096} step={64} />
                </div>
                <div className="flex gap-2">
                  <NumField label="步数" value={steps} onChange={setSteps} min={1} max={150} />
                  <NumField label="CFG Scale" value={cfgScale} onChange={setCfgScale} min={0} max={20} step={0.5} />
                </div>
                <div className="flex gap-2">
                  {mode !== 'xy' && (
                    <NumField label="每 prompt 张数" value={count} onChange={setCount} min={1} max={32} />
                  )}
                  <NumField label="种子（0=随机）" value={seed} onChange={setSeed} min={0} />
                </div>
              </div>
            </div>

            <div className="card" style={{ padding: 18 }}>
              <div className="text-md font-semibold mb-3">LoRA</div>
              <SidebarLoras loras={loras} onChange={setLoras} projectLoras={projectLoras} />
            </div>

            {mode === 'xy' && (
              <SidebarXYAxes
                xDraft={xDraft}
                yDraft={yDraft}
                onXChange={setXDraft}
                onYChange={setYDraft}
                loras={loras}
              />
            )}

            <div className="flex flex-col gap-1">
              <label className="caption">加速</label>
              <select
                className="input"
                value={attentionBackend}
                onChange={(e) => setAttentionBackend(e.target.value as AttentionBackend)}
              >
                <option value="flash_attn">Flash Attention</option>
                <option value="xformers">xformers</option>
                <option value="none">无（PyTorch SDPA）</option>
              </select>
            </div>

            <div className="flex gap-2">
              <button className="btn btn-primary flex-1" onClick={handleGenerate} disabled={busy}>
                {generateLabel}
              </button>
              {cancelable && (
                <button className="btn btn-ghost" onClick={handleCancel} title="取消当前任务">
                  取消
                </button>
              )}
            </div>
          </div>

          {/* 右：结果 */}
          <div className="flex-1 min-w-0">
            <div className="card" style={{ padding: 18 }}>
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

              {!currentTask ? (
                <div
                  className="grid place-items-center rounded-md border border-subtle bg-sunken text-fg-tertiary text-sm"
                  style={{ minHeight: 260 }}
                >
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
              ) : (
                <SampleGallery samples={samples} taskId={currentTask.id} />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
