import { useRef, useState } from 'react'
import {
  api,
  type AttentionBackend,
  type GenerateRequest,
  type LoraEntry,
  type MonitorState,
  type Task,
} from '../../api/client'
import PageHeader from '../../components/PageHeader'
import { useToast } from '../../components/Toast'
import { useEventStream } from '../../lib/useEventStream'
import LoraList from './generate/LoraList'
import NumField from './generate/NumField'
import PromptList from './generate/PromptList'
import SampleGallery from './generate/SampleGallery'
import StatusBadge from './generate/StatusBadge'
import { DEFAULT_NEG } from './generate/types'
import { useRecentLoras } from './generate/useRecentLoras'

export default function GeneratePage() {
  const { toast } = useToast()

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

  const [busy, setBusy] = useState(false)
  const [currentTask, setCurrentTask] = useState<Task | null>(null)
  const [monitorState, setMonitorState] = useState<MonitorState | null>(null)
  const taskIdRef = useRef<number | null>(null)
  taskIdRef.current = currentTask?.id ?? null

  const recentLoras = useRecentLoras()
  const samples = monitorState?.samples ?? []

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
    setBusy(true)
    setCurrentTask(null)
    setMonitorState(null)
    try {
      const body: GenerateRequest = {
        prompts: prompts.filter((p) => p.trim()),
        negative_prompt: negPrompt,
        width, height, steps, count, seed,
        cfg_scale: cfgScale,
        lora_configs: loras.filter((l) => l.path.trim()),
        attention_backend: attentionBackend,
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

  return (
    <div className="fade-in">
      <PageHeader eyebrow="工具" title="测试" subtitle="独立运行推理，复用训练采样逻辑（出图不保存，关页面即丢）" />

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
          <div className="flex flex-col gap-4 w-full xl:w-[320px] shrink-0">

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
                  <NumField label="每 prompt 张数" value={count} onChange={setCount} min={1} max={32} />
                  <NumField label="种子（0=随机）" value={seed} onChange={setSeed} min={0} />
                </div>
              </div>
            </div>

            <div className="card" style={{ padding: 18 }}>
              <div className="text-md font-semibold mb-3">LoRA</div>
              <LoraList loras={loras} onChange={setLoras} recent={recentLoras} />
            </div>

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
                {busy ? '生成中…' : '开始生成'}
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
            {currentTask ? (
              <div className="card" style={{ padding: 18 }}>
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-md font-semibold">生成结果</span>
                  <span className="caption">#{currentTask.id}</span>
                  <StatusBadge status={currentTask.status} />
                  {currentTask.error_msg && (
                    <span className="text-xs text-err ml-1">{currentTask.error_msg}</span>
                  )}
                </div>
                <SampleGallery samples={samples} taskId={currentTask.id} />
              </div>
            ) : (
              <div
                className="grid place-items-center rounded-md border border-subtle bg-sunken text-fg-tertiary text-sm"
                style={{ minHeight: 260 }}
              >
                填写参数后点击「开始生成」
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
