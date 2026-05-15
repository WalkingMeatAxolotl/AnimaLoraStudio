import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useOutletContext } from 'react-router-dom'
import {
  api,
  type Job,
  type PreprocessedItem,
  type PreprocessPendingItem,
  type ProjectDetail,
  type UpscalerVariant,
  type Version,
} from '../../../api/client'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import StepShell from '../../../components/StepShell'
import { useDialog } from '../../../components/Dialog'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

interface Status {
  job: Job | null
  log_tail: string
  summary: { download_count: number; processed_count: number; pending_count: number }
}

interface FilesView {
  processed: PreprocessedItem[]
  pending: PreprocessPendingItem[]
  summary: Status['summary']
}

const STATUS_COLOR: Record<Job['status'], string> = {
  pending: 'badge badge-neutral',
  running: 'badge badge-warn',
  done: 'badge badge-ok',
  failed: 'badge badge-err',
  canceled: 'badge badge-neutral',
}

const FALLBACK_MODEL = '4x-AnimeSharp'
const TILE_OPTIONS = [128, 192, 256, 384, 512] as const
type Device = 'auto' | 'cuda' | 'cpu'
const DEVICE_OPTIONS: { value: Device; label: string }[] = [
  { value: 'auto', label: '自动（优先 CUDA）' },
  { value: 'cuda', label: 'CUDA' },
  { value: 'cpu', label: 'CPU' },
]

// 目标分辨率预设 — LoRA 训练桶常用面积。
// value=null 是「关闭智能」模式，直接 4× 模型输出（老路径，盘费高）。
type TargetPreset = { label: string; edge: number | null }
const TARGET_PRESETS: TargetPreset[] = [
  { label: '768²',  edge: 768 },
  { label: '1024² (推荐)', edge: 1024 },
  { label: '1536²', edge: 1536 },
  { label: '2048²', edge: 2048 },
  { label: '自定义', edge: 0 },     // edge=0 触发自定义 input
  { label: '关闭 (4× 输出)', edge: null },
]
const DEFAULT_TARGET_EDGE = 1024

export default function PreprocessPage() {
  const { project, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()

  const [files, setFiles] = useState<FilesView | null>(null)
  const [status, setStatus] = useState<Status | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [tileSize, setTileSize] = useState<number>(256)
  const [device, setDevice] = useState<Device>('auto')
  // targetEdge: 边长（像素），平方就是面积；null = 关闭智能；0 = 自定义中
  const [targetEdge, setTargetEdge] = useState<number | null>(DEFAULT_TARGET_EDGE)
  const [customEdge, setCustomEdge] = useState<string>(String(DEFAULT_TARGET_EDGE))
  const [pendingSel, setPendingSel] = useState<Set<string>>(new Set())
  const [pendingAnchor, setPendingAnchor] = useState<string | null>(null)
  const [processedSel, setProcessedSel] = useState<Set<string>>(new Set())
  const [processedAnchor, setProcessedAnchor] = useState<string | null>(null)
  // 大图预览：bucket + index 引用 processedNames / pendingNames
  const [preview, setPreview] = useState<{ bucket: 'processed' | 'pending'; idx: number } | null>(null)

  // 模型权重就绪状态（catalog 取一次，下载完成后用户手动刷新或 SSE 更新）
  const [allUpscalers, setAllUpscalers] = useState<UpscalerVariant[]>([])
  // 当前选中的放大器 label。初值 fallback；refreshUpscaler 拉 catalog.upscalers.current 覆盖
  const [selectedModel, setSelectedModel] = useState<string>(FALLBACK_MODEL)
  const [downloadingModel, setDownloadingModel] = useState(false)
  const upscaler = useMemo<UpscalerVariant | null>(
    () => allUpscalers.find((x) => x.label === selectedModel) ?? null,
    [allUpscalers, selectedModel],
  )

  const refreshFiles = useCallback(async () => {
    try {
      const r = await api.listPreprocessFiles(project.id)
      setFiles(r)
    } catch {
      /* ignore */
    }
  }, [project.id])

  const refreshStatus = useCallback(async () => {
    try {
      const r = await api.getPreprocessStatus(project.id)
      setStatus(r)
      setLogs(r.log_tail ? r.log_tail.split('\n') : [])
    } catch {
      /* ignore */
    }
  }, [project.id])

  const refreshUpscaler = useCallback(async () => {
    try {
      const cat = await api.getModelsCatalog()
      const variants = cat.upscalers?.variants ?? []
      setAllUpscalers(variants)
      const current = cat.upscalers?.current
      // 后端兜底逻辑已经保证 current 合法（预设 or 已存在 custom）；前端只在
      // 后端返回空时回退 FALLBACK_MODEL。setSelectedModel 是 idempotent。
      setSelectedModel(current || FALLBACK_MODEL)
    } catch {
      /* ignore */
    }
  }, [])

  const changeSelectedModel = useCallback(async (label: string) => {
    setSelectedModel(label)
    try {
      await api.selectUpscaler(label)
    } catch (e) {
      toast(String(e), 'error')
      void refreshUpscaler()  // 失败回滚
    }
  }, [refreshUpscaler, toast])

  useEffect(() => {
    void refreshFiles()
    void refreshStatus()
    void refreshUpscaler()
  }, [refreshFiles, refreshStatus, refreshUpscaler])

  // SSE：日志 / job 状态 / 每图完成 / 模型下载 完成都即时反映。
  // 不轮询 — worker 每张图后通过 stdout 标记行 → supervisor publish
  // `preprocess_progress` 事件，前端收到后刷新 files。
  const jobIdRef = useRef<number | null>(null)
  jobIdRef.current = status?.job?.id ?? null
  useEventStream((evt) => {
    const jid = jobIdRef.current
    if (evt.type === 'job_log_appended' && jid && evt.job_id === jid) {
      setLogs((prev) => [...prev, String(evt.text ?? '')])
    } else if (evt.type === 'preprocess_progress' && jid && evt.job_id === jid) {
      // 每张图完成都刷一下 files —— grid / 进度条 / 盘占用 全部跟着动
      void refreshFiles()
    } else if (evt.type === 'job_state_changed' && jid && evt.job_id === jid) {
      void refreshStatus()
      if (evt.status === 'done' || evt.status === 'failed' || evt.status === 'canceled') {
        void refreshFiles()
        void reload()
      }
    } else if (evt.type === 'project_state_changed' && evt.project_id === project.id) {
      void refreshFiles()
    } else if (evt.type === 'model_download_changed') {
      void refreshUpscaler()
    }
  })

  const job = status?.job ?? null
  const isLive = job?.status === 'running' || job?.status === 'pending'
  const summary = files?.summary ?? status?.summary ?? {
    download_count: 0,
    processed_count: 0,
    pending_count: 0,
  }
  const modelReady = !!upscaler?.exists

  // ----- 操作 ---------------------------------------------------------------
  const downloadModel = async () => {
    if (downloadingModel) return
    // custom（kind='custom'）的不能从这里再下载——它本来就是用户手动下来的；
    // 走自定义下载流程在 Settings 页里处理。
    if (upscaler?.kind === 'custom') {
      toast('自定义模型请前往设置 → 预处理重新下载', 'error')
      return
    }
    setDownloadingModel(true)
    try {
      await api.startModelDownload({ model_id: 'upscaler', variant: selectedModel })
      toast(`开始下载 ${selectedModel}`, 'success')
      // 让 SSE 推 model_download_changed；这里也立即刷一下兜底
      setTimeout(() => void refreshUpscaler(), 1500)
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setDownloadingModel(false)
    }
  }

  const startPreprocess = async (
    mode: 'all' | 'selected' | 'all_force',
    names?: string[],
  ) => {
    if (!modelReady) {
      toast(`请先下载放大器：${selectedModel}`, 'error')
      return
    }
    // 解析 targetEdge → target_area。0 走 customEdge；null 关闭智能；正数平方
    let target_area: number | null = null
    if (targetEdge === null) {
      target_area = null
    } else if (targetEdge === 0) {
      const n = Number(customEdge)
      if (!Number.isFinite(n) || n < 256 || n > 4096) {
        toast('自定义边长需在 256-4096 之间', 'error')
        return
      }
      target_area = Math.round(n) * Math.round(n)
    } else {
      target_area = targetEdge * targetEdge
    }
    setBusy(true)
    try {
      const j = await api.startPreprocess(project.id, {
        mode,
        names,
        model: selectedModel,
        tile_size: tileSize,
        device,
        target_area,
      })
      setLogs([])
      setStatus((prev) => ({
        job: j,
        log_tail: '',
        summary: prev?.summary ?? summary,
      }))
      toast(`开始预处理 #${j.id}`, 'success')
      setPendingSel(new Set())
      setPendingAnchor(null)
      setProcessedSel(new Set())
      setProcessedAnchor(null)
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const cancel = async () => {
    if (!job) return
    try {
      await api.cancelJob(job.id)
      toast('已取消', 'success')
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const deleteProcessed = async () => {
    if (processedSel.size === 0) return
    if (!(await confirm(
      `从 preprocess/ 删除 ${processedSel.size} 张产物（这些图会回到「待处理」列表）？`,
      { tone: 'danger', okText: '删除' },
    ))) return
    try {
      const r = await api.deletePreprocessFiles(project.id, Array.from(processedSel))
      toast(
        `已删除 ${r.deleted.length} 张${r.missing.length ? ` · 跳过 ${r.missing.length}` : ''}`,
        'success',
      )
      setProcessedSel(new Set())
      setProcessedAnchor(null)
      await refreshFiles()
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  // ----- grids --------------------------------------------------------------
  const pendingNames = useMemo(
    () => (files?.pending ?? []).map((it) => it.name),
    [files],
  )
  const processedNames = useMemo(
    () => (files?.processed ?? []).map((it) => it.name),
    [files],
  )

  const pendingItems = useMemo(
    () =>
      pendingNames.map((n) => ({
        name: n,
        thumbUrl: api.projectThumbUrl(project.id, n),
      })),
    [pendingNames, project.id],
  )
  const processedItems = useMemo(
    () =>
      (files?.processed ?? []).map((it) => ({
        name: it.name,
        thumbUrl: api.preprocessThumbUrl(project.id, it.name),
        meta: it.dst_size
          ? [
              it.action ?? 'upscale',
              `${it.src_size?.join('×') ?? '?'} → ${it.dst_size.join('×')}`,
            ].join('  ')
          : undefined,
      })),
    [files, project.id],
  )

  return (
    <StepShell
      idx={2}
      title="预处理"
      subtitle="放大 / 裁剪 / 涂抹 — 第一阶段：放大"
      actions={
        <Link to="/tools/settings" className="btn btn-ghost btn-sm">
          设置
        </Link>
      }
    >
      <div className="flex flex-col h-full gap-3 min-h-0">
        <div className="grid gap-3 flex-1 min-h-0" style={{ gridTemplateColumns: '1fr 260px' }}>
          {/* 左栏 */}
          <div className="flex flex-col gap-2 min-h-0 min-w-0">
            <OperationPanel
              tileSize={tileSize}
              setTileSize={setTileSize}
              device={device}
              setDevice={setDevice}
              targetEdge={targetEdge}
              setTargetEdge={setTargetEdge}
              customEdge={customEdge}
              setCustomEdge={setCustomEdge}
              modelReady={modelReady}
              downloadingModel={downloadingModel}
              onDownloadModel={() => void downloadModel()}
              upscaler={upscaler}
              allUpscalers={allUpscalers}
              selectedModel={selectedModel}
              onSelectedModelChange={(label) => void changeSelectedModel(label)}
              pendingCount={summary.pending_count}
              pendingSelCount={pendingSel.size}
              busy={busy || isLive}
              onStartAll={() => void startPreprocess('all')}
              onStartSelected={() =>
                void startPreprocess('selected', Array.from(pendingSel))
              }
            />

            {job && (
              <JobStrip
                job={job}
                logs={logs}
                onCancel={isLive ? cancel : undefined}
              />
            )}

            {/* 已处理 grid */}
            <ProcessedSection
              items={processedItems}
              selected={processedSel}
              onSelect={(name, e) => {
                const r = applySelection(processedSel, name, e, processedNames, processedAnchor)
                setProcessedSel(r.next)
                setProcessedAnchor(r.anchor)
              }}
              onPreview={(name) => {
                const i = processedNames.indexOf(name)
                if (i >= 0) setPreview({ bucket: 'processed', idx: i })
              }}
              onSelectAll={() => setProcessedSel(new Set(processedNames))}
              onClear={() => {
                setProcessedSel(new Set())
                setProcessedAnchor(null)
              }}
              onDelete={() => void deleteProcessed()}
            />

            {/* 待处理 grid */}
            <PendingSection
              items={pendingItems}
              selected={pendingSel}
              onSelect={(name, e) => {
                const r = applySelection(pendingSel, name, e, pendingNames, pendingAnchor)
                setPendingSel(r.next)
                setPendingAnchor(r.anchor)
              }}
              onPreview={(name) => {
                const i = pendingNames.indexOf(name)
                if (i >= 0) setPreview({ bucket: 'pending', idx: i })
              }}
              onSelectAll={() => setPendingSel(new Set(pendingNames))}
              onClear={() => {
                setPendingSel(new Set())
                setPendingAnchor(null)
              }}
            />
          </div>

          {/* 右栏统计 */}
          <PreprocessSidebar
            summary={summary}
            upscaler={upscaler}
            selectedModel={selectedModel}
            tileSize={tileSize}
            processed={files?.processed ?? []}
            targetEdge={targetEdge}
          />
        </div>
      </div>

      {preview && (() => {
        const names = preview.bucket === 'processed' ? processedNames : pendingNames
        const name = names[preview.idx]
        if (!name) return null
        const src =
          preview.bucket === 'processed'
            ? api.preprocessThumbUrl(project.id, name, 1600)
            : api.projectThumbUrl(project.id, name, 'download', 1600)
        return (
          <ImagePreviewModal
            src={src}
            caption={name}
            hasPrev={preview.idx > 0}
            hasNext={preview.idx < names.length - 1}
            onClose={() => setPreview(null)}
            onPrev={() =>
              preview.idx > 0 && setPreview({ ...preview, idx: preview.idx - 1 })
            }
            onNext={() =>
              preview.idx < names.length - 1 &&
              setPreview({ ...preview, idx: preview.idx + 1 })
            }
          />
        )
      })()}
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// 操作 panel
// ---------------------------------------------------------------------------

interface OperationPanelProps {
  tileSize: number
  setTileSize: (n: number) => void
  device: Device
  setDevice: (d: Device) => void
  targetEdge: number | null
  setTargetEdge: (n: number | null) => void
  customEdge: string
  setCustomEdge: (s: string) => void
  modelReady: boolean
  downloadingModel: boolean
  onDownloadModel: () => void
  upscaler: UpscalerVariant | null
  allUpscalers: UpscalerVariant[]
  selectedModel: string
  onSelectedModelChange: (label: string) => void
  pendingCount: number
  pendingSelCount: number
  busy: boolean
  onStartAll: () => void
  onStartSelected: () => void
}

function OperationPanel({
  tileSize,
  setTileSize,
  device,
  setDevice,
  targetEdge,
  setTargetEdge,
  customEdge,
  setCustomEdge,
  modelReady,
  downloadingModel,
  onDownloadModel,
  upscaler,
  allUpscalers,
  selectedModel,
  onSelectedModelChange,
  pendingCount,
  pendingSelCount,
  busy,
  onStartAll,
  onStartSelected,
}: OperationPanelProps) {
  // 当前下拉选中哪一档：用 edge 作为 value（null→'off', 0→'custom', 正数→str）
  const selectValue =
    targetEdge === null ? 'off' : targetEdge === 0 ? 'custom' : String(targetEdge)
  const handlePresetChange = (v: string) => {
    if (v === 'off') setTargetEdge(null)
    else if (v === 'custom') setTargetEdge(0)
    else setTargetEdge(Number(v))
  }
  // 在 model panel 上方加一条"目标分辨率"说明，紧凑
  const targetHint = targetEdge === null
    ? '关闭：所有图都走 4× 模型（盘费高）'
    : targetEdge === 0
      ? `自定义：约 ${customEdge}² 像素`
      : `${targetEdge}² ≈ ${(targetEdge * targetEdge / 1e6).toFixed(2)}M px`
  return (
    <section className="flex flex-col gap-1.5 rounded-md border border-subtle bg-surface px-3 py-2.5 shrink-0">
      <h3 className="caption flex items-center gap-1.5">
        <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-accent" />
        放大设置
      </h3>

      {!modelReady && (
        <div className="flex items-center gap-2 text-sm px-2 py-1.5 rounded-sm bg-warn-soft border border-warn">
          <span className="text-warn font-medium">需要下载模型</span>
          <span className="text-fg-secondary text-xs flex-1 truncate">
            {upscaler?.kind === 'custom'
              ? '自定义模型未在本地（请到设置 → 预处理）'
              : `${upscaler?.hf_repo ?? upscaler?.ms_repo ?? '—'} · ~${upscaler?.size_mb ?? 64} MB`}
          </span>
          <button
            onClick={onDownloadModel}
            disabled={downloadingModel || upscaler?.kind === 'custom'}
            className="btn btn-primary btn-sm"
          >
            {downloadingModel ? '下载中...' : `下载 ${selectedModel}`}
          </button>
        </div>
      )}

      {/* 目标分辨率行：决定每张图最终输出多少像素（按训练桶面积估的） */}
      <div className="flex items-center gap-2 text-sm flex-wrap">
        <label className="flex items-center gap-1.5">
          <span className="text-fg-tertiary">目标分辨率</span>
          <select
            value={selectValue}
            onChange={(e) => handlePresetChange(e.target.value)}
            disabled={busy}
            className="input text-sm"
            style={{ width: 'auto', padding: '2px 6px' }}
          >
            {TARGET_PRESETS.map((p) => (
              <option
                key={p.edge === null ? 'off' : p.edge === 0 ? 'custom' : p.edge}
                value={p.edge === null ? 'off' : p.edge === 0 ? 'custom' : String(p.edge)}
              >{p.label}</option>
            ))}
          </select>
          {targetEdge === 0 && (
            <input
              type="number"
              min={256}
              max={4096}
              step={64}
              value={customEdge}
              onChange={(e) => setCustomEdge(e.target.value)}
              disabled={busy}
              className="input input-mono text-sm"
              style={{ width: 80, padding: '2px 6px' }}
              placeholder="边长"
            />
          )}
          <span className="text-fg-tertiary text-xs">{targetHint}</span>
        </label>
      </div>

      <div className="flex items-center gap-2 text-sm flex-wrap">
        <label className="flex items-center gap-1.5">
          <span className="text-fg-tertiary">模型</span>
          <select
            value={selectedModel}
            onChange={(e) => onSelectedModelChange(e.target.value)}
            disabled={busy}
            className="input text-sm mono"
            style={{ width: 'auto', padding: '2px 6px' }}
          >
            {allUpscalers.map((v) => (
              <option key={v.label} value={v.label}>
                {v.label}
                {!v.exists ? ' · 未下载' : ''}
                {v.kind === 'custom' ? ' · 自定义' : ''}
              </option>
            ))}
            {allUpscalers.length === 0 && (
              <option value={selectedModel}>{selectedModel}</option>
            )}
          </select>
        </label>

        <span className="text-dim">·</span>

        <label className="flex items-center gap-1.5">
          <span className="text-fg-tertiary">tile</span>
          <select
            value={tileSize}
            onChange={(e) => setTileSize(Number(e.target.value))}
            disabled={busy}
            className="input text-sm"
            style={{ width: 'auto', padding: '2px 6px' }}
          >
            {TILE_OPTIONS.map((n) => (
              <option key={n} value={n}>{n}px</option>
            ))}
          </select>
        </label>

        <span className="text-dim">·</span>

        <label className="flex items-center gap-1.5">
          <span className="text-fg-tertiary">设备</span>
          <select
            value={device}
            onChange={(e) => setDevice(e.target.value as Device)}
            disabled={busy}
            className="input text-sm"
            style={{ width: 'auto', padding: '2px 6px' }}
          >
            {DEVICE_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </label>

        <span className="flex-1" />

        <button
          onClick={onStartSelected}
          disabled={busy || !modelReady || pendingSelCount === 0}
          className="btn btn-secondary btn-sm"
          title={pendingSelCount === 0 ? '在「待处理」里选图后启用' : ''}
        >
          {`放大选中 ${pendingSelCount}`}
        </button>
        <button
          onClick={onStartAll}
          disabled={busy || !modelReady || pendingCount === 0}
          className="btn btn-primary btn-sm"
        >
          {pendingCount > 0 ? `放大全部 ${pendingCount}` : '没有待处理'}
        </button>
      </div>

      {/* 智能流水说明 + 未来 tabs 占位 */}
      <div className="flex items-center gap-2 mt-1 text-xs text-fg-tertiary flex-wrap">
        <span className="font-medium text-fg-secondary">阶段</span>
        <span className="px-1.5 py-0.5 rounded bg-accent-soft text-accent text-xs font-medium">放大</span>
        <span
          className="px-1.5 py-0.5 rounded bg-overlay opacity-50 cursor-not-allowed"
          title="未来阶段：交互式裁剪"
        >裁剪</span>
        <span
          className="px-1.5 py-0.5 rounded bg-overlay opacity-50 cursor-not-allowed"
          title="未来阶段：画笔涂抹（取色 + 高斯）"
        >涂抹</span>
        <span className="flex-1" />
        {targetEdge !== null && (
          <span title="像素够目标的图直接 LANCZOS 缩，省掉昂贵的 4× 推理">
            智能：像素够 → 跳过模型直接 LANCZOS 缩
          </span>
        )}
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// 已处理 / 待处理 sections
// ---------------------------------------------------------------------------

function ProcessedSection({
  items,
  selected,
  onSelect,
  onPreview,
  onSelectAll,
  onClear,
  onDelete,
}: {
  items: { name: string; thumbUrl: string; meta?: string }[]
  selected: Set<string>
  onSelect: (name: string, e: React.MouseEvent) => void
  onPreview: (name: string) => void
  onSelectAll: () => void
  onClear: () => void
  onDelete: () => void
}) {
  return (
    <section className="flex flex-col flex-1 min-h-0 rounded-md border border-subtle bg-surface overflow-hidden">
      <header className="flex items-center gap-2 shrink-0 px-2.5 py-1.5 border-b border-subtle text-sm">
        <h3 className="font-semibold">已处理</h3>
        <span className="text-fg-tertiary">{items.length} 张</span>
        {selected.size > 0 && (
          <span className="text-accent">· 已选 {selected.size}</span>
        )}
        <span className="flex-1" />
        <button
          onClick={onSelectAll}
          disabled={items.length === 0}
          className="btn btn-ghost btn-sm"
        >全选</button>
        <button
          onClick={onClear}
          disabled={selected.size === 0}
          className="btn btn-ghost btn-sm"
        >清空</button>
        <button
          onClick={onDelete}
          disabled={selected.size === 0}
          className="btn btn-sm bg-err-soft text-err"
          title="删除选中产物（源会重新出现在「待处理」）"
        >🗑 删除 {selected.size}</button>
      </header>
      <div className="flex-1 min-h-0 overflow-y-auto p-2">
        <ImageGrid
          items={items}
          selected={selected}
          onSelect={onSelect}
          onActivate={onPreview}
          onPreview={onPreview}
          clickMode="activate"
          ariaLabel="preprocess-processed-grid"
          emptyHint="还没有产物 — 点击上方「放大全部 N」"
        />
      </div>
    </section>
  )
}

function PendingSection({
  items,
  selected,
  onSelect,
  onPreview,
  onSelectAll,
  onClear,
}: {
  items: { name: string; thumbUrl: string }[]
  selected: Set<string>
  onSelect: (name: string, e: React.MouseEvent) => void
  onPreview: (name: string) => void
  onSelectAll: () => void
  onClear: () => void
}) {
  return (
    <section className="flex flex-col flex-1 min-h-0 rounded-md border border-subtle bg-surface overflow-hidden">
      <header className="flex items-center gap-2 shrink-0 px-2.5 py-1.5 border-b border-subtle text-sm">
        <h3 className="font-semibold">待处理</h3>
        <span className="text-fg-tertiary">{items.length} 张</span>
        {selected.size > 0 && (
          <span className="text-accent">· 已选 {selected.size}</span>
        )}
        <span className="flex-1" />
        <button
          onClick={onSelectAll}
          disabled={items.length === 0}
          className="btn btn-ghost btn-sm"
        >全选</button>
        <button
          onClick={onClear}
          disabled={selected.size === 0}
          className="btn btn-ghost btn-sm"
        >清空</button>
      </header>
      <div className="flex-1 min-h-0 overflow-y-auto p-2">
        <ImageGrid
          items={items}
          selected={selected}
          onSelect={onSelect}
          onActivate={onPreview}
          onPreview={onPreview}
          clickMode="activate"
          ariaLabel="preprocess-pending-grid"
          emptyHint="所有图都已预处理 ✓"
        />
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// JobStrip
// ---------------------------------------------------------------------------

function JobStrip({
  job,
  logs,
  onCancel,
}: {
  job: Job
  logs: string[]
  onCancel?: () => void
}) {
  const elapsed =
    job.started_at && (job.finished_at ?? Date.now() / 1000) - job.started_at
  const isLive = job.status === 'running' || job.status === 'pending'
  const lastLine = logs[logs.length - 1] ?? ''
  return (
    <details
      open={isLive}
      className="group rounded-md border border-subtle bg-surface overflow-hidden shrink-0"
    >
      <summary className="cursor-pointer flex items-center gap-2 list-none px-2.5 py-1.5 text-sm select-none">
        <span className="inline-block transition-transform group-open:rotate-90 text-fg-tertiary w-3">▸</span>
        <span className={STATUS_COLOR[job.status]}>{job.status}</span>
        <span className="mono text-fg-secondary">job #{job.id}</span>
        {elapsed && elapsed > 0 && (
          <span className="text-fg-tertiary">· {Math.round(elapsed)}s</span>
        )}
        <span className="mono truncate flex-1 min-w-0 text-fg-secondary text-xs">
          {lastLine}
        </span>
        {isLive && onCancel && (
          <button
            onClick={(e) => {
              e.preventDefault()
              onCancel()
            }}
            className="btn btn-ghost btn-sm text-err"
          >取消</button>
        )}
      </summary>
      <pre className="px-3 py-2 text-xs font-mono text-fg-secondary bg-sunken max-h-[224px] overflow-auto whitespace-pre-wrap border-t border-subtle m-0">
        {logs.length === 0 ? '(等待日志...)' : logs.slice(-1000).join('\n')}
      </pre>
    </details>
  )
}

// ---------------------------------------------------------------------------
// 右栏侧边栏
// ---------------------------------------------------------------------------

function PreprocessSidebar({
  summary,
  upscaler,
  selectedModel,
  tileSize,
  processed,
  targetEdge,
}: {
  summary: Status['summary']
  upscaler: UpscalerVariant | null
  selectedModel: string
  tileSize: number
  processed: PreprocessedItem[]
  targetEdge: number | null
}) {
  const { download_count, processed_count, pending_count } = summary
  const pct = download_count > 0 ? Math.round((processed_count / download_count) * 100) : 0
  // 粗略 VRAM 估算：tile²×scale²×2byte (fp16)×7倍中间张量，单位 MB。给个量级。
  const estVramMB = Math.round((tileSize * tileSize * 16 * 2 * 7) / (1024 * 1024))

  // action 分布：让用户看到智能流水的实际收益（有多少图根本没走模型）
  const actionStats = useMemo(() => {
    const stats = { resize: 0, upscale: 0, 'upscale+resize': 0, unknown: 0 }
    for (const it of processed) {
      const a = it.action ?? 'unknown'
      if (a in stats) (stats as Record<string, number>)[a]++
      else stats.unknown++
    }
    return stats
  }, [processed])

  // 产物总盘占 — 云端硬盘费才是真该警惕的
  const processedBytes = useMemo(
    () => processed.reduce((s, it) => s + (it.size ?? 0), 0),
    [processed],
  )
  const avgBytes = processed.length > 0 ? processedBytes / processed.length : 0
  const fmtBytes = (b: number) =>
    b >= 1024 * 1024 * 1024
      ? `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`
      : b >= 1024 * 1024
        ? `${(b / 1024 / 1024).toFixed(1)} MB`
        : `${(b / 1024).toFixed(0)} KB`

  return (
    <div className="flex flex-col gap-3 min-w-0">
      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <h3 className="caption flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-accent" />
          预处理进度
        </h3>
        <StatRow label="源 download/" value={`${download_count} 张`} />
        <StatRow label="已处理" value={`${processed_count} 张`} accent="ok" />
        <StatRow label="待处理" value={`${pending_count} 张`} accent={pending_count > 0 ? 'warn' : undefined} />
        <div className="mt-2 h-1.5 rounded bg-sunken overflow-hidden">
          <div
            className="h-full bg-accent rounded transition-[width] duration-300 ease-out"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="text-xs text-fg-tertiary mt-1 text-right">{pct}%</p>
      </div>

      {/* action 分布：跳过模型的图越多，整体越快 */}
      {processed.length > 0 && (
        <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
          <h3 className="caption flex items-center gap-1.5">
            <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-accent" />
            处理方式分布
          </h3>
          {actionStats.resize > 0 && (
            <StatRow label="直接缩 (大图)" value={`${actionStats.resize} 张`} accent="ok" />
          )}
          {actionStats['upscale+resize'] > 0 && (
            <StatRow label="放大 + 缩" value={`${actionStats['upscale+resize']} 张`} accent="warn" />
          )}
          {actionStats.upscale > 0 && (
            <StatRow label="纯 4× 放大" value={`${actionStats.upscale} 张`} />
          )}
          {actionStats.unknown > 0 && (
            <StatRow label="旧数据" value={`${actionStats.unknown} 张`} />
          )}
          <p className="text-[11px] text-fg-tertiary mt-1.5 leading-snug">
            「直接缩」走 LANCZOS 不调用模型，秒级；「放大+缩」才是慢的那条路。
            {targetEdge === null && '（关闭目标模式时所有图都走 4× 路径）'}
          </p>
        </div>
      )}

      {/* 盘占用 — 云端关注的真指标 */}
      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <h3 className="caption flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-ok" />
          盘占用
        </h3>
        <StatRow
          label="产物总大小"
          value={processed.length > 0 ? fmtBytes(processedBytes) : '—'}
          accent={processedBytes > 5 * 1024 ** 3 ? 'warn' : undefined}
        />
        {processed.length > 0 && (
          <StatRow label="均值 / 张" value={fmtBytes(avgBytes)} />
        )}
        <p className="text-[11px] text-fg-tertiary mt-1.5 leading-snug">
          {targetEdge === null
            ? '关闭目标模式：每张 4× PNG 巨大，盘费高。'
            : '智能模式：缩到目标面积后单张 PNG 通常 1-3 MB。'}
        </p>
      </div>

      {/* 设备 / 模型就绪 */}
      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <h3 className="caption flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-accent opacity-60" />
          设备 / 模型
        </h3>
        <StatRow
          label={selectedModel}
          value={upscaler?.exists ? '已就绪' : '未下载'}
          accent={upscaler?.exists ? 'ok' : 'warn'}
        />
        <StatRow label="估算 VRAM 峰值" value={`~${estVramMB} MB`} />
        <p className="text-[11px] text-fg-tertiary mt-1.5 leading-snug">
          tile 越大显存占用越高；OOM 时降到 128/192。
        </p>
      </div>

      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <h3 className="caption flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-warn opacity-60" />
          下一步
        </h3>
        <p className="text-xs text-fg-secondary leading-snug">
          预处理产物存在时，<strong className="text-accent">筛选</strong>页左侧自动切到 preprocess/。
          预处理是可选阶段 — 跳过也能直接进筛选。
        </p>
      </div>
    </div>
  )
}

function StatRow({
  label,
  value,
  accent,
}: {
  label: string
  value: string | number
  accent?: 'ok' | 'warn' | 'err'
}) {
  const cls =
    accent === 'ok' ? 'text-ok' :
    accent === 'warn' ? 'text-warn' :
    accent === 'err' ? 'text-err' :
    'text-fg-primary'
  return (
    <div className="flex justify-between items-baseline mt-1.5 text-xs">
      <span className="text-fg-tertiary">{label}</span>
      <span className={`font-mono font-medium ${cls}`}>{value}</span>
    </div>
  )
}
