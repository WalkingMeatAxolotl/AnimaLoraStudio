import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useOutletContext } from 'react-router-dom'
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
import PreprocessJobStrip from '../../../components/preprocess/PreprocessJobStrip'
import PreprocessToolsBar from '../../../components/preprocess/PreprocessToolsBar'
import StepShell from '../../../components/StepShell'
import BarHistogram from '../../../components/BarHistogram'
import { PX_BINS, pxBinFor, computePixelHist, type PxBinId } from '../../../lib/pixelBins'
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
  summary: { image_count: number }
}

interface FilesView {
  processed: PreprocessedItem[]
  pending: PreprocessPendingItem[]
  summary: Status['summary']
}

/** 单图视图：合并 pending + processed 成一份带状态的列表。
 *
 *  ADR 0004：用户视角只有一份图，「未处理 / 已处理」是图上的徽章而非分组。 */
interface ImageRow {
  name: string                  // download/ 下的原文件名（用作 selection key + thumb URL）
  productName: string           // preprocess/ 下的产物名（{stem}.png），还原 API 用这个
  status: 'pending' | 'processed'
  processed?: PreprocessedItem  // status=processed 时有
  size: number
  /** 实际像素尺寸（pending 取 download/，processed 取 preprocess/）。filter 用。 */
  w: number | null
  h: number | null
  /** 源文件 mtime — 当 thumb cache-buster 用。in-place 覆盖（同名裁剪 / 同名再放大）
   *  后 URL 不带这个 → 浏览器 memory image cache 会继续展示旧 decoded 像素。 */
  mtime: number
}

/** Pixel-area histogram bins — 共享于 sidebar histogram + grid filter chips +
 *  Overview 详情 tab。定义/逻辑移到 lib/pixelBins.ts。 */
type FilterMode = 'all' | PxBinId

const FALLBACK_MODEL = '4x-AnimeSharp'
const TILE_OPTIONS = [128, 192, 256, 384, 512] as const
type Device = 'auto' | 'cuda' | 'cpu'

// 目标分辨率预设 — LoRA 训练桶常用面积。
// value=null 是「关闭智能」模式，直接 4× 模型输出（老路径，盘费高）。
const DEFAULT_TARGET_EDGE = 1024

/** stem 工具（去扩展名）—— 前端 product name 拼装用。 */
function fileStem(name: string): string {
  const dot = name.lastIndexOf('.')
  return dot < 0 ? name : name.slice(0, dot)
}

export default function PreprocessPage() {
  const { t } = useTranslation()
  const { project, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()

  const [files, setFiles] = useState<FilesView | null>(null)
  const [status, setStatus] = useState<Status | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [tileSize, setTileSize] = useState<number>(256)
  const [device, setDevice] = useState<Device>('auto')
  // targetEdge: 边长（像素），平方就是面积；null = 关闭智能；0 = 自定义中
  const [targetEdge, setTargetEdge] = useState<number | null>(DEFAULT_TARGET_EDGE)
  const [customEdge, setCustomEdge] = useState<string>(String(DEFAULT_TARGET_EDGE))
  const [filter, setFilter] = useState<FilterMode>('all')
  const [sel, setSel] = useState<Set<string>>(new Set())
  const [selAnchor, setSelAnchor] = useState<string | null>(null)
  // 大图预览：index 引用 visibleRows[]（filter 当前的可见 ImageRow 列表）
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)

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
      // Logs are ephemeral per-session — don't hydrate from log_tail. After
      // refresh the user only sees a fresh, empty log; SSE appends as the
      // current job emits. Matches the crop tool's behavior so the two pages
      // are consistent.
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
      void refreshUpscaler()
    }
  }, [refreshUpscaler, toast])

  useEffect(() => {
    void refreshFiles()
    void refreshStatus()
    void refreshUpscaler()
  }, [refreshFiles, refreshStatus, refreshUpscaler])

  const jobIdRef = useRef<number | null>(null)
  jobIdRef.current = status?.job?.id ?? null
  useEventStream((evt) => {
    const jid = jobIdRef.current
    if (evt.type === 'job_log_appended' && jid && evt.job_id === jid) {
      setLogs((prev) => [...prev, String(evt.text ?? '')])
    } else if (evt.type === 'preprocess_progress' && jid && evt.job_id === jid) {
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
  const summary = files?.summary ?? status?.summary ?? { image_count: 0 }
  const modelReady = !!upscaler?.exists

  // ADR 0004：合并 pending + processed 成单一带状态的列表。
  const rows = useMemo<ImageRow[]>(() => {
    if (!files) return []
    const out: ImageRow[] = []
    for (const it of files.pending) {
      out.push({
        name: it.name, productName: `${fileStem(it.name)}.png`,
        status: 'pending', size: it.size,
        w: it.w, h: it.h,
        mtime: it.mtime,
      })
    }
    for (const p of files.processed) {
      const downloadName = p.source ?? p.name
      out.push({
        name: downloadName, productName: p.name,
        status: 'processed', processed: p, size: p.size,
        w: p.w, h: p.h,
        mtime: p.mtime,
      })
    }
    out.sort((a, b) => a.name.localeCompare(b.name))
    return out
  }, [files])

  // Per-bin counts — drives both filter chip labels and the "hide empty bins"
  // logic so users only see chips for bins they actually have images in.
  const binCounts = useMemo(() => {
    const m = new Map<PxBinId, number>()
    for (const r of rows) {
      const id = pxBinFor(r.w, r.h)
      if (id) m.set(id, (m.get(id) ?? 0) + 1)
    }
    return m
  }, [rows])

  const visibleRows = useMemo(
    () =>
      rows.filter((r) => {
        if (filter === 'all') return true
        return pxBinFor(r.w, r.h) === filter
      }),
    [rows, filter],
  )
  // visibleNames must match gridItems' unique keys (see gridItems below) so
  // shift-range selection stays consistent — multi-crop fan-out produces
  // processed rows sharing one download `name`, productName disambiguates.
  const visibleNames = useMemo(
    () =>
      visibleRows.map((r) =>
        r.status === 'processed' ? r.productName : r.name,
      ),
    [visibleRows],
  )

  const gridItems = useMemo(
    () =>
      visibleRows.map((r) => ({
        // Grid key — must be unique per row. Multi-crop fan-out produces
        // multiple processed rows that share the same download `name`; use
        // productName (= preprocess filename, including _c0/_c1 suffixes) to
        // disambiguate. Pending rows have unique download names.
        name: r.status === 'processed' ? r.productName : r.name,
        // For processed rows, address by preprocess product name (handles
        // multi-crop fan-out where multiple products share one origin). For
        // pending rows, still address by download name.
        thumbUrl: r.status === 'processed'
          ? api.projectThumbUrl(project.id, r.productName, 'preprocess', 256, r.mtime)
          : api.projectThumbUrl(project.id, r.name, 'download', 256, r.mtime),
        // ADR 0004 Addendum 1 §「Stage 不强制时序」：grid 不分"未处理 / 已处理"。
        // 只在派生路径标 action 给用户看上次处理留下的痕迹（resize / upscale
        // / crop 等），原图路径不挂角标 — 那只是"当前态从 download 读"，不是
        // 一种待办状态。
        meta: r.status === 'processed' ? (r.processed?.action ?? undefined) : undefined,
      })),
    [visibleRows, project.id],
  )

  // grid key（= productName / r.name，跟 gridItems 一致）→ resolve_targets 接收的
  // 后端名（processed 走 productName / pending 走 r.name 都是 ADR-0004 selected
  // mode selectable 集合的元素）。这俩在当前实现下等值，但保留这层映射方便未来
  // grid key 跟后端名分流时只动 byKey。
  const selectedTargets = useMemo(() => {
    const byKey = new Map(
      rows.map((r) => [
        r.status === 'processed' ? r.productName : r.name,
        r,
      ] as const),
    )
    const names: string[] = []
    for (const k of sel) {
      const r = byKey.get(k)
      if (!r) continue
      names.push(r.status === 'processed' ? r.productName : r.name)
    }
    return { count: names.length, names }
  }, [sel, rows])

  // ----- 操作 ---------------------------------------------------------------
  const downloadModel = async () => {
    if (downloadingModel) return
    if (upscaler?.kind === 'custom') {
      toast(t('preprocess.customModelGoSettings'), 'error')
      return
    }
    setDownloadingModel(true)
    try {
      await api.startModelDownload({ model_id: 'upscaler', variant: selectedModel })
      toast(t('preprocess.downloadingModel', { model: selectedModel }), 'success')
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
      toast(t('preprocess.needModelFirst', { model: selectedModel }), 'error')
      return
    }
    let target_area: number | null = null
    if (targetEdge === null) {
      target_area = null
    } else if (targetEdge === 0) {
      const n = Number(customEdge)
      if (!Number.isFinite(n) || n < 256 || n > 4096) {
        toast(t('preprocess.customEdgeRange'), 'error')
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
      toast(t('preprocess.started', { id: j.id }), 'success')
      setSel(new Set())
      setSelAnchor(null)
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
      toast(t('preprocess.canceled'), 'success')
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  // 撤销 (restore) 流程已迁移到「总览」tab (PreprocessOverview)，作为
  // 跨工具统一入口，避免每个工具页都有自己的撤销按钮。

  return (
    <StepShell
      idx={2}
      title={t('steps.preprocess.title')}
      subtitle={t('steps.preprocess.subtitle')}
    >
      <div className="flex flex-col h-full gap-3 min-h-0">
        <div className="grid gap-3 flex-1 min-h-0" style={{ gridTemplateColumns: '1fr 260px' }}>
          {/* 左栏 */}
          <div className="flex flex-col gap-2 min-h-0 min-w-0">
            <PreprocessToolsBar current="upscale" projectId={project.id} />
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
              totalCount={rows.length}
              selectedCount={selectedTargets.count}
              busy={busy || isLive}
              onStartAll={() => void startPreprocess('all')}
              onStartSelected={() =>
                void startPreprocess('selected', selectedTargets.names)
              }
            />

            {/* Show JobStrip only when there's a live job OR session has
                accumulated logs. After a page refresh the historic job may
                still be present in `status.job` but logs are reset (ephemeral
                per session) — rendering an empty JobStrip for a stale
                terminal job is just clutter. */}
            {job && (isLive || logs.length > 0) && (
              <PreprocessJobStrip
                job={job}
                logs={logs}
                onCancel={isLive ? cancel : undefined}
              />
            )}

            <ImagesPanel
              summary={summary}
              filter={filter}
              setFilter={(f) => {
                setFilter(f)
                setSel(new Set())
                setSelAnchor(null)
                setPreviewIdx(null)
              }}
              binCounts={binCounts}
              items={gridItems}
              selected={sel}
              onSelect={(name, e) => {
                const r = applySelection(sel, name, e, visibleNames, selAnchor)
                setSel(r.next)
                setSelAnchor(r.anchor)
              }}
              onPreview={(name) => {
                const i = visibleNames.indexOf(name)
                if (i >= 0) setPreviewIdx(i)
              }}
              onSelectAll={() => setSel(new Set(visibleNames))}
              onClear={() => {
                setSel(new Set())
                setSelAnchor(null)
              }}
            />
          </div>

          {/* 右栏统计 */}
          <PreprocessSidebar
            upscaler={upscaler}
            selectedModel={selectedModel}
            tileSize={tileSize}
            processed={files?.processed ?? []}
            pending={files?.pending ?? []}
            targetEdge={targetEdge}
          />
        </div>
      </div>

      {previewIdx !== null && visibleRows[previewIdx] && (
        <ImagePreviewModal
          src={visibleRows[previewIdx].status === 'processed'
            ? api.projectThumbUrl(project.id, visibleRows[previewIdx].productName, 'preprocess', 1600, visibleRows[previewIdx].mtime)
            : api.projectThumbUrl(project.id, visibleRows[previewIdx].name, 'download', 1600, visibleRows[previewIdx].mtime)}
          caption={`${visibleRows[previewIdx].name} · ${
            visibleRows[previewIdx].status === 'processed' ? '✓ 已处理' : '⊘ 未处理'
          }`}
          hasPrev={previewIdx > 0}
          hasNext={previewIdx < visibleRows.length - 1}
          onClose={() => setPreviewIdx(null)}
          onPrev={() => previewIdx > 0 && setPreviewIdx(previewIdx - 1)}
          onNext={() => previewIdx < visibleRows.length - 1 && setPreviewIdx(previewIdx + 1)}
        />
      )}
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
  totalCount: number
  selectedCount: number
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
  totalCount,
  selectedCount,
  busy,
  onStartAll,
  onStartSelected,
}: OperationPanelProps) {
  const { t } = useTranslation()

  const DEVICE_OPTIONS: { value: Device; label: string }[] = [
    { value: 'auto', label: t('preprocess.deviceAuto') },
    { value: 'cuda', label: 'CUDA' },
    { value: 'cpu', label: 'CPU' },
  ]

  type TargetPreset = { label: string; edge: number | null }
  const TARGET_PRESETS: TargetPreset[] = [
    { label: '768²',  edge: 768 },
    { label: `1024²${t('preprocess.targetRecommended')}`, edge: 1024 },
    { label: '1536²', edge: 1536 },
    { label: '2048²', edge: 2048 },
    { label: t('preprocess.targetCustomLabel'), edge: 0 },
    { label: t('preprocess.targetOffLabel'),    edge: null },
  ]

  const selectValue =
    targetEdge === null ? 'off' : targetEdge === 0 ? 'custom' : String(targetEdge)
  const handlePresetChange = (v: string) => {
    if (v === 'off') setTargetEdge(null)
    else if (v === 'custom') setTargetEdge(0)
    else setTargetEdge(Number(v))
  }

  const targetHint = targetEdge === null
    ? t('preprocess.targetHintOff')
    : targetEdge === 0
      ? t('preprocess.targetHintCustom', { edge: customEdge })
      : t('preprocess.targetHintEdge', { edge: targetEdge, mpx: (targetEdge * targetEdge / 1e6).toFixed(2) })

  return (
    <section className="flex flex-col gap-1.5 rounded-md border border-subtle bg-surface px-3 py-2.5 shrink-0">
      <h3 className="caption flex items-center gap-1.5">
        <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-accent" />
        {t('preprocess.panelTitle')}
      </h3>

      {!modelReady && (
        <div className="flex items-center gap-2 text-sm px-2 py-1.5 rounded-sm bg-warn-soft border border-warn">
          <span className="text-warn font-medium">{t('preprocess.needDownload')}</span>
          <span className="text-fg-secondary text-xs flex-1 truncate">
            {upscaler?.kind === 'custom'
              ? t('preprocess.customModelLocal')
              : `${upscaler?.hf_repo ?? upscaler?.ms_repo ?? '—'} · ~${upscaler?.size_mb ?? 64} MB`}
          </span>
          <button
            onClick={onDownloadModel}
            disabled={downloadingModel || upscaler?.kind === 'custom'}
            className="btn btn-primary btn-sm"
          >
            {downloadingModel ? t('preprocess.modelDownloading') : t('preprocess.downloadModel', { model: selectedModel })}
          </button>
        </div>
      )}

      {/* 目标分辨率行 */}
      <div className="flex items-center gap-2 text-sm flex-wrap">
        <label className="flex items-center gap-1.5">
          <span className="text-fg-tertiary">{t('preprocess.targetRes')}</span>
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
              placeholder={t('preprocess.edgePlaceholder')}
            />
          )}
          <span className="text-fg-tertiary text-xs">{targetHint}</span>
        </label>
      </div>

      <div className="flex items-center gap-2 text-sm flex-wrap">
        <label className="flex items-center gap-1.5">
          <span className="text-fg-tertiary">{t('preprocess.modelLabel')}</span>
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
                {!v.exists ? t('preprocess.notDownloaded') : ''}
                {v.kind === 'custom' ? t('preprocess.customModel') : ''}
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
          <span className="text-fg-tertiary">{t('preprocess.deviceLabel')}</span>
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
          disabled={busy || !modelReady || selectedCount === 0}
          className="btn btn-secondary btn-sm"
          title={selectedCount === 0 ? t('preprocess.upscaleSelectedHint') : ''}
        >
          {t('preprocess.upscaleSelected', { n: selectedCount })}
        </button>
        <button
          onClick={onStartAll}
          disabled={busy || !modelReady || totalCount === 0}
          className="btn btn-primary btn-sm"
        >
          {t('preprocess.upscaleAll', { n: totalCount })}
        </button>
      </div>

      {/* 智能流水提示（tools 切换已移到页面顶部 PreprocessToolsBar） */}
      {targetEdge !== null && (
        <div className="flex items-center gap-2 mt-1 text-xs text-fg-tertiary">
          <span title={t('preprocess.smartHint')}>{t('preprocess.smartHint')}</span>
        </div>
      )}
    </section>
  )
}

// ---------------------------------------------------------------------------
// 单 grid + 状态徽章 + filter chips（ADR 0004）
// ---------------------------------------------------------------------------

function ImagesPanel({
  summary,
  filter,
  setFilter,
  binCounts,
  items,
  selected,
  onSelect,
  onPreview,
  onSelectAll,
  onClear,
}: {
  summary: Status['summary']
  filter: FilterMode
  setFilter: (f: FilterMode) => void
  binCounts: Map<PxBinId, number>
  items: { name: string; thumbUrl: string; meta?: string }[]
  selected: Set<string>
  onSelect: (name: string, e: React.MouseEvent) => void
  onPreview: (name: string) => void
  onSelectAll: () => void
  onClear: () => void
}) {
  const { t } = useTranslation()
  // Pixel-bin chips replace the old 全部 / 未处理 / 已处理 filter.
  // 「未处理 / 已处理」was a workspace concept that doesn't apply to upscale —
  // unlike crop, upscale has no session state to remember; the meaningful
  // axis users want to filter by is resolution (which images need upscaling
  // and which are already large enough). Pixel bins double as a visual link
  // to the sidebar histogram so users see the same buckets on both sides.
  const chip = (key: FilterMode, label: string, count: number) => (
    <button
      onClick={() => setFilter(key)}
      className={
        'px-2 py-0.5 rounded-full text-xs font-medium transition-colors ' +
        (filter === key
          ? 'bg-accent text-white'
          : 'bg-overlay text-fg-secondary hover:bg-accent-soft')
      }
    >
      {label} {count}
    </button>
  )
  // Only show bin chips that have at least one image — keeps the chip row
  // tight on small datasets (a 10-image set isn't going to occupy all 6 bins).
  const nonEmptyBins = PX_BINS.filter((b) => (binCounts.get(b.id) ?? 0) > 0)

  return (
    <section className="flex flex-col flex-1 min-h-0 rounded-md border border-subtle bg-surface overflow-hidden">
      <header className="flex items-center gap-2 shrink-0 px-2.5 py-1.5 border-b border-subtle text-sm flex-wrap">
        <h3 className="font-semibold">{t('preprocess.imagesTitle')}</h3>
        <span className="text-fg-tertiary">{t('preprocess.totalCount', { n: summary.image_count })}</span>
        {selected.size > 0 && (
          <span className="text-accent">{t('preprocess.selectedCount', { n: selected.size })}</span>
        )}
        <span className="mx-1 text-dim">·</span>
        <div className="flex items-center gap-1 flex-wrap">
          {chip('all', t('preprocess.filterAll'), summary.image_count)}
          {nonEmptyBins.map((b) =>
            chip(b.id, b.label, binCounts.get(b.id) ?? 0),
          )}
        </div>
        <span className="flex-1" />
        <button
          onClick={onSelectAll}
          disabled={items.length === 0}
          className="btn btn-ghost btn-sm"
        >{t('common.selectAll')}</button>
        <button
          onClick={onClear}
          disabled={selected.size === 0}
          className="btn btn-ghost btn-sm"
        >{t('common.deselect')}</button>
      </header>
      <div className="flex-1 min-h-0 overflow-y-auto p-2">
        <ImageGrid
          items={items}
          selected={selected}
          onSelect={onSelect}
          onActivate={onPreview}
          onPreview={onPreview}
          clickMode="activate"
          ariaLabel="preprocess-grid"
          emptyHint={t('preprocess.emptyForBin')}
        />
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// 右栏侧边栏
// ---------------------------------------------------------------------------

function PreprocessSidebar({
  upscaler,
  selectedModel,
  tileSize,
  processed,
  pending,
  targetEdge,
}: {
  upscaler: UpscalerVariant | null
  selectedModel: string
  tileSize: number
  processed: PreprocessedItem[]
  pending: PreprocessPendingItem[]
  targetEdge: number | null
}) {
  const { t } = useTranslation()
  const estVramMB = Math.round((tileSize * tileSize * 16 * 2 * 7) / (1024 * 1024))

  // Histogram counts BOTH processed and pending — global view of the
  // dataset's resolution distribution. 按 ADR 0004 这俩只是 grid 来源不同
  // （preprocess/ 派生 vs download 原图），全部图就是 grid 上看到的全部图。
  const pixelHist = useMemo(
    () => computePixelHist([...processed, ...pending]),
    [processed, pending],
  )

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
      {/* 像素分布 — 用户最直接关心的"图够不够大"信息，置顶；按总像素面积
          分桶映射到常见 LoRA 训练分辨率，跟 grid 上的 chip 一一对应。 */}
      {pixelHist.length > 0 && (
        <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
          <h3 className="caption flex items-center gap-1.5">
            <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-accent" />
            {t('preprocess.sidebarPxDist')}
          </h3>
          <div className="mt-1.5">
            <BarHistogram bins={pixelHist.map((b) => ({ key: b.id, label: b.label, n: b.n }))} />
          </div>
        </div>
      )}

      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <h3 className="caption flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-ok" />
          {t('preprocess.sidebarDisk')}
        </h3>
        <StatRow
          label={t('preprocess.diskTotal')}
          value={processed.length > 0 ? fmtBytes(processedBytes) : '—'}
          accent={processedBytes > 5 * 1024 ** 3 ? 'warn' : undefined}
        />
        {processed.length > 0 && (
          <StatRow label={t('preprocess.diskAvg')} value={fmtBytes(avgBytes)} />
        )}
        <p className="text-[11px] text-fg-tertiary mt-1.5 leading-snug">
          {targetEdge === null ? t('preprocess.diskNoteOff') : t('preprocess.diskNoteSmart')}
        </p>
      </div>

      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <h3 className="caption flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-accent opacity-60" />
          {t('preprocess.sidebarDevice')}
        </h3>
        <StatRow
          label={selectedModel}
          value={upscaler?.exists ? t('preprocess.modelReady') : t('preprocess.modelNotDownloaded')}
          accent={upscaler?.exists ? 'ok' : 'warn'}
        />
        <StatRow label={t('preprocess.vramEst')} value={`~${estVramMB} MB`} />
        <p className="text-[11px] text-fg-tertiary mt-1.5 leading-snug">
          {t('preprocess.vramNote')}
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
