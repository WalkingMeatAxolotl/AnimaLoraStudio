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

/** 单图视图：合并 pending + processed 成一份带状态的列表。
 *
 *  ADR 0004：用户视角只有一份图，「未处理 / 已处理」是图上的徽章而非分组。 */
interface ImageRow {
  name: string                  // download/ 下的原文件名（用作 selection key + thumb URL）
  productName: string           // preprocess/ 下的产物名（{stem}.png），还原 API 用这个
  status: 'pending' | 'processed'
  processed?: PreprocessedItem  // status=processed 时有
  size: number
}

type FilterMode = 'all' | 'pending' | 'processed'

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
  const summary = files?.summary ?? status?.summary ?? {
    download_count: 0,
    processed_count: 0,
    pending_count: 0,
  }
  const modelReady = !!upscaler?.exists

  // ADR 0004：合并 pending + processed 成单一带状态的列表。
  const rows = useMemo<ImageRow[]>(() => {
    if (!files) return []
    const out: ImageRow[] = []
    for (const it of files.pending) {
      out.push({
        name: it.name, productName: `${fileStem(it.name)}.png`,
        status: 'pending', size: it.size,
      })
    }
    for (const p of files.processed) {
      const downloadName = p.source ?? p.name
      out.push({
        name: downloadName, productName: p.name,
        status: 'processed', processed: p, size: p.size,
      })
    }
    out.sort((a, b) => a.name.localeCompare(b.name))
    return out
  }, [files])

  const visibleRows = useMemo(
    () => rows.filter((r) => filter === 'all' || r.status === filter),
    [rows, filter],
  )
  const visibleNames = useMemo(() => visibleRows.map((r) => r.name), [visibleRows])

  const gridItems = useMemo(
    () =>
      visibleRows.map((r) => ({
        name: r.name,
        thumbUrl: api.projectThumbUrl(project.id, r.name),
        meta:
          r.status === 'processed'
            ? `✓ ${r.processed?.action ?? 'upscale'}`
            : t('preprocess.statusPending'),
      })),
    [visibleRows, project.id, t],
  )

  const { selPending, selProcessed } = useMemo(() => {
    const byName = new Map(rows.map((r) => [r.name, r]))
    let p = 0, q = 0
    const pNames: string[] = []
    const qProductNames: string[] = []
    for (const n of sel) {
      const r = byName.get(n)
      if (!r) continue
      if (r.status === 'pending') { p++; pNames.push(r.name) }
      else { q++; qProductNames.push(r.productName) }
    }
    return { selPending: { count: p, names: pNames },
             selProcessed: { count: q, productNames: qProductNames } }
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

  const restoreSelected = async () => {
    if (selProcessed.count === 0) return
    if (!(await confirm(
      t('preprocess.confirmRestore', { n: selProcessed.count }),
      { tone: 'danger', okText: t('preprocess.confirmRestoreOk') },
    ))) return
    try {
      const r = await api.restorePreprocessFiles(project.id, selProcessed.productNames)
      toast(
        t('preprocess.restoredToast', { restored: r.restored.length }) +
          (r.missing.length ? t('preprocess.restoredSkipped', { skipped: r.missing.length }) : ''),
        'success',
      )
      setSel(new Set())
      setSelAnchor(null)
      await refreshFiles()
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

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
              pendingSelCount={selPending.count}
              busy={busy || isLive}
              onStartAll={() => void startPreprocess('all')}
              onStartSelected={() =>
                void startPreprocess('selected', selPending.names)
              }
            />

            {job && (
              <JobStrip
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
              items={gridItems}
              selected={sel}
              selPendingCount={selPending.count}
              selProcessedCount={selProcessed.count}
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
              onRestore={() => void restoreSelected()}
              filterMode={filter}
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

      {previewIdx !== null && visibleRows[previewIdx] && (
        <ImagePreviewModal
          src={api.projectThumbUrl(project.id, visibleRows[previewIdx].name, 'download', 1600)}
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
          disabled={busy || !modelReady || pendingSelCount === 0}
          className="btn btn-secondary btn-sm"
          title={pendingSelCount === 0 ? t('preprocess.upscaleSelectedHint') : ''}
        >
          {t('preprocess.upscaleSelected', { n: pendingSelCount })}
        </button>
        <button
          onClick={onStartAll}
          disabled={busy || !modelReady || pendingCount === 0}
          className="btn btn-primary btn-sm"
        >
          {pendingCount > 0 ? t('preprocess.upscaleAll', { n: pendingCount }) : t('preprocess.noPending')}
        </button>
      </div>

      {/* 智能流水说明 + 未来 tabs 占位 */}
      <div className="flex items-center gap-2 mt-1 text-xs text-fg-tertiary flex-wrap">
        <span className="font-medium text-fg-secondary">{t('preprocess.stageLabel')}</span>
        <span className="px-1.5 py-0.5 rounded bg-accent-soft text-accent text-xs font-medium">
          {t('preprocess.stageUpscale')}
        </span>
        <span
          className="px-1.5 py-0.5 rounded bg-overlay opacity-50 cursor-not-allowed"
          title={t('preprocess.stageCropTitle')}
        >{t('preprocess.stageCrop')}</span>
        <span
          className="px-1.5 py-0.5 rounded bg-overlay opacity-50 cursor-not-allowed"
          title={t('preprocess.stageInpaintTitle')}
        >{t('preprocess.stageInpaint')}</span>
        <span className="flex-1" />
        {targetEdge !== null && (
          <span title={t('preprocess.smartHint')}>
            {t('preprocess.smartHint')}
          </span>
        )}
      </div>
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
  items,
  selected,
  selPendingCount,
  selProcessedCount,
  onSelect,
  onPreview,
  onSelectAll,
  onClear,
  onRestore,
  filterMode,
}: {
  summary: Status['summary']
  filter: FilterMode
  setFilter: (f: FilterMode) => void
  items: { name: string; thumbUrl: string; meta?: string }[]
  selected: Set<string>
  selPendingCount: number
  selProcessedCount: number
  onSelect: (name: string, e: React.MouseEvent) => void
  onPreview: (name: string) => void
  onSelectAll: () => void
  onClear: () => void
  onRestore: () => void
  filterMode: FilterMode
}) {
  const { t } = useTranslation()
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

  const emptyHint =
    filterMode === 'pending'
      ? t('preprocess.emptyPending')
      : filterMode === 'processed'
        ? t('preprocess.emptyProcessed')
        : t('preprocess.emptyAll')

  return (
    <section className="flex flex-col flex-1 min-h-0 rounded-md border border-subtle bg-surface overflow-hidden">
      <header className="flex items-center gap-2 shrink-0 px-2.5 py-1.5 border-b border-subtle text-sm flex-wrap">
        <h3 className="font-semibold">{t('preprocess.imagesTitle')}</h3>
        <span className="text-fg-tertiary">{t('preprocess.totalCount', { n: summary.download_count })}</span>
        {selected.size > 0 && (
          <span className="text-accent">{t('preprocess.selectedCount', { n: selected.size })}</span>
        )}
        <span className="mx-1 text-dim">·</span>
        <div className="flex items-center gap-1">
          {chip('all',       t('preprocess.filterAll'),       summary.download_count)}
          {chip('pending',   t('preprocess.filterPending'),   summary.pending_count)}
          {chip('processed', t('preprocess.filterProcessed'), summary.processed_count)}
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
        <button
          onClick={onRestore}
          disabled={selProcessedCount === 0}
          className="btn btn-sm bg-err-soft text-err"
          title={selProcessedCount === 0
            ? t('preprocess.restoreHintDisabled')
            : t('preprocess.restoreHint', { n: selProcessedCount })}
        >{t('preprocess.restoreBtn', { n: selProcessedCount })}</button>
      </header>
      {selected.size > 0 && selPendingCount > 0 && selProcessedCount > 0 && (
        <div className="px-2.5 py-1 bg-warn-soft text-warn text-xs">
          {t('preprocess.mixedSelectionWarning', { pending: selPendingCount, processed: selProcessedCount })}
        </div>
      )}
      <div className="flex-1 min-h-0 overflow-y-auto p-2">
        <ImageGrid
          items={items}
          selected={selected}
          onSelect={onSelect}
          onActivate={onPreview}
          onPreview={onPreview}
          clickMode="activate"
          ariaLabel="preprocess-grid"
          emptyHint={emptyHint}
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
  const { t } = useTranslation()
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
          >{t('common.cancel')}</button>
        )}
      </summary>
      <pre className="px-3 py-2 text-xs font-mono text-fg-secondary bg-sunken max-h-[224px] overflow-auto whitespace-pre-wrap border-t border-subtle m-0">
        {logs.length === 0 ? t('jobProgress.waitingLogs') : logs.slice(-1000).join('\n')}
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
  const { t } = useTranslation()
  const { download_count, processed_count, pending_count } = summary
  const pct = download_count > 0 ? Math.round((processed_count / download_count) * 100) : 0
  const estVramMB = Math.round((tileSize * tileSize * 16 * 2 * 7) / (1024 * 1024))

  const actionStats = useMemo(() => {
    const stats = { resize: 0, upscale: 0, 'upscale+resize': 0, unknown: 0 }
    for (const it of processed) {
      const a = it.action ?? 'unknown'
      if (a in stats) (stats as Record<string, number>)[a]++
      else stats.unknown++
    }
    return stats
  }, [processed])

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
          {t('preprocess.sidebarProgress')}
        </h3>
        <StatRow label={t('preprocess.sidebarSource')} value={t('preprocess.sidebarNImages', { n: download_count })} />
        <StatRow label={t('preprocess.sidebarProcessed')} value={t('preprocess.sidebarNImages', { n: processed_count })} accent="ok" />
        <StatRow label={t('preprocess.sidebarPending')} value={t('preprocess.sidebarNImages', { n: pending_count })} accent={pending_count > 0 ? 'warn' : undefined} />
        <div className="mt-2 h-1.5 rounded bg-sunken overflow-hidden">
          <div
            className="h-full bg-accent rounded transition-[width] duration-300 ease-out"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="text-xs text-fg-tertiary mt-1 text-right">{pct}%</p>
      </div>

      {processed.length > 0 && (
        <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
          <h3 className="caption flex items-center gap-1.5">
            <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-accent" />
            {t('preprocess.sidebarActionDist')}
          </h3>
          {actionStats.resize > 0 && (
            <StatRow label={t('preprocess.actionResize')} value={t('preprocess.sidebarNImages', { n: actionStats.resize })} accent="ok" />
          )}
          {actionStats['upscale+resize'] > 0 && (
            <StatRow label={t('preprocess.actionUpscaleResize')} value={t('preprocess.sidebarNImages', { n: actionStats['upscale+resize'] })} accent="warn" />
          )}
          {actionStats.upscale > 0 && (
            <StatRow label={t('preprocess.actionUpscale')} value={t('preprocess.sidebarNImages', { n: actionStats.upscale })} />
          )}
          {actionStats.unknown > 0 && (
            <StatRow label={t('preprocess.actionUnknown')} value={t('preprocess.sidebarNImages', { n: actionStats.unknown })} />
          )}
          <p className="text-[11px] text-fg-tertiary mt-1.5 leading-snug">
            {t('preprocess.actionNote')}
            {targetEdge === null && t('preprocess.actionNoteOff')}
          </p>
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
