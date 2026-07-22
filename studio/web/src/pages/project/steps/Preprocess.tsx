import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useOutletContext } from 'react-router-dom'
import {
  api,
  type Job,
  type ProjectDetail,
  type UpscalerVariant,
  type Version,
} from '../../../api/client'
import { parseFolderMeta } from '../../../lib/folderMeta'
import { compareImagePath } from '../../../lib/imageSort'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
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
  /** ADR 0010: train 集合所有图（替代老 pending+processed 二元）。 */
  images: import('../../../api/client').TrainImage[]
  summary: Status['summary']
}

/** 单图视图：从 train manifest 派生的带状态列表（ADR 0010）。
 *
 *  ADR 0010：用户视角只有一份图，「未处理 / 已处理」是图上的徽章而非分组；
 *  状态从 entry 字段差异推断（rel path 末段 ≠ origin → processed；相同 →
 *  pending/原样）。
 */
interface ImageRow {
  /** train rel path "1_data/X.png"（用作 selection key + manifest entry key）。 */
  name: string
  /** name 末段文件名（restore/crop 操作 + thumb URL 用）。 */
  filename: string
  /** name 的 folder 段（thumb URL 用）。 */
  folder: string
  status: 'pending' | 'processed'
  processed?: import('../../../api/client').TrainImage
  size: number
  w: number | null
  h: number | null
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

export default function PreprocessPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const vid = activeVersion?.id ?? 0

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
  const [folderFilter, setFolderFilter] = useState<string>('all')
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
    if (!vid) return
    try {
      const r = await api.listPreprocessFilesTrain(project.id, vid)
      setFiles(r)
    } catch {
      /* ignore */
    }
  }, [project.id, vid])

  const refreshStatus = useCallback(async () => {
    if (!vid) return
    try {
      const r = await api.getPreprocessStatusTrain(project.id, vid)
      setStatus(r)
      // 回放（issue #251）：进页面 / SSE 重连时用 log_tail 恢复日志；
      // 同一 job 且本地已有 SSE 积累时不覆盖（tail 只有 50 行，比本地短）。
      const rid = r.job?.id ?? null
      setLogs((prev) =>
        rid !== null && rid === jobIdRef.current && prev.length > 0
          ? prev
          : r.log_tail
            ? r.log_tail.split('\n')
            : [],
      )
    } catch {
      /* ignore */
    }
  }, [project.id, vid])

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
  }, { onOpen: () => void refreshStatus() })

  const job = status?.job ?? null
  const isLive = job?.status === 'running' || job?.status === 'pending'
  const summary = files?.summary ?? status?.summary ?? { image_count: 0 }
  const modelReady = !!upscaler?.exists

  // ADR 0010: TrainImage[] → ImageRow[]。processed 用 backend `_is_processed`
  // 推断（扩展名变 / _cN 后缀 / train size != download size），前端不自己算。
  const rows = useMemo<ImageRow[]>(() => {
    if (!files) return []
    const out: ImageRow[] = []
    for (const img of files.images) {
      if (img.duplicate_removed) continue // 软删除不进 grid
      const lastSlash = img.name.lastIndexOf('/')
      const folder = lastSlash >= 0 ? img.name.slice(0, lastSlash) : ''
      const filename = lastSlash >= 0 ? img.name.slice(lastSlash + 1) : img.name
      out.push({
        name: img.name,
        filename,
        folder,
        status: img.processed ? 'processed' : 'pending',
        processed: img.processed ? img : undefined,
        size: img.size,
        w: img.w, h: img.h,
        mtime: img.mtime,
      })
    }
    out.sort((a, b) => compareImagePath(a.name, b.name))
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

  // 数据集子文件夹列表（多分辨率：一次放大一个文件夹，目标分辨率可跟随 px 前缀）。
  const folders = useMemo(
    () => Array.from(new Set(rows.map((r) => r.folder).filter(Boolean))).sort(),
    [rows],
  )
  // 选中某文件夹时「全部放大」的范围 = 该文件夹全部图（忽略像素档 filter）；
  // 'all' 时为 null → 走全局 'all' 模式。
  const folderScopedNames = useMemo(
    () =>
      folderFilter === 'all'
        ? null
        : rows.filter((r) => r.folder === folderFilter).map((r) => r.name),
    [rows, folderFilter],
  )

  const visibleRows = useMemo(
    () =>
      rows.filter((r) => {
        if (folderFilter !== 'all' && r.folder !== folderFilter) return false
        if (filter === 'all') return true
        return pxBinFor(r.w, r.h) === filter
      }),
    [rows, filter, folderFilter],
  )

  // 选中带 px 前缀的文件夹 → 目标分辨率自动跟随该文件夹（如 1024px_xxx → 1024）。
  useEffect(() => {
    if (folderFilter === 'all') return
    const reso = parseFolderMeta(folderFilter).reso
    if (reso !== null) {
      setTargetEdge(reso)
      setCustomEdge(String(reso))
    }
  }, [folderFilter])
  // ADR 0010: grid key = rel path (manifest entry key)，跨 sub-folder 唯一。
  const visibleNames = useMemo(
    () => visibleRows.map((r) => r.name),
    [visibleRows],
  )

  const gridItems = useMemo(
    () =>
      visibleRows.map((r) => ({
        name: r.name,
        // train bucket thumb：folder + filename
        thumbUrl: api.versionThumbUrl(
          project.id, vid, 'train', r.filename, r.folder, 256,
        ) + `&_=${r.mtime}`,
        // ADR 0010: processed entry 显示 action 角标（继承自老 schema 透传）
        meta: r.status === 'processed' ? (r.processed?.action ?? undefined) : undefined,
      })),
    [visibleRows, project.id, vid],
  )

  // ADR 0010: 选中传给 start_job_train 的 names 直接用 rel path
  // （resolve_targets_train 接受 rel path，跟 manifest entry key 一致）。
  const selectedTargets = useMemo(() => {
    const names: string[] = []
    for (const k of sel) names.push(k)
    return { count: names.length, names }
  }, [sel])

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
      const j = await api.startPreprocessTrain(project.id, vid, {
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

  // ADR 0010: hooks 之后再做 vid guard（hooks 顺序不能被早 return 打断）
  if (!activeVersion) {
    return (
      <div className="p-6 text-fg-secondary">
        {t('projectStepper.selectVersion')}
      </div>
    )
  }

  // 放大操作数量：有文件夹筛选时按筛选范围算，否则全部
  const upscaleTotal = folderScopedNames ? folderScopedNames.length : rows.length
  const upscaleBusy = busy || isLive

  return (
    <StepShell
      idx={2}
      title={t('steps.preprocess.title')}
      subtitle={t('steps.preprocess.subtitle')}
      actions={
        <>
          {/* 放大全部 = ghost；放大选中 = primary + icon（选中项才启用），放最右 */}
          <button
            type="button"
            onClick={() =>
              void (folderScopedNames
                ? startPreprocess('selected', folderScopedNames)
                : startPreprocess('all'))
            }
            disabled={upscaleBusy || !modelReady || upscaleTotal === 0}
            className="btn btn-ghost btn-sm"
          >
            {t('preprocess.upscaleAll', { n: upscaleTotal })}
          </button>
          <button
            type="button"
            onClick={() => void startPreprocess('selected', selectedTargets.names)}
            disabled={upscaleBusy || !modelReady || selectedTargets.count === 0}
            className="btn btn-primary btn-sm"
            title={selectedTargets.count === 0 ? t('preprocess.upscaleSelectedHint') : ''}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
            <span>{t('preprocess.upscaleSelected', { n: selectedTargets.count })}</span>
          </button>
        </>
      }
      belowHeader={<PreprocessToolsBar current="upscale" projectId={project.id} versionId={vid} />}
      logSources={[
        job && {
          key: 'preprocess',
          label: t('logDrawer.preprocess'),
          status: job.status,
          lines: logs,
          startedAt: job.started_at,
          finishedAt: job.finished_at,
          onCancel: () => void cancel(),
        },
      ]}
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
              busy={busy || isLive}
            />

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
              folders={folders}
              folderFilter={folderFilter}
              setFolderFilter={(f) => {
                setFolderFilter(f)
                setSel(new Set())
                setSelAnchor(null)
                setPreviewIdx(null)
              }}
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
            images={files?.images ?? []}
            targetEdge={targetEdge}
          />
        </div>
      </div>

      {previewIdx !== null && visibleRows[previewIdx] && (
        <ImagePreviewModal
          src={api.versionThumbUrl(
            project.id, vid, 'train',
            visibleRows[previewIdx].filename,
            visibleRows[previewIdx].folder, 1600,
          ) + `&_=${visibleRows[previewIdx].mtime}`}
          caption={`${visibleRows[previewIdx].name} · ${
            visibleRows[previewIdx].status === 'processed' ? '✓ 已处理' : '⊘ 未处理'
          }`}
          index={previewIdx}
          total={visibleRows.length}
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
  busy: boolean
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
  busy,
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
  binCounts,
  folders,
  folderFilter,
  setFolderFilter,
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
  folders: string[]
  folderFilter: string
  setFolderFilter: (f: string) => void
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
        {folders.length > 0 && (
          <>
            <span className="mx-1 text-dim">·</span>
            <label className="flex items-center gap-1 text-xs text-fg-tertiary">
              {t('preprocess.folderFilter')}
              <select
                value={folderFilter}
                onChange={(e) => setFolderFilter(e.target.value)}
                className="input text-xs"
                style={{ width: 'auto', padding: '1px 6px' }}
              >
                <option value="all">{t('preprocess.folderAll')}</option>
                {folders.map((f) => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
            </label>
          </>
        )}
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
  images,
  targetEdge,
}: {
  upscaler: UpscalerVariant | null
  selectedModel: string
  tileSize: number
  images: import('../../../api/client').TrainImage[]
  targetEdge: number | null
}) {
  const { t } = useTranslation()
  const estVramMB = Math.round((tileSize * tileSize * 16 * 2 * 7) / (1024 * 1024))

  // ADR 0010: 直方图基于 train 集全图；状态从 entry 字段差异隐含推断。
  const pixelHist = useMemo(
    () => computePixelHist(images.filter((i) => !i.duplicate_removed)),
    [images],
  )

  // ADR 0010: backend `_is_processed` 推断（扩展名变 / _cN / size diff）
  const processedImages = useMemo(
    () => images.filter((i) => i.processed && !i.duplicate_removed),
    [images],
  )

  const processedBytes = useMemo(
    () => processedImages.reduce((s, it) => s + (it.size ?? 0), 0),
    [processedImages],
  )
  const avgBytes = processedImages.length > 0 ? processedBytes / processedImages.length : 0
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
          value={processedImages.length > 0 ? fmtBytes(processedBytes) : '—'}
          accent={processedBytes > 5 * 1024 ** 3 ? 'warn' : undefined}
        />
        {processedImages.length > 0 && (
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
