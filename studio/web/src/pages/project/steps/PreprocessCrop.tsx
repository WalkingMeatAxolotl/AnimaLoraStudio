import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useOutletContext } from 'react-router-dom'
import {
  api,
  type CropWorkspaceItem,
  type Job,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import FreeCropEditor, { type CropRect } from '../../../components/preprocess/FreeCropEditor'
import PreprocessJobStrip from '../../../components/preprocess/PreprocessJobStrip'
import PreprocessToolsBar from '../../../components/preprocess/PreprocessToolsBar'
import StepShell from '../../../components/StepShell'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'
import { arBucket, arLabel } from '../../../lib/aspectRatio'
import { clusterByAspectRatio } from '../../../lib/cropClustering'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

/** Aspect-ratio choices for the crop canvas. `free` = no lock; `custom` opens W:H fields. */
interface ArOption {
  id: string
  label: string
  w: number | null
  h: number | null
}
const AR_OPTIONS: ArOption[] = [
  { id: 'free',  label: '自由（不锁）', w: null, h: null },
  { id: '1:1',   label: '1:1 正方',    w: 1,  h: 1 },
  { id: '4:3',   label: '4:3 横',      w: 4,  h: 3 },
  { id: '3:2',   label: '3:2 横',      w: 3,  h: 2 },
  { id: '16:9',  label: '16:9 宽屏',   w: 16, h: 9 },
  { id: '3:4',   label: '3:4 竖',      w: 3,  h: 4 },
  { id: '2:3',   label: '2:3 竖',      w: 2,  h: 3 },
  { id: '9:16',  label: '9:16 手机',   w: 9,  h: 16 },
  { id: '4:5',   label: '4:5 竖',      w: 4,  h: 5 },
  { id: 'custom', label: '自定义…',    w: null, h: null },
]

type Filter = 'all' | 'pending' | 'cropped'

interface AutoParams {
  maxCropFraction: number
  kMin: number
  kMax: number
}

/** Reset / clear is "未裁" (pending), having any rect is "已裁" (cropped). Status quo:
 *  the page bypasses upscale-vs-pending distinction — every workspace image is a
 *  candidate; whether it has crops drawn is the only filter dimension. */
function genRectId(): string {
  return 'c' + Math.random().toString(36).slice(2, 9)
}

export default function PreprocessCropPage() {
  const { t } = useTranslation()
  const { project, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()

  // ────── Workspace data ──────
  const [images, setImages] = useState<CropWorkspaceItem[]>([])
  const [loading, setLoading] = useState(true)

  const refreshWorkspace = useCallback(async () => {
    try {
      const r = await api.listCropWorkspace(project.id)
      setImages(r.images)
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }, [project.id])

  useEffect(() => { void refreshWorkspace() }, [refreshWorkspace])

  // ────── Cropping state ──────
  // AR lock + cluster params live side-by-side; clustering is just a feature
  // inside the same crop flow, not a separate "mode" — once cluster runs it
  // pre-fills rects that the user can manually tweak via the same canvas.
  const [arSel, setArSel] = useState<string>('free')
  const [customAR, setCustomAR] = useState<{ w: number; h: number }>({ w: 5, h: 7 })
  const [autoParams, setAutoParams] = useState<AutoParams>({
    maxCropFraction: 0.10, kMin: 3, kMax: 6,
  })
  const [lastClusterK, setLastClusterK] = useState<number | null>(null)

  // ────── Editor state ──────
  const [activeName, setActiveName] = useState<string | null>(null)
  const [cropsByImage, setCropsByImage] = useState<Record<string, CropRect[]>>({})
  const [selectedRectId, setSelectedRectId] = useState<string | null>(null)
  const [filter, setFilter] = useState<Filter>('all')

  // Keep activeName in sync with the available images. Multi-crop fan-out
  // renames a source file (X.png → X_c0.png / X_c1.png), so after a crop job
  // the previous activeName disappears from the workspace; without this
  // fallback the editor would render blank until the user navigates manually.
  useEffect(() => {
    if (images.length === 0) return
    if (!activeName || !images.find((im) => im.name === activeName)) {
      setActiveName(images[0].name)
    }
  }, [images, activeName])

  // ────── Job tracking ──────
  const [job, setJob] = useState<Job | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const jobIdRef = useRef<number | null>(null)
  jobIdRef.current = job?.id ?? null
  const isLive = job?.status === 'running' || job?.status === 'pending'
  useEventStream((evt) => {
    const jid = jobIdRef.current
    if (evt.type === 'job_log_appended' && jid && evt.job_id === jid) {
      setLogs((prev) => [...prev, String(evt.text ?? '')])
    } else if (evt.type === 'job_state_changed' && jid && evt.job_id === jid) {
      // mirror status change in our job object so JobStrip renders the new badge
      setJob((prev) =>
        prev ? { ...prev, status: evt.status as Job['status'] } : prev,
      )
      if (evt.status === 'done' || evt.status === 'failed' || evt.status === 'canceled') {
        void refreshWorkspace()
        void reload()
        // Clear in-memory crops only on success — keep on failure for retry
        if (evt.status === 'done') setCropsByImage({})
      }
    } else if (evt.type === 'crop_progress' && jid && evt.job_id === jid) {
      // Backend throttles crop_progress to ≥1Hz; safe to refresh per event here
      if (evt.status === 'done') void refreshWorkspace()
    }
  })

  const cancelJob = useCallback(async () => {
    if (!job) return
    try {
      await api.cancelJob(job.id)
    } catch (e) {
      toast(String(e), 'error')
    }
  }, [job, toast])

  // ────── Derived ──────
  const arLock = useMemo<{ w: number; h: number } | null>(() => {
    if (arSel === 'free') return null
    if (arSel === 'custom') {
      const w = Math.max(1, customAR.w)
      const h = Math.max(1, customAR.h)
      return { w, h }
    }
    const o = AR_OPTIONS.find((x) => x.id === arSel)
    return o && o.w && o.h ? { w: o.w, h: o.h } : null
  }, [arSel, customAR])

  const totalRects = useMemo(
    () => Object.values(cropsByImage).reduce((s, arr) => s + arr.length, 0),
    [cropsByImage],
  )
  const configuredImages = useMemo(
    () => Object.entries(cropsByImage).filter(([, arr]) => arr.length > 0).length,
    [cropsByImage],
  )

  const activeImage = useMemo(
    () => images.find((im) => im.name === activeName) ?? null,
    [images, activeName],
  )
  const activeCrops = activeName ? (cropsByImage[activeName] ?? []) : []

  const counts = useMemo(() => {
    let pending = 0, cropped = 0
    for (const im of images) {
      const n = (cropsByImage[im.name] ?? []).length
      if (n === 0) pending++; else cropped++
    }
    return { all: images.length, pending, cropped }
  }, [images, cropsByImage])

  const filteredImages = useMemo(() => {
    return images.filter((im) => {
      const n = (cropsByImage[im.name] ?? []).length
      if (filter === 'pending') return n === 0
      if (filter === 'cropped') return n > 0
      return true
    })
  }, [images, filter, cropsByImage])

  // ────── Mutations ──────
  const updateRect = useCallback((id: string, newRect: CropRect) => {
    if (!activeName) return
    setCropsByImage((prev) => ({
      ...prev,
      [activeName]: (prev[activeName] ?? []).map((c) => c.id === id ? newRect : c),
    }))
  }, [activeName])

  const createRect = useCallback((r: Omit<CropRect, 'id' | 'label'>) => {
    if (!activeName) return
    const newId = genRectId()
    setCropsByImage((prev) => {
      const existing = prev[activeName] ?? []
      const newRect: CropRect = {
        id: newId,
        x: r.x, y: r.y, w: r.w, h: r.h,
        label: `${t('preprocessCrop.rectDefaultLabel')} ${existing.length + 1}`,
      }
      return { ...prev, [activeName]: [...existing, newRect] }
    })
    setSelectedRectId(newId)
  }, [activeName, t])

  const deleteRect = useCallback((id: string) => {
    if (!activeName) return
    setCropsByImage((prev) => ({
      ...prev,
      [activeName]: (prev[activeName] ?? []).filter((c) => c.id !== id),
    }))
    setSelectedRectId(null)
  }, [activeName])

  const duplicateRect = useCallback((id: string) => {
    if (!activeName) return
    const newId = genRectId()
    setCropsByImage((prev) => {
      const existing = prev[activeName] ?? []
      const src = existing.find((c) => c.id === id)
      if (!src) return prev
      return {
        ...prev,
        [activeName]: [
          ...existing,
          {
            ...src,
            id: newId,
            x: Math.min(1 - src.w, src.x + 0.04),
            y: Math.min(1 - src.h, src.y + 0.04),
            label: src.label + ' 副本',
          },
        ],
      }
    })
    setSelectedRectId(newId)
  }, [activeName])

  const clearActive = useCallback(() => {
    if (!activeName) return
    setCropsByImage((prev) => ({ ...prev, [activeName]: [] }))
    setSelectedRectId(null)
  }, [activeName])

  // ────── Clustering (optional prefill action) ──────
  const runClustering = useCallback(() => {
    if (images.length === 0) return
    const summary = clusterByAspectRatio(
      images.map((im) => ({ id: im.name, w: im.w, h: im.h })),
      {
        maxCropFraction: autoParams.maxCropFraction,
        kMin: autoParams.kMin,
        kMax: autoParams.kMax,
      },
    )
    const newCrops: Record<string, CropRect[]> = {}
    for (const a of summary.assignments) {
      if (a.skipped) continue
      newCrops[a.id] = [{
        id: 'cl_' + a.id,
        x: a.rect.x, y: a.rect.y, w: a.rect.w, h: a.rect.h,
        label: `聚类 ${a.targetAr.w}:${a.targetAr.h}`,
        fromCluster: true,
      }]
    }
    setCropsByImage(newCrops)
    setSelectedRectId(null)
    setLastClusterK(summary.kUsed)
  }, [images, autoParams])

  // ────── Submit crop job ──────
  const submitCrop = useCallback(async (onlySelected = false) => {
    const payload: Record<string, { x: number; y: number; w: number; h: number; label?: string }[]> = {}
    const entries = Object.entries(cropsByImage)
    for (const [name, rects] of entries) {
      if (rects.length === 0) continue
      if (onlySelected && name !== activeName) continue
      payload[name] = rects.map((r) => ({
        x: r.x, y: r.y, w: r.w, h: r.h,
        label: r.label || undefined,
      }))
    }
    if (Object.keys(payload).length === 0) {
      toast(t('preprocessCrop.toastNoCrops'), 'error')
      return
    }
    setBusy(true)
    try {
      const j = await api.startPreprocessCrop(project.id, payload)
      setJob(j)
      setLogs([])
      toast(t('preprocessCrop.toastStarted', { id: j.id }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }, [cropsByImage, activeName, project.id, toast, t])

  // ────── Render ──────
  return (
    <StepShell
      idx={2}
      title={t('steps.preprocess.title')}
      subtitle={t('preprocessCrop.subtitle')}
    >
      <div className="flex flex-col h-full gap-3 min-h-0">
        <div className="grid gap-3 flex-1 min-h-0" style={{ gridTemplateColumns: '1fr 260px' }}>
          {/* 左栏 */}
          <div className="flex flex-col gap-2 min-h-0 min-w-0">
            <PreprocessToolsBar current="crop" projectId={project.id} />
            <OperationPanel
              arSel={arSel} setArSel={setArSel}
              customAR={customAR} setCustomAR={setCustomAR}
              autoParams={autoParams} setAutoParams={setAutoParams}
              lastClusterK={lastClusterK}
              totalRects={totalRects}
              configuredImages={configuredImages}
              totalImages={images.length}
              activeHasCrops={(cropsByImage[activeName ?? ''] ?? []).length > 0}
              busy={busy}
              onApplyAll={() => void submitCrop(false)}
              onApplySelected={() => void submitCrop(true)}
              onRunCluster={runClustering}
            />
            {/* Job log below OperationPanel — same place as upscale page so the
                two tools have identical UX. Hide entirely when there's no
                live job AND no current-session logs (stale terminal job from
                refresh would otherwise show empty). */}
            {job && (isLive || logs.length > 0) && (
              <PreprocessJobStrip
                job={job}
                logs={logs}
                onCancel={isLive ? () => void cancelJob() : undefined}
              />
            )}

            <section className="flex flex-col flex-1 min-h-0 rounded-md border border-subtle bg-surface overflow-hidden">
              <header className="flex items-center gap-2 shrink-0 px-2.5 py-1.5 border-b border-subtle text-sm flex-wrap">
                <div className="flex items-center gap-1">
                  {(['all', 'pending', 'cropped'] as const).map((k) => (
                    <button
                      key={k}
                      onClick={() => setFilter(k)}
                      className={
                        'px-2 py-0.5 rounded-full text-xs font-medium transition-colors ' +
                        (filter === k
                          ? 'bg-accent text-white'
                          : 'bg-overlay text-fg-secondary hover:bg-accent-soft')
                      }
                    >
                      {t(`preprocessCrop.filter.${k}`)} {counts[k]}
                    </button>
                  ))}
                </div>
                {activeImage && (
                  <span className="text-fg-tertiary text-xs font-mono ml-2">
                    {activeImage.name} · {activeImage.w}×{activeImage.h} · {arLabel(activeImage.w, activeImage.h)}
                  </span>
                )}
                <span className="flex-1" />
                <button
                  onClick={clearActive}
                  disabled={!activeName || (cropsByImage[activeName] ?? []).length === 0}
                  className="btn btn-ghost btn-sm"
                >{t('preprocessCrop.clearActive')}</button>
              </header>

              <div className="flex-1 min-h-0 overflow-hidden p-3">
                {loading && (
                  <p className="text-fg-tertiary text-sm">{t('preprocessCrop.loading')}</p>
                )}
                {!loading && images.length === 0 && (
                  <p className="text-fg-tertiary text-sm">
                    {t('preprocessCrop.emptyWorkspace')}{' '}
                    <Link to={`/projects/${project.id}/preprocess`} className="text-accent hover:underline">
                      {t('preprocessCrop.goToUpscale')}
                    </Link>
                  </p>
                )}

                {activeImage && (
                  /* 3-column layout — filmstrip (left) / canvas (center) / rect list (right).
                     With 264+ image datasets, a bottom horizontal filmstrip gets squeezed to
                     a hairline. Vertical 3-col grid scrolls cleanly and gives the canvas
                     the full WorkArea height to render in. */
                  <div
                    className="grid gap-3 h-full min-h-0"
                    style={{ gridTemplateColumns: '220px minmax(0, 1fr) 260px' }}
                  >
                    {/* Always render the filmstrip column — when the active filter
                        produces 0 matches (e.g. 「已裁剪 0」), conditionally hiding
                        the whole component would collapse the 3-col grid and push
                        canvas + rect list one column left. The empty state lives
                        inside Filmstrip itself so layout stays put. */}
                    <Filmstrip
                      items={filteredImages}
                      activeName={activeName}
                      cropsByImage={cropsByImage}
                      onSelect={(name) => {
                        setActiveName(name)
                        setSelectedRectId(null)
                      }}
                      thumbUrl={(im) => api.projectThumbUrl(
                        project.id,
                        im.name,
                        im.processed ? 'preprocess' : 'download',
                        256,
                      )}
                      emptyHint={t(`preprocessCrop.filmstripEmpty.${filter}`)}
                    />

                    <div className="min-w-0 min-h-0 overflow-hidden">
                      <FreeCropEditor
                        image={{
                          id: activeImage.name,
                          name: activeImage.name,
                          w: activeImage.w,
                          h: activeImage.h,
                          thumbUrl: api.projectThumbUrl(
                            project.id,
                            activeImage.name,
                            activeImage.processed ? 'preprocess' : 'download',
                            1024,
                          ),
                        }}
                        crops={activeCrops}
                        selectedId={selectedRectId}
                        arLock={arLock}
                        onSelect={setSelectedRectId}
                        onChange={updateRect}
                        onCreate={createRect}
                      />
                    </div>

                    <RectListPanel
                      activeImage={activeImage}
                      crops={activeCrops}
                      selectedId={selectedRectId}
                      arLock={arLock}
                      onSelect={setSelectedRectId}
                      onLabelChange={(id, label) => {
                        const r = activeCrops.find((c) => c.id === id)
                        if (r) updateRect(id, { ...r, label })
                      }}
                      onDelete={deleteRect}
                      onDuplicate={duplicateRect}
                    />
                  </div>
                )}
              </div>
            </section>
          </div>

          {/* 右栏统计 */}
          <RightRail
            totalRects={totalRects}
            configuredImages={configuredImages}
            totalImages={images.length}
            lastClusterK={lastClusterK}
            cropsByImage={cropsByImage}
            images={images}
          />
        </div>
      </div>
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// OperationPanel
// ---------------------------------------------------------------------------

interface OperationPanelProps {
  arSel: string
  setArSel: (s: string) => void
  customAR: { w: number; h: number }
  setCustomAR: (v: { w: number; h: number }) => void
  autoParams: AutoParams
  setAutoParams: (v: AutoParams) => void
  lastClusterK: number | null
  totalRects: number
  configuredImages: number
  totalImages: number
  activeHasCrops: boolean
  busy: boolean
  onApplyAll: () => void
  onApplySelected: () => void
  onRunCluster: () => void
}

function OperationPanel({
  arSel, setArSel, customAR, setCustomAR,
  autoParams, setAutoParams,
  lastClusterK,
  totalRects, configuredImages, totalImages, activeHasCrops,
  busy,
  onApplyAll, onApplySelected, onRunCluster,
}: OperationPanelProps) {
  const { t } = useTranslation()
  // Cluster section is collapsed by default — it's an optional helper, not
  // something most users need every visit. Saves vertical space for the canvas.
  const [clusterOpen, setClusterOpen] = useState(false)
  return (
    <section className="flex flex-col gap-1.5 rounded-md border border-subtle bg-surface px-3 py-2.5 shrink-0">
      <h3 className="caption flex items-center gap-1.5">
        <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-accent" />
        {t('preprocessCrop.panelTitle')}
      </h3>

      {/* Row 1 — AR config + primary actions. AR options + actions always
          visible; cropping is one feature, no manual-vs-cluster mode split. */}
      <div className="flex items-center gap-2 text-sm flex-wrap">
        <label className="flex items-center gap-1.5">
          <span className="text-fg-tertiary">{t('preprocessCrop.aspectRatio')}</span>
          <select
            value={arSel}
            onChange={(e) => setArSel(e.target.value)}
            disabled={busy}
            className="input text-sm"
            style={{ width: 'auto', padding: '2px 6px' }}
          >
            {AR_OPTIONS.map((o) => (
              <option key={o.id} value={o.id}>{o.label}</option>
            ))}
          </select>
        </label>
        {arSel === 'custom' && (
          <label className="flex items-center gap-1.5">
            <span className="text-fg-tertiary">W : H</span>
            <input
              type="number" min={1} max={64}
              value={customAR.w}
              onChange={(e) => setCustomAR({ ...customAR, w: Number(e.target.value) || 1 })}
              className="input input-mono text-sm"
              style={{ width: 56, padding: '2px 6px' }}
            />
            <span className="text-fg-tertiary">:</span>
            <input
              type="number" min={1} max={64}
              value={customAR.h}
              onChange={(e) => setCustomAR({ ...customAR, h: Number(e.target.value) || 1 })}
              className="input input-mono text-sm"
              style={{ width: 56, padding: '2px 6px' }}
            />
          </label>
        )}
        <span className="text-dim">·</span>
        <span className="text-fg-secondary text-xs">
          {arSel === 'free'
            ? t('preprocessCrop.hintFree')
            : t('preprocessCrop.hintLocked', { ar: arSel === 'custom' ? `${customAR.w}:${customAR.h}` : arSel })}
        </span>
        <span className="flex-1" />
        <span className="font-mono text-xs text-fg-tertiary mr-1">
          {t('preprocessCrop.summary', { rects: totalRects, configured: configuredImages, total: totalImages })}
        </span>
        <button
          onClick={onApplySelected}
          disabled={busy || !activeHasCrops}
          className="btn btn-secondary btn-sm"
        >{t('preprocessCrop.cropActive')}</button>
        <button
          onClick={onApplyAll}
          disabled={busy || totalRects === 0}
          className="btn btn-primary btn-sm"
        >▶ {t('preprocessCrop.cropAll', { n: totalRects })}</button>
      </div>

      {/* Row 2 — 智能聚类 as an optional helper. Collapsed by default to keep
          OperationPanel compact (most sessions don't use it). Expand with the
          ▸ toggle to reveal sliders + Run button. */}
      <div className="flex flex-col gap-1.5 rounded-sm bg-sunken/40 border border-subtle px-2.5 py-1.5">
        <button
          type="button"
          onClick={() => setClusterOpen((v) => !v)}
          className="flex items-baseline gap-2 text-xs w-full text-left bg-transparent border-0 p-0 cursor-pointer"
        >
          <span className="text-fg-tertiary w-3 inline-block">{clusterOpen ? '▾' : '▸'}</span>
          <span className="text-accent">✦</span>
          <span className="font-medium text-fg-secondary">
            {t('preprocessCrop.clusterSectionTitle')}
          </span>
          <span className="text-fg-tertiary">
            {t('preprocessCrop.clusterSectionDesc')}
          </span>
          <span className="flex-1" />
          {lastClusterK !== null && (
            <span className="text-xs">
              <span className="inline-block px-1.5 py-0.5 rounded-full bg-ok-soft text-ok font-mono">
                ✓ {t('preprocessCrop.clusterDone')}
              </span>
              <span className="text-fg-tertiary ml-2">
                {t('preprocessCrop.clusterUsed', { k: lastClusterK })}
              </span>
            </span>
          )}
        </button>
        {clusterOpen && (
          <div className="flex items-center gap-3 text-sm flex-wrap">
            <ClusterSlider
              label="max_crop"
              min={0} max={0.3} step={0.01}
              value={autoParams.maxCropFraction}
              onChange={(v) => setAutoParams({ ...autoParams, maxCropFraction: v })}
              display={autoParams.maxCropFraction.toFixed(2)}
            />
            <ClusterSlider
              label="k_min"
              min={1} max={10} step={1}
              value={autoParams.kMin}
              onChange={(v) => setAutoParams({ ...autoParams, kMin: Math.min(v, autoParams.kMax) })}
              display={String(autoParams.kMin)}
            />
            <ClusterSlider
              label="k_max"
              min={2} max={15} step={1}
              value={autoParams.kMax}
              onChange={(v) => setAutoParams({ ...autoParams, kMax: Math.max(v, autoParams.kMin) })}
              display={String(autoParams.kMax)}
            />
            <button
              onClick={onRunCluster}
              disabled={busy || totalImages === 0}
              className="btn btn-secondary btn-sm"
            >▶ {t('preprocessCrop.runCluster')}</button>
          </div>
        )}
      </div>

    </section>
  )
}

function ClusterSlider({
  label, min, max, step, value, onChange, display,
}: {
  label: string
  min: number; max: number; step: number; value: number
  onChange: (v: number) => void
  display: string
}) {
  return (
    <label className="flex items-center gap-1.5 min-w-[180px] flex-1">
      <span className="text-fg-tertiary text-xs">{label}</span>
      <input
        type="range" min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 cursor-pointer accent-accent"
        style={{ height: 4, minWidth: 100 }}
      />
      <span className="font-mono text-xs text-fg-tertiary">{display}</span>
    </label>
  )
}

// ---------------------------------------------------------------------------
// Rect list panel (right side of editor)
// ---------------------------------------------------------------------------

function RectListPanel({
  activeImage,
  crops,
  selectedId,
  arLock,
  onSelect,
  onLabelChange,
  onDelete,
  onDuplicate,
}: {
  activeImage: CropWorkspaceItem
  crops: CropRect[]
  selectedId: string | null
  arLock: { w: number; h: number } | null
  onSelect: (id: string) => void
  onLabelChange: (id: string, label: string) => void
  onDelete: (id: string) => void
  onDuplicate: (id: string) => void
}) {
  const { t } = useTranslation()
  return (
    <div className="bg-sunken border border-subtle rounded-md p-2.5 flex flex-col gap-2 h-full min-h-0 overflow-y-auto">
      <header className="flex items-center gap-2 flex-wrap">
        <h3 className="caption">{t('preprocessCrop.rectListTitle')} · {crops.length}</h3>
        <span className="text-fg-tertiary text-[11px]">
          {arLock ? t('preprocessCrop.arLockedTo', { ar: `${arLock.w}:${arLock.h}` }) : t('preprocessCrop.arUnlocked')}
        </span>
        {/* Selected-rect actions — show only when a rect is selected; act as a
            second affordance for the per-row ⎘/✕ buttons so the user has a
            top-of-panel control even when scrolled in a long crop list. */}
        {selectedId && (
          <>
            <span className="flex-1" />
            <button
              className="bg-transparent border-none text-fg-tertiary cursor-pointer px-1.5 py-0.5 text-xs hover:bg-overlay hover:text-fg-primary rounded"
              onClick={() => onDuplicate(selectedId)}
              title={t('preprocessCrop.duplicate')}
            >⎘</button>
            <button
              className="bg-transparent border-none text-fg-tertiary cursor-pointer px-1.5 py-0.5 text-xs hover:bg-err-soft hover:text-err rounded"
              onClick={() => onDelete(selectedId)}
              title={t('preprocessCrop.delete')}
            >✕</button>
          </>
        )}
      </header>
      {crops.length === 0 && (
        <div className="flex flex-col items-center py-6 px-3 gap-1.5 text-center">
          <div className="text-fg-disabled text-2xl">⬚</div>
          <p className="text-fg-secondary text-xs">{t('preprocessCrop.emptyHintLine1')}</p>
          <p className="text-fg-tertiary text-[11px]">{t('preprocessCrop.emptyHintLine2')}</p>
        </div>
      )}
      {crops.map((c, i) => {
        const outW = Math.round(c.w * activeImage.w)
        const outH = Math.round(c.h * activeImage.h)
        const isSel = c.id === selectedId
        return (
          <div
            key={c.id}
            className={
              'grid items-center gap-2 p-1.5 rounded border cursor-pointer transition-colors ' +
              (isSel
                ? 'border-accent bg-accent-soft/40'
                : 'border-subtle bg-surface hover:border-dim')
            }
            style={{ gridTemplateColumns: '50px 1fr auto' }}
            onClick={() => onSelect(c.id)}
          >
            <div
              className="border border-dashed border-dim rounded bg-sunken flex items-center justify-center text-fg-tertiary text-[10px]"
              style={{ aspectRatio: `${outW}/${outH}`, minHeight: 24 }}
            >
              <span className="font-mono">{arLabel(outW, outH)}</span>
            </div>
            <div className="min-w-0 flex flex-col gap-0.5">
              <div className="flex items-center gap-1">
                <span className="text-[10px] text-fg-tertiary font-mono">#{i + 1}</span>
                <input
                  value={c.label}
                  onChange={(e) => onLabelChange(c.id, e.target.value)}
                  onClick={(e) => e.stopPropagation()}
                  className="bg-transparent border-none text-fg-primary text-[12.5px] outline-none w-full min-w-0"
                />
              </div>
              <div className="text-[11px] text-fg-tertiary font-mono">{outW}×{outH} px</div>
            </div>
            <div className="flex gap-0.5">
              <button
                onClick={(e) => { e.stopPropagation(); onDuplicate(c.id) }}
                className="bg-transparent border-none text-fg-tertiary cursor-pointer px-1.5 py-0.5 text-xs hover:bg-overlay hover:text-fg-primary rounded"
                title={t('preprocessCrop.duplicate')}
              >⎘</button>
              <button
                onClick={(e) => { e.stopPropagation(); onDelete(c.id) }}
                className="bg-transparent border-none text-fg-tertiary cursor-pointer px-1.5 py-0.5 text-xs hover:bg-err-soft hover:text-err rounded"
                title={t('preprocessCrop.delete')}
              >✕</button>
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Filmstrip
// ---------------------------------------------------------------------------

function Filmstrip({
  items,
  activeName,
  cropsByImage,
  onSelect,
  thumbUrl,
  emptyHint,
}: {
  items: CropWorkspaceItem[]
  activeName: string | null
  cropsByImage: Record<string, CropRect[]>
  onSelect: (name: string) => void
  thumbUrl: (im: CropWorkspaceItem) => string
  emptyHint?: string
}) {
  /* 3-col vertical grid with square cover thumbs. Squaring is intentional —
     a 264-image dataset spans both portrait and landscape source ARs and
     mixing them in a row leaves ragged gaps. Cover-crop keeps the grid tidy
     and recognisable enough as a navigator (full AR is visible in the
     canvas anyway). The existing crop rects still overlay in normalized
     percent coords, so the user can spot which images already have crops. */
  if (items.length === 0) {
    // Render the same container so the parent grid keeps 3 columns; empty
    // state lives inside.
    return (
      <div className="flex items-center justify-center bg-sunken/40 border border-subtle rounded p-3 h-full text-center text-fg-tertiary text-[11px] leading-snug">
        {emptyHint ?? ''}
      </div>
    )
  }
  return (
    <div className="grid grid-cols-3 gap-1 overflow-y-auto pr-1 bg-sunken/40 border border-subtle rounded p-1.5 h-full content-start">
      {items.map((im) => {
        const crops = cropsByImage[im.name] ?? []
        const isActive = im.name === activeName
        return (
          <div key={im.name} className="fs-thumb-sq-cell">
            <button
              onClick={() => onSelect(im.name)}
              className={'fs-thumb-sq ' + (isActive ? 'is-active' : '')}
              style={{ backgroundImage: `url(${thumbUrl(im)})` }}
              title={im.name}
            >
              {crops.length > 0 && crops.map((c, i) => (
                <div
                  key={c.id}
                  className={'fs-overlay ' + (crops.length > 1 ? 'is-multi' : '')}
                  style={{
                    left: `${c.x * 100}%`,
                    top: `${c.y * 100}%`,
                    width: `${c.w * 100}%`,
                    height: `${c.h * 100}%`,
                  }}
                  aria-label={`crop ${i + 1}`}
                />
              ))}
              {crops.length > 1 && <span className="fs-badge">×{crops.length}</span>}
            </button>
          </div>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Right rail
// ---------------------------------------------------------------------------

function RightRail({
  totalRects,
  configuredImages,
  totalImages,
  lastClusterK,
  cropsByImage,
  images,
}: {
  totalRects: number
  configuredImages: number
  totalImages: number
  lastClusterK: number | null
  cropsByImage: Record<string, CropRect[]>
  images: CropWorkspaceItem[]
}) {
  const { t } = useTranslation()
  const pct = totalImages > 0 ? Math.round((configuredImages / totalImages) * 100) : 0

  // AR histogram: if any crops exist, show crop AR; otherwise source AR.
  // Bins snap to common LoRA AR (1:1 / 2:3 / 3:2 / 16:9 / etc.) within ±5%;
  // others fall into `其他 X.XX`. Sorted wide → tall to match visual intuition.
  const arHist = useMemo(() => {
    const m = new Map<string, { n: number; sortKey: number }>()
    const cropARs: number[] = []
    for (const [name, rects] of Object.entries(cropsByImage)) {
      const im = images.find((x) => x.name === name)
      if (!im) continue
      for (const r of rects) {
        cropARs.push((r.w * im.w) / (r.h * im.h))
      }
    }
    const fromSource = cropARs.length === 0
    const values = fromSource ? images.map((im) => im.w / im.h) : cropARs
    for (const v of values) {
      const { label, sortKey } = arBucket(v)
      const prev = m.get(label)
      m.set(label, { n: (prev?.n ?? 0) + 1, sortKey })
    }
    const bins = Array.from(m.entries())
      .map(([label, { n, sortKey }]) => ({ label, n, sortKey }))
      .sort((a, b) => b.sortKey - a.sortKey) // wide first, tall last
    return { bins, fromSource }
  }, [cropsByImage, images])
  const maxBin = Math.max(1, ...arHist.bins.map((b) => b.n))

  return (
    <div className="flex flex-col gap-3 min-w-0">
      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <h3 className="caption flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-accent" />
          {t('preprocessCrop.rrProgress')}
        </h3>
        <StatRow label={t('preprocessCrop.rrWorkspace')} value={`${totalImages} 张`} />
        <StatRow label={t('preprocessCrop.rrConfigured')} value={`${configuredImages} 张`} accent={configuredImages > 0 ? 'ok' : undefined} />
        <StatRow label={t('preprocessCrop.rrPending')} value={`${totalImages - configuredImages} 张`} accent={totalImages - configuredImages > 0 ? 'warn' : undefined} />
        <div className="mt-2 h-1.5 rounded bg-sunken overflow-hidden">
          <div className="h-full bg-accent rounded transition-[width] duration-300 ease-out" style={{ width: `${pct}%` }} />
        </div>
        <p className="text-xs text-fg-tertiary mt-1 text-right">{pct}%</p>
      </div>

      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <h3 className="caption flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-ok" />
          {t('preprocessCrop.rrOutputs')}
        </h3>
        <StatRow label={t('preprocessCrop.rrOutputFiles')} value={`${totalRects} 张`} />
        <StatRow label={t('preprocessCrop.rrConfiguredImages')} value={`${configuredImages} / ${totalImages}`} />
        {lastClusterK !== null && (
          <StatRow label={t('preprocessCrop.rrSource')} value={`聚类 k=${lastClusterK}`} accent="ok" />
        )}
        <p className="text-[11px] text-fg-tertiary mt-1.5 leading-snug">
          {t('preprocessCrop.rrNote')}
        </p>
      </div>

      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <h3 className="caption flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-accent opacity-60" />
          {t('preprocessCrop.rrArDist')}
        </h3>
        <div className="text-[10px] text-fg-tertiary mt-1 mb-1 font-mono">
          {arHist.fromSource ? `· ${t('preprocessCrop.rrFromSource')}` : `· ${t('preprocessCrop.rrFromCrops')}`}
        </div>
        <div className="flex flex-col gap-1">
          {arHist.bins.map((b) => (
            <div key={b.label} className="grid items-center gap-1.5 text-[11px]" style={{ gridTemplateColumns: '96px 1fr 30px' }}>
              <span className="text-fg-tertiary font-mono">{b.label}</span>
              <div className="ar-bar"><div className="ar-bar-fill" style={{ width: `${(b.n / maxBin) * 100}%` }} /></div>
              <span className="font-mono text-right text-fg-secondary">{b.n}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function StatRow({ label, value, accent }: { label: string; value: string | number; accent?: 'ok' | 'warn' | 'err' }) {
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
