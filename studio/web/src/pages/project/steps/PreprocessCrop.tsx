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
import Filmstrip from '../../../components/preprocess/Filmstrip'
import FreeCropEditor, { type CropRect } from '../../../components/preprocess/FreeCropEditor'
import PreprocessToolsBar from '../../../components/preprocess/PreprocessToolsBar'
import StepShell from '../../../components/StepShell'
import BarHistogram from '../../../components/BarHistogram'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'
import { arBucket, arLabel } from '../../../lib/aspectRatio'
import { clusterByAspectRatio } from '../../../lib/cropClustering'
import { compareImagePath } from '../../../lib/imageSort'

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
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const vid = activeVersion?.id ?? 0

  // ────── Workspace data ──────
  const [images, setImages] = useState<CropWorkspaceItem[]>([])
  const [loading, setLoading] = useState(true)

  const refreshWorkspace = useCallback(async () => {
    if (!vid) return
    try {
      const r = await api.listCropWorkspaceTrain(project.id, vid)
      setImages([...r.images].sort((a, b) => compareImagePath(a.name, b.name)))
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }, [project.id, vid])

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
  // 自动聚类参数改由 header 按钮弹出的 modal 承载（不再占裁剪参数区）。
  const [clusterModalOpen, setClusterModalOpen] = useState(false)

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

  // 回放（issue #251）：crop 与放大共用 kind=preprocess，走同一 status 端点
  // 恢复最近一次 preprocess job + log_tail；同一 job 本地已有 SSE 积累时不覆盖。
  const refreshJobStatus = useCallback(async () => {
    if (!vid) return
    try {
      const r = await api.getPreprocessStatusTrain(project.id, vid)
      const rid = r.job?.id ?? null
      setJob(r.job)
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

  useEffect(() => { void refreshJobStatus() }, [refreshJobStatus])

  useEventStream((evt) => {
    const jid = jobIdRef.current
    if (evt.type === 'job_log_appended' && jid && evt.job_id === jid) {
      setLogs((prev) => [...prev, String(evt.text ?? '')])
    } else if (evt.type === 'job_state_changed' && jid && evt.job_id === jid) {
      // mirror status change in our job object so the log drawer renders the new badge
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
  }, { onOpen: () => void refreshJobStatus() })

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
    const n = summary.assignments.filter((a) => !a.skipped).length
    toast(t('preprocessCrop.clusterToast', { k: summary.kUsed, n }), 'success')
    setClusterModalOpen(false)
  }, [images, autoParams, toast, t])

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
      const j = await api.startPreprocessCropTrain(project.id, vid, payload)
      setJob(j)
      setLogs([])
      toast(t('preprocessCrop.toastStarted', { id: j.id }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }, [cropsByImage, activeName, project.id, vid, toast, t])

  // ────── Render ──────
  // ADR 0010: hooks 之后再做 vid guard
  if (!activeVersion) {
    return (
      <div className="p-6 text-fg-secondary">
        {t('projectStepper.selectVersion')}
      </div>
    )
  }

  return (
    <StepShell
      idx={2}
      title={t('steps.preprocess.title')}
      subtitle={t('preprocessCrop.subtitle')}
      actions={
        <>
          {/* 次操作 = ghost（对齐正则/打标页 header 范式）；主操作「裁剪当前图」= primary + icon，放最右 */}
          <button
            type="button"
            onClick={() => setClusterModalOpen(true)}
            disabled={busy || images.length === 0}
            className="btn btn-ghost btn-sm"
          >
            {t('preprocessCrop.autoCluster')}
          </button>
          <button
            type="button"
            onClick={() => void submitCrop(false)}
            disabled={busy || totalRects === 0}
            className="btn btn-ghost btn-sm"
          >
            {t('preprocessCrop.cropAll', { n: totalRects })}
          </button>
          <button
            type="button"
            onClick={() => void submitCrop(true)}
            disabled={busy || activeCrops.length === 0}
            className="btn btn-primary btn-sm"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
            <span>{t('preprocessCrop.cropActive')}</span>
          </button>
        </>
      }
      belowHeader={<PreprocessToolsBar current="crop" projectId={project.id} versionId={vid} />}
      logSources={[
        job && {
          key: 'preprocess',
          label: t('logDrawer.preprocess'),
          status: job.status,
          lines: logs,
          startedAt: job.started_at,
          finishedAt: job.finished_at,
          onCancel: () => void cancelJob(),
        },
      ]}
    >
      <div className="flex flex-col h-full gap-3 min-h-0">
        <div className="grid gap-3 flex-1 min-h-0" style={{ gridTemplateColumns: '1fr 260px' }}>
          {/* 左栏 —— 裁剪参数区已删，工作区占满高度；长宽比锁定移到工作区底部固定栏 */}
          <div className="flex flex-col gap-2 min-h-0 min-w-0">
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
                    <Link to={`/projects/${project.id}/v/${vid}/preprocess?tool=upscale`} className="text-accent hover:underline">
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
                      onSelect={(name) => {
                        setActiveName(name)
                        setSelectedRectId(null)
                      }}
                      thumbUrl={(im) => {
                        const i = im.name.lastIndexOf('/')
                        const folder = i >= 0 ? im.name.slice(0, i) : ''
                        const filename = i >= 0 ? im.name.slice(i + 1) : im.name
                        return api.versionThumbUrl(
                          project.id, vid, 'train', filename, folder, 256,
                        ) + `&_=${im.mtime}`
                      }}
                      emptyHint={t(`preprocessCrop.filmstripEmpty.${filter}`)}
                      renderOverlay={(im) => {
                        const crops = cropsByImage[im.name] ?? []
                        if (crops.length === 0) return null
                        return (
                          <>
                            {crops.map((c, i) => (
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
                          </>
                        )
                      }}
                    />

                    <div className="min-w-0 min-h-0 overflow-hidden">
                      <FreeCropEditor
                        image={{
                          id: activeImage.name,
                          name: activeImage.name,
                          w: activeImage.w,
                          h: activeImage.h,
                          thumbUrl: (() => {
                            const i = activeImage.name.lastIndexOf('/')
                            const folder = i >= 0 ? activeImage.name.slice(0, i) : ''
                            const filename = i >= 0 ? activeImage.name.slice(i + 1) : activeImage.name
                            return api.versionThumbUrl(
                              project.id, vid, 'train', filename, folder, 1024,
                            ) + `&_=${activeImage.mtime}`
                          })(),
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
                      arSel={arSel}
                      setArSel={setArSel}
                      customAR={customAR}
                      setCustomAR={setCustomAR}
                      busy={busy}
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

        {clusterModalOpen && (
          <ClusterModal
            autoParams={autoParams}
            setAutoParams={setAutoParams}
            lastClusterK={lastClusterK}
            busy={busy}
            totalImages={images.length}
            onRun={runClustering}
            onClose={() => setClusterModalOpen(false)}
          />
        )}
      </div>
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// 自动聚类 modal —— header 按钮弹出。按长宽比给所有图预填裁剪框（之后可手动微调）。
// 参数用数字 input 框（原来是左右拉 slider）。
// ---------------------------------------------------------------------------

function ClusterModal({
  autoParams, setAutoParams, lastClusterK, busy, totalImages, onRun, onClose,
}: {
  autoParams: AutoParams
  setAutoParams: (v: AutoParams) => void
  lastClusterK: number | null
  busy: boolean
  totalImages: number
  onRun: () => void
  onClose: () => void
}) {
  const { t } = useTranslation()
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="cluster-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-md"
      onClick={onClose}
    >
      <div
        className="w-[90%] max-w-[460px] flex flex-col gap-5 p-6 bg-elevated border border-dim rounded-lg shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="cluster-modal-title" className="m-0 text-lg font-semibold text-fg-primary">
          {t('preprocessCrop.autoCluster')}
        </h2>
        <p className="m-0 text-sm text-fg-secondary leading-relaxed">
          {t('preprocessCrop.clusterModalDesc')}
        </p>
        <div className="grid grid-cols-3 gap-3">
          <ClusterField
            label={t('preprocessCrop.clusterMaxCrop')}
            value={autoParams.maxCropFraction}
            min={0} max={0.3} step={0.01}
            onChange={(v) => setAutoParams({ ...autoParams, maxCropFraction: Math.max(0, Math.min(0.3, v)) })}
          />
          <ClusterField
            label={t('preprocessCrop.clusterKMin')}
            value={autoParams.kMin}
            min={1} max={10} step={1}
            onChange={(v) => setAutoParams({ ...autoParams, kMin: Math.min(v, autoParams.kMax) })}
          />
          <ClusterField
            label={t('preprocessCrop.clusterKMax')}
            value={autoParams.kMax}
            min={2} max={15} step={1}
            onChange={(v) => setAutoParams({ ...autoParams, kMax: Math.max(v, autoParams.kMin) })}
          />
        </div>
        {lastClusterK !== null && (
          <p className="m-0 text-xs text-ok font-mono">
            ✓ {t('preprocessCrop.clusterDone')} {t('preprocessCrop.clusterUsed', { k: lastClusterK })}
          </p>
        )}
        <div className="flex justify-end gap-2 pt-1">
          <button type="button" onClick={onClose} className="btn btn-secondary">
            {t('common.cancel', { defaultValue: '取消' })}
          </button>
          <button
            type="button"
            onClick={onRun}
            disabled={busy || totalImages === 0}
            className="btn btn-primary"
          >
            ▶ {t('preprocessCrop.runCluster')}
          </button>
        </div>
      </div>
    </div>
  )
}

function ClusterField({
  label, value, min, max, step, onChange,
}: {
  label: string
  value: number
  min: number; max: number; step: number
  onChange: (v: number) => void
}) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="text-fg-tertiary">{label}</span>
      <input
        type="number"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="input input-mono text-sm"
        style={{ padding: '4px 8px' }}
      />
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
  arSel,
  setArSel,
  customAR,
  setCustomAR,
  busy,
  onSelect,
  onLabelChange,
  onDelete,
  onDuplicate,
}: {
  activeImage: CropWorkspaceItem
  crops: CropRect[]
  selectedId: string | null
  arLock: { w: number; h: number } | null
  arSel: string
  setArSel: (s: string) => void
  customAR: { w: number; h: number }
  setCustomAR: (v: { w: number; h: number }) => void
  busy: boolean
  onSelect: (id: string) => void
  onLabelChange: (id: string, label: string) => void
  onDelete: (id: string) => void
  onDuplicate: (id: string) => void
}) {
  const { t } = useTranslation()
  return (
    <div className="bg-sunken border border-subtle rounded-md flex flex-col h-full min-h-0 overflow-hidden">
      <div className="flex flex-col gap-2 p-2.5 flex-1 min-h-0 overflow-y-auto">
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

      {/* 长宽比锁定固定栏：贴「本图裁剪」面板底部，画框时锁定裁剪框比例 */}
      <div className="shrink-0 border-t border-subtle px-2.5 py-2 flex flex-col gap-1.5">
        <label className="flex items-center gap-1.5 text-xs">
          <span className="text-fg-tertiary shrink-0">{t('preprocessCrop.aspectRatio')}</span>
          <select
            value={arSel}
            onChange={(e) => setArSel(e.target.value)}
            disabled={busy}
            className="input text-sm flex-1 min-w-0"
            style={{ padding: '2px 6px' }}
          >
            {AR_OPTIONS.map((o) => (
              <option key={o.id} value={o.id}>{o.label}</option>
            ))}
          </select>
        </label>
        {arSel === 'custom' && (
          <label className="flex items-center gap-1.5 text-xs">
            <span className="text-fg-tertiary shrink-0">W : H</span>
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
      </div>
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

  // AR histogram covers the whole dataset, always. Per image: if it has
  // crops, count each crop's AR; otherwise count the source AR. A single
  // crop on the current image must NOT collapse the histogram to a single
  // entry — the user is mid-edit and still needs to see how the rest of
  // their 264 images distribute.
  //
  // `crops` mode kicks in once ALL images have at least one crop (the user
  // has fully configured the dataset and now wants to see post-crop AR).
  const arHist = useMemo(() => {
    const m = new Map<string, { n: number; sortKey: number }>()
    const allCovered = images.length > 0
      && images.every((im) => (cropsByImage[im.name] ?? []).length > 0)
    for (const im of images) {
      const rects = cropsByImage[im.name] ?? []
      if (rects.length > 0) {
        for (const r of rects) {
          const v = (r.w * im.w) / (r.h * im.h)
          const { label, sortKey } = arBucket(v)
          const prev = m.get(label)
          m.set(label, { n: (prev?.n ?? 0) + 1, sortKey })
        }
      } else {
        const v = im.w / im.h
        const { label, sortKey } = arBucket(v)
        const prev = m.get(label)
        m.set(label, { n: (prev?.n ?? 0) + 1, sortKey })
      }
    }
    const bins = Array.from(m.entries())
      .map(([label, { n, sortKey }]) => ({ label, n, sortKey }))
      .sort((a, b) => b.sortKey - a.sortKey) // wide first, tall last
    return { bins, fromSource: !allCovered }
  }, [cropsByImage, images])

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
        <BarHistogram bins={arHist.bins} />
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
