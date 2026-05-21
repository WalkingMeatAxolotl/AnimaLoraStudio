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

/** Image-mode aspect-ratio choices. `free` = no lock; `custom` opens W:H fields. */
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

type Mode = 'manual' | 'auto'
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

  // ────── Mode + per-mode state ──────
  const [mode, setMode] = useState<Mode>('manual')
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

  // Initialize active image after first load
  useEffect(() => {
    if (!activeName && images.length > 0) setActiveName(images[0].name)
  }, [images, activeName])

  // ────── Job tracking ──────
  const [job, setJob] = useState<Job | null>(null)
  const [busy, setBusy] = useState(false)
  const jobIdRef = useRef<number | null>(null)
  jobIdRef.current = job?.id ?? null
  useEventStream((evt) => {
    const jid = jobIdRef.current
    if (evt.type === 'job_state_changed' && jid && evt.job_id === jid) {
      // refresh workspace when job done
      if (evt.status === 'done' || evt.status === 'failed' || evt.status === 'canceled') {
        void refreshWorkspace()
        void reload()
        // Clear the in-memory crops only on success (kept on failure so user can retry)
        if (evt.status === 'done') setCropsByImage({})
      }
    } else if (evt.type === 'crop_progress' && jid && evt.job_id === jid) {
      // optionally throttle workspace refresh; for now refresh per image done
      if (evt.status === 'done') void refreshWorkspace()
    }
  })

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

  // ────── Clustering (auto mode) ──────
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
            <OperationPanel
              mode={mode}
              setMode={setMode}
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
              projectId={project.id}
            />

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

              <div className="flex-1 min-h-0 overflow-auto p-3 flex flex-col gap-3">
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
                  <div className="grid gap-4" style={{ gridTemplateColumns: 'minmax(0, 1fr) 260px' }}>
                    <div className="flex justify-center items-start min-w-0 pt-2">
                      <FreeCropEditor
                        image={{
                          id: activeImage.name,
                          name: activeImage.name,
                          w: activeImage.w,
                          h: activeImage.h,
                          thumbUrl: api.projectThumbUrl(project.id, activeImage.source, 'download', 1024),
                        }}
                        crops={activeCrops}
                        selectedId={selectedRectId}
                        arLock={arLock}
                        onSelect={setSelectedRectId}
                        onChange={updateRect}
                        onCreate={createRect}
                        onDelete={deleteRect}
                        onDuplicate={duplicateRect}
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

                {/* Filmstrip */}
                {filteredImages.length > 0 && (
                  <Filmstrip
                    items={filteredImages}
                    activeName={activeName}
                    cropsByImage={cropsByImage}
                    onSelect={(name) => {
                      setActiveName(name)
                      setSelectedRectId(null)
                    }}
                    thumbUrl={(im) => api.projectThumbUrl(project.id, im.source, 'download', 256)}
                  />
                )}
              </div>
            </section>
          </div>

          {/* 右栏统计 */}
          <RightRail
            mode={mode}
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
  mode: Mode
  setMode: (m: Mode) => void
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
  projectId: number
}

function OperationPanel({
  mode, setMode,
  arSel, setArSel, customAR, setCustomAR,
  autoParams, setAutoParams,
  lastClusterK,
  totalRects, configuredImages, totalImages, activeHasCrops,
  busy,
  onApplyAll, onApplySelected, onRunCluster,
  projectId,
}: OperationPanelProps) {
  const { t } = useTranslation()
  return (
    <section className="flex flex-col gap-1.5 rounded-md border border-subtle bg-surface px-3 py-2.5 shrink-0">
      <h3 className="caption flex items-center gap-1.5">
        <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-accent" />
        {t('preprocessCrop.panelTitle')}
      </h3>

      {/* Row 1: mode segmented + main action */}
      <div className="flex items-center gap-2 text-sm flex-wrap">
        <div className="inline-flex gap-1 p-0.5 bg-sunken border border-subtle rounded-md">
          {(['manual', 'auto'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={
                'px-3 py-1 rounded font-medium text-xs flex flex-col items-start min-w-[140px] transition-colors ' +
                (mode === m
                  ? 'bg-surface text-accent shadow-sm'
                  : 'text-fg-secondary hover:text-fg-primary')
              }
            >
              <span className="text-sm">{t(`preprocessCrop.mode.${m}`)}</span>
              <span className="text-[10.5px] text-fg-tertiary font-normal">
                {t(`preprocessCrop.modeSub.${m}`)}
              </span>
            </button>
          ))}
        </div>

        <span className="flex-1" />

        {/* Action buttons stay identical across modes — both are disk writes,
            hierarchy stays put. Mode-specific tools (开始聚类) live in Row 2
            next to the inputs they operate on. */}
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

      {/* Row 2: mode-specific config */}
      {mode === 'manual' && (
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
          <span className="font-mono text-xs text-fg-tertiary">
            {t('preprocessCrop.summaryManual', { rects: totalRects, configured: configuredImages, total: totalImages })}
          </span>
        </div>
      )}

      {mode === 'auto' && (
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
          {/* 开始聚类 sits with the sliders it computes from — it's a preview /
              prefill action, not a disk write. Disk writes are the Row 1 buttons. */}
          <button
            onClick={onRunCluster}
            disabled={busy || totalImages === 0}
            className="btn btn-secondary btn-sm"
          >▶ {t('preprocessCrop.runCluster')}</button>
          <span className="flex-1" />
          {lastClusterK !== null ? (
            <span className="text-xs text-fg-secondary">
              <span className="inline-block px-1.5 py-0.5 rounded-full bg-ok-soft text-ok font-mono mr-2">
                ✓ {t('preprocessCrop.clusterDone')}
              </span>
              <span className="text-fg-tertiary">
                {t('preprocessCrop.clusterUsed', { k: lastClusterK })}
              </span>
            </span>
          ) : (
            <span className="text-xs text-fg-tertiary">{t('preprocessCrop.clusterHint')}</span>
          )}
        </div>
      )}

      {/* Stage pills — 裁剪 is current; 放大 links back. */}
      <div className="flex items-center gap-2 mt-1 pt-2 border-t border-dashed border-subtle text-xs">
        <span className="font-mono text-fg-tertiary text-[10px] uppercase tracking-wider">
          {t('preprocess.stageLabel')}
        </span>
        <Link
          to={`/projects/${projectId}/preprocess`}
          className="px-1.5 py-0.5 rounded bg-ok-soft text-ok hover:bg-ok-soft/80 font-mono"
        >
          {t('preprocess.stageUpscale')} ✓
        </Link>
        <span className="px-1.5 py-0.5 rounded bg-accent-soft text-accent font-mono font-semibold">
          {t('preprocess.stageCrop')}
        </span>
        <span className="px-1.5 py-0.5 rounded bg-overlay opacity-50 cursor-not-allowed font-mono"
          title={t('preprocess.stageInpaintTitle')}>
          {t('preprocess.stageInpaint')}
        </span>
        <span className="flex-1" />
        <span className="text-fg-tertiary">{t('preprocessCrop.stageNote')}</span>
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
    <div className="bg-sunken border border-subtle rounded-md p-2.5 flex flex-col gap-2 max-h-[640px] overflow-y-auto">
      <header className="flex items-center justify-between">
        <h3 className="caption">{t('preprocessCrop.rectListTitle')} · {crops.length}</h3>
        <span className="text-fg-tertiary text-[11px]">
          {arLock ? t('preprocessCrop.arLockedTo', { ar: `${arLock.w}:${arLock.h}` }) : t('preprocessCrop.arUnlocked')}
        </span>
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
}: {
  items: CropWorkspaceItem[]
  activeName: string | null
  cropsByImage: Record<string, CropRect[]>
  onSelect: (name: string) => void
  thumbUrl: (im: CropWorkspaceItem) => string
}) {
  return (
    <div className="flex gap-1.5 overflow-x-auto px-0.5 py-1 border-t border-dashed border-subtle pt-3 mt-1">
      {items.map((im) => {
        const crops = cropsByImage[im.name] ?? []
        const isActive = im.name === activeName
        return (
          <button
            key={im.name}
            onClick={() => onSelect(im.name)}
            className={'fs-thumb ' + (isActive ? 'is-active' : '')}
            style={{
              aspectRatio: `${im.w}/${im.h}`,
              backgroundImage: `url(${thumbUrl(im)})`,
            }}
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
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Right rail
// ---------------------------------------------------------------------------

function RightRail({
  mode,
  totalRects,
  configuredImages,
  totalImages,
  lastClusterK,
  cropsByImage,
  images,
}: {
  mode: Mode
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
        {mode === 'auto' && lastClusterK !== null && (
          <StatRow label={t('preprocessCrop.rrSource')} value={`聚类 k=${lastClusterK}`} accent="ok" />
        )}
        <p className="text-[11px] text-fg-tertiary mt-1.5 leading-snug">
          {mode === 'manual'
            ? t('preprocessCrop.rrNoteManual')
            : t('preprocessCrop.rrNoteAuto')}
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
