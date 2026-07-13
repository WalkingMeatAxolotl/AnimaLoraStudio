import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useOutletContext } from 'react-router-dom'
import {
  api,
  type CropWorkspaceItem,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import Filmstrip from '../../../components/preprocess/Filmstrip'
import InpaintCanvas, {
  renderInpaintedBlob,
  renderMaskBlob,
  type InpaintCanvasHandle,
  type InpaintMode,
  type InpaintStroke,
} from '../../../components/preprocess/InpaintCanvas'
import PreprocessToolsBar from '../../../components/preprocess/PreprocessToolsBar'
import StepShell from '../../../components/StepShell'
import { useDialog } from '../../../components/Dialog'
import { useToast } from '../../../components/Toast'
import { useLocalStorageState } from '../../../lib/useLocalStorageState'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

type Filter = 'all' | 'pending' | 'edited'

interface BrushState {
  color: string
  size: number
  hardness: number
}

const DEFAULT_BRUSH: BrushState = { color: '#ffffff', size: 24, hardness: 1 }

function splitRel(name: string): { folder: string; filename: string } {
  const i = name.lastIndexOf('/')
  return {
    folder: i >= 0 ? name.slice(0, i) : '',
    filename: i >= 0 ? name.slice(i + 1) : name,
  }
}

/** 状态模型对齐裁剪页：双数据面双桶（strokesByImage / maskStrokesByImage），
 *  随便切图改动都留在内存，保存按当前模式分发（§9 决策 2）。只有活动图挂
 *  真实 canvas，「保存全部」对非活动图走离屏重放。 */
export default function PreprocessInpaintPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const vid = activeVersion?.id ?? 0

  // ────── Workspace data（复用 crop workspace：name + w/h + mtime + mask_mtime）──────
  const [images, setImages] = useState<CropWorkspaceItem[]>([])
  const [loading, setLoading] = useState(true)

  const refreshWorkspace = useCallback(async () => {
    if (!vid) return
    try {
      const r = await api.listCropWorkspaceTrain(project.id, vid)
      setImages(r.images)
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }, [project.id, vid])

  useEffect(() => { void refreshWorkspace() }, [refreshWorkspace])

  // ────── Editor state ──────
  const [mode, setMode] = useState<InpaintMode>('paint')
  const [maskErase, setMaskErase] = useState(false)
  const [activeName, setActiveName] = useState<string | null>(null)
  const [strokesByImage, setStrokesByImage] = useState<Record<string, InpaintStroke[]>>({})
  const [redoByImage, setRedoByImage] = useState<Record<string, InpaintStroke[]>>({})
  const [maskStrokesByImage, setMaskStrokesByImage] = useState<Record<string, InpaintStroke[]>>({})
  const [maskRedoByImage, setMaskRedoByImage] = useState<Record<string, InpaintStroke[]>>({})
  const [maskCoverage, setMaskCoverage] = useState(0)
  const [filter, setFilter] = useState<Filter>('all')
  const [busy, setBusy] = useState(false)

  const [brush, setBrush] = useLocalStorageState<BrushState>(
    'studio:inpaint:brush', DEFAULT_BRUSH,
  )
  const [recentColors, setRecentColors] = useLocalStorageState<string[]>(
    'studio:inpaint:recent_colors', [],
  )

  const canvasRef = useRef<InpaintCanvasHandle | null>(null)

  useEffect(() => {
    if (images.length === 0) return
    if (!activeName || !images.find((im) => im.name === activeName)) {
      setActiveName(images[0].name)
    }
  }, [images, activeName])

  // ────── Derived ──────
  const activeImage = useMemo(
    () => images.find((im) => im.name === activeName) ?? null,
    [images, activeName],
  )
  const isMask = mode === 'mask'
  const activeStrokes = activeName
    ? ((isMask ? maskStrokesByImage : strokesByImage)[activeName] ?? [])
    : []
  const activeRedo = activeName
    ? ((isMask ? maskRedoByImage : redoByImage)[activeName] ?? [])
    : []

  /** 当前模式的 dirty 图集合（保存全部 / filter / header 计数共用）。 */
  const editedNames = useMemo(() => {
    const bucket = isMask ? maskStrokesByImage : strokesByImage
    return Object.entries(bucket)
      .filter(([, s]) => s.length > 0)
      .map(([n]) => n)
  }, [isMask, maskStrokesByImage, strokesByImage])

  const counts = useMemo(() => {
    const bucket = isMask ? maskStrokesByImage : strokesByImage
    const edited = images.filter((im) => (bucket[im.name] ?? []).length > 0).length
    return { all: images.length, pending: images.length - edited, edited }
  }, [images, isMask, maskStrokesByImage, strokesByImage])

  const filteredImages = useMemo(() => {
    const bucket = isMask ? maskStrokesByImage : strokesByImage
    return images.filter((im) => {
      const n = (bucket[im.name] ?? []).length
      if (filter === 'pending') return n === 0
      if (filter === 'edited') return n > 0
      return true
    })
  }, [images, filter, isMask, maskStrokesByImage, strokesByImage])

  const rawUrl = useCallback((im: CropWorkspaceItem) => {
    const { folder, filename } = splitRel(im.name)
    return api.versionThumbUrl(project.id, vid, 'train', filename, folder, 0)
      + `&_=${im.mtime}`
  }, [project.id, vid])

  const maskBaseUrlFor = useCallback((im: CropWorkspaceItem): string | null => {
    if (im.mask_mtime == null) return null
    return api.maskUrl(project.id, vid, im.name) + `&_=${im.mask_mtime}`
  }, [project.id, vid])

  // ────── Stroke mutations（按模式分发到对应桶）──────
  const pushRecentColor = useCallback((hex: string) => {
    setRecentColors((prev) => [hex, ...prev.filter((c) => c !== hex)].slice(0, 8))
  }, [setRecentColors])

  const onStrokeEnd = useCallback((s: InpaintStroke) => {
    if (!activeName) return
    setStrokesByImage((prev) => ({
      ...prev,
      [activeName]: [...(prev[activeName] ?? []), s],
    }))
    setRedoByImage((prev) => ({ ...prev, [activeName]: [] }))
    pushRecentColor(s.color)
  }, [activeName, pushRecentColor])

  const onMaskStrokeEnd = useCallback((s: InpaintStroke) => {
    if (!activeName) return
    setMaskStrokesByImage((prev) => ({
      ...prev,
      [activeName]: [...(prev[activeName] ?? []), s],
    }))
    setMaskRedoByImage((prev) => ({ ...prev, [activeName]: [] }))
  }, [activeName])

  const undo = useCallback(() => {
    if (!activeName) return
    const setBucket = isMask ? setMaskStrokesByImage : setStrokesByImage
    const setRedoBucket = isMask ? setMaskRedoByImage : setRedoByImage
    setBucket((prev) => {
      const cur = prev[activeName] ?? []
      if (cur.length === 0) return prev
      const last = cur[cur.length - 1]
      setRedoBucket((r) => ({
        ...r,
        [activeName]: [...(r[activeName] ?? []), last],
      }))
      return { ...prev, [activeName]: cur.slice(0, -1) }
    })
  }, [activeName, isMask])

  const redo = useCallback(() => {
    if (!activeName) return
    const setBucket = isMask ? setMaskStrokesByImage : setStrokesByImage
    const setRedoBucket = isMask ? setMaskRedoByImage : setRedoByImage
    setRedoBucket((prev) => {
      const cur = prev[activeName] ?? []
      if (cur.length === 0) return prev
      const last = cur[cur.length - 1]
      setBucket((s) => ({
        ...s,
        [activeName]: [...(s[activeName] ?? []), last],
      }))
      return { ...prev, [activeName]: cur.slice(0, -1) }
    })
  }, [activeName, isMask])

  const clearActive = useCallback(() => {
    if (!activeName) return
    if (isMask) {
      setMaskStrokesByImage((prev) => ({ ...prev, [activeName]: [] }))
      setMaskRedoByImage((prev) => ({ ...prev, [activeName]: [] }))
    } else {
      setStrokesByImage((prev) => ({ ...prev, [activeName]: [] }))
      setRedoByImage((prev) => ({ ...prev, [activeName]: [] }))
    }
  }, [activeName, isMask])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== 'z') return
      const el = e.target as HTMLElement | null
      if (
        el &&
        (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)
      ) {
        return
      }
      e.preventDefault()
      if (e.shiftKey) redo()
      else undo()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [undo, redo])

  const onPickColor = useCallback((hex: string) => {
    setBrush((prev) => ({ ...prev, color: hex }))
    pushRecentColor(hex)
  }, [setBrush, pushRecentColor])

  // ────── Save（按模式分发，§9 决策 2）──────
  const clearSavedPaint = useCallback((name: string) => {
    setStrokesByImage((prev) => {
      const next = { ...prev }
      delete next[name]
      return next
    })
    setRedoByImage((prev) => {
      const next = { ...prev }
      delete next[name]
      return next
    })
  }, [])

  const clearSavedMask = useCallback((name: string) => {
    setMaskStrokesByImage((prev) => {
      const next = { ...prev }
      delete next[name]
      return next
    })
    setMaskRedoByImage((prev) => {
      const next = { ...prev }
      delete next[name]
      return next
    })
  }, [])

  const saveActivePaint = useCallback(async () => {
    if (!activeName) return
    const blob = await canvasRef.current?.exportBlob()
    if (!blob) {
      toast(t('preprocessInpaint.toastNotReady'), 'error')
      return
    }
    setBusy(true)
    try {
      const res = await api.saveInpaintTrain(project.id, vid, activeName, blob)
      clearSavedPaint(activeName)
      toast(t('preprocessInpaint.toastSaved', { name: res.name }), 'success')
      await refreshWorkspace()
      if (res.name !== activeName) setActiveName(res.name)
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }, [activeName, project.id, vid, clearSavedPaint, refreshWorkspace, reload, toast, t])

  const saveActiveMask = useCallback(async () => {
    if (!activeName || !activeImage) return
    setBusy(true)
    try {
      const res = await canvasRef.current?.exportMaskBlob()
      if (res === undefined) {
        toast(t('preprocessInpaint.toastNotReady'), 'error')
        return
      }
      if (res === null) {
        // mask 为空 → 有旧文件则删（恢复全学），否则无事可做
        if (activeImage.mask_mtime != null) {
          await api.deleteMaskTrain(project.id, vid, activeName)
          toast(t('preprocessInpaint.toastMaskCleared'), 'success')
        }
      } else {
        await api.saveMaskTrain(project.id, vid, activeName, res.blob)
        toast(t('preprocessInpaint.toastMaskSaved', {
          pct: Math.round(res.coverage * 100),
        }), 'success')
      }
      clearSavedMask(activeName)
      await refreshWorkspace()
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }, [activeName, activeImage, project.id, vid, clearSavedMask, refreshWorkspace, reload, toast, t])

  const saveAll = useCallback(async () => {
    const dirty = editedNames
    if (dirty.length === 0) return
    setBusy(true)
    let ok = 0
    const failed: string[] = []
    try {
      for (const name of dirty) {
        const im = images.find((i) => i.name === name)
        if (!im) continue
        try {
          if (isMask) {
            const strokes = maskStrokesByImage[name] ?? []
            if (strokes.length === 0) continue
            const res = await renderMaskBlob(maskBaseUrlFor(im), im.w, im.h, strokes)
            if (res === null) {
              if (im.mask_mtime != null) {
                await api.deleteMaskTrain(project.id, vid, name)
              }
            } else {
              await api.saveMaskTrain(project.id, vid, name, res.blob)
            }
            clearSavedMask(name)
          } else {
            const strokes = strokesByImage[name] ?? []
            if (strokes.length === 0) continue
            const blob = await renderInpaintedBlob(rawUrl(im), im.w, im.h, strokes)
            await api.saveInpaintTrain(project.id, vid, name, blob)
            clearSavedPaint(name)
          }
          ok++
        } catch {
          failed.push(name)
        }
      }
      toast(
        failed.length > 0
          ? t('preprocessInpaint.toastSavedAllPartial', { ok, failed: failed.length })
          : t('preprocessInpaint.toastSavedAll', { n: ok }),
        failed.length > 0 ? 'error' : 'success',
      )
      await refreshWorkspace()
      void reload()
    } finally {
      setBusy(false)
    }
  }, [
    editedNames, images, isMask, maskStrokesByImage, strokesByImage,
    maskBaseUrlFor, rawUrl, project.id, vid,
    clearSavedMask, clearSavedPaint, refreshWorkspace, reload, toast, t,
  ])

  /** 清除 mask（U5）：即时 DELETE 服务器文件 + 清本地笔画。 */
  const clearMask = useCallback(async () => {
    if (!activeName || !activeImage) return
    const hasServer = activeImage.mask_mtime != null
    const hasLocal = (maskStrokesByImage[activeName] ?? []).length > 0
    if (!hasServer && !hasLocal) return
    if (hasServer && !(await confirm(
      t('preprocessInpaint.confirmClearMask'),
      { tone: 'danger', okText: t('preprocessInpaint.confirmClearMaskOk') },
    ))) return
    setBusy(true)
    try {
      if (hasServer) await api.deleteMaskTrain(project.id, vid, activeName)
      clearSavedMask(activeName)
      await refreshWorkspace()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }, [
    activeName, activeImage, maskStrokesByImage, project.id, vid,
    confirm, clearSavedMask, refreshWorkspace, toast, t,
  ])

  // ────── Render ──────
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
      subtitle={t('preprocessInpaint.subtitle')}
      actions={
        <>
          <button
            type="button"
            onClick={() => void saveAll()}
            disabled={busy || editedNames.length === 0}
            className="btn btn-ghost btn-sm"
          >
            {t(isMask ? 'preprocessInpaint.saveAllMask' : 'preprocessInpaint.saveAll', { n: editedNames.length })}
          </button>
          <button
            type="button"
            onClick={() => void (isMask ? saveActiveMask() : saveActivePaint())}
            disabled={busy || activeStrokes.length === 0}
            className="btn btn-primary btn-sm"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M17 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V7l-4-4zm-5 16a3 3 0 1 1 0-6 3 3 0 0 1 0 6zm3-10H5V5h10v4z" />
            </svg>
            <span>{t(isMask ? 'preprocessInpaint.saveActiveMask' : 'preprocessInpaint.saveActive')}</span>
          </button>
        </>
      }
      belowHeader={<PreprocessToolsBar current="inpaint" projectId={project.id} versionId={vid} />}
    >
      <div className="flex flex-col h-full gap-3 min-h-0">
        <section className="flex flex-col flex-1 min-h-0 rounded-md border border-subtle bg-surface overflow-hidden">
          <header className="flex items-center gap-2 shrink-0 px-2.5 py-1.5 border-b border-subtle text-sm flex-wrap">
            <div className="flex items-center gap-1">
              {(['all', 'pending', 'edited'] as const).map((k) => (
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
                  {t(`preprocessInpaint.filter.${k}`)} {counts[k]}
                </button>
              ))}
            </div>
            {activeImage && (
              <span className="text-fg-tertiary text-xs font-mono ml-2">
                {activeImage.name} · {activeImage.w}×{activeImage.h}
              </span>
            )}
            <span className="flex-1" />
            <button
              onClick={undo}
              disabled={!activeName || activeStrokes.length === 0}
              className="btn btn-ghost btn-sm"
              title="Ctrl+Z"
            >↶ {t('preprocessInpaint.undo')}</button>
            <button
              onClick={redo}
              disabled={!activeName || activeRedo.length === 0}
              className="btn btn-ghost btn-sm"
              title="Ctrl+Shift+Z"
            >↷ {t('preprocessInpaint.redo')}</button>
            <button
              onClick={clearActive}
              disabled={!activeName || activeStrokes.length === 0}
              className="btn btn-ghost btn-sm"
            >{t('preprocessInpaint.clearActive')}</button>
          </header>

          <div className="flex-1 min-h-0 overflow-hidden p-3">
            {loading && (
              <p className="text-fg-tertiary text-sm">{t('preprocessInpaint.loading')}</p>
            )}
            {!loading && images.length === 0 && (
              <p className="text-fg-tertiary text-sm">
                {t('preprocessInpaint.emptyWorkspace')}{' '}
                <Link to={`/projects/${project.id}/v/${vid}/preprocess`} className="text-accent hover:underline">
                  {t('preprocessInpaint.goToOverview')}
                </Link>
              </p>
            )}

            {activeImage && (
              <div
                className="grid gap-3 h-full min-h-0"
                style={{ gridTemplateColumns: '220px minmax(0, 1fr) 260px' }}
              >
                <Filmstrip
                  items={filteredImages}
                  activeName={activeName}
                  onSelect={setActiveName}
                  thumbUrl={(im) => {
                    const { folder, filename } = splitRel(im.name)
                    return api.versionThumbUrl(
                      project.id, vid, 'train', filename, folder, 256,
                    ) + `&_=${im.mtime}`
                  }}
                  emptyHint={t(`preprocessInpaint.filmstripEmpty.${filter}`)}
                  renderOverlay={(im) => {
                    const hasPaint = (strokesByImage[im.name] ?? []).length > 0
                    const hasMask = im.mask_mtime != null
                      || (maskStrokesByImage[im.name] ?? []).length > 0
                    if (!hasPaint && !hasMask) return null
                    return (
                      <span className="fs-badge">
                        {hasPaint ? '✎' : ''}{hasMask ? 'M' : ''}
                      </span>
                    )
                  }}
                />

                <div className="min-w-0 min-h-0 overflow-hidden">
                  <InpaintCanvas
                    key={activeImage.name}
                    ref={canvasRef}
                    imageUrl={rawUrl(activeImage)}
                    imageW={activeImage.w}
                    imageH={activeImage.h}
                    mode={mode}
                    strokes={activeName ? (strokesByImage[activeName] ?? []) : []}
                    maskStrokes={activeName ? (maskStrokesByImage[activeName] ?? []) : []}
                    maskBaseUrl={maskBaseUrlFor(activeImage)}
                    brush={brush}
                    maskErase={maskErase}
                    onStrokeEnd={onStrokeEnd}
                    onMaskStrokeEnd={onMaskStrokeEnd}
                    onPickColor={onPickColor}
                    onMaskCoverage={setMaskCoverage}
                  />
                </div>

                <ToolPanel
                  mode={mode}
                  setMode={setMode}
                  brush={brush}
                  setBrush={setBrush}
                  recentColors={recentColors}
                  maskErase={maskErase}
                  setMaskErase={setMaskErase}
                  maskCoverage={maskCoverage}
                  canClearMask={
                    activeImage.mask_mtime != null
                    || (activeName ? (maskStrokesByImage[activeName] ?? []).length > 0 : false)
                  }
                  onClearMask={() => void clearMask()}
                  busy={busy}
                />
              </div>
            )}
          </div>
        </section>
      </div>
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// Tool panel（right side）
// ---------------------------------------------------------------------------

function ToolPanel({
  mode,
  setMode,
  brush,
  setBrush,
  recentColors,
  maskErase,
  setMaskErase,
  maskCoverage,
  canClearMask,
  onClearMask,
  busy,
}: {
  mode: InpaintMode
  setMode: (m: InpaintMode) => void
  brush: BrushState
  setBrush: (v: BrushState | ((prev: BrushState) => BrushState)) => void
  recentColors: string[]
  maskErase: boolean
  setMaskErase: (v: boolean) => void
  maskCoverage: number
  canClearMask: boolean
  onClearMask: () => void
  busy: boolean
}) {
  const { t } = useTranslation()
  const [recentOpen, setRecentOpen] = useState(false)
  const coveragePct = Math.round(maskCoverage * 100)
  return (
    <div className="bg-sunken border border-subtle rounded-md flex flex-col h-full min-h-0 overflow-hidden">
      <div className="flex flex-col gap-2 p-2.5 flex-1 min-h-0 overflow-y-auto">
        {/* 模式切换：涂抹（改像素）| Mask（训练不学区域） */}
        <div className="flex items-center gap-0 rounded-md border border-subtle overflow-hidden">
          {(['paint', 'mask'] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={
                'flex-1 py-1.5 text-xs font-medium transition-colors ' +
                (mode === m
                  ? 'bg-accent text-white'
                  : 'bg-transparent text-fg-secondary hover:bg-overlay')
              }
            >{t(`preprocessInpaint.mode.${m}`)}</button>
          ))}
        </div>

        {mode === 'paint' ? (
          <div className="flex items-center gap-1.5 text-xs">
            <span className="text-fg-tertiary shrink-0 w-10">{t('preprocessInpaint.brushColor')}</span>
            <input
              type="color"
              value={brush.color}
              onChange={(e) => setBrush((p) => ({ ...p, color: e.target.value }))}
              className="flex-1 min-w-0 h-7 p-0 border border-subtle rounded cursor-pointer bg-transparent"
              title={t('preprocessInpaint.colorWheel')}
            />
            <button
              type="button"
              onClick={() => setRecentOpen((v) => !v)}
              disabled={recentColors.length === 0}
              className={
                'btn btn-ghost btn-sm justify-center shrink-0 ' +
                (recentOpen ? 'bg-overlay text-fg-primary' : '')
              }
              style={{ width: 56 }}
              title={t('preprocessInpaint.recentColors')}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M13 3a9 9 0 0 0-9 9H1l3.89 3.89.07.14L9 12H6a7 7 0 1 1 7 7 6.98 6.98 0 0 1-4.9-2l-1.42 1.42A8.96 8.96 0 0 0 13 21a9 9 0 0 0 0-18zm-1 5v5l4.28 2.54.72-1.21-3.5-2.08V8H12z" />
              </svg>
            </button>
          </div>
        ) : (
          <>
            {/* mask 工具行：画笔 / 橡皮 切换（与颜色行同版式） */}
            <div className="flex items-center gap-1.5 text-xs">
              <span className="text-fg-tertiary shrink-0 w-10">{t('preprocessInpaint.maskTool')}</span>
              <button
                type="button"
                onClick={() => setMaskErase(false)}
                className={
                  'btn btn-sm flex-1 min-w-0 ' + (!maskErase ? 'btn-primary' : 'btn-ghost')
                }
              >{t('preprocessInpaint.maskBrush')}</button>
              <button
                type="button"
                onClick={() => setMaskErase(true)}
                className={
                  'btn btn-sm flex-1 min-w-0 ' + (maskErase ? 'btn-primary' : 'btn-ghost')
                }
              >{t('preprocessInpaint.maskEraser')}</button>
            </div>
            <div className="flex items-center gap-1.5 text-xs">
              <span className="text-fg-tertiary shrink-0 w-10">{t('preprocessInpaint.maskCoverage')}</span>
              <span
                className={
                  'font-mono flex-1 ' +
                  (coveragePct > 50 ? 'text-warn' : 'text-fg-secondary')
                }
              >
                {coveragePct}%{coveragePct > 50 ? ` ${t('preprocessInpaint.maskCoverageWarn')}` : ''}
              </span>
              <button
                type="button"
                onClick={onClearMask}
                disabled={busy || !canClearMask}
                className="btn btn-ghost btn-sm shrink-0"
              >{t('preprocessInpaint.clearMask')}</button>
            </div>
          </>
        )}
        {mode === 'paint' && recentOpen && recentColors.length > 0 && (
          <div className="flex items-center gap-1 flex-wrap">
            {recentColors.map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => {
                  setBrush((p) => ({ ...p, color: c }))
                  setRecentOpen(false)
                }}
                className={
                  'w-5 h-5 rounded border transition-transform hover:scale-110 ' +
                  (c === brush.color ? 'border-accent' : 'border-dim')
                }
                style={{ backgroundColor: c }}
                title={c}
              />
            ))}
          </div>
        )}

        <label className="flex items-center gap-1.5 text-xs">
          <span className="text-fg-tertiary shrink-0 w-10">{t('preprocessInpaint.brushSize')}</span>
          <input
            type="range"
            min={1} max={400} step={1}
            value={brush.size}
            onChange={(e) => setBrush((p) => ({ ...p, size: Number(e.target.value) }))}
            className="flex-1 min-w-0"
          />
          <input
            type="number"
            min={1} max={400}
            value={brush.size}
            onChange={(e) => setBrush((p) => ({
              ...p, size: Math.max(1, Math.min(400, Number(e.target.value) || 1)),
            }))}
            className="input input-mono text-sm shrink-0"
            style={{ width: 56, padding: '2px 6px' }}
          />
        </label>
        <label className="flex items-center gap-1.5 text-xs">
          <span className="text-fg-tertiary shrink-0 w-10">{t('preprocessInpaint.brushHardness')}</span>
          <input
            type="range"
            min={0} max={100} step={5}
            value={Math.round(brush.hardness * 100)}
            onChange={(e) => setBrush((p) => ({ ...p, hardness: Number(e.target.value) / 100 }))}
            className="flex-1 min-w-0"
          />
          <input
            type="number"
            min={0} max={100} step={5}
            value={Math.round(brush.hardness * 100)}
            onChange={(e) => setBrush((p) => ({
              ...p,
              hardness: Math.max(0, Math.min(100, Number(e.target.value) || 0)) / 100,
            }))}
            className="input input-mono text-sm shrink-0"
            style={{ width: 56, padding: '2px 6px' }}
          />
        </label>
      </div>
    </div>
  )
}
