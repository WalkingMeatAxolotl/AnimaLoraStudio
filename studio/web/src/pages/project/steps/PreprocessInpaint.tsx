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
  type InpaintCanvasHandle,
  type InpaintStroke,
} from '../../../components/preprocess/InpaintCanvas'
import PreprocessToolsBar from '../../../components/preprocess/PreprocessToolsBar'
import StepShell from '../../../components/StepShell'
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

/** 状态模型对齐裁剪页：strokesByImage 存笔画矢量，随便切图改动都留在内存，
 *  统一提交。只有活动图挂真实 canvas（InpaintCanvas），「保存全部」对
 *  非活动图走 renderInpaintedBlob 离屏重放。 */
export default function PreprocessInpaintPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const vid = activeVersion?.id ?? 0

  // ────── Workspace data（复用 crop workspace：name + w/h + mtime）──────
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
  const [activeName, setActiveName] = useState<string | null>(null)
  const [strokesByImage, setStrokesByImage] = useState<Record<string, InpaintStroke[]>>({})
  const [redoByImage, setRedoByImage] = useState<Record<string, InpaintStroke[]>>({})
  const [filter, setFilter] = useState<Filter>('all')
  const [busy, setBusy] = useState(false)

  const [brush, setBrush] = useLocalStorageState<BrushState>(
    'studio:inpaint:brush', DEFAULT_BRUSH,
  )
  const [recentColors, setRecentColors] = useLocalStorageState<string[]>(
    'studio:inpaint:recent_colors', [],
  )

  const canvasRef = useRef<InpaintCanvasHandle | null>(null)

  // 保存产物可能改名（X.jpg → X.png），沿用裁剪页的 activeName 兜底
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
  const activeStrokes = activeName ? (strokesByImage[activeName] ?? []) : []
  const activeRedo = activeName ? (redoByImage[activeName] ?? []) : []

  const editedNames = useMemo(
    () => Object.entries(strokesByImage)
      .filter(([, s]) => s.length > 0)
      .map(([n]) => n),
    [strokesByImage],
  )

  const counts = useMemo(() => ({
    all: images.length,
    pending: images.filter((im) => (strokesByImage[im.name] ?? []).length === 0).length,
    edited: images.filter((im) => (strokesByImage[im.name] ?? []).length > 0).length,
  }), [images, strokesByImage])

  const filteredImages = useMemo(() => images.filter((im) => {
    const n = (strokesByImage[im.name] ?? []).length
    if (filter === 'pending') return n === 0
    if (filter === 'edited') return n > 0
    return true
  }), [images, filter, strokesByImage])

  const rawUrl = useCallback((im: CropWorkspaceItem) => {
    const { folder, filename } = splitRel(im.name)
    // size=0 → 原图直出（涂抹必须在原图分辨率上编辑，不能用 1024 缩略图）
    return api.versionThumbUrl(project.id, vid, 'train', filename, folder, 0)
      + `&_=${im.mtime}`
  }, [project.id, vid])

  // ────── Stroke mutations ──────
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

  const undo = useCallback(() => {
    if (!activeName) return
    setStrokesByImage((prev) => {
      const cur = prev[activeName] ?? []
      if (cur.length === 0) return prev
      const last = cur[cur.length - 1]
      setRedoByImage((r) => ({
        ...r,
        [activeName]: [...(r[activeName] ?? []), last],
      }))
      return { ...prev, [activeName]: cur.slice(0, -1) }
    })
  }, [activeName])

  const redo = useCallback(() => {
    if (!activeName) return
    setRedoByImage((prev) => {
      const cur = prev[activeName] ?? []
      if (cur.length === 0) return prev
      const last = cur[cur.length - 1]
      setStrokesByImage((s) => ({
        ...s,
        [activeName]: [...(s[activeName] ?? []), last],
      }))
      return { ...prev, [activeName]: cur.slice(0, -1) }
    })
  }, [activeName])

  const clearActive = useCallback(() => {
    if (!activeName) return
    setStrokesByImage((prev) => ({ ...prev, [activeName]: [] }))
    setRedoByImage((prev) => ({ ...prev, [activeName]: [] }))
  }, [activeName])

  // Ctrl+Z / Ctrl+Shift+Z（表单聚焦时不劫持）
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

  // ────── Save / restore ──────
  const clearSaved = useCallback((name: string) => {
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

  const saveActive = useCallback(async () => {
    if (!activeName || !activeImage) return
    const blob = await canvasRef.current?.exportBlob()
    if (!blob) {
      toast(t('preprocessInpaint.toastNotReady'), 'error')
      return
    }
    setBusy(true)
    try {
      const res = await api.saveInpaintTrain(project.id, vid, activeName, blob)
      clearSaved(activeName)
      toast(t('preprocessInpaint.toastSaved', { name: res.name }), 'success')
      await refreshWorkspace()
      if (res.name !== activeName) setActiveName(res.name)
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }, [activeName, activeImage, project.id, vid, clearSaved, refreshWorkspace, reload, toast, t])

  const saveAll = useCallback(async () => {
    const dirty = editedNames
    if (dirty.length === 0) return
    setBusy(true)
    let ok = 0
    const failed: string[] = []
    try {
      for (const name of dirty) {
        const im = images.find((i) => i.name === name)
        const strokes = strokesByImage[name] ?? []
        if (!im || strokes.length === 0) continue
        try {
          // 统一走离屏重放（活动图也一样），不依赖显示 canvas 的挂载状态
          const blob = await renderInpaintedBlob(rawUrl(im), im.w, im.h, strokes)
          await api.saveInpaintTrain(project.id, vid, name, blob)
          clearSaved(name)
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
  }, [editedNames, images, strokesByImage, rawUrl, project.id, vid, clearSaved, refreshWorkspace, reload, toast, t])

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
          {/* 次操作 = ghost / 主操作 = primary + icon（对齐裁剪页 header 范式）。
              还原不放这页 —— 总览页是还原统一入口。 */}
          <button
            type="button"
            onClick={() => void saveAll()}
            disabled={busy || editedNames.length === 0}
            className="btn btn-ghost btn-sm"
          >
            {t('preprocessInpaint.saveAll', { n: editedNames.length })}
          </button>
          <button
            type="button"
            onClick={() => void saveActive()}
            disabled={busy || activeStrokes.length === 0}
            className="btn btn-primary btn-sm"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M17 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V7l-4-4zm-5 16a3 3 0 1 1 0-6 3 3 0 0 1 0 6zm3-10H5V5h10v4z" />
            </svg>
            <span>{t('preprocessInpaint.saveActive')}</span>
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
              /* 3-column layout（对齐裁剪页）— filmstrip / canvas / tool panel */
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
                  renderOverlay={(im) =>
                    (strokesByImage[im.name] ?? []).length > 0 ? (
                      <span className="fs-badge">✎</span>
                    ) : null
                  }
                />

                <div className="min-w-0 min-h-0 overflow-hidden">
                  <InpaintCanvas
                    key={activeImage.name}
                    ref={canvasRef}
                    imageUrl={rawUrl(activeImage)}
                    imageW={activeImage.w}
                    imageH={activeImage.h}
                    strokes={activeStrokes}
                    brush={brush}
                    onStrokeEnd={onStrokeEnd}
                    onPickColor={onPickColor}
                  />
                </div>

                <ToolPanel
                  brush={brush}
                  setBrush={setBrush}
                  recentColors={recentColors}
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
// Tool panel（right side）— PR-B 在顶部加「涂抹 | Mask」模式切换，布局不重排
// ---------------------------------------------------------------------------

function ToolPanel({
  brush,
  setBrush,
  recentColors,
}: {
  brush: BrushState
  setBrush: (v: BrushState | ((prev: BrushState) => BrushState)) => void
  recentColors: string[]
}) {
  const { t } = useTranslation()
  const [recentOpen, setRecentOpen] = useState(false)
  return (
    <div className="bg-sunken border border-subtle rounded-md flex flex-col h-full min-h-0 overflow-hidden">
      <div className="flex flex-col gap-3 p-2.5 flex-1 min-h-0 overflow-y-auto">
        {/* 颜色一行：caption + 色轮色块（原生 picker 自带取色器 / RGB 输入）
            + 历史颜色 icon dropdown */}
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2">
            <h3 className="caption m-0">{t('preprocessInpaint.brushColor')}</h3>
            <input
              type="color"
              value={brush.color}
              onChange={(e) => setBrush((p) => ({ ...p, color: e.target.value }))}
              className="w-10 h-7 p-0 border border-subtle rounded cursor-pointer bg-transparent shrink-0"
              title={t('preprocessInpaint.colorWheel')}
            />
            <span className="flex-1" />
            <button
              type="button"
              onClick={() => setRecentOpen((v) => !v)}
              disabled={recentColors.length === 0}
              className="btn btn-ghost btn-sm"
              title={t('preprocessInpaint.recentColors')}
            >🎨 {recentOpen ? '▴' : '▾'}</button>
          </div>
          {recentOpen && recentColors.length > 0 && (
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
        </div>

        {/* 笔刷参数 */}
        <div className="flex flex-col gap-1.5">
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
              className="input input-mono text-sm"
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
            <span className="font-mono text-fg-secondary w-9 text-right">
              {Math.round(brush.hardness * 100)}%
            </span>
          </label>
        </div>

      </div>

      {/* 底部固定栏：保存语义提示（对齐裁剪页 AR 锁定固定栏位置） */}
      <div className="shrink-0 border-t border-subtle px-2.5 py-2">
        <p className="text-[11px] text-fg-tertiary leading-snug m-0">
          {t('preprocessInpaint.footNote')}
        </p>
      </div>
    </div>
  )
}
