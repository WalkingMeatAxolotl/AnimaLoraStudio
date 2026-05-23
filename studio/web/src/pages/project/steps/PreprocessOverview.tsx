import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useOutletContext } from 'react-router-dom'
import {
  api,
  type CropWorkspaceItem,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import { useDialog } from '../../../components/Dialog'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import PreprocessToolsBar from '../../../components/preprocess/PreprocessToolsBar'
import StepShell from '../../../components/StepShell'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

/** Preprocess overview — the gallery for everything that's been processed.
 *
 *  Lives at `?tool=overview`. Responsibilities:
 *  - Show all preprocess/ workspace files (post-upscale / crop / etc.)
 *  - Multi-select via shift-click region and ctrl-click toggle (standard
 *    behavior provided by `applySelection`)
 *  - Click → preview enlarged image in a modal (parity with the Download page
 *    review flow), separate from any selection state
 *  - Undo selected / undo all → calls `restorePreprocessFiles` / `resetPreprocessFiles`
 *
 *  Per-tool pages (upscale, crop) no longer carry undo controls — undo is
 *  always done here so users have one place to manage state. See
 *  docs/design/preprocess-crop-design.md (preprocess overview section).
 */
export default function PreprocessOverviewPage() {
  const { t } = useTranslation()
  const { project, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()

  const [images, setImages] = useState<CropWorkspaceItem[]>([])
  const [loading, setLoading] = useState(true)
  const [sel, setSel] = useState<Set<string>>(new Set())
  const [selAnchor, setSelAnchor] = useState<string | null>(null)
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)

  const refresh = useCallback(async () => {
    try {
      const r = await api.listCropWorkspace(project.id)
      setImages(r.images)
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }, [project.id])
  useEffect(() => { void refresh() }, [refresh])

  // Live-update on any preprocess SSE — upscale / crop / restore all change the
  // workspace; cheap to refetch (PIL header read for each file).
  useEventStream((evt) => {
    if (
      evt.type === 'project_state_changed' && evt.project_id === project.id ||
      evt.type === 'preprocess_progress' && evt.project_id === project.id ||
      evt.type === 'crop_progress' && evt.project_id === project.id
    ) {
      void refresh()
    }
  })

  // Filter: show only "processed" (= manifest-tracked) images so this tab is
  // exactly the set the user might want to undo. Originals (download/) are
  // shown in the Download tool, not here.
  const processed = useMemo(
    () => images.filter((im) => im.processed),
    [images],
  )

  const items = useMemo(
    () =>
      processed.map((im) => ({
        name: im.name,
        // Address by preprocess filename, not by origin/source — multi-crop
        // fan-out (X_c0.png / X_c1.png both with origin X.png) would otherwise
        // all show the same thumbnail (the [0] of resolve_origin).
        thumbUrl: api.projectThumbUrl(project.id, im.name, 'preprocess', 256),
        meta: `${im.w}×${im.h}`,
      })),
    [processed, project.id],
  )
  const visibleNames = useMemo(() => items.map((i) => i.name), [items])

  // The preview modal uses image source name (so the thumbnail endpoint
  // resolves through the manifest and picks the actual preprocess output).
  const previewItem = previewIdx !== null ? processed[previewIdx] : null

  const restoreNames = useCallback(async (names: string[]) => {
    if (names.length === 0) return
    if (!(await confirm(
      t('preprocessOverview.confirmRestore', { n: names.length }),
      { tone: 'danger', okText: t('preprocessOverview.confirmRestoreOk') },
    ))) return
    try {
      const r = await api.restorePreprocessFiles(project.id, names)
      toast(
        t('preprocessOverview.restoredToast', { n: r.restored.length }),
        'success',
      )
      setSel(new Set())
      setSelAnchor(null)
      await refresh()
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }, [confirm, project.id, t, toast, refresh, reload])

  const resetAll = useCallback(async () => {
    if (processed.length === 0) return
    if (!(await confirm(
      t('preprocessOverview.confirmResetAll', { n: processed.length }),
      { tone: 'danger', okText: t('preprocessOverview.confirmResetAllOk') },
    ))) return
    try {
      await api.resetPreprocessFiles(project.id)
      toast(t('preprocessOverview.resetAllToast'), 'success')
      setSel(new Set())
      setSelAnchor(null)
      await refresh()
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }, [confirm, processed.length, project.id, t, toast, refresh, reload])

  return (
    <StepShell
      idx={2}
      title={t('steps.preprocess.title')}
      subtitle={t('preprocessOverview.subtitle')}
    >
      <div className="flex flex-col h-full gap-3 min-h-0">
        <PreprocessToolsBar current="overview" projectId={project.id} />

        <section className="flex flex-col flex-1 min-h-0 rounded-md border border-subtle bg-surface overflow-hidden">
          <header className="flex items-center gap-2 shrink-0 px-3 py-2 border-b border-subtle text-sm flex-wrap">
            <h3 className="font-semibold">{t('preprocessOverview.title')}</h3>
            <span className="text-fg-tertiary text-xs">
              {t('preprocessOverview.totalCount', { n: processed.length })}
            </span>
            {sel.size > 0 && (
              <span className="text-accent text-xs">
                {t('preprocessOverview.selectedCount', { n: sel.size })}
              </span>
            )}
            <span className="flex-1" />
            <button
              onClick={() => setSel(new Set(visibleNames))}
              disabled={items.length === 0}
              className="btn btn-ghost btn-sm"
            >{t('common.selectAll')}</button>
            <button
              onClick={() => { setSel(new Set()); setSelAnchor(null) }}
              disabled={sel.size === 0}
              className="btn btn-ghost btn-sm"
            >{t('common.deselect')}</button>
            <button
              onClick={() => void restoreNames(Array.from(sel))}
              disabled={sel.size === 0}
              className="btn btn-sm bg-err-soft text-err"
              title={t('preprocessOverview.restoreSelectedTitle')}
            >{t('preprocessOverview.restoreSelected', { n: sel.size })}</button>
            <button
              onClick={() => void resetAll()}
              disabled={processed.length === 0}
              className="btn btn-sm btn-secondary"
              title={t('preprocessOverview.resetAllTitle')}
            >↶ {t('preprocessOverview.resetAll')}</button>
          </header>

          <div className="flex-1 min-h-0 overflow-y-auto p-3">
            {loading && (
              <p className="text-fg-tertiary text-sm">{t('common.loading')}</p>
            )}
            {!loading && processed.length === 0 && (
              <p className="text-fg-tertiary text-sm">
                {t('preprocessOverview.empty')}
              </p>
            )}
            {processed.length > 0 && (
              <ImageGrid
                items={items}
                selected={sel}
                onSelect={(name, e) => {
                  const r = applySelection(sel, name, e, visibleNames, selAnchor)
                  setSel(r.next)
                  setSelAnchor(r.anchor)
                }}
                onActivate={(name) => {
                  const i = visibleNames.indexOf(name)
                  if (i >= 0) setPreviewIdx(i)
                }}
                onPreview={(name) => {
                  const i = visibleNames.indexOf(name)
                  if (i >= 0) setPreviewIdx(i)
                }}
                clickMode="activate"
                ariaLabel="preprocess-overview-grid"
                emptyHint={t('preprocessOverview.empty')}
              />
            )}
          </div>
        </section>
      </div>

      {previewItem && (
        <ImagePreviewModal
          src={api.projectThumbUrl(project.id, previewItem.name, 'preprocess', 1600)}
          caption={`${previewItem.name} · ${previewItem.w}×${previewItem.h}`}
          hasPrev={previewIdx! > 0}
          hasNext={previewIdx! < processed.length - 1}
          onClose={() => setPreviewIdx(null)}
          onPrev={() => previewIdx! > 0 && setPreviewIdx(previewIdx! - 1)}
          onNext={() => previewIdx! < processed.length - 1 && setPreviewIdx(previewIdx! + 1)}
        />
      )}
    </StepShell>
  )
}
