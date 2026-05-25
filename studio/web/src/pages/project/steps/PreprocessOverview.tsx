import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useOutletContext } from 'react-router-dom'
import {
  api,
  type CropWorkspaceItem,
  type DuplicateRemovedItem,
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

type Tab = 'all' | 'processed' | 'removed'

/** Preprocess overview — 三 tab 视图：
 *
 *  - **all**：当前数据集真实状态。list_crop_workspace 合并了 download 未派生 +
 *    preprocess 派生产物（已 filter duplicate_removed）。每张图按各自来源
 *    取缩略图：processed → bucket=preprocess + name；否则 bucket=download + source。
 *    只读视图，点击放大。
 *  - **processed**：已处理产物子集。可选中恢复（撤销处理回 download/ 原图）
 *    或全部撤销。
 *  - **removed**：被去重审核标记的 entry。物理图仍在 download/{source}，
 *    缩略图按 download bucket 取。可选中恢复（删 manifest entry）。
 *
 *  恢复都走 restorePreprocessFiles —— restore() 对 duplicate_removed entry 也
 *  work（删 entry，对应 PNG 不存在静默跳过）。
 */
export default function PreprocessOverviewPage() {
  const { t } = useTranslation()
  const { project, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()

  const [tab, setTab] = useState<Tab>('all')
  const [workspace, setWorkspace] = useState<CropWorkspaceItem[]>([])
  const [removed, setRemoved] = useState<DuplicateRemovedItem[]>([])
  const [loading, setLoading] = useState(true)
  const [sel, setSel] = useState<Set<string>>(new Set())
  const [selAnchor, setSelAnchor] = useState<string | null>(null)
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)

  const refresh = useCallback(async () => {
    try {
      const [ws, rm] = await Promise.all([
        api.listCropWorkspace(project.id),
        api.listPreprocessDuplicatesRemoved(project.id),
      ])
      setWorkspace(ws.images)
      setRemoved(rm.images)
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }, [project.id])
  useEffect(() => { void refresh() }, [refresh])

  // Live-update on preprocess SSE — upscale / crop / restore / duplicate apply
  // all mutate manifest; cheap to refetch.
  useEventStream((evt) => {
    if (
      (evt.type === 'project_state_changed' && evt.project_id === project.id) ||
      (evt.type === 'preprocess_progress' && evt.project_id === project.id) ||
      (evt.type === 'crop_progress' && evt.project_id === project.id)
    ) {
      void refresh()
    }
  })

  // Tab 切换重置选择和预览
  useEffect(() => {
    setSel(new Set())
    setSelAnchor(null)
    setPreviewIdx(null)
  }, [tab])

  const processed = useMemo(
    () => workspace.filter((im) => im.processed),
    [workspace],
  )

  type GridItem = {
    name: string
    thumbUrl: string
    previewUrl: string
    caption: string
  }

  const wsThumb = useCallback(
    (im: CropWorkspaceItem, size: number) =>
      im.processed
        ? api.projectThumbUrl(project.id, im.name, 'preprocess', size, im.mtime)
        : api.projectThumbUrl(project.id, im.source, 'download', size, im.mtime),
    [project.id],
  )

  const allItems = useMemo<GridItem[]>(
    () => workspace.map((im) => ({
      name: im.name,
      thumbUrl: wsThumb(im, 256),
      previewUrl: wsThumb(im, 1600),
      caption: `${im.name} · ${im.w}×${im.h}`,
    })),
    [workspace, wsThumb],
  )
  const processedItems = useMemo<GridItem[]>(
    () => processed.map((im) => ({
      name: im.name,
      thumbUrl: wsThumb(im, 256),
      previewUrl: wsThumb(im, 1600),
      caption: `${im.name} · ${im.w}×${im.h}`,
    })),
    [processed, wsThumb],
  )
  const removedItems = useMemo<GridItem[]>(
    () => removed.map((im) => ({
      name: im.name,
      thumbUrl: api.projectThumbUrl(project.id, im.source, 'download', 256, im.mtime),
      previewUrl: api.projectThumbUrl(project.id, im.source, 'download', 1600, im.mtime),
      caption: im.w && im.h ? `${im.source} · ${im.w}×${im.h}` : im.source,
    })),
    [removed, project.id],
  )

  const items =
    tab === 'all' ? allItems
    : tab === 'processed' ? processedItems
    : removedItems
  const visibleNames = useMemo(() => items.map((i) => i.name), [items])
  const previewItem = previewIdx !== null ? items[previewIdx] : null

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

  const tabDefs: { id: Tab; label: string; count: number }[] = [
    { id: 'all', label: t('preprocessOverview.tabAll'), count: workspace.length },
    { id: 'processed', label: t('preprocessOverview.tabProcessed'), count: processed.length },
    { id: 'removed', label: t('preprocessOverview.tabRemoved'), count: removed.length },
  ]

  const emptyHint =
    tab === 'all' ? t('preprocessOverview.emptyAll')
    : tab === 'processed' ? t('preprocessOverview.empty')
    : t('preprocessOverview.emptyRemoved')

  const canMutate = tab === 'processed' || tab === 'removed'

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
            <div className="flex items-center gap-1">
              {tabDefs.map((td) => (
                <button
                  key={td.id}
                  onClick={() => setTab(td.id)}
                  className={`px-2.5 py-1 rounded-md text-sm font-medium ${
                    tab === td.id
                      ? 'bg-overlay text-fg-primary'
                      : 'text-fg-secondary hover:bg-overlay/50'
                  }`}
                >
                  {td.label}
                  <span className="ml-1 text-fg-tertiary text-xs">{td.count}</span>
                </button>
              ))}
            </div>
            {sel.size > 0 && (
              <span className="text-accent text-xs">
                {t('preprocessOverview.selectedCount', { n: sel.size })}
              </span>
            )}
            <span className="flex-1" />
            {canMutate && (
              <>
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
                {tab === 'processed' && (
                  <button
                    onClick={() => void resetAll()}
                    disabled={processed.length === 0}
                    className="btn btn-sm btn-secondary"
                    title={t('preprocessOverview.resetAllTitle')}
                  >↶ {t('preprocessOverview.resetAll')}</button>
                )}
              </>
            )}
          </header>

          <div className="flex-1 min-h-0 overflow-y-auto p-3">
            {loading && (
              <p className="text-fg-tertiary text-sm">{t('common.loading')}</p>
            )}
            {!loading && items.length === 0 && (
              <p className="text-fg-tertiary text-sm">{emptyHint}</p>
            )}
            {items.length > 0 && (
              <ImageGrid
                items={items}
                selected={sel}
                onSelect={canMutate ? (name, e) => {
                  const r = applySelection(sel, name, e, visibleNames, selAnchor)
                  setSel(r.next)
                  setSelAnchor(r.anchor)
                } : () => {}}
                onActivate={(name) => {
                  const i = visibleNames.indexOf(name)
                  if (i >= 0) setPreviewIdx(i)
                }}
                onPreview={(name) => {
                  const i = visibleNames.indexOf(name)
                  if (i >= 0) setPreviewIdx(i)
                }}
                clickMode="activate"
                ariaLabel={`preprocess-overview-grid-${tab}`}
                emptyHint={emptyHint}
              />
            )}
          </div>
        </section>
      </div>

      {previewItem && (
        <ImagePreviewModal
          src={previewItem.previewUrl}
          caption={previewItem.caption}
          hasPrev={previewIdx! > 0}
          hasNext={previewIdx! < items.length - 1}
          onClose={() => setPreviewIdx(null)}
          onPrev={() => previewIdx! > 0 && setPreviewIdx(previewIdx! - 1)}
          onNext={() => previewIdx! < items.length - 1 && setPreviewIdx(previewIdx! + 1)}
        />
      )}
    </StepShell>
  )
}
