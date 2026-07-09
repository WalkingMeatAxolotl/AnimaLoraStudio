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

type Tab = 'all' | 'removed'

/** Preprocess overview — 两 tab 视图：
 *
 *  - **all**：当前数据集真实状态（处理后数据集）。list_crop_workspace 合并
 *    了 download 未派生 + preprocess 派生产物（已 filter duplicate_removed）。
 *    每张图按各自来源取缩略图；processed 项右下角带「已处理」badge，点击放大
 *    走 split 布局（左 download 原图 + 右 preprocess 派生）；未处理项点击单图。
 *    可选中已处理项恢复（撤销处理回 download/ 原图）或全部撤销。
 *  - **removed**：被去重审核标记的 entry（已删除）。物理图仍在 download/{source}，
 *    缩略图按 download bucket 取。可选中恢复（删 manifest entry）。
 *
 *  恢复都走 restorePreprocessFiles —— restore() 对 duplicate_removed entry 也
 *  work（删 entry，对应 PNG 不存在静默跳过）。
 */
export default function PreprocessOverviewPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const vid = activeVersion?.id ?? 0

  const [tab, setTab] = useState<Tab>('all')
  const [workspace, setWorkspace] = useState<CropWorkspaceItem[]>([])
  const [removed, setRemoved] = useState<DuplicateRemovedItem[]>([])
  const [loading, setLoading] = useState(true)
  const [sel, setSel] = useState<Set<string>>(new Set())
  const [selAnchor, setSelAnchor] = useState<string | null>(null)
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)

  const refresh = useCallback(async () => {
    if (!vid) return
    try {
      const [ws, rm] = await Promise.all([
        api.listCropWorkspaceTrain(project.id, vid),
        api.listPreprocessDuplicatesRemovedTrain(project.id, vid),
      ])
      setWorkspace(ws.images)
      setRemoved(rm.images)
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }, [project.id, vid])
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
  const processedNames = useMemo(
    () => new Set(processed.map((p) => p.name)),
    [processed],
  )

  type GridItem = {
    name: string
    thumbUrl: string
    previewUrl: string
    /** 右侧对比图（preprocess 派生）。设了 modal 切 split 布局。仅 processed 项有。 */
    compareSrc?: string
    /** cell 右下角常显小角标。仅 processed 项有「已处理」徽章。 */
    badge?: string
    caption: string
  }

  // ADR 0010: workspace 的 name 是 train rel path "1_data/X.png"。
  // 拆 folder + filename 喂 versionThumbUrl(bucket='train')；split 预览左侧
  // 仍走 download bucket 看原图（origin 平铺名）。
  const splitRel = (rel: string) => {
    const i = rel.lastIndexOf('/')
    return i >= 0
      ? { folder: rel.slice(0, i), filename: rel.slice(i + 1) }
      : { folder: '', filename: rel }
  }

  const allItems = useMemo<GridItem[]>(
    () => workspace.map((im) => {
      const { folder, filename } = splitRel(im.name)
      const trainThumb = (size: number) =>
        api.versionThumbUrl(project.id, vid, 'train', filename, folder, size)
          + `&_=${im.mtime}`
      if (im.processed) {
        return {
          name: im.name,
          thumbUrl: trainThumb(256),
          // split 预览：左 = download 原图（origin 平铺名），右 = train 派生
          previewUrl: api.projectThumbUrl(project.id, im.source, 'download', 1600, im.mtime, true),
          compareSrc: trainThumb(1600),
          badge: t('preprocessOverview.badgeProcessed'),
          caption: `${im.name} · ${im.w}×${im.h}`,
        }
      }
      // 原样未处理：train 里的图就是 download 原图副本
      return {
        name: im.name,
        thumbUrl: trainThumb(256),
        previewUrl: trainThumb(1600),
        caption: `${im.name} · ${im.w}×${im.h}`,
      }
    }),
    [workspace, project.id, vid, t],
  )
  const removedItems = useMemo<GridItem[]>(
    () => removed.map((im) => ({
      name: im.name,
      // duplicate_removed 物理已删；缩略图走 download bucket + im.source (origin)
      thumbUrl: api.projectThumbUrl(project.id, im.source, 'download', 256, im.mtime, true),
      previewUrl: api.projectThumbUrl(project.id, im.source, 'download', 1600, im.mtime, true),
      caption: im.w && im.h ? `${im.source} · ${im.w}×${im.h}` : im.source,
    })),
    [removed, project.id],
  )

  const items = tab === 'all' ? allItems : removedItems
  const visibleNames = useMemo(() => items.map((i) => i.name), [items])
  const previewItem = previewIdx !== null ? items[previewIdx] : null

  const restoreNames = useCallback(async (names: string[]) => {
    if (names.length === 0) return
    if (!(await confirm(
      t('preprocessOverview.confirmRestore', { n: names.length }),
      { tone: 'danger', okText: t('preprocessOverview.confirmRestoreOk') },
    ))) return
    try {
      const r = await api.restorePreprocessFilesTrain(project.id, vid, names)
      toast(
        t('preprocessOverview.restoredToast', {
          n: r.restored.length,
          missing: r.no_origin.length,
        }),
        r.no_origin.length > 0 ? 'error' : 'success',
      )
      setSel(new Set())
      setSelAnchor(null)
      await refresh()
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }, [confirm, project.id, vid, t, toast, refresh, reload])

  const resetAll = useCallback(async () => {
    if (processed.length === 0) return
    if (!(await confirm(
      t('preprocessOverview.confirmResetAll', { n: processed.length }),
      { tone: 'danger', okText: t('preprocessOverview.confirmResetAllOk') },
    ))) return
    try {
      await api.resetPreprocessFilesTrain(project.id, vid)
      toast(t('preprocessOverview.resetAllToast'), 'success')
      setSel(new Set())
      setSelAnchor(null)
      await refresh()
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }, [confirm, processed.length, project.id, vid, t, toast, refresh, reload])

  // all tab 里 select all 只选「已处理」项 —— 未处理的 download 原图没什么
  // 可恢复（没有 manifest entry），加进选中会浪费一次 confirm。
  const selectableNames = useMemo(
    () => tab === 'all'
      ? visibleNames.filter((n) => processedNames.has(n))
      : visibleNames,
    [tab, visibleNames, processedNames],
  )

  const tabDefs: { id: Tab; label: string; count: number }[] = [
    { id: 'all', label: t('preprocessOverview.tabAll'), count: workspace.length },
    { id: 'removed', label: t('preprocessOverview.tabRemoved'), count: removed.length },
  ]

  const emptyHint =
    tab === 'all' ? t('preprocessOverview.emptyAll')
    : t('preprocessOverview.emptyRemoved')

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
      subtitle={t('preprocessOverview.subtitle')}
      belowHeader={<PreprocessToolsBar current="overview" projectId={project.id} versionId={vid} />}
    >
      <div className="flex flex-col h-full gap-3 min-h-0">
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
            <button
              onClick={() => setSel(new Set(selectableNames))}
              disabled={selectableNames.length === 0}
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
            {tab === 'all' && (
              <button
                onClick={() => void resetAll()}
                disabled={processed.length === 0}
                className="btn btn-sm btn-secondary"
                title={t('preprocessOverview.resetAllTitle')}
              >↶ {t('preprocessOverview.resetAll')}</button>
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
                onSelect={(name, e) => {
                  // all tab 里只有 processed 项可选；未处理 cell 点击直接 noop
                  // （保留 activate 单击放大）
                  if (tab === 'all' && !processedNames.has(name)) return
                  const r = applySelection(sel, name, e, selectableNames, selAnchor)
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
          compareSrc={previewItem.compareSrc}
          srcLabel={previewItem.compareSrc ? t('preprocessOverview.compareOriginal') : undefined}
          compareLabel={previewItem.compareSrc ? t('preprocessOverview.compareProcessed') : undefined}
          caption={previewItem.caption}
          index={previewIdx!}
          total={items.length}
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
