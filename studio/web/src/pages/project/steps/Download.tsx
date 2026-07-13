import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useOutletContext } from 'react-router-dom'
import {
  api,
  type DownloadFile,
  type Job,
  type ProjectDetail,
  type UploadResult,
  type Version,
} from '../../../api/client'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import PathPicker from '../../../components/PathPicker'
import StepShell from '../../../components/StepShell'
import UploadProgressBar from '../../../components/UploadProgressBar'
import { useDialog } from '../../../components/Dialog'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'
import { useUploadProgress } from '../../../lib/useUploadProgress'

// 跟 studio/datasets.py:IMAGE_EXTS 对齐 — 上传白名单 = 全链路图片白名单 + .zip。
const UPLOAD_ACCEPT =
  '.png,.jpg,.jpeg,.webp,.bmp,.gif,.zip,image/png,image/jpeg,image/webp,image/bmp,image/gif,application/zip'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

interface Estimate {
  tag: string
  api_source: 'gelbooru' | 'danbooru'
  exclude_tags: string[]
  effective_query: string
  count: number // -1 表示未知
}

// 信息密度优先：每个 panel 紧凑成单/双 inline 行；已下载 grid 占主区域。
export default function DownloadPage() {
  const { t } = useTranslation()
  const { project, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const [job, setJob] = useState<Job | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [files, setFiles] = useState<DownloadFile[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [anchor, setAnchor] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)
  const [tag, setTag] = useState('')
  const [apiSource, setApiSource] = useState<'gelbooru' | 'danbooru'>(
    'gelbooru'
  )
  const [estimate, setEstimate] = useState<Estimate | null>(null)
  const [count, setCount] = useState<number>(20)
  const [busy, setBusy] = useState(false)
  const [lastUpload, setLastUpload] = useState<UploadResult | null>(null)
  // 上传任务在 TaskLogDrawer 里独立成 LogSource —— server 端 on_log 推 SSE
  // `project_upload_log` / `project_upload_state`，前端 accumulate 成 lines + status。
  // 不同于 download 走 job 系统（log file 持久化 + 跨刷新可回放），upload 是
  // 同步路由没 job，SSE 期间错过的行就丢了 —— 跨页面 / 刷新拿不到上次的历史
  // 是预期行为（issue #251 "活着的优先" 的精神）。
  const [uploadLogs, setUploadLogs] = useState<string[]>([])
  const [uploadStatus, setUploadStatus] = useState<
    'pending' | 'running' | 'done' | 'failed' | null
  >(null)
  const [uploadStartedAt, setUploadStartedAt] = useState<number | null>(null)
  const [uploadFinishedAt, setUploadFinishedAt] = useState<number | null>(null)

  const refreshFiles = useCallback(async () => {
    try {
      const r = await api.listFiles(project.id)
      setFiles(r.items)
    } catch {
      /* ignore */
    }
  }, [project.id])

  const refreshStatus = useCallback(async () => {
    try {
      const r = await api.getDownloadStatus(project.id)
      setJob(r.job)
      setLogs(r.log_tail ? r.log_tail.split('\n') : [])
    } catch {
      /* ignore */
    }
  }, [project.id])

  useEffect(() => {
    void refreshStatus()
    void refreshFiles()
  }, [refreshStatus, refreshFiles])

  const jobIdRef = useRef<number | null>(null)
  jobIdRef.current = job?.id ?? null
  useEventStream((evt) => {
    const jid = jobIdRef.current
    if (evt.type === 'job_log_appended' && jid && evt.job_id === jid) {
      setLogs((prev) => [...prev, String(evt.text ?? '')])
    } else if (evt.type === 'job_state_changed' && jid && evt.job_id === jid) {
      void refreshStatus()
      if (evt.status === 'done' || evt.status === 'failed') {
        void refreshFiles()
        void reload()
      }
    } else if (
      evt.type === 'project_state_changed' &&
      evt.project_id === project.id
    ) {
      void refreshFiles()
    } else if (
      evt.type === 'project_upload_log' &&
      evt.project_id === project.id
    ) {
      const line = typeof evt.line === 'string' ? evt.line : ''
      if (line) setUploadLogs((prev) => [...prev, line])
    } else if (
      evt.type === 'project_upload_state' &&
      evt.project_id === project.id
    ) {
      const status = evt.status as typeof uploadStatus
      if (status === 'running') {
        setUploadLogs([])  // 新一轮上传，旧 lines 清掉
        setUploadStartedAt(Date.now() / 1000)
        setUploadFinishedAt(null)
      } else if (status === 'done' || status === 'failed') {
        setUploadFinishedAt(Date.now() / 1000)
      }
      setUploadStatus(status)
    }
  }, { onOpen: () => void refreshStatus() })

  useEffect(() => {
    setEstimate(null)
  }, [tag, apiSource])

  const doEstimate = async () => {
    if (!tag.trim()) {
      toast(t('download.tagEmpty'), 'error')
      return
    }
    setBusy(true)
    try {
      const r = await api.estimateDownload(project.id, {
        tag,
        api_source: apiSource,
      })
      setEstimate(r)
      if (r.count > 0) setCount(Math.min(r.count, 200))
      else if (r.count === 0) setCount(0)
      else setCount(20)
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const start = async () => {
    if (!estimate) return
    if (estimate.count === 0)
      return toast(t('download.noResults'), 'error')
    if (count < 1) return toast(t('download.countMin'), 'error')
    setBusy(true)
    try {
      const j = await api.startDownload(project.id, {
        tag,
        count,
        api_source: apiSource,
      })
      setJob(j)
      setLogs([])
      toast(t('download.started', { id: j.id }), 'success')
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
      toast(t('download.canceled'), 'success')
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const isLive = job?.status === 'running' || job?.status === 'pending'
  const maxCount = estimate && estimate.count > 0 ? estimate.count : 5000

  return (
    <StepShell
      idx={1}
      title={t('steps.download.title')}
      subtitle={t('steps.download.subtitle')}
      logSources={[
        job && {
          key: 'download',
          label: t('logDrawer.download'),
          status: job.status,
          lines: logs,
          startedAt: job.started_at,
          finishedAt: job.finished_at,
          onCancel: () => void cancel(),
        },
        uploadStatus && {
          key: 'upload',
          label: t('logDrawer.upload'),
          status: uploadStatus,
          lines: uploadLogs,
          startedAt: uploadStartedAt,
          finishedAt: uploadFinishedAt,
        },
      ]}
    >
    <div className="flex flex-col h-full gap-3 min-h-0">

      {/* 主体左右两栏：左（booru/upload + 状态 + grid） / 右（下载统计侧边栏） */}
      <div className="grid gap-3 flex-1 min-h-0" style={{ gridTemplateColumns: '1fr 240px' }}>

        {/* 左栏 */}
        <div className="flex flex-col gap-2 min-h-0 min-w-0">

          {/* 操作行：两个紧凑 panel 并排（窄屏堆叠） */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-2 shrink-0">
            <BooruPanel
              tag={tag}
              setTag={setTag}
              apiSource={apiSource}
              setApiSource={setApiSource}
              estimate={estimate}
              count={count}
              setCount={setCount}
              maxCount={maxCount}
              busy={busy}
              isLive={!!isLive}
              onEstimate={doEstimate}
              onStart={start}
            />
            <UploadPanel
              pid={project.id}
              onUploaded={(r) => {
                setLastUpload(r)
                void refreshFiles()
                void reload()
              }}
            />
          </div>

          {/* 上传结果条（下载任务日志走 StepShell 统一抽屉，issue #251） */}
          {lastUpload && (
            <div className="flex flex-col gap-1.5 shrink-0">
              <UploadResultStrip
                result={lastUpload}
                onDismiss={() => setLastUpload(null)}
              />
            </div>
          )}

          {/* 已下载 grid — 占满剩余高度，支持多选 + 删除 + 大图预览 */}
          <DownloadedGrid
            project={project}
            files={files}
            selected={selected}
            anchor={anchor}
            deleting={deleting}
            onSelect={(name, e) => {
              const r = applySelection(
                selected,
                name,
                e,
                files.map((f) => f.name),
                anchor
              )
              setSelected(r.next)
              setAnchor(r.anchor)
            }}
            onPreview={(name) => {
              const i = files.findIndex((f) => f.name === name)
              if (i >= 0) setPreviewIdx(i)
            }}
            onSelectAll={() => setSelected(new Set(files.map((f) => f.name)))}
            onClear={() => {
              setSelected(new Set())
              setAnchor(null)
            }}
            onDelete={async () => {
              if (selected.size === 0) return
              if (!(await confirm(
                t('download.confirmDelete', { n: selected.size }),
                { tone: 'danger', okText: t('common.delete') },
              ))) return
              setDeleting(true)
              try {
                const r = await api.deleteProjectFiles(
                  project.id,
                  Array.from(selected)
                )
                toast(
                  t('download.deletedToast', { deleted: r.deleted.length }) +
                    (r.missing.length ? t('download.deletedSkipped', { skipped: r.missing.length }) : ''),
                  'success'
                )
                setSelected(new Set())
                setAnchor(null)
                await refreshFiles()
                void reload()
              } catch (e) {
                toast(String(e), 'error')
              } finally {
                setDeleting(false)
              }
            }}
          />
        </div>

        {/* 右栏：下载统计侧边栏 */}
        <DownloadStatsSidebar files={files} projectDownloadCount={project.download_image_count} />
      </div>
    </div>

    {previewIdx !== null && files[previewIdx] && (
      <ImagePreviewModal
        src={api.projectThumbUrl(project.id, files[previewIdx].name, 'download', 0)}
        caption={files[previewIdx].name}
        index={previewIdx}
        total={files.length}
        hasPrev={previewIdx > 0}
        hasNext={previewIdx < files.length - 1}
        onClose={() => setPreviewIdx(null)}
        onPrev={() => previewIdx > 0 && setPreviewIdx(previewIdx - 1)}
        onNext={() => previewIdx < files.length - 1 && setPreviewIdx(previewIdx + 1)}
      />
    )}
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// 已下载 grid — 多选 + 删除
// ---------------------------------------------------------------------------

function DownloadedGrid({
  project,
  files,
  selected,
  anchor,
  deleting,
  onSelect,
  onPreview,
  onSelectAll,
  onClear,
  onDelete,
}: {
  project: ProjectDetail
  files: DownloadFile[]
  selected: Set<string>
  anchor: string | null
  deleting: boolean
  onSelect: (name: string, e: React.MouseEvent) => void
  onPreview: (name: string) => void
  onSelectAll: () => void
  onClear: () => void
  onDelete: () => void | Promise<void>
}) {
  const { t } = useTranslation()
  // anchor 仅父组件用，这里不读但保留参数避免未来漂移
  void anchor
  const items = useMemo(
    () =>
      files.map((f) => ({
        name: f.name,
        thumbUrl: api.projectThumbUrl(project.id, f.name),
      })),
    [files, project.id]
  )
  return (
    <section className="flex flex-col flex-1 min-h-0 rounded-md border border-subtle bg-surface overflow-hidden">
      <header className="flex items-center gap-2 shrink-0 px-2.5 py-1.5 border-b border-subtle text-sm">
        <h3 className="font-semibold">{t('download.sectionTitle')}</h3>
        <span className="text-fg-tertiary">{t('download.imageCount', { n: files.length })}</span>
        {selected.size > 0 && (
          <span className="text-accent">{t('download.selectedCount', { n: selected.size })}</span>
        )}
        <span className="flex-1" />
        <button
          onClick={onSelectAll}
          disabled={files.length === 0 || deleting}
          className="btn btn-ghost btn-sm"
        >
          {t('common.selectAll')}
        </button>
        <button
          onClick={onClear}
          disabled={selected.size === 0 || deleting}
          className="btn btn-ghost btn-sm"
        >
          {t('common.deselect')}
        </button>
        <button
          onClick={() => void onDelete()}
          disabled={selected.size === 0 || deleting}
          className="btn btn-sm bg-err-soft text-err"
          title={t('download.deleteTitle')}
        >
          {deleting ? t('download.deleting') : t('download.deleteBtn', { n: selected.size })}
        </button>
      </header>
      <div className="flex-1 min-h-0 overflow-y-auto p-2">
        <ImageGrid
          items={items}
          selected={selected}
          onSelect={onSelect}
          onActivate={onPreview}
          onPreview={onPreview}
          clickMode="activate"
          ariaLabel="downloaded-grid"
          emptyHint={t('download.emptyHint')}
        />
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Booru 紧凑 panel
// ---------------------------------------------------------------------------

interface BooruPanelProps {
  tag: string
  setTag: (v: string) => void
  apiSource: 'gelbooru' | 'danbooru'
  setApiSource: (v: 'gelbooru' | 'danbooru') => void
  estimate: Estimate | null
  count: number
  setCount: (n: number) => void
  maxCount: number
  busy: boolean
  isLive: boolean
  onEstimate: () => void
  onStart: () => void
}

function BooruPanel({
  tag,
  setTag,
  apiSource,
  setApiSource,
  estimate,
  count,
  setCount,
  maxCount,
  busy,
  isLive,
  onEstimate,
  onStart,
}: BooruPanelProps) {
  const { t } = useTranslation()
  const disabled = busy || isLive
  return (
    <section className="flex flex-col gap-1.5 rounded-md border border-subtle bg-surface px-3 py-2.5">
      <PanelTitle accent="cyan">{t('download.booruPanel')}</PanelTitle>
      <div className="flex items-center gap-1.5">
        <select
          value={apiSource}
          onChange={(e) =>
            setApiSource(e.target.value as 'gelbooru' | 'danbooru')
          }
          disabled={disabled}
          className="input text-sm"
          style={{ width: 'auto', padding: '3px 8px' }}
        >
          <option value="gelbooru">Gelbooru</option>
          <option value="danbooru">Danbooru</option>
        </select>
        <input
          value={tag}
          onChange={(e) => setTag(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && tag.trim() && !disabled) onEstimate()
          }}
          disabled={disabled}
          placeholder={t('download.tagPlaceholder')}
          className="input flex-1 text-sm"
          style={{ padding: '3px 8px' }}
        />
        <button
          onClick={onEstimate}
          disabled={disabled || !tag.trim()}
          className="btn btn-secondary btn-sm"
        >
          {busy && !estimate ? t('download.querying') : t('download.query')}
        </button>
      </div>
      {estimate && (
        <div className="flex items-center gap-1.5 flex-wrap text-sm text-fg-secondary">
          <span>
            {t('download.matches')}{' '}
            {estimate.count >= 0 ? (
              <strong className="text-accent">{estimate.count}</strong>
            ) : (
              <strong className="text-warn">{t('download.matchesUnknown')}</strong>
            )}
          </span>
          {estimate.count !== 0 && (
            <>
              <span className="text-dim">·</span>
              <span className="text-fg-tertiary">count</span>
              <input
                type="number"
                min={1}
                max={maxCount}
                value={count}
                onChange={(e) =>
                  setCount(Math.min(Number(e.target.value) || 1, maxCount))
                }
                disabled={disabled}
                className="input input-mono"
                style={{ width: 80, padding: '2px 6px' }}
              />
              {estimate.count > 0 && (
                <button
                  onClick={() => setCount(estimate.count)}
                  disabled={disabled}
                  className="btn btn-ghost btn-sm"
                >
                  {t('download.allN', { n: estimate.count })}
                </button>
              )}
              <button
                onClick={onStart}
                disabled={disabled || count < 1}
                className="btn btn-primary btn-sm ml-auto"
              >
                {isLive ? t('download.downloading') : t('download.startCount', { n: count })}
              </button>
            </>
          )}
          <span
            className="basis-full truncate text-xs text-fg-tertiary"
            title={estimate.effective_query}
          >
            query: <code>{estimate.effective_query}</code>
            {estimate.exclude_tags.length > 0 && (
              <>
                {' · exclude: '}
                <code>{estimate.exclude_tags.join(', ')}</code>
              </>
            )}
          </span>
        </div>
      )}
    </section>
  )
}

// ---------------------------------------------------------------------------
// 本地上传紧凑 panel
// ---------------------------------------------------------------------------

function UploadPanel({
  pid,
  onUploaded,
}: {
  pid: number
  onUploaded: (r: UploadResult) => void
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const inputRef = useRef<HTMLInputElement>(null)
  const [picked, setPicked] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [showPathPicker, setShowPathPicker] = useState(false)
  const uploadProgress = useUploadProgress()

  const choose = (fl: FileList | null) => {
    if (!fl || fl.length === 0) return
    setPicked(Array.from(fl))
  }
  const reset = () => {
    setPicked([])
    if (inputRef.current) inputRef.current.value = ''
  }
  const applyUploadResult = (r: UploadResult) => {
    const skipped = r.skipped.length
    toast(
      t('download.uploadAdded', { n: r.added.length }) +
        (skipped ? t('download.uploadSkippedSuffix', { skipped }) : ''),
      r.added.length > 0 ? 'success' : 'error'
    )
    onUploaded(r)
  }
  const submit = async () => {
    if (picked.length === 0) return
    const totalBytes = picked.reduce((s, f) => s + f.size, 0)
    setUploading(true)
    uploadProgress.start(totalBytes)
    try {
      const r = await api.uploadProjectFiles(pid, picked, uploadProgress.onProgress)
      uploadProgress.finish()
      applyUploadResult(r)
      reset()
      // 短延迟后清掉进度条；让用户看清完成状态
      window.setTimeout(() => uploadProgress.reset(), 800)
    } catch (e) {
      uploadProgress.fail(e)
      toast(String(e), 'error')
    } finally {
      setUploading(false)
    }
  }
  const importFromPath = async (path: string) => {
    setShowPathPicker(false)
    setUploading(true)
    try {
      applyUploadResult(await api.uploadProjectFileFromPath(pid, path))
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setUploading(false)
    }
  }
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    if (uploading) return
    if (e.dataTransfer.files?.length) choose(e.dataTransfer.files)
  }
  const totalBytes = picked.reduce((s, f) => s + f.size, 0)
  const fileNames = picked.map((f) => f.name).join(', ')

  return (
    <section className="flex flex-col gap-1.5 rounded-md border border-subtle bg-surface px-3 py-2.5">
      <PanelTitle accent="emerald">{t('download.uploadPanel')}</PanelTitle>
      <label
        onDragOver={(e) => {
          e.preventDefault()
          if (!uploading) setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={[
          'flex items-center gap-2 cursor-pointer transition-colors rounded-sm border border-dashed text-sm px-2.5 py-1.5',
          dragging ? 'border-accent text-accent bg-accent-soft' : 'border-dim text-fg-secondary',
        ].join(' ')}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={UPLOAD_ACCEPT}
          onChange={(e) => choose(e.target.files)}
          disabled={uploading}
          className="hidden"
        />
        <span className="font-medium">{t('download.clickOrDrop')}</span>
        <span className="text-fg-tertiary">{t('download.acceptedFormats')}</span>
        <span className="flex-1" />
        {picked.length > 0 && (
          <span className="text-ok">
            {t('download.filesSelected', { n: picked.length, mb: (totalBytes / 1024 / 1024).toFixed(1) })}
          </span>
        )}
      </label>
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          onClick={() => setShowPathPicker(true)}
          disabled={uploading}
          className="btn btn-secondary btn-sm"
        >
          {t('download.uploadFromPath')}
        </button>
        <span className="text-xs text-fg-tertiary">{t('download.uploadFromPathHint')}</span>
      </div>
      {picked.length > 0 && (
        <div className="flex items-center gap-1.5">
          <button
            onClick={submit}
            disabled={uploading}
            className="btn btn-primary btn-sm"
          >
            {uploading ? t('download.uploading') : t('download.uploadCount', { n: picked.length })}
          </button>
          <button
            onClick={reset}
            disabled={uploading}
            className="btn btn-ghost btn-sm"
          >
            {t('common.cancel')}
          </button>
          <span
            className="truncate min-w-0 flex-1 ml-1 text-xs text-fg-tertiary"
            title={fileNames}
          >
            {fileNames}
          </span>
        </div>
      )}
      {uploadProgress.state.phase !== 'idle' && (
        <UploadProgressBar state={uploadProgress.state} />
      )}
      {showPathPicker && (
        <PathPicker
          dirOnly={false}
          onClose={() => setShowPathPicker(false)}
          onPick={(path) => { void importFromPath(path) }}
        />
      )}
    </section>
  )
}

function UploadResultStrip({
  result,
  onDismiss,
}: {
  result: UploadResult
  onDismiss: () => void
}) {
  const { t } = useTranslation()
  const skipped = result.skipped.length
  const ok = result.added.length
  return (
    <details className="group rounded-md border border-subtle bg-surface overflow-hidden">
      <summary className="cursor-pointer flex items-center gap-2 list-none px-2.5 py-1.5 text-sm select-none">
        <span className="inline-block transition-transform group-open:rotate-90 text-fg-tertiary w-3">▸</span>
        <span className="badge badge-ok">upload</span>
        <span className="text-fg-secondary">
          {t('download.added')} <strong className="text-ok">{ok}</strong>
          {skipped > 0 && (
            <>
              {' · '}{t('download.skipped')} <strong className="text-warn">{skipped}</strong>
            </>
          )}
        </span>
        <span className="flex-1" />
        <button
          onClick={(e) => {
            e.preventDefault()
            onDismiss()
          }}
          className="btn btn-ghost btn-sm"
          title={t('common.close')}
        >
          ×
        </button>
      </summary>
      {skipped > 0 ? (
        <ul className="px-3 py-2 text-xs font-mono text-warn bg-sunken max-h-[160px] overflow-auto border-t border-subtle m-0 list-none">
          {result.skipped.map((s, i) => (
            <li key={`${s.name}-${i}`} className="truncate">
              {s.name} <span className="text-fg-tertiary">— {s.reason}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="px-3 py-2 text-xs text-fg-tertiary border-t border-subtle m-0">
          {t('download.allSucceeded')}
        </p>
      )}
    </details>
  )
}

// ---------------------------------------------------------------------------
// 杂项 — 小标题
// ---------------------------------------------------------------------------

function PanelTitle({
  accent,
  children,
}: {
  accent: 'cyan' | 'emerald'
  children: React.ReactNode
}) {
  const dotCls = accent === 'cyan' ? 'bg-accent' : 'bg-ok'
  return (
    <h3 className="caption flex items-center gap-1.5">
      <span className={`inline-block w-1.5 h-1.5 rounded-full shrink-0 ${dotCls}`} />
      {children}
    </h3>
  )
}

// ---------------------------------------------------------------------------
// 下载统计侧边栏
// ---------------------------------------------------------------------------
function DownloadStatsSidebar({
  files,
  projectDownloadCount,
}: {
  files: DownloadFile[]
  projectDownloadCount: number
}) {
  const { t } = useTranslation()
  // 按扩展名分组统计
  const extCounts = useMemo(() => {
    const m: Record<string, number> = {}
    for (const f of files) {
      const ext = f.name.split('.').pop()?.toLowerCase() ?? '?'
      m[ext] = (m[ext] ?? 0) + 1
    }
    return Object.entries(m).sort((a, b) => b[1] - a[1])
  }, [files])

  return (
    <div className="flex flex-col gap-3 min-w-0">
      {/* 总量卡片 */}
      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <PanelTitle accent="cyan">{t('download.statsTitle')}</PanelTitle>
        <StatRow label={t('download.statsTotal')} value={projectDownloadCount} />
        <StatRow label={t('download.statsVisible')} value={files.length} />
        {files.length > 0 && (
          <StatRow
            label={t('download.statsTotalSize')}
            value={files.reduce((s, f) => s + f.size, 0)}
            format="bytes"
          />
        )}
      </div>

      {/* 格式分布 */}
      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5 flex-1 flex flex-col min-h-0">
        <PanelTitle accent="emerald">{t('download.formatTitle')}</PanelTitle>
        {files.length === 0 ? (
          <p className="text-xs text-fg-tertiary m-0 mt-1.5">
            {t('common.noImages')}
          </p>
        ) : (
          <div className="flex flex-col gap-1.5 mt-1.5 flex-1 overflow-y-auto">
            {extCounts.map(([ext, count]) => {
              const pct = Math.round((count / files.length) * 100)
              return (
                <div key={ext} className="flex items-center gap-1.5">
                  <span className="text-xs font-mono text-fg-primary w-9 uppercase text-right">
                    {ext}
                  </span>
                  <div className="flex-1 h-1.5 rounded bg-sunken overflow-hidden">
                    <div
                      className="h-full bg-accent rounded transition-[width] duration-300 ease-out"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="text-xs text-fg-tertiary w-9 text-right">{count}</span>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

function StatRow({
  label,
  value,
  format,
}: {
  label: string
  value: number
  format?: 'bytes'
}) {
  const display = format === 'bytes'
    ? value > 1024 * 1024
      ? `${(value / 1024 / 1024).toFixed(1)} MB`
      : `${(value / 1024).toFixed(0)} KB`
    : String(value)
  return (
    <div className="flex justify-between items-baseline mt-1.5 text-xs">
      <span className="text-fg-tertiary">{label}</span>
      <span className="font-mono text-fg-primary font-medium">{display}</span>
    </div>
  )
}
