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
import ActionGroup from '../../../components/ActionGroup'
import Alert from '../../../components/Alert'
import Button from '../../../components/Button'
import Card from '../../../components/Card'
import { Input, Select } from '../../../components/FormControl'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import PathPicker from '../../../components/PathPicker'
import StepShell from '../../../components/StepShell'
import UploadProgressBar from '../../../components/UploadProgressBar'
import { useDialog } from '../../../components/Dialog'
import { useToast } from '../../../components/Toast'
import { compareImageName } from '../../../lib/imageSort'
import { useEventStream } from '../../../lib/useEventStream'
import { formatBytes, useUploadProgress } from '../../../lib/useUploadProgress'

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

// 信息密度优先：两种获取入口并列，图片检查区占主区域。
export default function DownloadPage() {
  const { t } = useTranslation()
  const { project, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const [job, setJob] = useState<Job | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [files, setFiles] = useState<DownloadFile[]>([])
  const [filesLoading, setFilesLoading] = useState(true)
  const [hasLoadedFiles, setHasLoadedFiles] = useState(false)
  const [filesError, setFilesError] = useState<string | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [anchor, setAnchor] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)
  const [tag, setTag] = useState('')
  const [apiSource, setApiSource] = useState<'gelbooru' | 'danbooru'>(
    'gelbooru'
  )
  const [estimate, setEstimate] = useState<Estimate | null>(null)
  const [count, setCount] = useState('20')
  const [booruAction, setBooruAction] = useState<'estimate' | 'start' | null>(null)
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
    setFilesLoading(true)
    setFilesError(null)
    try {
      const r = await api.listFiles(project.id)
      setFiles([...r.items].sort((a, b) => compareImageName(a.name, b.name)))
      setHasLoadedFiles(true)
    } catch {
      setFilesError(t('download.filesLoadError'))
    } finally {
      setFilesLoading(false)
    }
  }, [project.id, t])

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
    setBooruAction('estimate')
    try {
      const r = await api.estimateDownload(project.id, {
        tag,
        api_source: apiSource,
      })
      setEstimate(r)
      if (r.count > 0) setCount(String(Math.min(r.count, 200)))
      else if (r.count === 0) setCount('0')
      else setCount('20')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBooruAction(null)
    }
  }

  const start = async () => {
    if (!estimate) return
    if (estimate.count === 0)
      return toast(t('download.noResults'), 'error')
    const requestedCount = Number(count)
    if (!Number.isInteger(requestedCount) || requestedCount < 1) {
      return toast(t('download.countMin'), 'error')
    }
    setBooruAction('start')
    try {
      const j = await api.startDownload(project.id, {
        tag,
        count: requestedCount,
        api_source: apiSource,
      })
      setJob(j)
      setLogs([])
      toast(t('download.started', { id: j.id }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBooruAction(null)
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
  const totalBytes = files.reduce((sum, file) => sum + file.size, 0)

  return (
    <StepShell
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
      <div
        className="grid shrink-0 gap-3 xl:grid-cols-2"
        data-download-acquisition-grid
      >
        <Card
          as="section"
          radius="compact"
          className="min-w-0 overflow-hidden"
          aria-labelledby="download-booru-title"
        >
          <header className="border-b border-subtle px-3 py-2">
            <h2 id="download-booru-title" className="m-0 type-panel-title">
              {t('download.modeBooru')}
            </h2>
          </header>
          <div className="p-3">
            <BooruPanel
              tag={tag}
              setTag={setTag}
              apiSource={apiSource}
              setApiSource={setApiSource}
              estimate={estimate}
              count={count}
              setCount={setCount}
              maxCount={maxCount}
              action={booruAction}
              isLive={!!isLive}
              onEstimate={doEstimate}
              onStart={start}
            />
          </div>
        </Card>

        <Card
          as="section"
          radius="compact"
          className="min-w-0 overflow-hidden"
          aria-labelledby="download-import-title"
        >
          <header className="border-b border-subtle px-3 py-2">
            <h2 id="download-import-title" className="m-0 type-panel-title">
              {t('download.modeUpload')}
            </h2>
          </header>
          <div className="p-3">
            <UploadPanel
              pid={project.id}
              onUploaded={(result) => {
                setLastUpload(result)
                void refreshFiles()
                void reload()
              }}
            />
          </div>
        </Card>
      </div>

      {lastUpload && (
        <UploadResultStrip
          result={lastUpload}
          onDismiss={() => setLastUpload(null)}
        />
      )}

      {filesError && (
        <Alert
          tone={hasLoadedFiles ? 'warning' : 'danger'}
          size="sm"
          title={t(hasLoadedFiles ? 'download.refreshErrorTitle' : 'download.loadErrorTitle')}
          action={
            <Button variant="secondary" size="sm" onClick={() => void refreshFiles()}>
              {t('common.retry')}
            </Button>
          }
          role="alert"
        >
          {filesError}
        </Alert>
      )}

      <DownloadedGrid
        project={project}
        files={files}
        totalBytes={totalBytes}
        loading={filesLoading}
        emptyHint={
          filesLoading && !hasLoadedFiles
            ? t('common.loading')
            : filesError && !hasLoadedFiles
              ? t('download.loadFailedEmpty')
              : t('download.emptyHint')
        }
        selected={selected}
        anchor={anchor}
        deleting={deleting}
        onSelect={(name, event) => {
          const result = applySelection(
            selected,
            name,
            event,
            files.map((file) => file.name),
            anchor,
          )
          setSelected(result.next)
          setAnchor(result.anchor)
        }}
        onPreview={(name) => {
          const index = files.findIndex((file) => file.name === name)
          if (index >= 0) setPreviewIdx(index)
        }}
        onSelectAll={() => setSelected(new Set(files.map((file) => file.name)))}
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
            const result = await api.deleteProjectFiles(project.id, Array.from(selected))
            toast(
              t('download.deletedToast', { deleted: result.deleted.length }) +
                (result.missing.length
                  ? t('download.deletedSkipped', { skipped: result.missing.length })
                  : ''),
              'success',
            )
            setSelected(new Set())
            setAnchor(null)
            await refreshFiles()
            void reload()
          } catch (error) {
            toast(String(error), 'error')
          } finally {
            setDeleting(false)
          }
        }}
      />
    </div>

    {previewIdx !== null && files[previewIdx] && (
      <ImagePreviewModal
        src={api.projectThumbUrl(project.id, files[previewIdx].name, 'download', 1600)}
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
  totalBytes,
  loading,
  emptyHint,
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
  totalBytes: number
  loading: boolean
  emptyHint: string
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
    <Card
      as="section"
      radius="compact"
      className="flex flex-col flex-1 min-h-0 overflow-hidden"
      aria-busy={loading || undefined}
    >
      <header className="flex flex-wrap items-center gap-related shrink-0 px-3 py-2 border-b border-subtle text-sm">
        <h2 className="font-semibold">{t('download.sectionTitle')}</h2>
        <span className="text-fg-tertiary">
          {t('download.workspaceSummary', { n: files.length, size: formatBytes(totalBytes) })}
        </span>
        <ActionGroup
          className="ml-auto"
          aria-label={t('download.gridActionsLabel')}
          status={selected.size > 0 && (
            <span className="text-accent text-xs">
              {t('download.selectedCount', { n: selected.size })}
            </span>
          )}
          secondary={(
            <>
              <Button
                variant="ghost"
                size="sm"
                onClick={onSelectAll}
                disabled={files.length === 0 || deleting}
              >
                {t('common.selectAll')}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={onClear}
                disabled={selected.size === 0 || deleting}
              >
                {t('common.deselect')}
              </Button>
            </>
          )}
          primary={(
            <Button
              variant="danger"
              size="sm"
              onClick={() => void onDelete()}
              disabled={selected.size === 0 || deleting}
              loading={deleting}
              title={t('download.deleteTitle')}
            >
              {deleting ? t('download.deleting') : t('download.deleteBtn', { n: selected.size })}
            </Button>
          )}
        />
      </header>
      <ImageGrid
        className="flex-1 min-h-0"
        contentClassName="p-2"
        items={items}
        selected={selected}
        onSelect={onSelect}
        onActivate={onPreview}
        onPreview={onPreview}
        clickMode="activate"
        ariaLabel={t('download.gridLabel')}
        emptyHint={emptyHint}
      />
    </Card>
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
  count: string
  setCount: (n: string) => void
  maxCount: number
  action: 'estimate' | 'start' | null
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
  action,
  isLive,
  onEstimate,
  onStart,
}: BooruPanelProps) {
  const { t } = useTranslation()
  const disabled = action !== null || isLive
  return (
    <form
      className="flex flex-col gap-2"
      aria-label={t('download.booruFormLabel')}
      onSubmit={(event) => {
        event.preventDefault()
        if (!tag.trim() || disabled) return
        if (estimate && estimate.count !== 0) onStart()
        else onEstimate()
      }}
    >
      <div className="flex flex-wrap items-center gap-related">
        <Select
          aria-label={t('download.sourceLabel')}
          controlSize="sm"
          value={apiSource}
          onChange={(event) => {
            setApiSource(event.target.value as 'gelbooru' | 'danbooru')
          }}
          disabled={disabled}
          className="w-auto"
        >
          <option value="gelbooru">Gelbooru</option>
          <option value="danbooru">Danbooru</option>
        </Select>
        <Input
          aria-label={t('download.tagLabel')}
          controlSize="sm"
          value={tag}
          onChange={(event) => setTag(event.target.value)}
          disabled={disabled}
          placeholder={t('download.tagPlaceholder')}
          className="min-w-[16rem] flex-1"
        />
        <Button
          type={estimate && estimate.count !== 0 ? 'button' : 'submit'}
          variant="secondary"
          size="sm"
          onClick={estimate && estimate.count !== 0 ? onEstimate : undefined}
          loading={action === 'estimate'}
          disabled={disabled || !tag.trim()}
        >
          {action === 'estimate' ? t('download.querying') : t('download.query')}
        </Button>
      </div>

      <div className="flex flex-wrap items-center gap-related border-t border-subtle pt-2 text-sm text-fg-secondary">
        {!estimate ? (
          <span role="status" aria-live="polite" aria-atomic="true">
            {t('download.notQueried')}
          </span>
        ) : (
          <>
            <span role="status" aria-live="polite" aria-atomic="true">
              {t('download.matches')}{' '}
              <strong className={estimate.count === -1 ? 'text-warn' : 'text-accent'}>
                {estimate.count >= 0
                  ? estimate.count.toLocaleString()
                  : t('download.matchesUnknown')}
              </strong>
              {estimate.exclude_tags.length > 0 && (
                <span
                  className="ml-1 text-xs text-fg-tertiary"
                  title={estimate.effective_query}
                  aria-label={`${t('download.exclusionsApplied', { n: estimate.exclude_tags.length })}: ${estimate.effective_query}`}
                >
                  · {t('download.exclusionsApplied', { n: estimate.exclude_tags.length })}
                </span>
              )}
            </span>
            {estimate.count !== 0 && (
              <>
                <label htmlFor="download-count" className="text-xs text-fg-tertiary">
                  {t('download.countLabel')}
                </label>
                <Input
                  id="download-count"
                  type="number"
                  controlSize="sm"
                  mono
                  min={1}
                  max={maxCount}
                  value={count}
                  onChange={(event) => {
                    const next = event.target.value
                    if (next === '') {
                      setCount('')
                      return
                    }
                    setCount(String(Math.min(Number(next) || 1, maxCount)))
                  }}
                  disabled={disabled}
                  className="w-24"
                />
              </>
            )}
          </>
        )}
        <Button
          type="submit"
          variant="primary"
          size="sm"
          loading={action === 'start'}
          disabled={
            disabled ||
            !estimate ||
            estimate.count === 0 ||
            !Number.isInteger(Number(count)) ||
            Number(count) < 1
          }
          className="md:ml-auto"
        >
          {isLive ? t('download.downloading') : t('download.startDownload')}
        </Button>
      </div>
    </form>
  )
}

// ---------------------------------------------------------------------------
// 文件导入：两个来源，共用选择摘要与确认动作
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
  const [serverPath, setServerPath] = useState('')
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [showPathPicker, setShowPathPicker] = useState(false)
  const uploadProgress = useUploadProgress()

  const resetLocalInput = () => {
    setPicked([])
    if (inputRef.current) inputRef.current.value = ''
  }
  const chooseLocal = (files: FileList | null) => {
    if (!files || files.length === 0) return
    setServerPath('')
    setPicked(Array.from(files))
  }
  const chooseServer = (path: string) => {
    resetLocalInput()
    setServerPath(path)
    setShowPathPicker(false)
  }
  const clearSelection = () => {
    resetLocalInput()
    setServerPath('')
  }
  const applyUploadResult = (result: UploadResult) => {
    onUploaded(result)
  }
  const submit = async () => {
    if (picked.length === 0 && !serverPath) return
    const isLocal = picked.length > 0
    setUploading(true)
    if (isLocal) {
      uploadProgress.start(picked.reduce((sum, file) => sum + file.size, 0))
    }
    try {
      const result = isLocal
        ? await api.uploadProjectFiles(pid, picked, uploadProgress.onProgress)
        : await api.uploadProjectFileFromPath(pid, serverPath)
      if (isLocal) uploadProgress.finish()
      applyUploadResult(result)
      clearSelection()
      if (isLocal) {
        window.setTimeout(() => uploadProgress.reset(), 800)
      }
    } catch (error) {
      if (isLocal) uploadProgress.fail(error)
      toast(String(error), 'error')
    } finally {
      setUploading(false)
    }
  }
  const onDrop = (event: React.DragEvent) => {
    event.preventDefault()
    setDragging(false)
    if (!uploading && event.dataTransfer.files?.length) {
      chooseLocal(event.dataTransfer.files)
    }
  }
  const totalBytes = picked.reduce((sum, file) => sum + file.size, 0)
  const fileNames = picked.map((file) => file.name).join(', ')
  const source = picked.length > 0 ? 'local' : serverPath ? 'server' : null
  const selectionSummary = source === 'local'
    ? `${t('download.currentDeviceSource')} · ${t('download.filesSelected', {
      n: picked.length,
      mb: (totalBytes / 1024 / 1024).toFixed(1),
    })}`
    : source === 'server'
      ? `${t('download.serverSource')} · ${serverPath}`
      : t('download.noImportSelection')
  const selectionTitle = source === 'local' && fileNames
    ? `${selectionSummary} · ${fileNames}`
    : selectionSummary
  const importLabel = source === 'local'
    ? t('download.importDeviceAria', { n: picked.length })
    : source === 'server'
      ? t('download.importServerAria', { path: serverPath })
      : t('download.importButton')

  return (
    <div
      className="flex flex-col gap-2"
      onDragOver={(event) => {
        event.preventDefault()
        if (!uploading) setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={UPLOAD_ACCEPT}
        onChange={(event) => chooseLocal(event.target.files)}
        disabled={uploading}
        className="hidden"
        aria-label={t('download.localFilesLabel')}
      />

      <div className="grid grid-cols-2 gap-related">
        <Button
          variant="secondary"
          size="sm"
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
          title={t('download.acceptedFormats')}
          className={dragging ? 'border-accent bg-accent-soft' : ''}
        >
          {t('download.chooseFiles')}
        </Button>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => setShowPathPicker(true)}
          disabled={uploading}
        >
          {t('download.uploadFromPath')}
        </Button>
      </div>

      <div className="flex min-w-0 flex-wrap items-center gap-related border-t border-subtle pt-2">
        <p
          className="m-0 min-w-0 flex-1 truncate text-sm font-medium text-fg-primary"
          role="status"
          aria-live="polite"
          aria-atomic="true"
          title={source ? selectionTitle : undefined}
          data-download-import-summary
        >
          {selectionSummary}
        </p>
        <ActionGroup
          aria-label={t('download.importActionsLabel')}
          secondary={source && (
            <Button variant="ghost" size="sm" onClick={clearSelection} disabled={uploading}>
              {t('common.cancel')}
            </Button>
          )}
          primary={(
            <Button
              variant="primary"
              size="sm"
              onClick={() => void submit()}
              loading={uploading}
              disabled={!source}
              aria-label={importLabel}
              title={source ? importLabel : undefined}
            >
              {uploading ? t('download.importing') : t('download.importButton')}
            </Button>
          )}
        />
      </div>

      {uploadProgress.state.phase !== 'idle' && (
        <UploadProgressBar state={uploadProgress.state} />
      )}
      {showPathPicker && (
        <PathPicker
          dirOnly={false}
          onClose={() => setShowPathPicker(false)}
          onPick={chooseServer}
        />
      )}
    </div>
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
  const added = result.added.length
  return (
    <Alert
      tone={skipped > 0 ? 'warning' : 'success'}
      size="sm"
      title={(
        <span role="status" aria-live="polite" aria-atomic="true">
          {t('download.uploadResultTitle', { added, skipped })}
        </span>
      )}
      action={(
        <Button
          variant="ghost"
          size="xs"
          iconOnly
          onClick={onDismiss}
          aria-label={t('common.close')}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            aria-hidden="true"
          >
            <path d="m6 6 12 12M18 6 6 18" />
          </svg>
        </Button>
      )}
    >
      {skipped > 0 ? (
        <details>
          <summary className="cursor-pointer text-sm">
            {t('download.reviewSkipped', { n: skipped })}
          </summary>
          <ul className="mt-2 max-h-40 overflow-y-auto p-0 font-mono text-xs list-none">
            {result.skipped.map((item, index) => (
              <li key={`${item.name}-${index}`} className="truncate">
                {item.name} <span className="text-fg-tertiary">— {item.reason}</span>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </Alert>
  )
}
