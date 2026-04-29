import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useOutletContext } from 'react-router-dom'
import {
  api,
  type DownloadFile,
  type Job,
  type ProjectDetail,
  type UploadResult,
  type Version,
} from '../../../api/client'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'

const UPLOAD_ACCEPT =
  '.jpg,.jpeg,.png,.zip,image/jpeg,image/png,application/zip'

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

const STATUS_COLOR: Record<Job['status'], string> = {
  pending: 'bg-slate-700/60 text-slate-300',
  running: 'bg-amber-700/40 text-amber-200',
  done: 'bg-emerald-700/40 text-emerald-200',
  failed: 'bg-red-800/50 text-red-200',
  canceled: 'bg-slate-700/60 text-slate-300',
}

// 信息密度优先：每个 panel 紧凑成单/双 inline 行；已下载 grid 占主区域。
export default function DownloadPage() {
  const { project, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const [job, setJob] = useState<Job | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [files, setFiles] = useState<DownloadFile[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [anchor, setAnchor] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [tag, setTag] = useState('')
  const [apiSource, setApiSource] = useState<'gelbooru' | 'danbooru'>(
    'gelbooru'
  )
  const [estimate, setEstimate] = useState<Estimate | null>(null)
  const [count, setCount] = useState<number>(20)
  const [busy, setBusy] = useState(false)
  const [lastUpload, setLastUpload] = useState<UploadResult | null>(null)

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
    }
  })

  useEffect(() => {
    setEstimate(null)
  }, [tag, apiSource])

  const doEstimate = async () => {
    if (!tag.trim()) {
      toast('tag 不能为空', 'error')
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
      return toast('查询结果为 0，没有可下载的图', 'error')
    if (count < 1) return toast('count 必须 >= 1', 'error')
    setBusy(true)
    try {
      const j = await api.startDownload(project.id, {
        tag,
        count,
        api_source: apiSource,
      })
      setJob(j)
      setLogs([])
      toast(`开始下载 #${j.id}`, 'success')
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
      toast('已取消', 'success')
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const isLive = job?.status === 'running' || job?.status === 'pending'
  const maxCount = estimate && estimate.count > 0 ? estimate.count : 5000

  return (
    <div className="flex flex-col h-full gap-2 min-h-0">
      {/* 单行标题 */}
      <header className="flex items-baseline gap-2 flex-wrap shrink-0">
        <h2 className="text-base font-semibold">① 下载</h2>
        <span className="text-xs text-slate-500">
          Booru 抓取 + 本地上传，共享{' '}
          <code className="text-slate-400">{project.slug}/download/</code> ·{' '}
          <Link to="/tools/settings" className="text-cyan-400 hover:underline">
            设置
          </Link>{' '}
          配置 exclude / 凭据
        </span>
      </header>

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

      {/* 状态条：仅在有 job / 上次上传结果时出现，details 折叠 */}
      {(job || lastUpload) && (
        <div className="flex flex-col gap-1.5 shrink-0">
          {job && (
            <JobStrip
              job={job}
              logs={logs}
              onCancel={isLive ? cancel : undefined}
            />
          )}
          {lastUpload && (
            <UploadResultStrip
              result={lastUpload}
              onDismiss={() => setLastUpload(null)}
            />
          )}
        </div>
      )}

      {/* 已下载 grid — 占满剩余高度，支持多选 + 删除 */}
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
        onSelectAll={() => setSelected(new Set(files.map((f) => f.name)))}
        onClear={() => {
          setSelected(new Set())
          setAnchor(null)
        }}
        onDelete={async () => {
          if (selected.size === 0) return
          if (
            !window.confirm(
              `从 download/ 删除 ${selected.size} 张图片（含同名 caption metadata）？\n操作不可恢复。`
            )
          )
            return
          setDeleting(true)
          try {
            const r = await api.deleteProjectFiles(
              project.id,
              Array.from(selected)
            )
            toast(
              `已删除 ${r.deleted.length} 张${
                r.missing.length ? ` · 跳过 ${r.missing.length} 张（不存在）` : ''
              }`,
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
  onSelectAll: () => void
  onClear: () => void
  onDelete: () => void | Promise<void>
}) {
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
    <section className="rounded-lg border border-slate-700 bg-slate-800/30 overflow-hidden flex flex-col flex-1 min-h-0">
      <header className="px-3 py-1.5 border-b border-slate-700 flex items-center gap-2 shrink-0 text-xs">
        <h3 className="font-semibold text-slate-200">已下载</h3>
        <span className="text-slate-500">{files.length} 张</span>
        {selected.size > 0 && (
          <span className="text-cyan-300">· 已选 {selected.size}</span>
        )}
        <span className="flex-1" />
        <button
          onClick={onSelectAll}
          disabled={files.length === 0 || deleting}
          className="px-2 py-0.5 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-50"
        >
          全选
        </button>
        <button
          onClick={onClear}
          disabled={selected.size === 0 || deleting}
          className="px-2 py-0.5 rounded text-slate-400 hover:text-slate-200 disabled:opacity-30"
        >
          清空
        </button>
        <button
          onClick={() => void onDelete()}
          disabled={selected.size === 0 || deleting}
          className="px-2 py-0.5 rounded bg-red-700/80 hover:bg-red-700 text-red-100 disabled:opacity-40"
          title="删除选中的图片 + 同名 caption metadata"
        >
          {deleting ? '删除中...' : `🗑 删除 ${selected.size}`}
        </button>
      </header>
      <div className="flex-1 min-h-0 overflow-y-auto p-2">
        <ImageGrid
          items={items}
          selected={selected}
          onSelect={onSelect}
          ariaLabel="downloaded-grid"
          emptyHint="还没有图片 — 用上方 Booru 抓取或本地上传"
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
  const disabled = busy || isLive
  return (
    <section className="rounded-lg border border-slate-700 bg-slate-800/40 px-3 py-2 space-y-1.5">
      <PanelTitle accent="cyan">Booru 抓取</PanelTitle>
      <div className="flex items-center gap-1.5">
        <select
          value={apiSource}
          onChange={(e) =>
            setApiSource(e.target.value as 'gelbooru' | 'danbooru')
          }
          disabled={disabled}
          className="px-2 py-1 rounded bg-slate-950 border border-slate-700 text-xs w-24"
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
          placeholder="tag (如 character_x rating:safe)"
          className="flex-1 px-2 py-1 rounded bg-slate-950 border border-slate-700 text-xs focus:outline-none focus:border-cyan-500"
        />
        <button
          onClick={onEstimate}
          disabled={disabled || !tag.trim()}
          className="text-xs px-2.5 py-1 rounded bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-500"
        >
          {busy && !estimate ? '查询中...' : '查询'}
        </button>
      </div>
      {estimate && (
        <div className="flex items-center gap-1.5 flex-wrap text-xs">
          <span className="text-slate-300">
            匹配{' '}
            {estimate.count >= 0 ? (
              <strong className="text-cyan-300">{estimate.count}</strong>
            ) : (
              <strong className="text-amber-300">未知</strong>
            )}
          </span>
          {estimate.count !== 0 && (
            <>
              <span className="text-slate-500">·</span>
              <span className="text-slate-500">count</span>
              <input
                type="number"
                min={1}
                max={maxCount}
                value={count}
                onChange={(e) =>
                  setCount(Math.min(Number(e.target.value) || 1, maxCount))
                }
                disabled={disabled}
                className="px-1.5 py-0.5 rounded bg-slate-950 border border-slate-700 text-xs w-20 focus:outline-none focus:border-cyan-500"
              />
              {estimate.count > 0 && (
                <button
                  onClick={() => setCount(estimate.count)}
                  disabled={disabled}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-200"
                >
                  全部 {estimate.count}
                </button>
              )}
              <button
                onClick={onStart}
                disabled={disabled || count < 1}
                className="ml-auto text-xs px-2.5 py-1 rounded bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:text-slate-500"
              >
                {isLive ? '下载中...' : `开始 ${count}`}
              </button>
            </>
          )}
          <span
            className="basis-full text-[10px] text-slate-500 truncate"
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
  const { toast } = useToast()
  const inputRef = useRef<HTMLInputElement>(null)
  const [picked, setPicked] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)

  const choose = (fl: FileList | null) => {
    if (!fl || fl.length === 0) return
    setPicked(Array.from(fl))
  }
  const reset = () => {
    setPicked([])
    if (inputRef.current) inputRef.current.value = ''
  }
  const submit = async () => {
    if (picked.length === 0) return
    setUploading(true)
    try {
      const r = await api.uploadProjectFiles(pid, picked)
      const skipped = r.skipped.length
      toast(
        `已添加 ${r.added.length} 张${skipped ? ` · 跳过 ${skipped}` : ''}`,
        r.added.length > 0 ? 'success' : 'error'
      )
      reset()
      onUploaded(r)
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
    <section className="rounded-lg border border-slate-700 bg-slate-800/40 px-3 py-2 space-y-1.5">
      <PanelTitle accent="emerald">本地上传</PanelTitle>
      <label
        onDragOver={(e) => {
          e.preventDefault()
          if (!uploading) setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={
          'flex items-center gap-2 cursor-pointer rounded border border-dashed px-2.5 py-1.5 text-xs transition-colors ' +
          (dragging
            ? 'border-emerald-500 bg-emerald-950/30 text-emerald-200'
            : 'border-slate-600 hover:border-slate-400 text-slate-300')
        }
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
        <span className="font-medium">点击选择 / 拖入</span>
        <span className="text-slate-500">
          · .jpg / .png / .zip(自动解压)
        </span>
        <span className="flex-1" />
        {picked.length > 0 && (
          <span className="text-emerald-300">
            已选 {picked.length} · {(totalBytes / 1024 / 1024).toFixed(1)} MB
          </span>
        )}
      </label>
      {picked.length > 0 && (
        <div className="flex items-center gap-1.5 text-xs">
          <button
            onClick={submit}
            disabled={uploading}
            className="text-xs px-2.5 py-1 rounded bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500"
          >
            {uploading ? '上传中...' : `上传 ${picked.length}`}
          </button>
          <button
            onClick={reset}
            disabled={uploading}
            className="text-[11px] px-2 py-1 rounded text-slate-400 hover:text-slate-200"
          >
            清空
          </button>
          <span
            className="ml-1 text-[10px] text-slate-500 truncate min-w-0 flex-1"
            title={fileNames}
          >
            {fileNames}
          </span>
        </div>
      )}
    </section>
  )
}

// ---------------------------------------------------------------------------
// 状态条 — 1 行 summary，details 折叠完整内容
// ---------------------------------------------------------------------------

function JobStrip({
  job,
  logs,
  onCancel,
}: {
  job: Job
  logs: string[]
  onCancel?: () => void
}) {
  const elapsed =
    job.started_at && (job.finished_at ?? Date.now() / 1000) - job.started_at
  const isLive = job.status === 'running' || job.status === 'pending'
  const lastLine = logs[logs.length - 1] ?? ''
  return (
    <details
      open={isLive}
      className="group rounded-lg border border-slate-700 bg-slate-800/40 overflow-hidden"
    >
      <summary className="px-3 py-1.5 cursor-pointer flex items-center gap-2 text-xs select-none list-none">
        <span className="text-slate-500 group-open:rotate-90 transition-transform inline-block w-3">
          ▸
        </span>
        <span
          className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${STATUS_COLOR[job.status]}`}
        >
          {job.status}
        </span>
        <span className="text-slate-400 font-mono">job #{job.id}</span>
        {elapsed && elapsed > 0 && (
          <span className="text-slate-500">· {Math.round(elapsed)}s</span>
        )}
        <span className="text-slate-300 truncate flex-1 min-w-0 font-mono text-[11px]">
          {lastLine}
        </span>
        {isLive && onCancel && (
          <button
            onClick={(e) => {
              e.preventDefault()
              onCancel()
            }}
            className="text-[11px] px-2 py-0.5 rounded text-red-300 hover:text-red-200 hover:bg-red-900/30"
          >
            取消
          </button>
        )}
      </summary>
      <pre className="p-2.5 text-[11px] font-mono text-slate-300 bg-slate-950/40 max-h-56 overflow-y-auto whitespace-pre-wrap border-t border-slate-700">
        {logs.length === 0 ? '(等待日志...)' : logs.slice(-1000).join('\n')}
      </pre>
    </details>
  )
}

function UploadResultStrip({
  result,
  onDismiss,
}: {
  result: UploadResult
  onDismiss: () => void
}) {
  const skipped = result.skipped.length
  const ok = result.added.length
  return (
    <details className="group rounded-lg border border-slate-700 bg-slate-800/40 overflow-hidden">
      <summary className="px-3 py-1.5 cursor-pointer flex items-center gap-2 text-xs select-none list-none">
        <span className="text-slate-500 group-open:rotate-90 transition-transform inline-block w-3">
          ▸
        </span>
        <span className="text-[10px] px-1.5 py-0.5 rounded font-mono bg-emerald-700/40 text-emerald-200">
          upload
        </span>
        <span className="text-slate-300">
          添加 <strong className="text-emerald-300">{ok}</strong>
          {skipped > 0 && (
            <>
              {' · '}跳过 <strong className="text-amber-300">{skipped}</strong>
            </>
          )}
        </span>
        <span className="flex-1" />
        <button
          onClick={(e) => {
            e.preventDefault()
            onDismiss()
          }}
          className="text-[11px] px-1.5 py-0.5 rounded text-slate-400 hover:text-slate-200"
          title="关闭"
        >
          ×
        </button>
      </summary>
      {skipped > 0 ? (
        <ul className="p-2.5 text-[11px] font-mono text-amber-300/90 max-h-40 overflow-y-auto space-y-0.5 border-t border-slate-700">
          {result.skipped.map((s, i) => (
            <li key={`${s.name}-${i}`} className="truncate">
              {s.name} <span className="text-slate-500">— {s.reason}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="p-2.5 text-[11px] text-slate-500 border-t border-slate-700">
          全部成功，无跳过。
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
  const dot = accent === 'cyan' ? 'bg-cyan-400' : 'bg-emerald-400'
  return (
    <h3 className="text-[10px] font-semibold text-slate-400 flex items-center gap-1.5 uppercase tracking-wider">
      <span className={`inline-block w-1.5 h-1.5 rounded-full ${dot}`} />
      {children}
    </h3>
  )
}
