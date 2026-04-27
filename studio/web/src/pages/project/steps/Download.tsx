import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useOutletContext } from 'react-router-dom'
import {
  api,
  type DownloadFile,
  type Job,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import FileList from '../../../components/FileList'
import JobProgress from '../../../components/JobProgress'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'

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
  count: number  // -1 表示未知
}

export default function DownloadPage() {
  const { project, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const [job, setJob] = useState<Job | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [files, setFiles] = useState<DownloadFile[]>([])
  const [tag, setTag] = useState('')
  const [apiSource, setApiSource] = useState<'gelbooru' | 'danbooru'>(
    'gelbooru'
  )
  const [estimate, setEstimate] = useState<Estimate | null>(null)
  const [count, setCount] = useState<number>(20)
  const [busy, setBusy] = useState(false)

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

  // SSE 实时推
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

  // 当 tag / api_source 变化，丢掉旧估算，强制用户重新查
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
      // 默认 count：未知 → 20；已知 → min(全部, 200) 作为合理默认
      if (r.count > 0) {
        setCount(Math.min(r.count, 200))
      } else if (r.count === 0) {
        setCount(0)
      } else {
        setCount(20)
      }
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const start = async () => {
    if (!estimate) return
    if (estimate.count === 0) {
      toast('查询结果为 0，没有可下载的图', 'error')
      return
    }
    if (count < 1) {
      toast('count 必须 >= 1', 'error')
      return
    }
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
  // 已知总数时：count 上限 = 估算值；未知时不限
  const maxCount = estimate && estimate.count > 0 ? estimate.count : 5000

  return (
    <div className="space-y-4 max-w-3xl">
      <header>
        <h2 className="text-lg font-semibold">① 下载数据</h2>
        <p className="text-sm text-slate-400">
          来源：Gelbooru / Danbooru。下载到{' '}
          <code className="text-slate-300">{project.slug}/download/</code>，
          所有版本共享。
        </p>
      </header>

      <div className="text-xs text-slate-500">
        全局 exclude tag、Gelbooru / Danbooru 凭据请在{' '}
        <Link to="/tools/settings" className="text-cyan-400 hover:underline">
          设置
        </Link>{' '}
        配置。
      </div>

      {/* 第一步：tag + 渠道 + 查询 */}
      <section className="rounded-lg border border-slate-700 bg-slate-800/40 p-4 space-y-3">
        <div className="grid grid-cols-[120px_1fr] gap-3 items-center">
          <label className="text-xs text-slate-400 font-mono">tag</label>
          <input
            value={tag}
            onChange={(e) => setTag(e.target.value)}
            disabled={busy || isLive}
            placeholder="例：character_x rating:safe"
            className="px-2 py-1.5 rounded bg-slate-950 border border-slate-700 text-sm focus:outline-none focus:border-cyan-500"
          />
          <label className="text-xs text-slate-400 font-mono">api_source</label>
          <select
            value={apiSource}
            onChange={(e) =>
              setApiSource(e.target.value as 'gelbooru' | 'danbooru')
            }
            disabled={busy || isLive}
            className="px-2 py-1.5 rounded bg-slate-950 border border-slate-700 text-sm w-40 focus:outline-none focus:border-cyan-500"
          >
            <option value="gelbooru">Gelbooru</option>
            <option value="danbooru">Danbooru</option>
          </select>
        </div>
        <div>
          <button
            onClick={doEstimate}
            disabled={busy || isLive || !tag.trim()}
            className="px-3 py-1.5 rounded text-sm bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-500"
          >
            {busy ? '查询中...' : '查询数量'}
          </button>
        </div>
      </section>

      {/* 第二步：估算结果 + 选择 count + 启动 */}
      {estimate && (
        <section className="rounded-lg border border-cyan-800 bg-cyan-950/20 p-4 space-y-3">
          <div className="text-sm text-slate-200">
            匹配{' '}
            {estimate.count >= 0 ? (
              <strong className="text-cyan-300">{estimate.count}</strong>
            ) : (
              <strong className="text-amber-300">数量未知</strong>
            )}{' '}
            张
            <span className="text-slate-500 text-xs ml-2">
              query: <code>{estimate.effective_query}</code>
            </span>
          </div>
          {estimate.exclude_tags.length > 0 && (
            <div className="text-xs text-slate-400">
              已应用全局 exclude:{' '}
              <code className="text-slate-300">
                {estimate.exclude_tags.join(', ')}
              </code>
            </div>
          )}

          {estimate.count !== 0 && (
            <>
              <div className="grid grid-cols-[120px_1fr] gap-3 items-center">
                <label className="text-xs text-slate-400 font-mono">
                  count
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min={1}
                    max={maxCount}
                    value={count}
                    onChange={(e) =>
                      setCount(Math.min(Number(e.target.value) || 1, maxCount))
                    }
                    disabled={busy || isLive}
                    className="px-2 py-1.5 rounded bg-slate-950 border border-slate-700 text-sm w-32 focus:outline-none focus:border-cyan-500"
                  />
                  {estimate.count > 0 && (
                    <button
                      onClick={() => setCount(estimate.count)}
                      disabled={busy || isLive}
                      className="text-xs px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 text-slate-200"
                    >
                      全部 ({estimate.count})
                    </button>
                  )}
                </div>
              </div>
              <div>
                <button
                  onClick={start}
                  disabled={busy || isLive || count < 1}
                  className="px-3 py-1.5 rounded text-sm bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:text-slate-500"
                >
                  {isLive ? '下载中...' : `开始下载 ${count} 张`}
                </button>
              </div>
            </>
          )}
        </section>
      )}

      {job && <JobProgress job={job} logs={logs} onCancel={cancel} />}

      <section>
        <h3 className="text-sm font-semibold text-slate-300 mb-2">
          已下载 ({files.length})
        </h3>
        <FileList pid={project.id} items={files} />
      </section>
    </div>
  )
}
