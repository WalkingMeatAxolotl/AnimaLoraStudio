import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useOutletContext } from 'react-router-dom'
import {
  api,
  type Job,
  type ProjectDetail,
  type RegBuildRequest,
  type RegStatus,
  type RegTagCount,
  type Version,
} from '../../../api/client'
import ImageGrid from '../../../components/ImageGrid'
import JobProgress from '../../../components/JobProgress'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

export default function RegularizationPage() {
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()

  const [reg, setReg] = useState<RegStatus | null>(null)
  const [trainTags, setTrainTags] = useState<RegTagCount[]>([])
  const [excluded, setExcluded] = useState<Set<string>>(new Set())
  const [targetCount, setTargetCount] = useState<number | ''>('')
  const [autoTag, setAutoTag] = useState(true)
  const [apiSource, setApiSource] = useState<'gelbooru' | 'danbooru'>('gelbooru')

  const [job, setJob] = useState<Job | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const jobIdRef = useRef<number | null>(null)
  jobIdRef.current = job?.id ?? null

  const vid = activeVersion?.id ?? null

  const refreshReg = useCallback(async () => {
    if (!vid) return
    try {
      const s = await api.getRegStatus(project.id, vid)
      setReg(s)
    } catch (e) {
      toast(`加载 reg 状态失败: ${e}`, 'error')
    }
  }, [project.id, vid, toast])

  const refreshTrainTags = useCallback(async () => {
    if (!vid) return
    try {
      const items = await api.previewRegTags(project.id, vid, 30)
      setTrainTags(items)
    } catch (e) {
      // train 没图也会返回空，不报错
      setTrainTags([])
    }
  }, [project.id, vid])

  useEffect(() => {
    void refreshReg()
    void refreshTrainTags()
  }, [refreshReg, refreshTrainTags])

  useEventStream((evt) => {
    const jid = jobIdRef.current
    if (evt.type === 'job_log_appended' && jid && evt.job_id === jid) {
      setLogs((prev) => [...prev, String(evt.text ?? '')])
    } else if (evt.type === 'job_state_changed' && jid && evt.job_id === jid) {
      void api.getJob(jid).then(setJob).catch(() => {})
      if (evt.status === 'done' || evt.status === 'failed' || evt.status === 'canceled') {
        void refreshReg()
        void reload()
      }
    }
  })

  const trainImageCount = activeVersion?.stats?.train_image_count ?? 0
  const isLive = job?.status === 'running' || job?.status === 'pending'

  const toggleTag = (tag: string) => {
    setExcluded((prev) => {
      const next = new Set(prev)
      if (next.has(tag)) next.delete(tag)
      else next.add(tag)
      return next
    })
  }

  const startBuild = async () => {
    if (!vid) return
    if (trainImageCount <= 0) {
      toast('train 还没有图片，先去 ① 整理 / ② 下载', 'error')
      return
    }
    const body: RegBuildRequest = {
      target_count: targetCount === '' ? null : Number(targetCount),
      excluded_tags: Array.from(excluded),
      auto_tag: autoTag,
      api_source: apiSource,
    }
    try {
      const j = await api.startRegBuild(project.id, vid, body)
      setJob(j)
      setLogs([])
      toast(`已入队 #${j.id}`, 'success')
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const onDelete = async () => {
    if (!vid) return
    if (!confirm('删除当前 reg 集？这是不可恢复的（meta + 所有图片都会清掉）。')) return
    try {
      await api.deleteReg(project.id, vid)
      toast('已删除', 'success')
      setReg(null)
      void refreshReg()
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  if (!activeVersion || !vid) {
    return <p className="text-slate-500">请先选择 / 创建一个版本</p>
  }

  return (
    <div className="flex flex-col h-full w-full gap-3">
      <header className="flex items-baseline gap-2 flex-wrap shrink-0">
        <h2 className="text-base font-semibold">⑤ 正则集</h2>
        <span className="text-xs text-slate-500">
          基于 train tag 分布拉「相似但不同」的正则图 · 落到 reg/1_general/
        </span>
      </header>

      <RegStatusBar reg={reg} onDelete={onDelete} disabled={isLive} />

      <section className="rounded-lg border border-slate-700 bg-slate-800/40 p-3 flex flex-col gap-3 shrink-0">
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <span className="text-slate-400">来源</span>
          <select
            value={apiSource}
            onChange={(e) => setApiSource(e.target.value as 'gelbooru' | 'danbooru')}
            className="px-2 py-1 rounded bg-slate-950 border border-slate-700"
          >
            <option value="gelbooru">Gelbooru</option>
            <option value="danbooru">Danbooru</option>
          </select>
          <span className="text-slate-700">|</span>
          <span className="text-slate-400">目标数量</span>
          <input
            type="number"
            min={1}
            value={targetCount}
            onChange={(e) =>
              setTargetCount(e.target.value === '' ? '' : Number(e.target.value))
            }
            placeholder={`(默认 train 总数 = ${trainImageCount})`}
            className="px-2 py-1 rounded bg-slate-950 border border-slate-700 w-44 text-xs"
          />
          <span className="text-slate-700">|</span>
          <label className="flex items-center gap-1 cursor-pointer">
            <input
              type="checkbox"
              checked={autoTag}
              onChange={(e) => setAutoTag(e.target.checked)}
            />
            <span className="text-slate-300">拉完后自动 WD14 打标</span>
          </label>
          <span className="flex-1" />
          <button
            onClick={startBuild}
            disabled={isLive || trainImageCount <= 0}
            className="px-3 py-1 rounded bg-cyan-600 hover:bg-cyan-500 text-white disabled:bg-slate-700 disabled:text-slate-500"
          >
            {isLive ? '生成中...' : '开始生成'}
          </button>
        </div>

        <ExcludeTagsPicker
          trainTags={trainTags}
          excluded={excluded}
          onToggle={toggleTag}
        />
      </section>

      {job && (
        <JobProgress
          job={job}
          logs={logs}
          onCancel={async () => {
            try {
              await api.cancelJob(job.id)
              toast('已取消', 'success')
            } catch (e) {
              toast(String(e), 'error')
            }
          }}
        />
      )}

      {reg && reg.image_count > 0 && (
        <RegPreview pid={project.id} vid={vid} reg={reg} />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 子组件
// ---------------------------------------------------------------------------

function RegStatusBar({
  reg,
  onDelete,
  disabled,
}: {
  reg: RegStatus | null
  onDelete: () => void
  disabled: boolean
}) {
  if (!reg) {
    return (
      <section className="rounded border border-slate-700 bg-slate-800/30 px-3 py-2 text-xs text-slate-400 shrink-0">
        加载中...
      </section>
    )
  }
  if (!reg.exists) {
    return (
      <section className="rounded border border-slate-700 bg-slate-800/30 px-3 py-2 text-xs text-slate-400 shrink-0">
        当前版本 reg 集：<span className="text-slate-500">不存在</span>
      </section>
    )
  }
  const m = reg.meta
  const ago = m ? formatAgo(m.generated_at) : '?'
  return (
    <section className="rounded border border-slate-700 bg-slate-800/30 px-3 py-2 flex flex-wrap items-center gap-2 text-xs shrink-0">
      <span className="text-slate-300">
        reg 集存在：
        <span className="font-mono text-emerald-300">{reg.image_count} 张</span>
      </span>
      {m && (
        <>
          <span className="text-slate-700">·</span>
          <span className="text-slate-400">
            target {m.actual_count}/{m.target_count}
          </span>
          <span className="text-slate-700">·</span>
          <span className="text-slate-400">{m.api_source}</span>
          <span className="text-slate-700">·</span>
          <span className="text-slate-400">
            auto-tag:{' '}
            <span className={m.auto_tagged ? 'text-emerald-300' : 'text-slate-500'}>
              {m.auto_tagged ? '✓' : '×'}
            </span>
          </span>
          <span className="text-slate-700">·</span>
          <span className="text-slate-500">{ago}</span>
          {m.failed_tags.length > 0 && (
            <span
              className="text-amber-300"
              title={`搜索失败的 tag: ${m.failed_tags.join(', ')}`}
            >
              · {m.failed_tags.length} 失败 tag
            </span>
          )}
        </>
      )}
      <span className="flex-1" />
      <button
        onClick={onDelete}
        disabled={disabled}
        className="px-2 py-0.5 rounded text-xs bg-red-800/40 hover:bg-red-800/60 text-red-200 disabled:opacity-40"
      >
        清空
      </button>
    </section>
  )
}

function ExcludeTagsPicker({
  trainTags,
  excluded,
  onToggle,
}: {
  trainTags: RegTagCount[]
  excluded: Set<string>
  onToggle: (tag: string) => void
}) {
  if (trainTags.length === 0) {
    return (
      <p className="text-xs text-slate-500">
        train 还没有 tag 分布（先打标）。可以直接开始生成，会按 booru 默认搜索。
      </p>
    )
  }
  return (
    <div>
      <p className="text-[11px] text-slate-500 mb-1">
        排除 train top tag（项目特定 tag 不应作为搜索条件，例如角色名）：
      </p>
      <div className="flex flex-wrap gap-1">
        {trainTags.map((t) => {
          const on = excluded.has(t.tag)
          return (
            <button
              key={t.tag}
              onClick={() => onToggle(t.tag)}
              className={
                'px-2 py-0.5 rounded border text-[11px] font-mono transition ' +
                (on
                  ? 'bg-amber-700/40 border-amber-600 text-amber-200'
                  : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-slate-500')
              }
              title={on ? '点击取消排除' : '点击加入排除'}
            >
              {on ? '✕' : '+'} {t.tag} <span className="opacity-50">×{t.count}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function RegPreview({
  pid,
  vid,
  reg,
}: {
  pid: number
  vid: number
  reg: RegStatus
}) {
  // reg.files 是相对 reg/1_general/ 的路径，可能含子文件夹（镜像 train）
  const items = useMemo(
    () =>
      reg.files.map((rel) => {
        // rel 形如 "1_data/2001.png" 或 "2001.png"
        const idx = rel.lastIndexOf('/')
        const folder = idx >= 0 ? rel.slice(0, idx) : ''
        const name = idx >= 0 ? rel.slice(idx + 1) : rel
        const folderForUrl = folder ? `1_general/${folder}` : '1_general'
        return {
          name: rel,
          thumbUrl: api.versionThumbUrl(pid, vid, 'reg', name, folderForUrl),
        }
      }),
    [reg.files, pid, vid]
  )
  return (
    <section className="rounded-lg border border-slate-700 bg-slate-800/20 p-2 flex-1 min-h-0 overflow-y-auto">
      <p className="text-[11px] text-slate-500 px-1 pb-1">
        reg/1_general/（共 {reg.image_count} 张）
      </p>
      <ImageGrid
        items={items}
        selected={new Set()}
        onSelect={() => {}}
        ariaLabel="reg-preview"
      />
    </section>
  )
}

function formatAgo(unix: number): string {
  const now = Date.now() / 1000
  const dt = now - unix
  if (dt < 60) return '刚刚'
  if (dt < 3600) return `${Math.floor(dt / 60)} 分钟前`
  if (dt < 86400) return `${Math.floor(dt / 3600)} 小时前`
  return `${Math.floor(dt / 86400)} 天前`
}
