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
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import JobProgress from '../../../components/JobProgress'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

interface AdvancedParams {
  skip_similar: boolean
  aspect_ratio_filter_enabled: boolean
  min_aspect_ratio: number
  max_aspect_ratio: number
  postprocess_method: 'smart' | 'stretch' | 'crop'
  postprocess_max_crop_ratio: number
}

// batch_size 不暴露 — 多 train 子文件夹（5_concept / 1_general 等）共用同一 batch
// 概念在 UI 上意义不大，保持源脚本默认 5。
const ADVANCED_DEFAULTS: AdvancedParams = {
  skip_similar: true,
  aspect_ratio_filter_enabled: false,
  min_aspect_ratio: 0.5,
  max_aspect_ratio: 2.0,
  postprocess_method: 'smart',
  postprocess_max_crop_ratio: 0.1,
}

export default function RegularizationPage() {
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()

  const [reg, setReg] = useState<RegStatus | null>(null)
  const [trainTags, setTrainTags] = useState<RegTagCount[]>([])
  const [excluded, setExcluded] = useState<Set<string>>(new Set())
  const [autoTag, setAutoTag] = useState(true)
  const [apiSource, setApiSource] = useState<'gelbooru' | 'danbooru'>('gelbooru')
  const [advanced, setAdvanced] = useState<AdvancedParams>(ADVANCED_DEFAULTS)
  const [advancedOpen, setAdvancedOpen] = useState(false)

  const [job, setJob] = useState<Job | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const jobIdRef = useRef<number | null>(null)
  jobIdRef.current = job?.id ?? null

  // 预览 modal
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)
  const [previewCaption, setPreviewCaption] = useState<string>('')

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
    } catch {
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

  const startBuild = async (incremental = false) => {
    if (!vid) return
    if (trainImageCount <= 0) {
      toast('train 还没有图片，先去 ① 整理 / ② 下载', 'error')
      return
    }
    const body: RegBuildRequest = {
      excluded_tags: Array.from(excluded),
      auto_tag: autoTag,
      api_source: apiSource,
      incremental,
      ...advanced,
    }
    try {
      const j = await api.startRegBuild(project.id, vid, body)
      setJob(j)
      setLogs([])
      toast(incremental ? `已入队补足 #${j.id}` : `已入队 #${j.id}`, 'success')
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

  // 预览：点击缩略图 → 加载该图 caption → 打开 modal
  const openPreview = useCallback(
    async (idx: number) => {
      if (!reg || !vid) return
      const path = reg.files[idx]
      setPreviewIdx(idx)
      setPreviewCaption('加载中...')
      try {
        const r = await api.getRegCaption(project.id, vid, path)
        setPreviewCaption(r.tags.length ? r.tags.join(', ') : '(无 caption)')
      } catch (e) {
        setPreviewCaption(`加载失败: ${e}`)
      }
    },
    [reg, vid, project.id]
  )

  if (!activeVersion || !vid) {
    return <p className="text-slate-500">请先选择 / 创建一个版本</p>
  }

  return (
    <div className="flex flex-col h-full w-full gap-3">
      <header className="flex items-baseline gap-2 flex-wrap shrink-0">
        <h2 className="text-base font-semibold">⑤ 正则集</h2>
        <span className="text-xs text-slate-500">
          基于 train tag 分布拉「相似但不同」的正则图 · 镜像 train 子文件夹到 reg/
        </span>
      </header>

      <RegStatusBar
        reg={reg}
        onDelete={onDelete}
        onTopUp={() => void startBuild(true)}
        disabled={isLive}
      />

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
          <span className="text-slate-400">
            目标数量{' '}
            <span className="font-mono text-slate-300">{trainImageCount}</span>
            <span className="text-slate-600">（镜像 train）</span>
          </span>
          <span className="text-slate-700">|</span>
          <label className="flex items-center gap-1 cursor-pointer">
            <input
              type="checkbox"
              checked={autoTag}
              onChange={(e) => setAutoTag(e.target.checked)}
            />
            <span className="text-slate-300">拉完后自动 WD14 打标</span>
          </label>
          <button
            onClick={() => setAdvancedOpen((v) => !v)}
            className="text-slate-400 hover:text-slate-200 underline-offset-2 hover:underline"
          >
            {advancedOpen ? '⌃ 进阶' : '⌄ 进阶'}
          </button>
          <span className="flex-1" />
          <button
            onClick={() => void startBuild(false)}
            disabled={isLive || trainImageCount <= 0}
            className="px-3 py-1 rounded bg-cyan-600 hover:bg-cyan-500 text-white disabled:bg-slate-700 disabled:text-slate-500"
          >
            {isLive ? '生成中...' : '开始生成'}
          </button>
        </div>

        {advancedOpen && (
          <AdvancedPanel value={advanced} onChange={setAdvanced} />
        )}

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
        <RegPreview
          pid={project.id}
          vid={vid}
          reg={reg}
          onPick={(idx) => void openPreview(idx)}
        />
      )}

      {previewIdx !== null && reg && reg.files[previewIdx] && (
        <ImagePreviewModal
          src={regOrigUrl(project.id, vid, reg.files[previewIdx])}
          caption={previewCaption}
          hasPrev={previewIdx > 0}
          hasNext={previewIdx < reg.files.length - 1}
          onClose={() => setPreviewIdx(null)}
          onPrev={() =>
            previewIdx > 0 ? void openPreview(previewIdx - 1) : undefined
          }
          onNext={() =>
            previewIdx < reg.files.length - 1
              ? void openPreview(previewIdx + 1)
              : undefined
          }
        />
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
  onTopUp,
  disabled,
}: {
  reg: RegStatus | null
  onDelete: () => void
  onTopUp: () => void
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
  const shortfall = m ? m.target_count - m.actual_count : 0
  const canTopUp = m !== null && shortfall > 0
  return (
    <section className="rounded border border-slate-700 bg-slate-800/30 px-3 py-2 flex flex-col gap-1 shrink-0">
      <div className="flex flex-wrap items-center gap-2 text-xs">
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
            {m.incremental_runs > 0 && (
              <span className="text-slate-500" title="补足跑过的次数">
                · 补足 ×{m.incremental_runs}
              </span>
            )}
          </>
        )}
        <span className="flex-1" />
        {canTopUp && (
          <button
            onClick={onTopUp}
            disabled={disabled}
            className="px-2 py-0.5 rounded text-xs bg-cyan-700/50 hover:bg-cyan-600/60 text-cyan-100 disabled:opacity-40"
            title={`保留已下 ${m!.actual_count} 张，补足 ${shortfall} 张`}
          >
            补足 +{shortfall}
          </button>
        )}
        <button
          onClick={onDelete}
          disabled={disabled}
          className="px-2 py-0.5 rounded text-xs bg-red-800/40 hover:bg-red-800/60 text-red-200 disabled:opacity-40"
        >
          清空
        </button>
      </div>

      {m && (
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
          <span>分辨率聚类：</span>
          {m.postprocess_clusters !== null ? (
            <span
              className="text-slate-300"
              title={`方法 ${m.postprocess_method}，max_crop ${m.postprocess_max_crop_ratio}`}
            >
              {m.postprocess_clusters} 类（{m.postprocess_method},{' '}
              max_crop {m.postprocess_max_crop_ratio}）
            </span>
          ) : (
            <span
              className="text-slate-600"
              title="分辨率差异过大或未启用 — 训练靠 bucketing 处理"
            >
              未聚类
            </span>
          )}
        </div>
      )}
    </section>
  )
}

function AdvancedPanel({
  value,
  onChange,
}: {
  value: AdvancedParams
  onChange: (v: AdvancedParams) => void
}) {
  const set = <K extends keyof AdvancedParams>(k: K, v: AdvancedParams[K]) =>
    onChange({ ...value, [k]: v })
  return (
    <div className="rounded border border-slate-700 bg-slate-900/50 p-3 flex flex-col gap-3 text-xs">
      <p className="text-[10px] text-slate-500">
        默认值与原脚本一致；不熟悉就保持默认
      </p>

      {/* 选图 */}
      <Group label="选图">
        <label className="flex items-center gap-1 cursor-pointer">
          <input
            type="checkbox"
            checked={value.skip_similar}
            onChange={(e) => set('skip_similar', e.target.checked)}
          />
          <span
            className="text-slate-300"
            title="候选只取偶数索引，避免相邻相似图（默认 ✓）"
          >
            skip_similar
          </span>
        </label>
      </Group>

      {/* 长宽比过滤 */}
      <Group label="长宽比过滤">
        <label className="flex items-center gap-1 cursor-pointer">
          <input
            type="checkbox"
            checked={value.aspect_ratio_filter_enabled}
            onChange={(e) => set('aspect_ratio_filter_enabled', e.target.checked)}
          />
          <span className="text-slate-300">启用</span>
        </label>
        {value.aspect_ratio_filter_enabled && (
          <>
            <label className="flex items-center gap-1">
              <span className="text-slate-500">min</span>
              <input
                type="number"
                min={0.1}
                max={1}
                step={0.05}
                value={value.min_aspect_ratio}
                onChange={(e) =>
                  set(
                    'min_aspect_ratio',
                    Math.max(0.1, Math.min(1, Number(e.target.value) || 0.5))
                  )
                }
                className="w-16 px-1 py-0.5 rounded bg-slate-950 border border-slate-700"
              />
            </label>
            <label className="flex items-center gap-1">
              <span className="text-slate-500">max</span>
              <input
                type="number"
                min={1}
                max={10}
                step={0.1}
                value={value.max_aspect_ratio}
                onChange={(e) =>
                  set(
                    'max_aspect_ratio',
                    Math.max(1, Math.min(10, Number(e.target.value) || 2))
                  )
                }
                className="w-16 px-1 py-0.5 rounded bg-slate-950 border border-slate-700"
              />
            </label>
            <span className="text-[10px] text-slate-500">
              过滤极端长宽比图（例：0.5–2.0 = 1:2 到 2:1）
            </span>
          </>
        )}
      </Group>

      {/* 后处理 */}
      <Group label="后处理">
        <span className="text-slate-500">方法</span>
        <select
          value={value.postprocess_method}
          onChange={(e) =>
            set('postprocess_method', e.target.value as 'smart' | 'stretch' | 'crop')
          }
          className="px-1 py-0.5 rounded bg-slate-950 border border-slate-700"
        >
          <option value="smart">smart（缩放+居中裁，推荐）</option>
          <option value="stretch">stretch（拉伸，可能变形）</option>
          <option value="crop">crop（先裁后缩）</option>
        </select>
        <label className="flex items-center gap-1">
          <span className="text-slate-500">max_crop</span>
          <input
            type="number"
            min={0.05}
            max={0.5}
            step={0.05}
            value={value.postprocess_max_crop_ratio}
            onChange={(e) =>
              set(
                'postprocess_max_crop_ratio',
                Math.max(0.05, Math.min(0.5, Number(e.target.value) || 0.1))
              )
            }
            className="w-16 px-1 py-0.5 rounded bg-slate-950 border border-slate-700"
            title="单聚类内最大允许裁剪比例（默认 0.1 = 10%）"
          />
        </label>
      </Group>
    </div>
  )
}

function Group({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="flex items-baseline gap-3">
      <span className="text-[10px] text-slate-500 w-20 shrink-0 uppercase tracking-wide">
        {label}
      </span>
      <div className="flex flex-wrap items-center gap-3 flex-1">{children}</div>
    </div>
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
  const [draft, setDraft] = useState('')
  const trainTagSet = useMemo(
    () => new Set(trainTags.map((t) => t.tag)),
    [trainTags]
  )
  // 自定义 = excluded 里那些不在 train top tag 列表里的（含画师等 train 没出现的 tag）
  const customTags = useMemo(
    () => Array.from(excluded).filter((t) => !trainTagSet.has(t)).sort(),
    [excluded, trainTagSet]
  )

  // 与后端 `_normalize_tags` 对齐：小写、空白→下划线、去重。
  const normalize = (raw: string): string =>
    raw.trim().toLowerCase().replace(/\s+/g, '_')

  const addCustom = () => {
    // 支持一次粘多个：逗号 / 空格 / 换行分隔
    const items = draft
      .split(/[,，\n]+/)
      .map(normalize)
      .filter(Boolean)
    if (items.length === 0) return
    for (const tag of items) {
      if (!excluded.has(tag)) onToggle(tag)
    }
    setDraft('')
  }

  const showTrainList = trainTags.length > 0

  return (
    <div className="space-y-2">
      {showTrainList ? (
        <div>
          <p className="text-[11px] text-slate-500 mb-1">
            排除 train top tag（项目特定 tag，例如角色名）：
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
                  {on ? '✕' : '+'} {t.tag}{' '}
                  <span className="opacity-50">×{t.count}</span>
                </button>
              )
            })}
          </div>
        </div>
      ) : (
        <p className="text-xs text-slate-500">
          train 还没有 tag 分布（先打标）。也可以仅靠下方「自定义排除」继续。
        </p>
      )}

      <div>
        <p className="text-[11px] text-slate-500 mb-1">
          自定义排除（train 里没有也行，常用于画师 / 风格类 tag，例如{' '}
          <code className="text-slate-400">artist_foo</code>）：
        </p>
        <div className="flex items-center gap-1.5">
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                addCustom()
              }
            }}
            placeholder="如 artist_foo, sensitive 等；逗号 / 换行分隔可一次加多个"
            className="flex-1 px-2 py-1 rounded bg-slate-950 border border-slate-700 text-xs focus:outline-none focus:border-cyan-500"
          />
          <button
            onClick={addCustom}
            disabled={!draft.trim()}
            className="text-xs px-2.5 py-1 rounded bg-slate-700 hover:bg-slate-600 text-slate-200 disabled:bg-slate-800 disabled:text-slate-500"
          >
            + 添加
          </button>
        </div>
        {customTags.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1.5">
            {customTags.map((t) => (
              <span
                key={t}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded border border-amber-600 bg-amber-700/40 text-amber-200 text-[11px] font-mono"
                title="自定义排除（点 × 移除）"
              >
                {t}
                <button
                  onClick={() => onToggle(t)}
                  className="text-amber-300 hover:text-amber-100"
                  aria-label={`移除 ${t}`}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function RegPreview({
  pid,
  vid,
  reg,
  onPick,
}: {
  pid: number
  vid: number
  reg: RegStatus
  onPick: (idx: number) => void
}) {
  // reg.files 是相对 reg/ 的路径（含子文件夹镜像 train，例如 "5_concept/2001.png"）
  const items = useMemo(
    () =>
      reg.files.map((rel) => {
        const idx = rel.lastIndexOf('/')
        const folder = idx >= 0 ? rel.slice(0, idx) : ''
        const name = idx >= 0 ? rel.slice(idx + 1) : rel
        return {
          name: rel,
          thumbUrl: api.versionThumbUrl(pid, vid, 'reg', name, folder),
        }
      }),
    [reg.files, pid, vid]
  )
  const indexByName = useMemo(() => {
    const m = new Map<string, number>()
    items.forEach((it, i) => m.set(it.name, i))
    return m
  }, [items])
  return (
    <section className="rounded-lg border border-slate-700 bg-slate-800/20 p-2 flex-1 min-h-0 overflow-y-auto">
      <p className="text-[11px] text-slate-500 px-1 pb-1">
        reg/（共 {reg.image_count} 张）— 点击查看大图 + caption
      </p>
      <ImageGrid
        items={items}
        selected={new Set()}
        onSelect={(name) => {
          const i = indexByName.get(name)
          if (i !== undefined) onPick(i)
        }}
        ariaLabel="reg-preview"
      />
    </section>
  )
}

function regOrigUrl(pid: number, vid: number, rel: string): string {
  const idx = rel.lastIndexOf('/')
  const folder = idx >= 0 ? rel.slice(0, idx) : ''
  const name = idx >= 0 ? rel.slice(idx + 1) : rel
  // 768px 预览（与 PP3 alt-hover 同尺寸）
  return api.versionThumbUrl(pid, vid, 'reg', name, folder, 768)
}

function formatAgo(unix: number): string {
  const now = Date.now() / 1000
  const dt = now - unix
  if (dt < 60) return '刚刚'
  if (dt < 3600) return `${Math.floor(dt / 60)} 分钟前`
  if (dt < 86400) return `${Math.floor(dt / 3600)} 小时前`
  return `${Math.floor(dt / 86400)} 天前`
}
