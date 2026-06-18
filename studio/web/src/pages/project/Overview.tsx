/** 项目详情页 — Design v2 实装（pvt-detail-v2.jsx）
 *
 *  顶部：Identity strip（glyph + title/version/status + meta caption）
 *  → 横向 VersionRail（pill 行）
 *  → 5 状态 StatusBanner（preparing / training / completed / failed / canceled）
 *  → Tabs (详情 / Tasks / Output)
 *  → 详情 = 2+3 不对称 grid（训练集 hero + 标签分布 hero / 分辨率 + 长宽比 + 正则集）
 *
 *  TopBar (面包屑 + sys stats) 不实装 —— 已被 sidebar/全局区覆盖。
 *  Live 训练进度 (step/total/ETA) 不实装 —— 需 SSE/monitor state 整合，留 follow-up。
 *  "复制配置开新版本" / "调小 batch 重训" 需新后端 API，渲染为占位按钮 toast 提示。
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate, useOutletContext } from 'react-router-dom'
import {
  api,
  type CurationView,
  type ProjectDetail,
  type Task,
  type TaskOutputs,
  type Version,
  type VersionPhase,
  type VersionStatus,
} from '../../api/client'
import VersionStatusBadge from '../../components/VersionStatusBadge'
import BarHistogram from '../../components/BarHistogram'
import { TranslatedTag } from '../../components/tagDisplay/TranslatedTag'
import ImageGrid, { type ImageGridItem } from '../../components/ImageGrid'
import ImagePreviewModal from '../../components/ImagePreviewModal'
import { OutputsTab } from '../QueueDetail'
import { arBucket } from '../../lib/aspectRatio'
import { computePixelHist } from '../../lib/pixelBins'
import { useProjectCtx } from '../../context/ProjectContext'
import { useEventStream } from '../../lib/useEventStream'
import { useToast } from '../../components/Toast'

type OverviewTab = 'details' | 'tasks' | 'output'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
  onCreateVersion: (forkFromVid?: number) => void
  creatingVersionBusy: boolean
}

// ── ProjectGlyph (slug-deterministic gradient block) ─────────────────────

function ProjectGlyph({ slug, size = 52 }: { slug: string; size?: number }) {
  const h = [...slug].reduce((a, c) => a + c.charCodeAt(0), 0) % 360
  return (
    <div
      style={{
        width: size, height: size, flex: 'none',
        borderRadius: 'var(--r-lg)',
        background: `linear-gradient(135deg, oklch(0.58 0.16 ${h}), oklch(0.42 0.10 ${(h + 50) % 360}))`,
        display: 'grid', placeItems: 'center',
        fontFamily: 'var(--font-mono)',
        fontSize: size * 0.36, fontWeight: 700, letterSpacing: '-0.04em',
        color: 'rgba(255,255,255,0.95)',
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.15), inset 0 -1px 0 rgba(0,0,0,0.20), 0 1px 2px rgba(0,0,0,0.3)',
      }}
    >{slug.slice(0, 2).toUpperCase()}</div>
  )
}

// ── Identity strip (replaces old big title + slug + 3-stat metadata) ─────

function Identity({
  project, version, totalVersions,
}: {
  project: ProjectDetail
  version: Version | null
  totalVersions: number
}) {
  const { t } = useTranslation()
  const created = project.created_at
    ? new Date(project.created_at * 1000).toLocaleDateString()
    : '—'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
      <ProjectGlyph slug={project.slug} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <h1 style={{
          margin: 0, fontSize: 'var(--t-2xl)', fontWeight: 600,
          letterSpacing: '-0.025em', lineHeight: 1.1,
          display: 'flex', alignItems: 'baseline', gap: 8, flexWrap: 'wrap',
        }}>
          <span>{project.title}</span>
          {version && (
            <>
              <span style={{ color: 'var(--fg-tertiary)', fontWeight: 300, fontSize: 'var(--t-xl)' }}>/</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--t-xl)', color: 'var(--accent)' }}>{version.label}</span>
              <VersionStatusBadge status={version.status} />
            </>
          )}
        </h1>
        <div style={{
          marginTop: 6, display: 'flex', alignItems: 'center', gap: 10,
          fontFamily: 'var(--font-mono)', fontSize: 'var(--t-xs)', color: 'var(--fg-tertiary)',
          flexWrap: 'wrap',
        }}>
          <span><span style={{ color: 'var(--fg-secondary)' }}>{project.download_image_count ?? 0}</span> {t('overview.identity.datasetSuffix')}</span>
          <span>·</span>
          <span><span style={{ color: 'var(--fg-secondary)' }}>{totalVersions}</span> {t('overview.identity.versionSuffix')}</span>
          <span>·</span>
          <span>{t('overview.identity.createdLabel')} {created}</span>
        </div>
      </div>
    </div>
  )
}

// ── VersionRail (horizontal pill row) ────────────────────────────────────

function StatusDotMini({ status }: { status: VersionStatus }) {
  const cmap: Record<VersionStatus, string> = {
    preparing: 'var(--warn)',
    training:  'var(--accent)',
    completed: 'var(--ok)',
    failed:    'var(--err)',
    canceled:  'var(--fg-disabled)',
  }
  const running = status === 'training'
  return (
    <span
      style={{
        width: 7, height: 7, borderRadius: '50%',
        background: cmap[status] ?? 'var(--fg-disabled)',
        animation: running ? 'pulse 1.6s infinite' : 'none',
        flexShrink: 0,
      }}
    />
  )
}

const STATUS_LABEL: Record<VersionStatus, string> = {
  preparing: 'versionStatus.preparing',
  training:  'versionStatus.training',
  completed: 'versionStatus.completed',
  failed:    'versionStatus.failed',
  canceled:  'versionStatus.canceled',
}

function VersionRail({
  versions, currentVid, onSelect, onCreate, onExport, exporting, exportEnabled,
}: {
  versions: Version[]
  currentVid: number | null
  onSelect: (vid: number) => void
  onCreate: () => void
  onExport: () => void
  exporting: boolean
  exportEnabled: boolean
}) {
  const { t } = useTranslation()
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap',
      paddingTop: 14, paddingBottom: 2,
      borderTop: '1px solid var(--border-subtle)',
    }}>
      <span style={{
        fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
        color: 'var(--fg-tertiary)', textTransform: 'uppercase', letterSpacing: '0.08em',
        marginRight: 4,
      }}>{t('overview.rail.label')}</span>
      {versions.map((v) => {
        const isCurrent = v.id === currentVid
        return (
          <button
            key={v.id}
            onClick={() => onSelect(v.id)}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 7,
              padding: '5px 10px 5px 8px',
              background: isCurrent ? 'var(--bg-surface)' : 'transparent',
              border: '1px solid ' + (isCurrent ? 'var(--accent)' : 'var(--border-subtle)'),
              borderRadius: 'var(--r-md)',
              cursor: 'pointer',
              color: 'var(--fg-primary)',
              boxShadow: isCurrent ? '0 0 0 3px var(--accent-soft)' : 'none',
            }}
          >
            <StatusDotMini status={v.status} />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--t-sm)', fontWeight: 600 }}>{v.label}</span>
            <span style={{ fontSize: 'var(--t-xs)', color: 'var(--fg-tertiary)' }}>{t(STATUS_LABEL[v.status])}</span>
          </button>
        )
      })}
      <button
        onClick={onCreate}
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 4,
          padding: '5px 10px',
          background: 'transparent',
          border: '1px dashed var(--border-default)',
          borderRadius: 'var(--r-md)',
          cursor: 'pointer',
          color: 'var(--fg-tertiary)',
          fontSize: 'var(--t-sm)',
        }}
      >+ {t('overview.versionSelector.newVersion')}</button>
      <span style={{ flex: 1 }} />
      <button
        onClick={onExport}
        disabled={!exportEnabled || exporting}
        className={`btn btn-secondary btn-sm ${!exportEnabled ? 'opacity-40' : ''}`}
      >
        {exporting ? t('sidebar.exporting') : t('sidebar.export')}
      </button>
    </div>
  )
}

// ── StatusBanner shared bits ─────────────────────────────────────────────

const bannerMetaRow: React.CSSProperties = {
  display: 'flex', gap: 18, flexWrap: 'wrap',
  paddingTop: 10, marginTop: 4,
  borderTop: '1px dashed var(--border-subtle)',
}
const bannerActions: React.CSSProperties = {
  display: 'flex', gap: 6, marginTop: 12, marginLeft: 'auto',
  justifyContent: 'flex-end', flexWrap: 'wrap',
}

function BannerMeta({ k, v }: { k: string; v: string | number }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      <span style={{
        fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
        color: 'var(--fg-tertiary)', textTransform: 'uppercase', letterSpacing: '0.06em',
      }}>{k}</span>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--t-sm)', color: 'var(--fg-primary)', fontWeight: 600 }}>{v}</span>
    </div>
  )
}

function BannerShell({
  tint, iconChar, iconColor, iconPulse, title, sub, children,
}: {
  tint: 'err' | 'warn' | 'accent' | 'ok'
  iconChar: string
  iconColor: string
  iconPulse?: boolean
  title: string
  sub?: string
  children: ReactNode
}) {
  const tintMap = {
    err:    { bg: 'rgba(232, 118, 92, 0.06)', border: 'rgba(232, 118, 92, 0.30)' },
    warn:   { bg: 'rgba(224, 162, 58, 0.05)', border: 'rgba(224, 162, 58, 0.25)' },
    accent: { bg: 'rgba(237, 107, 58, 0.06)', border: 'rgba(237, 107, 58, 0.35)' },
    ok:     { bg: 'rgba(95, 199, 140, 0.05)', border: 'rgba(95, 199, 140, 0.25)' },
  }
  const tCfg = tintMap[tint]
  return (
    <div className="banner-shell" style={{
      background: tCfg.bg,
      border: '1px solid ' + tCfg.border,
      borderRadius: 'var(--r-lg)',
      display: 'flex', flexDirection: 'column', gap: 4,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span className="banner-shell-icon" style={{
          flexShrink: 0,
          borderRadius: '50%',
          background: 'var(--bg-surface)',
          border: '1px solid ' + tCfg.border,
          display: 'grid', placeItems: 'center',
          color: iconColor, fontWeight: 700,
          animation: iconPulse ? 'pulse 1.6s infinite' : 'none',
        }}>{iconChar}</span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 'var(--t-md)', fontWeight: 600, color: 'var(--fg-primary)' }}>{title}</div>
          {sub && <div style={{ fontSize: 'var(--t-xs)', color: 'var(--fg-secondary)', marginTop: 1 }}>{sub}</div>}
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, paddingTop: 6 }}>
        {children}
      </div>
    </div>
  )
}

function BannerProgress({
  now, total, running, muted, fail,
}: { now: number; total: number; running?: boolean; muted?: boolean; fail?: boolean }) {
  const pct = total > 0 ? Math.min(100, (now / total) * 100) : 0
  const color = fail ? 'var(--err)' : muted ? 'var(--fg-disabled)' : 'var(--accent)'
  return (
    <div style={{
      height: 6, borderRadius: 'var(--r-pill)',
      background: 'var(--bg-sunken)',
      overflow: 'hidden', position: 'relative',
    }}>
      <div style={{
        width: `${pct}%`, height: '100%',
        background: color,
        animation: running ? 'pulse 2s infinite' : 'none',
        borderRadius: 'var(--r-pill)',
      }}/>
      {muted && (
        <div style={{
          position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
          background: 'repeating-linear-gradient(45deg, transparent, transparent 4px, rgba(255,255,255,0.04) 4px, rgba(255,255,255,0.04) 8px)',
        }}/>
      )}
    </div>
  )
}

const PHASE_ORDER_TIMELINE: { id: VersionPhase; n: string; key: string }[] = [
  { id: 'curating',      n: '①', key: 'nav.curate' },
  { id: 'preprocessing', n: '②', key: 'nav.preprocess' },
  { id: 'tagging',       n: '③', key: 'nav.tag' },
  { id: 'editing',       n: '④', key: 'nav.tagEdit' },
  { id: 'regularizing',  n: '⑤', key: 'nav.reg' },
  { id: 'ready',         n: '⑥', key: 'nav.train' },
]

function PhaseTimeline({
  current, onPhaseClick,
}: {
  current: VersionPhase
  /** 点 phase box → 跳到对应 phase 页面 */
  onPhaseClick?: (phase: VersionPhase) => void
}) {
  const { t } = useTranslation()
  const ci = PHASE_ORDER_TIMELINE.findIndex((p) => p.id === current)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 0, padding: '6px 0', flexWrap: 'wrap' }}>
      {PHASE_ORDER_TIMELINE.map((p, i) => {
        const done = i < ci
        const here = i === ci
        const skip = p.id === 'regularizing'
        // ADR-0007 §11.5-A：strict —— cursor 之后 (i > ci) 全部不许跳。
        // cursor+1 推进必须经 banner "继续 X →" 按钮（会调 advance API 校验完成条件）。
        const disabled = i > ci
        const clickable = !disabled && !!onPhaseClick
        return (
          <span key={p.id} style={{ display: 'inline-flex', alignItems: 'center' }}>
            <button
              type="button"
              onClick={() => { if (!disabled) onPhaseClick?.(p.id) }}
              disabled={disabled}
              title={t(p.key)}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '4px 10px',
                borderRadius: 'var(--r-md)',
                background: 'transparent',
                border: '1px solid transparent',
                cursor: disabled ? 'not-allowed' : clickable ? 'pointer' : 'default',
                opacity: disabled ? 0.4 : 1,
                font: 'inherit',
              }}
            >
              <span style={{
                fontFamily: 'var(--font-mono)', fontSize: 'var(--t-sm)', fontWeight: 600,
                color: done ? 'var(--ok)' : here ? 'var(--accent)' : 'var(--fg-disabled)',
              }}>{p.n}</span>
              <span className="phase-timeline-label" style={{
                fontSize: 'var(--t-xs)',
                color: done ? 'var(--fg-secondary)' : here ? 'var(--fg-primary)' : 'var(--fg-tertiary)',
                fontWeight: here ? 600 : 400,
              }}>{t(p.key)}{skip ? <span style={{ color: 'var(--fg-tertiary)', fontWeight: 400 }}> · {t('overview.banner.skippableHint')}</span> : ''}</span>
            </button>
            {i < PHASE_ORDER_TIMELINE.length - 1 && (
              <span style={{
                display: 'inline-block', width: 14, height: 1,
                background: i < ci ? 'var(--ok)' : 'var(--border-subtle)',
              }}/>
            )}
          </span>
        )
      })}
    </div>
  )
}

// ── StatusBanner ─────────────────────────────────────────────────────────

function StatusBanner({
  projectId, version, latestTask, onOpenOutput,
}: {
  projectId: number
  version: Version
  latestTask: Task | null
  /** "下载" CTA 切到下方 [Output] tab */
  onOpenOutput: () => void
}) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { toast } = useToast()
  const ctx = useProjectCtx()
  const taskId = latestTask?.id

  // 拉 task outputs —— completed 状态时 version.output_lora_path 可能为空（早期
  // 训练 supervisor 未回填），用 task outputs.files (is_lora) 兜底找产物名。
  const [taskOutputs, setTaskOutputs] = useState<TaskOutputs | null>(null)
  useEffect(() => {
    if (!taskId || version.status !== 'completed') { setTaskOutputs(null); return }
    let cancelled = false
    void api.getTaskOutputs(taskId)
      .then((res) => { if (!cancelled) setTaskOutputs(res) })
      .catch(() => { if (!cancelled) setTaskOutputs(null) })
    return () => { cancelled = true }
  }, [taskId, version.status])

  const goLog = () => taskId && navigate(`/queue/${taskId}#log`)
  const goMonitor = () => taskId && navigate(`/queue/${taskId}#monitor`)
  const goPhase = (step: string) => navigate(`/projects/${projectId}/v/${version.id}/${step}`)

  const fmtTime = (ts: number | null | undefined) =>
    ts ? new Date(ts * 1000).toLocaleString('zh-CN', { hour12: false, dateStyle: 'short', timeStyle: 'short' }) : '—'

  if (version.status === 'canceled') {
    const cancelReason = latestTask?.error_msg || t('overview.banner.canceledReasonDefault')
    return (
      <BannerShell
        tint="err" iconChar="⊘" iconColor="var(--fg-tertiary)"
        title={`${version.label} · ${t('versionStatus.canceled')}`}
        sub={cancelReason}
      >
        <div style={{ ...bannerMetaRow, alignItems: 'center' }}>
          <BannerMeta k={t('overview.banner.metaTime')} v={fmtTime(latestTask?.finished_at)} />
          <span style={{ flex: 1 }} />
          {taskId && <button onClick={goLog} className="btn btn-ghost btn-sm">{t('overview.banner.viewLog')} →</button>}
          <button
            onClick={() => ctx && void ctx.onDeleteVersion(version.id)}
            className="btn btn-secondary btn-sm"
          >{t('overview.banner.deleteVersion')}</button>
          <button
            onClick={() => ctx?.onCreateVersion(version.id)}
            className="btn btn-primary btn-sm"
          >+ {t('overview.banner.forkConfigNew')}</button>
        </div>
      </BannerShell>
    )
  }

  if (version.status === 'failed') {
    const reason = version.last_failure_reason || latestTask?.error_msg || t('overview.banner.failedReasonDefault')
    return (
      <BannerShell
        tint="err" iconChar="!" iconColor="var(--err)"
        title={`${version.label} · ${t('overview.banner.failedTitle')}`}
        sub={reason}
      >
        <div style={{ ...bannerMetaRow, alignItems: 'center' }}>
          <BannerMeta k={t('overview.banner.metaTime')} v={fmtTime(latestTask?.finished_at)} />
          <span style={{ flex: 1 }} />
          {taskId && <button onClick={goLog} className="btn btn-ghost btn-sm">{t('overview.banner.viewLog')} →</button>}
          <button
            onClick={() => ctx && void ctx.onDeleteVersion(version.id)}
            className="btn btn-secondary btn-sm"
          >{t('overview.banner.deleteVersion')}</button>
          <button
            onClick={() => {
              ctx?.onCreateVersion(version.id)
              toast(t('overview.banner.smallerBatchHint'), 'info')
            }}
            className="btn btn-primary btn-sm"
          >{t('overview.banner.smallerBatchRetry')} ↻</button>
        </div>
      </BannerShell>
    )
  }

  if (version.status === 'training') {
    const startedAt = latestTask?.started_at
    return (
      <BannerShell
        tint="accent" iconChar="●" iconColor="var(--accent)" iconPulse
        title={`${version.label} · ${t('versionStatus.training')}`}
        sub={startedAt ? `${t('overview.banner.startedAt')} ${fmtTime(startedAt)}` : undefined}
      >
        <BannerProgress now={0} total={1} running />
        <div style={bannerMetaRow}>
          <BannerMeta k={t('overview.banner.metaStarted')} v={fmtTime(startedAt)} />
          {latestTask?.is_pausable && (
            <BannerMeta k={t('overview.banner.metaPausable')} v={t('overview.banner.yes')} />
          )}
        </div>
        <div style={bannerActions}>
          {latestTask?.is_pausable && (
            <button
              onClick={() => taskId && api.pauseTask(taskId).catch((e) => toast(String(e), 'error'))}
              className="btn btn-ghost btn-sm"
            >{t('overview.banner.pause')}</button>
          )}
          {taskId && (
            <button
              onClick={() => api.cancelTask(taskId).catch((e) => toast(String(e), 'error'))}
              className="btn btn-secondary btn-sm"
            >{t('overview.banner.cancelTraining')}</button>
          )}
          {taskId && <button onClick={goMonitor} className="btn btn-primary btn-sm">{t('overview.banner.openMonitor')} →</button>}
        </div>
      </BannerShell>
    )
  }

  if (version.status === 'completed') {
    // 测试中加载用完整 path：version 字段优先；空时用 task outputs 第一个 LoRA 文件 兜底
    const loraFromTask = taskOutputs?.files.find((f) => f.is_lora)?.name ?? null
    const loraPathForTest = version.output_lora_path
      || (taskOutputs?.output_dir && loraFromTask ? `${taskOutputs.output_dir}/${loraFromTask}` : null)
    return (
      <BannerShell
        tint="ok" iconChar="✓" iconColor="var(--ok)"
        title={`${version.label} · ${t('versionStatus.completed')}`}
        sub={fmtTime(latestTask?.finished_at)}
      >
        <div style={{ ...bannerMetaRow, alignItems: 'center' }}>
          {taskId && <BannerMeta k={t('overview.banner.metaTaskId')} v={`#${taskId}`} />}
          {taskOutputs && (
            <BannerMeta
              k={t('overview.banner.metaLoraCount')}
              v={taskOutputs.files.filter((f) => f.is_lora).length}
            />
          )}
          <span style={{ flex: 1 }} />
          <button
            onClick={() => ctx?.onCreateVersion(version.id)}
            className="btn btn-ghost btn-sm"
          >{t('overview.banner.copyAsNew')}</button>
          <button
            onClick={() => {
              if (!loraPathForTest) {
                toast(t('overview.banner.noArtifact'), 'error')
                return
              }
              const sp = new URLSearchParams()
              sp.set('lora', loraPathForTest)
              sp.set('projectId', String(projectId))
              sp.set('versionId', String(version.id))
              navigate(`/tools/generate?${sp.toString()}`)
            }}
            className="btn btn-secondary btn-sm"
          >{t('overview.banner.loadInTest')} →</button>
          <button
            onClick={onOpenOutput}
            className="btn btn-primary btn-sm"
          >{t('overview.banner.downloadLora')} ↓</button>
        </div>
      </BannerShell>
    )
  }

  // preparing
  const phase = version.phase
  // banner 按钮始终反映 current phase，让用户去当前阶段页面做事。
  // cursor advance 走 Sidebar 的 "cursor+1" 行入口（见 Sidebar.handleAdvanceToNext），
  // banner 不再自己调 advance/skip — 之前的 next-phase 文案会让用户误以为
  // 当前阶段已完成（如 curating 0/0 时显示「继续 ② 打标」）。
  const continueTarget = PHASE_ORDER_TIMELINE.find((p) => p.id === phase) ?? null

  const handleContinue = () => {
    const step = continueTarget ? PHASE_TO_STEP_LOCAL[continueTarget.id] : null
    if (step) goPhase(step)
  }

  return (
    <BannerShell
      tint="warn" iconChar="◐" iconColor="var(--warn)"
      title={`${version.label} · ${t('versionStatus.preparing')}`}
      sub={t('overview.banner.preparingSub')}
    >
      <PhaseTimeline
        current={phase}
        onPhaseClick={(p) => {
          const step = PHASE_TO_STEP_LOCAL[p]
          if (step) goPhase(step)
        }}
      />
      <div style={{ ...bannerMetaRow, alignItems: 'center' }}>
        <BannerMeta
          k={t('overview.banner.metaCurrentPhase')}
          v={t(PHASE_ORDER_TIMELINE.find((p) => p.id === phase)?.key ?? 'nav.curate')}
        />
        {version.stats && (
          <BannerMeta
            k={t('overview.banner.metaTagged')}
            v={`${version.stats.tagged_image_count} / ${version.stats.train_image_count}`}
          />
        )}
        <span style={{ flex: 1 }} />
        {continueTarget && (
          <button
            onClick={() => void handleContinue()}
            className="btn btn-primary btn-sm"
          >{t('overview.banner.continueLabel')} {continueTarget.n} {t(continueTarget.key)} →</button>
        )}
      </div>
    </BannerShell>
  )
}

/** phase enum → URL step key（StatusBanner 内用，独立于 sidebar 的同名 map）。 */
const PHASE_TO_STEP_LOCAL: Record<VersionPhase, string> = {
  curating:      'curate',
  preprocessing: 'preprocess',
  tagging:       'tag',
  editing:       'edit',
  regularizing:  'reg',
  ready:         'train',
}

/** cursor 校验：preparing 态下只允许 cursor 及之前的 phase（cursor+1 也禁，
 *  推进必须走 banner 的 "继续 X →" 按钮，那里会调 advance API 校验完成条件）。
 *  非 preparing 态（已训练 / 训练中 / 终态）所有 phase 都允许跳（回看历史）。 */
function canGoVersionPhase(version: Version | null, phase: VersionPhase): boolean {
  if (!version) return false
  if (version.status !== 'preparing') return true
  const cursorIdx = PHASE_ORDER_TIMELINE.findIndex((p) => p.id === version.phase)
  const targetIdx = PHASE_ORDER_TIMELINE.findIndex((p) => p.id === phase)
  return targetIdx <= cursorIdx
}

// ── HeroCard / 详情 card 通用 shell ──────────────────────────────────────

function HeroCard({
  title, count, countSub, action, phase, children,
}: {
  title: string
  count?: number | null
  countSub?: string
  /** disabled 时按钮 opacity 0.4 + cursor not-allowed，点击不触发 onClick（cursor 校验失败用） */
  action?: { label: string; onClick: () => void; disabled?: boolean }
  phase?: string
  children: ReactNode
}) {
  return (
    <div style={{
      padding: 16,
      background: 'var(--bg-surface)',
      border: '1px solid var(--border-subtle)',
      borderRadius: 'var(--r-lg)',
      display: 'flex', flexDirection: 'column', gap: 12,
      height: '100%', minHeight: 0,
    }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
        <h3 style={{ margin: 0, fontSize: 'var(--t-sm)', fontWeight: 600, color: 'var(--fg-primary)' }}>{title}</h3>
        <span style={{ flex: 1 }} />
        {count != null && (
          <span style={{ display: 'inline-flex', alignItems: 'baseline', gap: 4, fontFamily: 'var(--font-mono)' }}>
            <span style={{ fontSize: 'var(--t-lg)', fontWeight: 600, color: 'var(--fg-primary)' }}>{count}</span>
            {countSub && <span style={{ fontSize: 'var(--t-xs)', color: 'var(--fg-tertiary)' }}>{countSub}</span>}
          </span>
        )}
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10, minHeight: 0, overflow: 'hidden' }}>{children}</div>
      {action && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6, paddingTop: 8,
          borderTop: '1px dashed var(--border-subtle)',
        }}>
          {phase && (
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
              color: 'var(--fg-tertiary)', textTransform: 'uppercase', letterSpacing: '0.06em', flex: 1,
            }}>{phase}</span>
          )}
          <button
            onClick={() => { if (!action.disabled) action.onClick() }}
            disabled={action.disabled}
            style={{
              padding: '4px 10px',
              fontSize: 'var(--t-xs)', color: 'var(--fg-primary)',
              background: 'var(--bg-sunken)', border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--r-sm)',
              cursor: action.disabled ? 'not-allowed' : 'pointer',
              opacity: action.disabled ? 0.4 : 1,
              fontWeight: 500,
            }}
          >{action.label} →</button>
        </div>
      )}
    </div>
  )
}

// ── TrainSetCard (hero) ──────────────────────────────────────────────────
// 文件夹 chips 放 header 右边（跟 Curation 训练集 panel 同款）；body 用
// ImageGrid 渲染当前 folder 的图，点击放大走 ImagePreviewModal。

const EMPTY_SELECTED: Set<string> = new Set()

function TrainSetCard({ project, version }: { project: ProjectDetail; version: Version | null }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [view, setView] = useState<CurationView | null>(null)
  const [selectedFolder, setSelectedFolder] = useState<string>('all')
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)

  useEffect(() => {
    if (!version) { setView(null); return }
    let cancelled = false
    void api.getCuration(project.id, version.id)
      .then((res) => { if (!cancelled) setView(res) })
      .catch(() => { if (!cancelled) setView(null) })
    return () => { cancelled = true }
  }, [project.id, version])

  const folders = view?.folders ?? []
  const folderCounts: Record<string, number> = useMemo(() => {
    if (!view) return {}
    const out: Record<string, number> = {}
    for (const f of view.folders) out[f] = (view.right[f] ?? []).length
    return out
  }, [view])
  const total = view?.train_total ?? version?.stats?.train_image_count ?? 0

  // 当前选中 folder 的图，转换为 ImageGridItem[]
  const items = useMemo<Array<ImageGridItem & { folder: string; pureName: string }>>(() => {
    if (!view || !version) return []
    const out: Array<ImageGridItem & { folder: string; pureName: string }> = []
    const list = selectedFolder === 'all' ? view.folders : [selectedFolder]
    for (const folder of list) {
      const arr = view.right[folder] ?? []
      for (const it of arr) {
        out.push({
          name: `${folder}/${it.name}`,
          pureName: it.name,
          folder,
          thumbUrl: api.versionThumbUrl(project.id, version.id, 'train', it.name, folder, 256),
        })
      }
    }
    return out
  }, [view, version, project.id, selectedFolder])

  // 预览大图 src（1600 大小）
  const previewItem = previewIdx != null ? items[previewIdx] : null
  const previewSrc = previewItem && version
    ? api.versionThumbUrl(project.id, version.id, 'train', previewItem.pureName, previewItem.folder, 1600)
    : ''

  const actionDisabled = !canGoVersionPhase(version, 'curating')
  const phaseLine = `① ${t('nav.download')} → ② ${t('nav.preprocess')} → ③ ${t('nav.curate')}`

  return (
    <div style={{
      padding: 16,
      background: 'var(--bg-surface)',
      border: '1px solid var(--border-subtle)',
      borderRadius: 'var(--r-lg)',
      display: 'flex', flexDirection: 'column', gap: 12,
      height: '100%', minHeight: 0,
    }}>
      {/* Header: title + folder chips on right */}
      <div className="flex items-center gap-3 flex-wrap">
        <h3 className="m-0 text-sm font-semibold" style={{ color: 'var(--fg-primary)' }}>
          {t('overview.detail.folders')}
        </h3>
        <span className="flex-1" />
        {folders.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <FolderChip
              label={t('overview.detail.allFolders')}
              count={total}
              active={selectedFolder === 'all'}
              onClick={() => setSelectedFolder('all')}
            />
            {folders.map((f) => (
              <FolderChip
                key={f}
                label={f}
                count={folderCounts[f] ?? 0}
                active={selectedFolder === f}
                onClick={() => setSelectedFolder(f)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Body: ImageGrid 或 empty */}
      <div className="flex-1 min-h-0">
        {!version || items.length === 0 ? (
          <p className="m-0 text-xs text-fg-tertiary italic">{t('overview.detail.emptyCurate')}</p>
        ) : (
          <ImageGrid
            items={items}
            selected={EMPTY_SELECTED}
            onSelect={() => { /* read-only */ }}
            clickMode="activate"
            onActivate={(name) => setPreviewIdx(items.findIndex((i) => i.name === name))}
            onPreview={(name) => setPreviewIdx(items.findIndex((i) => i.name === name))}
            ariaLabel="overview-train-grid"
          />
        )}
      </div>

      {/* Action row */}
      {version && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6, paddingTop: 8,
          borderTop: '1px dashed var(--border-subtle)',
        }}>
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
            color: 'var(--fg-tertiary)', textTransform: 'uppercase', letterSpacing: '0.06em', flex: 1,
          }}>{phaseLine}</span>
          <button
            onClick={() => { if (!actionDisabled) navigate(`/projects/${project.id}/v/${version.id}/curate`) }}
            disabled={actionDisabled}
            style={{
              padding: '4px 10px',
              fontSize: 'var(--t-xs)', color: 'var(--fg-primary)',
              background: 'var(--bg-sunken)', border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--r-sm)',
              cursor: actionDisabled ? 'not-allowed' : 'pointer',
              opacity: actionDisabled ? 0.4 : 1,
              fontWeight: 500,
            }}
          >③ {t('nav.curate')} · {t('overview.detail.reorganize')} →</button>
        </div>
      )}

      {/* Preview modal */}
      {previewItem && previewIdx != null && (
        <ImagePreviewModal
          src={previewSrc}
          caption={previewItem.name}
          hasPrev={previewIdx > 0}
          hasNext={previewIdx < items.length - 1}
          onClose={() => setPreviewIdx(null)}
          onPrev={() => setPreviewIdx((i) => (i != null && i > 0 ? i - 1 : i))}
          onNext={() => setPreviewIdx((i) => (i != null && i < items.length - 1 ? i + 1 : i))}
        />
      )}
    </div>
  )
}

function FolderChip({
  label, count, active, onClick,
}: { label: string; count: number; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`px-2 py-0.5 rounded-md font-mono transition-colors ${
        active
          ? 'border border-accent bg-accent-soft text-accent'
          : 'border border-dim bg-surface text-fg-secondary hover:bg-overlay'
      }`}
    >
      {label}
      <span className="text-fg-tertiary"> ({count})</span>
    </button>
  )
}

// ── TagDistCard (hero) ───────────────────────────────────────────────────

function TagDistCard({ project, version }: { project: ProjectDetail; version: Version | null }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const triggerWord = version?.trigger_word ?? ''
  const [tags, setTags] = useState<Array<{ tag: string; n: number }>>([])
  const [uniqueTotal, setUniqueTotal] = useState(0)

  useEffect(() => {
    if (!version) { setTags([]); setUniqueTotal(0); return }
    let cancelled = false
    void api.listCaptionsFull(project.id, version.id)
      .then((res) => {
        if (cancelled) return
        const counter = new Map<string, number>()
        for (const it of res.items) {
          for (const tg of it.tags) counter.set(tg, (counter.get(tg) ?? 0) + 1)
        }
        const arr = Array.from(counter.entries())
          .map(([tag, n]) => ({ tag, n }))
          .sort((a, b) => b.n - a.n || a.tag.localeCompare(b.tag))
        setUniqueTotal(arr.length)
        setTags(arr)
      })
      .catch(() => {
        if (cancelled) return
        setTags([]); setUniqueTotal(0)
      })
    return () => { cancelled = true }
  }, [project.id, version])

  const max = useMemo(() => Math.max(1, ...tags.map((t) => t.n)), [tags])

  return (
    <HeroCard
      title={t('overview.detail.tagDist')}
      count={uniqueTotal}
      countSub={t('overview.detail.tagSuffix')}
      action={version ? {
        label: `⑤ ${t('nav.tagEdit')}`,
        onClick: () => navigate(`/projects/${project.id}/v/${version.id}/edit`),
        disabled: !canGoVersionPhase(version, 'editing'),
      } : undefined}
      phase={`④ ${t('nav.tag')} → ⑤ ${t('nav.tagEdit')}`}
    >
      {tags.length === 0 ? (
        <p style={{ margin: 0, fontSize: 'var(--t-xs)', color: 'var(--fg-tertiary)', fontStyle: 'italic' }}>
          {t('overview.detail.emptyTag')}
        </p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3, fontFamily: 'var(--font-mono)', overflowY: 'auto', flex: 1, minHeight: 0 }}>
          {tags.map((row) => {
            const pct = row.n / max
            const isTrigger = !!triggerWord && row.tag === triggerWord
            return (
              <div key={row.tag} style={{
                display: 'grid', gridTemplateColumns: '1fr 36px',
                alignItems: 'center', gap: 8,
                padding: '4px 8px',
                minHeight: 22, flexShrink: 0,
                borderRadius: 'var(--r-sm)',
                background: isTrigger ? 'var(--accent-soft)' : 'transparent',
                position: 'relative', overflow: 'hidden',
              }}>
                <div style={{
                  position: 'absolute', left: 0, top: 0, bottom: 0,
                  width: `${pct * 100}%`,
                  background: isTrigger ? 'rgba(237,107,58,0.18)' : 'rgba(237,107,58,0.08)',
                  zIndex: 0,
                }}/>
                <span style={{
                  position: 'relative', zIndex: 1,
                  fontSize: 'var(--t-xs)',
                  color: isTrigger ? 'var(--accent)' : 'var(--fg-primary)',
                  fontWeight: isTrigger ? 700 : 500,
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>{isTrigger ? '★ ' : ''}<TranslatedTag tag={row.tag} /></span>
                <span style={{
                  position: 'relative', zIndex: 1,
                  fontSize: 'var(--t-xs)',
                  color: 'var(--fg-primary)', textAlign: 'right', fontWeight: 600,
                }}>{row.n}</span>
              </div>
            )
          })}
        </div>
      )}
    </HeroCard>
  )
}

// ── HistTile / RegTile (下排 3 tile) ─────────────────────────────────────

function HistTileCard({
  title, bins, action, phase, emptyHint,
}: {
  title: string
  bins: Array<{ key?: string; label: string; n: number }>
  action?: { label: string; onClick: () => void; disabled?: boolean }
  phase?: string
  emptyHint: string
}) {
  return (
    <HeroCard title={title} action={action} phase={phase}>
      {bins.length === 0 ? (
        <p style={{ margin: 0, fontSize: 'var(--t-xs)', color: 'var(--fg-tertiary)', fontStyle: 'italic' }}>
          {emptyHint}
        </p>
      ) : (
        <div style={{ overflowY: 'auto' }}>
          <BarHistogram bins={bins} />
        </div>
      )}
    </HeroCard>
  )
}

function RegTileCard({
  regCount, onGoReg, disabled,
}: {
  regCount: number
  onGoReg: () => void
  disabled?: boolean
}) {
  const { t } = useTranslation()
  if (regCount > 0) {
    return (
      <HeroCard
        title={t('overview.detail.regSet')}
        count={regCount}
        countSub={t('overview.detail.imagesSuffix')}
        action={{ label: `⑥ ${t('nav.reg')}`, onClick: onGoReg, disabled }}
        phase={`⑥ ${t('nav.reg')}`}
      >
        <p style={{ margin: 0, fontSize: 'var(--t-xs)', color: 'var(--fg-tertiary)' }}>
          {t('overview.detail.regCount', { n: regCount })}
        </p>
      </HeroCard>
    )
  }
  return (
    <HeroCard
      title={t('overview.detail.regSet')}
      action={{ label: `⑥ ${t('nav.reg')}`, onClick: onGoReg, disabled }}
      phase={`⑥ ${t('nav.reg')} · ${t('overview.banner.skippableHint')}`}
    >
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        gap: 8, padding: '20px 0', color: 'var(--fg-tertiary)',
        background: 'repeating-linear-gradient(45deg, transparent 0 8px, rgba(255,255,255,0.015) 8px 16px)',
        borderRadius: 'var(--r-md)',
      }}>
        <div style={{
          width: 36, height: 36, borderRadius: 'var(--r-md)',
          border: '1px dashed var(--border-default)',
          display: 'grid', placeItems: 'center', color: 'var(--fg-tertiary)', fontSize: 16,
        }}>∅</div>
        <span style={{ fontSize: 'var(--t-xs)', fontStyle: 'italic', textAlign: 'center', maxWidth: 200 }}>
          {t('overview.detail.regEmptyHint')}
        </span>
      </div>
    </HeroCard>
  )
}

// ── DetailGrid (2 hero + 3 tile) ─────────────────────────────────────────

function DetailGrid({ project, version }: { project: ProjectDetail; version: Version | null }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const vid = version?.id

  // ADR 0010: preprocess 已下沉 version scope；project 概览 hist 用 active
  // version 的 train 数据。没 active version → 空。
  const [preprocessItems, setPreprocessItems] = useState<Array<{ w: number | null; h: number | null }>>([])
  useEffect(() => {
    if (vid == null) { setPreprocessItems([]); return }
    let cancelled = false
    void api.listPreprocessFilesTrain(project.id, vid)
      .then((res) => {
        if (cancelled) return
        setPreprocessItems(res.images.map((i) => ({ w: i.w, h: i.h })))
      })
      .catch(() => { if (!cancelled) setPreprocessItems([]) })
    return () => { cancelled = true }
  }, [project.id, vid])

  // crop workspace - 长宽比 hist 数据源（train scope）
  const [cropItems, setCropItems] = useState<Array<{ w: number; h: number }>>([])
  useEffect(() => {
    if (vid == null) { setCropItems([]); return }
    let cancelled = false
    void api.listCropWorkspaceTrain(project.id, vid)
      .then((res) => {
        if (cancelled) return
        setCropItems(res.images.map((i) => ({ w: i.w, h: i.h })))
      })
      .catch(() => { if (!cancelled) setCropItems([]) })
    return () => { cancelled = true }
  }, [project.id, vid])

  const pixelBins = useMemo(
    () => computePixelHist(preprocessItems).map((b) => ({ key: b.id, label: b.label, n: b.n })),
    [preprocessItems],
  )
  const arBins = useMemo(() => {
    const m = new Map<string, { label: string; n: number; sortKey: number }>()
    for (const im of cropItems) {
      if (im.w <= 0 || im.h <= 0) continue
      const { label, sortKey } = arBucket(im.w / im.h)
      const prev = m.get(label)
      m.set(label, { label, sortKey, n: (prev?.n ?? 0) + 1 })
    }
    return Array.from(m.values())
      .sort((a, b) => b.sortKey - a.sortKey)
      .map((b) => ({ label: b.label, n: b.n }))
  }, [cropItems])

  const regCount = version?.stats?.reg_image_count ?? 0

  return (
    <div style={{ display: 'grid', gridTemplateRows: '1.4fr 1fr', gap: 12, height: '100%', minHeight: 0 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 12, minHeight: 0 }}>
        <TrainSetCard project={project} version={version} />
        <TagDistCard project={project} version={version} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, minHeight: 0 }}>
        <HistTileCard
          title={t('overview.detail.resolutionDist')}
          bins={pixelBins}
          emptyHint={t('overview.detail.emptyResolution')}
          action={version ? { label: `② ${t('nav.preprocess')}`, onClick: () => navigate(`/projects/${project.id}/v/${version.id}/preprocess?tool=upscale`) } : undefined}
          phase={`② ${t('nav.preprocess')}`}
        />
        <HistTileCard
          title={t('overview.detail.aspectDist')}
          bins={arBins}
          emptyHint={t('overview.detail.emptyAspect')}
          action={version ? { label: `② ${t('nav.preprocess')}`, onClick: () => navigate(`/projects/${project.id}/v/${version.id}/preprocess?tool=crop`) } : undefined}
          phase={`② ${t('nav.preprocess')}`}
        />
        <RegTileCard
          regCount={regCount}
          onGoReg={() => version && navigate(`/projects/${project.id}/v/${version.id}/reg`)}
          disabled={!canGoVersionPhase(version, 'regularizing')}
        />
      </div>
    </div>
  )
}

// ── Tasks / Output 面板（version scope，沿用） ───────────────────────────

const TASK_STATUS_BADGE: Record<string, string> = {
  pending: 'neutral', running: 'accent', paused: 'warn',
  done: 'ok', failed: 'err', canceled: 'neutral',
}

function VersionTasksPanel({ projectId, versionId }: { projectId: number; versionId: number | null }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [tasks, setTasks] = useState<Task[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    void api.listQueue()
      .then((items) => {
        if (cancelled) return
        const filtered = items
          .filter((tk) => tk.project_id === projectId && (versionId == null || tk.version_id === versionId))
          .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
        setTasks(filtered)
      })
      .catch(() => { if (!cancelled) setTasks([]) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [projectId, versionId])

  if (loading) return <div className="p-6 text-fg-tertiary text-sm">{t('common.loading')}</div>
  if (tasks.length === 0) return <div className="p-6 text-fg-tertiary text-sm italic">{t('overview.tasksEmpty')}</div>

  const fmtTime = (ts: number | null) => ts ? new Date(ts * 1000).toLocaleString() : '—'

  return (
    <div className="p-6">
      <table className="w-full text-sm">
        <thead className="text-fg-tertiary text-xs">
          <tr className="border-b border-subtle">
            <th className="text-left py-2 px-3 font-normal">{t('overview.tasksTable.name')}</th>
            <th className="text-left py-2 px-3 font-normal">{t('overview.tasksTable.status')}</th>
            <th className="text-left py-2 px-3 font-normal">{t('overview.tasksTable.started')}</th>
            <th className="text-left py-2 px-3 font-normal">{t('overview.tasksTable.finished')}</th>
          </tr>
        </thead>
        <tbody>
          {tasks.map((tk) => (
            <tr
              key={tk.id}
              className="border-b border-subtle cursor-pointer hover:bg-overlay"
              onClick={() => navigate(`/queue/${tk.id}`)}
            >
              <td className="py-2 px-3 font-mono">#{tk.id} {tk.name}</td>
              <td className="py-2 px-3"><span className={`badge badge-${TASK_STATUS_BADGE[tk.status] ?? 'neutral'}`}>{tk.status}</span></td>
              <td className="py-2 px-3 text-fg-tertiary text-xs">{fmtTime(tk.started_at ?? null)}</td>
              <td className="py-2 px-3 text-fg-tertiary text-xs">{fmtTime(tk.finished_at ?? null)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function VersionOutputPanel({
  version, latestTask,
}: {
  version: Version | null
  latestTask: Task | null
}) {
  const { t } = useTranslation()
  if (!version) return <div className="p-6 text-fg-tertiary text-sm italic">{t('overview.outputEmpty')}</div>
  if (!latestTask) return <div className="p-6 text-fg-tertiary text-sm italic">{t('overview.outputEmptyVersion')}</div>
  // 复用 QueueDetail OutputsTab：列表 + 排序 + 单文件下载 + 批量打 zip + 打开
  // 文件夹 + 导出 data_exports（跟 task 详情页同款行为）
  return <OutputsTab taskId={latestTask.id} />
}

// ── Main ─────────────────────────────────────────────────────────────────

export default function ProjectOverview() {
  const { t } = useTranslation()
  const { project, activeVersion } = useOutletContext<Ctx>()
  const ctx = useProjectCtx()

  // 初值优先级：URL `?version=N` (从 /queue 项目链接跳来时带) → project.active_version_id → activeVersion
  // 读完 URL 后用 history.replaceState 清掉 query，避免刷新覆盖用户后续在 dropdown 选的版本
  const [selectedVid, setSelectedVid] = useState<number | null>(() => {
    try {
      const sp = new URLSearchParams(window.location.search)
      const v = sp.get('version')
      if (v) {
        const n = Number(v)
        if (Number.isFinite(n)) return n
      }
    } catch { /* ignore */ }
    return project.active_version_id ?? activeVersion?.id ?? null
  })
  useEffect(() => {
    try {
      const url = new URL(window.location.href)
      if (url.searchParams.has('version')) {
        url.searchParams.delete('version')
        window.history.replaceState({}, '', url.toString())
      }
    } catch { /* ignore */ }
  }, [])
  useEffect(() => {
    const stillExists = project.versions.some((v) => v.id === selectedVid)
    if (!stillExists) setSelectedVid(project.active_version_id ?? null)
  }, [project.versions, project.active_version_id, selectedVid])

  const selectedVersion: Version | null =
    project.versions.find((v) => v.id === selectedVid) ?? null

  // 拉 selected version 的最新 task — banner 状态叙事 + CTA 数据源
  const [latestTask, setLatestTask] = useState<Task | null>(null)
  // seq 守卫：SSE 防抖重拉与切版本拉取可能并发，旧响应不许覆盖新状态。
  const latestSeqRef = useRef(0)
  const reloadLatestTask = useCallback(async () => {
    if (!selectedVid) { setLatestTask(null); return }
    const seq = ++latestSeqRef.current
    try {
      const items = await api.listQueue()
      if (seq !== latestSeqRef.current) return
      const list = items
        .filter((tk) => tk.project_id === project.id && tk.version_id === selectedVid)
        .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
      setLatestTask(list[0] ?? null)
    } catch {
      if (seq === latestSeqRef.current) setLatestTask(null)
    }
  }, [project.id, selectedVid])

  useEffect(() => { void reloadLatestTask() }, [reloadLatestTask])

  // latestTask 是独立 local state（不是 project prop）—— Layout 的
  // version_state_changed reload 只刷 project，碰不到它。但训练态 banner 的
  // 暂停按钮（latestTask.is_pausable）要靠 train_loop_started +
  // auto_epoch_backup_written 翻 true，task 生命周期变化（新任务启动 / 结束落
  // finished_at·error_msg）也得让 banner 的 CTA 跟上，所以这几个事件都要重拉。
  // 不订阅就会出现「暂停按钮一直不出现」「完成/失败 banner 时间停在 —」等
  // 滞后，必须切版本 / 刷新页面才更新。100ms 防抖合并启动瞬间的事件风暴。
  const latestReloadTimer = useRef<number | null>(null)
  useEventStream((evt) => {
    if (
      evt.type === 'task_state_changed' ||
      evt.type === 'train_loop_started' ||
      evt.type === 'auto_epoch_backup_written'
    ) {
      if (latestReloadTimer.current) return
      latestReloadTimer.current = window.setTimeout(() => {
        latestReloadTimer.current = null
        void reloadLatestTask()
      }, 100)
    }
  })
  useEffect(() => () => {
    if (latestReloadTimer.current) window.clearTimeout(latestReloadTimer.current)
  }, [])

  const [activeTab, setActiveTab] = useState<OverviewTab>('details')

  const tabBtnCls = (tab: OverviewTab) => [
    'px-4 py-2 text-sm border-none bg-transparent cursor-pointer border-b-2 transition-colors',
    activeTab === tab
      ? 'text-fg-primary font-semibold border-accent'
      : 'text-fg-secondary border-transparent hover:text-fg-primary',
  ].join(' ')

  return (
    <div className="fade-in flex flex-col h-full min-h-0">
      {/* ── 顶部三段：Identity / VersionRail / StatusBanner ──── */}
      <div
        className="shrink-0 border-b border-subtle"
        style={{ padding: '14px 24px 10px', display: 'flex', flexDirection: 'column', gap: 8, background: 'var(--bg-canvas)' }}
      >
        <Identity
          project={project}
          version={selectedVersion}
          totalVersions={project.versions.length}
        />
        <VersionRail
          versions={project.versions}
          currentVid={selectedVid}
          onSelect={(vid) => { setSelectedVid(vid); ctx?.onSelectVersion(vid) }}
          onCreate={() => ctx && ctx.onCreateVersion()}
          onExport={() => ctx && ctx.onExportTrain()}
          exporting={ctx?.exporting ?? false}
          exportEnabled={!!selectedVersion}
        />
        {selectedVersion && (
          <StatusBanner
            projectId={project.id}
            version={selectedVersion}
            latestTask={latestTask}
            onOpenOutput={() => setActiveTab('output')}
          />
        )}
      </div>

      {/* ── Tabs ──── */}
      <div className="border-b border-subtle px-6 shrink-0">
        <div className="flex gap-1">
          <button className={tabBtnCls('details')} onClick={() => setActiveTab('details')}>
            {t('overview.tabDetails')}
          </button>
          <button className={tabBtnCls('tasks')} onClick={() => setActiveTab('tasks')}>
            {t('overview.tabTasks')}
          </button>
          <button className={tabBtnCls('output')} onClick={() => setActiveTab('output')}>
            {t('overview.tabOutput')}
          </button>
        </div>
      </div>

      {/* ── Tab body ──── */}
      {activeTab === 'details' && (
        <div className="px-6 pt-3 pb-6 flex-1 min-h-0 overflow-y-auto">
          <DetailGrid project={project} version={selectedVersion} />
        </div>
      )}
      {activeTab === 'tasks' && (
        <div className="flex-1 min-h-0 overflow-y-auto">
          <VersionTasksPanel projectId={project.id} versionId={selectedVid} />
        </div>
      )}
      {activeTab === 'output' && (
        <div className="flex-1 min-h-0 overflow-y-auto">
          <VersionOutputPanel version={selectedVersion} latestTask={latestTask} />
        </div>
      )}
    </div>
  )
}
