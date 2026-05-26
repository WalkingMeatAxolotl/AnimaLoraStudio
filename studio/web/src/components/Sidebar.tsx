import { Fragment, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { api, PHASE_ORDER, PHASE_SKIPPABLE, type Version, type VersionPhase, type VersionStatus } from '../api/client'
import { useSettingsDrawer } from '../lib/SettingsDrawer'
import { getStoredTheme, toggleTheme, type Theme } from '../lib/theme'
import { useToast } from './Toast'

/** ADR-0007 §11.2 / §11.5: cursor 派生 step 完成态。
 *
 * STEPS 顺序：0 download / 1 preprocess / 2 curate / 3 tag / 4 edit / 5 reg / 6 train
 * 项目级 ①②：`*_image_count > 0` 派生
 * version 级 ③-⑦：`PHASE_ORDER.indexOf(STEP_KEY_TO_PHASE[key]) < cursorIdx`
 */
const STEP_KEY_TO_PHASE: Record<string, VersionPhase> = {
  curate: 'curating',
  tag:    'tagging',
  edit:   'editing',
  reg:    'regularizing',
  train:  'ready',
}

const PHASE_TO_STEP_KEY: Record<VersionPhase, string> = {
  curating:     'curate',
  tagging:      'tag',
  editing:      'edit',
  regularizing: 'reg',
  ready:        'train',
}
import { useProjectCtx } from '../context/ProjectContext'

// ── icons ──────────────────────────────────────────────────────────────────
const I = {
  folder:  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>,
  queue:   <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M4 6h16M4 12h10M4 18h16"/><circle cx="18" cy="12" r="2" fill="currentColor"/></svg>,
  preset:  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><line x1="6" y1="4" x2="6" y2="20"/><line x1="12" y1="4" x2="12" y2="20"/><line x1="18" y1="4" x2="18" y2="20"/><circle cx="6" cy="9" r="2" fill="var(--bg-sunken)"/><circle cx="12" cy="15" r="2" fill="var(--bg-sunken)"/><circle cx="18" cy="7" r="2" fill="var(--bg-sunken)"/></svg>,
  monitor: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M3 17l4-6 4 3 5-9 5 7"/></svg>,
  cog:     <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h0a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>,
  image:   <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="1.6" fill="currentColor"/><path d="m21 15-5-5L5 21"/></svg>,
  check:   <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="m4 12 5 5 11-12"/></svg>,
  chevL:   <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M15 6l-6 6 6 6"/></svg>,
  chevR:   <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M9 6l6 6-6 6"/></svg>,
  download:<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 4v12m0 0-4-4m4 4 4-4M4 20h16"/></svg>,
  upscale: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="13" width="8" height="8" rx="1"/><path d="M14 10V4h-6"/><path d="M21 3 14 10"/></svg>,
  filter:  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 5h18l-7 9v6l-4-2v-4z"/></svg>,
  tag:     <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M20 12 12 20l-9-9V3h8z"/><circle cx="7" cy="7" r="1.5" fill="currentColor"/></svg>,
  edit:    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 21h4l11-11-4-4L3 17z"/><path d="m14 5 4 4"/></svg>,
  reg:     <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><circle cx="17.5" cy="17.5" r="3.5"/></svg>,
  train:   <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 18 9 12l4 4 8-9"/><path d="M15 7h6v6"/></svg>,
  export:  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/></svg>,
  plus:    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M12 5v14M5 12h14"/></svg>,
  sun:     <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>,
  moon:    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>,
}

// ── version status dot (ADR-0007 §11.3-B) ─────────────────────────────────
const STATUS_DOT: Record<VersionStatus, string> = {
  preparing: 'dot dot-warn',
  training:  'dot dot-running',
  completed: 'dot dot-ok',
  failed:    'dot dot-err',
  canceled:  'dot dot-neutral',
}

// ── logo ───────────────────────────────────────────────────────────────────
function Logo({ collapsed }: { collapsed: boolean }) {
  const [version, setVersion] = useState<string | null>(null)
  useEffect(() => {
    let alive = true
    api.health().then((h) => { if (alive) setVersion(h.version) }).catch(() => {})
    return () => { alive = false }
  }, [])
  return (
    <div className="flex items-center gap-2.5">
      <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden>
        <rect x="2" y="2" width="22" height="22" rx="5" fill="var(--accent)" />
        <path d="M8 18 L13 7 L18 18" stroke="var(--accent-fg)" strokeWidth="2" fill="none" strokeLinejoin="round" strokeLinecap="round" />
        <line x1="10.5" y1="14" x2="15.5" y2="14" stroke="var(--accent-fg)" strokeWidth="2" strokeLinecap="round" />
      </svg>
      {!collapsed && (
        <div className="flex flex-col leading-[1.1]">
          <span className="font-semibold text-md tracking-[-0.01em]">Anima</span>
          <span className="text-xs text-fg-tertiary font-mono">
            lora studio{version ? ` · ${version}` : ''}
          </span>
        </div>
      )}
    </div>
  )
}

// ── nav item ───────────────────────────────────────────────────────────────
function navItemClass(active: boolean, collapsed: boolean, prominent: boolean): string {
  return [
    'flex w-full items-center gap-2.5 rounded-md no-underline transition-colors relative bg-transparent border-none cursor-pointer',
    prominent ? 'text-md' : 'text-sm',
    collapsed
      ? 'py-[9px] px-0 justify-center'
      : prominent ? 'py-2.5 px-3 justify-start' : 'py-2 px-3 justify-start',
    active
      ? 'bg-surface text-fg-primary font-semibold shadow-sm'
      : 'text-fg-secondary font-medium hover:bg-overlay',
  ].join(' ')
}

function NavItem({ to, label, icon, active, collapsed, prominent = false }: {
  to: string; label: string; icon: React.ReactNode; active: boolean; collapsed: boolean
  /** 顶级 tab 用更大字号 + 更大 padding，跟项目下属 sub-nav 区分。 */
  prominent?: boolean
}) {
  return (
    <Link
      to={to}
      title={collapsed ? label : undefined}
      className={navItemClass(active, collapsed, prominent)}
    >
      {active && !collapsed && (
        <span className="absolute left-0 top-2 bottom-2 w-[3px] bg-accent rounded-[2px]" />
      )}
      {icon}
      {!collapsed && <span className="flex-1">{label}</span>}
    </Link>
  )
}

/** NavItem 的 button 变体 —— 给设置抽屉用，不走路由。 */
function NavButton({ onClick, label, icon, active, collapsed, prominent = false }: {
  onClick: () => void; label: string; icon: React.ReactNode; active: boolean; collapsed: boolean
  prominent?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={collapsed ? label : undefined}
      className={navItemClass(active, collapsed, prominent) + ' text-left'}
    >
      {active && !collapsed && (
        <span className="absolute left-0 top-2 bottom-2 w-[3px] bg-accent rounded-[2px]" />
      )}
      {icon}
      {!collapsed && <span className="flex-1">{label}</span>}
    </button>
  )
}

// ── project info block (项目名 + active version label，放最上) ──────────────
function ProjectInfoBlock({ collapsed }: { collapsed: boolean }) {
  const ctx = useProjectCtx()
  if (!ctx) return null
  if (collapsed) return null
  const { project } = ctx
  return (
    <div className="px-3 py-1.5">
      <div className="font-semibold text-fg-primary text-sm overflow-hidden text-ellipsis whitespace-nowrap" title={project.title}>
        {project.title}
      </div>
      <div className="font-mono text-xs text-fg-tertiary mt-0.5 overflow-hidden text-ellipsis whitespace-nowrap" title={project.slug}>
        slug / {project.slug}
      </div>
    </div>
  )
}

// ── version picker block (header "训练 vX [+/-]" 始终显示，展开后下方再渲染完整 list) ─
function VersionPickerBlock({ collapsed }: { collapsed: boolean }) {
  const { t } = useTranslation()
  const ctx = useProjectCtx()
  const [expanded, setExpanded] = useState(false)
  if (!ctx) return null
  if (collapsed) return null
  const { project, activeVersion, onSelectVersion, onCreateVersion, onExportTrain, onDeleteVersion, exporting } = ctx

  const header = (
    <div className="flex items-center gap-2.5 rounded-md py-2 px-3 text-sm text-fg-secondary hover:bg-overlay transition-colors">
      <span className="w-5 h-5 grid place-items-center text-fg-tertiary shrink-0">
        {I.train}
      </span>
      <span className="flex-1 overflow-hidden text-ellipsis whitespace-nowrap">
        {t('sidebar.trainingVersionPrefix')} <span className="font-mono text-fg-primary">{activeVersion?.label ?? '—'}</span>
      </span>
      <button
        onClick={() => setExpanded((v) => !v)}
        title={expanded ? t('sidebar.collapseVersions') : t('sidebar.expandVersions')}
        className="w-5 h-5 grid place-items-center text-fg-tertiary text-sm bg-transparent border border-dim rounded-sm cursor-pointer hover:bg-surface hover:text-accent shrink-0"
      >
        {expanded ? '−' : '+'}
      </button>
    </div>
  )

  if (!expanded) return header

  // 展开态：header（[-]）+ 下方完整版本列表 + 新建/导出
  return (
    <>
      {header}
      <div className="rounded-md border border-subtle bg-overlay px-2 pt-2 pb-1.5 flex flex-col gap-1 mb-1">
      <div className="flex flex-col gap-px">
        {project.versions.map((v) => {
          const isActive = v.id === project.active_version_id
          return (
            <div key={v.id} className="flex items-center gap-0.5">
              <button
                onClick={() => onSelectVersion(v.id)}
                className={[
                  'flex-1 text-left px-1.5 py-0.5 rounded-sm font-mono text-xs flex items-center gap-1.5 border-none cursor-pointer transition-colors',
                  isActive
                    ? 'bg-accent-soft text-accent font-semibold'
                    : 'text-fg-secondary font-normal bg-transparent hover:bg-surface',
                ].join(' ')}
              >
                <span className={STATUS_DOT[v.status] ?? 'dot dot-neutral'} />
                <span className="flex-1 overflow-hidden text-ellipsis whitespace-nowrap">{v.label}</span>
              </button>
              {isActive && project.versions.length > 1 && (
                <button
                  onClick={() => onDeleteVersion(v.id)}
                  title={t('sidebar.deleteVersionTitle')}
                  className="px-[5px] py-0.5 text-fg-tertiary text-xs bg-transparent border-none cursor-pointer rounded-sm hover:text-err transition-colors shrink-0"
                >
                  ×
                </button>
              )}
            </div>
          )
        })}
      </div>

      <div className="flex gap-1 mt-0.5">
        <button
          onClick={() => onCreateVersion()}
          className="flex-1 flex items-center justify-center gap-1 py-1 px-1.5 text-xs text-fg-secondary bg-transparent border border-dashed border-dim rounded-sm cursor-pointer hover:bg-surface hover:text-accent transition-colors"
        >
          {I.plus} {t('sidebar.newVersion')}
        </button>
        <button
          onClick={onExportTrain}
          disabled={!activeVersion || exporting}
          title={exporting ? t('sidebar.exporting') : t('sidebar.exportTitle')}
          className={`flex items-center justify-center gap-1 py-1 px-2 text-xs text-fg-secondary bg-transparent border border-dim rounded-sm cursor-pointer hover:bg-surface hover:text-fg-primary transition-colors ${!activeVersion ? 'opacity-40' : ''}`}
        >
          {I.export}
          {exporting ? t('sidebar.exporting') : t('sidebar.export')}
        </button>
      </div>
      </div>
    </>
  )
}

// ── project stepper nav ────────────────────────────────────────────────────
function ProjectStepperNav({ pid, activeVid, currentStep, version, collapsed }: {
  pid: string
  activeVid: string | null
  currentStep: string | null
  version: Version | null
  collapsed: boolean
}) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { toast } = useToast()

  // 项目级 ①② 跟"概览"同款：圆盘里放 icon、无序号、无完成绿色态。
  // version 级 phase 重编号 1-5（每个 version 自己一段流水线）。
  const STEPS = [
    { key: 'download',   labelKey: 'nav.download',   idx: '',  icon: I.download, scope: 'project' as const },
    { key: 'preprocess', labelKey: 'nav.preprocess', idx: '',  icon: I.upscale,  scope: 'project' as const },
    { key: 'curate',     labelKey: 'nav.curate',     idx: '1', icon: I.filter,   scope: 'version' as const },
    { key: 'tag',        labelKey: 'nav.tag',        idx: '2', icon: I.tag,      scope: 'version' as const },
    { key: 'edit',       labelKey: 'nav.tagEdit',    idx: '3', icon: I.edit,     scope: 'version' as const },
    { key: 'reg',        labelKey: 'nav.reg',        idx: '4', icon: I.reg,      scope: 'version' as const },
    { key: 'train',      labelKey: 'nav.train',      idx: '5', icon: I.train,    scope: 'version' as const },
  ]

  const overviewActive = currentStep === null
  // ADR-0007 §11.5 cursor 派生：cursor 之前的 phase = done（仅 version 级）。
  // 项目级 ①② 不参与 cursor、不显示完成态。
  const cursorPhase: VersionPhase = (version?.phase as VersionPhase | undefined) ?? 'curating'
  const cursorIdx = PHASE_ORDER.indexOf(cursorPhase)

  const isStepDone = (key: string): boolean => {
    const phase = STEP_KEY_TO_PHASE[key]
    if (!phase) return false
    return PHASE_ORDER.indexOf(phase) < cursorIdx
  }

  // ADR-0007 §11.5-A: 推进 cursor 的唯一入口 —— 点击 cursor+1 那一行触发。
  // skippable 调 skip，必经调 advance；校验失败给 warning toast。
  const handleAdvanceToNext = async () => {
    if (!activeVid) return
    const nextIdx = cursorIdx + 1
    if (nextIdx >= PHASE_ORDER.length) return
    const nextPhase = PHASE_ORDER[nextIdx]
    const isSkippable = PHASE_SKIPPABLE.includes(cursorPhase)
    try {
      const res = isSkippable
        ? await api.skipVersionPhase(Number(pid), Number(activeVid))
        : await api.advanceVersionPhase(Number(pid), Number(activeVid))
      if (!res.ok) {
        toast(res.reason || t('sidebar.advanceFailed'), 'error')
        return
      }
      navigate(`/projects/${pid}/v/${activeVid}/${PHASE_TO_STEP_KEY[nextPhase]}`)
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const linkCls = (active: boolean, indent = false) => [
    'flex items-center gap-2.5 rounded-md text-sm no-underline transition-colors',
    collapsed ? 'py-2 px-0 justify-center' : 'py-2 px-3 justify-start',
    !collapsed && indent ? 'ml-3' : '',
    active ? 'bg-surface text-fg-primary font-semibold shadow-sm' : 'text-fg-secondary font-normal hover:bg-overlay',
  ].join(' ')

  return (
    <div className="flex flex-col gap-px">
      {/* 项目名 + 当前 version label，放最顶 */}
      <ProjectInfoBlock collapsed={collapsed} />

      <Link
        to={`/projects/${pid}`}
        title={collapsed ? t('nav.overview') : undefined}
        className={linkCls(overviewActive)}
      >
        <span className={`w-5 h-5 rounded-full grid place-items-center text-[12px] shrink-0 ${overviewActive ? 'bg-accent-soft text-accent' : 'bg-overlay text-fg-tertiary'}`}>
          ≡
        </span>
        {!collapsed && <span className="flex-1">{t('nav.overview')}</span>}
      </Link>

      {STEPS.map((s, i) => {
        const label = t(s.labelKey)
        const isActive = s.key === currentStep
        const isProject = s.scope === 'project'
        const phase = STEP_KEY_TO_PHASE[s.key]
        const phaseIdx = phase != null ? PHASE_ORDER.indexOf(phase) : -1
        // ADR-0007 §11.5-A: sidebar 是 cursor 推进主通道。
        // - phase_idx < cursorIdx  → done (绿数字，Link)
        // - phase_idx == cursorIdx → 当前 cursor (Link)
        // - phase_idx == cursorIdx+1 → "下一步"，呼吸 button，点击触发 advance/skip
        // - phase_idx > cursorIdx+1 → disabled
        const isCursorCurrent = !isProject && phaseIdx === cursorIdx
        const isNextStep = !isProject && phaseIdx === cursorIdx + 1
        const isFuturePhase = !isProject && phaseIdx > cursorIdx + 1
        const isDone = isProject ? false : isStepDone(s.key)

        const href = (isFuturePhase || isNextStep)
          ? null
          : isProject
            ? `/projects/${pid}/${s.key}`
            : activeVid ? `/projects/${pid}/v/${activeVid}/${s.key}` : null

        // 视觉强度：cursor 当前（实心 accent）> cursor+1（弱底 + 静态 ring）> 已完成（绿）> 默认
        // 注：之前 cursor+1 用 animate-pulse 反而盖过 cursor 当前，去掉动画；改用 ring 静态提示。
        const badgeCls = isDone
          ? 'bg-ok-soft text-ok'
          : isCursorCurrent
            ? 'bg-accent text-accent-fg'
            : isNextStep
              ? 'bg-overlay text-fg-secondary ring-1 ring-accent'
              : isActive
                ? 'bg-accent-soft text-accent'
                : 'bg-overlay text-fg-tertiary'

        // 项目级 ①② badge 内放 icon（跟"概览" ≡ 同款），不要数字 / 不要绿色态。
        // version 级 badge 始终放数字；完成时数字变绿；cursor+1 时背景 accent + 呼吸。
        const badgeContent = isProject ? s.icon : s.idx

        const inner = (
          <>
            <span className={`w-5 h-5 rounded-full grid place-items-center text-[10px] font-bold font-mono shrink-0 ${badgeCls}`}>
              {badgeContent}
            </span>
            {!collapsed && <span className="flex-1 text-left">{label}</span>}
            {!collapsed && isActive && <span className="dot dot-running" />}
          </>
        )

        // version 级 step（筛选→训练）整体再缩进一层，表达从属于上方的 version 选择器。
        const indent = s.scope === 'version'

        let stepNode: React.ReactNode
        if (isNextStep) {
          // 推进入口：button + onClick
          stepNode = (
            <button
              key={s.key}
              type="button"
              onClick={() => void handleAdvanceToNext()}
              title={collapsed ? `→ ${label}` : t('sidebar.advanceToTitle', { label })}
              className={linkCls(false, indent) + ' cursor-pointer text-fg-primary font-medium'}
            >
              {inner}
            </button>
          )
        } else if (!href) {
          stepNode = (
            <span key={s.key} title={collapsed ? (s.idx ? `${s.idx}. ${label}` : label) : undefined}
              className={linkCls(false, indent) + ' opacity-40 cursor-default'}>
              {inner}
            </span>
          )
        } else {
          stepNode = (
            <Link
              key={s.key}
              to={href}
              title={collapsed ? (s.idx ? `${s.idx}. ${label}` : label) : undefined}
              className={linkCls(isActive, indent)}
            >
              {inner}
            </Link>
          )
        }

        // 在 scope 从 project 切到 version 的边界（"预处理"和"筛选"之间）
        // 插入 VersionPickerBlock —— 版本选择紧靠 version 级 phase 上方。
        const prev = STEPS[i - 1]
        const isBoundary = prev && prev.scope === 'project' && s.scope === 'version'
        if (!isBoundary) return stepNode
        return (
          <Fragment key={s.key}>
            <VersionPickerBlock collapsed={collapsed} />
            {stepNode}
          </Fragment>
        )
      })}
    </div>
  )
}

// ── theme toggle ───────────────────────────────────────────────────────────
function ThemeToggle({ collapsed }: { collapsed: boolean }) {
  const { t } = useTranslation()
  const [theme, setTheme] = useState<Theme>(() => getStoredTheme())

  const handleToggle = () => {
    setTheme(toggleTheme())
  }

  const isDark = theme === 'dark'
  return (
    <button
      onClick={handleToggle}
      title={isDark ? t('sidebar.switchToLight') : t('sidebar.switchToDark')}
      className={[
        'flex items-center gap-2.5 rounded-md text-sm no-underline transition-colors bg-transparent border-none cursor-pointer w-full',
        collapsed ? 'py-[9px] px-0 justify-center' : 'py-2 px-3 justify-start',
        'text-fg-secondary font-medium hover:bg-overlay',
      ].join(' ')}
    >
      {isDark ? I.sun : I.moon}
      {!collapsed && <span>{isDark ? t('sidebar.themeLight') : t('sidebar.themeDark')}</span>}
    </button>
  )
}

// ── sidebar ────────────────────────────────────────────────────────────────
const SIDEBAR_KEY = 'studio.sidebar.expanded'

export default function Sidebar() {
  const { t } = useTranslation()
  const location = useLocation()
  const ctx = useProjectCtx()
  const settingsDrawer = useSettingsDrawer()

  const pid = location.pathname.match(/^\/projects\/([^/]+)/)?.[1] ?? null
  const urlVid = location.pathname.match(/\/v\/([^/]+)/)?.[1] ?? null
  const stepMatch = location.pathname.match(/\/v\/[^/]+\/([^/]+)$/)
  const projectScopeStep = location.pathname.match(/^\/projects\/[^/]+\/(download|preprocess)$/)?.[1] ?? null
  const currentStep = stepMatch?.[1] ?? projectScopeStep

  const activeVid = ctx?.activeVersion?.id?.toString() ?? urlVid

  const inProject = pid !== null

  const [expandedOverride, setExpandedOverride] = useState<boolean | null>(() => {
    try {
      const v = sessionStorage.getItem(SIDEBAR_KEY)
      return v === '1' ? true : v === '0' ? false : null
    } catch { return null }
  })

  const expanded = expandedOverride ?? true
  const collapsed = !expanded

  const toggle = () => {
    const next = !expanded
    setExpandedOverride(next)
    try { sessionStorage.setItem(SIDEBAR_KEY, next ? '1' : '0') } catch { /* ignore */ }
  }

  const isMain = (path: string) => {
    if (path === '/') return location.pathname === '/'
    return location.pathname.startsWith(path)
  }

  return (
    <aside
      className="shrink-0 bg-sunken border-r border-subtle flex flex-col overflow-hidden h-full transition-[width] duration-[160ms] ease-in-out"
      style={{ width: collapsed ? 'var(--sidebar-collapsed-w)' : 'var(--sidebar-w)' }}
    >
      <div
        className={`flex items-center border-b border-subtle shrink-0 px-3.5 ${collapsed ? 'justify-center' : ''}`}
        style={{ height: 'var(--topbar-h)' }}
      >
        <Logo collapsed={collapsed} />
      </div>

      <nav className={`flex-1 flex flex-col gap-0.5 overflow-hidden ${collapsed ? 'px-2 py-2.5' : 'px-2 py-3.5'}`}>
        <NavItem to="/" label={t('nav.projects')} icon={I.folder} active={!inProject && location.pathname === '/'} collapsed={collapsed} prominent />

        {/* 当前项目下的全部内容（概览 + ①② + VersionPanel + ③-⑦）夹在 项目 / 队列 之间。
            sub-nav 性质，缩进表达从属（折叠态不缩进）。
            VersionPanel 由 ProjectStepperNav 在 project→version scope 切换点插入。 */}
        {inProject && pid && (
          <div className={`flex flex-col gap-0.5 ${collapsed ? '' : 'ml-3'}`}>
            <ProjectStepperNav pid={pid} activeVid={activeVid} currentStep={currentStep} version={ctx?.activeVersion ?? null} collapsed={collapsed} />
          </div>
        )}

        <NavItem to="/queue" label={t('nav.queue')} icon={I.queue} active={isMain('/queue')} collapsed={collapsed} prominent />
        <NavItem to="/tools/generate" label={t('nav.generate')} icon={I.image} active={isMain('/tools/generate')} collapsed={collapsed} prominent />
      </nav>

      <div className={`border-t border-subtle flex flex-col gap-0.5 shrink-0 ${collapsed ? 'px-1.5 py-2' : 'p-2.5'}`}>
        <NavItem to="/tools/presets" label={t('nav.presets')} icon={I.preset} active={isMain('/tools/presets')} collapsed={collapsed} />
        <NavItem to="/tools/monitor" label={t('nav.monitor')} icon={I.monitor} active={isMain('/tools/monitor')} collapsed={collapsed} />
        <NavButton onClick={() => settingsDrawer.open()} label={t('nav.settings')} icon={I.cog} active={settingsDrawer.isOpen} collapsed={collapsed} />
        <ThemeToggle collapsed={collapsed} />
        <button
          onClick={toggle}
          title={collapsed ? t('sidebar.expand') : t('sidebar.collapse')}
          className={`text-fg-tertiary bg-transparent border-none rounded cursor-pointer hover:bg-overlay transition-colors ${collapsed ? 'flex justify-center p-2 mt-1' : 'flex items-center gap-1.5 p-2 mt-1 text-xs'}`}
        >
          {collapsed ? I.chevR : <>{I.chevL}<span>{t('sidebar.collapseLabel')}</span></>}
        </button>
      </div>
    </aside>
  )
}
