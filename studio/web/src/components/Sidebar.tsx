import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useLocation } from 'react-router-dom'
import { api, PHASE_ORDER, type ProjectDetail, type Version, type VersionPhase, type VersionStatus } from '../api/client'
import { getStoredTheme, toggleTheme, type Theme } from '../lib/theme'

/** ADR-0007 §11.2 / §11.5-A: 把 STEPS 的 version-scope step key 映射到 phase enum。
 *
 * STEPS 顺序：0 download / 1 preprocess / 2 curate / 3 tag / 4 edit / 5 reg / 6 train
 * phase enum: curating → tagging → editing → regularizing → ready（PR-3 加）
 *
 * 完成判定（§11.5）：step 的 phase index < version.phase 的 cursor index → 已完成。
 * cursor 之后 disabled（§11.5-A）；cursor 当前 = active；cursor 之前 = done。
 *
 * preprocess 是项目级（§6.1），不进 version phase；用 preprocess_image_count > 0 派生。
 */
const STEP_KEY_TO_PHASE: Record<string, VersionPhase> = {
  curate: 'curating',
  tag:    'tagging',
  edit:   'editing',
  reg:    'regularizing',
  train:  'ready',
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
function NavItem({ to, label, icon, active, collapsed }: {
  to: string; label: string; icon: React.ReactNode; active: boolean; collapsed: boolean
}) {
  return (
    <Link
      to={to}
      title={collapsed ? label : undefined}
      className={[
        'flex w-full items-center gap-2.5 rounded-md text-sm no-underline transition-colors relative',
        collapsed ? 'py-[9px] px-0 justify-center' : 'py-2 px-3 justify-start',
        active
          ? 'bg-surface text-fg-primary font-semibold shadow-sm'
          : 'text-fg-secondary font-medium hover:bg-overlay',
      ].join(' ')}
    >
      {active && !collapsed && (
        <span className="absolute left-0 top-2 bottom-2 w-[3px] bg-accent rounded-[2px]" />
      )}
      {icon}
      {!collapsed && <span className="flex-1">{label}</span>}
    </Link>
  )
}

// ── version panel ──────────────────────────────────────────────────────────
function VersionPanel({ collapsed }: { collapsed: boolean }) {
  const { t } = useTranslation()
  const ctx = useProjectCtx()
  if (!ctx) return null
  const { project, activeVersion, onSelectVersion, onCreateVersion, onExportTrain, onDeleteVersion, exporting } = ctx

  if (collapsed) return null

  return (
    <div className="rounded-md border border-subtle bg-overlay px-2 pt-2 pb-1.5 flex flex-col gap-1">
      <div className="px-0.5">
        <div className="font-semibold text-fg-primary text-sm">
          {project.title}
        </div>
        <div className="font-mono text-xs text-fg-tertiary mt-0.5">
          v / {activeVersion?.label ?? '—'}
        </div>
      </div>

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
          onClick={onCreateVersion}
          className="flex-1 flex items-center justify-center gap-1 py-1 px-1.5 text-xs text-fg-secondary bg-transparent border border-dashed border-dim rounded-sm cursor-pointer hover:bg-surface hover:text-accent transition-colors"
        >
          {I.plus} {t('sidebar.newVersion')}
        </button>
        <button
          onClick={onExportTrain}
          disabled={!activeVersion || exporting}
          title={exporting ? t('sidebar.exporting') : t('sidebar.deleteVersionTitle')}
          className={`flex items-center justify-center gap-1 py-1 px-2 text-xs text-fg-secondary bg-transparent border border-dim rounded-sm cursor-pointer hover:bg-surface hover:text-fg-primary transition-colors ${!activeVersion ? 'opacity-40' : ''}`}
        >
          {I.export}
          {exporting ? t('sidebar.exporting') : t('sidebar.export')}
        </button>
      </div>
    </div>
  )
}

// ── project stepper nav ────────────────────────────────────────────────────
function ProjectStepperNav({ pid, activeVid, currentStep, project, version, collapsed }: {
  pid: string
  activeVid: string | null
  currentStep: string | null
  project: ProjectDetail | null
  version: Version | null
  collapsed: boolean
}) {
  const { t } = useTranslation()

  const STEPS = [
    { key: 'download',   labelKey: 'nav.download',   idx: '1', icon: I.download, scope: 'project' as const },
    { key: 'preprocess', labelKey: 'nav.preprocess',  idx: '2', icon: I.upscale,  scope: 'project' as const },
    { key: 'curate',     labelKey: 'nav.curate',      idx: '3', icon: I.filter,   scope: 'version' as const },
    { key: 'tag',        labelKey: 'nav.tag',         idx: '4', icon: I.tag,      scope: 'version' as const },
    { key: 'edit',       labelKey: 'nav.tagEdit',     idx: '5', icon: I.edit,     scope: 'version' as const },
    { key: 'reg',        labelKey: 'nav.reg',         idx: '6', icon: I.reg,      scope: 'version' as const },
    { key: 'train',      labelKey: 'nav.train',       idx: '7', icon: I.train,    scope: 'version' as const },
  ]

  const overviewActive = currentStep === null
  const downloadCount = project?.download_image_count ?? 0
  const preprocessCount = project?.preprocess_image_count ?? 0

  // ADR-0007 §11.2 派生：cursor 之前的 phase = done
  const cursorPhase: VersionPhase = (version?.phase as VersionPhase | undefined) ?? 'curating'
  const cursorIdx = PHASE_ORDER.indexOf(cursorPhase)

  const isStepDone = (key: string): boolean => {
    if (key === 'download') return downloadCount > 0
    if (key === 'preprocess') return preprocessCount > 0
    const phase = STEP_KEY_TO_PHASE[key]
    if (!phase) return false
    return PHASE_ORDER.indexOf(phase) < cursorIdx
  }

  const linkCls = (active: boolean) => [
    'flex items-center gap-2.5 rounded-md text-sm no-underline transition-colors',
    collapsed ? 'py-2 px-0 justify-center' : 'py-2 px-3 justify-start',
    active ? 'bg-surface text-fg-primary font-semibold shadow-sm' : 'text-fg-secondary font-normal hover:bg-overlay',
  ].join(' ')

  return (
    <div className="flex flex-col gap-px">
      <Link
        to={`/projects/${pid}`}
        title={collapsed ? t('nav.overview') : undefined}
        className={linkCls(overviewActive) + ' mb-1'}
      >
        <span className={`w-5 h-5 rounded-full grid place-items-center text-[12px] shrink-0 ${overviewActive ? 'bg-accent-soft text-accent' : 'bg-overlay text-fg-tertiary'}`}>
          ≡
        </span>
        {!collapsed && <span className="flex-1">{t('nav.overview')}</span>}
      </Link>

      {STEPS.map((s) => {
        const label = t(s.labelKey)
        const isActive = s.key === currentStep
        const isDone = isStepDone(s.key)
        // ADR-0007 §11.2: 数字本身变绿 = phase 已完成；当前页 = active 整行高亮。
        const numColorCls = isDone
          ? 'text-ok'
          : isActive
            ? 'text-accent'
            : 'text-fg-tertiary'

        const href = s.scope === 'project'
          ? `/projects/${pid}/${s.key}`
          : activeVid ? `/projects/${pid}/v/${activeVid}/${s.key}` : null

        // 项目级 ①② 文案带 (N) 文件数；version 级保持纯 label
        let labelText = label
        if (s.key === 'download') labelText = `${label} (${downloadCount})`
        else if (s.key === 'preprocess') labelText = `${label} (${preprocessCount})`

        const inner = (
          <>
            <span className={`w-5 h-5 grid place-items-center text-[13px] font-bold font-mono shrink-0 ${numColorCls}`}>
              {s.idx}
            </span>
            {!collapsed && <span className="flex-1 text-left">{labelText}</span>}
            {!collapsed && isActive && <span className="dot dot-running" />}
          </>
        )

        if (!href) {
          return (
            <span key={s.key} title={collapsed ? `${s.idx}. ${label}` : undefined}
              className={linkCls(false) + ' opacity-40 cursor-default'}>
              {inner}
            </span>
          )
        }

        return (
          <Link
            key={s.key}
            to={href}
            title={collapsed ? `${s.idx}. ${label}` : undefined}
            className={linkCls(isActive)}
          >
            {inner}
          </Link>
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
        <NavItem to="/" label={t('nav.projects')} icon={I.folder} active={!inProject && location.pathname === '/'} collapsed={collapsed} />
        <NavItem to="/queue" label={t('nav.queue')} icon={I.queue} active={isMain('/queue')} collapsed={collapsed} />
        <NavItem to="/tools/generate" label={t('nav.generate')} icon={I.image} active={isMain('/tools/generate')} collapsed={collapsed} />

        {inProject && pid && (
          <div className="mt-2.5 flex flex-col gap-1">
            <VersionPanel collapsed={collapsed} />
            <ProjectStepperNav pid={pid} activeVid={activeVid} currentStep={currentStep} project={ctx?.project ?? null} version={ctx?.activeVersion ?? null} collapsed={collapsed} />
          </div>
        )}
      </nav>

      <div className={`border-t border-subtle flex flex-col gap-0.5 shrink-0 ${collapsed ? 'px-1.5 py-2' : 'p-2.5'}`}>
        <NavItem to="/tools/presets" label={t('nav.presets')} icon={I.preset} active={isMain('/tools/presets')} collapsed={collapsed} />
        <NavItem to="/tools/monitor" label={t('nav.monitor')} icon={I.monitor} active={isMain('/tools/monitor')} collapsed={collapsed} />
        <NavItem to="/tools/settings" label={t('nav.settings')} icon={I.cog} active={isMain('/tools/settings')} collapsed={collapsed} />
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
