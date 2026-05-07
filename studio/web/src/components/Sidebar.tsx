import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import type { Version, VersionStage } from '../api/client'
import { getStoredTheme, toggleTheme, type Theme } from '../lib/theme'

/** Map version stage → 0-based index of the current active step.
 *
 * 注意：后端 stage 集合是 {curating, tagging, regularizing, ready, training,
 * done}，**没有 editing 这个值**——打标完成后到正则启动前，stage 一直停在
 * "tagging"。所以单看 stage，tag(2) 和 edit(3) 都不会变 done 状态。下方
 * `isStepDone` 用 version.stats 派生覆盖，做到打完标 → tag/edit 立刻打勾。
 */
const STAGE_TO_STEP_IDX: Record<VersionStage, number> = {
  curating: 1,     // download done, curate active
  tagging: 2,      // download+curate done, tag active
  regularizing: 4, // download+curate+tag+edit done, reg active
  ready: 5,        // download+curate+tag+edit+reg done, train active
  training: 5,     // same, train running
  done: 6,         // all 6 steps done
}
import { useProjectCtx } from '../context/ProjectContext'

// ── icons ──────────────────────────────────────────────────────────────────
const I = {
  folder:  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>,
  queue:   <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M4 6h16M4 12h10M4 18h16"/><circle cx="18" cy="12" r="2" fill="currentColor"/></svg>,
  preset:  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><line x1="6" y1="4" x2="6" y2="20"/><line x1="12" y1="4" x2="12" y2="20"/><line x1="18" y1="4" x2="18" y2="20"/><circle cx="6" cy="9" r="2" fill="var(--bg-sunken)"/><circle cx="12" cy="15" r="2" fill="var(--bg-sunken)"/><circle cx="18" cy="7" r="2" fill="var(--bg-sunken)"/></svg>,
  monitor: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M3 17l4-6 4 3 5-9 5 7"/></svg>,
  generate:<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/></svg>,
  cog:     <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h0a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>,
  check:   <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="m4 12 5 5 11-12"/></svg>,
  chevL:   <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M15 6l-6 6 6 6"/></svg>,
  chevR:   <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M9 6l6 6-6 6"/></svg>,
  download:<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 4v12m0 0-4-4m4 4 4-4M4 20h16"/></svg>,
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

// ── version stage dot ──────────────────────────────────────────────────────
const STAGE_DOT: Record<VersionStage, string> = {
  curating:     'dot dot-warn',
  tagging:      'dot dot-warn',
  regularizing: 'dot dot-warn',
  ready:        'dot dot-ok',
  training:     'dot dot-running',
  done:         'dot dot-ok',
}

// ── logo ───────────────────────────────────────────────────────────────────
function Logo({ collapsed }: { collapsed: boolean }) {
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
          <span className="text-xs text-fg-tertiary font-mono">lora studio · 0.4</span>
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
        'flex items-center gap-2.5 rounded-md text-sm no-underline transition-colors relative',
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
  const ctx = useProjectCtx()
  if (!ctx) return null
  const { project, activeVersion, onSelectVersion, onCreateVersion, onExportTrain, onDeleteVersion, exporting } = ctx

  if (collapsed) {
    return (
      <button
        onClick={onExportTrain}
        disabled={!activeVersion || exporting}
        title={exporting ? '打包中...' : '导出训练集'}
        className={`flex items-center justify-center py-[7px] w-full rounded-md text-fg-tertiary bg-transparent border-none cursor-pointer hover:bg-overlay transition-colors ${!activeVersion ? 'opacity-40' : ''}`}
      >
        {I.export}
      </button>
    )
  }

  return (
    <div className="rounded-md border border-subtle bg-overlay px-2 pt-2 pb-1.5 flex flex-col gap-1">
      {/* Project name header */}
      <div className="px-0.5">
        <div className="font-semibold text-fg-primary text-sm">
          {project.title}
        </div>
        <div className="font-mono text-xs text-fg-tertiary mt-0.5">
          v / {activeVersion?.label ?? '—'}
        </div>
      </div>

      {/* Version list */}
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
                <span className={STAGE_DOT[v.stage]} />
                <span className="flex-1 overflow-hidden text-ellipsis whitespace-nowrap">{v.label}</span>
              </button>
              {isActive && project.versions.length > 1 && (
                <button
                  onClick={() => onDeleteVersion(v.id)}
                  title="删除此版本（移到回收站）"
                  className="px-[5px] py-0.5 text-fg-tertiary text-xs bg-transparent border-none cursor-pointer rounded-sm hover:text-err transition-colors shrink-0"
                >
                  ×
                </button>
              )}
            </div>
          )
        })}
      </div>

      {/* Actions row */}
      <div className="flex gap-1 mt-0.5">
        <button
          onClick={onCreateVersion}
          className="flex-1 flex items-center justify-center gap-1 py-1 px-1.5 text-xs text-fg-secondary bg-transparent border border-dashed border-dim rounded-sm cursor-pointer hover:bg-surface hover:text-accent transition-colors"
        >
          {I.plus} 新版本
        </button>
        <button
          onClick={onExportTrain}
          disabled={!activeVersion || exporting}
          title={exporting ? '打包中...' : '导出当前版本训练集 (.zip)'}
          className={`flex items-center justify-center gap-1 py-1 px-2 text-xs text-fg-secondary bg-transparent border border-dim rounded-sm cursor-pointer hover:bg-surface hover:text-fg-primary transition-colors ${!activeVersion ? 'opacity-40' : ''}`}
        >
          {I.export}
          {exporting ? '打包...' : '导出'}
        </button>
      </div>
    </div>
  )
}

// ── project stepper nav ────────────────────────────────────────────────────
const STEPS = [
  { key: 'download', label: '下载',     idx: '1', icon: I.download },
  { key: 'curate',   label: '筛选',     idx: '2', icon: I.filter },
  { key: 'tag',      label: '打标',     idx: '3', icon: I.tag },
  { key: 'edit',     label: '标签编辑', idx: '4', icon: I.edit },
  { key: 'reg',      label: '正则集',   idx: '5', icon: I.reg },
  { key: 'train',    label: '训练',     idx: '6', icon: I.train },
]

function ProjectStepperNav({ pid, activeVid, currentStep, version, collapsed }: {
  pid: string
  activeVid: string | null
  currentStep: string | null
  version: Version | null
  collapsed: boolean
}) {
  const overviewActive = currentStep === null
  const stage: VersionStage = version?.stage ?? 'curating'
  const stageStepIdx = STAGE_TO_STEP_IDX[stage] ?? 0
  const stats = version?.stats

  // 派生覆盖（stats / output_lora_path）：打完标 → tag+edit 立即 done；
  // 正则集生成 → reg 立即 done；output_lora_path 存在 → train done。
  // 这样不依赖后端 stage 跳转就能让侧边的勾勾跟上数据真相。
  const isStepDone = (key: string, idx: number): boolean => {
    if (idx < stageStepIdx) return true
    if (
      (key === 'tag' || key === 'edit') &&
      stats &&
      stats.train_image_count > 0 &&
      stats.tagged_image_count >= stats.train_image_count
    ) return true
    if (
      key === 'reg' &&
      stats &&
      stats.reg_meta_exists &&
      stats.reg_image_count > 0
    ) return true
    if (key === 'train' && version?.output_lora_path) return true
    return false
  }

  const linkCls = (active: boolean) => [
    'flex items-center gap-2.5 rounded-md text-sm no-underline transition-colors',
    collapsed ? 'py-2 px-0 justify-center' : 'py-2 px-3 justify-start',
    active ? 'bg-surface text-fg-primary font-semibold shadow-sm' : 'text-fg-secondary font-normal hover:bg-overlay',
  ].join(' ')

  return (
    <div className="flex flex-col gap-px">
      {/* 概览 */}
      <Link
        to={`/projects/${pid}`}
        title={collapsed ? '概览' : undefined}
        className={linkCls(overviewActive) + ' mb-1'}
      >
        <span className={`w-5 h-5 rounded-full grid place-items-center text-[12px] shrink-0 ${overviewActive ? 'bg-accent-soft text-accent' : 'bg-overlay text-fg-tertiary'}`}>
          ≡
        </span>
        {!collapsed && <span className="flex-1">概览</span>}
      </Link>

      {STEPS.map((s, i) => {
        const isActive = s.key === currentStep
        const isDone = isStepDone(s.key, i)

        const href = s.key === 'download'
          ? `/projects/${pid}/download`
          : activeVid ? `/projects/${pid}/v/${activeVid}/${s.key}` : null

        const badgeCls = isDone
          ? 'bg-ok-soft text-ok'
          : isActive
            ? 'bg-accent-soft text-accent'
            : 'bg-overlay text-fg-tertiary'

        const inner = (
          <>
            <span className={`w-5 h-5 rounded-full grid place-items-center text-[10px] font-bold font-mono shrink-0 ${badgeCls}`}>
              {isDone ? I.check : s.idx}
            </span>
            {!collapsed && <span className="flex-1 text-left">{s.label}</span>}
            {!collapsed && isActive && <span className="dot dot-running" />}
          </>
        )

        if (!href) {
          return (
            <span key={s.key} title={collapsed ? `${s.idx}. ${s.label}` : undefined}
              className={linkCls(false) + ' opacity-40 cursor-default'}>
              {inner}
            </span>
          )
        }

        return (
          <Link
            key={s.key}
            to={href}
            title={collapsed ? `${s.idx}. ${s.label}` : undefined}
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
  const [theme, setTheme] = useState<Theme>(() => getStoredTheme())

  const handleToggle = () => {
    setTheme(toggleTheme())
  }

  const isDark = theme === 'dark'
  return (
    <button
      onClick={handleToggle}
      title={isDark ? '切到日间模式' : '切到暗色模式'}
      className={[
        'flex items-center gap-2.5 rounded-md text-sm no-underline transition-colors bg-transparent border-none cursor-pointer w-full',
        collapsed ? 'py-[9px] px-0 justify-center' : 'py-2 px-3 justify-start',
        'text-fg-secondary font-medium hover:bg-overlay',
      ].join(' ')}
    >
      {isDark ? I.sun : I.moon}
      {!collapsed && <span>{isDark ? '日间模式' : '暗色模式'}</span>}
    </button>
  )
}

// ── sidebar ────────────────────────────────────────────────────────────────
const SIDEBAR_KEY = 'studio.sidebar.expanded'

export default function Sidebar() {
  const location = useLocation()
  const ctx = useProjectCtx()

  const pid = location.pathname.match(/^\/projects\/([^/]+)/)?.[1] ?? null
  const urlVid = location.pathname.match(/\/v\/([^/]+)/)?.[1] ?? null
  const stepMatch = location.pathname.match(/\/v\/[^/]+\/([^/]+)$/)
  const currentStep = stepMatch?.[1] ?? (location.pathname.endsWith('/download') ? 'download' : null)

  // Prefer active version from context; fall back to URL vid (handles page reload)
  const activeVid = ctx?.activeVersion?.id?.toString() ?? urlVid

  const inProject = pid !== null

  const [expandedOverride, setExpandedOverride] = useState<boolean | null>(() => {
    try {
      const v = sessionStorage.getItem(SIDEBAR_KEY)
      return v === '1' ? true : v === '0' ? false : null
    } catch { return null }
  })

  const expanded = expandedOverride ?? !inProject
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
      {/* header / logo */}
      <div
        className={`flex items-center border-b border-subtle shrink-0 px-3.5 ${collapsed ? 'justify-center' : ''}`}
        style={{ height: 'var(--topbar-h)' }}
      >
        <Logo collapsed={collapsed} />
      </div>

      {/* main nav */}
      <nav className={`flex-1 flex flex-col gap-0.5 overflow-hidden ${collapsed ? 'px-2 py-2.5' : 'px-2 py-3.5'}`}>
        <NavItem to="/" label="项目" icon={I.folder} active={!inProject && location.pathname === '/'} collapsed={collapsed} />
        <NavItem to="/queue" label="队列" icon={I.queue} active={isMain('/queue')} collapsed={collapsed} />

        {inProject && pid && (
          <div className="mt-2.5 flex flex-col gap-1">
            {/* Version selector + export with project name embedded */}
            <VersionPanel collapsed={collapsed} />

            <ProjectStepperNav pid={pid} activeVid={activeVid} currentStep={currentStep} version={ctx?.activeVersion ?? null} collapsed={collapsed} />
          </div>
        )}
      </nav>

      {/* tools + toggle */}
      <div className={`border-t border-subtle flex flex-col gap-0.5 shrink-0 ${collapsed ? 'px-1.5 py-2' : 'p-2.5'}`}>
        <NavItem to="/tools/presets" label="预设" icon={I.preset} active={isMain('/tools/presets')} collapsed={collapsed} />
        <NavItem to="/tools/generate" label="生成" icon={I.generate} active={isMain('/tools/generate')} collapsed={collapsed} />
        <NavItem to="/tools/monitor" label="监控" icon={I.monitor} active={isMain('/tools/monitor')} collapsed={collapsed} />
        <NavItem to="/tools/settings" label="设置" icon={I.cog} active={isMain('/tools/settings')} collapsed={collapsed} />
        <ThemeToggle collapsed={collapsed} />
        <button
          onClick={toggle}
          title={collapsed ? '展开' : '折叠'}
          className={`text-fg-tertiary bg-transparent border-none rounded cursor-pointer hover:bg-overlay transition-colors ${collapsed ? 'flex justify-center p-2 mt-1' : 'flex items-center gap-1.5 p-2 mt-1 text-xs'}`}
        >
          {collapsed ? I.chevR : <>{I.chevL}<span>收起</span></>}
        </button>
      </div>
    </aside>
  )
}
