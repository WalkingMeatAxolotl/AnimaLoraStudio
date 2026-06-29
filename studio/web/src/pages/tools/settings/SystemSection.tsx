import { useEffect, useMemo, useState } from 'react'
import type { TFunction } from 'i18next'
import { Trans, useTranslation } from 'react-i18next'
import {
  api,
  type DevCommit,
  type DevCommitsResult,
  type ModelsRootInfo,
  type PreflightResult,
  type StudioDataInfo,
  type SystemPrefsConfig,
  type SystemUpdateCheck,
  type SystemUpdateStatus,
  type SystemVersion,
} from '../../../api/client'
import { useAnnouncements, extractReleaseHighlights } from '../../../lib/Announcements'
import { useDialog } from '../../../components/Dialog'
import {
  clearOnboardingDone,
  ONBOARDING_EVENTS,
} from '../../../components/FirstRunOnboardingModal'
import { InfoButton } from '../../../components/InfoButton'
import PathPicker from '../../../components/PathPicker'
import StudioDataMigrateModal from '../../../components/StudioDataMigrateModal'
import ModelsRootMigrateModal from '../../../components/ModelsRootMigrateModal'
import { useToast } from '../../../components/Toast'
import {
  formatMasterStateText,
  formatDevStateText,
  shouldShowMasterUpdateButton,
  shouldShowSwitchToStableButton,
  isDevSwitchButtonDisabled,
} from '../../../lib/versionPanel'
import i18n from '../../../i18n'
import { textInputClass } from './constants'
import { SettingsField, SettingsSection } from './fields'

// ── System Section（系统 tab）─────────────────────────────────────────────
//
// PR-B 起拆成两个 sub-section：
//   - VersionSection：当前版本 / 检查更新 / 立即更新（master 通道）
//   - ServiceSection：重启 server
//
// 共用流程"触发后端退出 → 轮询 /api/health 等回来 → 刷新页面"由
// `pollHealthThenReload` 抽出。restart 超时 5 分钟，update 超时 10 分钟
// （要多跑 git pull + 可能 pip install / npm install 的时间）。
export function SystemSection() {
  return (
    <>
      <VersionSection />
      <StorageSection />
      <ServiceSection />
    </>
  )
}

// ── 公共：触发后端退出后轮询 health 并刷新 ─────────────────────────────
//
// 调用者负责在 await 之前已经成功触发了 server 退出（POST /restart 或 /update
// 已经 200 回来）。这里只管"等服务回来 + 刷页面 + 失败提示"。
export type ToastFn = (msg: string, kind?: 'info' | 'success' | 'error') => void

export async function pollHealthThenReload(
  toast: ToastFn,
  timeoutMs: number,
  label: string,
  onTimeout: () => void,
  t: TFunction,
): Promise<void> {
  const deadline = Date.now() + timeoutMs
  const pollInterval = 500
  // 间隔后开始轮询：给 server 时间真正退出，避免命中还没死的旧进程
  await new Promise((r) => setTimeout(r, 1500))
  while (Date.now() < deadline) {
    try {
      await api.health()
      toast(t('settings.operationCompletedReloading', { label }), 'success')
      setTimeout(() => window.location.reload(), 800)
      return
    } catch {
      // server 还没回来，继续轮询
    }
    await new Promise((r) => setTimeout(r, pollInterval))
  }
  const mins = Math.round(timeoutMs / 60_000)
  toast(t('settings.operationTimeout', { label, mins }), 'error')
  onTimeout()
}

// ── 版本 Section（ADR 0005 重设计 — 单视图 + 通道偏好）───────────────
//
// 产品模型：
// - 通道（channel）是**用户视图偏好**：你想订阅哪条更新轨道（稳定 / 开发）
// - 与 git 工作树状态**解耦**：切 toggle 不动 git；真正"切到 dev HEAD" /
//   "更新到 vX.Y.Z" 是单独按钮
// - 同屏只显示当前选中通道的卡片（不并排）—— 通道是互斥视图，并排会让
//   用户陷入"我究竟在哪里"的矛盾
// - 文案语言只有"版本号"+"状态"，绝不出现"commits"/"sha"等 git 词汇
//
// 数据：
// - version.installed_kind (stable / dev / custom) + installed_label：装了什么
// - check.state (up_to_date / update_available / ahead / detached)：相对所选
//   通道的状态
// - prefs.update_channel：用户偏好（"stable" / "dev"）
//
// 自动检查 + Topbar 红点仍然只看 master（ADR 0002 决策）。
export function VersionSection() {
  const { t } = useTranslation()
  const { toast } = useToast()
  // chunk 4：update 预览走 inline preview 面板（不用 dialog）。例外：工作树 dirty
  // 时确认更新前要弹一个"强制覆盖"确认 modal，复用 useDialog().confirm。
  const dialog = useDialog()
  // release notes 已并入公告栏：版本 section 的入口直接打开公告栏 modal。
  const { posts, openCenter } = useAnnouncements()
  const [version, setVersion] = useState<SystemVersion | null>(null)
  // 版本面板「更新内容」概览：取当前版（匹配不到则最新）release post 的前 5 条要点；
  // 「查看更新公告」与详细内容统一打开公告栏（不再是旧的 detail modal）。
  const announcementLang = i18n.language.toLowerCase().startsWith('zh') ? 'zh' : 'en'
  const releaseOverview = useMemo(() => {
    const releases = posts.filter((p) => p.tag === 'release')
    if (!releases.length) return null
    const post = releases.find((p) => p.version === version?.version) ?? releases[0]
    const highlights = extractReleaseHighlights(post.body[announcementLang], 5)
    return highlights.length ? { title: post.title[announcementLang], highlights } : null
  }, [posts, version, announcementLang])
  const [check, setCheck] = useState<SystemUpdateCheck | null>(null)
  const [status, setStatus] = useState<SystemUpdateStatus | null>(null)
  // status 是否已拉过（成功 / 失败都置 true）。回滚提示行占位骨架靠它判「加载中」，
  // 不能直接判 status===null —— 拉取失败时 status 永远 null，骨架会永久不消失。
  const [statusLoaded, setStatusLoaded] = useState(false)
  const [prefs, setPrefs] = useState<SystemPrefsConfig | null>(null)
  const [devCheck, setDevCheck] = useState<SystemUpdateCheck | null>(null)
  // chunk 3 — dev 通道最近 commit 列表 + 选中状态（用户点 commit 准备切换）
  const [devCommits, setDevCommits] = useState<DevCommitsResult | null>(null)
  const [selectedSha, setSelectedSha] = useState<string | null>(null)
  // chunk 4 — 状态机 + preview / progress 数据。CardState / PendingTarget 类型
  // 在模块底部声明（同时给 MasterCardProps / DevCardProps 用，避免重复定义）。
  const [masterState, setMasterState] = useState<CardState>('idle')
  const [devState, setDevState] = useState<CardState>('idle')
  const [pendingTarget, setPendingTarget] = useState<PendingTarget | null>(null)
  const [preflight, setPreflight] = useState<PreflightResult | null>(null)
  const [preflightLoading, setPreflightLoading] = useState(false)
  const [checking, setChecking] = useState(false)
  const [checkingDev, setCheckingDev] = useState(false)
  const [busy, setBusy] = useState(false)
  const [logModal, setLogModal] = useState<{ open: boolean; content: string; loading: boolean }>(
    { open: false, content: '', loading: false },
  )
  // 0.8.1 hotfix — zip 安装用户首次 init git 仓库
  const [initing, setIniting] = useState(false)
  const [initError, setInitError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    void (async () => {
      const v = await api.getSystemVersion().catch(() => null)
      if (cancelled) return
      if (v) setVersion(v)
      // zip 模式下 check_update 会失败（git fetch 在没 .git/ 时报错），
      // 没必要发请求。等用户 init 完后再触发。
      if (v?.is_git_repo !== false) {
        void api.checkSystemUpdate('master').then((r) => { if (!cancelled) setCheck(r) }).catch(() => { /* silent */ })
      }
    })()
    void api.getSystemUpdateStatus().then(setStatus).catch(() => { /* silent */ }).finally(() => setStatusLoaded(true))
    void api.getSecrets().then((s) => setPrefs(s.system)).catch(() => { /* silent */ })
    return () => { cancelled = true }
  }, [])

  // 0.8.1 hotfix — 触发 zip → git 自动 normalize。成功后刷一遍 version + check
  // 让 banner 消失、版本面板正常显示。bootstrap 不重启 server（只动 .git/），
  // 不需要 pollHealthThenReload。
  const handleInitGit = async () => {
    setIniting(true)
    setInitError(null)
    try {
      await api.initGitRepo()
      toast(t('settings.gitInitEnabled'), 'success')
      const v = await api.getSystemVersion().catch(() => null)
      if (v) setVersion(v)
      // init 后立刻拉一次 check，banner 消失 + 同屏显示「已是最新 / 有新版」
      void api.checkSystemUpdate('master', true).then(setCheck).catch(() => { /* silent */ })
    } catch (e) {
      const err = e as Error & { detail?: { message?: string } }
      const msg = err.detail?.message ?? err.message ?? String(e)
      setInitError(msg)
      toast(t('settings.gitInitFailed', { error: msg }), 'error')
    } finally {
      setIniting(false)
    }
  }

  // 选中 dev 通道时自动拉 dev_commits + dev check（用户不用先手动按 [抓取 dev]）。
  // 即便装的是 stable，用户切到 dev 通道偏好时也要能立刻看到 dev HEAD 信息。
  const channelPref: 'stable' | 'dev' = prefs?.update_channel ?? 'stable'
  const showDevView = channelPref === 'dev'
  // 两个 fetch 拆独立 effect：避免「commits 先 resolve 触发 re-render，
  // effect 用 devCommits !== null 早 return 跳过 check fetch」race
  // —— 实测会导致 devCheck 一直 null、"切到 dev HEAD" 按钮不知道
  // 该 disabled，UI 显示 enabled 但点了 no-op。
  useEffect(() => {
    if (!showDevView || devCommits !== null) return
    let cancelled = false
    void api.getDevCommits(10).then((r) => { if (!cancelled) setDevCommits(r) }).catch(() => { /* silent */ })
    return () => { cancelled = true }
  }, [showDevView, devCommits])
  useEffect(() => {
    if (!showDevView || devCheck !== null) return
    let cancelled = false
    void api.checkSystemUpdate('dev', true).then((r) => { if (!cancelled) setDevCheck(r) }).catch(() => { /* silent */ })
    return () => { cancelled = true }
  }, [showDevView, devCheck])

  const handleCheck = async () => {
    setChecking(true)
    try {
      const r = await api.checkSystemUpdate('master', true)
      setCheck(r)
      if (r.error) {
        toast(t('settings.checkFailed', { error: r.error }), 'error')
      } else if (r.state === 'update_available') {
        const target = r.latest_version ?? r.latest_tag ?? r.latest_commit.slice(0, 8)
        toast(t('settings.stableUpdateAvailable', { version: target }), 'info')
      } else if (r.state === 'ahead') {
        toast(t('settings.stableAhead'), 'info')
      } else {
        toast(t('settings.upToDateStable', { version: r.latest_version ? ` ${r.latest_version}` : '' }), 'success')
      }
    } catch (e) {
      toast(t('settings.checkUpdateFailed', { error: String(e) }), 'error')
    } finally {
      setChecking(false)
    }
  }

  // 公用的 422 / 其它错误分流（update 和 rollback 都用）
  const _formatActionError = (e: unknown, action: string): string => {
    const err = e as Error & { status?: number; code?: string; detail?: { tasks?: { name: string; id?: number }[] } }
    if (err.code === 'system.tasks_running') {
      const names = (err.detail?.tasks ?? []).map((task) => task.name || `task#${task.id ?? '?'}`).join(', ')
      return t('settings.taskRunningCancelFirst', { names })
    }
    if (err.code === 'system.working_tree_dirty') {
      return t('settings.dirtyWorkingTree')
    }
    if (err.code === 'system.no_rollback_target') {
      return t('settings.noRollbackTarget')
    }
    return t('settings.triggerActionFailed', { action, error: err.message ?? String(e) })
  }

  // chunk 4 — inline preview/progress/failed 状态机：所有 "更新 / 切换 / 回滚"
  // 动作不再走 dialog 模态，而是把卡 body 替换成 preview 面板（含 release
  // notes / commit info + pre-flight 检查 + 取消/确认 按钮）。确认后切到
  // progress 状态显示 spinner，pollHealthThenReload 触发页面刷新。
  const enterPreview = (target: PendingTarget) => {
    setPendingTarget(target)
    setSelectedSha(null)
    setPreflight(null)
    setPreflightLoading(true)
    if (target.kind === 'master') setMasterState('preview')
    else setDevState('preview')
    void api.getPreflight(target.ref)
      .then((r) => setPreflight(r))
      .catch(() => setPreflight(null))
      .finally(() => setPreflightLoading(false))
  }

  const cancelPreview = () => {
    setMasterState('idle')
    setDevState('idle')
    setPendingTarget(null)
    setPreflight(null)
    setPreflightLoading(false)
  }

  const confirmPreview = async () => {
    if (!pendingTarget) return
    const t = pendingTarget
    // 工作树脏（真实改动，自动 churn 已在后端剔除）→ 弹确认 modal。reset --hard
    // 会丢弃这些未提交改动且不可恢复，必须用户显式点确认才带 force。取消则留在
    // preview 不动。(i18n 用 i18n.t —— 此作用域 t 已被 pendingTarget 遮蔽。)
    const force = preflight?.working_tree_dirty === true
    if (force) {
      const ok = await dialog.confirm(i18n.t('settings.forceUpdateConfirm'), {
        tone: 'warn',
        okText: i18n.t('settings.forceUpdateConfirmOk'),
      })
      if (!ok) return
    }
    if (t.kind === 'master') setMasterState('progress')
    else setDevState('progress')
    setBusy(true)
    try {
      await api.performSystemUpdate(t.ref, force)
    } catch (e) {
      toast(_formatActionError(e, t.kind === 'master' ? i18n.t('settings.actionUpdate') : i18n.t('settings.actionSwitch')), 'error')
      setBusy(false)
      if (t.kind === 'master') setMasterState('idle')
      else setDevState('idle')
      return
    }
    void pollHealthThenReload(
      toast,
      10 * 60_000,
      t.kind === 'master' ? i18n.t('settings.actionUpdate') : i18n.t('settings.actionSwitch'),
      () => {
        setBusy(false)
        if (t.kind === 'master') setMasterState('idle')
        else setDevState('idle')
      },
      i18n.t.bind(i18n),
    )
  }

  // 各 action 入口：构造 PendingTarget 后委托给 enterPreview。
  const handleUpdate = () => {
    if (!check?.has_update) return
    const label = check.latest_tag ?? check.latest_commit.slice(0, 8)
    enterPreview({ kind: 'master', ref: 'origin/master', label })
  }

  const handleSwitchToMaster = () => {
    const label = check?.latest_tag ?? check?.latest_commit?.slice(0, 8) ?? 'master'
    enterPreview({ kind: 'master', ref: 'origin/master', label })
  }

  const handleRollback = () => {
    if (!status?.rollback_target) return
    enterPreview({
      kind: 'master',
      ref: status.rollback_target,
      label: status.rollback_target.slice(0, 8),
    })
  }

  const handleViewLog = async () => {
    setLogModal({ open: true, content: '', loading: true })
    try {
      const r = await api.getSystemUpdateLog()
      setLogModal({ open: true, content: r.content || t('settings.emptyLog'), loading: false })
    } catch (e) {
      setLogModal({ open: true, content: t('settings.loadFailedWithError', { error: String(e) }), loading: false })
    }
  }

  // ADR 0005 — 通道偏好持久化到 secrets.json。**不触发任何 git 操作**，
  // 只是切换 UI 视图。乐观更新 + 失败回滚。
  const handleSwitchChannel = async (next: 'stable' | 'dev') => {
    const prev = prefs
    setPrefs((p) => p
      ? { ...p, update_channel: next }
      : { update_channel: next, show_dev_channel: next === 'dev' })
    if (next === 'stable') setDevCheck(null)  // 切回稳定版时清掉 dev 缓存
    try {
      // 同步写 show_dev_channel 字段，保留对老版本回滚兼容
      const updated = await api.updateSecrets({
        system: { update_channel: next, show_dev_channel: next === 'dev' },
      })
      setPrefs(updated.system)
    } catch (e) {
      setPrefs(prev)
      toast(t('settings.saveChannelFailed', { error: (e as Error).message ?? String(e) }), 'error')
    }
  }

  const handleCheckDev = async () => {
    setCheckingDev(true)
    try {
      // chunk 3：同时拉 update_check（HEAD 比对）和 dev_commits（commit 时间线）
      const [check, commits] = await Promise.all([
        api.checkSystemUpdate('dev', true),
        api.getDevCommits(10),
      ])
      setDevCheck(check)
      setDevCommits(commits)
      if (check.error) {
        toast(t('settings.devCheckFailed', { error: check.error }), 'error')
      } else if (commits.error && !commits.fetched) {
        toast(t('settings.devFetchPartialFailed', { error: commits.error }), 'error')
      } else if (check.state === 'update_available') {
        toast(t('settings.devHasNewCommits', { count: check.behind_count }), 'info')
      } else if (check.state === 'ahead') {
        toast(t('settings.devAhead'), 'info')
      } else {
        toast(t('settings.devUpToDate'), 'success')
      }
    } catch (e) {
      toast(t('settings.devCheckFailed', { error: String(e) }), 'error')
    } finally {
      setCheckingDev(false)
    }
  }

  // chunk 3 + chunk 4：选中 commit 后进 preview 面板（dev 卡）。
  const handleSwitchToCommit = (commit: DevCommit) => {
    enterPreview({
      kind: 'dev',
      ref: commit.sha,
      label: commit.short_sha,
      msg: commit.msg,
      author: commit.author,
    })
  }

  // "切到 dev (HEAD)" 当 master 用户初次切到 dev：进 preview 面板。
  const handleUpdateDev = () => {
    const headCommit = devCommits?.commits?.[0]
    if (!headCommit) {
      // 还没抓取过 dev，先 fetch 再 retry（避免空 ref）
      void handleCheckDev()
      return
    }
    enterPreview({
      kind: 'dev',
      ref: 'origin/dev',
      label: headCommit.short_sha,
      msg: headCommit.msg,
      author: headCommit.author,
    })
  }

  // 派生状态（ADR 0005）：installed_kind / state 取代 branch / has_update
  const installedIsDevHead = version?.installed_kind === 'dev'
  const masterHasUpdate = check?.state === 'update_available'
  const hasRollback = !!status?.rollback_target
  // 上次 update 失败 banner（aborted / failed / partial 时显示红色提示）
  const statusBadFailed = !!status && (status.status === 'failed' || status.status === 'aborted' || status.status === 'partial')

  return (
    <SettingsSection
      id="version"
      title={t('settings.version')}
      headerExtras={
        <>
          <InfoButton>
            <ul>
              <li>{t('settings.versionInfoChannel')}</li>
              <li>{t('settings.versionInfoAutoCheck')}</li>
              <li>{t('settings.versionInfoUpdateImpl')}</li>
              <li>{t('settings.versionInfoPreflight')}</li>
            </ul>
          </InfoButton>
          {/* 更新内容已并入公告栏：入口直接打开公告栏 modal（铃铛同款） */}
          <button
            type="button"
            className="btn btn-ghost btn-sm text-xs text-fg-tertiary ml-auto inline-flex items-center gap-1"
            onClick={() => openCenter()}
          >
            <VersionIcon name="log" />
            {t('settings.viewAnnouncements')}
          </button>
        </>
      }
    >
      {/* 0.8.1 hotfix — zip 安装用户首次启用自更新功能的 banner。
          version.is_git_repo=false 时显示；git 不可用 vs 可用分两种文案。
          init 成功后 setVersion 刷新，banner 自动消失。 */}
      {version && !version.is_git_repo && (
        <div className="vs-zip-banner">
          {!version.git_available ? (
            <>
              <div className="vs-zip-banner-title">{t('settings.gitNotDetected')}</div>
              <div className="vs-zip-banner-body">
                <Trans
                  i18nKey="settings.gitRequiredHelp"
                  components={{ a: <a href="https://git-scm.com/downloads" target="_blank" rel="noreferrer" /> }}
                />
              </div>
            </>
          ) : (
            <>
              <div className="vs-zip-banner-title">{t('settings.enableAutoUpdate')}</div>
              <div className="vs-zip-banner-body">
                <Trans
                  i18nKey="settings.zipInstallGitInitHelp"
                  values={{ version: `v${version.stable_version?.replace(/^v/, '') ?? version.version}` }}
                  components={{ b: <b /> }}
                />
              </div>
              <div className="vs-zip-banner-actions">
                <button
                  type="button"
                  className="btn btn-sm btn-primary"
                  onClick={() => void handleInitGit()}
                  disabled={initing}
                >
                  {initing ? t('settings.initializingGit') : t('settings.enableAutoUpdate')}
                </button>
                {initError && <span className="vs-zip-banner-error">{t('settings.failedWithError', { error: initError })}</span>}
              </div>
            </>
          )}
        </div>
      )}

      {/* 顶部：你装的是什么（一行事实状态，与通道偏好解耦） */}
      <div className="vs-installed-row">
        <span className="vs-installed-label">{t('settings.installedVersionLabel')}</span>
        <b className="vs-installed-value">{version?.installed_label ?? t('settings.loadingEllipsis')}</b>
        {version?.is_dirty && !version.installed_label.includes(t('settings.uncommittedChangesText')) && (
          <span className="vs-installed-warn">· {t('settings.localChanges')}</span>
        )}
      </div>

      {/* 通道偏好：radio toggle（不触发 git） */}
      <div className="vs-channel-toggle-row">
        <span className="vs-channel-toggle-label">{t('settings.updateChannel')}</span>
        <button
          type="button"
          role="radio"
          aria-checked={channelPref === 'stable'}
          className={`vs-channel-radio${channelPref === 'stable' ? ' on' : ''}`}
          onClick={() => { if (channelPref !== 'stable') void handleSwitchChannel('stable') }}
        >
          <span className="vs-channel-dot" />{t('settings.stable')}
        </button>
        <button
          type="button"
          role="radio"
          aria-checked={channelPref === 'dev'}
          className={`vs-channel-radio${channelPref === 'dev' ? ' on' : ''}`}
          onClick={() => { if (channelPref !== 'dev') void handleSwitchChannel('dev') }}
        >
          <span className="vs-channel-dot" />{t('settings.devBuild')}
        </button>
        <span className="vs-channel-hint">{t('settings.channelUiOnly')}</span>
      </div>

      <div className="vs-sec-card">
        <div className="vs-channels">
          {!showDevView ? (
            <MasterCard
              on={true}
              solo={true}
              version={version}
              check={check}
              status={status}
              statusLoaded={statusLoaded}
              hasUpdate={masterHasUpdate}
              hasRollback={hasRollback}
              statusBadFailed={statusBadFailed}
              checking={checking}
              busy={busy}
              cardState={masterState}
              pendingTarget={pendingTarget}
              preflight={preflight}
              preflightLoading={preflightLoading}
              onCancelPreview={cancelPreview}
              onConfirmPreview={confirmPreview}
              onCheck={handleCheck}
              onUpdate={handleUpdate}
              onSwitchToMaster={handleSwitchToMaster}
              onRollback={handleRollback}
              onViewLog={handleViewLog}
            />
          ) : (
            <DevCard
              on={installedIsDevHead}
              check={devCheck}
              commits={devCommits}
              currentSha={version?.commit ?? ''}
              installedKind={version?.installed_kind}
              selectedSha={selectedSha}
              setSelectedSha={setSelectedSha}
              checking={checkingDev}
              busy={busy}
              cardState={devState}
              pendingTarget={pendingTarget}
              preflight={preflight}
              preflightLoading={preflightLoading}
              onCancelPreview={cancelPreview}
              onConfirmPreview={confirmPreview}
              onCheck={handleCheckDev}
              onSwitchToDev={handleUpdateDev}
              onSwitchToCommit={handleSwitchToCommit}
            />
          )}
        </div>
        {releaseOverview && (
          <div className="mt-3 pt-3 border-t border-dim">
            <div className="flex items-center justify-between gap-3 mb-2">
              <span className="text-sm font-semibold text-fg-primary">{releaseOverview.title}</span>
              <button type="button" className="btn btn-ghost btn-sm" onClick={() => openCenter()}>
                <VersionIcon name="note" />{t('settings.viewAnnouncements')}
              </button>
            </div>
            <ul className="m-0 pl-5 list-disc space-y-1 text-sm text-fg-secondary marker:text-fg-tertiary">
              {releaseOverview.highlights.map((h, i) => (
                <li key={i} className="leading-relaxed">{h}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {logModal.open && (
        <UpdateLogModal
          loading={logModal.loading}
          content={logModal.content}
          onClose={() => setLogModal({ open: false, content: '', loading: false })}
        />
      )}
    </SettingsSection>
  )
}

// ── 子组件：图标 / Master 卡 / Dev 卡 ─────────────────────────────────
//
// 双卡布局拆成独立函数组件方便 chunk 2/3/4 各自扩展：
//   - chunk 2 把 release notes 填进 MasterCard.change-block
//   - chunk 3 给 DevCard 加 commits 列表 + 选中状态
//   - chunk 4 给两卡都加 preview / progress 状态机

export const VERSION_ICON_PATHS: Record<string, React.ReactNode> = {
  refresh:  <><path d="M14 8a6 6 0 1 1-1.76-4.24" /><path d="M14 3v3.4h-3.4" /></>,
  log:      <><rect x="3" y="2.5" width="10" height="11" rx="1.5" /><path d="M5.5 5.5h5M5.5 8h5M5.5 10.5h3" /></>,
  rollback: <><path d="M3 8h7a3 3 0 1 1 0 6h-1" /><path d="M5.5 5.5L3 8l2.5 2.5" /></>,
  note:     <><path d="M4 3.5h6l2 2v7a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-8a1 1 0 0 1 1-1z" /><path d="M5.5 7h5M5.5 9.5h5M5.5 12h3" /></>,
  lock:     <><rect x="3.5" y="7" width="9" height="6.5" rx="1" /><path d="M5.5 7v-2a2.5 2.5 0 0 1 5 0v2" /></>,
}

export function VersionIcon({ name }: { name: keyof typeof VERSION_ICON_PATHS | string }) {
  const path = VERSION_ICON_PATHS[name]
  if (!path) return null
  return (
    <svg width={12} height={12} viewBox="0 0 16 16" fill="none" stroke="currentColor"
      strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round">
      {path}
    </svg>
  )
}

export type CardState = 'idle' | 'preview' | 'progress'
export type PendingTarget = {
  kind: 'master' | 'dev'
  ref: string
  label: string
  msg?: string
  author?: string
}

export type MasterCardProps = {
  on: boolean
  solo: boolean
  version: SystemVersion | null
  check: SystemUpdateCheck | null
  status: SystemUpdateStatus | null
  statusLoaded: boolean
  hasUpdate: boolean
  hasRollback: boolean
  statusBadFailed: boolean
  checking: boolean
  busy: boolean
  cardState: CardState
  pendingTarget: PendingTarget | null
  preflight: PreflightResult | null
  preflightLoading: boolean
  onCancelPreview: () => void
  onConfirmPreview: () => void
  onCheck: () => void
  onUpdate: () => void
  onSwitchToMaster: () => void
  onRollback: () => void
  onViewLog: () => void
}

// chunk 4 — preview / progress 通用面板。channel 决定主按钮配色（master=primary
// orange / dev=warn yellow）。details 可选，由 caller 决定渲染什么（dev 用
// commit msg / author；master 不再内联 release notes，留空）。
export type PreviewPaneProps = {
  channel: 'master' | 'dev'
  fromLabel: string
  toLabel: string
  badge?: string
  details?: React.ReactNode
  preflight: PreflightResult | null
  loading: boolean
  busy: boolean
  onCancel: () => void
  onConfirm: () => void
}

export function PreviewPane(p: PreviewPaneProps) {
  const { t } = useTranslation()
  const confirmDisabled = !p.preflight || p.preflight.blocking || p.busy
  return (
    <div className="vs-preview-pane">
      <div className="vs-preview-head">
        <span className="vs-from">{p.fromLabel}</span>
        <span className="vs-arr">→</span>
        <span className="vs-to">{p.toLabel}</span>
        {p.badge && <span className="vs-badge">{p.badge}</span>}
      </div>

      {p.details}

      <div className="vs-preflight">
        <div className="vs-h">{t('settings.preflightCheck')}</div>
        {p.loading ? (
          <div className="vs-row">
            <span className="vs-glyph">·</span>
            <span>{t('settings.checkingEllipsis')}</span>
          </div>
        ) : p.preflight ? (
          p.preflight.checks.map((c, i) => (
            <div key={i} className={`vs-row ${c.level}`}>
              <span className="vs-glyph">
                {c.level === 'ok' ? '✓' : c.level === 'warn' ? '!' : '✗'}
              </span>
              <span>{c.label}</span>
            </div>
          ))
        ) : (
          <div className="vs-row err">
            <span className="vs-glyph">✗</span>
            <span>{t('settings.preflightFailedRetry')}</span>
          </div>
        )}
      </div>

      <div className="vs-chan-foot" style={{ borderTop: 0, paddingTop: 0 }}>
        <div className="vs-info">
          {t('settings.preflightInfo')}
        </div>
        <div className="vs-actions">
          <button onClick={p.onCancel} disabled={p.busy} className="btn btn-sm">
            {t('settings.cancel')}
          </button>
          <button
            onClick={p.onConfirm}
            disabled={confirmDisabled}
            className={`btn btn-sm ${p.channel === 'master' ? 'btn-primary' : 'btn-warn'}`}
          >
            {p.busy
              ? t('settings.processing')
              : t('settings.confirmActionTo', {
                action: p.channel === 'master' ? t('settings.actionUpdate') : t('settings.actionSwitch'),
                label: p.toLabel,
              })}
          </button>
        </div>
      </div>
    </div>
  )
}

export function ProgressPane({ fromLabel, toLabel }: { fromLabel: string; toLabel: string }) {
  const { t } = useTranslation()
  return (
    <div className="vs-progress-pane">
      <div className="vs-preview-head">
        <span className="vs-from">{fromLabel}</span>
        <span className="vs-arr">→</span>
        <span className="vs-to">{toLabel}</span>
      </div>
      <div className="vs-progress-bar">
        <div className="vs-progress-fill" style={{ width: '100%' }} />
      </div>
      <div className="vs-progress-step">
        <span>{t('settings.progressStarted')}</span>
        <span>{t('settings.progressWaitReload')}</span>
      </div>
      <p style={{ color: 'var(--fg-tertiary)', fontSize: 11, lineHeight: 1.5, margin: 0 }}>
        {t('settings.progressDetail')}
      </p>
    </div>
  )
}

export function MasterCard(p: MasterCardProps) {
  const { t } = useTranslation()
  // 装的是 stable 时显示当前稳定版号，否则 ver-tag 区不显示 from（"你装的"
  // 顶部行已经表达了装了什么，避免 "v0.8.0 → v0.8.0" 这种因 __version__
  // 字符串与目标 tag 字面相同导致的伪箭头）
  const installedIsStable = p.version?.installed_kind === 'stable'
  const currentTag = installedIsStable
    ? (p.version?.stable_version ?? p.version?.tag ?? `v${p.version?.version ?? ''}`)
    : null
  // 远端最新稳定版（state=update_available 时显示）
  const targetTag = p.check?.latest_version ?? p.check?.latest_tag ?? ''
  const stateText = formatMasterStateText(p.check, t)
  const showUpdateButton = shouldShowMasterUpdateButton(p.check, p.version?.installed_kind)
  const showSwitchToStableButton = shouldShowSwitchToStableButton(p.check, p.version?.installed_kind)
  if (p.cardState === 'preview' && p.pendingTarget && p.pendingTarget.kind === 'master') {
    return (
      <div className="vs-chan">
        <div className="vs-chan-head">
          <div className="vs-lhs">
            <span className="vs-name">{t('settings.stableConfirmUpdate')}</span>
            <span className="vs-pill vs-pill-stable"><span className="vs-dot" />{t('settings.stable')}</span>
          </div>
          <button className="btn btn-sm btn-ghost" onClick={p.onCancelPreview} disabled={p.busy}>
            {t('settings.back')}
          </button>
        </div>
        <PreviewPane
          channel="master"
          fromLabel={currentTag ?? p.version?.installed_label ?? t('settings.currentShort')}
          toLabel={p.pendingTarget.label}
          preflight={p.preflight}
          loading={p.preflightLoading}
          busy={p.busy}
          onCancel={p.onCancelPreview}
          onConfirm={p.onConfirmPreview}
        />
      </div>
    )
  }
  if (p.cardState === 'progress' && p.pendingTarget && p.pendingTarget.kind === 'master') {
    return (
      <div className="vs-chan">
        <div className="vs-chan-head">
          <div className="vs-lhs">
            <span className="vs-name">{t('settings.stableUpdating')}</span>
            <span className="vs-pill vs-pill-stable"><span className="vs-dot" />{t('settings.stable')}</span>
          </div>
        </div>
        <ProgressPane fromLabel={currentTag ?? p.version?.installed_label ?? t('settings.currentShort')} toLabel={p.pendingTarget.label} />
      </div>
    )
  }
  const checkedAt = p.check?.checked_at
    ? new Date(p.check.checked_at * 1000).toLocaleString()
    : t('settings.notChecked')
  const releasedAt = p.version?.commit_time_iso
    ? new Date(p.version.commit_time_iso).toLocaleDateString()
    : null

  return (
    <div className="vs-chan">
      <div className="vs-chan-head">
        <div className="vs-lhs">
          <span className="vs-name">{t('settings.stable')}</span>
          <span className="vs-pill vs-pill-stable"><span className="vs-dot" />{t('settings.stable')}</span>
        </div>
        <div className={`vs-meta${p.hasUpdate ? ' attn' : ''}`}>{stateText}</div>
      </div>

      {p.statusBadFailed && p.status && (
        <div className="vs-fail-banner">
          <div className="vs-h">
            <span>
              {t('settings.lastUpdate')}
              {p.status.status === 'aborted' ? t('settings.statusAborted')
                : p.status.status === 'partial' ? t('settings.statusPartial')
                : t('settings.statusFailed')}
            </span>
            {!!p.status.finished_at && (
              <span className="vs-when">
                {new Date(p.status.finished_at * 1000).toLocaleString()}
              </span>
            )}
          </div>
          <div className="vs-d">
            {p.status.reason || t('settings.unknownReason')}
            {p.status.target && <> · target = <code>{p.status.target}</code></>}
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {p.hasUpdate && (
              <button className="btn btn-primary btn-sm" onClick={p.onUpdate} disabled={p.busy}>
                {t('settings.retryUpdateTo', { tag: targetTag })}
              </button>
            )}
            <button className="btn btn-sm" onClick={p.onViewLog} disabled={p.busy}>
              <VersionIcon name="log" />{t('settings.viewFullLog')}
            </button>
          </div>
        </div>
      )}

      <div className={`vs-chan-body ${p.solo ? 'solo' : 'split'}`}>
        <div className="vs-ver-block" style={{ flex: p.solo ? '0 0 220px' : 1 }}>
          <div className="vs-ver-tag">
            {/* 装的是 stable 且有新稳定版：显示 from → to 箭头
                装的是 stable 但已是最新：只显示当前版本号
                装的不是 stable（dev / custom）：只显示目标稳定版号（如有），
                  不再显示 "v0.8.0 → v0.8.0" 伪箭头 */}
            {installedIsStable && p.hasUpdate && targetTag && currentTag !== targetTag ? (
              <>
                <span className="vs-dim">{currentTag}</span>
                <span className="vs-arrow">→</span>
                <span className="vs-target">{targetTag}</span>
              </>
            ) : installedIsStable ? (
              currentTag
            ) : targetTag ? (
              <span className="vs-target">{targetTag}</span>
            ) : null}
          </div>
          <div className="vs-ver-meta">
            {releasedAt && <span>{t('settings.releasedAt')} <b>{releasedAt}</b></span>}
          </div>
          {!p.hasUpdate && p.solo && (
            <div className="vs-ver-tagline">{t('settings.topbarMasterOnly')}</div>
          )}
        </div>
      </div>

      <div className="vs-chan-foot">
        <div className="vs-info">
          {p.check?.error
            ? <span style={{ color: 'var(--err)' }}>{p.check.error}</span>
            : <span>{t('settings.lastCheck', { time: checkedAt })}</span>}
        </div>
        <div className="vs-actions">
          <button onClick={p.onCheck} disabled={p.checking || p.busy} className="btn btn-sm">
            <VersionIcon name="refresh" />{p.checking ? t('settings.checkingEllipsis') : t('settings.checkUpdates')}
          </button>
          {/* state=update_available 才显示更新按钮；up_to_date / ahead / detached 不显示
              （上面 vs-meta 已经把"已是最新 / 本地领先 / 当前不在历史上"说清楚了）*/}
          {showUpdateButton && (
            <button onClick={p.onUpdate} disabled={p.busy || p.checking} className="btn btn-sm btn-primary">
              {p.busy ? t('settings.updatingEllipsis') : t('settings.updateTo', { tag: targetTag })}
            </button>
          )}
          {/* 装在非稳定版（dev / custom）时显示"切到最新稳定版"按钮；
              与"更新到 X"按钮互斥（shouldShowMasterUpdateButton 内部已按
              installed_kind 排除了非 stable），避免同屏显示两个做同样事的按钮 */}
          {showSwitchToStableButton && (
            <button onClick={p.onSwitchToMaster} disabled={p.busy || p.checking} className="btn btn-sm btn-primary">
              {p.busy ? t('settings.switchingEllipsis') : t('settings.switchToStable', { tag: p.check?.latest_version })}
            </button>
          )}
        </div>
      </div>

      {!p.statusLoaded ? (
        // status 加载中：回滚提示行占位骨架（与真实行同高），有可回滚版本时平滑
        // 填入而非弹入把下方内容顶开；几乎总有可回滚版本（除从未更新过的首版）
        <div className="vs-rollback-collapse" aria-hidden="true">
          <span className="vs-rollback-summary vs-rollback-sk">
            <span className="vs-caret">▸</span>
            <span className="vs-rollback-sk-bar" />
          </span>
        </div>
      ) : p.hasRollback && p.status?.rollback_target ? (() => {
        // rollback 显示优先 tag（"v0.6.0"），否则 sha 前 8 位
        const sha = p.status.rollback_target
        const tag = p.status.rollback_target_tag
        const label = tag || sha.slice(0, 8)
        return (
          // 回滚是潜在破坏性操作（reset --hard 丢失当前 commit 上的本地未
          // commit 改动 / GC 后 reflog 也可能消失），UI 默认折叠成小字提示
          // 让用户主动确认才展开按钮，降低误触概率。
          <details className="vs-rollback-collapse">
            <summary className="vs-rollback-summary">
              <span className="vs-caret">▸</span>
              {t('settings.rollbackAvailable', { label })}
            </summary>
            <div className="vs-rollback-inline-row">
              <div className="vs-lhs">
                <span className="vs-ico"><VersionIcon name="rollback" /></span>
                <span>{t('settings.previousVersion')}</span>
                <b>{label}</b>
                {tag && <span className="vs-when">{sha.slice(0, 8)}</span>}
              </div>
              <button onClick={p.onRollback} disabled={p.busy || p.checking} className="btn btn-sm">
                {t('settings.switchBackTo', { label })}
              </button>
            </div>
          </details>
        )
      })() : null}
    </div>
  )
}

export type DevCardProps = {
  on: boolean
  check: SystemUpdateCheck | null
  commits: DevCommitsResult | null
  currentSha: string
  installedKind: 'stable' | 'dev' | 'custom' | 'zip' | undefined
  selectedSha: string | null
  setSelectedSha: (sha: string | null) => void
  checking: boolean
  busy: boolean
  cardState: CardState
  pendingTarget: PendingTarget | null
  preflight: PreflightResult | null
  preflightLoading: boolean
  onCancelPreview: () => void
  onConfirmPreview: () => void
  onCheck: () => void
  onSwitchToDev: () => void
  onSwitchToCommit: (commit: DevCommit) => void
}

export function DevCard(p: DevCardProps) {
  const { t } = useTranslation()
  const commits = p.commits?.commits ?? []
  const head = commits[0]?.short_sha ?? p.check?.latest_commit?.slice(0, 8)
  const selectedCommit = p.selectedSha ? commits.find((c) => c.sha === p.selectedSha) ?? null : null
  const fetchError = p.commits?.error ?? p.check?.error
  const currentShortSha = p.currentSha ? p.currentSha.slice(0, 8) : t('settings.currentShort')
  const stateText = formatDevStateText(p.check, t)
  // installedKind 用作 check 还没 resolve 期间的 fallback：装 dev tip 时
  // 按钮 disabled，避免显示可点但点了 no-op
  const devSwitchDisabled = isDevSwitchButtonDisabled(p.check, p.installedKind)
  if (p.cardState === 'preview' && p.pendingTarget && p.pendingTarget.kind === 'dev') {
    const target = p.pendingTarget
    return (
      <div className="vs-chan">
        <div className="vs-chan-head">
          <div className="vs-lhs">
            <span className="vs-name">{t('settings.devConfirmSwitch')}</span>
            <span className="vs-pill vs-pill-dev"><span className="vs-dot" />{t('settings.devBuild')}</span>
          </div>
          <button className="btn btn-sm btn-ghost" onClick={p.onCancelPreview} disabled={p.busy}>
            {t('settings.back')}
          </button>
        </div>
        <PreviewPane
          channel="dev"
          fromLabel={currentShortSha}
          toLabel={target.label}
          details={
            <div className="vs-change-block">
              <div className="vs-h">{t('settings.targetThisCommit', { label: target.label })}</div>
              {target.msg ? (
                <>
                  <div style={{ fontSize: 13, color: 'var(--fg-primary)', marginTop: 4, lineHeight: 1.5 }}>
                    {target.msg}
                  </div>
                  {target.author && (
                    <div className="vs-ver-meta" style={{ marginTop: 6 }}>
                      <span>author <b>{target.author}</b></span>
                    </div>
                  )}
                </>
              ) : (
                <div style={{ fontSize: 13, color: 'var(--fg-tertiary)', marginTop: 4 }}>
                  {t('settings.switchToDevHeadDesc')}
                </div>
              )}
            </div>
          }
          preflight={p.preflight}
          loading={p.preflightLoading}
          busy={p.busy}
          onCancel={p.onCancelPreview}
          onConfirm={p.onConfirmPreview}
        />
      </div>
    )
  }
  if (p.cardState === 'progress' && p.pendingTarget && p.pendingTarget.kind === 'dev') {
    return (
      <div className="vs-chan">
        <div className="vs-chan-head">
          <div className="vs-lhs">
            <span className="vs-name">{t('settings.devSwitching')}</span>
            <span className="vs-pill vs-pill-dev"><span className="vs-dot" />{t('settings.devBuild')}</span>
          </div>
        </div>
        <ProgressPane fromLabel={currentShortSha} toLabel={p.pendingTarget.label} />
      </div>
    )
  }

  return (
    <div className="vs-chan">
      <div className="vs-chan-head">
        <div className="vs-lhs">
          <span className="vs-name">{t('settings.devBuild')}</span>
          <span className="vs-pill vs-pill-dev"><span className="vs-dot" />{t('settings.devBuild')}</span>
        </div>
        <div className={`vs-meta${p.check?.state === 'update_available' ? ' attn' : ''}`}>
          {fetchError && !head ? (
            <span style={{ color: 'var(--err)' }}>{fetchError}</span>
          ) : head ? (
            <>
              dev HEAD <b style={{ color: 'var(--fg-secondary)', fontWeight: 500 }}>{head}</b>
              {p.check && <>{' · '}{stateText}</>}
            </>
          ) : (
            <span>{t('settings.notFetched')}</span>
          )}
        </div>
      </div>

      <div className="vs-change-block" style={{ paddingTop: 4, paddingBottom: 4 }}>
        <div className="vs-h">{t('settings.recentCommits')}</div>
        {commits.length === 0 ? (
          <ul className="vs-change-list">
            <li>
              <span className="vs-glyph">·</span>
              <span className="vs-txt">
                {fetchError
                  ? <span style={{ color: 'var(--err)' }}>{fetchError}</span>
                  : t('settings.fetchDevHint')}
              </span>
            </li>
          </ul>
        ) : (
          <>
            <ul className="vs-commits">
              {commits.map((c, i) => {
                const isHead = i === 0
                const isCurrent = !!p.currentSha && c.sha === p.currentSha
                const isSelected = c.sha === p.selectedSha
                const clickable = !isCurrent
                // 行 class 同时跟 isHead / isCurrent / clickable / selected。
                // accent glyph 走 .current（"你在这里"）；HEAD 只在 pill 里
                // 用文字标记（不抢 glyph）。
                const classes = ['vs-commit']
                if (isHead) classes.push('head')
                if (isCurrent) classes.push('current')
                if (clickable) classes.push('clickable')
                if (isSelected) classes.push('selected')
                return (
                  <li
                    key={c.sha}
                    className={classes.join(' ')}
                    onClick={() => clickable && p.setSelectedSha(isSelected ? null : c.sha)}
                    title={c.msg}
                  >
                    <span className="vs-glyph" />
                    <span className="vs-msg">{c.msg}</span>
                    <span className="vs-sha">{c.short_sha}</span>
                    <span className="vs-pill-slot">
                      {isCurrent ? (
                        <span className="vs-head-pill">{t('settings.currentMarker')}</span>
                      ) : isHead ? (
                        <span className="vs-head-pill">HEAD</span>
                      ) : isSelected ? (
                        <span className="vs-switch-hint">{t('settings.selectedMarker')}</span>
                      ) : (
                        <span className="vs-switch-hint">{t('settings.switchHere')}</span>
                      )}
                    </span>
                  </li>
                )
              })}
            </ul>
            {p.commits && !p.commits.fetched && p.commits.error && (
              <p className="vs-d" style={{ color: 'var(--warn)', marginTop: 6 }}>
                {t('settings.fetchFailedCached', { error: p.commits.error })}
              </p>
            )}
          </>
        )}
      </div>

      {p.selectedSha && selectedCommit ? (
        // 选中确认条：仅 sha + 取消/确认 按钮。commit 信息上方 list 已可见，
        // 这里只是 action 收尾，info 段去掉避免长 message 挤换行。
        <div className="vs-selection-foot">
          <span className="vs-info" title={selectedCommit.msg}>
            <b>{selectedCommit.short_sha}</b>
          </span>
          <div className="vs-actions">
            <button onClick={() => p.setSelectedSha(null)} disabled={p.busy} className="btn btn-sm btn-ghost">
              {t('settings.cancel')}
            </button>
            <button
              onClick={() => p.onSwitchToCommit(selectedCommit)}
              disabled={p.busy || p.checking}
              className="btn btn-sm btn-warn"
            >
              {p.busy ? t('settings.switchingEllipsis') : t('settings.switchToCommit', { sha: selectedCommit.short_sha })}
            </button>
          </div>
        </div>
      ) : (
        <div className="vs-chan-foot">
          <div className="vs-info">
            {p.check?.checked_at
              ? <span>{t('settings.lastFetch', { time: new Date(p.check.checked_at * 1000).toLocaleString() })}</span>
              : <span style={{ color: 'var(--fg-tertiary)' }}>{t('settings.notFetched')}</span>}
          </div>
          <div className="vs-actions">
            <button onClick={p.onCheck} disabled={p.checking || p.busy} className="btn btn-sm">
              <VersionIcon name="refresh" />{p.checking ? t('settings.fetchingEllipsis') : t('settings.fetchDev')}
            </button>
            {/* 切按钮 disabled 条件改用 commit 比较（state=up_to_date），不再
                看 branch / installed_kind —— 因为 release 直后存在"装的是 stable
                但 commit 恰好等于 dev HEAD"的边界，此时切操作是 no-op */}
            {devSwitchDisabled ? (
              <button disabled className="btn btn-sm">{t('settings.alreadyAtDevHead')}</button>
            ) : commits.length > 0 ? (
              <button
                onClick={p.onSwitchToDev}
                disabled={p.busy || p.checking}
                className="btn btn-sm btn-warn"
              >
                {p.busy ? t('settings.switchingEllipsis') : t('settings.switchToDev', { head: head ? ` (${head})` : '' })}
              </button>
            ) : null}
          </div>
        </div>
      )}
    </div>
  )
}

// 简易的 modal：点遮罩 / 按 ESC 关闭，pre + 等宽字体显示日志。
// 没用 useDialog 是因为它返回的是命令式 confirm/prompt 接口，不适合长文本展示。
export function UpdateLogModal({
  loading, content, onClose,
}: { loading: boolean; content: string; onClose: () => void }) {
  const { t } = useTranslation()
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={onClose}
    >
      <div
        className="bg-surface border border-subtle rounded-md shadow-lg max-w-4xl w-[92vw] max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-subtle px-4 py-2.5">
          <h3 className="text-sm font-semibold text-fg-primary">{t('settings.updateLogTitle')}</h3>
          <button
            onClick={onClose}
            className="text-fg-dim hover:text-fg-primary text-lg leading-none"
            aria-label={t('common.close')}
          >×</button>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <span className="text-fg-dim text-sm">{t('common.loading')}</span>
          ) : (
            <pre className="text-2xs font-mono text-fg-primary whitespace-pre-wrap break-words">
              {content}
            </pre>
          )}
        </div>
      </div>
    </div>
  )
}

// ── 存储位置 Section（studio_data 自定义位置 + 迁移）──────────────────────
export function StorageSection() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [info, setInfo] = useState<StudioDataInfo | null>(null)
  const [pickerOpen, setPickerOpen] = useState(false)
  // 迁移 modal 的目标目录；非 null = modal 打开（迁移期间 modal 不可关，
  // 不存在"后台迁移中"的游离状态，section 无需跟踪迁移进度）
  const [migrateTarget, setMigrateTarget] = useState<string | null>(null)
  const [restartBusy, setRestartBusy] = useState(false)
  // 模型根目录（同区块第二张卡；迁移完无需重启，立即生效）
  const [modelsInfo, setModelsInfo] = useState<ModelsRootInfo | null>(null)
  const [modelsPickerOpen, setModelsPickerOpen] = useState(false)
  const [modelsMigrateTarget, setModelsMigrateTarget] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    void api.getStudioDataInfo(false).then((i) => {
      if (!cancelled) setInfo(i)
    }).catch(() => { /* section 显示用，拉不到不阻塞页面 */ })
    return () => { cancelled = true }
  }, [])

  const refreshModelsInfo = () => {
    void api.getModelsRootInfo(false).then(setModelsInfo).catch(() => { /* 显示用 */ })
  }
  useEffect(() => {
    refreshModelsInfo()
  }, [])

  // done 态「立即重启」：modal 上下文已是确认语境，不再二次 confirm
  const handleRestart = async () => {
    setRestartBusy(true)
    try {
      await api.restartServer()
    } catch (e) {
      const err = e as Error & { status?: number; code?: string; detail?: { tasks?: { name: string; id?: number }[] } }
      if (err.code === 'system.tasks_running') {
        const names = (err.detail?.tasks ?? []).map((task) => task.name || `task#${task.id ?? '?'}`).join(', ')
        toast(t('settings.taskRunningCancelFirst', { names }), 'error')
      } else {
        toast(t('settings.restartTriggerFailed', { error: err.message ?? String(e) }), 'error')
      }
      setRestartBusy(false)
      return
    }
    void pollHealthThenReload(toast, 5 * 60_000, t('settings.restart'), () => setRestartBusy(false), t)
  }

  return (
    <SettingsSection id="storage" title={t('settings.storage.sectionTitle')}>
      <SettingsField
        label={t('settings.storage.locationLabel')}
        helpTooltip={
          <>
            <p>{t('settings.storage.help1')}</p>
            <p>{t('settings.storage.help2')}</p>
          </>
        }
      >
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2 min-w-0">
            <input
              type="text"
              readOnly
              value={info?.current ?? '…'}
              className={`${textInputClass} font-mono text-xs flex-1 min-w-0 cursor-default`}
            />
            <button
              className="btn btn-secondary btn-sm shrink-0"
              onClick={() => setPickerOpen(true)}
              disabled={restartBusy}
            >
              {t('settings.storage.changeLocation')}
            </button>
          </div>
          {info?.is_custom && (
            <span className="text-2xs text-fg-tertiary">
              {t('settings.storage.customBadge')}
            </span>
          )}
        </div>
      </SettingsField>

      <SettingsField
        label={t('settings.storage.modelsRootLabel')}
        helpTooltip={<p>{t('settings.storage.modelsRootHelp')}</p>}
      >
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2 min-w-0">
            <input
              type="text"
              readOnly
              value={modelsInfo?.current ?? '…'}
              className={`${textInputClass} font-mono text-xs flex-1 min-w-0 cursor-default`}
            />
            <button
              className="btn btn-secondary btn-sm shrink-0"
              onClick={() => setModelsPickerOpen(true)}
            >
              {t('settings.storage.changeLocation')}
            </button>
          </div>
          {modelsInfo?.is_custom && (
            <span className="text-2xs text-fg-tertiary">
              {t('settings.storage.customBadge')}
            </span>
          )}
        </div>
      </SettingsField>

      {pickerOpen && (
        <PathPicker
          dirOnly
          initialPath={info?.current}
          onPick={(path) => {
            setPickerOpen(false)
            setMigrateTarget(path)
          }}
          onClose={() => setPickerOpen(false)}
        />
      )}

      {migrateTarget != null && (
        <StudioDataMigrateModal
          target={migrateTarget}
          onClose={() => setMigrateTarget(null)}
          onRestart={() => void handleRestart()}
        />
      )}

      {modelsPickerOpen && (
        <PathPicker
          dirOnly
          initialPath={modelsInfo?.current}
          onPick={(path) => {
            setModelsPickerOpen(false)
            setModelsMigrateTarget(path)
          }}
          onClose={() => setModelsPickerOpen(false)}
        />
      )}

      {modelsMigrateTarget != null && (
        <ModelsRootMigrateModal
          target={modelsMigrateTarget}
          onClose={() => setModelsMigrateTarget(null)}
          onDone={refreshModelsInfo}
        />
      )}
    </SettingsSection>
  )
}

// ── 服务 Section（重新运行首次引导 + 重启 server）─────────────────────
export function ServiceSection() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const dialog = useDialog()
  const [busy, setBusy] = useState(false)

  // 清掉 localStorage 的 onboarding done 标记 + dispatch event,触发
  // FirstRunOnboardingModal 显示。不重启服务、不动 secrets。
  const handleReopenOnboarding = () => {
    clearOnboardingDone()
    window.dispatchEvent(new Event(ONBOARDING_EVENTS.open))
  }

  const handleRestart = async () => {
    const ok = await dialog.confirm(
      t('settings.confirmRestartService'),
      { tone: 'warn', okText: t('settings.restart') },
    )
    if (!ok) return

    setBusy(true)
    try {
      await api.restartServer()
    } catch (e) {
      const err = e as Error & { status?: number; code?: string; detail?: { tasks?: { name: string; id?: number }[] } }
      if (err.code === 'system.tasks_running') {
        const names = (err.detail?.tasks ?? []).map((task) => task.name || `task#${task.id ?? '?'}`).join(', ')
        toast(t('settings.taskRunningCancelFirst', { names }), 'error')
      } else {
        toast(t('settings.restartTriggerFailed', { error: err.message ?? String(e) }), 'error')
      }
      setBusy(false)
      return
    }

    void pollHealthThenReload(toast, 5 * 60_000, t('settings.restart'), () => setBusy(false), t)
  }

  return (
    <SettingsSection id="service" title={t('settings.service')}>
      <SettingsField
        label={t('settings.onboardingReopenTitle')}
        helpTooltip={<p>{t('settings.onboardingReopenHelp')}</p>}
      >
        <button
          type="button"
          onClick={handleReopenOnboarding}
          className="btn btn-secondary btn-sm self-start"
        >
          {t('settings.onboardingReopen')}
        </button>
      </SettingsField>

      <SettingsField
        label={t('settings.serviceRestartTitle')}
        helpTooltip={
          <>
            <p>{t('settings.serviceRestartHelp1')}</p>
            <p><Trans i18nKey="settings.serviceRestartHelp2" components={{ code: <code /> }} /></p>
          </>
        }
      >
        <button
          onClick={() => void handleRestart()}
          disabled={busy}
          className="btn btn-secondary btn-sm self-start"
        >
          {busy ? t('settings.restarting') : t('settings.restartServer')}
        </button>
      </SettingsField>
    </SettingsSection>
  )
}
