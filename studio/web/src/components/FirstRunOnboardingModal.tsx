// FirstRunOnboardingModal —— 首次启动引导 modal。
// 设计文档：docs/todo/onboarding-first-run.md
//
// 流程：FirstRunLangModal 选完语言 → dispatch 'studio:lang-set' →
// 本组件检测到 lang 已设 + onboarding 未 done → 弹出。
// Settings 里"重新运行首次引导"按钮 → dispatch 'studio:open-onboarding' → 强制弹出。
//
// 数据全部复用 SettingsData (catalog + secrets + SSE 推送) + 现有 download API，
// 后端不需要任何改动。
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type ModelsCatalog } from '../api/client'
import { useSettingsData } from '../lib/SettingsData'
import { useToast } from './Toast'

const ONBOARDING_DONE_KEY = 'studio.onboarding.done'
const LANG_KEY = 'studio.lang'

// CustomEvent name 集中在这里，跟 LangModal / Settings 共享。
export const ONBOARDING_EVENTS = {
  langSet: 'studio:lang-set',
  open: 'studio:open-onboarding',
} as const

export function isOnboardingDone(): boolean {
  try { return localStorage.getItem(ONBOARDING_DONE_KEY) === 'true' } catch { return false }
}
function setOnboardingDone() {
  try { localStorage.setItem(ONBOARDING_DONE_KEY, 'true') } catch { /* ignore */ }
}
export function clearOnboardingDone() {
  try { localStorage.removeItem(ONBOARDING_DONE_KEY) } catch { /* ignore */ }
}
function langIsSet(): boolean {
  try { return localStorage.getItem(LANG_KEY) !== null } catch { return false }
}

// 按 i18n 推断默认下载源：中文 → ModelScope；其余 → HuggingFace。
// 用户在国内的概率高且大多不知道 ModelScope 是啥；海外用户走 HF 直连。
export function defaultDownloadSource(lang: string | null | undefined): 'huggingface' | 'modelscope' {
  if ((lang ?? '').toLowerCase().startsWith('zh')) return 'modelscope'
  return 'huggingface'
}

// ---- item states ---------------------------------------------------------

type ItemKey = 'base' | 'tagger' | 'accel' | 'upscaler'
type ItemStatus = 'checking' | 'idle' | 'queued' | 'installing' | 'done' | 'failed'

interface ItemDerived {
  status: ItemStatus
  // 多组件聚合时显当前 X/N
  progress?: { done: number; total: number }
}

// 从 catalog 派生底模套件状态（4 个子文件全 exists 算 done）。
function deriveBaseStatus(c: ModelsCatalog | null, dl: Record<ItemKey, ItemStatus>): ItemDerived {
  if (!c) return { status: 'checking' }
  const animaLatest = c.anima_main.variants.find((v) => v.is_latest)
  const checks: boolean[] = [
    !!animaLatest?.exists,
    !!c.anima_vae.exists,
    c.qwen3.files.length > 0 && c.qwen3.files.every((f) => f.exists),
    c.t5_tokenizer.files.length > 0 && c.t5_tokenizer.files.every((f) => f.exists),
  ]
  const done = checks.filter(Boolean).length
  const total = checks.length
  if (done === total) return { status: 'done', progress: { done, total } }
  // catalog 不在 done 状态时，叠加本地 install state（installing/failed/queued）
  if (dl.base === 'failed') return { status: 'failed', progress: { done, total } }
  if (dl.base === 'installing' || dl.base === 'queued') {
    return { status: dl.base, progress: { done, total } }
  }
  return { status: 'idle', progress: { done, total } }
}

// 打标功能 = WD14 当前 variant 装好 + ONNX runtime 已装。
function deriveTaggerStatus(
  c: ModelsCatalog | null,
  hasOnnx: boolean,
  runtimeChecked: boolean,
  dl: Record<ItemKey, ItemStatus>,
): ItemDerived {
  if (!c || !runtimeChecked) return { status: 'checking' }
  const current = c.wd14.variants.find((v) => v.is_current) ?? c.wd14.variants[0]
  const wd14Ok = !!current?.exists
  const checks = [wd14Ok, hasOnnx]
  const done = checks.filter(Boolean).length
  const total = checks.length
  if (done === total) return { status: 'done', progress: { done, total } }
  if (dl.tagger === 'failed') return { status: 'failed', progress: { done, total } }
  if (dl.tagger === 'installing' || dl.tagger === 'queued') {
    return { status: dl.tagger, progress: { done, total } }
  }
  return { status: 'idle', progress: { done, total } }
}

// 训练加速：检测 flash_attn 或 xformers 任一已装(尊重用户已有环境),
// 安装时统一装 flash_attn(推荐方案,新人不用选)。
function deriveAccelStatus(
  hasFlash: boolean,
  hasXformers: boolean,
  runtimeChecked: boolean,
  dl: Record<ItemKey, ItemStatus>,
): ItemDerived {
  if (!runtimeChecked) return { status: 'checking' }
  if (hasFlash || hasXformers) return { status: 'done' }
  if (dl.accel === 'failed') return { status: 'failed' }
  if (dl.accel === 'installing' || dl.accel === 'queued') return { status: dl.accel }
  return { status: 'idle' }
}

// Upscaler = 当前 selected variant exists。
function deriveUpscalerStatus(
  c: ModelsCatalog | null,
  dl: Record<ItemKey, ItemStatus>,
): ItemDerived {
  if (!c) return { status: 'checking' }
  if (!c.upscalers) return { status: dl.upscaler }
  const current =
    c.upscalers.variants.find((v) => v.is_current)
    ?? c.upscalers.variants.find((v) => v.label === c.upscalers!.default)
  if (current?.exists) return { status: 'done' }
  if (dl.upscaler === 'failed') return { status: 'failed' }
  if (dl.upscaler === 'installing' || dl.upscaler === 'queued') return { status: dl.upscaler }
  return { status: 'idle' }
}

// ---- main component ------------------------------------------------------

export function FirstRunOnboardingModal() {
  const { t, i18n } = useTranslation()
  const { secrets, setSecrets, catalog, reloadCatalog } = useSettingsData()
  const { toast } = useToast()

  const [open, setOpen] = useState<boolean>(false)
  // 选中要装的条目；初次 mount 时按"推荐勾选"填充。
  const [selected, setSelected] = useState<Set<ItemKey>>(new Set(['base', 'tagger']))
  // 安装本地状态机（catalog 还没反映到结果时用）。
  const [installState, setInstallState] = useState<Record<ItemKey, ItemStatus>>({
    base: 'idle', tagger: 'idle', accel: 'idle', upscaler: 'idle',
  })
  const [failureLogs, setFailureLogs] = useState<Record<ItemKey, string[]>>({
    base: [], tagger: [], accel: [], upscaler: [],
  })
  const [runtimeStatus, setRuntimeStatus] = useState<{
    onnx: boolean
    flashAttn: boolean
    xformers: boolean
  }>({ onnx: false, flashAttn: false, xformers: false })
  // runtime status 是否拉过(区分"还没拉"和"拉完都没装");catalog 有自己的 null
  // 表示未拉,runtime 是个对象所以需要单独 flag。
  const [runtimeChecked, setRuntimeChecked] = useState<boolean>(false)
  const [restartRequired, setRestartRequired] = useState<boolean>(false)
  const [restarting, setRestarting] = useState<boolean>(false)
  const [currentItem, setCurrentItem] = useState<ItemKey | null>(null)
  const [savingSource, setSavingSource] = useState<boolean>(false)
  // first-run 自动按 i18n 写一次 secrets.download_source(只跑一次,避免覆盖用户手动改)。
  const autoSourceWrittenRef = useRef<boolean>(false)
  // catalog 最新值 ref —— installItem polling 闭包里看不到 catalog state 更新,
  // 必须经 ref 拿。catalog 本身通过 SettingsData 的 SSE 自动刷新。
  const catalogRef = useRef(catalog)
  useEffect(() => { catalogRef.current = catalog }, [catalog])

  // 触发：lang 已设 + onboarding 未 done → 自动弹；显式 open 事件 → 强制弹。
  useEffect(() => {
    const evaluate = () => { if (langIsSet() && !isOnboardingDone()) setOpen(true) }
    const forceOpen = () => setOpen(true)
    // mount 时尝试一次（lang 可能在本组件 mount 前就设好了）
    evaluate()
    window.addEventListener(ONBOARDING_EVENTS.langSet, evaluate)
    window.addEventListener(ONBOARDING_EVENTS.open, forceOpen)
    return () => {
      window.removeEventListener(ONBOARDING_EVENTS.langSet, evaluate)
      window.removeEventListener(ONBOARDING_EVENTS.open, forceOpen)
    }
  }, [])

  // 拉运行时（ONNX / Flash-Attn / Xformers）状态。open 时拉一次，重启回来也会重新 mount。
  useEffect(() => {
    if (!open) return
    let cancelled = false
    Promise.all([
      api.getWD14Runtime().catch(() => null),
      api.getFlashAttnStatus().catch(() => null),
      api.getXformersStatus().catch(() => null),
    ]).then(([wd, fa, xf]) => {
      if (cancelled) return
      setRuntimeStatus({
        onnx: !!wd?.installed,
        flashAttn: !!fa?.installed,
        xformers: !!xf?.installed,
      })
      setRuntimeChecked(true)
    })
    return () => { cancelled = true }
  }, [open])

  // 派生每个条目状态。
  const itemStatus = useMemo<Record<ItemKey, ItemDerived>>(() => ({
    base: deriveBaseStatus(catalog, installState),
    tagger: deriveTaggerStatus(catalog, runtimeStatus.onnx, runtimeChecked, installState),
    accel: deriveAccelStatus(runtimeStatus.flashAttn, runtimeStatus.xformers, runtimeChecked, installState),
    upscaler: deriveUpscalerStatus(catalog, installState),
  }), [catalog, installState, runtimeStatus, runtimeChecked])

  // 全部装完判定：选中的条目全 done。
  const allSelectedDone = useMemo(() => {
    for (const k of selected) {
      if (itemStatus[k].status !== 'done') return false
    }
    return selected.size > 0
  }, [selected, itemStatus])

  // 是否正在跑一键装。
  const installingNow = useMemo(() => {
    return (['base', 'tagger', 'accel', 'upscaler'] as ItemKey[])
      .some((k) => installState[k] === 'installing' || installState[k] === 'queued')
  }, [installState])

  // catalog 或 runtime status 还在拉,任一条目处于 checking 态。
  const isChecking = useMemo(() => {
    return (['base', 'tagger', 'accel', 'upscaler'] as ItemKey[])
      .some((k) => itemStatus[k].status === 'checking')
  }, [itemStatus])

  // 派生每个条目对应 catalog.downloads 里 running 任务的 log_tail,实时显示
  // 下载速度 / tqdm 进度;accel(pip install) 不进 catalog.downloads,空。
  const liveLogs = useMemo<Record<ItemKey, string[]>>(() => {
    const empty = { base: [], tagger: [], accel: [], upscaler: [] }
    if (!catalog) return empty
    const collect = (filter: (k: string) => boolean): string[] => {
      const tails = Object.entries(catalog.downloads)
        .filter(([k]) => filter(k))
        .filter(([, v]) => v.status === 'running' || v.status === 'pending')
        .map(([, v]) => v.log_tail)
      return tails.flat()
    }
    return {
      base: collect((k) => k.startsWith('anima_main') || k.startsWith('anima_vae')
                       || k.startsWith('qwen3') || k.startsWith('t5_tokenizer')),
      tagger: collect((k) => k.startsWith('wd14')),
      accel: [],
      upscaler: collect((k) => k.startsWith('upscaler')),
    }
  }, [catalog])

  const toggleSelected = useCallback((key: ItemKey) => {
    if (installingNow) return
    setSelected((s) => {
      const n = new Set(s)
      if (n.has(key)) n.delete(key); else n.add(key)
      return n
    })
  }, [installingNow])

  // 保存下载源到 secrets。
  const saveDownloadSource = useCallback(async (next: 'huggingface' | 'modelscope') => {
    if (!secrets) return
    setSavingSource(true)
    try {
      const updated = await api.updateSecrets({ download_source: next })
      setSecrets(updated)
    } catch {
      // 静默失败：保留 UI 选择，用户可重试
    } finally {
      setSavingSource(false)
    }
  }, [secrets, setSecrets])

  // 当前下载源（secrets 为空时按 i18n 兜底显示）。
  const downloadSource: 'huggingface' | 'modelscope' = useMemo(() => {
    const fromSecrets = (secrets?.download_source ?? '').toLowerCase()
    if (fromSecrets === 'huggingface' || fromSecrets === 'modelscope') return fromSecrets
    return defaultDownloadSource(i18n.language)
  }, [secrets, i18n.language])

  // First-run 时按 i18n 推断把默认源写入 secrets,让中文用户跳过 onboarding 后
  // 进 Settings 也能看到 ModelScope(而不是后端 hardcoded default 'huggingface')。
  // 只在 onboarding 未 done(首次启动)时跑,跑过一次后 ref 锁住,用户主动改不会被覆盖。
  useEffect(() => {
    if (!open || !secrets || autoSourceWrittenRef.current) return
    if (isOnboardingDone()) {
      autoSourceWrittenRef.current = true
      return
    }
    const inferred = defaultDownloadSource(i18n.language)
    if (secrets.download_source !== inferred) {
      void saveDownloadSource(inferred)
    }
    autoSourceWrittenRef.current = true
  }, [open, secrets, i18n.language, saveDownloadSource])

  // 等 catalog 反映"装齐"或"失败",最长 timeoutMs。SettingsData 通过 SSE
  // model_download_changed 自动 reloadCatalog,所以这里只要轮询 ref 就行。
  // 任意一个 relevant download 进 failed → 立刻返回失败;全 done → 成功。
  const waitForCatalog = useCallback(async (
    isAllDone: (c: ModelsCatalog) => boolean,
    keyFilter: (k: string) => boolean,
    timeoutMs = 30 * 60_000,
  ): Promise<{ ok: boolean; errors: string[] }> => {
    const deadline = Date.now() + timeoutMs
    while (Date.now() < deadline) {
      const c = catalogRef.current
      if (c) {
        if (isAllDone(c)) return { ok: true, errors: [] }
        const failed = Object.entries(c.downloads)
          .filter(([k]) => keyFilter(k))
          .map(([, v]) => v)
          .filter((d) => d.status === 'failed')
        if (failed.length > 0) {
          const tail = failed.flatMap((d) => d.log_tail.slice(-10))
          return { ok: false, errors: tail.length > 0 ? tail : [failed.map((d) => d.message).join('\n')] }
        }
      }
      await new Promise((r) => setTimeout(r, 1500))
    }
    return { ok: false, errors: ['Timed out waiting for download to finish'] }
  }, [])

  // 安装单个条目。触发后端开始下载/装包,然后 polling catalog 等到真完成,
  // 才把状态切到 done/failed —— startModelDownload 只是 fire-and-forget,
  // 立刻 set done 会让派生状态短暂闪一下又回 idle。
  const installItem = useCallback(async (key: ItemKey): Promise<{ ok: boolean; needsRestart: boolean }> => {
    setInstallState((s) => ({ ...s, [key]: 'installing' }))
    setCurrentItem(key)
    let needsRestart = false
    const errors: string[] = []
    try {
      if (key === 'base' && catalog) {
        const latest = catalog.anima_main.variants.find((v) => v.is_latest)
        const triggers: Promise<unknown>[] = []
        if (latest && !latest.exists) {
          triggers.push(api.startModelDownload({ model_id: 'anima_main', variant: latest.variant }))
        }
        if (!catalog.anima_vae.exists) {
          triggers.push(api.startModelDownload({ model_id: 'anima_vae' }))
        }
        if (!catalog.qwen3.files.every((f) => f.exists)) {
          triggers.push(api.startModelDownload({ model_id: 'qwen3' }))
        }
        if (!catalog.t5_tokenizer.files.every((f) => f.exists)) {
          triggers.push(api.startModelDownload({ model_id: 't5_tokenizer' }))
        }
        if (triggers.length > 0) {
          await Promise.all(triggers)
          const res = await waitForCatalog(
            (c) => {
              const a = c.anima_main.variants.find((v) => v.is_latest)
              return !!a?.exists
                && !!c.anima_vae.exists
                && c.qwen3.files.length > 0 && c.qwen3.files.every((f) => f.exists)
                && c.t5_tokenizer.files.length > 0 && c.t5_tokenizer.files.every((f) => f.exists)
            },
            (k) => k.startsWith('anima_main') || k.startsWith('anima_vae')
                || k.startsWith('qwen3') || k.startsWith('t5_tokenizer'),
          )
          if (!res.ok) errors.push(...res.errors)
        }
      } else if (key === 'tagger' && catalog) {
        const current = catalog.wd14.variants.find((v) => v.is_current) ?? catalog.wd14.variants[0]
        const needWd14 = current && !current.exists
        if (needWd14) {
          await api.startModelDownload({ model_id: 'wd14', variant: current.model_id })
          const res = await waitForCatalog(
            (c) => {
              const cur = c.wd14.variants.find((v) => v.is_current) ?? c.wd14.variants[0]
              return !!cur?.exists
            },
            (k) => k.startsWith('wd14'),
          )
          if (!res.ok) errors.push(...res.errors)
        }
        if (errors.length === 0 && !runtimeStatus.onnx) {
          const r = await api.installWD14Runtime('auto')
          if (r.installed) {
            setRuntimeStatus((s) => ({ ...s, onnx: true }))
            needsRestart = needsRestart || !!r.restart_required
          } else {
            errors.push(r.stdout_tail.split('\n').slice(-10).join('\n'))
          }
        }
      } else if (key === 'accel') {
        if (!runtimeStatus.flashAttn && !runtimeStatus.xformers) {
          const r = await api.installFlashAttn(null)
          if (r.installed) {
            setRuntimeStatus((s) => ({ ...s, flashAttn: true }))
            needsRestart = needsRestart || !!r.restart_required
          } else {
            errors.push(r.stdout_tail.split('\n').slice(-10).join('\n'))
          }
        }
      } else if (key === 'upscaler' && catalog?.upscalers) {
        const target =
          catalog.upscalers.variants.find((v) => v.is_current)
          ?? catalog.upscalers.variants.find((v) => v.label === catalog.upscalers!.default)
        if (target && !target.exists) {
          await api.startModelDownload({ model_id: 'upscaler', variant: target.label })
          const res = await waitForCatalog(
            (c) => {
              if (!c.upscalers) return true
              const t = c.upscalers.variants.find((v) => v.is_current)
                ?? c.upscalers.variants.find((v) => v.label === c.upscalers!.default)
              return !!t?.exists
            },
            (k) => k.startsWith('upscaler'),
          )
          if (!res.ok) errors.push(...res.errors)
        }
      }
    } catch (e) {
      errors.push(String(e))
    }
    const ok = errors.length === 0
    if (!ok) {
      setFailureLogs((s) => ({ ...s, [key]: errors }))
      setInstallState((s) => ({ ...s, [key]: 'failed' }))
    } else {
      setInstallState((s) => ({ ...s, [key]: 'done' }))
    }
    await reloadCatalog()
    return { ok, needsRestart }
  }, [catalog, runtimeStatus, reloadCatalog, waitForCatalog])

  // 一键装：串行跑选中条目（后端有些接口本身是同步阻塞 pip，无法并行）。
  const installAll = useCallback(async () => {
    // 先把选中且非 done 的条目标为 queued
    const queue: ItemKey[] = (['base', 'tagger', 'accel', 'upscaler'] as ItemKey[])
      .filter((k) => selected.has(k) && itemStatus[k].status !== 'done')
    setInstallState((s) => {
      const n = { ...s }
      for (const k of queue) n[k] = 'queued'
      return n
    })
    let anyRestart = false
    for (const k of queue) {
      const { needsRestart } = await installItem(k)
      anyRestart = anyRestart || needsRestart
    }
    setCurrentItem(null)
    if (anyRestart) setRestartRequired(true)
  }, [selected, itemStatus, installItem])

  const retryItem = useCallback((key: ItemKey) => {
    setInstallState((s) => ({ ...s, [key]: 'queued' }))
    setFailureLogs((s) => ({ ...s, [key]: [] }))
    void installItem(key)
  }, [installItem])

  const handleRestart = useCallback(async () => {
    setRestarting(true)
    try {
      await api.restartServer()
      // restart_required 后会有专用轮询；这里简化为 5s 后 reload，跟 Settings 里
      // 的 pollHealthThenReload 行为对齐（详细轮询逻辑暂用 location.reload 兜底）。
      setTimeout(() => { window.location.reload() }, 5000)
    } catch {
      setRestarting(false)
    }
  }, [])

  const handleClose = useCallback(() => {
    setOnboardingDone()
    setOpen(false)
  }, [])

  const handleSkip = useCallback(() => {
    // 装中也允许关 modal —— 后端 daemon thread 不受影响,继续下载。
    // 给个 toast 提示让用户知道不是"按了就停",真要停得重启 Studio。
    if (installingNow) toast(t('onboarding.skipWhileInstalling'), 'info')
    handleClose()
  }, [installingNow, handleClose, toast, t])

  if (!open) return null

  const selectedCount = (['base', 'tagger', 'accel', 'upscaler'] as ItemKey[])
    .filter((k) => selected.has(k) && itemStatus[k].status !== 'done').length

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="first-run-onboarding-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-md p-4"
      data-testid="first-run-onboarding-modal"
    >
      <div className="w-full max-w-[640px] max-h-[90vh] flex flex-col bg-elevated border border-dim rounded-lg shadow-xl overflow-hidden">
        <div className="p-6 pb-4 border-b border-dim">
          <h1
            id="first-run-onboarding-title"
            className="m-0 text-2xl font-semibold text-fg-primary"
          >
            {t('onboarding.title')}
          </h1>
          <p className="mt-2 mb-0 text-sm text-fg-secondary">
            {t('onboarding.subtitle')}
          </p>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-4 flex flex-col gap-4">
          <DownloadSourceRow
            value={downloadSource}
            onChange={saveDownloadSource}
            disabled={savingSource}
            installHint={installingNow ? t('onboarding.downloadSourceInstallHint') : null}
          />

          <div className="border-t border-dim" />

          <ChecklistItem
            itemKey="base"
            label={t('onboarding.items.base.label')}
            description={t('onboarding.items.base.description')}
            checked={selected.has('base')}
            onToggle={() => toggleSelected('base')}
            status={itemStatus.base}
            failureLog={failureLogs.base}
            liveLog={liveLogs.base}
            onRetry={() => retryItem('base')}
            disabled={installingNow}
          />

          <ChecklistItem
            itemKey="tagger"
            label={t('onboarding.items.tagger.label')}
            description={t('onboarding.items.tagger.description')}
            checked={selected.has('tagger')}
            onToggle={() => toggleSelected('tagger')}
            status={itemStatus.tagger}
            failureLog={failureLogs.tagger}
            liveLog={liveLogs.tagger}
            onRetry={() => retryItem('tagger')}
            disabled={installingNow}
          />

          <ChecklistItem
            itemKey="accel"
            label={t('onboarding.items.accel.label')}
            description={t('onboarding.items.accel.description')}
            checked={selected.has('accel')}
            onToggle={() => toggleSelected('accel')}
            status={itemStatus.accel}
            failureLog={failureLogs.accel}
            liveLog={liveLogs.accel}
            onRetry={() => retryItem('accel')}
            disabled={installingNow}
          />

          <ChecklistItem
            itemKey="upscaler"
            label={t('onboarding.items.upscaler.label')}
            description={t('onboarding.items.upscaler.description')}
            checked={selected.has('upscaler')}
            onToggle={() => toggleSelected('upscaler')}
            status={itemStatus.upscaler}
            failureLog={failureLogs.upscaler}
            liveLog={liveLogs.upscaler}
            onRetry={() => retryItem('upscaler')}
            disabled={installingNow}
          />
        </div>

        <div className="border-t border-dim p-4 flex items-center justify-between bg-surface">
          <button
            type="button"
            onClick={handleSkip}
            disabled={false}
            className="px-3 py-1.5 text-sm text-fg-tertiary hover:text-fg-primary disabled:opacity-50 disabled:cursor-not-allowed bg-transparent border-none cursor-pointer"
            data-testid="onboarding-skip"
          >
            {t('onboarding.skip')}
          </button>

          {installingNow ? (
            <div className="text-sm text-fg-secondary">
              {currentItem ? t(`onboarding.installing.${currentItem}`) : t('onboarding.installing.generic')}
            </div>
          ) : restartRequired ? (
            <div className="flex items-center gap-3">
              <span className="text-sm text-fg-secondary">{t('onboarding.restartHint')}</span>
              <button
                type="button"
                onClick={handleRestart}
                disabled={restarting}
                className="px-4 py-2 text-sm font-medium bg-accent text-white rounded hover:opacity-90 disabled:opacity-50 cursor-pointer"
                data-testid="onboarding-restart"
              >
                {restarting ? t('onboarding.restarting') : t('onboarding.restart')}
              </button>
            </div>
          ) : isChecking ? (
            <button
              type="button"
              disabled
              className="px-4 py-2 text-sm font-medium bg-accent text-white rounded opacity-50 cursor-not-allowed"
              data-testid="onboarding-checking"
            >
              {t('onboarding.checkingState')}
            </button>
          ) : allSelectedDone ? (
            <button
              type="button"
              onClick={handleClose}
              className="px-4 py-2 text-sm font-medium bg-accent text-white rounded hover:opacity-90 cursor-pointer"
              data-testid="onboarding-finish"
            >
              {t('onboarding.finish')}
            </button>
          ) : (
            <button
              type="button"
              onClick={installAll}
              disabled={selectedCount === 0}
              className="px-4 py-2 text-sm font-medium bg-accent text-white rounded hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
              data-testid="onboarding-install-all"
            >
              {t('onboarding.installSelected', { count: selectedCount })}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

// ---- subcomponents --------------------------------------------------------

function DownloadSourceRow(props: {
  value: 'huggingface' | 'modelscope'
  onChange: (v: 'huggingface' | 'modelscope') => void
  disabled: boolean
  installHint: string | null
}) {
  const { t } = useTranslation()
  return (
    <div className="flex flex-col gap-2">
      <div className="text-sm font-medium text-fg-primary">{t('onboarding.downloadSource')}</div>
      <div className="flex gap-2">
        {(['modelscope', 'huggingface'] as const).map((src) => (
          <button
            key={src}
            type="button"
            onClick={() => props.onChange(src)}
            disabled={props.disabled}
            className={`flex-1 px-3 py-2 text-sm border rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer ${
              props.value === src
                ? 'border-accent bg-accent-soft text-fg-primary'
                : 'border-dim bg-surface text-fg-secondary hover:border-accent/50'
            }`}
            data-testid={`onboarding-source-${src}`}
          >
            {src === 'modelscope' ? 'ModelScope' : 'HuggingFace'}
          </button>
        ))}
      </div>
      <div className="text-xs text-fg-tertiary">
        {props.installHint ?? t('onboarding.downloadSourceHint')}
      </div>
    </div>
  )
}

function StatusBadge({ status, progress }: { status: ItemStatus; progress?: { done: number; total: number } }) {
  const { t } = useTranslation()
  const baseClass = 'inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded'
  switch (status) {
    case 'done':
      return <span className={`${baseClass} bg-accent-soft text-accent`}>✓ {t('onboarding.status.done')}</span>
    case 'installing': {
      const suffix = progress ? ` (${progress.done}/${progress.total})` : ''
      return <span className={`${baseClass} bg-info-soft text-info`}>⏳ {t('onboarding.status.installing')}{suffix}</span>
    }
    case 'queued':
      return <span className={`${baseClass} bg-surface text-fg-tertiary`}>⏱ {t('onboarding.status.queued')}</span>
    case 'failed':
      return <span className={`${baseClass} bg-err-soft text-err`}>✕ {t('onboarding.status.failed')}</span>
    case 'checking':
      return <span className={`${baseClass} bg-surface text-fg-tertiary`}>⋯ {t('onboarding.status.checking')}</span>
    default: {
      const suffix = progress && progress.done > 0 ? ` (${progress.done}/${progress.total})` : ''
      return <span className={`${baseClass} bg-surface text-fg-tertiary`}>○ {t('onboarding.status.idle')}{suffix}</span>
    }
  }
}

function ChecklistItem(props: {
  itemKey: ItemKey
  label: string
  description: string
  checked: boolean
  onToggle: () => void
  status: ItemDerived
  failureLog: string[]
  liveLog: string[]
  onRetry: () => void
  disabled: boolean
}) {
  const { t } = useTranslation()
  const [logOpen, setLogOpen] = useState(false)
  const isDone = props.status.status === 'done'
  const isInstalling = props.status.status === 'installing'
  const isFailed = props.status.status === 'failed'
  // installing 时默认展开 log,让用户能看到下载速度(tqdm 输出);failed 默认折叠
  // (避免错误信息一下糊到屏幕)。这里用 useEffect 跟着状态切默认值。
  useEffect(() => {
    if (isInstalling) setLogOpen(true)
    if (isDone) setLogOpen(false)
  }, [isInstalling, isDone])
  const shownLog = isFailed ? props.failureLog : props.liveLog
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-start gap-3">
        <input
          type="checkbox"
          checked={props.checked || isDone}
          onChange={props.onToggle}
          disabled={props.disabled}
          className="mt-1 cursor-pointer disabled:cursor-not-allowed"
          style={{ accentColor: 'var(--accent)' }}
          data-testid={`onboarding-check-${props.itemKey}`}
        />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <div className="text-sm font-medium text-fg-primary">{props.label}</div>
            <StatusBadge status={props.status.status} progress={props.status.progress} />
            {(isInstalling || isFailed) && shownLog.length > 0 && (
              <button
                type="button"
                onClick={() => setLogOpen((v) => !v)}
                className="ml-auto px-2 py-0.5 text-xs text-fg-tertiary hover:text-fg-primary bg-transparent border-none cursor-pointer"
              >
                {logOpen ? t('onboarding.hideLog') : t('onboarding.showLog')}
              </button>
            )}
          </div>
          <div className="text-xs text-fg-tertiary mt-0.5">{props.description}</div>
          {isFailed && (
            <div className="mt-2 flex items-center gap-2">
              <button
                type="button"
                onClick={props.onRetry}
                className="px-2 py-1 text-xs bg-surface border border-dim rounded hover:border-accent cursor-pointer"
                data-testid={`onboarding-retry-${props.itemKey}`}
              >
                {t('onboarding.retry')}
              </button>
            </div>
          )}
          {logOpen && shownLog.length > 0 && (
            <pre className="mt-2 p-2 text-xs bg-surface border border-dim rounded overflow-x-auto whitespace-pre-wrap max-h-40 font-mono leading-snug">
              {shownLog.join('\n')}
            </pre>
          )}
        </div>
      </div>
    </div>
  )
}

