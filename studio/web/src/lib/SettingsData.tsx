// SettingsData.tsx —— Settings 全局数据层。
//
// 把 secrets / catalog / downloadBusy / SSE 订阅从 SettingsPage 提到根级 Provider，
// 让 SettingsPage 本身可以 unmount + remount 不付重新拉数据的代价：
// - secrets：一次 fetch，常驻 context；save 后由 SettingsPage 调 setSecrets 更新
// - catalog：reloadCatalog + model_download_changed SSE 订阅常驻，跟下载组件共享
// - downloadBusy：跟 startDownload 配对的 in-flight Set
//
// 这层只持有数据，不渲染 UI。SettingsDrawer 关闭时 SettingsPage 卸载，
// 第二次打开瞬间渲染——数据已经在 context 里。
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { useTranslation } from 'react-i18next'
import { api, type ModelsCatalog, type Secrets, type SecretsPatch } from '../api/client'
import { useToast } from '../components/Toast'
import { useEventStream } from './useEventStream'

// 全局「已保存」状态指示（instant-apply 下取代旧的保存按钮 dirty 态）。
export type SaveStatus =
  | { state: 'idle' }
  | { state: 'saving' }
  | { state: 'saved'; at: number }
  | { state: 'error'; error: string }

// 把单字段 patch 浅合并进本地 secrets（乐观更新用）。secrets 是两层结构
// （section → fields），顶层标量字段（如 download_source）直接覆盖。
function mergePatchLocal(base: Secrets, patch: SecretsPatch): Secrets {
  const out = { ...base } as Record<string, unknown>
  for (const key of Object.keys(patch)) {
    const pv = (patch as Record<string, unknown>)[key]
    const bv = out[key]
    if (pv && typeof pv === 'object' && !Array.isArray(pv)
        && bv && typeof bv === 'object' && !Array.isArray(bv)) {
      out[key] = { ...(bv as object), ...(pv as object) }
    } else {
      out[key] = pv
    }
  }
  return out as unknown as Secrets
}

interface SettingsData {
  secrets: Secrets | null
  secretsError: string | null
  setSecrets: (s: Secrets) => void
  /** instant-apply 统一写入入口：乐观更新 + 串行 PUT 单字段 patch。 */
  commitSecrets: (patch: SecretsPatch) => void
  saveStatus: SaveStatus
  catalog: ModelsCatalog | null
  catalogError: string | null
  reloadCatalog: () => Promise<ModelsCatalog | null>
  downloadBusy: Set<string>
  startDownload: (model_id: string, variant?: string) => Promise<void>
  setDownloadSource: (type: string, source: string) => Promise<void>
}

const Ctx = createContext<SettingsData | null>(null)

export function SettingsDataProvider({ children }: { children: ReactNode }) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [secrets, setSecrets] = useState<Secrets | null>(null)
  const [secretsError, setSecretsError] = useState<string | null>(null)
  const [catalog, setCatalog] = useState<ModelsCatalog | null>(null)
  const [catalogError, setCatalogError] = useState<string | null>(null)
  const [downloadBusy, setDownloadBusy] = useState<Set<string>>(new Set())
  const [saveStatus, setSaveStatus] = useState<SaveStatus>({ state: 'idle' })
  // 串行 PUT 队列 + in-flight 计数：保证顺序避免后端读改写竞态。
  const saveQueueRef = useRef<Promise<unknown>>(Promise.resolve())
  const pendingRef = useRef(0)

  useEffect(() => {
    api.getSecrets()
      .then((s) => { setSecrets(s); setSecretsError(null) })
      .catch((e) => setSecretsError(String(e)))
  }, [])

  const reloadCatalog = useCallback(async (): Promise<ModelsCatalog | null> => {
    try {
      const c = await api.getModelsCatalog()
      setCatalog(c)
      setCatalogError(null)
      return c
    } catch (e) {
      setCatalogError(String(e))
      return null
    }
  }, [])

  useEffect(() => { void reloadCatalog() }, [reloadCatalog])

  // model_download_changed 既驱动 catalog 刷新，也是下载失败时的唯一全局信号：
  // 下载在后台线程跑，失败原因（如 gated 仓库缺 token）只进 download status，
  // 不会让 startDownload 的 await 抛错。这里在 failed 时弹一个 error toast，
  // 把后端汇总的可操作 message 顶到用户面前——否则用户只看到卡片上一个红 badge，
  // 原因埋在另一个 tab 的折叠「下载日志」里（甚至只在终端）。
  useEventStream((evt) => {
    if (evt.type !== 'model_download_changed') return
    void reloadCatalog().then((c) => {
      if (evt.status !== 'failed' || !c) return
      const key = String(evt.key ?? '')
      const dl = c.downloads[key]
      toast(dl?.message || t('settings.downloadFailed', { error: key }), 'error')
    })
  })

  const startDownload = useCallback(async (model_id: string, variant?: string) => {
    const key = variant ? `${model_id}:${variant}` : model_id
    setDownloadBusy((s) => new Set(s).add(key))
    try {
      await api.startModelDownload({ model_id, variant })
      toast(t('settings.downloadStarted', { name: key }), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setDownloadBusy((s) => { const n = new Set(s); n.delete(key); return n })
    }
  }, [reloadCatalog, t, toast])

  // 按类型选下载源：即时存（跟「下载」/ models.root 一样是立即动作，不进表单
  // draft）。刻意不 setSecrets —— 否则会让 SettingsPage 的 draft/server 失同步，
  // 表单 Save 时把这次改动 clobber 回去。dropdown 当前值读 catalog（reloadCatalog
  // 刷新），不依赖表单 secrets。
  const setDownloadSource = useCallback(async (type: string, source: string) => {
    try {
      await api.updateSecrets({ download_sources: { [type]: source } })
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    }
  }, [reloadCatalog, toast])

  // instant-apply 统一写入：乐观更新本地 secrets 让控件立即反映，PUT 单字段
  // patch 入串行队列。队列全部清空后用后端权威结果回写一次（拿 validator
  // 规范化 + 敏感字段 mask）——避免连改多个字段时早 PUT 的权威结果覆盖掉
  // 后面字段的乐观值（中途闪回）。失败时重拉 secrets 恢复一致。
  const commitSecrets = useCallback((patch: SecretsPatch) => {
    setSecrets((s) => (s ? mergePatchLocal(s, patch) : s))
    setSaveStatus({ state: 'saving' })
    pendingRef.current += 1
    saveQueueRef.current = saveQueueRef.current
      .then(() => api.updateSecrets(patch))
      .then((authoritative) => {
        pendingRef.current -= 1
        // 每个 PUT 完成都刷新「已保存」时间戳，让连续保存每次都有可见反馈；
        // 权威结果只在队列清空时回写一次，避免中途覆盖后续字段的乐观值。
        if (pendingRef.current === 0) setSecrets(authoritative)
        setSaveStatus({ state: 'saved', at: Date.now() })
      })
      .catch((e) => {
        pendingRef.current -= 1
        setSaveStatus({ state: 'error', error: String(e) })
        toast(String(e), 'error')
        api.getSecrets().then((s) => setSecrets(s)).catch(() => {})
      })
  }, [toast])

  return (
    <Ctx.Provider value={{
      secrets, secretsError, setSecrets, commitSecrets, saveStatus,
      catalog, catalogError, reloadCatalog,
      downloadBusy, startDownload, setDownloadSource,
    }}>
      {children}
    </Ctx.Provider>
  )
}

export function useSettingsData(): SettingsData {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useSettingsData must be used inside <SettingsDataProvider>')
  return ctx
}
