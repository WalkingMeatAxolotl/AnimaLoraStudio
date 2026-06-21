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
  useState,
  type ReactNode,
} from 'react'
import { useTranslation } from 'react-i18next'
import { api, type ModelsCatalog, type Secrets } from '../api/client'
import { useToast } from '../components/Toast'
import { useEventStream } from './useEventStream'

interface SettingsData {
  secrets: Secrets | null
  secretsError: string | null
  setSecrets: (s: Secrets) => void
  catalog: ModelsCatalog | null
  catalogError: string | null
  reloadCatalog: () => Promise<void>
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

  useEffect(() => {
    api.getSecrets()
      .then((s) => { setSecrets(s); setSecretsError(null) })
      .catch((e) => setSecretsError(String(e)))
  }, [])

  const reloadCatalog = useCallback(async () => {
    try {
      const c = await api.getModelsCatalog()
      setCatalog(c)
      setCatalogError(null)
    } catch (e) {
      setCatalogError(String(e))
    }
  }, [])

  useEffect(() => { void reloadCatalog() }, [reloadCatalog])

  useEventStream((evt) => {
    if (evt.type === 'model_download_changed') { void reloadCatalog() }
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

  return (
    <Ctx.Provider value={{
      secrets, secretsError, setSecrets,
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
