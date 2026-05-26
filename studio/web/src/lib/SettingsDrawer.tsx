// SettingsDrawer.tsx —— Settings 抽屉的全局打开/关闭 store。
//
// 路由化的旧设置页（/tools/settings）已删除，现在所有"打开设置"动作都走
// useSettingsDrawer().open({ section? })。状态在内存，刷新页面默认关闭，
// 跟用户对"抽屉"的心智模型一致。
//
// dirty guard：SettingsPage 通过 registerDirtyGuard 注册一个查询函数，
// drawer 关闭前调用它决定是否弹 confirm。这样守护逻辑住在表单侧（它本来就知道
// 自己 dirty 不 dirty），drawer 侧只负责询问用户。
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
import { useDialog } from '../components/Dialog'

interface OpenOptions {
  /** 跳转到指定 section（对应 SettingsPage 内的 DOM id）；不传则维持上次 tab。 */
  section?: string
}

interface SettingsDrawerApi {
  isOpen: boolean
  /** 上一次 open 调用带的 section；SettingsPage 内 effect 监听它做 scrollIntoView。
   *  每次 open 哪怕同一 section 也会换引用，便于触发 effect。 */
  sectionRequest: { section: string; nonce: number } | null
  open: (opts?: OpenOptions) => void
  close: () => void
  /** SettingsPage mount 时注册「当前 draft 是否 dirty」的查询函数；unmount 时传 null。 */
  registerDirtyGuard: (fn: (() => boolean) | null) => void
}

const Ctx = createContext<SettingsDrawerApi | null>(null)

export function SettingsDrawerProvider({ children }: { children: ReactNode }) {
  const { t } = useTranslation()
  const { confirm } = useDialog()
  const [isOpen, setIsOpen] = useState(false)
  const [sectionRequest, setSectionRequest] = useState<SettingsDrawerApi['sectionRequest']>(null)
  const dirtyGuardRef = useRef<(() => boolean) | null>(null)
  const nonceRef = useRef(0)

  const open = useCallback((opts?: OpenOptions) => {
    // 每次 open 都重置 sectionRequest：没传 section 就显式清空，避免上次 open 留下
    // 的 section 在下一次"无 section 打开 → SettingsPage 重新 mount"时被旧 effect
    // 再次消费，导致预期外的滚动。
    if (opts?.section) {
      nonceRef.current += 1
      setSectionRequest({ section: opts.section, nonce: nonceRef.current })
    } else {
      setSectionRequest(null)
    }
    setIsOpen(true)
  }, [])

  const close = useCallback(async () => {
    if (dirtyGuardRef.current?.()) {
      const ok = await confirm(t('settings.closeDirtyConfirm'), {
        tone: 'warn',
        title: t('settings.closeDirtyTitle'),
        okText: t('settings.closeDirtyOk'),
      })
      if (!ok) return
    }
    setIsOpen(false)
  }, [confirm, t])

  const registerDirtyGuard = useCallback((fn: (() => boolean) | null) => {
    dirtyGuardRef.current = fn
  }, [])

  // ESC 关：放在 Provider 里而非 Drawer 组件，避免 lazy 加载期间快捷键失效。
  useEffect(() => {
    if (!isOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        void close()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [isOpen, close])

  return (
    <Ctx.Provider value={{ isOpen, sectionRequest, open, close, registerDirtyGuard }}>
      {children}
    </Ctx.Provider>
  )
}

export function useSettingsDrawer(): SettingsDrawerApi {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useSettingsDrawer must be used inside <SettingsDrawerProvider>')
  return ctx
}
