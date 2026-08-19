import {
  createContext,
  useCallback,
  useContext,
  useState,
  type ReactNode,
} from 'react'

type Kind = 'info' | 'success' | 'error'

interface ToastItem {
  id: number
  kind: Kind
  message: string
}

interface ToastApi {
  toast: (msg: string, kind?: Kind) => void
}

const Ctx = createContext<ToastApi | null>(null)

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([])

  const toast = useCallback((message: string, kind: Kind = 'info') => {
    const id = Date.now() + Math.random()
    setItems((arr) => [...arr, { id, kind, message }])
    window.setTimeout(() => {
      setItems((arr) => arr.filter((t) => t.id !== id))
    }, kind === 'error' ? 6000 : 3000)
  }, [])

  return (
    <Ctx.Provider value={{ toast }}>
      {children}
      <div className="fixed bottom-4 right-4 z-50 space-y-2 max-w-sm">
        {items.map((t) => (
          <div
            key={t.id}
            className={
              'px-4 py-2 rounded-lg shadow-lg text-sm border ' +
              (t.kind === 'error'
                ? 'bg-err-soft border-err text-err'
                : t.kind === 'success'
                ? 'bg-ok-soft border-ok text-ok'
                : 'bg-elevated border-subtle text-fg-primary')
            }
          >
            {t.message}
          </div>
        ))}
      </div>
    </Ctx.Provider>
  )
}

export function useToast(): ToastApi {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useToast must be used inside <ToastProvider>')
  return ctx
}

const _noopToast: ToastApi = { toast: () => {} }

/** 可选版：不在 ToastProvider 内（单测 / 独立挂载）时退化为 no-op，不抛。
 *  给可复用的展示组件（LogView）用——它的 toast 只是反馈，不是功能前提。 */
export function useOptionalToast(): ToastApi {
  return useContext(Ctx) ?? _noopToast
}
