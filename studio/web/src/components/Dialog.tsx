// Dialog.tsx —— 命令式 confirm / prompt / alert,替代浏览器原生 window.* 三件套。
//
// 设计动机:浏览器原生对话框样式跟应用 UI 完全脱节(灰蒙蒙系统弹框),且无法
// 支持自定义按钮文案 / 危险操作语义 / 输入校验。仓库里散落 20+ 处 confirm/
// prompt/alert,这里集中成一个 Provider + hook 的命令式 API,call site 改动
// 最小(`if (!confirm(...))` → `if (!await confirm(...))`)。
//
// API 形态:
//   const { confirm, prompt, alert } = useDialog()
//   const ok    = await confirm('删除版本 v1？', { tone: 'danger' })   // Promise<boolean>
//   const name  = await prompt('新预设名称', { defaultValue, validate }) // Promise<string|null>
//   await alert('JSON 解析失败', { tone: 'error' })
//
// 不替代:已有的复杂表单对话框(NewVersionDialog 之类),那些有自定义字段/
// 嵌入 SchemaForm,声明式 JSX 写更清楚,继续保留。
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { useTranslation } from 'react-i18next'
import ActionGroup from './ActionGroup'
import Button from './Button'
import { Input } from './FormControl'
import Modal from './Modal'

export type DialogTone = 'default' | 'danger' | 'warn'

export interface ConfirmOptions {
  /** 标记确认的紧迫语义；按钮仍遵循全局单一 primary 层级，不按 tone 改色。 */
  tone?: DialogTone
  okText?: string
  cancelText?: string
  /** 对话框标题(默认"确认操作") */
  title?: string
}

export interface PromptOptions {
  defaultValue?: string
  placeholder?: string
  /** 同步校验:返回 null = 通过,返回 string = 错误信息。每次输入立刻跑。 */
  validate?: (v: string) => string | null
  okText?: string
  cancelText?: string
  title?: string
  /** 跟 ConfirmOptions 对齐 — 用于「输入后会做危险操作」的场景。 */
  tone?: DialogTone
}

export interface AlertOptions {
  tone?: DialogTone
  okText?: string
  title?: string
}

type DialogState =
  | {
      type: 'confirm'
      message: string
      options: ConfirmOptions
      resolve: (ok: boolean) => void
    }
  | {
      type: 'prompt'
      label: string
      options: PromptOptions
      resolve: (v: string | null) => void
    }
  | {
      type: 'alert'
      message: string
      options: AlertOptions
      resolve: () => void
    }

interface DialogApi {
  confirm: (message: string, options?: ConfirmOptions) => Promise<boolean>
  prompt: (label: string, options?: PromptOptions) => Promise<string | null>
  alert: (message: string, options?: AlertOptions) => Promise<void>
}

const Ctx = createContext<DialogApi | null>(null)

export function DialogProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<DialogState | null>(null)

  const confirm = useCallback(
    (message: string, options: ConfirmOptions = {}): Promise<boolean> =>
      new Promise((resolve) => {
        setState({ type: 'confirm', message, options, resolve })
      }),
    [],
  )

  const prompt = useCallback(
    (label: string, options: PromptOptions = {}): Promise<string | null> =>
      new Promise((resolve) => {
        setState({ type: 'prompt', label, options, resolve })
      }),
    [],
  )

  const alert = useCallback(
    (message: string, options: AlertOptions = {}): Promise<void> =>
      new Promise((resolve) => {
        setState({ type: 'alert', message, options, resolve })
      }),
    [],
  )

  // 取消通用入口(ESC / 点遮罩 / 取消按钮):confirm → false, prompt → null,
  // alert → 直接 resolve。
  const cancel = useCallback(() => {
    setState((cur) => {
      if (!cur) return null
      if (cur.type === 'confirm') cur.resolve(false)
      else if (cur.type === 'prompt') cur.resolve(null)
      else cur.resolve()
      return null
    })
  }, [])

  const confirmOk = useCallback((value: boolean | string) => {
    setState((cur) => {
      if (!cur) return null
      if (cur.type === 'confirm') cur.resolve(value as boolean)
      else if (cur.type === 'prompt') cur.resolve(value as string)
      else cur.resolve()
      return null
    })
  }, [])

  return (
    <Ctx.Provider value={{ confirm, prompt, alert }}>
      {children}
      {state && (
        <DialogRoot state={state} onCancel={cancel} onOk={confirmOk} />
      )}
    </Ctx.Provider>
  )
}

export function useDialog(): DialogApi {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useDialog must be used inside <DialogProvider>')
  return ctx
}

// ────────────────────────────────────────────────────────────────────────────

interface RootProps {
  state: DialogState
  onCancel: () => void
  onOk: (value: boolean | string) => void
}

function DialogRoot({ state, onCancel, onOk }: RootProps) {
  const { t } = useTranslation()
  // input value 用 ref 而非 state,避免每次按键触发 DialogRoot rerender。
  // 错误信息走 state,因为要触发 re-render。
  const [inputValue, setInputValue] = useState(
    state.type === 'prompt' ? (state.options.defaultValue ?? '') : '',
  )
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const errorId = useId()

  // Prompt 自动 focus + select 默认值,方便用户改名
  useEffect(() => {
    if (state.type === 'prompt') {
      requestAnimationFrame(() => {
        inputRef.current?.focus()
        inputRef.current?.select()
      })
    }
  }, [state.type])

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault()
    if (state.type === 'confirm') {
      onOk(true)
    } else if (state.type === 'prompt') {
      const v = inputValue
      const err = state.options.validate?.(v) ?? null
      if (err) {
        setError(err)
        inputRef.current?.focus()
        return
      }
      onOk(v)
    } else {
      onOk(true)
    }
  }

  const title =
    state.options.title ??
    (state.type === 'confirm'
      ? t('common.dialogConfirmTitle')
      : state.type === 'prompt'
        ? t('common.dialogPromptTitle')
        : t('common.dialogAlertTitle'))

  return (
    <Modal
      as="form"
      title={title}
      description={state.type === 'prompt' ? undefined : state.message}
      onClose={onCancel}
      onSubmit={handleSubmit}
      size="sm"
      role={state.options.tone && state.options.tone !== 'default' ? 'alertdialog' : 'dialog'}
      initialFocusRef={state.type === 'prompt' ? inputRef : undefined}
      footer={(
        <ActionGroup
          secondary={state.type !== 'alert' ? (
            <Button
              type="button"
              variant="secondary"
              onClick={onCancel}
              className="min-w-[96px] justify-center"
            >
              {state.options.cancelText ?? t('common.cancel')}
            </Button>
          ) : undefined}
          primary={(
            <Button
              type="submit"
              variant="primary"
              className="min-w-[96px] justify-center"
            >
              {state.options.okText ??
                (state.type === 'confirm'
                  ? t('common.confirm')
                  : state.type === 'prompt'
                    ? t('common.ok')
                    : t('common.gotIt'))}
            </Button>
          )}
        />
      )}
    >
      {state.type === 'prompt' ? (
        <label className="flex flex-col gap-1.5">
          <span className="type-field-label">{state.label}</span>
          <Input
            ref={inputRef}
            mono
            value={inputValue}
            placeholder={state.options.placeholder}
            onChange={(e) => {
              setInputValue(e.target.value)
              if (error) setError(null)
            }}
            invalid={Boolean(error)}
            aria-describedby={error ? errorId : undefined}
          />
          {error && (
            <span id={errorId} className="text-xs text-err" aria-live="polite">
              {error}
            </span>
          )}
        </label>
      ) : undefined}
    </Modal>
  )
}
