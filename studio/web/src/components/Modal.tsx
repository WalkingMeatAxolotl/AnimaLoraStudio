import {
  useEffect,
  useId,
  useRef,
  type FormEventHandler,
  type ReactNode,
  type RefObject,
} from 'react'
import { createPortal } from 'react-dom'

export type ModalSize = 'sm' | 'md' | 'lg' | 'wide'

interface ModalBaseProps {
  title: ReactNode
  description?: ReactNode
  children?: ReactNode
  footer?: ReactNode
  headerActions?: ReactNode
  onClose: () => void
  size?: ModalSize
  role?: 'dialog' | 'alertdialog'
  closeOnBackdrop?: boolean
  closeOnEscape?: boolean
  initialFocusRef?: RefObject<HTMLElement | null>
  panelClassName?: string
  bodyClassName?: string
  testId?: string
}

export type ModalProps = ModalBaseProps & (
  | {
      as?: 'div'
      onSubmit?: never
    }
  | {
      as: 'form'
      onSubmit?: FormEventHandler<HTMLFormElement>
    }
)

const SIZE_CLASS: Record<ModalSize, string> = {
  sm: 'max-w-[440px]',
  md: 'max-w-[560px]',
  lg: 'max-w-[720px]',
  wide: 'w-[80vw] max-w-[1440px]',
}

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',')

let bodyLockCount = 0
let bodyOverflowBeforeLock = ''

function lockBodyScroll() {
  if (bodyLockCount === 0) {
    bodyOverflowBeforeLock = document.body.style.overflow
    document.body.style.overflow = 'hidden'
  }
  bodyLockCount += 1
}

function unlockBodyScroll() {
  bodyLockCount = Math.max(0, bodyLockCount - 1)
  if (bodyLockCount === 0) document.body.style.overflow = bodyOverflowBeforeLock
}

function getFocusable(panel: HTMLElement): HTMLElement[] {
  return Array.from(panel.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR))
    .filter((element) => (
      element.tabIndex >= 0 &&
      !element.hidden &&
      !element.closest('[hidden], [aria-hidden="true"]')
    ))
}

export default function Modal({
  title,
  description,
  children,
  footer,
  headerActions,
  onClose,
  size = 'md',
  role = 'dialog',
  closeOnBackdrop = true,
  closeOnEscape = true,
  initialFocusRef,
  as = 'div',
  onSubmit,
  panelClassName = '',
  bodyClassName = '',
  testId,
}: ModalProps) {
  const titleId = useId()
  const descriptionId = useId()
  const panelRef = useRef<HTMLElement | null>(null)
  const previousFocusRef = useRef<HTMLElement | null>(null)

  useEffect(() => {
    previousFocusRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null
    lockBodyScroll()

    const frame = requestAnimationFrame(() => {
      const panel = panelRef.current
      if (!panel || panel.contains(document.activeElement)) return
      const target = initialFocusRef?.current ?? getFocusable(panel)[0] ?? panel
      target.focus()
    })

    return () => {
      cancelAnimationFrame(frame)
      unlockBodyScroll()
      previousFocusRef.current?.focus()
    }
  }, [initialFocusRef])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const panel = panelRef.current
      if (!panel) return

      if (event.key === 'Escape' && closeOnEscape) {
        event.preventDefault()
        event.stopPropagation()
        onClose()
        return
      }

      if (event.key !== 'Tab') return
      const focusable = getFocusable(panel)
      if (focusable.length === 0) {
        event.preventDefault()
        panel.focus()
        return
      }

      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      const active = document.activeElement
      if (event.shiftKey && (active === first || !panel.contains(active))) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && (active === last || !panel.contains(active))) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [closeOnEscape, onClose])

  const sharedPanelProps = {
    role,
    'aria-modal': true,
    'aria-labelledby': titleId,
    'aria-describedby': description ? descriptionId : undefined,
    tabIndex: -1,
    className: [
      'flex w-full min-h-0 flex-col overflow-hidden rounded-lg border border-dim bg-elevated shadow-xl',
      SIZE_CLASS[size],
      panelClassName,
    ].filter(Boolean).join(' '),
    style: { maxHeight: 'calc(100dvh - (2 * var(--space-page)))' },
  }

  const content = (
    <>
      <header className="shrink-0 px-page pt-page">
        <div className="flex min-w-0 items-start gap-related">
          <div className="min-w-0 flex-1">
            <h2 id={titleId} className="type-section-title">{title}</h2>
            {description && (
              <p id={descriptionId} className="type-page-description mt-related">
                {description}
              </p>
            )}
          </div>
          {headerActions && (
            <div className="flex shrink-0 items-center gap-related">
              {headerActions}
            </div>
          )}
        </div>
      </header>
      {children && (
        <div className={`min-h-0 overflow-y-auto px-page pt-section ${footer ? 'pb-0' : 'pb-page'} ${bodyClassName}`}>
          {children}
        </div>
      )}
      {footer && (
        <footer className={`shrink-0 px-page pb-page ${children ? 'pt-section' : 'pt-field'}`}>
          {footer}
        </footer>
      )}
    </>
  )

  const panel = as === 'form' ? (
    <form
      {...sharedPanelProps}
      ref={(node) => { panelRef.current = node }}
      onSubmit={onSubmit}
    >
      {content}
    </form>
  ) : (
    <div
      {...sharedPanelProps}
      ref={(node) => { panelRef.current = node }}
    >
      {content}
    </div>
  )

  return createPortal(
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center overflow-y-auto bg-black/50 p-page"
      data-testid={testId}
      onMouseDown={(event) => {
        if (closeOnBackdrop && event.target === event.currentTarget) onClose()
      }}
    >
      {panel}
    </div>,
    document.body,
  )
}
