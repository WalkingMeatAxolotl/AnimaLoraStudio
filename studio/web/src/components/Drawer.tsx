import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type AnimationEvent,
  type ReactNode,
  type RefObject,
} from 'react'
import { createPortal } from 'react-dom'

export type DrawerSize = 'md' | 'lg' | 'page'
export type DrawerPhase = 'closed' | 'opening' | 'open' | 'closing'

export interface DrawerProps {
  open: boolean
  title: ReactNode
  onClose: () => void
  children?: ReactNode
  size?: DrawerSize
  showTitle?: boolean
  closeOnBackdrop?: boolean
  closeOnEscape?: boolean
  initialFocusRef?: RefObject<HTMLElement | null>
  onEntered?: () => void
  panelClassName?: string
  testId?: string
}

const MOTION_FALLBACK_MS: Record<'opening' | 'closing', number> = {
  opening: 320,
  closing: 240,
}

const SIZE_CLASS: Record<DrawerSize, string> = {
  md: 'ui-drawer-size-md',
  lg: 'ui-drawer-size-lg',
  page: 'ui-drawer-size-page',
}

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',')

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' &&
    Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches)
}

function getFocusable(panel: HTMLElement): HTMLElement[] {
  return Array.from(panel.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR))
    .filter((element) => (
      element.tabIndex >= 0 &&
      !element.hidden &&
      !element.closest('[hidden], [aria-hidden="true"]')
    ))
}

function isTopmostDialog(panel: HTMLElement): boolean {
  const dialogs = Array.from(
    document.querySelectorAll<HTMLElement>('[role="dialog"], [role="alertdialog"]'),
  ).filter((element) => (
    !element.hidden &&
    !element.closest('[hidden], [aria-hidden="true"]')
  ))
  return dialogs[dialogs.length - 1] === panel
}

export default function Drawer({
  open,
  title,
  onClose,
  children,
  size = 'md',
  showTitle = true,
  closeOnBackdrop = true,
  closeOnEscape = true,
  initialFocusRef,
  onEntered,
  panelClassName = '',
  testId,
}: DrawerProps) {
  const titleId = useId()
  const panelRef = useRef<HTMLElement | null>(null)
  const previousFocusRef = useRef<HTMLElement | null>(null)
  const enteredRef = useRef(false)
  const openRef = useRef(open)
  openRef.current = open
  const [phase, setPhase] = useState<DrawerPhase>(() => {
    if (!open) return 'closed'
    return prefersReducedMotion() ? 'open' : 'opening'
  })
  const layerActive = phase !== 'closed'

  useEffect(() => {
    setPhase((current) => {
      if (open) {
        if (current === 'open' || current === 'opening') return current
        return prefersReducedMotion() ? 'open' : 'opening'
      }
      if (current === 'closed' || current === 'closing') return current
      return prefersReducedMotion() ? 'closed' : 'closing'
    })
  }, [open])

  const settlePhase = useCallback((expected: 'opening' | 'closing') => {
    setPhase((current) => {
      if (current !== expected) return current
      if (current === 'opening') return openRef.current ? 'open' : 'closing'
      return openRef.current ? 'opening' : 'closed'
    })
  }, [])

  useEffect(() => {
    if (phase !== 'opening' && phase !== 'closing') return
    const timer = window.setTimeout(() => settlePhase(phase), MOTION_FALLBACK_MS[phase])
    return () => window.clearTimeout(timer)
  }, [phase, settlePhase])

  useEffect(() => {
    if (!open) {
      enteredRef.current = false
      return
    }
    if (phase === 'open' && !enteredRef.current) {
      enteredRef.current = true
      onEntered?.()
    }
  }, [onEntered, open, phase])

  useEffect(() => {
    if (!open) return

    if (!previousFocusRef.current) {
      previousFocusRef.current = document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null
    }

    const frame = requestAnimationFrame(() => {
      const panel = panelRef.current
      if (!panel || panel.contains(document.activeElement)) return
      const target = initialFocusRef?.current ?? panel
      target.focus()
    })

    return () => cancelAnimationFrame(frame)
  }, [initialFocusRef, open])

  useEffect(() => {
    if (!layerActive) {
      previousFocusRef.current?.focus()
      previousFocusRef.current = null
      return
    }

    const appRoot = document.getElementById('root')
    const appWasInert = appRoot?.inert ?? false
    if (appRoot) appRoot.inert = true
    return () => {
      if (appRoot) appRoot.inert = appWasInert
    }
  }, [layerActive])

  useEffect(() => () => {
    previousFocusRef.current?.focus()
    previousFocusRef.current = null
  }, [])

  useEffect(() => {
    if (!open) return

    const onKeyDown = (event: KeyboardEvent) => {
      const panel = panelRef.current
      if (!panel || !isTopmostDialog(panel)) return

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
      const activeElement = document.activeElement
      if (event.shiftKey && (activeElement === first || !panel.contains(activeElement))) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && (activeElement === last || !panel.contains(activeElement))) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [closeOnEscape, onClose, open])

  const finishMotion = (event: AnimationEvent<HTMLElement>) => {
    if (event.target !== event.currentTarget) return
    if (phase === 'opening' && event.animationName === 'drawer-panel-in') {
      settlePhase('opening')
    } else if (phase === 'closing' && event.animationName === 'drawer-panel-out') {
      settlePhase('closing')
    }
  }

  return createPortal(
    <div
      className="ui-drawer-root"
      data-testid={testId}
      data-state={phase}
      aria-hidden={phase === 'closed' || undefined}
    >
      <div
        className="ui-drawer-backdrop"
        onMouseDown={(event) => {
          if (closeOnBackdrop && event.target === event.currentTarget) onClose()
        }}
      />
      <aside
        ref={(node) => { panelRef.current = node }}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-hidden={phase === 'closed' || undefined}
        tabIndex={-1}
        className={[
          'ui-drawer-panel',
          SIZE_CLASS[size],
          panelClassName,
        ].filter(Boolean).join(' ')}
        onAnimationEnd={finishMotion}
      >
        <h2
          id={titleId}
          className={showTitle ? 'type-section-title shrink-0 px-page pt-page' : 'sr-only'}
        >
          {title}
        </h2>
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
          {children}
        </div>
      </aside>
    </div>,
    document.body,
  )
}
