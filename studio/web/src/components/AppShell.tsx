import type { ReactNode } from 'react'

export const APP_MAIN_ID = 'app-main'

interface AppShellProps {
  navigation: ReactNode
  topbar: ReactNode
  children: ReactNode
  skipLabel: string
  overlay?: ReactNode
  mainId?: string
}

/**
 * Desktop workspace frame.
 *
 * Owns the viewport, fixed navigation/topbar tracks, and the default document
 * scroll container. Pages may create local scroll owners inside `main`, but
 * should not add another viewport-height shell around this component.
 */
export default function AppShell({
  navigation,
  topbar,
  children,
  skipLabel,
  overlay,
  mainId = APP_MAIN_ID,
}: AppShellProps) {
  return (
    <>
      <div className="ui-app-shell">
        <a className="ui-app-shell-skip" href={`#${mainId}`}>
          {skipLabel}
        </a>
        {navigation}
        <div className="ui-app-shell-workspace">
          {topbar}
          <main id={mainId} className="ui-app-shell-main" tabIndex={-1}>
            {children}
          </main>
        </div>
      </div>
      {overlay}
    </>
  )
}
