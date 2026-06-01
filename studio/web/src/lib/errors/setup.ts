/**
 * 全局 JS 错误监听安装（ADR-0009 §5.1 / PR-3 C2）。
 *
 * 在 main.tsx 启动时调一次 installGlobalErrorHandlers()。装两个 listener：
 *   - window.addEventListener('error', ...)             同步脚本 / resource load
 *   - window.addEventListener('unhandledrejection', ...)  未 catch 的 Promise
 *
 * **不**全代理 console.error（开发期 React warnings / devtools msg 太多噪音）。
 * 只接 React 外的未捕获错误；React 内部 ErrorBoundary 单独 hook（PR-3 C3）。
 *
 * 装一次即可 — 重复调 noop（_installed sentinel）。
 */
import { reportClientError } from './report'

let _installed = false

export function installGlobalErrorHandlers(): void {
  if (_installed) return
  _installed = true

  if (typeof globalThis === 'undefined' || typeof globalThis.addEventListener !== 'function') {
    return
  }

  globalThis.addEventListener('error', (ev: ErrorEvent) => {
    // ev.error 可能是 null（cross-origin script error / resource 404）
    const err = ev.error as Error | null
    reportClientError({
      kind: 'window.error',
      message: err?.message || ev.message || '(unknown error)',
      stack: err?.stack,
      source: ev.filename,
      line: ev.lineno,
      col: ev.colno,
    })
  })

  globalThis.addEventListener('unhandledrejection', (ev: PromiseRejectionEvent) => {
    const reason = ev.reason as unknown
    let message: string
    let stack: string | undefined
    if (reason instanceof Error) {
      message = reason.message
      stack = reason.stack
    } else if (typeof reason === 'string') {
      message = reason
    } else {
      try {
        message = JSON.stringify(reason)
      } catch {
        message = String(reason)
      }
    }
    reportClientError({
      kind: 'unhandledrejection',
      message,
      stack,
    })
  })
}

/** 测试钩子 — 清 sentinel 让单测可重复 install。 */
export function _resetInstalledForTests(): void {
  _installed = false
}
