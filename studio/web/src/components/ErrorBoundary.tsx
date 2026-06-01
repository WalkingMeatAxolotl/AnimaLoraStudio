import { Component, type ErrorInfo, type ReactNode } from 'react'
import i18n from '../i18n'
import { getLastApiTraceId, reportClientError } from '../lib/errors/report'

interface State { error: Error | null }

export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary]', error, info)
    // ADR-0009 PR-3 C3: 上报到 /api/client-errors → studio.log studio.client logger。
    // silent swallow on fail（防 ErrorBoundary 已经在 catch state 时再炸第二次）。
    try {
      reportClientError({
        kind: 'react.boundary',
        message: error.message,
        stack: error.stack,
        componentStack: info.componentStack ?? undefined,
      })
    } catch {
      // 上报本身炸了也吞
    }
  }

  render() {
    if (this.state.error) {
      // ADR-0009 PR-3 C3: 显示最近一次 API 4xx/5xx 的 trace_id 末 8 字符
      // （ErrorBoundary 自己拿不到 traceId — 用前端 lastApiTraceId 兜底）。
      // 用户截图给开发：grep 这串能定位整条 trace 链路。
      const traceId = getLastApiTraceId()
      const traceSuffix = traceId ? traceId.slice(-8) : null
      return (
        <div className="min-h-screen flex items-center justify-center p-8 bg-canvas">
          <div className="card max-w-[560px] w-full p-6">
            <h1 className="text-err font-semibold text-lg mb-2">{i18n.t('errorBoundary.title')}</h1>
            <pre className="text-sm text-fg-secondary whitespace-pre-wrap break-all">
              {this.state.error.message}
            </pre>
            {traceSuffix && (
              <div className="text-xs text-fg-tertiary mt-2 font-mono">
                trace {traceSuffix}
              </div>
            )}
            <button
              className="btn btn-primary btn-sm mt-4"
              onClick={() => window.location.reload()}
            >
              {i18n.t('errorBoundary.reload')}
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
