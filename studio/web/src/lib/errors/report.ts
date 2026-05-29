/**
 * 前端错误上报（ADR-0009 §5 / PR-3 C2）。
 *
 * 三个 caller：
 *   - window.addEventListener('error')           同步脚本 / resource load 错
 *   - window.addEventListener('unhandledrejection')  Promise.reject 没 catch
 *   - ErrorBoundary.componentDidCatch            React 渲染 / lifecycle 抛错
 *
 * 全部进 `POST /api/client-errors` → 后端 logger.error → studio.log。
 *
 * 强约束：
 *   - **silent swallow on fail** — 上报失败不能再 throw（防级联到 ErrorBoundary 死循环）
 *   - 用 keepalive：tab 关闭瞬间也尽量发出去
 *   - 不阻塞主流程（async fire-and-forget）
 */

export type ClientErrorKind =
  | 'react.boundary'
  | 'window.error'
  | 'unhandledrejection'
  | 'manual'

export interface ClientErrorReport {
  kind: ClientErrorKind
  message: string
  stack?: string
  componentStack?: string  // react.boundary 专属
  source?: string          // window.error script URL
  line?: number
  col?: number
  /** 调用时**不**用填；report() 自动注入 location.href / userAgent 等。 */
}

interface InternalReportBody extends ClientErrorReport {
  url: string
  user_agent: string
  client_ts: string
  app_version: string
  build_hash?: string
  trace_id_last_4xx?: string
}

// 由 client.ts 在 4xx/5xx 时 set；report() 上报时读出来。让开发者
// 在 server log 能 join "用户 toast 看到的错" 跟 "前端崩前最后一次 API 失败"。
let _lastApiTraceId: string | undefined

export function setLastApiTraceId(traceId: string | undefined): void {
  _lastApiTraceId = traceId
}

export function getLastApiTraceId(): string | undefined {
  return _lastApiTraceId
}

/**
 * 上报。fire-and-forget — 不返 Promise（防 caller await 阻塞）。
 *
 * 失败 silently swallow（log to console.warn 防完全失明）。
 */
export function reportClientError(input: ClientErrorReport): void {
  // 防 listener 注册前的早期错误：globalThis 检查
  if (typeof globalThis === 'undefined' || typeof fetch === 'undefined') return

  let buildHash: string | undefined
  let appVersion = '0.0.0'
  try {
    // Vite 的 build-time 注入；缺失时 silently 兜底
    const env = (import.meta as ImportMeta & { env?: Record<string, string> }).env
    buildHash = env?.VITE_BUILD_HASH
    appVersion = env?.VITE_APP_VERSION || appVersion
  } catch {
    // import.meta 不可用（很罕见），兜底
  }

  const body: InternalReportBody = {
    ...input,
    url: globalThis.location?.href || '',
    user_agent: globalThis.navigator?.userAgent || '',
    client_ts: new Date().toISOString(),
    app_version: appVersion,
    build_hash: buildHash,
    trace_id_last_4xx: _lastApiTraceId,
  }

  // fire-and-forget；keepalive 让 tab 关闭也尽量送出
  try {
    void fetch('/api/client-errors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      keepalive: true,
    }).catch(() => {
      // silently swallow — 上报失败不能再 throw 进 ErrorBoundary
    })
  } catch {
    // fetch 都抛了（极罕见）也吞
    try {
      console.warn('[reportClientError] failed silently')
    } catch {
      // even console.warn 不能用了，放弃
    }
  }
}
