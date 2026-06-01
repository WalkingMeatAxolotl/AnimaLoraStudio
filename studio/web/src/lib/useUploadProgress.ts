/**
 * useUploadProgress — 浏览器上传进度状态机。
 *
 * 协议：
 *   1. caller 在请求前 `start(totalBytes)` 让进度条立即 0%
 *   2. caller 把 `onProgress` 透传给 `api.xxx(..., onProgress)` —— XHR.upload
 *      progress 事件回调（fetch 没有 request body progress）
 *   3. loaded === total 时自动切到 'processing'（server 还在解 zip / 落盘，
 *      没有事件，UI 转菊花区分「上传完了在等服务端」）
 *   4. 请求 resolve / reject 后 caller 调 `finish()` / `fail(e)` / `reset()`
 *
 * 速度计算用 1s 滑动窗口避免数字跳得太厉害；ETA = (total - loaded) / speed。
 */
import { useCallback, useRef, useState } from 'react'

export type UploadPhase = 'idle' | 'uploading' | 'processing' | 'done' | 'error'

export interface UploadProgressState {
  phase: UploadPhase
  loaded: number
  total: number
  /** bytes/sec，1s 滑动平均；processing/idle/done 时为 0 */
  speedBps: number
  /** 秒；null = 无法计算（speed=0 或 total=0） */
  etaSec: number | null
  error: string | null
}

const INITIAL: UploadProgressState = {
  phase: 'idle',
  loaded: 0,
  total: 0,
  speedBps: 0,
  etaSec: null,
  error: null,
}

export interface UseUploadProgress {
  state: UploadProgressState
  /** 请求前调，让 UI 立即显示 0% / total */
  start: (totalBytes: number) => void
  /** 透传给 api.xxx 作为 onProgress 回调 */
  onProgress: (e: { loaded: number; total: number; lengthComputable: boolean }) => void
  finish: () => void
  fail: (e: unknown) => void
  /** 重置回 idle，用于关闭对话框 / 隐藏面板 */
  reset: () => void
}

interface Sample {
  t: number
  loaded: number
}

const SPEED_WINDOW_MS = 1500

export function useUploadProgress(): UseUploadProgress {
  const [state, setState] = useState<UploadProgressState>(INITIAL)
  const samplesRef = useRef<Sample[]>([])

  const start = useCallback((totalBytes: number) => {
    samplesRef.current = [{ t: performance.now(), loaded: 0 }]
    setState({
      phase: 'uploading',
      loaded: 0,
      total: totalBytes,
      speedBps: 0,
      etaSec: null,
      error: null,
    })
  }, [])

  const onProgress = useCallback((e: { loaded: number; total: number; lengthComputable: boolean }) => {
    const now = performance.now()
    const samples = samplesRef.current
    samples.push({ t: now, loaded: e.loaded })
    const cutoff = now - SPEED_WINDOW_MS
    while (samples.length > 1 && samples[0].t < cutoff) samples.shift()
    const first = samples[0]
    const dt = (now - first.t) / 1000
    const speed = dt > 0 ? Math.max(0, (e.loaded - first.loaded) / dt) : 0
    const total = e.lengthComputable && e.total > 0 ? e.total : 0
    const remaining = total > 0 ? Math.max(0, total - e.loaded) : 0
    const eta = speed > 0 && remaining > 0 ? remaining / speed : null
    const isComplete = total > 0 && e.loaded >= total
    setState({
      phase: isComplete ? 'processing' : 'uploading',
      loaded: e.loaded,
      total,
      speedBps: isComplete ? 0 : speed,
      etaSec: isComplete ? null : eta,
      error: null,
    })
  }, [])

  const finish = useCallback(() => {
    samplesRef.current = []
    setState((s) => ({
      ...s,
      phase: 'done',
      speedBps: 0,
      etaSec: null,
      loaded: s.total > 0 ? s.total : s.loaded,
    }))
  }, [])

  const fail = useCallback((e: unknown) => {
    samplesRef.current = []
    setState((s) => ({
      ...s,
      phase: 'error',
      speedBps: 0,
      etaSec: null,
      error: e instanceof Error ? e.message : String(e),
    }))
  }, [])

  const reset = useCallback(() => {
    samplesRef.current = []
    setState(INITIAL)
  }, [])

  return { state, start, onProgress, finish, fail, reset }
}

// ── 格式化 helpers（独立 export，方便单测 / 别处复用） ──────────────────

export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v < 10 && i > 0 ? v.toFixed(1) : Math.round(v)} ${units[i]}`
}

export function formatSpeed(bps: number): string {
  if (!Number.isFinite(bps) || bps <= 0) return '—'
  return `${formatBytes(bps)}/s`
}

export function formatEta(sec: number | null): string {
  if (sec == null || !Number.isFinite(sec) || sec < 0) return '—'
  if (sec < 1) return '<1s'
  if (sec < 60) return `${Math.ceil(sec)}s`
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  if (m < 60) return `${m}m${String(s).padStart(2, '0')}s`
  const h = Math.floor(m / 60)
  const mm = m % 60
  return `${h}h${String(mm).padStart(2, '0')}m`
}
