/**
 * useMonitorProgress — 共享的 monitor state 订阅 hook（PR #37 增量协议）。
 *
 * 协议：
 *   - mount + SSE 重连：GET /api/state?task_id=X 拉全量快照
 *   - 之后 SSE monitor_progress 推 delta，本 hook 把 appended_losses/lr/samples
 *     合并进 state；scalar 字段（step/speed/...）每次替换
 *   - dedup：appended_losses/lr 按 step 过滤已知；samples 按 (step, path) 过滤
 *     —— 防止重连时 snapshot 与 poller delta 边界重叠造成的重复点
 *   - cap：losses/lr_history 各 50000 上限；samples 50 上限（同 backend cap）
 *
 * taskId 为 null 时 hook 完全 idle（用于 Queue/Topbar 跨 task 视图，无运行
 * 任务时不订阅）。taskId 切换时清状态 + 重新拉快照。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type ApiError, type MonitorState } from '../api/client'
import { isEventStreamOpen, useEventStream } from './useEventStream'

interface MonitorProgressDelta {
  step?: number
  total_steps?: number
  epoch?: number
  total_epochs?: number
  speed?: number
  start_time?: number | null
  appended_losses?: Array<{ step: number; loss: number; time?: number }>
  appended_lr?: Array<{ step: number; lr: number }>
  appended_optimizer_metrics?: NonNullable<MonitorState['optimizer_metrics_history']>
  appended_samples?: NonNullable<MonitorState['samples']>
  config?: Record<string, string | number | boolean>
}

// 与 backend train_monitor.update_monitor 的内置裁尾上限对齐 (runtime/train_monitor.py:108-116)。
// 早期 5000 上限是因为 server 默认降采样 1500；改成全量 snapshot 后 cold-start
// 已经可能 ≥10k 点，5000 会立刻 slice 掉早期。50000 跟 backend 一致，cap 由 disk
// 端兜底；前端 chart 内部 downsample(600) 渲染，不影响 perf。
const MAX_LOSSES = 50000
const MAX_LR = 50000
const MAX_SAMPLES = 50

function mergeDelta(prev: MonitorState | null, delta: MonitorProgressDelta): MonitorState {
  const base: MonitorState = prev ?? {}

  // dedup cursors — 由 prev 末尾推断，避免重连时 snapshot 与下条 delta 重叠
  const losses = base.losses ?? []
  const lrHistory = base.lr_history ?? []
  const optimizerMetricsHistory = base.optimizer_metrics_history ?? []
  const samples = base.samples ?? []
  const lastLossStep = losses.length ? losses[losses.length - 1].step : -1
  const lastLrStep = lrHistory.length ? lrHistory[lrHistory.length - 1].step : -1
  const lastOptimizerMetricsStep = optimizerMetricsHistory.length
    ? optimizerMetricsHistory[optimizerMetricsHistory.length - 1].step
    : -1
  const knownSamples = new Set(samples.map((s) => `${s.step ?? ''}|${s.path}`))

  const newLosses = (delta.appended_losses ?? []).filter((l) => l.step > lastLossStep)
  const newLr = (delta.appended_lr ?? []).filter((l) => l.step > lastLrStep)
  const newOptimizerMetrics = (delta.appended_optimizer_metrics ?? [])
    .filter((l) => l.step > lastOptimizerMetricsStep)
  const newSamples = (delta.appended_samples ?? []).filter(
    (s) => !knownSamples.has(`${s.step ?? ''}|${s.path}`),
  )

  const mergedLosses = newLosses.length ? [...losses, ...newLosses] : losses
  const mergedLr = newLr.length ? [...lrHistory, ...newLr] : lrHistory
  const mergedOptimizerMetrics = newOptimizerMetrics.length
    ? [...optimizerMetricsHistory, ...newOptimizerMetrics]
    : optimizerMetricsHistory
  const mergedSamples = newSamples.length ? [...samples, ...newSamples] : samples

  return {
    ...base,
    step: delta.step ?? base.step,
    total_steps: delta.total_steps ?? base.total_steps,
    epoch: delta.epoch ?? base.epoch,
    total_epochs: delta.total_epochs ?? base.total_epochs,
    speed: delta.speed ?? base.speed,
    start_time: delta.start_time ?? base.start_time,
    losses: mergedLosses.length > MAX_LOSSES ? mergedLosses.slice(-MAX_LOSSES) : mergedLosses,
    lr_history: mergedLr.length > MAX_LR ? mergedLr.slice(-MAX_LR) : mergedLr,
    optimizer_metrics_history: mergedOptimizerMetrics.length > MAX_LR
      ? mergedOptimizerMetrics.slice(-MAX_LR)
      : mergedOptimizerMetrics,
    samples: mergedSamples.length > MAX_SAMPLES ? mergedSamples.slice(-MAX_SAMPLES) : mergedSamples,
    config: delta.config ?? base.config,
  }
}

export type MonitorLoadStatus = 'idle' | 'loading' | 'ready' | 'unavailable' | 'error'
export type MonitorStreamStatus = 'idle' | 'connecting' | 'live' | 'reconnecting'

export interface MonitorProgress {
  state: MonitorState | null
  /** 快照本身的读取状态；已有数据刷新失败时保持 ready，并通过 error 报告。 */
  status: MonitorLoadStatus
  /** SSE transport 的真实状态；HTTP 快照成功不会把它改成 live。 */
  streamStatus: MonitorStreamStatus
  /** 当前任务最近一次快照/增量的前端接收时间（毫秒 epoch）。 */
  lastUpdatedAt: number | null
  /** 初次/手动/重连补拉正在进行。 */
  refreshing: boolean
  /** 最近一次快照读取错误；新数据到达后清除。 */
  error: string | null
  /** 兼容现有消费者；只表示 SSE 当前 open。 */
  connected: boolean
  /** 主动重新拉一次快照。失败被反映在 status/error，不向调用方抛出。 */
  refetch: () => Promise<void>
}

export function useMonitorProgress(taskId: number | null): MonitorProgress {
  const [state, setState] = useState<MonitorState | null>(null)
  const [status, setStatus] = useState<MonitorLoadStatus>(taskId == null ? 'idle' : 'loading')
  const [streamStatus, setStreamStatus] = useState<MonitorStreamStatus>(taskId == null ? 'idle' : 'connecting')
  const [lastUpdatedAt, setLastUpdatedAt] = useState<number | null>(null)
  const [refreshing, setRefreshing] = useState(taskId != null)
  const [error, setError] = useState<string | null>(null)
  const stateRef = useRef<MonitorState | null>(null)
  const requestGenerationRef = useRef(0)
  // 用 ref 接 taskId 给事件 handler 闭包用，避免每次 taskId 变化重订阅 SSE
  const taskIdRef = useRef(taskId)
  taskIdRef.current = taskId

  const refetch = useCallback(async () => {
    const tid = taskIdRef.current
    if (tid == null) return
    const generation = ++requestGenerationRef.current
    setRefreshing(true)
    if (stateRef.current === null) setStatus('loading')
    try {
      const snapshot = await api.getMonitorState(tid)
      if (taskIdRef.current !== tid || requestGenerationRef.current !== generation) return
      stateRef.current = snapshot
      setState(snapshot)
      setStatus('ready')
      setError(null)
      setLastUpdatedAt(Date.now())
    } catch (caught) {
      if (taskIdRef.current !== tid || requestGenerationRef.current !== generation) return
      const apiError = caught as ApiError
      setError(String(caught))
      // 已显示的证据不因一次失败消失；404 只在尚无快照时表示监控产物不可用。
      if (stateRef.current === null) {
        setStatus(apiError.status === 404 ? 'unavailable' : 'error')
      }
    } finally {
      if (taskIdRef.current === tid && requestGenerationRef.current === generation) {
        setRefreshing(false)
      }
    }
  }, [])

  // taskId 切换 → 清 state + 重新拉
  useEffect(() => {
    requestGenerationRef.current += 1
    stateRef.current = null
    setState(null)
    setLastUpdatedAt(null)
    setError(null)
    if (taskId == null) {
      setStatus('idle')
      setStreamStatus('idle')
      setRefreshing(false)
      return
    }
    setStatus('loading')
    setStreamStatus(isEventStreamOpen() ? 'live' : 'connecting')
    void refetch()
  }, [taskId, refetch])

  useEventStream(
    (evt) => {
      const tid = taskIdRef.current
      if (tid == null) return
      if (evt.type !== 'monitor_progress') return
      if (String(evt.task_id) !== String(tid)) return
      const delta = evt.delta as MonitorProgressDelta | undefined
      if (!delta) return
      setState((prev) => {
        const next = mergeDelta(prev, delta)
        stateRef.current = next
        return next
      })
      setStatus('ready')
      setStreamStatus('live')
      setError(null)
      setLastUpdatedAt(Date.now())
    },
    {
      onOpen: () => {
        if (taskIdRef.current == null) return
        setStreamStatus('live')
        void refetch()
      },
      onError: () => {
        if (taskIdRef.current != null) setStreamStatus('reconnecting')
      },
    },
  )

  return {
    state,
    status,
    streamStatus,
    lastUpdatedAt,
    refreshing,
    error,
    connected: streamStatus === 'live',
    refetch,
  }
}

// 暴露给测试用
export { mergeDelta as _mergeDeltaForTest }
