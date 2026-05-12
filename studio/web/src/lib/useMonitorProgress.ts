/**
 * useMonitorProgress — 共享的 monitor state 订阅 hook（PR #37 增量协议）。
 *
 * 协议：
 *   - mount + SSE 重连：GET /api/state?task_id=X&max_points=1000 拉降采样快照
 *   - 之后 SSE monitor_progress 推 delta，本 hook 把 appended_losses/lr/samples
 *     合并进 state；scalar 字段（step/speed/...）每次替换
 *   - dedup：appended_losses/lr 按 step 过滤已知；samples 按 (step, path) 过滤
 *     —— 防止重连时 snapshot 与 poller delta 边界重叠造成的重复点
 *   - cap：losses/lr_history 各 5000 上限；samples 50 上限（同 backend cap）
 *
 * taskId 为 null 时 hook 完全 idle（用于 Queue/Topbar 跨 task 视图，无运行
 * 任务时不订阅）。taskId 切换时清状态 + 重新拉快照。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type MonitorState } from '../api/client'
import { useEventStream } from './useEventStream'

interface MonitorProgressDelta {
  step?: number
  total_steps?: number
  epoch?: number
  total_epochs?: number
  speed?: number
  start_time?: number | null
  appended_losses?: Array<{ step: number; loss: number; time?: number }>
  appended_lr?: Array<{ step: number; lr: number }>
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
  const samples = base.samples ?? []
  const lastLossStep = losses.length ? losses[losses.length - 1].step : -1
  const lastLrStep = lrHistory.length ? lrHistory[lrHistory.length - 1].step : -1
  const knownSamples = new Set(samples.map((s) => `${s.step ?? ''}|${s.path}`))

  const newLosses = (delta.appended_losses ?? []).filter((l) => l.step > lastLossStep)
  const newLr = (delta.appended_lr ?? []).filter((l) => l.step > lastLrStep)
  const newSamples = (delta.appended_samples ?? []).filter(
    (s) => !knownSamples.has(`${s.step ?? ''}|${s.path}`),
  )

  const mergedLosses = newLosses.length ? [...losses, ...newLosses] : losses
  const mergedLr = newLr.length ? [...lrHistory, ...newLr] : lrHistory
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
    samples: mergedSamples.length > MAX_SAMPLES ? mergedSamples.slice(-MAX_SAMPLES) : mergedSamples,
    config: delta.config ?? base.config,
  }
}

export interface MonitorProgress {
  state: MonitorState | null
  connected: boolean
  /** 主动重新拉一次快照（消费者一般不用，hook 内部自动调）。 */
  refetch: () => Promise<void>
}

export function useMonitorProgress(taskId: number | null): MonitorProgress {
  const [state, setState] = useState<MonitorState | null>(null)
  const [connected, setConnected] = useState(false)
  const lastUpdateRef = useRef(0)
  // 用 ref 接 taskId 给事件 handler 闭包用，避免每次 taskId 变化重订阅 SSE
  const taskIdRef = useRef(taskId)
  taskIdRef.current = taskId

  const refetch = useCallback(async () => {
    const tid = taskIdRef.current
    if (tid == null) return
    try {
      const s = await api.getMonitorState(tid)
      setState(s)
      setConnected(true)
      lastUpdateRef.current = Date.now()
    } catch {
      if (Date.now() - lastUpdateRef.current > 5000) setConnected(false)
    }
  }, [])

  // taskId 切换 → 清 state + 重新拉
  useEffect(() => {
    setState(null)
    if (taskId == null) {
      setConnected(false)
      return
    }
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
      setState((prev) => mergeDelta(prev, delta))
      setConnected(true)
      lastUpdateRef.current = Date.now()
    },
    { onOpen: () => void refetch() },
  )

  return { state, connected, refetch }
}

// 暴露给测试用
export { mergeDelta as _mergeDeltaForTest }
