import { useEffect, useRef, useState } from 'react'
import { api, type EvalMetricResult, type Task } from '../api/client'
import { useEventStream } from './useEventStream'

// 训练结束后评估（after-training eval）的可见性：训练进程一退出 task 就标 done，
// 评估作为独立 jobs（每个 checkpoint：eval_samples→eval_clip→eval_dino）在后面跑，
// 不在 queue 列表里，用户没感知。这里靠现有 listEvalMetrics 的逐 checkpoint 状态
// 拼出「评估中 done/total」，配合 eval_auto_after_training_queued 事件即时开始观察、
// 刷新时也能从仍在跑的 metric 状态接上。纯前端，不依赖新后端事件。

const ACTIVE_STATUS = new Set(['pending', 'running'])
const METRIC_KEYS = ['clip_t', 'clip_i', 'dino_i'] as const
const POLL_MS = 4000

export interface EvalProgress {
  /** 还有 checkpoint 在出图 / 算指标 */
  active: boolean
  /** 已评完的 checkpoint 数 */
  done: number
  /** 本次评估的 checkpoint 总数 */
  total: number
}

/** 从 listEvalMetrics 的 results 聚合出「评估中 done/total」。
 *  一个 checkpoint「在评估」= run 级 status 或任一指标 status 为 pending/running
 *  （run pending/running 覆盖出图阶段，指标 status 覆盖算指标阶段）。 */
export function evalProgressFromResults(results: EvalMetricResult[]): EvalProgress {
  let active = false
  let done = 0
  for (const r of results) {
    const runActive = ACTIVE_STATUS.has(r.status)
    const metricActive = METRIC_KEYS.some((k) => {
      const s = r.metric_states?.[k]?.status
      return s != null && ACTIVE_STATUS.has(s)
    })
    if (runActive || metricActive) active = true
    else done += 1
  }
  return { active, done, total: results.length }
}

/** 单个 task 的评估进度（QueueDetail 头部用）。`enabled` 时首拉一次（刷新兜底），
 *  评估期间 4s 轮询，评估结束后自停（保留最终 progress，active=false 调用方不再
 *  显示徽标）；无评估则首拉后不再轮询。after-training 入队事件即时触发开始观察。 */
export function useTaskEvalProgress(
  pid: number | null | undefined,
  vid: number | null | undefined,
  taskId: number | null | undefined,
  enabled: boolean,
): EvalProgress | null {
  const [progress, setProgress] = useState<EvalProgress | null>(null)
  const timer = useRef<number | null>(null)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    const clear = () => {
      if (timer.current != null) { window.clearTimeout(timer.current); timer.current = null }
    }
    const poll = async () => {
      if (!enabled || !pid || !vid || !taskId) return
      let p: EvalProgress | null = null
      try {
        const r = await api.listEvalMetrics(pid, vid, taskId)
        p = evalProgressFromResults(r.results ?? [])
      } catch {
        // 评估进度是辅助信息，拉失败不打扰；下一拍再试（若仍在 watch）
      }
      if (!mounted.current) return
      if (p) setProgress(p)
      clear()
      if (p?.active) timer.current = window.setTimeout(() => void poll(), POLL_MS)
    }
    if (enabled) void poll()
    else setProgress(null)
    return () => { mounted.current = false; clear() }
  }, [enabled, pid, vid, taskId])

  // after-training 评估入队 → 即时开始观察（首拉可能早于入队而错过）
  useEventStream((evt) => {
    if (evt.type !== 'eval_auto_after_training_queued') return
    if (evt.task_id !== taskId || !enabled || !pid || !vid || !taskId) return
    void (async () => {
      try {
        const r = await api.listEvalMetrics(pid, vid, taskId)
        if (mounted.current) setProgress(evalProgressFromResults(r.results ?? []))
      } catch { /* ignore */ }
    })()
  })

  return progress
}

/** Queue 列表用：多 task 的评估进度 Map。观察集合靠 eval_auto_after_training_queued
 *  事件（即时）+ tasks 变化时对「最近完成的 done task」做一次兜底探测（刷新时正在
 *  评估也能接上）。只轮询观察集合里的 task（通常 0~1 个），评估结束即移出。 */
export function useEvaluatingTasks(tasks: Task[]): Map<number, EvalProgress> {
  const [progress, setProgress] = useState<Map<number, EvalProgress>>(new Map())
  const watch = useRef<Set<number>>(new Set())
  const meta = useRef<Map<number, { pid: number; vid: number }>>(new Map())

  // id → pid/vid（评估 task 都是绑定 project/version 的训练 task）
  useEffect(() => {
    const m = new Map<number, { pid: number; vid: number }>()
    for (const t of tasks) {
      if (t.project_id != null && t.version_id != null) {
        m.set(t.id, { pid: t.project_id, vid: t.version_id })
      }
    }
    meta.current = m
  }, [tasks])

  useEventStream((evt) => {
    if (evt.type === 'eval_auto_after_training_queued' && typeof evt.task_id === 'number') {
      watch.current.add(evt.task_id)
    }
  })

  // 兜底：最近完成的 done task 探一下是否仍在评估（覆盖刷新时漏掉事件）
  useEffect(() => {
    const done = tasks.filter(
      (t) => t.status === 'done' && t.project_id != null && t.version_id != null,
    )
    if (done.length === 0) return
    const latest = done.reduce((a, b) => ((b.finished_at ?? 0) > (a.finished_at ?? 0) ? b : a))
    let cancelled = false
    void (async () => {
      try {
        const r = await api.listEvalMetrics(latest.project_id!, latest.version_id!, latest.id)
        if (cancelled) return
        const p = evalProgressFromResults(r.results ?? [])
        if (p.active) {
          watch.current.add(latest.id)
          setProgress((prev) => new Map(prev).set(latest.id, p))
        }
      } catch { /* ignore */ }
    })()
    return () => { cancelled = true }
  }, [tasks])

  useEffect(() => {
    let alive = true
    const poll = async () => {
      const ids = Array.from(watch.current)
      if (ids.length === 0) return
      await Promise.all(ids.map(async (id) => {
        const m = meta.current.get(id)
        if (!m) { watch.current.delete(id); return }
        try {
          const r = await api.listEvalMetrics(m.pid, m.vid, id)
          const p = evalProgressFromResults(r.results ?? [])
          if (!alive) return
          setProgress((prev) => new Map(prev).set(id, p))
          if (!p.active && p.total > 0) watch.current.delete(id)  // 评估结束，停观察
        } catch { /* ignore */ }
      }))
    }
    const interval = window.setInterval(() => void poll(), POLL_MS)
    void poll()
    return () => { alive = false; window.clearInterval(interval) }
  }, [])

  return progress
}
