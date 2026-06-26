import { describe, expect, it } from 'vitest'
import { evalProgressFromResults } from './useEvalProgress'
import type { EvalMetricResult } from '../api/client'

function result(status: string, metricStatuses: Record<string, string> = {}): EvalMetricResult {
  return {
    schema_version: 1,
    has_metrics: true,
    status,
    run_id: 'r',
    metrics: {},
    metric_states: Object.fromEntries(
      Object.entries(metricStatuses).map(([k, s]) => [k, { key: k, status: s, value: null }]),
    ),
  } as EvalMetricResult
}

describe('evalProgressFromResults', () => {
  it('空结果 = 无评估', () => {
    expect(evalProgressFromResults([])).toEqual({ active: false, done: 0, total: 0 })
  })

  it('run 级 running 算 active（覆盖出图阶段，指标还没起）', () => {
    expect(evalProgressFromResults([result('running')])).toEqual({ active: true, done: 0, total: 1 })
  })

  it('指标 pending 算 active（覆盖排队/算指标阶段）', () => {
    expect(
      evalProgressFromResults([result('done', { clip_t: 'pending', clip_i: 'done', dino_i: 'done' })]),
    ).toEqual({ active: true, done: 0, total: 1 })
  })

  it('全终态 = 不 active，done 计数', () => {
    expect(
      evalProgressFromResults([
        result('done', { clip_t: 'done', clip_i: 'done', dino_i: 'done' }),
        result('done', { clip_t: 'done', clip_i: 'failed', dino_i: 'done' }),
      ]),
    ).toEqual({ active: false, done: 2, total: 2 })
  })

  it('混合：一个在评、一个评完 → active 且 done=1/total=2', () => {
    expect(
      evalProgressFromResults([
        result('done', { clip_t: 'done', clip_i: 'done', dino_i: 'done' }),
        result('running', { clip_t: 'running' }),
      ]),
    ).toEqual({ active: true, done: 1, total: 2 })
  })
})
