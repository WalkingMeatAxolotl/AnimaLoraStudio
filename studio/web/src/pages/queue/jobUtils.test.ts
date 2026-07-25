import { describe, expect, it } from 'vitest'
import type { Task } from '../../api/client'
import { jobJumpPath } from './jobUtils'

function task(over: Partial<Task>): Task {
  return {
    id: 1, name: 'j', config_name: 'j', status: 'done', priority: 0,
    created_at: 0, started_at: null, finished_at: null, pid: null,
    exit_code: null, output_dir: null, error_msg: null,
    project_id: 7, version_id: 9, ...over,
  } as Task
}

describe('jobJumpPath', () => {
  it('评估跳概览的评估 tab —— 不是训练页', () => {
    const path = jobJumpPath(task({ task_type: 'eval_session' }))
    expect(path).toBe('/projects/7?version=9&tab=eval')
  })

  it('给了 session id 就深链到那一次（否则会落到该 version 最新一次）', () => {
    expect(jobJumpPath(task({ task_type: 'eval_session' }), 42))
      .toBe('/projects/7?version=9&tab=eval&session=42')
  })

  it('上一代 eval 子作业的存量行同样归到评估 tab', () => {
    expect(jobJumpPath(task({ task_type: 'eval_samples' })))
      .toBe('/projects/7?version=9&tab=eval')
  })

  it('其余数据作业跳各自的原生步骤页', () => {
    expect(jobJumpPath(task({ task_type: 'tag' }))).toBe('/projects/7/v/9/tag')
    expect(jobJumpPath(task({ task_type: 'reg_build' }))).toBe('/projects/7/v/9/reg')
    expect(jobJumpPath(task({ task_type: 'preprocess' }))).toBe('/projects/7/v/9/preprocess')
    expect(jobJumpPath(task({ task_type: 'download' }))).toBe('/projects/7/download')
  })

  it('非作业类型（train/generate）另有专链，这里返回 null', () => {
    expect(jobJumpPath(task({ task_type: 'train' }))).toBeNull()
    expect(jobJumpPath(task({ task_type: 'generate' }))).toBeNull()
  })

  it('缺 version 的作业不给版本级深链', () => {
    expect(jobJumpPath(task({ task_type: 'eval_session', version_id: null }))).toBeNull()
  })
})
