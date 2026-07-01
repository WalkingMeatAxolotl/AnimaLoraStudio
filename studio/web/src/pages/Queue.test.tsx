/** Queue page — taskKind 契约测试（0.17 P-D）。
 *
 *  taskKind 从后端权威 `task.task_type`（train/reg_ai/generate）派生队列行的类型，
 *  取代旧 inferKind 的 config_name 子串猜测。核心契约：
 *   1) 直接返回后端 task_type；
 *   2) 缺字段（老 mock / 极老行）兜底 'train'；
 *   3) 不再受 config_name 影响 —— 修掉旧 inferKind 把名字含 "reg"/"tag" 的
 *      训练任务误判成别的类型的 latent bug。 */
import { describe, expect, it } from 'vitest'
import type { Task } from '../api/client'
import { taskKind } from './Queue'

function makeTask(overrides: Partial<Task> = {}): Task {
  return {
    id: 1, name: 'train', config_name: 'train',
    status: 'running', priority: 0,
    created_at: 1000, started_at: 1100, finished_at: null,
    pid: 1234, exit_code: null, output_dir: null, error_msg: null,
    ...overrides,
  }
}

describe('taskKind', () => {
  it('直接返回后端 task_type', () => {
    expect(taskKind(makeTask({ task_type: 'train' }))).toBe('train')
    expect(taskKind(makeTask({ task_type: 'reg_ai' }))).toBe('reg_ai')
    expect(taskKind(makeTask({ task_type: 'generate' }))).toBe('generate')
  })

  it('缺 task_type（老行 / 老 mock）兜底 train', () => {
    expect(taskKind(makeTask())).toBe('train')
  })

  it('不看 config_name —— 名字含 reg/tag 的训练任务仍是 train', () => {
    expect(taskKind(makeTask({ task_type: 'train', config_name: 'my_reg_lora' }))).toBe('train')
    expect(taskKind(makeTask({ task_type: 'train', config_name: 'wd14_tag_run' }))).toBe('train')
  })
})
