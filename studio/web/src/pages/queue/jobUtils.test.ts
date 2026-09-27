import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Task } from '../../api/client'
import i18n from '../../i18n'
import { fmtJobAgo, fmtJobTime, fmtJobUntil, fmtParamValue, jobJumpPath } from './jobUtils'

function task(over: Partial<Task>): Task {
  return {
    id: 1, name: 'j', config_name: 'j', status: 'done', priority: 0,
    created_at: 0, started_at: null, finished_at: null, pid: null,
    exit_code: null, output_dir: null, error_msg: null,
    project_id: 7, version_id: 9, ...over,
  } as Task
}

describe('队列时间与完整参数', () => {
  afterEach(async () => {
    vi.restoreAllMocks()
    await i18n.changeLanguage('zh')
  })

  it.each([
    { lang: 'zh', locale: 'zh-CN', now: '刚刚', ago: '2m 前', until: '1h 2m 后', soon: '即将开始' },
    { lang: 'en', locale: 'en-US', now: 'Just now', ago: '2m ago', until: 'In 1h 2m', soon: 'Starting soon' },
  ])('$lang 使用界面语言显示过去/计划时间', async (copy) => {
    await i18n.changeLanguage(copy.lang)
    const now = new Date('2026-09-17T04:00:00Z').getTime()
    vi.spyOn(Date, 'now').mockReturnValue(now)
    expect(fmtJobAgo(now / 1000)).toBe(copy.now)
    expect(fmtJobAgo(now / 1000 - 120)).toBe(copy.ago)
    expect(fmtJobUntil(now / 1000 + 3720)).toBe(copy.until)
    expect(fmtJobUntil(now / 1000 - 1)).toBe(copy.soon)
    expect(fmtJobTime(now / 1000)).toBe(new Date(now).toLocaleString(copy.locale, { hour12: false }))
    expect(fmtJobTime(null)).toBe('—')
  })

  it('长字符串、对象和嵌套数组不在格式化阶段截断', () => {
    const long = `${'x'.repeat(500)} full-tail`
    const t = (key: string) => key
    expect(fmtParamValue(long, t)).toBe(long)
    expect(fmtParamValue({ prompt: long }, t)).toBe(JSON.stringify({ prompt: long }))
    expect(fmtParamValue([{ prompt: long }], t)).toContain(long)
    expect(fmtParamValue(true, t)).toBe('field.yes')
    expect(fmtParamValue(null, t)).toBe('—')
  })
})

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
