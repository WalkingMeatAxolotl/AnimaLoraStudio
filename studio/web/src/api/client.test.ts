import { describe, expect, it } from 'vitest'
import { makeApiError } from './client'

// ADR-0009 Phase 2：makeApiError 统一把后端错误信封解析成 ApiError。
// 测试 locale = zh（setup.ts），故本地化断言用中文。
describe('makeApiError', () => {
  it('body.error.code → errors.<code> i18n 本地化 + details 插值', () => {
    const e = makeApiError(404, 'Not Found', {
      detail: 'Preset "foo" not found',
      error: { code: 'preset.not_found', message: 'Preset "foo" not found', details: { name: 'foo' }, trace_id: 'abc123' },
    })
    expect(e.message).toBe('预设「foo」不存在')
    expect(e.code).toBe('preset.not_found')
    expect(e.traceId).toBe('abc123')
    expect(e.status).toBe(404)
  })

  it('未知 code → 回退 error.message（defaultValue）', () => {
    const e = makeApiError(400, 'Bad Request', {
      detail: 'something specific',
      error: { code: 'http.400', message: 'something specific', trace_id: 't1' },
    })
    expect(e.message).toBe('something specific')
    expect(e.code).toBe('http.400')
  })

  it('无 error 信封、detail 为 string → 用 detail 作文案（老路径 fallback）', () => {
    const e = makeApiError(400, 'Bad Request', { detail: 'legacy message' })
    expect(e.message).toBe('legacy message')
    expect(e.code).toBeUndefined()
  })

  it('结构化数据在 error.details（409 冲突）→ 挂到 err.detail 给 callsite', () => {
    const e = makeApiError(409, 'Conflict', {
      detail: 'Preset "foo" already exists',
      error: {
        code: 'preset.exists',
        message: 'Preset "foo" already exists',
        details: { name: 'foo', config: { a: 1 }, suggested_name: 'foo-2' },
        trace_id: 't2',
      },
    })
    expect(e.message).toBe('预设「foo」已存在')
    expect(e.detail).toEqual({ name: 'foo', config: { a: 1 }, suggested_name: 'foo-2' })
  })

  it('非 JSON body（null）→ statusText 兜底 + header trace', () => {
    const e = makeApiError(500, 'Internal Server Error', null, 'hdr-trace')
    expect(e.message).toBe('500 Internal Server Error')
    expect(e.traceId).toBe('hdr-trace')
  })
})
