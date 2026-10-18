import { describe, expect, it } from 'vitest'
import { coerceGenerateFamily, parseSkipLayers } from './types'

describe('parseSkipLayers', () => {
  it('逗号 / 空白 / 全角逗号分隔的非负整数', () => {
    expect(parseSkipLayers('22')).toEqual([22])
    expect(parseSkipLayers('7, 8 9，10')).toEqual([7, 8, 9, 10])
  })
  it('空串 = 关闭（空列表）', () => {
    expect(parseSkipLayers('')).toEqual([])
    expect(parseSkipLayers('  ')).toEqual([])
  })
  it('含非法项返回 null', () => {
    expect(parseSkipLayers('22, a')).toBeNull()
    expect(parseSkipLayers('-1')).toBeNull()
    expect(parseSkipLayers('1.5')).toBeNull()
  })
})

describe('coerceGenerateFamily', () => {
  it('已知族原样，其余落回 anima', () => {
    expect(coerceGenerateFamily('moss')).toBe('moss')
    expect(coerceGenerateFamily('krea2')).toBe('krea2')
    expect(coerceGenerateFamily(undefined)).toBe('anima')
    expect(coerceGenerateFamily('sdxl')).toBe('anima')
  })
})
