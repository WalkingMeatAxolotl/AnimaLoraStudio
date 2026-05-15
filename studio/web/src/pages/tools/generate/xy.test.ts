import { describe, expect, it } from 'vitest'
import type { LoraEntry } from '../../../api/client'
import { cellCount, draftToSpec, parseAxisValues, type XYAxisDraft } from './xy'

describe('parseAxisValues', () => {
  it('parses int axis (steps) splits by comma', () => {
    expect(parseAxisValues('steps', '20, 25, 30')).toEqual([20, 25, 30])
  })

  it('parses float axis (cfg_scale)', () => {
    expect(parseAxisValues('cfg_scale', '3.0, 4.5, 5')).toEqual([3.0, 4.5, 5])
  })

  it('parses string axis (lora_ckpt 路径)', () => {
    expect(parseAxisValues('lora_ckpt', '/a/step100.safetensors, /a/step200.safetensors'))
      .toEqual(['/a/step100.safetensors', '/a/step200.safetensors'])
  })

  it('rejects empty values', () => {
    expect(() => parseAxisValues('steps', '')).toThrow()
    expect(() => parseAxisValues('steps', ' , , ')).toThrow()
  })

  it('rejects non-numeric on int axis', () => {
    expect(() => parseAxisValues('steps', '20, foo, 30')).toThrow(/不是合法数字/)
  })

  it('rejects float on int axis', () => {
    expect(() => parseAxisValues('steps', '20, 25.5')).toThrow(/必须是整数/)
  })

  it('accepts whitespace and trims', () => {
    expect(parseAxisValues('steps', '  20  ,30 , 40 ')).toEqual([20, 30, 40])
  })
})

describe('draftToSpec', () => {
  const loras: LoraEntry[] = [
    { path: '/a.safetensors', scale: 1.0 },
    { path: '/b.safetensors', scale: 0.8 },
  ]

  it('builds spec for non-lora axis without lora_index', () => {
    const d: XYAxisDraft = { axis: 'steps', raw: '20, 30', loraIndex: null }
    const s = draftToSpec(d, loras)
    expect(s.axis).toBe('steps')
    expect(s.values).toEqual([20, 30])
    expect(s.lora_index).toBeUndefined()
  })

  it('lora_scale 改为全局轴：不再要求 lora_index（透传 spec.lora_index=undefined）', () => {
    const d: XYAxisDraft = { axis: 'lora_scale', raw: '0.5, 1.0', loraIndex: null }
    const s = draftToSpec(d, loras)
    expect(s.axis).toBe('lora_scale')
    expect(s.values).toEqual([0.5, 1.0])
    expect(s.lora_index).toBeUndefined()
  })

  it('lora_ckpt axis 仍要求 loraIndex（指向 caller 自己 push 的 anchor 槽）', () => {
    const d: XYAxisDraft = { axis: 'lora_ckpt', raw: '/x/step.safetensors', loraIndex: null }
    expect(() => draftToSpec(d, loras)).toThrow(/必须绑定一个 LoRA/)
  })

  it('lora_ckpt axis lora_index 越界 → throw', () => {
    const d: XYAxisDraft = { axis: 'lora_ckpt', raw: '/x/step.safetensors', loraIndex: 5 }
    expect(() => draftToSpec(d, loras)).toThrow(/不存在/)
  })

  it('lora_ckpt axis with valid lora_index → spec.lora_index 填入', () => {
    const d: XYAxisDraft = { axis: 'lora_ckpt', raw: '/x/step.safetensors', loraIndex: 1 }
    const s = draftToSpec(d, loras)
    expect(s.axis).toBe('lora_ckpt')
    expect(s.lora_index).toBe(1)
    expect(s.values).toEqual(['/x/step.safetensors'])
  })
})

describe('cellCount', () => {
  it('returns x for y=null (单轴退化)', () => {
    expect(cellCount(3, null)).toBe(3)
  })

  it('returns x*y for 2D matrix', () => {
    expect(cellCount(3, 4)).toBe(12)
    expect(cellCount(5, 5)).toBe(25)
  })

  it('handles 0 length gracefully (callers guard)', () => {
    expect(cellCount(0, 3)).toBe(0)
  })
})
