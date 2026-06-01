import { describe, expect, it } from 'vitest'
import type { LoraEntry } from '../../../api/client'
import { buildXYMatrix, cellCount, draftToSpec, parseAxisValues, type XYAxisDraft } from './xy'

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

describe('buildXYMatrix（只发被轴引用的 anchor，丢弃 picker 沉积的孤儿）', () => {
  const CHEN = { path: 'G:/chen-bin_v3.4.safetensors', scale: 1, project_id: 2, version_id: 4 }
  const ORPHAN = { path: 'G:/chen-bin_v3.2.safetensors', scale: 1, project_id: 1, version_id: 1 }
  const HOSHI = { path: 'G:/hoshi.safetensors', scale: 1, project_id: 3, version_id: 9 }

  it('非 lora_ckpt 轴（steps）→ lora_configs 为空（孤儿不当 base LoRA 发）', () => {
    // 复现根因：xyLoras 里有 ORPHAN（picker 残留），但当前 X 轴是 steps，没引用它。
    const x: XYAxisDraft = { axis: 'steps', raw: '20, 25, 30', loraIndex: null }
    const { xy_matrix, loraConfigs } = buildXYMatrix(x, null, [ORPHAN, HOSHI])
    expect(loraConfigs).toEqual([]) // ← 修前会把 [ORPHAN, HOSHI] 整桶发出去
    expect(xy_matrix.x.values).toEqual([20, 25, 30])
    expect(xy_matrix.y).toBeNull()
  })

  it('lora_ckpt 轴只发被引用的那条 anchor，loraIndex 重映射到 0', () => {
    // xyLoras=[ORPHAN, CHEN]，X 轴 loraIndex=1 指向 CHEN；ORPHAN 没被引用 → 丢弃。
    const x: XYAxisDraft = { axis: 'lora_ckpt', raw: CHEN.path, loraIndex: 1 }
    const { xy_matrix, loraConfigs } = buildXYMatrix(x, null, [ORPHAN, CHEN])
    expect(loraConfigs).toEqual([CHEN]) // ← 没选过的 ORPHAN(v3.2) 不混进来
    expect(xy_matrix.x.lora_index).toBe(0) // 1 → 0 重映射
    expect(xy_matrix.x.values).toEqual([CHEN.path])
  })

  it('X/Y 都是 lora_ckpt 引用不同 anchor → 两条都保留，各自重映射', () => {
    const x: XYAxisDraft = { axis: 'lora_ckpt', raw: CHEN.path, loraIndex: 2 }
    const y: XYAxisDraft = { axis: 'lora_ckpt', raw: HOSHI.path, loraIndex: 0 }
    // loras=[HOSHI, ORPHAN, CHEN]：X→idx2(CHEN)，Y→idx0(HOSHI)，ORPHAN(idx1) 丢弃
    const { xy_matrix, loraConfigs } = buildXYMatrix(x, y, [HOSHI, ORPHAN, CHEN])
    expect(loraConfigs).toEqual([CHEN, HOSHI]) // 按出现顺序：X 先 → CHEN=0, Y → HOSHI=1
    expect(xy_matrix.x.lora_index).toBe(0)
    expect(xy_matrix.y?.lora_index).toBe(1)
  })

  it('X/Y 引用同一 anchor → 去重成一条，两轴指同一索引', () => {
    const x: XYAxisDraft = { axis: 'lora_ckpt', raw: CHEN.path, loraIndex: 0 }
    const y: XYAxisDraft = { axis: 'lora_scale', raw: '0.6, 0.8', loraIndex: 0 }
    const { loraConfigs, xy_matrix } = buildXYMatrix(x, y, [CHEN])
    expect(loraConfigs).toEqual([CHEN])
    expect(xy_matrix.x.lora_index).toBe(0)
    // lora_scale 不要求 lora_index，透传 undefined
    expect(xy_matrix.y?.lora_index).toBeUndefined()
  })

  it('lora_ckpt 轴 loraIndex 越界 → 抛错（不静默吞掉）', () => {
    const x: XYAxisDraft = { axis: 'lora_ckpt', raw: CHEN.path, loraIndex: 5 }
    expect(() => buildXYMatrix(x, null, [CHEN])).toThrow(/不存在/)
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
