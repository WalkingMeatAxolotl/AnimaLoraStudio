import { describe, expect, it } from 'vitest'
import type { LoraEntry } from '../../../api/client'
import {
  applyLoraText,
  enabledLoras,
  LoraTextError,
  normalizeLoraPath,
  normalizeLoraUi,
  serializeLoraText,
} from './loraSelection'

const loras: LoraEntry[] = [
  { path: 'G:/loras/alice.safetensors', scale: 1, project_id: 1, version_id: 2 },
  { path: 'D:/ComfyUI/models/loras/styles/ink.safetensors', scale: 0.6 },
]

describe('loraSelection', () => {
  it('migrates missing or corrupt sidecars to stable enabled entries', () => {
    const result = normalizeLoraUi(loras, [{ id: 'kept', enabled: false }, { id: 'kept' }])
    expect(result[0]).toEqual({ id: 'kept', enabled: false })
    expect(result[1].id).not.toBe('kept')
    expect(result[1].enabled).toBe(true)
    expect(normalizeLoraUi(loras, result)).toEqual(result)
  })

  it('filters disabled and missing entries before generation', () => {
    const ui = normalizeLoraUi(loras, [
      { id: 'a', enabled: false },
      { id: 'b', enabled: true },
    ])
    expect(enabledLoras([...loras, { path: '', scale: 1, name: 'missing' }], ui))
      .toEqual([loras[1]])
  })

  it('normalizes Windows paths case-insensitively without collapsing case-sensitive POSIX paths', () => {
    expect(normalizeLoraPath('G:\\Loras\\Alice.safetensors')).toBe('g:/loras/alice.safetensors')
    expect(normalizeLoraPath('/models/Alice.safetensors')).toBe('/models/Alice.safetensors')
  })

  it('serializes enabled entries and applies edited weights without deleting cards', () => {
    const ui = normalizeLoraUi(loras, [
      { id: 'a', enabled: true },
      { id: 'b', enabled: true },
    ])
    expect(serializeLoraText(loras, ui)).toBe('<lora:alice:1>\n<lora:ink:0.6>')

    const result = applyLoraText('<lora:ink:0.75>', loras, ui)
    expect(result.loras).toEqual([loras[0], { ...loras[1], scale: 0.75 }])
    expect(result.ui.map((item) => item.enabled)).toEqual([false, true])
  })

  it.each([
    ['<lora:unknown:1>', 'unknown'],
    ['not a lora', 'invalid'],
    ['<lora:alice:1>\n<lora:alice:0.5>', 'duplicate'],
  ])('reports invalid text %s as %s', (text, code) => {
    const ui = normalizeLoraUi(loras, undefined)
    try {
      applyLoraText(text, loras, ui)
      throw new Error('expected applyLoraText to fail')
    } catch (error) {
      expect(error).toBeInstanceOf(LoraTextError)
      expect((error as LoraTextError).code).toBe(code)
    }
  })

  it('rejects ambiguous basenames', () => {
    const duplicate = [...loras, { path: 'E:/other/alice.safetensors', scale: 1 }]
    expect(() => applyLoraText('<lora:alice:1>', duplicate, normalizeLoraUi(duplicate, undefined)))
      .toThrowError(expect.objectContaining({ code: 'ambiguous' }))
  })
})
