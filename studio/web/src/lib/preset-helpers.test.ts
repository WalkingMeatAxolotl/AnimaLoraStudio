// preset-helpers 纯函数单测；UI 流程改造（项目页一键新建预设）依赖这层。
import { describe, it, expect } from 'vitest'
import { generateUniquePresetName, PRESET_NAME_RE } from './preset-helpers'

describe('generateUniquePresetName', () => {
  it('returns base unchanged when not taken', () => {
    expect(generateUniquePresetName('myproj_v1', [])).toBe('myproj_v1')
    expect(generateUniquePresetName('myproj_v1', [{ name: 'other' }])).toBe('myproj_v1')
  })

  it('appends _1 when base is taken', () => {
    expect(generateUniquePresetName('myproj_v1', [{ name: 'myproj_v1' }])).toBe('myproj_v1_1')
  })

  it('walks up to first free suffix', () => {
    const existing = [
      { name: 'myproj_v1' },
      { name: 'myproj_v1_1' },
      { name: 'myproj_v1_2' },
    ]
    expect(generateUniquePresetName('myproj_v1', existing)).toBe('myproj_v1_3')
  })

  it('skips holes — picks lowest free', () => {
    const existing = [
      { name: 'myproj_v1' },
      { name: 'myproj_v1_2' }, // _1 不存在
    ]
    expect(generateUniquePresetName('myproj_v1', existing)).toBe('myproj_v1_1')
  })

  it('produces names that satisfy PRESET_NAME_RE — backend will reject otherwise', () => {
    // base 本身合法时，后缀也必须保持合法
    const name = generateUniquePresetName('proj_v1', [{ name: 'proj_v1' }, { name: 'proj_v1_1' }])
    expect(PRESET_NAME_RE.test(name)).toBe(true)
  })
})
