import { describe, expect, it } from 'vitest'
import { configToYaml } from './yamlPreview'

// 预期值以 PyYAML 实际输出为准：
// yaml.safe_dump(..., allow_unicode=True, sort_keys=False, default_flow_style=False)
describe('configToYaml', () => {
  it('renders scalars like PyYAML safe_dump', () => {
    expect(
      configToYaml({
        optimizer_type: 'adamw',
        learning_rate: 0.0001,
        epochs: 10,
        flag: true,
        off_flag: false,
        nothing: null,
      })
    ).toBe(
      [
        'optimizer_type: adamw',
        'learning_rate: 0.0001',
        'epochs: 10',
        'flag: true',
        'off_flag: false',
        'nothing: null',
      ].join('\n')
    )
  })

  it('keeps insertion order (no sorting)', () => {
    expect(configToYaml({ zeta: 1, alpha: 2 })).toBe('zeta: 1\nalpha: 2')
  })

  it('leaves paths and enums unquoted', () => {
    expect(configToYaml({ path: 'G:/models/anima.safetensors', rel: './dataset' })).toBe(
      'path: G:/models/anima.safetensors\nrel: ./dataset'
    )
  })

  it('quotes ambiguous strings the way PyYAML does', () => {
    expect(
      configToYaml({
        empty: '',
        numeric_str: '123',
        bool_str: 'true',
        colon_space: 'a: b',
      })
    ).toBe(["empty: ''", "numeric_str: '123'", "bool_str: 'true'", "colon_space: 'a: b'"].join('\n'))
  })

  it('renders lists in block style, empty list inline', () => {
    expect(configToYaml({ resolution: [1024, 768], tags: [] })).toBe(
      'resolution:\n- 1024\n- 768\ntags: []'
    )
  })

  it('escapes multiline strings as valid YAML', () => {
    // 偏差点：PyYAML 用单引号 + 空行折行，这里用双引号转义 —— 均为合法 YAML
    expect(configToYaml({ prompt: 'line1\nline2' })).toBe('prompt: "line1\\nline2"')
  })
})
