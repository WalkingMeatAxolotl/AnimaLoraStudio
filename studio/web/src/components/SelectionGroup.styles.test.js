import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { expect, it } from 'vitest'

// Source contract, not a rendered-layout test. Keep Node filesystem access out
// of the browser TypeScript project (which intentionally has no Node typings).
// Vitest's css:false stubs CSS imports; read the source from its studio/web cwd.
const tokens = readFileSync(resolve('src/styles/tokens.css'), 'utf8')

it('scopes natural sizing, wrapping and full labels to the opt-in selection recipe', () => {
  const rule = (selector) => {
    const start = tokens.indexOf(`${selector} {`)
    expect(start, `Missing CSS rule: ${selector}`).toBeGreaterThanOrEqual(0)
    return tokens.slice(start).split('}')[0]
  }
  expect(rule('.ui-selection-segmented.ui-selection-content')).toContain('flex-wrap: wrap')
  expect(rule('.ui-selection-segmented.ui-selection-content .ui-selection-item')).toContain('flex: 0 1 auto')
  expect(rule('.ui-selection-segmented.ui-selection-content .ui-selection-item')).toContain('white-space: normal')
  expect(rule('.ui-selection-segmented.ui-selection-content .ui-selection-label')).toContain('overflow: visible')
  expect(rule('.ui-selection-segmented.ui-selection-content .ui-selection-label')).toContain('overflow-wrap: anywhere')
  expect(rule('.ui-selection-segmented .ui-selection-item')).toContain('flex: 1 1 0')
})
