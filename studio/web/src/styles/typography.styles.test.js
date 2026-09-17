import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { expect, it } from 'vitest'

const tokens = readFileSync(resolve('src/styles/tokens.css'), 'utf8')
const responsive = readFileSync(resolve('src/styles/responsive.css'), 'utf8')

function rule(selector, source = tokens) {
  const start = source.indexOf(`${selector} {`)
  expect(start, `Missing CSS rule: ${selector}`).toBeGreaterThanOrEqual(0)
  return source.slice(start).split('}')[0]
}

it('uses the readable secondary role for section labels', () => {
  const sectionLabel = rule('.type-section-label')
  expect(sectionLabel).toContain('color: var(--fg-secondary)')
  expect(sectionLabel).not.toContain('color: var(--fg-tertiary)')
})

it('distinguishes read-only data labels from form labels', () => {
  const dataLabel = rule('.type-data-label')
  expect(dataLabel).toContain('font-size: var(--t-xs)')
  expect(dataLabel).toContain('color: var(--fg-secondary)')
  expect(rule('.type-field-label')).toContain('font-size: var(--t-sm)')
})

it('stretches paired overview cards to a common row height', () => {
  expect(rule('.ui-queue-overview-grid', responsive)).toContain('align-items: stretch')
})
