import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { expect, it } from 'vitest'

// Source contract for authored Train geometry. Browser verification still owns
// the actual pixel alignment across viewports.
const responsive = readFileSync(resolve('src/styles/responsive.css'), 'utf8')

function rule(selector) {
  const start = responsive.indexOf(`${selector} {`)
  expect(start, `Missing CSS rule: ${selector}`).toBeGreaterThanOrEqual(0)
  return responsive.slice(start).split('}')[0]
}

it('places parameter display in the far-right column above the section index', () => {
  expect(rule('.train-config-toolbar')).toContain(
    'grid-template-columns: minmax(0, 1fr) var(--train-section-index-width)',
  )
  expect(rule('.train-config-toolbar-mode')).toContain('grid-column: 2')
  expect(rule('.train-config-toolbar-mode')).toContain('justify-self: end')
  expect(rule('.train-section-index')).toContain('width: var(--train-section-index-width)')
})

it('keeps the display-mode control at the far right when the index is hidden', () => {
  const media = responsive.slice(responsive.indexOf('@media (max-width: 1280px)'))
  const start = media.indexOf('.train-config-toolbar {')
  expect(start).toBeGreaterThanOrEqual(0)
  expect(media.slice(start).split('}')[0]).toContain(
    'grid-template-columns: minmax(0, 1fr) auto',
  )
})
