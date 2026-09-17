// @vitest-environment node
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const css = readFileSync(new URL('../../styles/responsive.css', import.meta.url), 'utf8')

describe('Queue row link layout', () => {
  it('keeps a focusable layout box and inherits row tracks without nesting actions', () => {
    const rule = css.match(/\.ui-queue-row-link\s*\{([^}]+)\}/)?.[1] ?? ''
    // Chromium cannot focus the previous display:contents anchor.
    expect(rule).toMatch(/display:\s*grid\s*;/)
    expect(rule).not.toMatch(/display:\s*contents/)
    expect(rule).toMatch(/grid-column:\s*1\s*\/\s*-2\s*;/)
    expect(rule).toMatch(/grid-template-columns:\s*subgrid\s*;/)
    expect(css).toMatch(/\.ui-queue-row-link:focus-visible::after\s*\{[^}]*outline:\s*2px solid var\(--accent\)/)
  })
})
