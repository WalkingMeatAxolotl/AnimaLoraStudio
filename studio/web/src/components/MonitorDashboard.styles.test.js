import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const source = readFileSync(resolve('src/components/MonitorDashboard.tsx'), 'utf8')

describe('MonitorDashboard layout contracts', () => {
  it('uses the shared compact-desktop breakpoint without hiding expert metrics', () => {
    expect(source).toContain('min-[1281px]:grid-cols-[minmax(280px,0.9fr)_minmax(0,1.5fr)]')
    expect(source).toContain('min-[1080px]:grid-cols-2 min-[1281px]:grid-cols-1')
    expect(source).not.toContain('grid-cols-6')
    expect(source).not.toContain('hidden xl:')
  })

  it('keeps all three expert charts mounted directly', () => {
    expect(source.match(/<SeriesChart/g)).toHaveLength(3)
    expect(source).toContain("title={t('monitor.chart.loss')}")
    expect(source).toContain("title={t('monitor.chart.learningRate')}")
    expect(source).toContain("title={t('monitor.chart.optimizerD')}")
  })

  it('uses semantic density spacing and theme colors for the dashboard shell', () => {
    expect(source).toContain('gap-section overflow-y-auto p-page')
    expect(source).toContain('gap-related')
    expect(source).toContain('rawColor="var(--fg-tertiary)"')
    expect(source).not.toContain('rawColor="rgba(')
  })
})
