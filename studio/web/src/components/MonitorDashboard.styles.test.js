import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const source = readFileSync(resolve('src/components/MonitorDashboard.tsx'), 'utf8')

describe('MonitorDashboard layout contracts', () => {
  it('preserves the established six-metric and sample/chart workspace layout', () => {
    expect(source).toContain('grid shrink-0 grid-cols-6 gap-related')
    expect(source).toContain('grid-cols-[minmax(0,1fr)_minmax(0,1.5fr)]')
    expect(source).toContain('flex min-h-0 min-w-0 flex-col gap-related')
    expect(source).not.toContain('MetricGroup')
    expect(source).not.toContain('min-[1281px]:grid-cols')
  })

  it('keeps all three expert charts mounted directly', () => {
    expect(source.match(/<SeriesChart/g)).toHaveLength(3)
    expect(source).toContain("title={t('monitor.chart.loss')}")
    expect(source).toContain("title={t('monitor.chart.learningRate')}")
    expect(source).toContain("title={t('monitor.chart.optimizerD')}")
  })

  it('uses semantic density spacing and theme colors without nested metric cards', () => {
    expect(source).toContain('gap-field overflow-y-auto p-section')
    expect(source).toContain('card min-w-0 px-field py-related')
    expect(source).toContain('gap-related')
    expect(source).toContain('rawColor="var(--fg-tertiary)"')
    expect(source).not.toContain('rawColor="rgba(')
  })
})
