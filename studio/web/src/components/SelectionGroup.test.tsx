import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it } from 'vitest'
import { SegmentedControl, Tabs, selectionItemId } from './SelectionGroup'

type View = 'overview' | 'log' | 'output'

const tabItems = [
  { value: 'overview', label: '详情', controls: 'panel-overview' },
  { value: 'log', label: '日志', controls: 'panel-log', disabled: true },
  { value: 'output', label: '输出', controls: 'panel-output' },
] as const

function TabsHarness({ appearance, layout }: { appearance?: 'underline' | 'segmented'; layout?: 'equal' | 'content' } = {}) {
  const [value, setValue] = useState<View>('overview')
  return (
    <>
      <Tabs
        appearance={appearance}
        layout={layout}
        items={tabItems}
        value={value}
        onChange={setValue}
        ariaLabel="任务详情"
        idPrefix="task-tab"
        className="px-page"
      />
      <div
        id={`panel-${value}`}
        role="tabpanel"
        aria-labelledby={`task-tab-${value}`}
      >
        {value}
      </div>
    </>
  )
}

function ModeHarness({ layout }: { layout?: 'equal' | 'content' } = {}) {
  const [value, setValue] = useState<'single' | 'xy'>('single')
  return (
    <SegmentedControl
      items={[
        { value: 'single', label: '单图' },
        { value: 'xy', label: 'XY 矩阵' },
      ]}
      value={value}
      onChange={setValue}
      ariaLabel="生成模式"
      idPrefix="generate-mode"
      size="sm"
      layout={layout}
    />
  )
}

describe('Tabs', () => {
  it('links tabs to panels and uses the underline appearance by default', async () => {
    const user = userEvent.setup()
    render(<TabsHarness />)

    const group = screen.getByRole('tablist', { name: '任务详情' })
    const overview = screen.getByRole('tab', { name: '详情' })
    const output = screen.getByRole('tab', { name: '输出' })

    expect(group).toHaveClass('ui-selection-underline', 'px-page')
    expect(overview).toHaveAttribute('id', 'task-tab-overview')
    expect(overview).toHaveAttribute('aria-controls', 'panel-overview')
    expect(overview).toHaveAttribute('aria-selected', 'true')
    expect(overview).toHaveAttribute('tabindex', '0')
    expect(output).toHaveAttribute('tabindex', '-1')

    await user.click(output)
    expect(output).toHaveAttribute('aria-selected', 'true')
    expect(output).toHaveAttribute('tabindex', '0')
    expect(screen.getByRole('tabpanel')).toHaveAttribute('id', 'panel-output')
    expect(screen.getByRole('tabpanel')).toHaveAttribute('aria-labelledby', output.id)
  })

  it('keeps tab semantics when using the segmented appearance', () => {
    render(<TabsHarness appearance="segmented" />)

    const group = screen.getByRole('tablist', { name: '任务详情' })
    const overview = screen.getByRole('tab', { name: '详情' })
    expect(group).toHaveClass('ui-selection-segmented')
    expect(group).not.toHaveClass('ui-selection-underline', 'ui-selection-content')
    expect(overview).toHaveAttribute('aria-controls', 'panel-overview')
    expect(screen.getByRole('tabpanel')).toHaveAttribute('id', 'panel-overview')
  })

  it('wraps keyboard focus, supports Home/End, and skips disabled items', async () => {
    const user = userEvent.setup()
    render(<TabsHarness />)

    const overview = screen.getByRole('tab', { name: '详情' })
    const log = screen.getByRole('tab', { name: '日志' })
    const output = screen.getByRole('tab', { name: '输出' })

    overview.focus()
    await user.keyboard('{ArrowRight}')
    expect(output).toHaveFocus()
    expect(output).toHaveAttribute('aria-selected', 'true')
    expect(log).toBeDisabled()

    await user.keyboard('{ArrowRight}')
    expect(overview).toHaveFocus()

    await user.keyboard('{End}')
    expect(output).toHaveFocus()

    await user.keyboard('{Home}')
    expect(overview).toHaveFocus()
  })
})

describe('SegmentedControl', () => {
  it('uses radio semantics for mutually exclusive values', async () => {
    const user = userEvent.setup()
    render(<ModeHarness />)

    const group = screen.getByRole('radiogroup', { name: '生成模式' })
    const single = screen.getByRole('radio', { name: '单图' })
    const xy = screen.getByRole('radio', { name: 'XY 矩阵' })

    expect(group).toHaveClass('ui-selection-segmented', 'ui-selection-sm')
    expect(single).toHaveAttribute('aria-checked', 'true')
    expect(single).not.toHaveAttribute('aria-selected')

    single.focus()
    await user.keyboard('{ArrowLeft}')
    expect(xy).toHaveFocus()
    expect(xy).toHaveAttribute('aria-checked', 'true')
  })
})

describe('Content-sized selections', () => {
  it('opts both segmented APIs into intrinsic sizing without changing their semantics', async () => {
    const user = userEvent.setup()
    render(<><TabsHarness appearance="segmented" layout="content" /><ModeHarness layout="content" /></>)
    expect(screen.getByRole('tablist')).toHaveClass('ui-selection-content')
    expect(screen.getByRole('radiogroup')).toHaveClass('ui-selection-content')
    screen.getByRole('tab', { name: '详情' }).focus()
    await user.keyboard('{End}')
    expect(screen.getByRole('tab', { name: '输出' })).toHaveFocus()
    screen.getByRole('radio', { name: '单图' }).focus()
    await user.keyboard('{ArrowRight}')
    expect(screen.getByRole('radio', { name: 'XY 矩阵' })).toHaveFocus()
  })

})

describe('selectionItemId', () => {
  it('normalizes caller values into stable DOM ids', () => {
    expect(selectionItemId('mode', 'XY Matrix')).toBe('mode-xy-matrix')
  })
})
