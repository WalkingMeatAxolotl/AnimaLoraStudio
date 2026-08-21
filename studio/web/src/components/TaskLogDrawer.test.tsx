import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import TaskLogDrawer, { type LogSource } from './TaskLogDrawer'

function makeSource(overrides: Partial<LogSource> = {}): LogSource {
  return {
    key: 'tag',
    label: '打标任务',
    status: 'running',
    lines: ['line a', 'line b'],
    startedAt: 1700000000,
    finishedAt: null,
    ...overrides,
  }
}

describe('TaskLogDrawer (issue #251)', () => {
  it('无 source 时整体隐藏', () => {
    const { container } = render(<TaskLogDrawer sources={[null, false, undefined]} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('挂载即 running（回放场景）→ 自动展开，日志面板升起且显示尾部', () => {
    render(<TaskLogDrawer sources={[makeSource()]} />)
    expect(screen.getByRole('button', { name: /打标任务/ })).toHaveAttribute(
      'aria-expanded',
      'true',
    )
    expect(screen.getByTestId('log-drawer-body')).toHaveStyle({ height: '40vh' })
    expect(screen.getByText("line a")).toBeInTheDocument()
    expect(screen.getAllByText("line b").length).toBeGreaterThan(0)
  })

  it('挂载即 done（历史回放）→ 默认收起（body 高度 0），条上显示最后一行', () => {
    render(
      <TaskLogDrawer
        sources={[makeSource({ status: 'done', finishedAt: 1700000010 })]}
      />,
    )
    const strip = screen.getByRole('button', { name: /打标任务/ })
    expect(strip).toHaveAttribute('aria-expanded', 'false')
    // body 常驻 DOM 做高度动画，收起 = 高度 0；收起条上有最后一行
    expect(screen.getByTestId('log-drawer-body')).toHaveStyle({ height: '0px' })
    expect(screen.getByText('line b')).toBeInTheDocument()
  })

  it('任务结束不自动收起（done / failed 都保持展开，收起只靠手动或切页卸载）', () => {
    const { rerender } = render(<TaskLogDrawer sources={[makeSource()]} />)
    const strip = () => screen.getByRole('button', { name: /打标任务/ })
    expect(strip()).toHaveAttribute('aria-expanded', 'true')

    rerender(<TaskLogDrawer sources={[makeSource({ status: 'done' })]} />)
    expect(strip()).toHaveAttribute('aria-expanded', 'true')

    rerender(<TaskLogDrawer sources={[makeSource()]} />)
    rerender(<TaskLogDrawer sources={[makeSource({ status: 'failed' })]} />)
    expect(strip()).toHaveAttribute('aria-expanded', 'true')
  })

  it('点击收起条切换展开/收起，手动状态保持', async () => {
    const user = userEvent.setup()
    render(<TaskLogDrawer sources={[makeSource()]} />)
    const strip = screen.getByRole('button', { name: /打标任务/ })
    await user.click(strip)
    expect(strip).toHaveAttribute('aria-expanded', 'false')
    await user.click(strip)
    expect(strip).toHaveAttribute('aria-expanded', 'true')
  })

  it('多 source 单显：活着的优先于更晚启动的终态任务', () => {
    render(
      <TaskLogDrawer
        sources={[
          makeSource({
            key: 'reg_build',
            label: 'Booru 建集',
            status: 'done',
            startedAt: 1700009999,
          }),
          makeSource({ key: 'reg_ai', label: 'AI 先验生成', status: 'running' }),
        ]}
      />,
    )
    expect(screen.getByText('AI 先验生成')).toBeInTheDocument()
    expect(screen.queryByText('Booru 建集')).toBeNull()
  })

  it('都是终态时选最近启动的', () => {
    render(
      <TaskLogDrawer
        sources={[
          makeSource({ key: 'a', label: '旧任务', status: 'done', startedAt: 100 }),
          makeSource({ key: 'b', label: '新任务', status: 'failed', startedAt: 200 }),
        ]}
      />,
    )
    expect(screen.getByText('新任务')).toBeInTheDocument()
    expect(screen.queryByText('旧任务')).toBeNull()
  })

  it('live 且提供 onCancel 时显示取消按钮，点击不触发开合', async () => {
    const onCancel = vi.fn()
    const user = userEvent.setup()
    render(<TaskLogDrawer sources={[makeSource({ onCancel })]} />)
    const strip = screen.getByRole('button', { name: /打标任务/ })
    expect(strip).toHaveAttribute('aria-expanded', 'true')
    await user.click(screen.getByRole('button', { name: '取消' }))
    expect(onCancel).toHaveBeenCalledOnce()
    expect(strip).toHaveAttribute('aria-expanded', 'true')
  })

  it('终态不显示取消按钮', () => {
    render(
      <TaskLogDrawer sources={[makeSource({ status: 'done', onCancel: () => {} })]} />,
    )
    expect(screen.queryByRole('button', { name: '取消' })).toBeNull()
  })

  it('failed 且提供 onRetry 时显示重试按钮，点击不触发开合', async () => {
    const onRetry = vi.fn()
    const user = userEvent.setup()
    render(<TaskLogDrawer sources={[makeSource({ status: 'failed', onRetry })]} />)
    const strip = screen.getByRole('button', { name: /打标任务/ })
    expect(strip).toHaveAttribute('aria-expanded', 'false')
    await user.click(screen.getByRole('button', { name: '重试' }))
    expect(onRetry).toHaveBeenCalledOnce()
    expect(strip).toHaveAttribute('aria-expanded', 'false')
  })

  it('非 failed 不显示重试按钮', () => {
    render(
      <TaskLogDrawer sources={[makeSource({ status: 'done', onRetry: () => {} })]} />,
    )
    expect(screen.queryByRole('button', { name: '重试' })).toBeNull()
  })
})

describe('抽屉一层化（日志二期 UI）', () => {
  const HDR = (lvl: string, msg: string) =>
    `2026-08-19 14:03:22.417 ${lvl.padEnd(5)} studio.workers.tag_worker: ${msg}`

  it('收起态预览行跳过 DEBUG，显示最后一条 INFO+ 记录', () => {
    render(
      <TaskLogDrawer
        sources={[makeSource({
          status: 'done',
          finishedAt: 1700000010,
          lines: [HDR('INFO', 'tagging done 43/43'), HDR('DEBUG', 'internal detail')],
        })]}
      />,
    )
    expect(screen.getByText(/tagging done 43\/43/)).toBeInTheDocument()
    expect(screen.queryByText(/internal detail/)).not.toBeInTheDocument()
  })

  it('展开后 LogView 工具栏渲染在 header 内（一层化），预览行让位', async () => {
    const user = userEvent.setup()
    render(<TaskLogDrawer sources={[makeSource({ status: 'done', finishedAt: 1700000010, lines: [HDR('INFO', 'hello world')] })]} />)
    // 收起：预览可见、无工具栏
    expect(screen.getByText(/hello world/)).toBeInTheDocument()
    expect(screen.queryByTestId('log-view-toolbar')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { expanded: false }))
    // 展开：工具栏出现且位于 header（aria-expanded 的开合条）内部
    const toolbar = await screen.findByTestId('log-view-toolbar')
    const header = screen.getByRole('button', { expanded: true })
    expect(header.contains(toolbar)).toBe(true)
  })
})
