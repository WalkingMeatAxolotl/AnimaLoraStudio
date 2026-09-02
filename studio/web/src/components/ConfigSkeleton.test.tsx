/**
 * ConfigSkeleton 单元测试。
 *
 * 验证：
 *   - card 变体外层是 <section>，flat 变体外层是 <div>
 *   - role="status" + aria-label + sr-only 文本一致
 *   - groups prop 控制组数，row 数控制每组行数
 *   - label 透传到 aria + sr-only
 */
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import ConfigSkeleton from './ConfigSkeleton'

describe('ConfigSkeleton', () => {
  it('renders card variant with section + per-group border', () => {
    const { container } = render(<ConfigSkeleton groups={[2, 3]} />)
    expect(container.querySelector('section')).toBeTruthy()
    // 2 个组 × 1 个 border 卡片
    const cards = container.querySelectorAll('.border.border-subtle.bg-surface')
    expect(cards.length).toBe(2)
  })

  it('renders flat variant with div and no per-group border', () => {
    const { container } = render(<ConfigSkeleton variant="flat" groups={[2, 3]} />)
    expect(container.querySelector('section')).toBeFalsy()
    expect(container.querySelector('div[role="status"]')).toBeTruthy()
    expect(container.querySelectorAll('.border.border-subtle.bg-surface').length).toBe(0)
  })

  it('uses label for aria-label and sr-only text', () => {
    render(<ConfigSkeleton label="加载训练配置中" />)
    const status = screen.getByRole('status')
    expect(status.getAttribute('aria-label')).toBe('加载训练配置中')
    expect(status).toHaveAttribute('aria-busy', 'true')
    expect(screen.getByText('加载训练配置中...')).toBeInTheDocument()
  })

  it('uses the shared motion-safe skeleton class', () => {
    const { container, rerender } = render(<ConfigSkeleton groups={[1]} />)
    expect(container.querySelector('.ui-skeleton')).toBeInTheDocument()
    expect(container.querySelector('.animate-pulse')).not.toBeInTheDocument()

    rerender(<ConfigSkeleton variant="flat" groups={[1]} />)
    expect(screen.getByRole('status')).toHaveClass('ui-skeleton')
  })

  it('defaults to 4 groups when groups prop is omitted', () => {
    const { container } = render(<ConfigSkeleton />)
    // section > 4 children groups + 1 sr-only span = 5 direct children
    const section = container.querySelector('section')
    expect(section).toBeTruthy()
    // 4 group cards
    expect(container.querySelectorAll('.border.border-subtle.bg-surface').length).toBe(4)
  })

  it('renders the requested row count per group', () => {
    const { container } = render(<ConfigSkeleton groups={[1, 5]} />)
    const groups = container.querySelectorAll('section > div')
    expect(groups.length).toBe(2)
    // 第 1 组 1 行 = 1 label bar + 1 input bar = 2 row 容器内的 div
    // 第 2 组 5 行 = 5 row 容器（每行 2 个 bar）
    const rowsInGroup1 = groups[0].querySelectorAll('.flex.flex-col.gap-1')
    const rowsInGroup2 = groups[1].querySelectorAll('.flex.flex-col.gap-1')
    expect(rowsInGroup1.length).toBe(1)
    expect(rowsInGroup2.length).toBe(5)
  })
})
