import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ComponentProps } from 'react'
import { describe, expect, it, vi } from 'vitest'

import BulkActionBar from './BulkActionBar'
import { ToastProvider } from './Toast'

const cache = new Map<string, string[]>([
  ['a.png', ['cat', 'solo']],
  ['b.png', ['cat']],
  ['c.png', ['dog']],
])

function renderBar(overrides: Partial<ComponentProps<typeof BulkActionBar>> = {}) {
  const onApply = vi.fn()
  const props: ComponentProps<typeof BulkActionBar> = {
    cache,
    selectedKeys: [],
    onApply,
    tagSuggestions: ['cat', 'dog', 'solo', 'warm'],
    filterTag: 'cat',
    onFilterTagChange: vi.fn(),
    filteredKeys: ['a.png', 'b.png'],
    totalCount: 3,
    filteredCount: 2,
    onSelectAll: vi.fn(),
    onClearSelection: vi.fn(),
    ...overrides,
  }
  render(
    <ToastProvider>
      <BulkActionBar {...props} />
    </ToastProvider>,
  )
  return { onApply }
}

describe('BulkActionBar', () => {
  it('can apply a bulk operation to the current filtered result without touching hidden images', async () => {
    const user = userEvent.setup()
    const { onApply } = renderBar()

    await user.selectOptions(screen.getByRole('combobox'), 'filtered')
    await user.click(screen.getByRole('button', { name: '+ 加 tag' }))
    await user.type(screen.getByPlaceholderText('tag1, tag2 (逗号分隔)'), 'warm')
    await user.click(screen.getByRole('button', { name: '执行' }))

    expect(onApply).toHaveBeenCalledOnce()
    const updates = onApply.mock.calls[0][0] as Map<string, string[]>
    expect([...updates.keys()].sort()).toEqual(['a.png', 'b.png'])
    expect(updates.get('a.png')).toEqual(['warm', 'cat', 'solo'])
    expect(updates.get('b.png')).toEqual(['warm', 'cat'])
    expect(updates.has('c.png')).toBe(false)
  })

  it('keeps selected scope disabled until a visible image is selected', () => {
    renderBar({ selectedKeys: [] })
    expect(screen.getByRole('button', { name: '+ 加 tag' })).toBeDisabled()
  })
})
