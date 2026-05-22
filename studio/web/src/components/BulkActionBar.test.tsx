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
    selectedKeys: ['a.png', 'b.png'],
    onApply,
    tagSuggestions: ['cat', 'dog', 'solo', 'warm'],
    onSelectAll: vi.fn(),
    onClearSelection: vi.fn(),
    totalCount: 3,
    ...overrides,
  }
  render(
    <ToastProvider>
      <BulkActionBar {...props} />
    </ToastProvider>,
  )
  return { onApply }
}

describe('BulkActionBar (no popover, no scope — operates on selectedKeys)', () => {
  it('add to front prepends typed tag to every selected image', async () => {
    const user = userEvent.setup()
    const { onApply } = renderBar()

    await user.type(screen.getByPlaceholderText('tag1, tag2 (逗号分隔)'), 'warm')
    await user.click(screen.getByRole('button', { name: '+ 加到首部' }))

    expect(onApply).toHaveBeenCalledOnce()
    const updates = onApply.mock.calls[0][0] as Map<string, string[]>
    expect([...updates.keys()].sort()).toEqual(['a.png', 'b.png'])
    expect(updates.get('a.png')).toEqual(['warm', 'cat', 'solo'])
    expect(updates.get('b.png')).toEqual(['warm', 'cat'])
    expect(updates.has('c.png')).toBe(false)
  })

  it('add to back appends typed tag', async () => {
    const user = userEvent.setup()
    const { onApply } = renderBar()

    await user.type(screen.getByPlaceholderText('tag1, tag2 (逗号分隔)'), 'warm')
    await user.click(screen.getByRole('button', { name: '+ 加到尾部' }))

    const updates = onApply.mock.calls[0][0] as Map<string, string[]>
    expect(updates.get('a.png')).toEqual(['cat', 'solo', 'warm'])
    expect(updates.get('b.png')).toEqual(['cat', 'warm'])
  })

  it('remove drops typed tag from selected only', async () => {
    const user = userEvent.setup()
    const { onApply } = renderBar()

    await user.type(screen.getByPlaceholderText('tag1, tag2 (逗号分隔)'), 'cat')
    await user.click(screen.getByRole('button', { name: '− 删除' }))

    const updates = onApply.mock.calls[0][0] as Map<string, string[]>
    expect(updates.get('a.png')).toEqual(['solo'])
    expect(updates.get('b.png')).toEqual([])
    expect(updates.has('c.png')).toBe(false)
  })

  it('replace swaps old→new (and dedupes if collision)', async () => {
    const user = userEvent.setup()
    const { onApply } = renderBar()

    await user.type(screen.getByPlaceholderText('old'), 'cat')
    await user.type(screen.getByPlaceholderText('new'), 'feline')
    await user.click(screen.getByRole('button', { name: '✓' }))

    const updates = onApply.mock.calls[0][0] as Map<string, string[]>
    expect(updates.get('a.png')).toEqual(['feline', 'solo'])
    expect(updates.get('b.png')).toEqual(['feline'])
  })

  it('dedupe needs no input', async () => {
    const dupCache = new Map<string, string[]>([
      ['a.png', ['cat', 'cat', 'solo']],
    ])
    const user = userEvent.setup()
    const { onApply } = renderBar({ cache: dupCache, selectedKeys: ['a.png'] })

    await user.click(screen.getByRole('button', { name: '去重' }))
    const updates = onApply.mock.calls[0][0] as Map<string, string[]>
    expect(updates.get('a.png')).toEqual(['cat', 'solo'])
  })

  it('all op buttons disabled when nothing selected', () => {
    renderBar({ selectedKeys: [] })
    expect(screen.getByRole('button', { name: '+ 加到首部' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '+ 加到尾部' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '− 删除' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '去重' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '✓' })).toBeDisabled()
  })
})
