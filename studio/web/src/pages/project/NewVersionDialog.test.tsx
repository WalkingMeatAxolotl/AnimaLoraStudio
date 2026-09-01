import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { NewVersionDialog } from './Layout'

describe('NewVersionDialog (PP10.1)', () => {
  it('hides 从…创建 dropdown when no existing versions', () => {
    render(
      <NewVersionDialog
        existingLabels={[]}
        existingVersions={[]}
        onCancel={() => {}}
        onSubmit={() => {}}
      />,
    )
    expect(screen.queryByText(/从…创建/)).toBeNull()
  })

  it('shows dropdown with existing versions and submits with null forkFrom by default', async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    render(
      <NewVersionDialog
        existingLabels={['baseline']}
        existingVersions={[{ id: 7, label: 'baseline' }]}
        onCancel={() => {}}
        onSubmit={onSubmit}
      />,
    )
    expect(screen.getByText(/从…创建/)).toBeInTheDocument()
    expect(screen.getByRole('option', { name: '从空白开始' })).toBeInTheDocument()
    expect(
      screen.getByRole('option', { name: /从 baseline 复制/ }),
    ).toBeInTheDocument()

    await user.type(screen.getByPlaceholderText(/baseline/), 'high-lr')
    await user.click(screen.getByRole('button', { name: '创建' }))
    expect(onSubmit).toHaveBeenCalledWith('high-lr', null)
  })

  it('passes fork_from_version_id when user picks a source version', async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    render(
      <NewVersionDialog
        existingLabels={['baseline']}
        existingVersions={[{ id: 7, label: 'baseline' }]}
        onCancel={() => {}}
        onSubmit={onSubmit}
      />,
    )
    await user.type(screen.getByPlaceholderText(/baseline/), 'forked')
    await user.selectOptions(screen.getByRole('combobox'), '7')
    // 选了源 version 后应该出现 hint
    expect(screen.getByText(/将复制 train\/、reg\//)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '创建' }))
    expect(onSubmit).toHaveBeenCalledWith('forked', 7)
  })

  it('rejects duplicate label', async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    render(
      <NewVersionDialog
        existingLabels={['baseline']}
        existingVersions={[{ id: 7, label: 'baseline' }]}
        onCancel={() => {}}
        onSubmit={onSubmit}
      />,
    )
    await user.type(screen.getByPlaceholderText(/baseline/), 'baseline')
    await user.click(screen.getByRole('button', { name: '创建' }))
    expect(onSubmit).not.toHaveBeenCalled()
    expect(screen.getByText(/label 已存在/)).toBeInTheDocument()
    expect(screen.getByPlaceholderText(/baseline/)).toHaveAttribute('aria-invalid', 'true')
    expect(screen.getByPlaceholderText(/baseline/)).toHaveAccessibleDescription(/label 已存在/)
  })

  it('locks dismissal and exposes loading semantics while creation is busy', async () => {
    const user = userEvent.setup()
    const onCancel = vi.fn()
    render(
      <NewVersionDialog
        existingLabels={[]}
        existingVersions={[]}
        busy
        onCancel={onCancel}
        onSubmit={() => {}}
      />,
    )

    const create = screen.getByRole('button', { name: '创建' })
    expect(create).toBeDisabled()
    expect(create).toHaveAttribute('aria-busy', 'true')

    await user.keyboard('{Escape}')
    fireEvent.mouseDown(screen.getByRole('dialog').parentElement!)
    expect(onCancel).not.toHaveBeenCalled()
  })
})
