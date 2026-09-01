import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import '../i18n'
import ExportBundleDialog from './ExportBundleDialog'

describe('ExportBundleDialog', () => {
  it('uses the shared labelled modal and submits the default bundle options', async () => {
    const user = userEvent.setup()
    const onConfirm = vi.fn()
    render(<ExportBundleDialog onConfirm={onConfirm} onCancel={() => {}} />)

    expect(screen.getByRole('dialog', { name: '导出内容选择' })).toBeInTheDocument()
    expect(screen.getAllByRole('checkbox')[0]).toHaveClass('form-checkbox')
    await user.click(screen.getByRole('button', { name: '导出' }))
    expect(onConfirm).toHaveBeenCalledWith({
      train: true,
      trainCaptions: true,
      reg: false,
      regCaptions: false,
      includeConfig: false,
      trainLatentCache: false,
      regLatentCache: false,
      trainMasks: false,
      destination: 'download',
    })
  })

  it('requires at least one content group before export', async () => {
    const user = userEvent.setup()
    render(<ExportBundleDialog onConfirm={() => {}} onCancel={() => {}} />)

    await user.click(screen.getByRole('checkbox', { name: '训练集' }))
    expect(screen.getByRole('button', { name: '导出' })).toBeDisabled()
    expect(screen.getByText('至少选择一项导出内容')).toBeInTheDocument()

    await user.click(screen.getByRole('checkbox', { name: /训练配置/ }))
    expect(screen.getByRole('button', { name: '导出' })).toBeEnabled()
  })

  it('preserves cancellation behavior', async () => {
    const user = userEvent.setup()
    const onCancel = vi.fn()
    render(<ExportBundleDialog onConfirm={() => {}} onCancel={onCancel} />)

    await user.click(screen.getByRole('button', { name: '取消' }))
    expect(onCancel).toHaveBeenCalledTimes(1)
  })
})
