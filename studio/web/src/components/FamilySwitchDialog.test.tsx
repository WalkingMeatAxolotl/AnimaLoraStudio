import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import '../i18n'

vi.mock('../api/client', () => ({
  api: { switchModelFamily: vi.fn() },
}))

import { api } from '../api/client'
import FamilySwitchDialog from './FamilySwitchDialog'

const switched = { model_family: 'krea2', sample_sampler_name: 'euler' }
const response = {
  config: switched,
  changes: [
    { field: 'model_family', from: 'anima', to: 'krea2' },
    { field: 'transformer_path', from: 'G:/models/anima.safetensors', to: 'G:/models/krea2-raw-bf16.safetensors' },
    { field: 't5_tokenizer_path', from: 'G:/models/t5_tokenizer', to: '' },
    { field: 'sample_sampler_name', from: 'er_sde', to: 'euler' },
    { field: 'shuffle_caption', from: true, to: false },
  ],
}

describe('FamilySwitchDialog', () => {
  it('renders grouped changes and applies switched config on confirm', async () => {
    vi.mocked(api.switchModelFamily).mockResolvedValue(response)
    const onApply = vi.fn()
    render(
      <FamilySwitchDialog
        target="krea2"
        config={{ model_family: 'anima' }}
        onApply={onApply}
        onCancel={() => {}}
      />,
    )
    // 路径区（等宽新旧对照）与参数区分组渲染；model_family 自身行不列出
    await screen.findByText('G:/models/krea2-raw-bf16.safetensors')
    expect(screen.getByText('Transformer Path')).toBeInTheDocument()
    expect(screen.getByText('Sample Sampler Name')).toBeInTheDocument()
    expect(screen.queryByText('Model Family')).not.toBeInTheDocument()
    // 空值占位 + 布尔值本地化
    expect(screen.getByText('（空）')).toBeInTheDocument()
    expect(screen.getByText(/是/)).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: '切换' }))
    expect(onApply).toHaveBeenCalledWith(switched)
  })

  it('cancel keeps old values and never applies', async () => {
    vi.mocked(api.switchModelFamily).mockResolvedValue(response)
    const onApply = vi.fn()
    const onCancel = vi.fn()
    render(
      <FamilySwitchDialog
        target="krea2"
        config={{ model_family: 'anima' }}
        onApply={onApply}
        onCancel={onCancel}
      />,
    )
    await screen.findByText('Transformer Path')
    await userEvent.click(screen.getByRole('button', { name: '取消' }))
    expect(onCancel).toHaveBeenCalled()
    expect(onApply).not.toHaveBeenCalled()
  })

  it('disables confirm and surfaces error when preview fails', async () => {
    vi.mocked(api.switchModelFamily).mockRejectedValue(new Error('boom'))
    render(
      <FamilySwitchDialog
        target="krea2"
        config={{ model_family: 'anima' }}
        onApply={() => {}}
        onCancel={() => {}}
      />,
    )
    await waitFor(() => {
      expect(screen.getByText(/boom/)).toBeInTheDocument()
    })
    expect(screen.getByRole('button', { name: '切换' })).toBeDisabled()
  })
})
