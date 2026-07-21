import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createRef } from 'react'
import { describe, expect, it, vi } from 'vitest'
import '../i18n'

vi.mock('../api/client', () => ({
  api: { getModelPathChoices: vi.fn() },
}))

import { api } from '../api/client'
import ModelPathPicker from './ModelPathPicker'

const CHOICES = {
  choices: {
    transformer_path: [
      { label: 'anima-base-v1.0.safetensors', path: 'G:/m/anima-base-v1.0.safetensors', group: 'official', note: 'latest' },
      { label: 'anima-preview2.safetensors', path: 'G:/m/anima-preview2.safetensors', group: 'official', note: '' },
      { label: 'my-finetune.safetensors', path: 'G:/other/my-finetune.safetensors', group: 'custom', note: '' },
    ],
    t5_tokenizer_path: [],
  },
} as never

function renderPicker(field: string, onChange = vi.fn(), value = '') {
  const anchorRef = createRef<HTMLElement>()
  render(
    <ModelPathPicker
      field={field}
      family="anima"
      value={value}
      onChange={onChange}
      onClose={() => {}}
      anchorRef={anchorRef}
    />,
  )
  return onChange
}

describe('ModelPathPicker', () => {
  it('picking a choice writes back its absolute path', async () => {
    vi.mocked(api.getModelPathChoices).mockResolvedValue(CHOICES)
    const onChange = renderPicker('transformer_path')

    const row = await screen.findByText('my-finetune.safetensors')
    await userEvent.click(row)
    expect(onChange).toHaveBeenCalledWith('G:/other/my-finetune.safetensors')
  })

  it('groups choices and only requests the current field family', async () => {
    vi.mocked(api.getModelPathChoices).mockResolvedValue(CHOICES)
    renderPicker('transformer_path')

    await screen.findByText('anima-base-v1.0.safetensors')
    expect(api.getModelPathChoices).toHaveBeenCalledWith('anima')
    // group / note 是翻译 id，渲染出来的是译文而不是 id 本身
    expect(screen.getByText('官方')).toBeInTheDocument()
    expect(screen.getByText('自定义')).toBeInTheDocument()
    expect(screen.getByText('最新')).toBeInTheDocument()
  })

  it('shows an empty hint when the family has no ready asset for the field', async () => {
    vi.mocked(api.getModelPathChoices).mockResolvedValue(CHOICES)
    renderPicker('t5_tokenizer_path')

    await waitFor(() =>
      expect(screen.getByText(/还没有已下载的可选模型/)).toBeInTheDocument(),
    )
  })
})
