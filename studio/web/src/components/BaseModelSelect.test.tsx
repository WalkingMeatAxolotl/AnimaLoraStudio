import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import '../i18n'

vi.mock('../api/client', () => ({
  api: { getModelsCatalog: vi.fn() },
}))

import { api } from '../api/client'
import BaseModelSelect from './BaseModelSelect'

const catalog = {
  anima_main: {
    variants: [
      { variant: '1.0', exists: true },
      { variant: 'preview2', exists: false },
    ],
    custom: [{ path: 'G:/m/ft.safetensors', name: 'ft.safetensors', exists: true }],
    selected: '1.0',
  },
  krea2_main: {
    variants: [
      { variant: 'raw', exists: true, purpose: 'training' },
      { variant: 'turbo', exists: true, purpose: 'inference' },
    ],
    custom: [],
    selected: 'raw',
  },
} as never

function optionValues() {
  return Array.from(
    (screen.getByRole('combobox') as HTMLSelectElement).options,
  ).map((o) => o.value)
}

describe('BaseModelSelect per family（多模型 P4-4）', () => {
  it('default family lists anima variants + custom, hiding missing files', async () => {
    vi.mocked(api.getModelsCatalog).mockResolvedValue(catalog)
    render(<BaseModelSelect value={null} onChange={() => {}} />)
    await screen.findByRole('option', { name: '1.0' })
    expect(optionValues()).toEqual(['1.0', 'G:/m/ft.safetensors'])
  })

  it('krea2 family lists raw/turbo with purpose badges', async () => {
    vi.mocked(api.getModelsCatalog).mockResolvedValue(catalog)
    render(<BaseModelSelect value={null} onChange={() => {}} family="krea2" />)
    await screen.findByRole('option', { name: /raw/ })
    expect(optionValues()).toEqual(['raw', 'turbo'])
    // purpose 徽标（raw=训练 / turbo=推理）出现在 label 里
    expect(screen.getByRole('option', { name: /turbo · 推理/ })).toBeInTheDocument()
    // defaultValue 跟随该族 selected
    expect((screen.getByRole('combobox') as HTMLSelectElement).value).toBe('raw')
  })

  it('anima variants show layer count + community group + custom arch', async () => {
    vi.mocked(api.getModelsCatalog).mockResolvedValue({
      anima_main: {
        variants: [
          { variant: '1.0', exists: true, group: 'official', arch: { num_blocks: 28, model_channels: 2048, param_count: 1 } },
          { variant: '2.9b-preview-v1', label: 'Anima-2.9B preview-v1', exists: true, group: 'community', arch: { num_blocks: 40, model_channels: 2048, param_count: 1 } },
        ],
        custom: [{ path: 'G:/m/ft.safetensors', name: 'ft.safetensors', exists: true, arch: null }],
        selected: '1.0',
      },
    } as never)
    render(<BaseModelSelect value={null} onChange={() => {}} />)
    await screen.findByRole('option', { name: /1\.0/ })
    // 层数进 label（LoRA 互换性的判据）；显示名替代 variant key；无 arch 的 custom 不带后缀
    expect(screen.getByRole('option', { name: '1.0 · 28 层' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Anima-2.9B preview-v1 · 40 层' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'ft.safetensors' })).toBeInTheDocument()
    // 三组 → optgroup 分组（官方 / 第三方 / 自定义）
    const groups = Array.from(document.querySelectorAll('optgroup')).map((g) => g.label)
    expect(groups).toEqual(['官方', '第三方', '自定义'])
    expect(optionValues()).toEqual(['1.0', '2.9b-preview-v1', 'G:/m/ft.safetensors'])
  })
})
