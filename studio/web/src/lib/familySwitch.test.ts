import { describe, expect, it, vi } from 'vitest'

vi.mock('../api/client', () => ({
  api: { switchModelFamily: vi.fn() },
}))

import { api } from '../api/client'
import { confirmFamilySwitch } from './familySwitch'

const t = ((key: string, opts?: Record<string, unknown>) => {
  if (opts && 'changes' in opts) return `msg:${opts.family}\n${opts.changes}`
  if (opts && 'family' in opts) return `msg:${opts.family}`
  return key
}) as never

const switched = { model_family: 'krea2', sample_sampler_name: 'euler' }
const response = {
  config: switched,
  changes: [
    { field: 'model_family', from: 'anima', to: 'krea2' },
    { field: 'sample_sampler_name', from: 'er_sde', to: 'euler' },
    { field: 't5_tokenizer_path', from: '/models/t5', to: '' },
  ],
}

describe('confirmFamilySwitch', () => {
  it('applies backend-computed config after user confirms', async () => {
    vi.mocked(api.switchModelFamily).mockResolvedValue(response)
    const confirm = vi.fn().mockResolvedValue(true)
    const result = await confirmFamilySwitch(
      'krea2', { model_family: 'anima' }, confirm, t)
    expect(api.switchModelFamily).toHaveBeenCalledWith(
      'krea2', { model_family: 'anima' })
    expect(result).toBe(switched)
    // 确认文案含变更清单（不含 model_family 自身行；空值有占位符）
    const msg = confirm.mock.calls[0][0] as string
    expect(msg).toContain('euler')
    expect(msg).toContain('familySwitch.empty')
    expect(msg).not.toContain('Model Family:')
  })

  it('returns null when user cancels — caller keeps old values', async () => {
    vi.mocked(api.switchModelFamily).mockResolvedValue(response)
    const confirm = vi.fn().mockResolvedValue(false)
    const result = await confirmFamilySwitch(
      'krea2', { model_family: 'anima' }, confirm, t)
    expect(result).toBeNull()
  })
})
