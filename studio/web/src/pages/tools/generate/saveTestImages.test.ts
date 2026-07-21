import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { GenerateParamsSnapshot } from './paramsSnapshot'
import { saveSingleSamples } from './saveTestImages'

const snapshot: GenerateParamsSnapshot = {
  schema_version: 1,
  mode: 'single',
  prompts: ['raw prompt'],
  negative_prompt: '',
  width: 1024,
  height: 1024,
  steps: 25,
  cfg_scale: 4,
  count: 1,
  seed: 7,
  loras: [],
}

describe('saveSingleSamples metadata identity', () => {
  beforeEach(() => {
    vi.unstubAllGlobals()
  })

  it('sends the daemon source filename so the server selects the actual prompt index', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        blob: async () => new Blob(['png'], { type: 'image/png' }),
      })
      .mockImplementationOnce(async (_url: string, init: RequestInit) => {
        const form = init.body as FormData
        expect(form.get('source_filename')).toBe('gen_0001_p1_c0_s7.png')
        expect(form.get('task_id')).toBe('42')
        return {
          ok: true,
          json: async () => ({ path: 'saved.png', index: 1, filename: 'saved.png' }),
        }
      })
    vi.stubGlobal('fetch', fetchMock)

    await expect(saveSingleSamples(
      42,
      ['gen_0001_p1_c0_s7.png'],
      snapshot,
    )).resolves.toEqual(['saved.png'])
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })
})
