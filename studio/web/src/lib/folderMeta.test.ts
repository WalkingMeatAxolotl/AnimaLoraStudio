import { describe, expect, it } from 'vitest'
import { parseFolderMeta } from './folderMeta'

describe('parseFolderMeta', () => {
  it('parses [Npx_][R_]label (mirror of Python _parse_folder_meta)', () => {
    expect(parseFolderMeta('1024px_2_data')).toEqual({ reso: 1024, repeat: 2, label: 'data' })
    expect(parseFolderMeta('768px_concept')).toEqual({ reso: 768, repeat: 1, label: 'concept' })
    expect(parseFolderMeta('1024px_data')).toEqual({ reso: 1024, repeat: 1, label: 'data' })
    expect(parseFolderMeta('5_concept')).toEqual({ reso: null, repeat: 5, label: 'concept' })
    expect(parseFolderMeta('concept')).toEqual({ reso: null, repeat: 1, label: 'concept' })
  })

  it('snaps resolution to /64 (half-up, matches Python) and clamps to [256, 4096]', () => {
    expect(parseFolderMeta('1000px_data').reso).toBe(1024) // round(1000/64)*64
    expect(parseFolderMeta('288px_data').reso).toBe(320) // 288=64×4.5 → half-up to 5×64
    expect(parseFolderMeta('100px_data').reso).toBe(256) // clamp low
    expect(parseFolderMeta('9000px_data').reso).toBe(4096) // clamp high
  })
})
