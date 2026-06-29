import { describe, expect, it } from 'vitest'
import { extractReleaseEntries } from './Announcements'

const BODY = `本版总览一句话

### 新增

- **多分辨率训练：桶分布预览（#296）**
  官方示范 LoRA 常用多个分辨率混训。

  - 子要点不算 highlight
- **CLTagger V2（#297）**
  细节。

### 修复

- **修了个崩溃（#299, #305）**
- **又一条**
`

describe('extractReleaseEntries', () => {
  it('抽顶层加粗首句、带所属分组 kind、剥 PR 号、跳过子要点', () => {
    expect(extractReleaseEntries(BODY)).toEqual([
      { kind: 'added', summary: '多分辨率训练：桶分布预览' },
      { kind: 'added', summary: 'CLTagger V2' },
      { kind: 'fixed', summary: '修了个崩溃' },
      { kind: 'fixed', summary: '又一条' },
    ])
  })

  it('英文分组标题也映射 kind', () => {
    expect(extractReleaseEntries('### Fixed\n\n- **fixed a crash (#299)**')).toEqual([
      { kind: 'fixed', summary: 'fixed a crash' },
    ])
  })

  it('无加粗要点 → 空', () => {
    expect(extractReleaseEntries('### 新增\n\n- 没加粗的要点\n普通段落')).toEqual([])
  })
})
