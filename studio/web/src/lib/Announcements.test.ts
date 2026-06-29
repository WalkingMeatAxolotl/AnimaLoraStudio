import { describe, expect, it } from 'vitest'
import { extractReleaseHighlights } from './Announcements'

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

describe('extractReleaseHighlights', () => {
  it('抽顶层加粗首句、剥 PR 号、跳过子要点', () => {
    expect(extractReleaseHighlights(BODY, 5)).toEqual([
      '多分辨率训练：桶分布预览',
      'CLTagger V2',
      '修了个崩溃',
      '又一条',
    ])
  })

  it('受 max 限制', () => {
    expect(extractReleaseHighlights(BODY, 2)).toEqual([
      '多分辨率训练：桶分布预览',
      'CLTagger V2',
    ])
  })

  it('无加粗要点 → 空', () => {
    expect(extractReleaseHighlights('### 新增\n\n- 没加粗的要点\n普通段落')).toEqual([])
  })
})
