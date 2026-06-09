/** paramsSnapshot 单测：applySnapshot reducer + resolveSnapshotLora 三层 fallback。
 *
 * 决策 #8（单一应用快照入口） / plan §3 LoRA placeholder 兜底。
 */
import { describe, expect, it } from 'vitest'
import {
  applySnapshot, loraBasename, resolveSnapshotLora, transformAxisRawForSnapshot,
  type GenerateParamsSnapshot, type SnapshotLora,
} from './paramsSnapshot'
import type { ProjectLora } from './types'
import type { XYAxisDraft } from './xy'

const projects: ProjectLora[] = [
  {
    projectId: 1, projectTitle: 'cute_chibi',
    versionId: 11, versionLabel: 'v3', status: 'training',
    path: '/loras/cute_chibi/v3.safetensors', createdAt: 100,
  },
  {
    projectId: 2, projectTitle: 'noir',
    versionId: 21, versionLabel: 'v1', status: 'completed',
    path: '/loras/noir/v1.safetensors', createdAt: 200,
  },
]

function snapshot(overrides: Partial<GenerateParamsSnapshot> = {}): GenerateParamsSnapshot {
  return {
    schema_version: 1,
    mode: 'single',
    prompts: ['1girl'],
    negative_prompt: 'blurry',
    width: 1024,
    height: 768,
    steps: 20,
    cfg_scale: 7,
    count: 1,
    seed: 42,
    loras: [],
    xy_draft: null,
    dataset_pick: null,
    ...overrides,
  }
}

describe('loraBasename', () => {
  it('strips POSIX path', () => {
    expect(loraBasename('/a/b/c/my.safetensors')).toBe('my.safetensors')
  })
  it('strips Windows path', () => {
    expect(loraBasename('G:\\a\\b\\my.safetensors')).toBe('my.safetensors')
  })
  it('no separator → return whole', () => {
    expect(loraBasename('only.safetensors')).toBe('only.safetensors')
  })
})

describe('transformAxisRawForSnapshot', () => {
  it('lora_ckpt: raw paths → basename list', () => {
    const draft: XYAxisDraft = {
      axis: 'lora_ckpt',
      raw: '/a/b/step_1000.safetensors, /a/b/step_2000.safetensors',
      loraIndex: 0,
    }
    expect(transformAxisRawForSnapshot(draft).raw).toBe('step_1000.safetensors, step_2000.safetensors')
  })
  it('non lora_ckpt axes: raw 原样', () => {
    const draft: XYAxisDraft = { axis: 'steps', raw: '10, 20, 30', loraIndex: null }
    expect(transformAxisRawForSnapshot(draft).raw).toBe('10, 20, 30')
  })
})

describe('resolveSnapshotLora — 三层 fallback', () => {
  it('1. ids 命中 → 返 projectLoras path + 保留 snapshot 的 scale/ids', () => {
    const snap: SnapshotLora = {
      name: 'cute_chibi.safetensors', scale: 0.8,
      project_id: 1, version_id: 11,
    }
    const r = resolveSnapshotLora(snap, projects)
    expect(r.path).toBe('/loras/cute_chibi/v3.safetensors')
    expect(r.scale).toBe(0.8)
    expect(r.project_id).toBe(1)
    expect(r.version_id).toBe(11)
  })

  it('2. ids 未命中但 basename 匹配 → 用 projectLoras 的 ids/path', () => {
    const snap: SnapshotLora = {
      name: 'v3.safetensors', scale: 1.0,
      project_id: 999, version_id: 999,  // 不存在
    }
    const r = resolveSnapshotLora(snap, projects)
    expect(r.path).toBe('/loras/cute_chibi/v3.safetensors')
    expect(r.project_id).toBe(1)  // 用了 projectLoras 的，不是 snapshot 的 999
    expect(r.version_id).toBe(11)
  })

  it('3. 都不命中 → placeholder：path 空 + name 保留 + 原 ids', () => {
    const snap: SnapshotLora = {
      name: 'gone.safetensors', scale: 0.5,
      project_id: 999, version_id: 999,
    }
    const r = resolveSnapshotLora(snap, projects)
    expect(r.path).toBe('')
    expect(r.name).toBe('gone.safetensors')  // ← placeholder UI 渲染会读这个字段
    expect(r.project_id).toBe(999)
    expect(r.version_id).toBe(999)
    expect(r.scale).toBe(0.5)
  })

  it('snapshot 无 ids 时按 name 兜底', () => {
    const snap: SnapshotLora = {
      name: 'v3.safetensors', scale: 1.0,
    }
    const r = resolveSnapshotLora(snap, projects)
    expect(r.path).toBe('/loras/cute_chibi/v3.safetensors')
  })
})

describe('applySnapshot', () => {
  it('single 模式：所有字段灌入 + loras 替换 singleLoras', () => {
    const snap = snapshot({
      mode: 'single',
      seed: 42,
      loras: [{ name: 'v3.safetensors', scale: 0.7, project_id: 1, version_id: 11 }],
    })
    const r = applySnapshot(snap, projects)
    expect(r.mode).toBe('single')
    expect(r.seed).toBe(42)
    expect(r.loras).toHaveLength(1)
    expect(r.loras[0].path).toBe('/loras/cute_chibi/v3.safetensors')
    expect(r.unresolvedLoraCount).toBe(0)
    expect(r.xDraft).toBeUndefined()  // single 不灌 xDraft
    expect(r.yDraft).toBeUndefined()
  })

  it('compare 模式映射到 xy（子视图无 selectedIndices 不能直接进）', () => {
    const snap = snapshot({ mode: 'compare' })
    expect(applySnapshot(snap, projects).mode).toBe('xy')
  })

  it('xy 模式：xDraft + yDraft 灌入', () => {
    const snap = snapshot({
      mode: 'xy',
      xy_draft: {
        x: { axis: 'cfg_scale', raw: '4, 5, 6', loraIndex: null },
        y: { axis: 'steps', raw: '10, 20', loraIndex: null },
      },
    })
    const r = applySnapshot(snap, projects)
    expect(r.mode).toBe('xy')
    expect(r.xDraft?.axis).toBe('cfg_scale')
    expect(r.xDraft?.raw).toBe('4, 5, 6')
    expect(r.yDraft?.axis).toBe('steps')
  })

  it('未 resolve 的 LoRA → unresolvedLoraCount > 0', () => {
    const snap = snapshot({
      loras: [
        { name: 'gone1.safetensors', scale: 1, project_id: 99, version_id: 99 },
        { name: 'v3.safetensors', scale: 1, project_id: 1, version_id: 11 },
        { name: 'gone2.safetensors', scale: 1, project_id: 88, version_id: 88 },
      ],
    })
    const r = applySnapshot(snap, projects)
    expect(r.unresolvedLoraCount).toBe(2)
    expect(r.loras[0].path).toBe('')
    expect(r.loras[1].path).toBe('/loras/cute_chibi/v3.safetensors')
    expect(r.loras[2].path).toBe('')
  })
})
