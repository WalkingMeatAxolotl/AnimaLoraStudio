/** paramsSnapshot 单测：applySnapshot reducer + resolveSnapshotLora 三层 fallback。
 *
 * 决策 #8（单一应用快照入口） / plan §3 LoRA placeholder 兜底。
 */
import { describe, expect, it } from 'vitest'
import {
  applySnapshot, buildCellSnapshot, loraBasename, resolveSnapshotLora, transformAxisRawForSnapshot,
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

  it('base_model 回填：显式值原样，缺省 → null', () => {
    expect(applySnapshot(snapshot({ base_model: 'preview2' }), projects).baseModel).toBe('preview2')
    expect(applySnapshot(snapshot({ base_model: '/loras/ft.safetensors' }), projects).baseModel)
      .toBe('/loras/ft.safetensors')
    // 老快照无此字段 → null（沿用设置页默认底模）
    expect(applySnapshot(snapshot(), projects).baseModel).toBeNull()
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

  it('dataset_pick fallback：project 在 projectLoras 还在 → datasetPick 保留', () => {
    const snap = snapshot({
      dataset_pick: {
        projectId: 1, versionId: 11,
        name: '0001.txt', tags: ['tag-a', 'tag-b'],
      },
    })
    const r = applySnapshot(snap, projects)
    expect(r.datasetPick?.projectId).toBe(1)
    expect(r.prompts).toEqual(['1girl'])  // 没污染 prompts
  })

  it('dataset_pick fallback：project 在 projectLoras 找不到 → tags 拼进第一条 prompt + datasetPick=null', () => {
    const snap = snapshot({
      prompts: ['base prompt'],
      dataset_pick: {
        projectId: 999, versionId: 88,  // 不在 projects 列表里
        name: '0001.txt', tags: ['fall-tag-1', 'fall-tag-2'],
      },
    })
    const r = applySnapshot(snap, projects)
    expect(r.datasetPick).toBeNull()
    expect(r.prompts[0]).toBe('base prompt, fall-tag-1, fall-tag-2')
  })

  it('dataset_pick fallback：第一条 prompt 空 → 直接是 tags 串', () => {
    const snap = snapshot({
      prompts: [''],
      dataset_pick: {
        projectId: 999, versionId: 88,
        name: '0001.txt', tags: ['x', 'y'],
      },
    })
    const r = applySnapshot(snap, projects)
    expect(r.prompts[0]).toBe('x, y')
  })

  it('dataset_pick fallback：tags 已在第一条末尾 → 不重复追加（防双击同一历史 entry）', () => {
    const snap = snapshot({
      prompts: ['base, x, y'],
      dataset_pick: {
        projectId: 999, versionId: 88,
        name: '0001.txt', tags: ['x', 'y'],
      },
    })
    const r = applySnapshot(snap, projects)
    expect(r.prompts[0]).toBe('base, x, y')
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

describe('buildCellSnapshot', () => {
  const baseXy = snapshot({
    mode: 'xy',
    steps: 20,
    cfg_scale: 5,
    loras: [{ name: 'a.safetensors', scale: 0.5 }, { name: 'b.safetensors', scale: 0.7 }],
  })

  it('steps axis: 顶层 steps 覆盖；mode → single；xy_draft → null；xy_origin 记位置', () => {
    const cell = buildCellSnapshot(baseXy, { xi: 2, yi: 0 }, {
      x: { axis: 'steps', loraIndex: null, value: 30 },
      y: null,
    })
    expect(cell.mode).toBe('single')
    expect(cell.steps).toBe(30)
    expect(cell.cfg_scale).toBe(5)  // 未被轴改
    expect(cell.xy_draft).toBeNull()
    expect(cell.xy_origin).toEqual({
      xi: 2, yi: 0, xv: 30, yv: null, x_axis: 'steps', y_axis: null,
    })
  })

  it('cfg_scale axis: 顶层 cfg_scale 覆盖（字符串 input 也接受）', () => {
    const cell = buildCellSnapshot(baseXy, { xi: 1, yi: 0 }, {
      x: { axis: 'cfg_scale', loraIndex: null, value: '7.5' },
      y: null,
    })
    expect(cell.cfg_scale).toBe(7.5)
    expect(cell.steps).toBe(20)
  })

  it('lora_scale axis: 全 LoRA 共用 cell value（不按 loraIndex 单独改）', () => {
    const cell = buildCellSnapshot(baseXy, { xi: 0, yi: 0 }, {
      x: { axis: 'lora_scale', loraIndex: null, value: 0.9 },
      y: null,
    })
    expect(cell.loras).toEqual([
      { name: 'a.safetensors', scale: 0.9 },
      { name: 'b.safetensors', scale: 0.9 },
    ])
  })

  it('lora_ckpt axis: 仅指定 loraIndex 的 name 改成 basename，原 ids 失效', () => {
    const cell = buildCellSnapshot(baseXy, { xi: 1, yi: 0 }, {
      x: { axis: 'lora_ckpt', loraIndex: 0, value: '/some/path/new.safetensors' },
      y: null,
    })
    expect(cell.loras[0]).toEqual({
      name: 'new.safetensors', scale: 0.5, project_id: null, version_id: null,
    })
    // loras[1] 不动
    expect(cell.loras[1].name).toBe('b.safetensors')
  })

  it('2D: x + y 双轴都物化', () => {
    const cell = buildCellSnapshot(baseXy, { xi: 1, yi: 2 }, {
      x: { axis: 'steps', loraIndex: null, value: 25 },
      y: { axis: 'cfg_scale', loraIndex: null, value: 8.0 },
    })
    expect(cell.steps).toBe(25)
    expect(cell.cfg_scale).toBe(8.0)
    expect(cell.xy_origin?.yi).toBe(2)
    expect(cell.xy_origin?.y_axis).toBe('cfg_scale')
  })

  it('不污染原 XY snapshot（loras 是深复制）', () => {
    const before = JSON.parse(JSON.stringify(baseXy.loras))
    buildCellSnapshot(baseXy, { xi: 0, yi: 0 }, {
      x: { axis: 'lora_scale', loraIndex: null, value: 0.1 },
      y: null,
    })
    expect(baseXy.loras).toEqual(before)
  })
})
