import { describe, expect, it } from 'vitest'
import { availabilityOverrides } from './Tagging'

// issue #477：可用性检查必须带上页面的模型版本覆盖，helper 负责挑出
// 「与全局默认不同、且影响可用性」的字段。
describe('availabilityOverrides', () => {
  const defaults = {
    model_id: 'cella110n/cl_tagger',
    model_path: 'cl_tagger_1_02/model.onnx',
    tag_mapping_path: 'cl_tagger_1_02/tag_mapping.json',
    threshold_general: 0.35,
  }

  it('form / defaults 未加载时不产生覆盖', () => {
    expect(availabilityOverrides(null, defaults, ['model_id'])).toBeUndefined()
    expect(availabilityOverrides(defaults, null, ['model_id'])).toBeUndefined()
  })

  it('与默认一致时不产生覆盖', () => {
    expect(availabilityOverrides({ ...defaults }, defaults, ['model_id', 'model_path'])).toBeUndefined()
  })

  it('只挑列出的且不同的字段（改版本 → 三元组里变了的进覆盖，阈值不进）', () => {
    const form = {
      ...defaults,
      model_id: 'cella110n/cl_tagger_v2',
      model_path: 'v2_01a/model.onnx',
      tag_mapping_path: 'v2_01a/model_vocabulary.json',
      threshold_general: 0.5,
    }
    expect(
      availabilityOverrides(form, defaults, ['model_id', 'model_path', 'tag_mapping_path']),
    ).toEqual({
      model_id: 'cella110n/cl_tagger_v2',
      model_path: 'v2_01a/model.onnx',
      tag_mapping_path: 'v2_01a/model_vocabulary.json',
    })
  })
})
