/** 测试出图参数快照：落盘 sidecar + IndexedDB HistoryEntry.params 共用。
 *
 * 用途：
 * - 落盘 image_N.json sidecar 时存 → 历史栏点击磁盘 entry 时取出回填 prefs
 * - IndexedDB HistoryEntry.params 同 shape → 未落盘的 entry 也能回填
 *
 * 设计为 prefs 视图（含 xy_draft.raw / dataset_pick），不是 daemon 接口视图：
 * 回填时直接灌回 prefs 各字段，UI 完整重建（dataset picker / xy 文本框等）。
 */
import type { LoraEntry } from '../../../api/client'
import type { DatasetPick } from './PromptFromDatasetPicker'
import type { XYAxisDraft } from './xy'

export const PARAMS_SNAPSHOT_VERSION = 1

export interface GenerateParamsSnapshot {
  schema_version: number
  /** 当时的 mode；回填时按 mode 决定灌 singleLoras 还是 xyLoras + xDraft/yDraft */
  mode: 'single' | 'xy' | 'compare'
  /** prefs.prompts 原文（未与 datasetPick.tags 合并） */
  prompts: string[]
  negative_prompt: string
  width: number
  height: number
  steps: number
  cfg_scale: number
  /** xy 模式下 daemon 端强制 1，仅 single 有意义 */
  count: number
  seed: number
  /** 当时 mode 对应的 LoRA 列表（single→singleLoras，xy→xyLoras） */
  lora_configs: LoraEntry[]
  /** 仅 xy 模式：prefs 视图（含 raw 字符串），回填时直接灌回 xDraft/yDraft；
   *  重 submit 时前端从这里重 buildXYMatrix（不冗余存 daemon xy_matrix） */
  xy_draft?: { x: XYAxisDraft; y: XYAxisDraft | null } | null
  /** 训练集 caption picker 选择（保留 picker UI 上下文） */
  dataset_pick?: DatasetPick | null
}
