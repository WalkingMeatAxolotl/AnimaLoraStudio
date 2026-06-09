/** 测试出图参数快照：落盘 PNG metadata + IndexedDB HistoryEntry.params 共用。
 *
 * 用途：
 * - 落盘 PNG `anima_params` tEXt 块 → 历史栏点击磁盘 entry 时取出回填 prefs
 * - IndexedDB HistoryEntry.params 同 shape → 未落盘的 entry 也能回填
 *
 * 设计为 prefs 视图（含 xy_draft.raw / dataset_pick），不是 daemon 接口视图：
 * 回填时直接灌回 prefs 各字段，UI 完整重建（dataset picker / xy 文本框等）。
 *
 * **不存绝对路径**（避免泄露本地文件系统结构、跨机器死链、用户挪文件失效）：
 * - LoRA：只存 name + project_id + version_id + scale；回填时按 ids→path resolve
 * - XY lora_ckpt 轴 values：存 basename（去目录、保留 .safetensors 后缀）；
 *   回填时灌回 raw 字段，用户重 submit 前需要 picker 重选定位到具体 ckpt
 * - dataset_pick：只存 projectId/versionId/name/tags，name 是相对路径无机密
 */
import type { LoraEntry, XYAxisType } from '../../../api/client'
import type { DatasetPick } from './PromptFromDatasetPicker'
import type { ProjectLora } from './types'
import type { XYAxisDraft } from './xy'

export const PARAMS_SNAPSHOT_VERSION = 1

/** snapshot 里的 LoRA 引用 —— 仅身份（name + ids），无绝对路径。 */
export interface SnapshotLora {
  /** LoRA 文件 basename（含 .safetensors 后缀）。回填 fallback 用。 */
  name: string
  scale: number
  /** picker 选的项目 / 版本；外部文件无 */
  project_id?: number | null
  version_id?: number | null
}

/** XY draft 的 snapshot 形式 —— lora_ckpt 轴 raw 已 transform 为 basename 列表。 */
export interface SnapshotXYAxis {
  axis: XYAxisType
  /** lora_ckpt 时：basename 逗号串；其它轴：原样数值串 */
  raw: string
  loraIndex: number | null
}

/** 仅 XY cell PNG 用：链回所属 XY plot 的位置（拖进 Comfy / A1111 时
 *  能识别"这是 XY 第 (xi,yi) 格"；本程序回填走 mode='single' 主路径不读它）。 */
export interface XYCellOrigin {
  xi: number
  yi: number
  xv: string | number
  yv: string | number | null
  x_axis: XYAxisType
  y_axis: XYAxisType | null
}

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
  /** 当时 mode 对应的 LoRA 列表，已转为 name+ids 形式 */
  loras: SnapshotLora[]
  /** 仅 xy 模式：prefs 视图（raw 字符串），lora_ckpt 轴 raw 已转 basename */
  xy_draft?: { x: SnapshotXYAxis; y: SnapshotXYAxis | null } | null
  /** 训练集 caption picker 选择（保留 picker UI 上下文）。
   *  name 是相对路径（如 "5_concept/0001.txt"），不含本地绝对路径。 */
  dataset_pick?: DatasetPick | null
  /** XY cell PNG 专有；composite / single PNG 永远是 undefined。
   *  forward-compat 字段，老代码读不到不影响 v2 migrate 透传。 */
  xy_origin?: XYCellOrigin | null
}

/** path → basename（去目录），保留 .safetensors 等后缀。
 *  写 metadata 时用，避免泄露本地路径结构。 */
export function loraBasename(path: string): string {
  return path.split(/[\\/]/).pop() ?? path
}

/** XY lora_ckpt 轴的 raw 字符串（逗号分隔的 ckpt 路径列表）→ basename 列表。
 *  其它轴 raw 是数字串，原样返回。 */
export function transformAxisRawForSnapshot(draft: XYAxisDraft): SnapshotXYAxis {
  if (draft.axis !== 'lora_ckpt') {
    return { axis: draft.axis, raw: draft.raw, loraIndex: draft.loraIndex }
  }
  const raw = draft.raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(loraBasename)
    .join(', ')
  return { axis: draft.axis, raw, loraIndex: draft.loraIndex }
}

/** 回填：SnapshotLora → LoraEntry，按 ids 主键、name 兜底 resolve 当前机器上的 path。
 *  resolve 失败 → path 留空（submit 时 `.filter(l => l.path.trim())` 会跳过这条），
 *  UI 上用户能看到一条 path 空的 LoRA 卡片提示重选。 */
export function resolveSnapshotLora(
  snap: SnapshotLora, projectLoras: ProjectLora[],
): LoraEntry {
  if (snap.project_id != null && snap.version_id != null) {
    const hit = projectLoras.find(
      (p) => p.projectId === snap.project_id && p.versionId === snap.version_id,
    )
    if (hit) return {
      path: hit.path, scale: snap.scale,
      project_id: snap.project_id, version_id: snap.version_id,
    }
  }
  // 兜底：按 basename 匹配（外部 LoRA 无 ids，或 picker 重建 ids 变了）
  const byName = projectLoras.find((p) => loraBasename(p.path) === snap.name)
  if (byName) return {
    path: byName.path, scale: snap.scale,
    project_id: byName.projectId, version_id: byName.versionId,
  }
  return {
    path: '', scale: snap.scale,
    project_id: snap.project_id ?? null, version_id: snap.version_id ?? null,
    // placeholder: 保留 name 让 SidebarLoras 渲染 ⚠ 提示卡片
    name: snap.name,
  }
}

/** applySnapshot 输出（决策 #8 / Arch v2 Step 3）：把 snapshot 转成"prefs 字段补丁"。
 *
 * 不直接 import GeneratePrefs（避免循环依赖）；调用方接到这个 shape 自己 spread
 * 进 setPrefs。所有"应用快照"路径（历史回填 / URL ?lora= / Stepper 跳 / 入库回写）
 * 统一走这一个函数 —— 单一入口杜绝散落分支 + 每加一个调用点漏一个字段的 bug。
 */
export interface AppliedSnapshot {
  mode: 'single' | 'xy'  // compare 视图回填映射到 xy
  prompts: string[]
  negPrompt: string
  width: number
  height: number
  steps: number
  cfgScale: number
  count: number
  seed: number
  datasetPick: DatasetPick | null
  /** 按 mode 二选一灌入 prefs.singleLoras / prefs.xyLoras */
  loras: LoraEntry[]
  /** 仅 xy 模式回填；single 时为 undefined（不动 prev.xDraft/yDraft） */
  xDraft?: SnapshotXYAxis
  yDraft?: SnapshotXYAxis | null
  /** resolve 失败的 LoRA 数量（>0 时调用方应 toast 提示重选） */
  unresolvedLoraCount: number
}

/** dataset_pick 的 project 是否还在当前机器上能 resolve（heuristic：projectLoras
 *  里有任何这个 projectId 的条目就算"活着"）。用作 applySnapshot 决定是 picker
 *  受控回填还是 fallback 到正向 prompt 的判据。 */
function isDatasetProjectAlive(
  pick: DatasetPick, projectLoras: ProjectLora[],
): boolean {
  return projectLoras.some((p) => p.projectId === pick.projectId)
}

/** dataset_pick 失败兜底：把 tags 追加到第一条 prompt 末尾。
 *  和 handleGenerate 里 `datasetSuffix` 拼法保持一致（join(', ')），避免视觉差异。 */
function mergeTagsIntoFirstPrompt(prompts: string[], tags: string[]): string[] {
  if (tags.length === 0) return prompts
  const suffix = tags.join(', ')
  const first = (prompts[0] ?? '').trimEnd()
  // 已经以 tags 结尾（用户在前一次回填后又点了一次同 entry）→ 不重复追加
  if (first.endsWith(suffix)) return prompts
  const sep = first === '' ? '' : (first.endsWith(',') ? ' ' : ', ')
  return [`${first}${sep}${suffix}`, ...prompts.slice(1)]
}

export function applySnapshot(
  snap: GenerateParamsSnapshot,
  projectLoras: ProjectLora[],
): AppliedSnapshot {
  const resolved = snap.loras.map((l) => resolveSnapshotLora(l, projectLoras))
  const unresolved = resolved.filter((l) => !l.path).length
  // compare 视图回填到 xy（compare 是 xy 子视图，无 selectedIndices 不直接进）
  const mode: 'single' | 'xy' = snap.mode === 'single' ? 'single' : 'xy'

  // dataset_pick fallback：snap 存了 dataset_pick 但 project 在当前机器上没了
  // （project 被删 / 跨机器 / projectLoras 还没 load）→ tags 拼进 prompts[0]，
  // 不再回填 picker（datasetPick=null）。用户能在正向 textarea 里直接看到具体
  // 内容，否则 picker 关着 tags 不可见、又会偷偷在 handleGenerate 里拼一次。
  let prompts = snap.prompts
  let datasetPick = snap.dataset_pick ?? null
  if (datasetPick && datasetPick.tags.length > 0 && !isDatasetProjectAlive(datasetPick, projectLoras)) {
    prompts = mergeTagsIntoFirstPrompt(prompts, datasetPick.tags)
    datasetPick = null
  }

  const applied: AppliedSnapshot = {
    mode,
    prompts,
    negPrompt: snap.negative_prompt,
    width: snap.width,
    height: snap.height,
    steps: snap.steps,
    cfgScale: snap.cfg_scale,
    count: snap.count,
    seed: snap.seed,
    datasetPick,
    loras: resolved,
    unresolvedLoraCount: unresolved,
  }
  if (mode === 'xy' && snap.xy_draft) {
    applied.xDraft = snap.xy_draft.x
    applied.yDraft = snap.xy_draft.y
  }
  return applied
}

// ---------------------------------------------------------------------------
// XY cell single-snapshot 物化（落盘 cell PNG metadata 用）
// ---------------------------------------------------------------------------

/** XY snapshot 按 (xi, yi) 物化成单格 single-snapshot。
 *
 * 用途：落盘 cell PNG 时每张要带"如果只看这一张是什么参数"的快照 — 拖进
 * Comfy / A1111 能识别 steps/cfg/seed/lora 全套；本程序回填走 mode='single'
 * 主路径（同 single 出图 PNG）。
 *
 * 轴语义（daemon `_apply_axis` 对齐）：
 * - `steps` / `cfg_scale`：顶层标量字段覆盖
 * - `lora_scale`：全 LoRA 共用一个 scale（不按 loraIndex 单独改）
 * - `lora_ckpt`：仅 `loras[loraIndex]` 的 name 改成 cell value 的 basename，
 *   原 ids 失效（ckpt 换了，project_id/version_id 不再准）
 *
 * 输出额外带 `xy_origin: {xi, yi, xv, yv, x_axis, y_axis}` 链回所属 XY plot。
 */
export function buildCellSnapshot(
  xy: GenerateParamsSnapshot,
  cellPos: { xi: number; yi: number },
  axes: {
    x: { axis: XYAxisType; loraIndex: number | null; value: string | number }
    y: { axis: XYAxisType; loraIndex: number | null; value: string | number } | null
  },
): GenerateParamsSnapshot {
  const out: GenerateParamsSnapshot = {
    ...xy,
    mode: 'single',
    xy_draft: null,
    // 浅克隆 loras 数组让下面 mutate 不污染 xy.loras
    loras: xy.loras.map((l) => ({ ...l })),
  }
  applyAxisToCell(out, axes.x.axis, axes.x.value, axes.x.loraIndex)
  if (axes.y) {
    applyAxisToCell(out, axes.y.axis, axes.y.value, axes.y.loraIndex)
  }
  out.xy_origin = {
    xi: cellPos.xi,
    yi: cellPos.yi,
    xv: axes.x.value,
    yv: axes.y?.value ?? null,
    x_axis: axes.x.axis,
    y_axis: axes.y?.axis ?? null,
  }
  return out
}

function applyAxisToCell(
  snap: GenerateParamsSnapshot,
  axis: XYAxisType,
  value: string | number,
  loraIndex: number | null,
): void {
  switch (axis) {
    case 'steps':
      snap.steps = Math.trunc(Number(value))
      return
    case 'cfg_scale':
      snap.cfg_scale = Number(value)
      return
    case 'lora_scale': {
      const scale = Number(value)
      snap.loras = snap.loras.map((l) => ({ ...l, scale }))
      return
    }
    case 'lora_ckpt': {
      if (loraIndex == null || !snap.loras[loraIndex]) return
      // value 已经是 basename（snapshot 写时 transformAxisRawForSnapshot 处理过；
      // 但 buildCellSnapshot 也可能被传 raw path，统一过一遍 basename）
      const name = loraBasename(String(value))
      snap.loras = snap.loras.map((l, i) =>
        i === loraIndex ? { ...l, name, project_id: null, version_id: null } : l,
      )
      return
    }
  }
}
