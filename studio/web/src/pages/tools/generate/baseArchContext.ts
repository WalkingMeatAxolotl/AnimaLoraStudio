import { createContext, useContext } from 'react'
import type { LoraCkpt } from '../../../api/client'

/** 当前出图底模的 DiT 层数（catalog arch，header 探测）；未知为 null。
 *
 *  Generate 页在 sidebar 外层 Provide；InlineLoraPicker 用它给每个 LoRA chip 标
 *  「N 层」并按 lora_compat 契约的同一规则标不匹配（后端 apply 时真正拒绝 / 告警，
 *  前端只做预检展示，不自己再算一套语义）。 */
export const BaseNumBlocksContext = createContext<number | null>(null)

export function useBaseNumBlocks(): number | null {
  return useContext(BaseNumBlocksContext)
}

export type LoraCompatLevel = 'ok' | 'warn' | 'reject' | 'unknown'

/** 前端预检（与后端 lora_compat.check_lora_compat 同规则）：
 *  - 元数据确证的层数 ≠ 底模层数 → reject
 *  - 键扫描（下界）> 底模层数 → reject；< 底模层数 → warn
 *  - 一方未知 → unknown（不标） */
export function loraCompatLevel(c: Pick<LoraCkpt, 'base_num_blocks' | 'base_arch_source'>, baseNumBlocks: number | null): LoraCompatLevel {
  const n = c.base_num_blocks ?? null
  if (n == null || baseNumBlocks == null) return 'unknown'
  if (c.base_arch_source === 'metadata') return n === baseNumBlocks ? 'ok' : 'reject'
  if (n > baseNumBlocks) return 'reject'
  if (n < baseNumBlocks) return 'warn'
  return 'ok'
}
