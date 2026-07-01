/** entryTaskId —— 0.17 P-H `?task=` 深链按 task_id 命中历史条目的匹配逻辑。
 *  cache 条目顶层带 taskId；disk 条目的 task_id 由 server enrich 进 PNG anima_params。 */
import { describe, expect, it } from 'vitest'
import { entryTaskId, type HistoryEntry } from './entryAdapter'

describe('entryTaskId', () => {
  it('cache 条目取顶层 taskId', () => {
    const e = { source: 'cache', taskId: 42 } as HistoryEntry
    expect(entryTaskId(e)).toBe(42)
  })

  it('disk 条目从 params.task_id 取', () => {
    const e = { source: 'disk', params: { task_id: 7 } } as unknown as HistoryEntry
    expect(entryTaskId(e)).toBe(7)
  })

  it('disk 条目无 task_id → undefined（老落盘图无 enrich）', () => {
    const e = { source: 'disk', params: {} } as unknown as HistoryEntry
    expect(entryTaskId(e)).toBeUndefined()
  })
})
