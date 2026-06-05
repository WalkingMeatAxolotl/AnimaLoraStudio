/** Tag 翻译词典 — 类型定义。
 *
 * 数据流：后端 GET /api/tag-dictionary/data 返回 `{entries, meta}`，
 * 前端解析建正向 Map + 反向数组用于 autocomplete。
 */

export interface TagDictMeta {
  source_name: string
  source_url: string
  entry_count: number
  downloaded_at: number
  kind: 'default' | 'user'
}

export interface TagDictPayload {
  entries: Record<string, string[]>
  meta: TagDictMeta
}

export interface TagDictMetaResponse {
  loaded: boolean
  meta: TagDictMeta | null
}

/** 单条 autocomplete 候选；UI 渲染 `${tag}  ${zh.join(' ')}`。 */
export interface TagSuggestion {
  tag: string
  zh: string[]
  matchType: 'prefix' | 'substring'
}

export type TagDictStatus = 'idle' | 'loading' | 'ready' | 'error' | 'empty'

/** 反向索引一条 entry：单个 zh token → 含它的所有英文 tag。 */
export interface ReverseEntry {
  zh: string
  tags: string[]
}
