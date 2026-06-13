/** Tag 翻译词典 — 模块级 singleton + useSyncExternalStore hook。
 *
 * 设计：dict 约 7MB JSON（默认源 20 万条，gzip 后约 3.5MB），启动拉一次后放
 * 内存；浏览器 HTTP cache 处理 304。
 * 不写 IndexedDB（项目无先例，复杂度不值；体量更大后再换）。
 *
 * 对外 API：
 *   - useTagDict()  组件订阅，首次 mount 触发 loadDict()
 *   - reloadDict()  上传 / reset 后调用，强刷
 *   - lookupTag(t)  非 hook 形式快查（chip / 单点查询）
 */
import { useEffect, useSyncExternalStore } from 'react'

import type { ReverseEntry, TagDictMeta, TagDictPayload, TagDictStatus } from './types'

interface State {
  status: TagDictStatus
  entries: Map<string, string[]>
  /** tag 列表（保持词典文件行序；默认源即 post_count DESC 的热度排序，
   * 用户上传的词典无此保证。autocomplete 扫描用）。 */
  tagKeys: string[]
  /** tagKeys 的紧凑形式（去空格/_），同下标对齐。加载时一次算好，
   * 避免 suggest 每次按键对全字典逐个 regex replace。 */
  compactedKeys: string[]
  reverse: ReverseEntry[]
  meta: TagDictMeta | null
  error: string | null
}

let state: State = {
  status: 'idle',
  entries: new Map(),
  tagKeys: [],
  compactedKeys: [],
  reverse: [],
  meta: null,
  error: null,
}

const listeners = new Set<() => void>()
let inFlight: Promise<void> | null = null

function setState(next: Partial<State>): void {
  state = { ...state, ...next }
  listeners.forEach((l) => l())
}

function subscribe(l: () => void): () => void {
  listeners.add(l)
  return () => { listeners.delete(l) }
}

/** 把 entries map 摊成反向索引：每个 zh token → 含它的英文 tags 数组。
 *
 * 同一 tag 的多 zh alias 都各自建一条 reverse entry。zh 完全重复的合并 tags。 */
function buildReverse(entries: Map<string, string[]>): ReverseEntry[] {
  const zhToTags = new Map<string, string[]>()
  entries.forEach((aliases, tag) => {
    aliases.forEach((zh) => {
      const existing = zhToTags.get(zh)
      if (existing) existing.push(tag)
      else zhToTags.set(zh, [tag])
    })
  })
  // 按 zh 长度升序，让 prefix match 短的优先；suggest.ts 按此扫描序直接输出。
  return Array.from(zhToTags.entries())
    .map(([zh, tags]) => ({ zh, tags }))
    .sort((a, b) => a.zh.length - b.zh.length)
}

async function fetchDict(): Promise<void> {
  setState({ status: 'loading', error: null })
  try {
    const resp = await fetch('/api/tag-dictionary/data')
    if (resp.status === 404) {
      setState({
        status: 'empty',
        entries: new Map(),
        tagKeys: [],
        compactedKeys: [],
        reverse: [],
        meta: null,
      })
      return
    }
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const payload = (await resp.json()) as TagDictPayload
    const entries = new Map(Object.entries(payload.entries || {}))
    // 行序以后端 keys 数组为准：JS 对象会把整数型 key（"69"、年份 tag 等）
    // 重排到最前，entries 自身的 key 序不可靠。旧后端没有 keys 时退回对象序。
    const tagKeys = payload.keys && payload.keys.length === entries.size
      ? payload.keys
      : Array.from(entries.keys())
    const compactedKeys = tagKeys.map((t) => t.replace(/[\s_]/g, ''))
    const reverse = buildReverse(entries)
    setState({
      status: 'ready',
      entries,
      tagKeys,
      compactedKeys,
      reverse,
      meta: payload.meta || null,
      error: null,
    })
  } catch (err) {
    setState({
      status: 'error',
      error: err instanceof Error ? err.message : String(err),
    })
  }
}

/** 首次 / 强制刷新：idempotent；已在 loading 中直接复用 in-flight Promise。 */
export function loadDict(force = false): Promise<void> {
  if (!force && (state.status === 'ready' || state.status === 'loading')) {
    return inFlight ?? Promise.resolve()
  }
  inFlight = fetchDict().finally(() => { inFlight = null })
  return inFlight
}

/** 上传 / reset 后调用 → 强刷。 */
export function reloadDict(): Promise<void> {
  return loadDict(true)
}

/** 给 React 组件订阅用。首次 mount 触发加载（idempotent，多组件挂载只发一次请求）。 */
export function useTagDict(): State {
  const snapshot = useSyncExternalStore(subscribe, () => state, () => state)
  useEffect(() => {
    if (state.status === 'idle') void loadDict()
  }, [])
  return snapshot
}

/** Chip 渲染等非 hook 场景的快查。dict 未加载时返回 undefined。 */
export function lookupTag(tag: string): string[] | undefined {
  return state.entries.get(tag)
}

/** 测试用：直接注入 state，绕过网络。生产代码不要调。 */
export function __setStateForTest(next: Partial<State>): void {
  setState(next)
}

/** 内部读：suggest.ts 给的 store 句柄；不导出给外部应用代码。 */
export function _getInternalState(): State {
  return state
}
