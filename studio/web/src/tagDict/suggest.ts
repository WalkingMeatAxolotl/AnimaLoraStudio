/** Tag autocomplete — 纯函数。
 *
 * 两件事：
 * 1. extractCurrentToken：从输入文本 + cursor 位置算出"当前正在输入的 token"
 *    及其在原串里的 range（commit 时用来切片替换）。逗号边界兼容 ASCII `,`
 *    和中文 `，`；换行也作为分隔（textarea 多行场景）。
 * 2. findSuggestions：根据 token 找 prefix + substring 候选；含 CJK 自动走
 *    反向索引；按 prefix 优先 + 长度升序。
 */
import type { ReverseEntry, TagSuggestion } from './types'

const CJK_RE = /[一-鿿]/

const SEPARATORS = new Set([',', '，', '\n'])

export interface ExtractedToken {
  /** trim 后的查询文本（喂给 findSuggestions）。 */
  token: string
  /** 原串中 token 实际占位的 start 索引（含 leading 空格）。 */
  start: number
  /** end 索引（exclusive；含 trailing 空格但不含分隔符）。 */
  end: number
}

/** 从光标位置往两边扫到最近分隔符或边界，返回当前 token + range。
 *
 * range 边界规则：包含 token 周围的空白（commit 时一并替换，避免双空格）。
 * 不修改原串；不处理分隔符本身。 */
export function extractCurrentToken(value: string, cursor: number): ExtractedToken {
  const cur = Math.max(0, Math.min(cursor, value.length))
  let start = cur
  while (start > 0 && !SEPARATORS.has(value[start - 1])) start--
  let end = cur
  while (end < value.length && !SEPARATORS.has(value[end])) end++
  const raw = value.slice(start, end)
  const token = raw.trim()
  return { token, start, end }
}

/** suggest 入口需要的 store 切片；分离出来便于测试注入。 */
export interface SuggestStore {
  entries: Map<string, string[]>
  tagKeys: string[]
  reverse: ReverseEntry[]
}

/** 是否含 CJK 字符 —— 决定走英文正向还是中文反向。 */
export function hasCjk(s: string): boolean {
  return CJK_RE.test(s)
}

interface InternalCandidate {
  tag: string
  zh: string[]
  /** prefix=0；substring=1；用于稳定排序。 */
  matchOrder: 0 | 1
  /** 排序键：英文走 tag 长度；中文走匹配到的 zh 长度。 */
  weight: number
}

function toSuggestion(c: InternalCandidate): TagSuggestion {
  return {
    tag: c.tag,
    zh: c.zh,
    matchType: c.matchOrder === 0 ? 'prefix' : 'substring',
  }
}

function pushUnique(out: InternalCandidate[], seen: Set<string>, c: InternalCandidate): void {
  if (seen.has(c.tag)) return
  seen.add(c.tag)
  out.push(c)
}

/** 给定查询 token 找候选；空 token / 未加载 → []。
 *
 * 排序：先 matchOrder（prefix 优先），再 weight（短优先）。 */
export function findSuggestions(
  rawToken: string,
  store: SuggestStore,
  limit = 8,
): TagSuggestion[] {
  const token = rawToken.trim().toLowerCase()
  if (!token) return []
  if (!store.tagKeys.length && !store.reverse.length) return []

  const candidates: InternalCandidate[] = []
  const seen = new Set<string>()
  const goChinese = hasCjk(token)

  if (goChinese) {
    // 中文反向：扫 reverse 数组里 zh 字段 prefix / substring
    for (const re of store.reverse) {
      if (re.zh === token || re.zh.startsWith(token)) {
        for (const tag of re.tags) {
          pushUnique(candidates, seen, {
            tag, zh: store.entries.get(tag) ?? [],
            matchOrder: 0, weight: re.zh.length,
          })
          if (candidates.length >= limit * 4) break
        }
      }
      if (candidates.length >= limit * 4) break
    }
    if (candidates.length < limit * 2) {
      for (const re of store.reverse) {
        if (!re.zh.startsWith(token) && re.zh.includes(token)) {
          for (const tag of re.tags) {
            pushUnique(candidates, seen, {
              tag, zh: store.entries.get(tag) ?? [],
              matchOrder: 1, weight: re.zh.length,
            })
            if (candidates.length >= limit * 4) break
          }
        }
        if (candidates.length >= limit * 4) break
      }
    }
  } else {
    // 英文正向：扫 tagKeys（已按长度升序），prefix 先，substring 后
    for (const tag of store.tagKeys) {
      if (tag === token || tag.startsWith(token)) {
        pushUnique(candidates, seen, {
          tag, zh: store.entries.get(tag) ?? [],
          matchOrder: 0, weight: tag.length,
        })
        if (candidates.length >= limit * 4) break
      }
    }
    if (candidates.length < limit * 2) {
      for (const tag of store.tagKeys) {
        if (!tag.startsWith(token) && tag.includes(token)) {
          pushUnique(candidates, seen, {
            tag, zh: store.entries.get(tag) ?? [],
            matchOrder: 1, weight: tag.length,
          })
          if (candidates.length >= limit * 4) break
        }
      }
    }
  }

  candidates.sort((a, b) => a.matchOrder - b.matchOrder || a.weight - b.weight)
  return candidates.slice(0, limit).map(toSuggestion)
}
