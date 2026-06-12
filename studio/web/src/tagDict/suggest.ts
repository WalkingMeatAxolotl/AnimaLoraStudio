/** Tag autocomplete — 纯函数。
 *
 * 两件事：
 * 1. extractCurrentToken：从输入文本 + cursor 位置算出"当前正在输入的 token"
 *    及其在原串里的 range（commit 时用来切片替换）。逗号边界兼容 ASCII `,`
 *    和中文 `，`；换行也作为分隔（textarea 多行场景）。
 * 2. findSuggestions：根据 token 找 prefix + substring 候选；含 CJK 自动走
 *    反向索引；按 prefix 优先，同组内保持词典原始顺序（CSV 按 post_count DESC）。
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
  /** tagKeys 的紧凑形式（去空格/_），同下标对齐；store 加载时预计算。 */
  compactedKeys: string[]
  reverse: ReverseEntry[]
}

/** 是否含 CJK 字符 —— 决定走英文正向还是中文反向。 */
export function hasCjk(s: string): boolean {
  return CJK_RE.test(s)
}

interface InternalCandidate {
  tag: string
  zh: string[]
  /** prefix=0；substring=1。两轮扫描天然 prefix 组在前，无需再排序。 */
  matchOrder: 0 | 1
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
 * 顺序：prefix 组在前，组内保持扫描顺序（英文 = 词典行序即热度；
 * 中文 = zh 长度升序，见 store.buildReverse）。 */
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
            tag, zh: store.entries.get(tag) ?? [], matchOrder: 0,
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
              tag, zh: store.entries.get(tag) ?? [], matchOrder: 1,
            })
            if (candidates.length >= limit * 4) break
          }
        }
        if (candidates.length >= limit * 4) break
      }
    }
  } else {
    // 英文正向：扫 tagKeys（词典行序即热度序），prefix 先，substring 后。
    // 双方都用紧凑形式（去空格/_）比较：tag 侧用预计算的 compactedKeys，
    // token 侧现算一次 —— "redey" / "red_ey" / "red ey" 都能命中 "red eyes"。
    // 原串 startsWith/includes 为真时紧凑形式必然也为真，无需重复判断。
    const compactToken = token.replace(/[\s_]/g, '')
    if (compactToken) {
      for (let i = 0; i < store.tagKeys.length; i++) {
        if (store.compactedKeys[i].startsWith(compactToken)) {
          pushUnique(candidates, seen, {
            tag: store.tagKeys[i],
            zh: store.entries.get(store.tagKeys[i]) ?? [],
            matchOrder: 0,
          })
          if (candidates.length >= limit * 4) break
        }
      }
      if (candidates.length < limit * 2) {
        for (let i = 0; i < store.tagKeys.length; i++) {
          const compacted = store.compactedKeys[i]
          if (!compacted.startsWith(compactToken) && compacted.includes(compactToken)) {
            pushUnique(candidates, seen, {
              tag: store.tagKeys[i],
              zh: store.entries.get(store.tagKeys[i]) ?? [],
              matchOrder: 1,
            })
            if (candidates.length >= limit * 4) break
          }
        }
      }
    }
  }

  // 两轮扫描天然 prefix 在前 + 组内保持扫描顺序，直接截断即可。
  return candidates.slice(0, limit).map(toSuggestion)
}
