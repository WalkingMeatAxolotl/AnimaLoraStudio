import { describe, expect, it } from 'vitest'
import { extractCurrentToken, findSuggestions, hasCjk, type SuggestStore } from './suggest'
import type { ReverseEntry } from './types'

// ---------------------------------------------------------------------------
// extractCurrentToken
// ---------------------------------------------------------------------------

describe('extractCurrentToken', () => {
  it('returns whole string when no separator', () => {
    expect(extractCurrentToken('1gir', 4)).toEqual({ token: '1gir', start: 0, end: 4 })
  })

  it('extracts token after a comma', () => {
    // 光标在 "1gir,solo" 末尾时，当前 token 应该是 "solo"
    expect(extractCurrentToken('1girl,solo', 10)).toEqual({ token: 'solo', start: 6, end: 10 })
  })

  it('extracts token after a space-comma sequence', () => {
    // 光标在 "1girl, sol" 末尾；start 含 leading 空格
    expect(extractCurrentToken('1girl, sol', 10)).toEqual({ token: 'sol', start: 6, end: 10 })
  })

  it('handles Chinese comma', () => {
    expect(extractCurrentToken('1girl，sol', 9)).toEqual({ token: 'sol', start: 6, end: 9 })
  })

  it('handles cursor in the middle of a token', () => {
    // value="1girl,solo,long"; cursor=8 在 "solo" 中间
    const r = extractCurrentToken('1girl,solo,long', 8)
    expect(r.token).toBe('solo')
    expect(r.start).toBe(6)
    expect(r.end).toBe(10)
  })

  it('handles newline as separator (textarea multiline)', () => {
    const r = extractCurrentToken('foo\nbar', 7)
    expect(r).toEqual({ token: 'bar', start: 4, end: 7 })
  })

  it('returns empty token when cursor right after comma', () => {
    const r = extractCurrentToken('1girl,', 6)
    expect(r).toEqual({ token: '', start: 6, end: 6 })
  })

  it('clamps cursor to string range', () => {
    expect(extractCurrentToken('abc', 100)).toEqual({ token: 'abc', start: 0, end: 3 })
    expect(extractCurrentToken('abc', -5)).toEqual({ token: 'abc', start: 0, end: 3 })
  })
})

// ---------------------------------------------------------------------------
// hasCjk
// ---------------------------------------------------------------------------

describe('hasCjk', () => {
  it('detects Chinese characters', () => {
    expect(hasCjk('猫')).toBe(true)
    expect(hasCjk('1girl 长发')).toBe(true)
  })

  it('false for pure ASCII', () => {
    expect(hasCjk('1girl')).toBe(false)
    expect(hasCjk('long_hair')).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// findSuggestions
// ---------------------------------------------------------------------------

function buildStore(entries: Record<string, string[]>): SuggestStore {
  const map = new Map(Object.entries(entries))
  const reverseMap = new Map<string, string[]>()
  map.forEach((aliases, tag) => {
    aliases.forEach((zh) => {
      const ex = reverseMap.get(zh)
      if (ex) ex.push(tag); else reverseMap.set(zh, [tag])
    })
  })
  const reverse: ReverseEntry[] = Array.from(reverseMap.entries())
    .map(([zh, tags]) => ({ zh, tags }))
    .sort((a, b) => a.zh.length - b.zh.length)
  // 与生产 store.fetchDict 一致：tagKeys 保持词典行序（热度序），不重排
  const tagKeys = Array.from(map.keys())
  const compactedKeys = tagKeys.map((t) => t.replace(/[\s_]/g, ''))
  return { entries: map, tagKeys, compactedKeys, reverse }
}

describe('findSuggestions — English path', () => {
  const store = buildStore({
    '1girl': ['1女孩'],
    'girl': ['女孩'],
    'long hair': ['长发'],
    'longest day': [],
    'solo': ['单人'],
  })

  it('returns [] for empty token', () => {
    expect(findSuggestions('', store)).toEqual([])
  })

  it('returns [] when dict empty', () => {
    expect(findSuggestions('foo', buildStore({}))).toEqual([])
  })

  it('finds prefix matches in dictionary order', () => {
    const r = findSuggestions('lon', store)
    expect(r.map((s) => s.tag)).toEqual(['long hair', 'longest day'])
    expect(r[0].matchType).toBe('prefix')
  })

  it('orders prefix matches by dictionary (popularity) order, not length', () => {
    // 'red eyes' 在词典里靠前（更热门）但比 'red' 长；热度序应当赢
    const s = buildStore({
      'red eyes': ['红眼'],
      'red': ['红'],
    })
    expect(findSuggestions('re', s).map((x) => x.tag)).toEqual(['red eyes', 'red'])
  })

  it('case-insensitive', () => {
    const r = findSuggestions('GIRL', store)
    expect(r.map((s) => s.tag)).toContain('girl')
    expect(r.map((s) => s.tag)).toContain('1girl')
  })

  it('returns matched zh translations on each suggestion', () => {
    const r = findSuggestions('girl', store)
    const girl = r.find((s) => s.tag === 'girl')
    expect(girl?.zh).toEqual(['女孩'])
  })

  it('falls back to substring after prefix exhausted', () => {
    // '1girl' has substring 'girl', but 'girl' itself is the prefix match;
    // 1girl should show up via substring path
    const r = findSuggestions('girl', store, 5)
    expect(r.find((s) => s.tag === '1girl')?.matchType).toBe('substring')
  })

  it('matches compacted form (skip spaces/_): "redey" → "red eyes"', () => {
    const store = buildStore({
      '1girl': ['1女孩'],
      'red eyes': ['红眼'],
      'red hair': ['红发'],
      'solo': ['单人'],
    })
    const r = findSuggestions('redey', store)
    expect(r.map((s) => s.tag)).toContain('red eyes')
    expect(r[0].matchType).toBe('prefix')
  })

  it('matches when token uses booru underscore form: "red_ey" → "red eyes"', () => {
    const s = buildStore({ 'red eyes': ['红眼'], 'solo': ['单人'] })
    const r = findSuggestions('red_ey', s)
    expect(r.map((x) => x.tag)).toContain('red eyes')
    expect(r[0].matchType).toBe('prefix')
  })

  it('returns [] for token made of separators only', () => {
    // compactToken 为空时必须直接放弃，否则 ''.startsWith 让全字典都命中
    expect(findSuggestions('_', store)).toEqual([])
  })

  it('honors limit', () => {
    const big = buildStore(Object.fromEntries(
      Array.from({ length: 20 }, (_, i) => [`xtag${i}`, []]),
    ))
    expect(findSuggestions('xtag', big, 3)).toHaveLength(3)
  })
})

describe('findSuggestions — Chinese path', () => {
  const store = buildStore({
    '1girl': ['1女孩'],
    'girl': ['女孩'],
    'cat girl': ['猫娘', '兽耳'],
    'cat': ['猫'],
    'long hair': ['长发'],
  })

  it('matches Chinese prefix → English tags', () => {
    const r = findSuggestions('猫', store)
    const tags = r.map((s) => s.tag)
    expect(tags).toContain('cat')
    expect(tags).toContain('cat girl')   // 猫娘 prefix 也包含 '猫'
  })

  it('multi-char Chinese prefix', () => {
    const r = findSuggestions('女孩', store)
    expect(r.map((s) => s.tag)).toContain('girl')
  })

  it('returns [] for Chinese query with no match', () => {
    expect(findSuggestions('海洋', store)).toEqual([])
  })
})
