/** Autocomplete 行为 hook —— 给 input / textarea 复用。
 *
 * 设计原则：hook 不修改用户的 value，仅在用户选中候选时回调 `onPick`。
 * caller 决定怎么落进数据（替换 token range / 推到 tags 数组 / append）。
 *
 * 两种 token 模式：
 *   - `wholeAsToken: false`（默认）：根据 cursor + 逗号边界算当前 token，commit
 *     时给 caller `range` 以便切片替换。
 *   - `wholeAsToken: true`：整个 value 作为单 token。给 TagEditor chip 模式 input
 *     用（input 是 draft，本来就一段，commit 时直接 addTag(s.tag)）。
 */
import { useEffect, useMemo, useState } from 'react'
import type React from 'react'

import { useTagDict } from '../../tagDict/store'
import { extractCurrentToken, findSuggestions } from '../../tagDict/suggest'
import type { TagSuggestion } from '../../tagDict/types'

export interface TagSuggestPick {
  /** 选中的候选。 */
  suggestion: TagSuggestion
  /** 当前 token 在原 value 里的范围；wholeAsToken 时是 [0, value.length]。 */
  range: { start: number; end: number }
}

interface Args {
  value: string
  inputRef: React.RefObject<HTMLInputElement | HTMLTextAreaElement | null>
  onPick: (pick: TagSuggestPick) => void
  wholeAsToken?: boolean
  /** 关掉 autocomplete（dict 未加载、字段 disabled 等场景）。 */
  disabled?: boolean
}

export interface TagSuggestApi {
  open: boolean
  suggestions: TagSuggestion[]
  activeIdx: number
  setActiveIdx: (i: number) => void
  setOpen: (open: boolean) => void
  /** 当前 caret 位置（state）；传给 TagSuggestList 让它做 positionDep。 */
  cursor: number
  /** 在 input 的 onKeyDown 里第一句调；返回 true 表示已处理（caller 应 return）。 */
  handleKeyDown: (e: React.KeyboardEvent) => boolean
  /** 在 input 的 onChange 里调（cursor 跟踪 + 自动 open）。 */
  notifyChange: () => void
  /** 在 input 的 onFocus 里调。 */
  notifyFocus: () => void
  /** 在 input 的 onBlur 里调（延迟关闭，给点击留时间）。 */
  notifyBlur: () => void
  /** 在 input 的 onClick / onKeyUp 里调（用户移动光标）。 */
  notifySelect: () => void
  /** 鼠标点选 / 程序触发用。 */
  pickAt: (i: number) => void
}

export function useTagSuggest({
  value, inputRef, onPick, wholeAsToken = false, disabled = false,
}: Args): TagSuggestApi {
  const dict = useTagDict()
  const [open, setOpen] = useState(false)
  const [activeIdx, setActiveIdx] = useState(0)
  const [cursor, setCursor] = useState(0)

  const tokenInfo = useMemo(() => {
    if (wholeAsToken) return { token: value.trim(), start: 0, end: value.length }
    return extractCurrentToken(value, cursor)
  }, [value, cursor, wholeAsToken])

  const suggestions = useMemo(() => {
    if (disabled || !open || dict.status !== 'ready' || !tokenInfo.token) return []
    return findSuggestions(tokenInfo.token, {
      entries: dict.entries,
      tagKeys: dict.tagKeys,
      compactedKeys: dict.compactedKeys,
      reverse: dict.reverse,
    })
  }, [
    disabled, open, tokenInfo.token,
    dict.status, dict.entries, dict.tagKeys, dict.compactedKeys, dict.reverse,
  ])

  // suggestions 列表变化时重置 active
  const sugKey = suggestions.map((s) => s.tag).join('|')
  useEffect(() => { setActiveIdx(0) }, [sugKey])

  const syncCursor = () => {
    const el = inputRef.current
    if (!el) return
    setCursor(el.selectionStart ?? el.value.length)
  }

  const pickAt = (i: number) => {
    const s = suggestions[i]
    if (!s) return
    onPick({ suggestion: s, range: { start: tokenInfo.start, end: tokenInfo.end } })
    setOpen(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent): boolean => {
    if (disabled) return false
    if (!open || suggestions.length === 0) return false
    if (e.key === 'ArrowDown') {
      e.preventDefault(); setActiveIdx((activeIdx + 1) % suggestions.length); return true
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIdx((activeIdx - 1 + suggestions.length) % suggestions.length); return true
    }
    if (e.key === 'Enter' || e.key === 'Tab') {
      e.preventDefault(); pickAt(activeIdx); return true
    }
    if (e.key === 'Escape') {
      e.preventDefault(); setOpen(false); return true
    }
    return false
  }

  return {
    open, suggestions, activeIdx, setActiveIdx, setOpen,
    cursor,
    handleKeyDown,
    notifyChange: () => { syncCursor(); if (!disabled) setOpen(true) },
    notifyFocus: () => { syncCursor(); if (!disabled) setOpen(true) },
    // 100ms 延迟：给 onMouseDown(pick) 时间完成；用户切到别处不会卡 popover
    notifyBlur: () => { setTimeout(() => setOpen(false), 120) },
    notifySelect: () => { syncCursor() },
    pickAt,
  }
}
