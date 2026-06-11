/** Autocomplete 候选列表 — Portal + caret-anchored fixed positioning。
 *
 * 老版用 absolute + align="top|bottom" 锚定到 input 容器，textarea 高时
 * popover 会甩到屏幕顶/底栏外。新版：用 mirror-div 算法拿 caret 在 input/textarea
 * 内的像素坐标，portal 到 body 后用 fixed 定位到光标正下方；视口底部不够时
 * 自动翻到光标上方。
 */
import { useLayoutEffect, useState } from 'react'
import type { RefObject } from 'react'
import { createPortal } from 'react-dom'

import { getCaretCoordinates } from '../../tagDict/caretCoords'
import type { TagSuggestion } from '../../tagDict/types'

interface Props {
  open: boolean
  suggestions: TagSuggestion[]
  activeIdx: number
  onPick: (s: TagSuggestion) => void
  onHover: (idx: number) => void
  inputRef: RefObject<HTMLInputElement | HTMLTextAreaElement | null>
  /** 把光标位置传进来当 dep；不传也行（按需补 dep）。 */
  cursor?: number
  /** 触发位置重新计算的额外依赖（比如 value 字符串），改变就重新量 caret。 */
  positionDeps?: ReadonlyArray<unknown>
}

interface Position {
  top: number
  left: number
  /** 翻到光标上方时为 true（popover 底部对齐 caret 顶部）。 */
  flipUp: boolean
}

export function TagSuggestList({
  open, suggestions, activeIdx, onPick, onHover, inputRef, cursor, positionDeps = [],
}: Props) {
  const [pos, setPos] = useState<Position | null>(null)

  useLayoutEffect(() => {
    if (!open || !inputRef.current || suggestions.length === 0) {
      setPos(null)
      return
    }
    const el = inputRef.current
    const cursorPos = cursor ?? el.selectionStart ?? el.value.length
    const caret = getCaretCoordinates(el, cursorPos)
    const rect = el.getBoundingClientRect()

    // caret 在视口坐标中的 y/x
    const caretTopVp = rect.top + caret.top - el.scrollTop
    const caretLeftVp = rect.left + caret.left - el.scrollLeft

    // popover 估算高度（每条 ~26px + 8px padding，上限 260）
    const POPOVER_H = Math.min(260, suggestions.length * 26 + 8)
    const SPACE_BELOW = window.innerHeight - (caretTopVp + caret.height) - 8
    const SPACE_ABOVE = caretTopVp - 8
    const flipUp = SPACE_BELOW < POPOVER_H && SPACE_ABOVE > SPACE_BELOW

    const top = flipUp ? caretTopVp - POPOVER_H - 4 : caretTopVp + caret.height + 4
    // 不让 popover 跨右边界（粗略：保 200px 最小宽度内）
    const left = Math.min(caretLeftVp, window.innerWidth - 220)

    setPos({ top, left: Math.max(8, left), flipUp })
    // 依赖列表：open / suggestions count / cursor / value 任何一个变就重算
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, suggestions.length, cursor, ...positionDeps])

  if (!open || !pos || suggestions.length === 0) return null

  return createPortal(
    <ul
      className="bg-elevated border border-subtle rounded-sm shadow-lg max-h-[260px] overflow-y-auto min-w-[220px] list-none p-1 m-0"
      role="listbox"
      style={{ position: 'fixed', top: pos.top, left: pos.left, zIndex: 1000 }}
    >
      {suggestions.map((s, i) => (
        <li
          key={s.tag}
          role="option"
          aria-selected={i === activeIdx}
          onMouseEnter={() => onHover(i)}
          // onMouseDown + preventDefault：input 不丢 focus
          onMouseDown={(e) => { e.preventDefault(); onPick(s) }}
          className={
            'px-2.5 py-1 text-xs font-mono cursor-pointer rounded-sm flex items-center gap-2 ' +
            (i === activeIdx ? 'bg-overlay text-fg-primary' : 'text-fg-secondary hover:bg-overlay')
          }
        >
          <span>{s.tag}</span>
          {s.zh.length > 0 && (
            <span className="text-fg-tertiary truncate">{s.zh.join(' ')}</span>
          )}
        </li>
      ))}
    </ul>,
    document.body,
  )
}
