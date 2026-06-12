import { useLayoutEffect, type RefObject } from 'react'

/** textarea 随内容自动撑高，无上限；最小高度 = rows 属性决定的初始高度。
 *
 * 先把 height 重置为 auto 让浏览器回落到 rows 高度，再设为 scrollHeight —
 * 内容少时 scrollHeight 被 clientHeight 托底，正好回到 rows 最小高度。
 * 配合 className 加 resize-none overflow-hidden（手动拖拽会被下次输入覆盖，
 * 干脆禁掉；hidden 防止撑高瞬间滚动条闪烁）。
 */
export function useAutoGrowTextarea(
  ref: RefObject<HTMLTextAreaElement>,
  value: string,
): void {
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    // scrollHeight 不含 border；box-sizing: border-box 下补回去，否则每次少
    // 2px 出现滚动条
    const border = el.offsetHeight - el.clientHeight
    el.style.height = `${el.scrollHeight + border}px`
  }, [ref, value])
}
