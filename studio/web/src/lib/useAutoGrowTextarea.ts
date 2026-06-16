import { useLayoutEffect, type RefObject } from 'react'

/** textarea 随内容自动撑高，无上限；最小高度 = rows 属性决定的初始高度。
 *
 * 先把 height 重置为 auto 让浏览器回落到 rows 高度，再设为 scrollHeight —
 * 内容少时 scrollHeight 被 clientHeight 托底，正好回到 rows 最小高度。
 * 配合 className 加 resize-none overflow-hidden（手动拖拽会被下次输入覆盖，
 * 干脆禁掉；hidden 防止撑高瞬间滚动条闪烁）。
 *
 * 元素在 display:none 容器里时（如未激活的 tab 分页）scrollHeight=0，此时直接
 * bail，别把高度压成 0；ResizeObserver 会在它重新可见（尺寸 0→实际）时回调重算，
 * 避免切回该 tab 后 textarea 塌缩。
 */
export function useAutoGrowTextarea(
  ref: RefObject<HTMLTextAreaElement>,
  value: string,
): void {
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const fit = () => {
      // offsetParent===null ⇒ 自身或祖先 display:none，scrollHeight 不可信，先不动高度
      if (el.offsetParent === null) return
      el.style.height = 'auto'
      // scrollHeight 不含 border；box-sizing: border-box 下补回去，否则每次少
      // 2px 出现滚动条
      const border = el.offsetHeight - el.clientHeight
      el.style.height = `${el.scrollHeight + border}px`
    }
    fit()
    // 监听尺寸变化：tab 切换让元素从隐藏变可见时重算，宽度变化（换行影响高度）时也重算
    const ro = new ResizeObserver(fit)
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref, value])
}
