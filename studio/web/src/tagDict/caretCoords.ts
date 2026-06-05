/** 计算 input/textarea 内 caret 的像素坐标（相对元素左上角，已考虑 border / padding）。
 *
 * 经典 "mirror div" 算法：拿目标元素的所有相关样式建一个隐藏 div，把内容填到
 * cursor 位置 + 一个 marker span，量 span 的 offset 就是 caret 在元素内的坐标。
 *
 * 改编自 component/textarea-caret-position (MIT)。
 */

const PROPS_TO_COPY = [
  'direction',
  'boxSizing',
  'width',
  'height',
  'overflowX',
  'overflowY',
  'borderTopWidth',
  'borderRightWidth',
  'borderBottomWidth',
  'borderLeftWidth',
  'borderStyle',
  'paddingTop',
  'paddingRight',
  'paddingBottom',
  'paddingLeft',
  'fontStyle',
  'fontVariant',
  'fontWeight',
  'fontStretch',
  'fontSize',
  'fontSizeAdjust',
  'lineHeight',
  'fontFamily',
  'textAlign',
  'textTransform',
  'textIndent',
  'textDecoration',
  'letterSpacing',
  'wordSpacing',
  'tabSize',
  'MozTabSize',
] as const

export interface CaretCoords {
  /** caret 顶部 y（相对元素左上角，含 border + padding 偏移）。 */
  top: number
  /** caret 左侧 x。 */
  left: number
  /** 单行行高（让调用方知道 caret 下边在哪、popover 跳几像素合适）。 */
  height: number
}

export function getCaretCoordinates(
  element: HTMLInputElement | HTMLTextAreaElement,
  position: number,
): CaretCoords {
  const isInput = element.tagName.toLowerCase() === 'input'
  const div = document.createElement('div')
  document.body.appendChild(div)
  const style = div.style
  const computed = window.getComputedStyle(element)

  style.whiteSpace = isInput ? 'nowrap' : 'pre-wrap'
  if (!isInput) style.wordWrap = 'break-word'
  style.position = 'absolute'
  style.visibility = 'hidden'
  style.top = '0'
  style.left = '0'

  for (const prop of PROPS_TO_COPY) {
    // CSSStyleDeclaration 索引签名兼容性问题：直接当 record 写
    ;(style as unknown as Record<string, string>)[prop] = computed[prop as keyof CSSStyleDeclaration] as string
  }

  // input：单行强制 nowrap；空白要变成 nbsp 让 div 不折叠
  let pre = element.value.substring(0, position)
  if (isInput) pre = pre.replace(/\s/g, ' ')
  div.textContent = pre

  const span = document.createElement('span')
  // 用 cursor 处的字符当 marker；缺字符（cursor 在末尾）就用 '.'
  span.textContent = element.value.substring(position) || '.'
  div.appendChild(span)

  const lineHeight = parseInt(computed.lineHeight || '0', 10) || parseInt(computed.fontSize || '16', 10)
  const result: CaretCoords = {
    top: span.offsetTop + parseInt(computed.borderTopWidth || '0', 10),
    left: span.offsetLeft + parseInt(computed.borderLeftWidth || '0', 10),
    height: lineHeight,
  }

  document.body.removeChild(div)
  return result
}
