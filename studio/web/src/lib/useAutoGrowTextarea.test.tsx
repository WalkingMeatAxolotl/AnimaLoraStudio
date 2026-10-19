import { render } from '@testing-library/react'
import { useRef } from 'react'
import { describe, expect, it } from 'vitest'
import { useAutoGrowTextarea } from './useAutoGrowTextarea'

function Harness({ value }: { value: string }) {
  const ref = useRef<HTMLTextAreaElement>(null)
  useAutoGrowTextarea(ref, value)
  return (
    <div data-testid="wrap" style={{ minHeight: '7px' }}>
      <textarea ref={ref} value={value} readOnly rows={3} />
    </div>
  )
}

describe('useAutoGrowTextarea', () => {
  // regression：测量时 height=auto 让长文本 textarea 瞬间塌缩，外层滚动容器
  // scrollTop 被夹小、再被 scroll anchoring 推到底（LLM 预设提示词编辑乱跳）。
  // 测量期间必须钉住父元素高度，测完还原父元素原有 min-height。
  it('测量期间钉住父元素高度，测完还原', () => {
    const seen: string[] = []
    const proto = HTMLTextAreaElement.prototype
    const saved = {
      offsetParent: Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'offsetParent'),
      offsetHeight: Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'offsetHeight'),
    }
    Object.defineProperty(HTMLElement.prototype, 'offsetParent', { configurable: true, get: () => document.body })
    Object.defineProperty(HTMLElement.prototype, 'offsetHeight', {
      configurable: true,
      get(this: HTMLElement) { return this instanceof HTMLDivElement ? 500 : 0 },
    })
    Object.defineProperty(proto, 'scrollHeight', {
      configurable: true,
      get(this: HTMLTextAreaElement) {
        seen.push(this.parentElement!.style.minHeight)
        return 300
      },
    })
    try {
      const { getByTestId } = render(<Harness value="long text" />)
      const wrap = getByTestId('wrap')
      expect(seen).toEqual(['500px'])
      expect(wrap.style.minHeight).toBe('7px')
      expect((wrap.firstElementChild as HTMLTextAreaElement).style.height).toBe('320px')
    } finally {
      delete (proto as unknown as Record<string, unknown>).scrollHeight
      for (const [key, desc] of Object.entries(saved)) {
        if (desc) Object.defineProperty(HTMLElement.prototype, key, desc)
      }
    }
  })
})
