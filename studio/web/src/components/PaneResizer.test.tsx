import { fireEvent, render, screen } from '@testing-library/react'
import { createRef } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import PaneResizer, { clampPaneValue, normalizePanePair } from './PaneResizer'

// jsdom 无 layout & 无 pointer capture：容器宽度和 capture API 都得自己补，
// 否则 onPointerDown 直接 early-return（width <= 0）。
const CONTAINER_WIDTH = 1000

// jsdom 也没有 PointerEvent —— fireEvent 找不到构造函数会退化成裸 Event，
// button / clientX 全丢，组件当成非左键点击直接 return。继承 MouseEvent 补齐。
class PointerEventPolyfill extends MouseEvent {
  pointerId: number
  constructor(type: string, props: PointerEventInit = {}) {
    super(type, props)
    this.pointerId = props.pointerId ?? 0
  }
}
const w = window as unknown as { PointerEvent?: typeof PointerEvent }
if (!w.PointerEvent) {
  w.PointerEvent = PointerEventPolyfill as unknown as typeof PointerEvent
}

function setup(props: Partial<React.ComponentProps<typeof PaneResizer>> = {}) {
  const onChange = vi.fn()
  const ref = createRef<HTMLDivElement>()
  const view = render(
    <div ref={ref}>
      <PaneResizer
        containerRef={ref}
        value={40}
        onChange={onChange}
        ariaLabel="resize"
        {...props}
      />
    </div>,
  )
  vi.spyOn(ref.current!, 'getBoundingClientRect').mockReturnValue({
    width: CONTAINER_WIDTH,
  } as DOMRect)
  return { onChange, handle: screen.getByRole('separator'), unmount: view.unmount }
}

/** 从 startX 拖到 endX（按下 → 移动 → 抬起） */
function drag(handle: HTMLElement, startX: number, endX: number) {
  fireEvent.pointerDown(handle, { button: 0, clientX: startX, pointerId: 1 })
  fireEvent.pointerMove(handle, { clientX: endX, pointerId: 1 })
  fireEvent.pointerUp(handle, { pointerId: 1 })
}

beforeEach(() => {
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
  Object.assign(HTMLElement.prototype, {
    setPointerCapture: vi.fn(),
    hasPointerCapture: vi.fn(() => true),
    releasePointerCapture: vi.fn(),
  })
})

describe('pane value bounds', () => {
  it('clamps invalid persisted values to the declared range', () => {
    expect(clampPaneValue(90, 25, 70)).toBe(70)
    expect(clampPaneValue(Number.NaN, 25, 70)).toBe(25)
  })

  it('repairs a three-pane budget proportionally while preserving valid values', () => {
    expect(normalizePanePair(40, 32, {
      startMin: 15,
      endMin: 20,
      flexibleMin: 15,
    })).toEqual({ start: 40, end: 32 })

    const repaired = normalizePanePair(90, 90, {
      startMin: 15,
      endMin: 20,
      flexibleMin: 15,
    })
    expect(repaired.start + repaired.end).toBeCloseTo(85)
    expect(repaired.start).toBeGreaterThanOrEqual(15)
    expect(repaired.end).toBeGreaterThanOrEqual(20)
  })
})

describe('PaneResizer', () => {
  it('拖动按容器宽度换算成百分比增量', () => {
    const { onChange, handle } = setup()
    drag(handle, 500, 600) // +100px / 1000px = +10%
    expect(onChange).toHaveBeenLastCalledWith(50)
  })

  it('anchor=end 时方向取反（受控的是右侧栏宽度）', () => {
    const { onChange, handle } = setup({ anchor: 'end' })
    drag(handle, 500, 600)
    expect(onChange).toHaveBeenLastCalledWith(30)
  })

  it('拖过头夹在 min / max 之间', () => {
    const { onChange, handle } = setup({ min: 20, max: 45 })
    drag(handle, 500, 900)
    expect(onChange).toHaveBeenLastCalledWith(45)
    drag(handle, 500, 100)
    expect(onChange).toHaveBeenLastCalledWith(20)
  })

  it('方向键与 Home / End 调整，同样受 min / max 约束', () => {
    const { onChange, handle } = setup({ value: 41, min: 15, max: 42 })
    fireEvent.keyDown(handle, { key: 'ArrowRight' })
    expect(onChange).toHaveBeenLastCalledWith(42)
    fireEvent.keyDown(handle, { key: 'ArrowLeft' })
    expect(onChange).toHaveBeenLastCalledWith(39)
    fireEvent.keyDown(handle, { key: 'Home' })
    expect(onChange).toHaveBeenLastCalledWith(15)
    fireEvent.keyDown(handle, { key: 'End' })
    expect(onChange).toHaveBeenLastCalledWith(42)
  })

  it('抬起后继续移动不再改值', () => {
    const { onChange, handle } = setup()
    drag(handle, 500, 600)
    onChange.mockClear()
    fireEvent.pointerMove(handle, { clientX: 800, pointerId: 1 })
    expect(onChange).not.toHaveBeenCalled()
  })

  it('pointer cancel 或卸载时恢复全局 cursor 与文字选择', () => {
    const { handle, unmount } = setup()
    fireEvent.pointerDown(handle, { button: 0, clientX: 500, pointerId: 1 })
    expect(document.body.style.cursor).toBe('col-resize')
    expect(document.body.style.userSelect).toBe('none')
    fireEvent.pointerCancel(handle, { pointerId: 1 })
    expect(document.body.style.cursor).toBe('')
    expect(document.body.style.userSelect).toBe('')

    fireEvent.pointerDown(handle, { button: 0, clientX: 500, pointerId: 2 })
    unmount()
    expect(document.body.style.cursor).toBe('')
    expect(document.body.style.userSelect).toBe('')
  })

  it('暴露 separator 语义、受控 pane 关联与修复后的当前值', () => {
    const { handle } = setup({ value: 80, min: 15, max: 55, ariaControls: 'start-pane' })
    expect(handle).toHaveAttribute('aria-orientation', 'vertical')
    expect(handle).toHaveAttribute('aria-controls', 'start-pane')
    expect(handle).toHaveAttribute('aria-valuenow', '55')
    expect(handle).toHaveAttribute('aria-valuemin', '15')
    expect(handle).toHaveAttribute('aria-valuemax', '55')
    expect(handle).toHaveAttribute('aria-valuetext', '55%')
  })
})
