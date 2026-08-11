import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeAll, describe, expect, it, vi } from 'vitest'

import ImagePreviewModal from './ImagePreviewModal'

// jsdom 没实现 pointer capture；useZoomPan 的查看器 handlers 在 pointerdown
// 时会调它，stub 掉避免抛错（不影响 tap 判定——命中记录走 e.target）。
beforeAll(() => {
  Element.prototype.setPointerCapture = vi.fn()
  Element.prototype.releasePointerCapture = vi.fn()
})

/** 视口 wrap = ZoomableImage 里 img 的直接父元素（handlers 挂在它上面）。 */
function viewportOf(img: HTMLElement): HTMLElement {
  return img.parentElement!
}

/** jsdom 无 PointerEvent，fireEvent.pointerDown 会退化成裸 Event 丢掉 button/
 *  clientX —— pan 手势建立不起来。用 MouseEvent 显式构造派发（React 按事件
 *  type 路由到 onPointerDown 等，构造器类型无所谓）。 */
function firePointer(el: Element, type: 'pointerdown' | 'pointermove' | 'pointerup', init: MouseEventInit = {}) {
  fireEvent(el, new MouseEvent(type, { bubbles: true, cancelable: true, button: 0, ...init }))
}

describe('ImagePreviewModal', () => {
  it('× 按钮常显，点击关闭', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    render(<ImagePreviewModal src="/a.png" caption="a.png" onClose={onClose} />)
    await user.click(screen.getByRole('button', { name: '关闭' }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('ESC 关闭；焦点在 input 内时不抢（防御）', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    render(
      <>
        <input aria-label="outside" />
        <ImagePreviewModal src="/a.png" onClose={onClose} />
      </>
    )
    await user.click(screen.getByRole('textbox', { name: 'outside' }))
    await user.keyboard('{Escape}')
    expect(onClose).not.toHaveBeenCalled()
    await user.keyboard('{Tab}')
    await user.keyboard('{Escape}')
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('caption 与 index/total 计数渲染在底 bar', () => {
    render(
      <ImagePreviewModal src="/a.png" caption="shot_042.png" index={2} total={10} onClose={() => {}} />
    )
    expect(screen.getByText('shot_042.png')).toBeInTheDocument()
    expect(screen.getByText('3 / 10')).toBeInTheDocument()
  })

  it('点视口空白（未拖拽）关闭；点图本体、拖拽后松手都不关', () => {
    const onClose = vi.fn()
    render(<ImagePreviewModal src="/a.png" caption="a.png" onClose={onClose} />)
    const img = screen.getByAltText('a.png')
    const viewport = viewportOf(img)

    // 点图本体 → 不关
    firePointer(img, 'pointerdown', { clientX: 50, clientY: 50 })
    firePointer(viewport, 'pointerup')
    expect(onClose).not.toHaveBeenCalled()

    // 拖拽（down → move 有位移 → up）→ 不关
    firePointer(viewport, 'pointerdown', { clientX: 50, clientY: 50 })
    firePointer(viewport, 'pointermove', { clientX: 80, clientY: 60 })
    firePointer(viewport, 'pointerup')
    expect(onClose).not.toHaveBeenCalled()

    // 点视口空白（无位移）→ 关
    firePointer(viewport, 'pointerdown', { clientX: 50, clientY: 50 })
    firePointer(viewport, 'pointerup')
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('Enter 触发 onAccept、Delete 触发 onDelete（传了才生效）', async () => {
    const user = userEvent.setup()
    const onAccept = vi.fn()
    const onDelete = vi.fn()
    render(
      <ImagePreviewModal src="/a.png" onClose={() => {}} onAccept={onAccept} onDelete={onDelete} />
    )
    await user.keyboard('{Enter}')
    expect(onAccept).toHaveBeenCalledTimes(1)
    await user.keyboard('{Delete}')
    expect(onDelete).toHaveBeenCalledTimes(1)
  })

  it('四方向键与屏上箭头按 hasX 生效', async () => {
    const user = userEvent.setup()
    const onPrev = vi.fn()
    const onUp = vi.fn()
    render(
      <ImagePreviewModal
        src="/a.png" onClose={() => {}}
        hasPrev onPrev={onPrev} hasUp onUp={onUp}
      />
    )
    // hasNext/hasDown 未传 → 对应按钮不渲染
    expect(screen.getByRole('button', { name: '上一张' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '下一张' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '上一行' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '下一行' })).not.toBeInTheDocument()
    await user.keyboard('{ArrowLeft}')
    expect(onPrev).toHaveBeenCalledTimes(1)
    await user.keyboard('{ArrowUp}')
    expect(onUp).toHaveBeenCalledTimes(1)
    // 无邻居方向按键 no-op（不抛错即可）
    await user.keyboard('{ArrowRight}{ArrowDown}')
  })

  it('compareSrc → 左右分屏，两个 pane 都是可缩放视口且带 label', () => {
    render(
      <ImagePreviewModal
        src="/orig.png" compareSrc="/proc.png"
        srcLabel="原图" compareLabel="处理后"
        caption="x.png" onClose={() => {}}
      />
    )
    expect(screen.getByText('原图')).toBeInTheDocument()
    expect(screen.getByText('处理后')).toBeInTheDocument()
    // 两个 pane 各有一条 readout（zoom 适应窗口按钮）= 可缩放视口
    expect(screen.getAllByRole('button', { name: '适应窗口' })).toHaveLength(2)
    expect(screen.getByText('x.png')).toBeInTheDocument()
  })
})
