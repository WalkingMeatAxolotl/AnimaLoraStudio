import { StrictMode, useState } from 'react'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeAll, describe, expect, it, vi } from 'vitest'

import ImagePreviewModal from './ImagePreviewModal'
import Modal from './Modal'
import i18n from '../i18n'

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

function PreviewHarness() {
  const [open, setOpen] = useState(false)
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>Open preview</button>
      <button type="button">Background action</button>
      {open && <ImagePreviewModal src="/a.png" caption="a.png" onClose={() => setOpen(false)} />}
    </>
  )
}

describe('ImagePreviewModal', () => {
  it('× 按钮常显，点击关闭', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    render(<ImagePreviewModal src="/a.png" caption="a.png" onClose={onClose} />)
    await user.click(screen.getByRole('button', { name: '关闭' }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('portals a named dialog outside the workspace and initially focuses the preview', () => {
    const { container } = render(<ImagePreviewModal src="/a.png" caption="a.png" onClose={() => {}} />)
    const dialog = screen.getByRole('dialog', { name: '图片预览' })
    expect(container).not.toContainElement(dialog)
    expect(document.body).toContainElement(dialog)
    expect(dialog).toHaveAttribute('aria-modal', 'true')
    expect(dialog).toHaveAccessibleDescription('a.png')
    expect(dialog).toHaveFocus()
  })

  it('traps Tab in preview controls and restores the connected opener after Escape', async () => {
    const user = userEvent.setup()
    render(<StrictMode><PreviewHarness /></StrictMode>)
    const opener = screen.getByRole('button', { name: 'Open preview' })
    await user.click(opener)
    const dialog = screen.getByRole('dialog', { name: '图片预览' })
    const close = within(dialog).getByRole('button', { name: '关闭' })
    const last = within(dialog).getByRole('button', { name: '100%' })
    expect(dialog).toHaveFocus()
    await user.tab({ shift: true })
    expect(last).toHaveFocus()
    await user.tab()
    expect(dialog).toHaveFocus()
    await user.tab()
    expect(close).toHaveFocus()
    await user.tab({ shift: true })
    expect(dialog).toHaveFocus()
    await user.tab({ shift: true })
    expect(last).toHaveFocus()
    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(opener).toHaveFocus()
  })

  it.each(['{Enter}', ' '])('keeps the close button native for %s without accepting an image', async (key) => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    const onAccept = vi.fn()
    const onDelete = vi.fn()
    render(<ImagePreviewModal src="/a.png" onClose={onClose} onAccept={onAccept} onDelete={onDelete} />)
    screen.getByRole('button', { name: '关闭' }).focus()
    await user.keyboard(key)
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(onAccept).not.toHaveBeenCalled()
    expect(onDelete).not.toHaveBeenCalled()
  })

  it('keeps zoom and navigation buttons native instead of triggering business shortcuts', async () => {
    const user = userEvent.setup()
    const onAccept = vi.fn()
    const onDelete = vi.fn()
    const onPrev = vi.fn()
    render(<ImagePreviewModal src="/a.png" onClose={() => {}} hasPrev onPrev={onPrev} onAccept={onAccept} onDelete={onDelete} />)
    const zoom = screen.getByRole('button', { name: '适应窗口' })
    const zoomClick = vi.fn()
    zoom.addEventListener('click', zoomClick)
    zoom.focus()
    await user.keyboard('{Enter} {Delete}{Backspace}')
    expect(zoomClick).toHaveBeenCalledTimes(2)
    screen.getByRole('button', { name: '上一张' }).focus()
    await user.keyboard('{Enter}')
    expect(onPrev).toHaveBeenCalledTimes(1)
    expect(onAccept).not.toHaveBeenCalled()
    expect(onDelete).not.toHaveBeenCalled()
  })

  it('does not handle business shortcuts or Escape dispatched outside the preview', () => {
    const onClose = vi.fn()
    const onAccept = vi.fn()
    const onDelete = vi.fn()
    render(<ImagePreviewModal src="/a.png" onClose={onClose} onAccept={onAccept} onDelete={onDelete} />)
    for (const key of ['Enter', ' ', 'Delete', 'Backspace', 'Escape']) fireEvent.keyDown(document, { key })
    expect(onClose).not.toHaveBeenCalled()
    expect(onAccept).not.toHaveBeenCalled()
    expect(onDelete).not.toHaveBeenCalled()
  })

  it('blocks Ctrl/Cmd+K inside the preview without leaking to global search', () => {
    render(<ImagePreviewModal src="/a.png" onClose={() => {}} />)
    const globalSearch = vi.fn()
    window.addEventListener('keydown', globalSearch)
    try {
      const dialog = screen.getByRole('dialog', { name: '图片预览' })
      for (const modifier of ['ctrlKey', 'metaKey']) {
        expect(fireEvent.keyDown(dialog, { key: 'k', [modifier]: true })).toBe(false)
        expect(fireEvent.keyDown(screen.getByRole('button', { name: '关闭' }), { key: 'K', [modifier]: true })).toBe(false)
      }
      expect(globalSearch).not.toHaveBeenCalled()
    } finally {
      window.removeEventListener('keydown', globalSearch)
    }
  })

  it('ignores modified, composing, and repeated mutation shortcuts', () => {
    const onAccept = vi.fn()
    const onDelete = vi.fn()
    render(<ImagePreviewModal src="/a.png" onClose={() => {}} onAccept={onAccept} onDelete={onDelete} />)
    const dialog = screen.getByRole('dialog', { name: '图片预览' })
    for (const key of ['Enter', ' ', 'Delete', 'Backspace']) {
      for (const flag of ['ctrlKey', 'metaKey', 'altKey', 'shiftKey', 'repeat', 'isComposing']) {
        fireEvent.keyDown(dialog, { key, [flag]: true })
      }
    }
    expect(onAccept).not.toHaveBeenCalled()
    expect(onDelete).not.toHaveBeenCalled()
  })

  it('preserves focus on an existing control when the displayed image changes', () => {
    const onClose = vi.fn()
    const view = render(<ImagePreviewModal src="/a.png" caption="a.png" onClose={onClose} />)
    const zoom = screen.getByRole('button', { name: '100%' })
    zoom.focus()
    view.rerender(<ImagePreviewModal src="/b.png" caption="b.png" onClose={onClose} />)
    expect(zoom).toHaveFocus()
    expect(screen.getByRole('dialog')).toHaveAccessibleDescription('b.png')
  })

  it('returns to the image surface after controls so keyboard-only acceptance remains reachable', async () => {
    const user = userEvent.setup()
    const onAccept = vi.fn()
    render(<ImagePreviewModal src="/a.png" onClose={() => {}} onAccept={onAccept} />)
    screen.getByRole('button', { name: '100%' }).focus()
    await user.tab()
    expect(screen.getByRole('dialog')).toHaveFocus()
    await user.keyboard('{Enter}')
    expect(onAccept).toHaveBeenCalledTimes(1)
  })

  it.each(['body', 'documentElement'] as const)('recovers %s fallback when the focused navigation button disappears', (fallback) => {
    const props = { src: '/a.png', onClose: vi.fn(), onNext: vi.fn(), hasNext: true }
    const view = render(<ImagePreviewModal {...props} />)
    screen.getByRole('button', { name: '下一张' }).focus()
    // Simulate the alternate document fallback; do not claim this executes Firefox.
    const active = fallback === 'documentElement'
      ? vi.spyOn(document, 'activeElement', 'get').mockReturnValue(document.documentElement)
      : null
    try {
      view.rerender(<ImagePreviewModal {...props} src="/b.png" hasNext={false} />)
    } finally {
      active?.mockRestore()
    }
    expect(screen.getByRole('dialog')).toHaveFocus()
  })

  it('includes both compare readouts in the same focus cycle and skips disabled controls', async () => {
    const user = userEvent.setup()
    render(<ImagePreviewModal src="/a.png" compareSrc="/b.png" onClose={() => {}} />)
    const dialog = screen.getByRole('dialog')
    const zoomButtons = within(dialog).getAllByRole('button', { name: '100%' })
    zoomButtons[1].setAttribute('disabled', '')
    await user.tab({ shift: true })
    expect(within(dialog).getAllByRole('button', { name: '适应窗口' })[1]).toHaveFocus()
    await user.tab()
    expect(dialog).toHaveFocus()
    await user.tab()
    expect(within(dialog).getByRole('button', { name: '关闭' })).toHaveFocus()
  })

  it('does not handle Escape from a later confirmation and regains focus when it closes', async () => {
    const user = userEvent.setup()
    const onPreviewClose = vi.fn()
    function ConfirmationHarness() {
      const [confirm, setConfirm] = useState(false)
      return (
        <>
          <ImagePreviewModal src="/a.png" onClose={onPreviewClose} onDelete={() => setConfirm(true)} />
          {confirm && <Modal title="Remove image?" onClose={() => setConfirm(false)}><button>Cancel removal</button></Modal>}
        </>
      )
    }
    render(<ConfirmationHarness />)
    const preview = screen.getByRole('dialog', { name: '图片预览' })
    await user.keyboard('{Delete}')
    await waitFor(() => expect(screen.getByRole('button', { name: 'Cancel removal' })).toHaveFocus())
    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog', { name: 'Remove image?' })).not.toBeInTheDocument()
    expect(onPreviewClose).not.toHaveBeenCalled()
    expect(preview).toHaveFocus()
  })

  it('updates English accessible names without resetting focused controls', async () => {
    const view = render(<ImagePreviewModal src="/a.png" onClose={() => {}} />)
    screen.getByRole('button', { name: '100%' }).focus()
    try {
      await act(async () => { await i18n.changeLanguage('en') })
      expect(screen.getByRole('dialog', { name: 'Image preview' })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: 'Close' })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: '100%' })).toHaveFocus()
    } finally {
      view.unmount()
      await act(async () => { await i18n.changeLanguage('zh') })
    }
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
