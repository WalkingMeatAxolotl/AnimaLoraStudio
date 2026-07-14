/**
 * useZoomPan — 图片 / 画布查看视口：滚轮指针中心缩放 + 拖拽平移 + fit/100%。
 *
 * 从涂抹页 InpaintCanvas 的视图层抽出（PR-B 系列），供全部单图查看场景复用：
 * FullscreenViewer / ImagePreviewModal / TagEdit 内嵌单图 / InpaintCanvas。
 *
 * 用法（查看器，左键拖拽 = pan）：
 *   const zp = useZoomPan({ contentW, contentH, primaryButtonPans: true })
 *   <div ref={zp.wrapRef} {...zp.handlers} style={{ touchAction: 'none', ... }}>
 *     <img ref={zp.contentRef} style={{ position:'absolute', left:0, top:0,
 *          transformOrigin:'0 0' }} ... />
 *   </div>
 *
 * 用法（画笔类，左键留给画笔，空格 / 中键 = pan）：primaryButtonPans 缺省
 * false，调用方在自己的 pointer handler 里先问 panPointerDown/Move —— 返回
 * true 表示本事件是 pan 手势已被接管，画笔逻辑应跳过。
 *
 * 行为约定（与涂抹页一致）：
 * - wheel 缩放以指针为中心（non-passive，preventDefault 防页面滚动）
 * - 空格按住 = pan 修饰键（表单元素聚焦时不劫持）
 * - 容器 resize 时若用户没手动动过视图则保持 fit
 * - contentW/H 变化（换图）自动重新 fit
 * - 视图状态放 ref 直改 DOM transform —— pan/zoom 高频路径不走 React 渲染，
 *   只有 zoomPct（readout 用）走 state
 */
import { useCallback, useEffect, useRef, useState } from 'react'

export interface ZoomPanView {
  scale: number
  tx: number
  ty: number
}

export interface UseZoomPanOptions {
  /** 内容原始尺寸（图片 naturalWidth/Height 或 canvas 尺寸）。0 时视为未就绪。 */
  contentW: number
  contentH: number
  /** true = 左键拖拽即 pan（纯查看器）；false = 仅空格 / 中键（画笔类）。 */
  primaryButtonPans?: boolean
  /** 缩放的 DOM 应用方式：
   *  - 'transform'（默认）：translate+scale，纯 composite 最快；内容整体
   *    视觉缩放（含边框 / 子元素）。
   *  - 'size'：改宽高 + translate 平移 —— 内容按新尺寸重排，px 单位的
   *    border / handle / label 不跟着缩放（DOM overlay 编辑器用，如
   *    FreeCropEditor 的 % 定位 rect）。每帧触发 layout，小子树可接受。 */
  applyMode?: 'transform' | 'size'
  /** fit 时的留边系数，默认 0.98。 */
  fitPadding?: number
  /** 缩放上下限（scale 值）。 */
  minScale?: number
  maxScale?: number
}

export function useZoomPan({
  contentW,
  contentH,
  primaryButtonPans = false,
  applyMode = 'transform',
  fitPadding = 0.98,
  minScale = 0.02,
  maxScale = 32,
}: UseZoomPanOptions) {
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const contentRef = useRef<HTMLElement | null>(null)
  const viewRef = useRef<ZoomPanView>({ scale: 1, tx: 0, ty: 0 })
  const interactedRef = useRef(false)
  const spaceRef = useRef(false)
  const panRef = useRef<{ x: number; y: number } | null>(null)
  // pan 手势发生过拖动（区分「点击遮罩关闭」与「拖完松手」——查看器 modal 用）
  const draggedRef = useRef(false)
  const [zoomPct, setZoomPct] = useState(100)

  const applyToDom = useCallback(() => {
    const v = viewRef.current
    const el = contentRef.current
    if (!el) return
    if (applyMode === 'size') {
      el.style.width = `${contentW * v.scale}px`
      el.style.height = `${contentH * v.scale}px`
      el.style.transform = `translate(${v.tx}px, ${v.ty}px)`
    } else {
      el.style.transform = `translate(${v.tx}px, ${v.ty}px) scale(${v.scale})`
    }
  }, [applyMode, contentW, contentH])

  const applyView = useCallback(() => {
    applyToDom()
    setZoomPct(Math.round(viewRef.current.scale * 100))
  }, [applyToDom])

  const fit = useCallback(() => {
    const wrap = wrapRef.current
    if (!wrap || contentW <= 0 || contentH <= 0) return
    const rect = wrap.getBoundingClientRect()
    if (rect.width < 10 || rect.height < 10) return
    const scale = Math.min(rect.width / contentW, rect.height / contentH) * fitPadding
    viewRef.current = {
      scale,
      tx: (rect.width - contentW * scale) / 2,
      ty: (rect.height - contentH * scale) / 2,
    }
    interactedRef.current = false
    applyView()
  }, [contentW, contentH, fitPadding, applyView])

  /** 以视口中心为锚点回 100%。 */
  const reset100 = useCallback(() => {
    const wrap = wrapRef.current
    if (!wrap) return
    const rect = wrap.getBoundingClientRect()
    const v = viewRef.current
    const cx = rect.width / 2
    const cy = rect.height / 2
    viewRef.current = {
      scale: 1,
      tx: cx - (cx - v.tx) / v.scale,
      ty: cy - (cy - v.ty) / v.scale,
    }
    interactedRef.current = true
    applyView()
  }, [applyView])

  /** 屏幕坐标 → 内容坐标（原始像素）。容器未挂载返回 null。 */
  const toContentPoint = useCallback((clientX: number, clientY: number) => {
    const wrap = wrapRef.current
    if (!wrap) return null
    const rect = wrap.getBoundingClientRect()
    const v = viewRef.current
    return {
      x: (clientX - rect.left - v.tx) / v.scale,
      y: (clientY - rect.top - v.ty) / v.scale,
    }
  }, [])

  // 换图 / 初次就绪自动 fit
  useEffect(() => {
    fit()
  }, [fit])

  // 容器 resize：用户没手动动过视图时保持 fit
  useEffect(() => {
    const wrap = wrapRef.current
    if (!wrap) return
    const ro = new ResizeObserver(() => {
      if (!interactedRef.current) fit()
    })
    ro.observe(wrap)
    return () => ro.disconnect()
  }, [fit])

  // wheel zoom —— React onWheel 是 passive，必须手动挂 non-passive 才能 preventDefault
  useEffect(() => {
    const wrap = wrapRef.current
    if (!wrap) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = wrap.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      const v = viewRef.current
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
      const next = Math.min(maxScale, Math.max(minScale, v.scale * factor))
      viewRef.current = {
        scale: next,
        tx: mx - ((mx - v.tx) * next) / v.scale,
        ty: my - ((my - v.ty) * next) / v.scale,
      }
      interactedRef.current = true
      applyView()
    }
    wrap.addEventListener('wheel', onWheel, { passive: false })
    return () => wrap.removeEventListener('wheel', onWheel)
  }, [applyView, minScale, maxScale])

  // 空格 = pan 修饰键（表单元素聚焦时不劫持）
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return
      const el = e.target as HTMLElement | null
      if (
        el &&
        (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)
      ) {
        return
      }
      spaceRef.current = true
      e.preventDefault()
    }
    const up = (e: KeyboardEvent) => {
      if (e.code === 'Space') spaceRef.current = false
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [])

  /** pointerdown 是否为 pan 手势；是则接管（调用方跳过自己的逻辑）。 */
  const panPointerDown = useCallback((e: React.PointerEvent): boolean => {
    const isPan =
      spaceRef.current ||
      e.button === 1 ||
      (primaryButtonPans && e.button === 0)
    if (!isPan) return false
    panRef.current = { x: e.clientX, y: e.clientY }
    draggedRef.current = false
    return true
  }, [primaryButtonPans])

  /** pan 进行中的移动；返回是否消费了本事件。 */
  const panPointerMove = useCallback((e: React.PointerEvent): boolean => {
    if (!panRef.current) return false
    const dx = e.clientX - panRef.current.x
    const dy = e.clientY - panRef.current.y
    if (dx !== 0 || dy !== 0) draggedRef.current = true
    panRef.current = { x: e.clientX, y: e.clientY }
    viewRef.current = {
      ...viewRef.current,
      tx: viewRef.current.tx + dx,
      ty: viewRef.current.ty + dy,
    }
    interactedRef.current = true
    applyToDom()
    return true
  }, [applyToDom])

  /** 结束 pan；返回本次手势是否发生过拖动（modal 据此决定是否当作点击关闭）。 */
  const endPan = useCallback((): boolean => {
    panRef.current = null
    const dragged = draggedRef.current
    draggedRef.current = false
    return dragged
  }, [])

  /** 查看器一把梭绑定（画笔类不要用——自己组合 panPointerDown/Move/endPan）。 */
  const handlers = {
    onPointerDown: (e: React.PointerEvent<HTMLDivElement>) => {
      if (panPointerDown(e)) e.currentTarget.setPointerCapture(e.pointerId)
    },
    onPointerMove: (e: React.PointerEvent<HTMLDivElement>) => {
      panPointerMove(e)
    },
    onPointerUp: () => {
      endPan()
    },
    onPointerCancel: () => {
      endPan()
    },
  }

  return {
    wrapRef,
    contentRef,
    viewRef,
    zoomPct,
    fit,
    reset100,
    toContentPoint,
    interactedRef,
    spaceRef,
    panPointerDown,
    panPointerMove,
    endPan,
    handlers,
  }
}
