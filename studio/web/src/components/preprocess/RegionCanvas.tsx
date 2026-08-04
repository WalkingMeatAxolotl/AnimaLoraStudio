import { useCallback, useRef, type PointerEvent } from 'react'
import type { RegionBox } from '../../api/client'

interface Props {
  src: string
  width: number
  height: number
  box: RegionBox | null
  onChange: (box: RegionBox) => void
}

function clamp(value: number, lo = 0, hi = 1): number {
  return Math.max(lo, Math.min(hi, value))
}

/** Draw-one primary rectangle editor. Drag anywhere to replace the old box. */
export default function RegionCanvas({ src, width, height, box, onChange }: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null)
  const startRef = useRef<{ x: number; y: number } | null>(null)

  const point = useCallback((event: PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current
    if (!svg) return { x: 0, y: 0 }
    const rect = svg.getBoundingClientRect()
    const imageRatio = width / height
    const viewportRatio = rect.width / rect.height
    let drawW = rect.width
    let drawH = rect.height
    let offsetX = 0
    let offsetY = 0
    if (viewportRatio > imageRatio) {
      drawW = rect.height * imageRatio
      offsetX = (rect.width - drawW) / 2
    } else {
      drawH = rect.width / imageRatio
      offsetY = (rect.height - drawH) / 2
    }
    return {
      x: clamp((event.clientX - rect.left - offsetX) / Math.max(drawW, 1)),
      y: clamp((event.clientY - rect.top - offsetY) / Math.max(drawH, 1)),
    }
  }, [width, height])

  const onPointerDown = (event: PointerEvent<SVGSVGElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId)
    startRef.current = point(event)
  }
  const onPointerMove = (event: PointerEvent<SVGSVGElement>) => {
    if (!startRef.current) return
    const end = point(event)
    const start = startRef.current
    onChange({
      x: Math.min(start.x, end.x),
      y: Math.min(start.y, end.y),
      w: Math.max(0.002, Math.abs(end.x - start.x)),
      h: Math.max(0.002, Math.abs(end.y - start.y)),
    })
  }
  const onPointerUp = (event: PointerEvent<SVGSVGElement>) => {
    if (startRef.current) onPointerMove(event)
    startRef.current = null
    event.currentTarget.releasePointerCapture(event.pointerId)
  }

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      className="w-full h-full select-none touch-none bg-black/70 cursor-crosshair"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={() => { startRef.current = null }}
      aria-label="primary-region-editor"
    >
      <image href={src} x="0" y="0" width={width} height={height} preserveAspectRatio="none" />
      {box && (
        <rect
          x={box.x * width}
          y={box.y * height}
          width={box.w * width}
          height={box.h * height}
          fill="rgba(255,130,30,0.13)"
          stroke="rgb(255,130,30)"
          strokeWidth={Math.max(2, Math.min(width, height) / 250)}
          vectorEffect="non-scaling-stroke"
          pointerEvents="none"
        />
      )}
    </svg>
  )
}
