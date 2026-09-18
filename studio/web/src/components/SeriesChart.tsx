// 通用 step×value 折线图（纯 SVG）。
//
// 训练 monitor（loss / lr / d）和评估面板（指标随 checkpoint 的走势）都用它，所以
// 它不属于任何一方 —— 从 MonitorDashboard 抽出来只为让两边都能 import 而不互相依赖。
import { useLayoutEffect, useRef, useState } from 'react'

function calcEMA(data: number[], alpha = 0.02): number[] {
  if (!data.length) return []
  const out = [data[0]]
  for (let i = 1; i < data.length; i++) out.push(alpha * data[i] + (1 - alpha) * out[i - 1])
  return out
}

function downsample<T>(arr: T[], n: number): T[] {
  if (arr.length <= n) return arr
  return Array.from({ length: n }, (_, i) => arr[Math.round((i * (arr.length - 1)) / (n - 1))])
}

// ── StatCard ───────────────────────────────────────────────────────────────
// ── SeriesChart (pure SVG) ─────────────────────────────────────────────────
// 通用的 step×value 折线图：raw + EMA smooth 双线 + xy 轴 tick。
// loss / lr / d 都复用：传 rawColor/smoothColor 自定义配色，传 yFormat 控制
// y 轴数字格式（科学计数法 vs 定点）。

export function SeriesChart({ data, rawColor, smoothColor, fillColor, emaAlpha, yFormat, height, minHeight, axes = true, refLine }: {
  data: Array<{ step: number; value: number }>
  rawColor: string
  smoothColor: string
  fillColor?: string
  emaAlpha: number
  yFormat: (v: number) => string
  /** 固定像素高度（用于次要图，e.g. d value） */
  height?: number
  /** flex 模式下的最低像素高度；视口足够高时随父高度自动拉伸（用于主图，e.g. loss / lr） */
  minHeight?: number
  /** 是否绘制坐标轴 + tick label + 网格线；false 时退化为纯 sparkline 适合小高度图（d value） */
  axes?: boolean
  /** 可选的水平参考线（eval：纯底模 baseline 值）；y 范围会纳入它。 */
  refLine?: number
}) {
  // ResizeObserver 测真实像素尺寸，viewBox 用真实尺寸 → SVG 1:1 渲染，
  // 文本/线宽不会被 preserveAspectRatio 非等比缩放扭曲。
  const wrapperRef = useRef<HTMLDivElement | null>(null)
  const [size, setSize] = useState<{ w: number; h: number } | null>(null)

  useLayoutEffect(() => {
    const el = wrapperRef.current
    if (!el) return
    const measure = (w: number, h: number) => {
      if (w <= 0 || h <= 0) return
      setSize((prev) =>
        prev && Math.abs(prev.w - w) < 1 && Math.abs(prev.h - h) < 1 ? prev : { w, h },
      )
    }
    const rect = el.getBoundingClientRect()
    measure(rect.width, rect.height)
    const ro = new ResizeObserver(([entry]) => {
      const { width, height: h } = entry.contentRect
      measure(width, h)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const wrapperStyle: React.CSSProperties = height != null
    ? { height, width: '100%' }
    : { flex: 1, minHeight: minHeight ?? 0, width: '100%' }

  return (
    <div ref={wrapperRef} style={wrapperStyle}>
      {!data.length ? (
        <div className="grid place-items-center text-fg-tertiary text-sm h-full">
          等待数据…
        </div>
      ) : size ? (
        <ChartSvg
          data={data}
          W={size.w}
          H={size.h}
          rawColor={rawColor}
          smoothColor={smoothColor}
          fillColor={fillColor}
          emaAlpha={emaAlpha}
          yFormat={yFormat}
          axes={axes}
          refLine={refLine}
        />
      ) : null}
    </div>
  )
}

function ChartSvg({ data, W, H, rawColor, smoothColor, fillColor, emaAlpha, yFormat, axes, refLine }: {
  data: Array<{ step: number; value: number }>
  W: number
  H: number
  rawColor: string
  smoothColor: string
  fillColor?: string
  emaAlpha: number
  yFormat: (v: number) => string
  axes: boolean
  refLine?: number
}) {
  const pts = downsample(data, 600)
  const raw = pts.map((p) => p.value)
  // alpha = 1 → 跳过 EMA，纯 raw（avoid 双重曲线视觉冗余）
  const smooth = emaAlpha >= 0.999 ? raw : calcEMA(raw, emaAlpha)
  const steps = pts.map((p) => p.step)

  // sparkline 模式（axes=false）省掉 y label 左侧空间，path 填满；
  // 带 axes 时 PX 留 48 给 y tick（13pt 字宽约 7-8px × "0.0796" 6字 ≈ 46）；
  // PY 留 18 给 x tick（13pt 字高 ≈ 14，再留 4px 呼吸）。
  const PX = axes ? 48 : 0
  const PY = axes ? 18 : 2
  const RX = axes ? 8 : 0  // 右侧留白
  // y 范围按 smooth 算（无 smooth 时退化为 raw）—— raw 尖刺超出顶部会被裁掉，
  // 这是有意的：换取 smooth 信号占满高度、趋势可读。原 LossChart 同款行为。
  const refVals = emaAlpha >= 0.999 ? raw : smooth
  const hasRef = typeof refLine === 'number' && Number.isFinite(refLine)
  // y 范围纳入 baseline 参考线，保证它落在可视区（曲线在 base 线上/下方一目了然）。
  const minV = Math.min(...refVals, ...(hasRef ? [refLine as number] : []))
  const maxV = Math.max(...refVals, ...(hasRef ? [refLine as number] : []))
  const range = maxV - minV || Math.max(Math.abs(maxV), 1e-9) * 1e-3 || 1e-9
  const x = (i: number) => PX + (i / Math.max(1, pts.length - 1)) * (W - PX - RX)
  const y = (v: number) => PY + (1 - (v - minV) / range) * (H - PY - PY)

  const smoothPath = smooth.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')
  const areaPath = fillColor
    ? smoothPath + ` L${x(smooth.length - 1).toFixed(1)},${H - PY} L${PX},${H - PY}Z`
    : null
  const rawPath = raw.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')

  const yTicks = [minV, (minV + maxV) / 2, maxV].map((v) => ({
    v, y: y(v), label: yFormat(v),
  }))
  // 点少时（eval 只有几个 checkpoint）5 个分位会 round 到重复索引（如 3 点 →
  // 0,1,1,2,2），导致标签 "20 20 40 40" 叠在同一 x 上重叠。按索引去重。
  const xTicks = [...new Set(
    [0, 0.25, 0.5, 0.75, 1].map((t) => Math.round(t * Math.max(1, pts.length - 1))),
  )].map((i) => ({ i, x: x(i), label: String(steps[i] ?? '') }))

  const lastY = y(smooth[smooth.length - 1])
  const showSmoothLayer = emaAlpha < 0.999

  // viewBox 与真实尺寸 1:1，省掉 preserveAspectRatio="none" 的非等比缩放——
  // 文字/线宽在真实像素下渲染，不再被父容器宽高比扭曲。
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: '100%', display: 'block' }}>
      {axes && (
        <>
          {/* axis lines */}
          <line x1={PX} y1={PY} x2={PX} y2={H - PY} stroke="var(--border-subtle)" />
          <line x1={PX} y1={H - PY} x2={W - RX} y2={H - PY} stroke="var(--border-subtle)" />
          {/* grid */}
          {[0.25, 0.5, 0.75].map((t) => (
            <line key={t} x1={PX} y1={PY + t * (H - 2 * PY)} x2={W - RX} y2={PY + t * (H - 2 * PY)}
              stroke="var(--border-subtle)" strokeDasharray="3 3" />
          ))}
        </>
      )}
      {/* area (smooth fill, optional) */}
      {areaPath && <path d={areaPath} fill={fillColor} opacity="0.5" />}
      {/* raw —— smooth 模式下淡显，无 smooth 模式下当主线 */}
      <path
        d={rawPath}
        stroke={showSmoothLayer ? rawColor : smoothColor}
        strokeWidth={showSmoothLayer ? 1 : 2}
        strokeOpacity={showSmoothLayer ? 0.45 : 1}
        fill="none"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {/* smooth */}
      {showSmoothLayer && (
        <path d={smoothPath} stroke={smoothColor} strokeWidth="2" fill="none" strokeLinejoin="round" strokeLinecap="round" />
      )}
      {/* last point */}
      <circle cx={x(smooth.length - 1)} cy={lastY} r="4" fill={smoothColor} stroke="var(--bg-surface)" strokeWidth="2" />
      {/* baseline 参考线（纯底模）+ "base" 标注：曲线在它上方=优于底模，下方=不如底模 */}
      {hasRef && (
        <>
          <line
            x1={PX} y1={y(refLine as number)} x2={W - RX} y2={y(refLine as number)}
            stroke="var(--fg-tertiary)" strokeWidth="1" strokeDasharray="4 3" opacity="0.75"
          />
          <text
            x={W - RX - 1} y={y(refLine as number) - 3} fontSize="10"
            fill="var(--fg-tertiary)" fontFamily="var(--font-mono)" textAnchor="end"
          >base</text>
        </>
      )}
      {axes && (
        <>
          {/* y axis labels —— y offset +4.5 = fontSize/2 + 准基线微调，把字垂直居中到 tick */}
          {yTicks.map(({ v, y: yt, label }) => (
            <text key={v} x={PX - 4} y={yt + 4.5} fontSize="13" fill="var(--fg-tertiary)"
              fontFamily="var(--font-mono)" textAnchor="end">{label}</text>
          ))}
          {/* x axis labels —— 首/末两 tick 在 SVG 边缘上，middle 锚点会让一半字宽溢出被裁，
              改 start/end 锚点把字往内推；中间 tick 维持 middle 居中。 */}
          {xTicks.map(({ i: idx, x: xt, label }, i, arr) => {
            const anchor = i === 0 ? 'start' : i === arr.length - 1 ? 'end' : 'middle'
            return (
              <text key={idx} x={xt} y={H - 3} fontSize="13" fill="var(--fg-tertiary)"
                fontFamily="var(--font-mono)" textAnchor={anchor}>{label}</text>
            )
          })}
        </>
      )}
    </svg>
  )
}
