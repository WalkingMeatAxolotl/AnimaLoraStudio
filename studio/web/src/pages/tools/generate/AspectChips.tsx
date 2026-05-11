/** 画幅预设卡片（8 个常用比例 + Swap 横竖切换）。
 *
 * 每个卡片：glyph（按比例画的小矩形）+ 比例数字 + 名称 + W×H。
 * 选中态：accent 色描边 + 比例数字 accent 色。
 * grid-cols-4 两行排，4×2=8 张；sidebar 420px 内塞得下。
 *
 * Swap 按钮独立一个：物理交换 W↔H，命中预设则 aspect 跟着切（精确匹配
 * w/h 找当前 chip）；找不到时 aspect='custom'。
 */

export interface AspectPreset {
  /** 唯一 key，也是显示在大字号位置的比例标签 */
  ratio: string
  /** 'Square' / 'Landscape' / 'Portrait' */
  kind: 'Square' | 'Landscape' | 'Portrait'
  w: number
  h: number
}

export const ASPECT_PRESETS: AspectPreset[] = [
  { ratio: '3:2',   kind: 'Landscape', w: 1254, h: 836  },
  { ratio: '1:1',   kind: 'Square',    w: 1024, h: 1024 },
  { ratio: '4:5',   kind: 'Portrait',  w: 915,  h: 1144 },
  { ratio: '7:9',   kind: 'Portrait',  w: 896,  h: 1152 },
  { ratio: '3:4',   kind: 'Portrait',  w: 768,  h: 1024 },
  { ratio: '2:3',   kind: 'Portrait',  w: 832,  h: 1216 },
  { ratio: '9:16',  kind: 'Portrait',  w: 768,  h: 1344 },
  { ratio: '5:12',  kind: 'Portrait',  w: 640,  h: 1536 },
]

export type AspectName = string  // 用 "ratio" 做 name；自定义 = 'custom'

/** 给定 w/h 反查匹配的预设 ratio；不命中返回 'custom'。 */
export function aspectFromDimensions(w: number, h: number): AspectName {
  for (const p of ASPECT_PRESETS) {
    if (p.w === w && p.h === h) return p.ratio
  }
  return 'custom'
}

function RatioGlyph({ w, h, color }: { w: number; h: number; color: string }) {
  const max = 22
  const ratio = w / h
  const ww = ratio >= 1 ? max : max * ratio
  const hh = ratio >= 1 ? max / ratio : max
  return (
    <span style={{ width: max, height: max, display: 'inline-grid', placeItems: 'center' }}>
      <span style={{
        width: ww, height: hh,
        border: `1.5px solid ${color}`,
        borderRadius: 2,
      }} />
    </span>
  )
}

function PresetCard({
  preset, active, onClick,
}: {
  preset: AspectPreset
  active: boolean
  onClick: () => void
}) {
  const accent = active ? 'var(--accent)' : 'var(--fg-tertiary)'
  return (
    <button
      onClick={onClick}
      title={`${preset.kind} · ${preset.w}×${preset.h}`}
      style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        gap: 4,
        padding: '8px 4px',
        borderRadius: 'var(--r-md)',
        border: active ? '1px solid var(--accent)' : '1px solid var(--border-subtle)',
        background: active ? 'var(--accent-soft)' : 'var(--bg-sunken)',
        cursor: 'pointer',
        minWidth: 0,
      }}
    >
      <RatioGlyph w={preset.w} h={preset.h} color={accent} />
      <span
        className="font-mono"
        style={{
          fontSize: 13, fontWeight: 600,
          color: active ? 'var(--accent)' : 'var(--fg-secondary)',
        }}
      >
        {preset.ratio}
      </span>
      <span
        className="font-mono"
        style={{ fontSize: 10, color: 'var(--fg-tertiary)', whiteSpace: 'nowrap' }}
      >
        {preset.w}×{preset.h}
      </span>
    </button>
  )
}

export default function AspectChips({
  aspect, onPick,
}: {
  aspect: AspectName
  onPick: (ratio: AspectName, w?: number, h?: number) => void
}) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 6,
      }}
    >
      {ASPECT_PRESETS.map((p) => (
        <PresetCard
          key={p.ratio}
          preset={p}
          active={aspect === p.ratio}
          onClick={() => onPick(p.ratio, p.w, p.h)}
        />
      ))}
    </div>
  )
}
