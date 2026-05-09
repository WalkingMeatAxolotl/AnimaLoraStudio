import { useState } from 'react'
import type { LoraEntry } from '../../../api/client'
import PathPicker from '../../../components/PathPicker'
import type { RecentLora } from './types'

/** 多 LoRA 编辑列表（commit 4 会被 InlineLoraPicker 替换；当前保持原 UX）。 */
export default function LoraList({ loras, onChange, recent }: {
  loras: LoraEntry[]
  onChange: (l: LoraEntry[]) => void
  recent: RecentLora[]
}) {
  const [pickerIdx, setPickerIdx] = useState<number | null>(null)
  const [recentOpenIdx, setRecentOpenIdx] = useState<number | null>(null)

  const add = () => onChange([...loras, { path: '', scale: 1.0 }])
  const del = (i: number) => onChange(loras.filter((_, idx) => idx !== i))
  const setPath = (i: number, path: string) =>
    onChange(loras.map((l, idx) => idx === i ? { ...l, path } : l))
  const setScale = (i: number, scale: number) =>
    onChange(loras.map((l, idx) => idx === i ? { ...l, scale } : l))

  return (
    <div className="flex flex-col gap-2">
      {loras.map((l, i) => (
        <div key={i} className="flex gap-1.5 items-center">
          <div className="flex-1 flex gap-1 items-center bg-sunken border border-dim rounded-md px-2 py-1.5">
            <span className="text-xs text-fg-tertiary shrink-0 w-4 text-center font-mono">{i + 1}</span>
            <input
              type="text"
              className="input input-mono flex-1 border-0 bg-transparent p-0 text-xs"
              style={{ outline: 'none', boxShadow: 'none' }}
              placeholder="LoRA 路径…"
              value={l.path}
              onChange={(e) => setPath(i, e.target.value)}
            />
            {recent.length > 0 && (
              <button
                onClick={() => setRecentOpenIdx(recentOpenIdx === i ? null : i)}
                className="btn btn-ghost btn-sm text-xs shrink-0 px-1.5 text-fg-tertiary"
                title="最近训出的 LoRA"
              >
                最近
              </button>
            )}
            <button
              onClick={() => setPickerIdx(i)}
              className="btn btn-ghost btn-sm text-xs shrink-0 px-1.5"
              title="浏览文件"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>
            </button>
          </div>
          <div className="flex items-center gap-1 shrink-0">
            <span className="text-xs text-fg-tertiary">×</span>
            <input
              type="number"
              className="input text-center text-sm"
              style={{ width: 60, padding: '5px 6px' }}
              min={0} max={2} step={0.05}
              value={l.scale}
              onChange={(e) => setScale(i, Number(e.target.value))}
              title="权重倍率"
            />
          </div>
          <button onClick={() => del(i)} className="btn btn-ghost btn-sm text-fg-tertiary hover:text-err shrink-0 px-1.5">×</button>
        </div>
      ))}
      <button onClick={add} className="btn btn-ghost btn-sm self-start text-xs text-fg-tertiary">
        + 添加 LoRA
      </button>

      {/* 最近 LoRA 浮层（按行下方展开） */}
      {recentOpenIdx !== null && recent.length > 0 && (
        <div className="rounded-md border border-subtle bg-overlay px-2 py-1.5 flex flex-col gap-px text-sm">
          <div className="caption pb-1">最近训出的 LoRA</div>
          {recent.slice(0, 12).map((r) => (
            <button
              key={r.path}
              onClick={() => {
                setPath(recentOpenIdx, r.path)
                setRecentOpenIdx(null)
              }}
              className="flex items-center gap-2 text-left px-2 py-1 rounded text-xs cursor-pointer border-none bg-transparent text-fg-secondary hover:bg-surface"
            >
              <span className="flex-1 truncate">{r.label}</span>
              <span className="font-mono text-fg-tertiary text-2xs truncate" style={{ maxWidth: 280 }}>
                {r.path}
              </span>
            </button>
          ))}
          <button
            onClick={() => setRecentOpenIdx(null)}
            className="btn btn-ghost btn-sm self-end text-2xs text-fg-tertiary mt-1"
          >
            关闭
          </button>
        </div>
      )}

      {pickerIdx !== null && (
        <PathPicker
          dirOnly={false}
          onPick={(p) => { setPath(pickerIdx, p); setPickerIdx(null) }}
          onClose={() => setPickerIdx(null)}
        />
      )}
    </div>
  )
}
