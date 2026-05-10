import { useEffect, useRef, useState } from 'react'
import { api, type LoraCkpt, type StateCkpt, type VersionCkptGroup } from '../api/client'

/**
 * 字段内 dropdown picker：用户点字段旁的「📁 浏览本项目」按钮触发，
 * 弹出按 version 分组的可用文件列表，选中后写绝对路径回字段。
 *
 * 跟 PathPicker 的区别：不显示文件系统树，只显示语义化文件 label
 * （"baseline / step 2476"）。用户看不到深路径，但底层 onChange 仍写入
 * 真实绝对路径，保持 schema 字段值的兼容性。
 *
 * resume_state 字段 → kind="state"，列项目所有 versions 的 training_state_step*.pt
 * resume_lora  字段 → kind="lora"，列项目所有 versions 的 *.safetensors（含 final）
 *
 * 外部文件 / 别项目的 ckpt 用户直接在字段 input 手填即可，本 picker 只覆盖
 * 「接本项目某 version 的产出」这个最常见路径。
 */
export default function ResumeFieldPicker({
  pid,
  kind,
  value,
  onChange,
  onClose,
  anchorRef,
}: {
  pid: number
  kind: 'state' | 'lora'
  value: string
  onChange: (path: string) => void
  onClose: () => void
  anchorRef: React.RefObject<HTMLElement | null>
}) {
  const [stateGroups, setStateGroups] = useState<VersionCkptGroup<StateCkpt>[] | null>(null)
  const [loraGroups, setLoraGroups] = useState<VersionCkptGroup<LoraCkpt>[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const popRef = useRef<HTMLDivElement | null>(null)

  // 拉数据
  useEffect(() => {
    let alive = true
    if (kind === 'state') {
      api.listProjectStateCkpts(pid)
        .then((g) => { if (alive) setStateGroups(g) })
        .catch((e) => { if (alive) setError(String(e)) })
    } else {
      api.listProjectLoraCkpts(pid)
        .then((g) => { if (alive) setLoraGroups(g) })
        .catch((e) => { if (alive) setError(String(e)) })
    }
    return () => { alive = false }
  }, [pid, kind])

  // 点外面关闭 + Esc 关闭
  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (popRef.current?.contains(e.target as Node)) return
      if (anchorRef.current?.contains(e.target as Node)) return
      onClose()
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [onClose, anchorRef])

  const groups = kind === 'state' ? stateGroups : loraGroups
  const loaded = groups !== null
  const totalItems = groups?.reduce((s, g) => s + g.items.length, 0) ?? 0

  return (
    <div
      ref={popRef}
      className="absolute z-40 mt-1 w-[420px] max-h-[360px] overflow-y-auto rounded-md border border-dim bg-elevated shadow-xl text-xs"
      style={{ top: '100%', left: 0 }}
    >
      {error ? (
        <div className="px-3 py-2 text-err">{error}</div>
      ) : !loaded ? (
        <div className="px-3 py-2 text-fg-tertiary italic">加载中…</div>
      ) : totalItems === 0 ? (
        <div className="px-3 py-2 text-fg-tertiary italic">
          项目还没产出{kind === 'state' ? ' training_state_step*.pt' : ' LoRA ckpt'} —
          {kind === 'state' ? '先按 save_every 跑一轮训练' : '先训出至少一个 ckpt'}
        </div>
      ) : (
        groups!.map((g) => (
          <div key={g.version_id} className="border-b border-subtle last:border-0">
            <div className="px-3 py-1.5 bg-sunken text-fg-secondary font-mono font-semibold sticky top-0">
              {g.label}
              {g.items.length === 0 && (
                <span className="ml-2 text-fg-tertiary font-normal italic">(无)</span>
              )}
            </div>
            {kind === 'state'
              ? (g.items as StateCkpt[]).map((it) => (
                  <PickRow
                    key={it.path}
                    label={it.label}
                    selected={it.path === value}
                    onPick={() => { onChange(it.path); onClose() }}
                  />
                ))
              : (g.items as LoraCkpt[]).map((it) => (
                  <PickRow
                    key={it.path}
                    label={it.label}
                    selected={it.path === value}
                    onPick={() => { onChange(it.path); onClose() }}
                  />
                ))}
          </div>
        ))
      )}
    </div>
  )
}

function PickRow({
  label, selected, onPick,
}: {
  label: string
  selected: boolean
  onPick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onPick}
      className={
        'w-full text-left px-4 py-1.5 font-mono cursor-pointer transition-colors ' +
        (selected
          ? 'bg-accent-soft text-accent-fg'
          : 'text-fg-primary hover:bg-overlay')
      }
    >
      {selected && <span className="mr-1.5">✓</span>}
      {label}
    </button>
  )
}
