import { useEffect, useRef, useState } from 'react'
import { api } from '../../../api/client'

/** 测试 task 的 sample 缩略图列 + 大图预览。 */
export default function SampleGallery({ samples, taskId }: {
  samples: Array<{ path: string; step?: number }>
  taskId: number
}) {
  const [active, setActive] = useState(0)
  const prevLen = useRef(0)

  useEffect(() => {
    if (samples.length > prevLen.current) setActive(samples.length - 1)
    prevLen.current = samples.length
  }, [samples.length])

  if (!samples.length) {
    return (
      <div className="grid place-items-center rounded-md border border-subtle bg-sunken text-fg-tertiary text-sm" style={{ minHeight: 220 }}>
        等待生成图…
      </div>
    )
  }

  const cur = samples[active]
  const filename = cur.path.split(/[\\/]/).pop() ?? cur.path
  const fullUrl = api.generateSampleUrl(taskId, filename)

  return (
    <div className="flex flex-col gap-2">
      <div className="flex gap-1.5 overflow-x-auto pb-0.5" style={{ scrollbarWidth: 'thin' }}>
        {samples.map((s, i) => {
          const fn = s.path.split(/[\\/]/).pop() ?? s.path
          return (
            <button
              key={i}
              onClick={() => setActive(i)}
              className={`shrink-0 w-14 h-14 rounded overflow-hidden border-2 p-0 cursor-pointer bg-transparent transition-colors ${
                i === active ? 'border-accent' : 'border-transparent hover:border-dim'
              }`}
            >
              <img src={api.generateSampleUrl(taskId, fn)} className="w-full h-full object-cover" alt="" />
            </button>
          )
        })}
      </div>
      <a href={fullUrl} target="_blank" rel="noreferrer">
        <img
          src={fullUrl}
          className="w-full rounded-md border border-subtle object-contain"
          style={{ maxHeight: 480 }}
          alt={filename}
        />
      </a>
      <div className="text-xs text-fg-tertiary font-mono truncate">{filename}</div>
    </div>
  )
}
