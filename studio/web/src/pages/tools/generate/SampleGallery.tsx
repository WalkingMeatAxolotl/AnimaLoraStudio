import { useState } from 'react'
import { api } from '../../../api/client'

/** 测试 task 的最新一张图大图预览。
 *
 * 之前内部还渲染过缩略图列；用户反馈「我们已经有专门的 history rail，
 * SampleGallery 的 thumbnail 列冗余」 → 删。本组件只负责显示**最新**
 * 一张大图填满容器；空状态由 caller 处理。
 */
export default function SampleGallery({ samples, taskId }: {
  samples: Array<{ path: string; step?: number }>
  taskId: number
}) {
  const [errored, setErrored] = useState(false)

  if (!samples.length) return null

  const cur = samples[samples.length - 1]
  const filename = cur.path.split(/[\\/]/).pop() ?? cur.path
  const fullUrl = api.generateSampleUrl(taskId, filename)

  if (errored) {
    return (
      <div className="flex-1 grid place-items-center rounded-md border border-subtle bg-sunken text-fg-tertiary text-sm">
        图正在生成…（cache 暂未就绪，等下一张 step 进度）
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col items-center gap-2 min-h-0">
      <a
        href={fullUrl} target="_blank" rel="noreferrer"
        className="flex-1 min-h-0 flex items-center justify-center w-full"
      >
        <img
          key={fullUrl}
          src={fullUrl}
          className="rounded-md border border-subtle object-contain"
          style={{ maxWidth: '100%', maxHeight: '100%' }}
          alt={filename}
          onError={() => setErrored(true)}
          onLoad={() => setErrored(false)}
        />
      </a>
      <div className="text-xs text-fg-tertiary font-mono truncate w-full text-center">
        {filename}
      </div>
    </div>
  )
}
