import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api } from '../../../api/client'
import ZoomableImage from '../../../components/ZoomableImage'

/** 测试 task 的最新一张图大图预览。
 *
 * 之前内部还渲染过缩略图列；用户反馈「我们已经有专门的 history rail，
 * SampleGallery 的 thumbnail 列冗余」 → 删。本组件只负责显示**最新**
 * 一张大图填满容器（内嵌缩放平移，useZoomPan）；空状态由 caller 处理。
 */
export default function SampleGallery({ samples, taskId }: {
  samples: Array<{ path: string; step?: number }>
  taskId: number
}) {
  const { t } = useTranslation()
  const [errored, setErrored] = useState(false)

  const cur = samples[samples.length - 1]
  const filename = cur ? (cur.path.split(/[\\/]/).pop() ?? cur.path) : ''
  const fullUrl = cur ? api.generateSampleUrl(taskId, filename) : ''
  // 换图重置 error（原实现靠 onLoad 清；ZoomableImage 内部 onLoad 用于取
  // natural size，这里改为 src 驱动重置 + 失败重试语义不变）
  useEffect(() => { setErrored(false) }, [fullUrl])

  if (!samples.length) return null

  if (errored) {
    return (
      <div className="flex-1 grid place-items-center rounded-md border border-subtle bg-sunken text-fg-tertiary text-sm">
        {t('generate.imageCachePending')}
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col items-center gap-2 min-h-0">
      {/* ZoomableImage 自带视口边框 / 背景 + readout 条，外层不再包样式 */}
      <div className="flex-1 min-h-0 w-full">
        <ZoomableImage
          key={fullUrl}
          src={fullUrl}
          alt={filename}
          onError={() => setErrored(true)}
        />
      </div>
      <div className="text-xs text-fg-tertiary font-mono truncate w-full text-center">
        {filename}
      </div>
    </div>
  )
}
