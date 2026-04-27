import { api, type DownloadFile } from '../api/client'

interface Props {
  pid: number
  bucket?: 'download'
  items: DownloadFile[]
  onPreview?: (name: string) => void
  emptyHint?: string
}

export default function FileList({
  pid,
  bucket = 'download',
  items,
  onPreview,
  emptyHint = '还没有图片',
}: Props) {
  if (items.length === 0) {
    return <p className="text-slate-500 text-sm">{emptyHint}</p>
  }
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2">
      {items.slice(0, 200).map((f) => (
        <button
          key={f.name}
          onClick={() => onPreview?.(f.name)}
          className="group aspect-square overflow-hidden rounded border border-slate-800 hover:border-cyan-700 bg-slate-900"
          title={f.name}
        >
          <img
            src={api.projectThumbUrl(pid, f.name, bucket)}
            alt={f.name}
            loading="lazy"
            className="w-full h-full object-cover group-hover:scale-105 transition-transform"
          />
        </button>
      ))}
      {items.length > 200 && (
        <p className="col-span-full text-xs text-slate-500">
          仅显示前 200 张（共 {items.length} 张）
        </p>
      )}
    </div>
  )
}
