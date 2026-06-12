import { memo, useEffect, useRef, useState, type SyntheticEvent } from 'react'
import { useTranslation } from 'react-i18next'
import { VirtuosoGrid } from 'react-virtuoso'

export interface ImageGridItem {
  name: string
  thumbUrl: string
  /** 鼠标悬停时显示在角标的小字（可选）：例如标签预览。 */
  meta?: string
  /** 常显小角标，cell 右下角（可选）。例如 "已处理"，用于在合并视图里区分状态。 */
  badge?: string
}

interface Props {
  items: ImageGridItem[]
  selected: Set<string>
  /** 单击 = checkbox 切换；shift+click = 区间选；详见 applySelection。 */
  onSelect: (name: string, e: React.MouseEvent) => void
  /** 鼠标悬停时回调：用于驱动外部「大图预览面板」。 */
  onHover?: (name: string) => void
  /** 全屏 modal 预览，由 cell 上的放大镜按钮触发（可选）。 */
  onPreview?: (name: string) => void
  /** 主点击行为：默认选择；activate 模式下普通点击交给外部打开/激活。 */
  onActivate?: (name: string) => void
  clickMode?: 'select' | 'activate'
  emptyHint?: string
  /** 测试 / 长列表场景下传入用于 grid 标识的 aria-label。 */
  ariaLabel?: string
  /** 列数（默认按宽度自适应）。FolderColumn 这种窄列会传 2-3。 */
  columnsClass?: string
  /** 当前「活跃」项（如 TagEdit 正在编辑的那张），名字精确匹配 item.name。
   *
   * **传了**这个 prop 就启用「解耦视觉」模式：
   * - border / ring 只跟 activeName 走（标识活跃项）
   * - checkbox 只跟 selected 走（标识多选）
   *
   * **不传**沿用旧行为：selected 同时驱动 border 和 checkbox（其他用 ImageGrid
   * 的页面，如 Curation / Download / Reg 未引入「活跃项」概念，行为不变）。 */
  activeName?: string
}

// 默认按容器宽度自动塞满：每格最小 120px，剩余宽度均分给最后一列；
// 容器越宽列越多，无需断点切换。
const DEFAULT_COLUMNS = 'grid-cols-[repeat(auto-fill,minmax(120px,1fr))]'

// 虚拟滚动 buffer：约 5-6 行 cell。暗主题 cell 底色是 #110f0b（接近纯黑），
// 滚动时新 mount 的 cell 在 img decode 完成前会闪一下黑色；overscan 足够大
// 才能让 buffer 区图提前 decode 好，进入视口直接显示而非"先黑后图"。代价是
// DOM 多 ~50 个节点 — 缩略图轻量，可接受。
const OVERSCAN_PX = 600

// 主导色 placeholder cache：thumbUrl → '#RRGGBB'。
//
// 模块级 Map（不是 React state）— 跨 ImageGrid 实例、跨 cell unmount/mount 都
// 保留。Cell 第一次 mount 时 lookup 拿不到色 → 显示默认 bg-sunken（黑）；img
// onLoad 后用 canvas 取色入 map + setState；之后用户滚出再滚回时 cell 重 mount
// → lookup 命中 → 立即用主导色填底，img 还在 decode 时用户看到的是色块而非黑
// 块，"滚动闪黑"消失。
//
// 缺点：首次浏览（cache 冷）还是黑 → 图；但常规浏览（滚来滚去）从第二次起就
// 有色。不持久化到 sessionStorage —— 每次刷页面 cache 重建几秒内就满，开销
// 可忽略；持久化反而要处理 mtime invalidation 之类。
const colorCache = new Map<string, string>()

// 复用单个 1×1 canvas：drawImage(img, 0, 0, 1, 1) 让浏览器内部做完整缩放求平均
// （比手动遍历 ImageData 快 1-2 个数量级）；getImageData 拿这一个像素 ≈ 平均色。
let _colorCanvas: HTMLCanvasElement | null = null
function extractAvgColor(img: HTMLImageElement): string | null {
  try {
    if (!_colorCanvas) {
      _colorCanvas = document.createElement('canvas')
      _colorCanvas.width = 1
      _colorCanvas.height = 1
    }
    const ctx = _colorCanvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) return null
    ctx.drawImage(img, 0, 0, 1, 1)
    const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data
    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
  } catch {
    // canvas tainted（跨域）/ jsdom 无真实 canvas / 其它失败 — 静默 fallback
    // 到默认 bg-sunken。生产环境 thumb 同源不会 tainted。
    return null
  }
}

export default function ImageGrid({
  items,
  selected,
  onSelect,
  onHover,
  onPreview,
  onActivate,
  clickMode = 'select',
  emptyHint,
  ariaLabel,
  columnsClass = DEFAULT_COLUMNS,
  activeName,
}: Props) {
  const { t } = useTranslation()
  if (items.length === 0) {
    return <p className="text-fg-tertiary text-sm py-2">{emptyHint ?? t('imageGrid.noImages')}</p>
  }
  const decoupled = activeName !== undefined

  // role="grid" + aria-label 放在外层 wrapper：VirtuosoGrid 内部 List/Item 包多
  // 层 div（scroller/list/item），role 直接挂内层会被 Virtuoso 改写 className/
  // style；外层 wrapper 是稳定的。getAllByRole('gridcell') 会穿透中间 div 找到
  // Cell 上的 role="gridcell"，AT / 测试都不受影响。
  return (
    <div role="grid" aria-label={ariaLabel} className="h-full">
      <VirtuosoGrid
        style={{ height: '100%' }}
        totalCount={items.length}
        overscan={OVERSCAN_PX}
        listClassName={`grid ${columnsClass} gap-1`}
        itemContent={(index) => {
          const it = items[index]
          const isSel = selected.has(it.name)
          const isActive = decoupled && it.name === activeName
          // border = 旧行为时跟 selected 走；解耦时跟 activeName 走
          const borderHighlight = decoupled ? isActive : isSel
          return (
            <Cell
              item={it}
              selected={isSel}
              borderHighlight={borderHighlight}
              onSelect={onSelect}
              onHover={onHover}
              onPreview={onPreview}
              onActivate={onActivate}
              clickMode={clickMode}
            />
          )
        }}
      />
    </div>
  )
}

/** Cell 用 memo 包起来：父组件每次因为 hover 改 focus 都会重渲，但绝大多数
 * cell 的 selected / onSelect / item 引用都没变，能跳过重渲，避免 N 张缩略图
 * 全部重新创建 DOM。
 *
 * `borderHighlight` 控制 accent border + ring（"高亮"视觉），跟 `selected`
 * （checkbox 状态）解耦：旧路径上两者一致，TagEdit 解耦模式下 border 跟
 * activeName 走，checkbox 跟多选走。 */
const Cell = memo(function Cell({
  item,
  selected,
  borderHighlight,
  onSelect,
  onHover,
  onPreview,
  onActivate,
  clickMode,
}: {
  item: ImageGridItem
  selected: boolean
  borderHighlight: boolean
  onSelect: (name: string, e: React.MouseEvent) => void
  onHover?: (name: string) => void
  onPreview?: (name: string) => void
  onActivate?: (name: string) => void
  clickMode: 'select' | 'activate'
}) {
  const { t } = useTranslation()
  // mount 时同步从 cache lookup —— 命中就立刻用主导色填底（避免黑闪），
  // miss 就 undefined → 类名里的 bg-sunken 兜底。lazy init 保证只查一次。
  const [bg, setBg] = useState<string | undefined>(() => colorCache.get(item.thumbUrl))
  // img 是否已 load —— 默认 false（opacity-0），onLoad 后 true（opacity-100）。
  // 配合 transition-opacity 让"色块 → 图"是 150ms 淡入而非突变，进一步柔化
  // cache 命中场景下的视觉跳变；cache miss 场景也从"黑 → 啪一下出图"变成
  // "黑 → 图淡入"。
  const [loaded, setLoaded] = useState(false)
  const imgRef = useRef<HTMLImageElement>(null)
  // unmount 时 abort 还在下载的缩略图（src='' 是 <img> 唯一的取消手段）。
  // 浏览器不会因为节点被移除就取消请求 —— 切路由后几十张半途的 thumb 会
  // 继续占满同源 HTTP/1.1 的 6 个连接，新页面的 /api fetch 全在队尾排队，
  // 用户视角就是"路由被图片卡住"。半途取消的图不进 HTTP 缓存，但后端
  // thumb_cache 已落盘 + 完整加载过的走 304，重进页面的代价很小。
  useEffect(() => {
    // mount 时抓住元素：unmount 时 React 已把 ref 置 null，cleanup 里直接读
    // imgRef.current 拿不到节点。脱离 DOM 的 <img> 改 src 同样会 abort 请求。
    const img = imgRef.current
    return () => {
      if (img && !img.complete) img.src = ''
    }
  }, [])
  const handleCellClick = (e: React.MouseEvent) => {
    if (clickMode === 'activate' && onActivate && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
      onActivate(item.name)
      return
    }
    onSelect(item.name, e)
  }

  const handleSelectionClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    onSelect(item.name, e)
  }

  // img 加载完成：(1) 取色入 cache（如果还没有）；(2) 标记 loaded → 触发淡入。
  const handleImgLoad = (e: SyntheticEvent<HTMLImageElement>) => {
    if (!colorCache.has(item.thumbUrl)) {
      const color = extractAvgColor(e.currentTarget)
      if (color) {
        colorCache.set(item.thumbUrl, color)
        setBg(color)
      }
    }
    setLoaded(true)
  }

  return (
    <div
      role="gridcell"
      aria-selected={selected}
      onMouseEnter={onHover ? () => onHover(item.name) : undefined}
      onClick={handleCellClick}
      title={item.meta ? `${item.name}\n${item.meta}` : item.name}
      style={bg ? { background: bg } : undefined}
      className={
        'group relative aspect-square overflow-hidden rounded border cursor-pointer select-none ' +
        (borderHighlight
          ? 'border-accent ring-2 ring-accent-soft'
          : 'border-subtle hover:border-dim') +
        ' bg-sunken'
      }
    >
      {/* 虚拟化场景不能用 loading="lazy"：cell 进入 DOM（包括 overscan 区）
       * 时浏览器不主动 load，要等真正进入视口才 fetch，overscan 的预热效果
       * 全废。这里改 eager（默认）让 Virtuoso 一 mount cell 浏览器就开 fetch
       * + decode，配合大 overscan 几乎看不到滚动闪。 */}
      <img
        ref={imgRef}
        src={item.thumbUrl}
        alt={item.name}
        decoding="async"
        draggable={false}
        // 让尚未开始的 thumb 请求排在数据 fetch 之后，页内操作不被图片饿死。
        // React 18 只透传小写 unknown attribute（camelCase fetchPriority 是
        // React 19 的事），spread 绕开 TS 对未知 prop 的检查。
        {...{ fetchpriority: 'low' }}
        onLoad={handleImgLoad}
        className={
          'w-full h-full object-cover pointer-events-none transition-opacity duration-150 ' +
          (loaded ? 'opacity-100' : 'opacity-0')
        }
      />
      <button
        type="button"
        onClick={handleSelectionClick}
        aria-label={`${selected ? t('common.deselect') : t('common.select')} ${item.name}`}
        className={
          'absolute top-1 left-1 w-5 h-5 rounded-sm flex items-center justify-center text-[12px] font-bold transition-opacity ' +
          (selected
            ? 'bg-accent text-accent-fg opacity-100'
            : 'bg-black/50 border border-subtle text-transparent opacity-0 group-hover:opacity-100')
        }
      >
        ✓
      </button>
      {/* 放大镜：悬停时出现，点击触发 modal 全屏预览（不影响选择状态） */}
      {onPreview && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation()
            onPreview(item.name)
          }}
          aria-label={`${t('common.preview')} ${item.name}`}
          className="absolute top-1 right-1 w-5 h-5 rounded-sm bg-black/60 text-white text-[11px] opacity-0 group-hover:opacity-100 hover:bg-black/80"
        >
          ⤢
        </button>
      )}
      {item.badge && (
        <div className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded-sm bg-accent text-accent-fg text-[10px] font-medium pointer-events-none">
          {item.badge}
        </div>
      )}
    </div>
  )
})

/** 给 caller 用的工具：单击 = checkbox 切换；shift+click = 区间选。
 *
 * - 单击已选中 → 取消选中
 * - 单击未选中 → 加入选中
 * - shift+click：从 anchor 到当前位置之间所有项加入选中（不取消已选中的）
 *
 * 注意：不再要求 ctrl/cmd —— 单击就是切换，符合「checkbox 多选」UX。
 */
export function applySelection(
  current: Set<string>,
  name: string,
  e: React.MouseEvent,
  names: string[],
  lastAnchor: string | null
): { next: Set<string>; anchor: string } {
  if (e.shiftKey && lastAnchor && names.includes(lastAnchor)) {
    const i = names.indexOf(lastAnchor)
    const j = names.indexOf(name)
    if (j === -1) return _toggle(current, name)
    const [lo, hi] = i < j ? [i, j] : [j, i]
    const next = new Set(current)
    for (let k = lo; k <= hi; k++) next.add(names[k])
    return { next, anchor: name }
  }
  return _toggle(current, name)
}

function _toggle(
  current: Set<string>,
  name: string
): { next: Set<string>; anchor: string } {
  const next = new Set(current)
  if (next.has(name)) next.delete(name)
  else next.add(name)
  return { next, anchor: name }
}
