// SettingsDrawer.tsx —— 右侧滑出的设置抽屉外壳。
//
// 渲染策略（解决"打开瞬间 mount 1000+ 行 Settings 卡帧"问题）：
//
// 1. 代码分割（React.lazy）：Settings 不进首屏 bundle，首次 open 才下载
// 2. Skeleton-first：抽屉外壳立即滑入显示骨架，下一帧（rAF）才真正 mount Settings；
//    动画跑在 GPU 上（transform + opacity），mount 卡顿被动画掩盖
// 3. 关闭对称：isOpen=false 时立刻卸掉 Settings，再播 slide-out；
//    避免 React 卸 8 个 tab 组件树跟动画抢主线程
//
// Settings 数据（secrets / catalog / SSE）住在 SettingsDataProvider 里常驻，
// 所以这里 mount/unmount Settings 不付重新拉数据的成本，只付 React render 的成本。
import { lazy, Suspense, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useSettingsDrawer } from '../lib/SettingsDrawer'

const SettingsPageLazy = lazy(() => import('../pages/tools/Settings'))

// 响应式宽度：
//   < 1600 viewport（笔记本 / 小桌面）：80vw —— 1280屏≈1024，1440屏≈1152，1600屏≈1280
//   1600–2559（含 1k FHD）：1280px 上限，避免大屏吞太多
//   ≥ 2560（2k QHD 及以上）：1960px —— 大屏给到接近 80% 可视宽
// 用 Tailwind arbitrary breakpoint 表达。
const DRAWER_WIDTH_CLASS = 'w-[min(1280px,80vw)] min-[2560px]:w-[1960px]'
const ANIM_MS = 220

export default function SettingsDrawer() {
  const { isOpen, close } = useSettingsDrawer()
  const [contentReady, setContentReady] = useState(false)

  // open: 让外壳先滑进来（GPU 动画），下一帧再 mount Settings
  // close: 立刻卸 Settings，外壳的 slide-out 在空骨架上跑
  useEffect(() => {
    if (!isOpen) {
      setContentReady(false)
      return
    }
    const id = requestAnimationFrame(() => setContentReady(true))
    return () => cancelAnimationFrame(id)
  }, [isOpen])

  return (
    <div
      aria-hidden={!isOpen}
      className={`absolute inset-0 z-30 ${isOpen ? '' : 'pointer-events-none'}`}
    >
      {/* backdrop：absolute 铺满 viewport，让 panel slide-in 中途不会从右边露出底页。
       *  之前用 flex + flex-1 时 backdrop 只占 panel 左边那条，panel 滑动期间右侧
       *  panel 槽位无人覆盖 → 看见底层页面 / 出现黑白闪屏。 */}
      <div
        onClick={() => void close()}
        className={`absolute inset-0 transition-[background-color,backdrop-filter,-webkit-backdrop-filter] ease-out ${
          isOpen ? 'bg-black/25 backdrop-blur-sm' : 'bg-black/0 backdrop-blur-0'
        }`}
        style={{ transitionDuration: `${ANIM_MS}ms` }}
        aria-label="close settings"
      />
      <aside
        role="dialog"
        aria-modal="true"
        className={`absolute top-0 right-0 bottom-0 flex flex-col bg-canvas border-l border-subtle shadow-2xl transition-transform ease-out ${DRAWER_WIDTH_CLASS} ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
        style={{ transitionDuration: `${ANIM_MS}ms` }}
      >
        {contentReady && isOpen ? (
          <Suspense fallback={<DrawerSkeleton />}>
            <SettingsPageLazy />
          </Suspense>
        ) : (
          <DrawerSkeleton />
        )}
      </aside>
    </div>
  )
}

function DrawerSkeleton() {
  const { t } = useTranslation()
  return (
    <div className="flex flex-col h-full">
      {/* 跟 PageHeader 同结构的占位条 —— 高度对齐，slide-in 切到真内容时不跳 */}
      <div className="px-6 pt-5 pb-4 bg-canvas border-b border-subtle">
        <div className="h-7 w-24 rounded bg-overlay" />
        <div className="mt-4 flex gap-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-7 w-16 rounded bg-overlay/60" />
          ))}
        </div>
      </div>
      <div className="flex-1 grid place-items-center text-fg-tertiary text-sm">
        {t('settings.drawerLoading')}
      </div>
    </div>
  )
}
