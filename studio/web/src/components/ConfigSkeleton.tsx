/**
 * ConfigSkeleton — Schema 表单的加载骨架。
 *
 * 抽自 Train.tsx 的 `ConfigSkeleton` 和 Presets.tsx 的 `SkeletonGroups`。两边
 * 结构（N 组 × M 行的"label bar + input bar"）几乎一致，只是外观风格不同：
 *   - Train 页：每组带 border + bg-surface 卡片，沉浸感强（variant='card'）
 *   - Presets 页：扁平展开节省空间（variant='flat'）
 *
 * 顺带修 Designer P2 提到的 a11y 问题：role="status" 放外层在卸载时不会播报
 * "加载完成"。这里 sr-only 文本保留，但本组件返回时 caller 应保证只在
 * `loading === true` 期间挂载它（卸载 = 隐式表示加载完成），调用方页面如果
 * 要"加载完成"播报应在页面顶层挂一个独立的 aria-live region。
 */

interface ConfigSkeletonProps {
  /** sr-only / aria-label 文本，默认 "加载配置中" */
  label?: string
  /** 各组的行数，默认 [5, 6, 4, 5] */
  groups?: number[]
  /** 'card' = 每组带卡片边框（Train 风格）；'flat' = 扁平（Presets 风格）。默认 'card' */
  variant?: 'card' | 'flat'
}

const DEFAULT_GROUPS = [5, 6, 4, 5]

export default function ConfigSkeleton({
  label = '加载配置中',
  groups = DEFAULT_GROUPS,
  variant = 'card',
}: ConfigSkeletonProps) {
  const isCard = variant === 'card'
  const Wrapper = isCard ? 'section' : 'div'
  const wrapperClass = isCard
    ? 'flex-1 min-h-0 overflow-y-auto pr-1 space-y-3'
    : 'flex flex-col gap-3 ui-skeleton'
  const groupClass = isCard
    ? 'ui-skeleton rounded-md border border-subtle bg-surface p-3.5'
    : 'flex flex-col gap-2'
  const titleBarClass = isCard
    ? 'h-3.5 w-32 rounded-sm bg-sunken mb-2.5'
    : 'h-3 w-24 rounded-sm bg-sunken opacity-60'
  const rowsContainerClass = isCard
    ? 'flex flex-col gap-2'
    : 'flex flex-col gap-1.5'
  const rowClass = isCard
    ? 'flex flex-col gap-1'
    : 'flex flex-col gap-0.5'
  const labelBarClass = isCard
    ? 'h-2.5 w-24 rounded-sm bg-sunken opacity-70'
    : 'h-2 w-20 rounded-sm bg-sunken opacity-50'
  const inputBarClass = isCard
    ? 'h-7 rounded-sm bg-canvas border border-subtle'
    : 'h-[26px] rounded-sm border border-subtle bg-canvas'

  return (
    <Wrapper className={wrapperClass} role="status" aria-label={label} aria-busy="true">
      {groups.map((rows, gi) => (
        <div key={gi} className={groupClass}>
          <div className={titleBarClass} />
          <div className={rowsContainerClass}>
            {Array.from({ length: rows }).map((_, ri) => (
              <div key={ri} className={rowClass}>
                <div className={labelBarClass} />
                <div className={inputBarClass} />
              </div>
            ))}
          </div>
        </div>
      ))}
      <span className="sr-only">{label}...</span>
    </Wrapper>
  )
}
