import { useEffect, useState, type RefObject } from 'react'
import { useTranslation } from 'react-i18next'
import { schemaGroupLabel } from '../lib/schema'

/** SchemaForm 各分区的右侧锚点导航。
 *
 * active 判定走 scroll 监听：找滚动容器视口顶端下方 50px 处之上、DOM 顺序最后
 * 一个 section 作为 active。比 IntersectionObserver + rootMargin "激活带" 稳，
 * 不会出现多个 section 同时在带内时 active 来回跳。 */
export default function SchemaSectionIndex({
  groups,
  scrollContainer,
}: {
  groups: Array<{ key: string; label: string }>
  scrollContainer: RefObject<HTMLElement | null>
}) {
  const { t } = useTranslation()
  const [active, setActive] = useState<string>(groups[0]?.key ?? '')

  useEffect(() => {
    setActive(groups[0]?.key ?? '')
  }, [groups])

  useEffect(() => {
    const root = scrollContainer.current
    if (!root || groups.length === 0) return

    const compute = () => {
      const rootTop = root.getBoundingClientRect().top
      // 容器顶端往下 50px 当判定线 —— 这条线上方的最后一个 section 就是 active。
      // 50 是经验值：太小（如 0）切换太晚，section 头已经露出来才 active；太大
      // section 还远着就 active 上了。
      const probe = rootTop + 50
      let next = groups[0].key
      for (const g of groups) {
        const el = document.getElementById(`schema-group-${g.key}`)
        if (!el) continue
        if (el.getBoundingClientRect().top <= probe) {
          next = g.key
        } else {
          break
        }
      }
      setActive((prev) => (prev === next ? prev : next))
    }

    compute()
    root.addEventListener('scroll', compute, { passive: true })
    // 折叠 / 展开 section 会改变高度但不触发 scroll，靠 ResizeObserver 兜底。
    let ro: ResizeObserver | null = null
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(compute)
      ro.observe(root)
    }
    return () => {
      root.removeEventListener('scroll', compute)
      ro?.disconnect()
    }
  }, [groups, scrollContainer])

  const onJump = (key: string) => {
    const el = document.getElementById(`schema-group-${key}`)
    if (!el) return
    el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    setActive(key)
  }

  if (groups.length === 0) return null

  return (
    <nav className="flex flex-col gap-0.5">
      <div className="caption mb-2 px-2">{t('settings.pageIndex')}</div>
      {groups.map((g) => (
        <button
          key={g.key}
          onClick={() => onJump(g.key)}
          className={`text-left text-xs px-2 py-1.5 rounded-sm transition-colors border-l-2 bg-transparent ${
            active === g.key
              ? 'border-accent text-accent bg-accent-soft/40'
              : 'border-transparent text-fg-tertiary hover:text-fg-secondary hover:bg-overlay/40'
          }`}
        >
          {schemaGroupLabel(g.key, g.label, t)}
        </button>
      ))}
    </nav>
  )
}
