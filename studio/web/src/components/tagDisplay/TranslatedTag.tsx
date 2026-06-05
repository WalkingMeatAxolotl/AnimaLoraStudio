/** Chip 上显示的 tag 文本：根据全局 toggle + dict 命中，渲染英文 or 英文 + 中文翻译。
 *
 * 用法：在原本写 `{tag}` 的地方换成 `<TranslatedTag tag={tag} />`。chip 容器
 * 自己保留 border / padding / 颜色 —— 本组件只关心文本内容。
 *
 * 显示文本规则：caller 传啥就显示啥（保留 `_` 或者空格形式）；只是字典查找时
 * 内部归一成空格（dict 存储一律 canonical）。这样 Reg/booru 边界 tag 也能命中。
 */
import { useTagDict } from '../../tagDict/store'
import { useShowTagTranslation } from '../../tagDict/showToggle'

export function TranslatedTag({ tag }: { tag: string }) {
  const [show] = useShowTagTranslation()
  const dict = useTagDict()
  const zh = dict.entries.get(tag) ?? dict.entries.get(tag.replace(/_/g, ' '))
  if (!show || !zh || zh.length === 0) {
    return <span>{tag}</span>
  }
  return (
    <span>
      {tag}
      <span className="ml-1.5 text-fg-tertiary">{zh.join(' ')}</span>
    </span>
  )
}
