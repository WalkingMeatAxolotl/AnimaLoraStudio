import { useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'

export type PreprocessTool = 'overview' | 'dedupe' | 'upscale' | 'crop' | 'inpaint'

interface ToolDef {
  id: PreprocessTool
  /** i18n key suffix under `preprocess.tools.*`. */
  i18nKey: string
  /** Disabled in this milestone; pill is dim placeholder. */
  disabled?: boolean
}

/** Overview comes first — it's the gallery + multi-select + undo entry that
 *  governs the dataset, not a transform like upscale/crop/inpaint. */
const TOOLS: ReadonlyArray<ToolDef> = [
  { id: 'overview', i18nKey: 'overview' },
  { id: 'dedupe',   i18nKey: 'dedupe' },
  { id: 'upscale',  i18nKey: 'upscale' },
  { id: 'crop',     i18nKey: 'crop' },
  { id: 'inpaint',  i18nKey: 'inpaint', disabled: true },
]

interface Props {
  current: PreprocessTool
  projectId: number
  versionId: number
}

/** Top-of-page tools bar shared by every preprocess sub-tool.
 *
 *  The tools (放大 / 裁剪 / 涂抹) are peer **tools**, not pipeline stages — they
 *  don't have a completion state, and any one can be used at any time in any
 *  order. URL convention（ADR 0010 后）：
 *  `/projects/:pid/v/:vid/preprocess?tool=...`. The query string lets the
 *  sidebar's `/preprocess` matcher stay simple AND keeps the parent route
 *  mounted across tool switches.
 */
export default function PreprocessToolsBar({ current, projectId, versionId }: Props) {
  const { t } = useTranslation()
  const base = `/projects/${projectId}/v/${versionId}/preprocess`
  return (
    <nav className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md border border-subtle bg-surface shrink-0">
      <span className="text-fg-tertiary font-medium uppercase tracking-wider text-[10px] mr-1">
        {t('preprocess.toolsLabel')}
      </span>
      {TOOLS.map((tool) => {
        const label = t(`preprocess.tools.${tool.i18nKey}`)
        const isActive = tool.id === current
        if (tool.disabled) {
          return (
            <span
              key={tool.id}
              className="px-2.5 py-1 rounded text-fg-disabled bg-overlay/40 cursor-not-allowed font-medium"
              title={t(`preprocess.tools.${tool.i18nKey}Title`, { defaultValue: '' })}
            >{label}</span>
          )
        }
        // overview is the default tool (no ?tool= query); everyone else needs a tool param
        const href = tool.id === 'overview' ? base : `${base}?tool=${tool.id}`
        if (isActive) {
          return (
            <span
              key={tool.id}
              className="px-2.5 py-1 rounded bg-accent text-accent-fg font-medium"
              aria-current="page"
            >{label}</span>
          )
        }
        return (
          <Link
            key={tool.id}
            to={href}
            className="px-2.5 py-1 rounded text-fg-secondary hover:bg-accent-soft hover:text-accent transition-colors font-medium"
          >{label}</Link>
        )
      })}
    </nav>
  )
}
