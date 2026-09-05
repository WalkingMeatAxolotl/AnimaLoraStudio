import { useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'

export type PreprocessTool = 'overview' | 'dedupe' | 'upscale' | 'crop' | 'inpaint'

interface ToolDef {
  id: PreprocessTool
  /** i18n key suffix under `preprocess.tools.*`. */
  i18nKey: string
}

/** Overview comes first — it's the gallery + multi-select + undo entry that
 *  governs the dataset, not a transform like upscale/crop/inpaint. */
const TOOLS: ReadonlyArray<ToolDef> = [
  { id: 'overview', i18nKey: 'overview' },
  { id: 'dedupe',   i18nKey: 'dedupe' },
  { id: 'upscale',  i18nKey: 'upscale' },
  { id: 'crop',     i18nKey: 'crop' },
  { id: 'inpaint',  i18nKey: 'inpaint' },
]

interface Props {
  current: PreprocessTool
  projectId: number
  versionId: number
}

/** Route navigation, not local content tabs. The underline recipe is shared
 * with Tabs, but every tool remains a Link with native browser navigation.
 * Overview omits the query parameter; other tools use ?tool=... (ADR 0010).
 * The local horizontal scrollport keeps every tool reachable on compact desktop.
 */
export default function PreprocessToolsBar({ current, projectId, versionId }: Props) {
  const { t } = useTranslation()
  const base = `/projects/${projectId}/v/${versionId}/preprocess`
  return (
    <nav aria-label={t('preprocess.toolsLabel')} className="ui-selection-group ui-selection-underline ui-selection-md px-page shrink-0">
      {TOOLS.map((tool) => {
        const label = t(`preprocess.tools.${tool.i18nKey}`)
        const isActive = tool.id === current
        // Overview is the default tool; preserve native Link behavior even when active.
        const href = tool.id === 'overview' ? base : `${base}?tool=${tool.id}`
        return (
          <Link
            key={tool.id}
            to={href}
            aria-current={isActive ? 'page' : undefined}
            data-state={isActive ? 'active' : 'inactive'}
            className="ui-selection-item"
          >{label}</Link>
        )
      })}
    </nav>
  )
}
