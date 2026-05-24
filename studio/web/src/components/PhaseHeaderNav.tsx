/** ADR-0007 §11.5-A / §11.8-B: phase 页面 header 右上角"上一步 / 下一步"按钮。
 *
 *  渲染规则：
 *  - 仅在 URL 为 version phase 页面（curate/tag/edit/reg/train）时显示
 *  - 项目级 step（download/preprocess）不显示（不属于 phase cursor）
 *  - 按钮永远可点（§11.5-A: 不 disabled，对新手友好）
 *  - 文案：[← ④ 打标]  [⑥ 正则集 →]（带编号 + phase 名）
 *  - 下一步行为：
 *    - 当前 phase 是 regularizing（SKIPPABLE）→ skip-phase（无校验直接过）
 *    - 否则 → advance-phase（校验失败给 toast）
 *    - 推进成功后 navigate 到下一个 phase 页面
 *  - ready phase（最后一个）next 按钮 hidden（用户在 Train 页用 "开始训练" 按钮）
 *  - curating phase（第一个）prev 按钮 hidden
 */
import { useTranslation } from 'react-i18next'
import { useLocation, useNavigate } from 'react-router-dom'
import { api, PHASE_ORDER, PHASE_SKIPPABLE, type VersionPhase } from '../api/client'
import { useProjectCtx } from '../context/ProjectContext'
import { useToast } from './Toast'

/** STEP key（URL）→ phase enum（§11.2 STEP_KEY_TO_PHASE 同步映射）。 */
const STEP_KEY_TO_PHASE: Record<string, VersionPhase> = {
  curate: 'curating',
  tag:    'tagging',
  edit:   'editing',
  reg:    'regularizing',
  train:  'ready',
}

const PHASE_TO_STEP_KEY: Record<VersionPhase, string> = {
  curating:     'curate',
  tagging:      'tag',
  editing:      'edit',
  regularizing: 'reg',
  ready:        'train',
}

/** phase enum → 编号（与 sidebar version-scope STEPS 对齐，1-5）。 */
const PHASE_TO_IDX: Record<VersionPhase, string> = {
  curating:     '1',
  tagging:      '2',
  editing:      '3',
  regularizing: '4',
  ready:        '5',
}

const PHASE_TO_LABEL_KEY: Record<VersionPhase, string> = {
  curating:     'nav.curate',
  tagging:      'nav.tag',
  editing:      'nav.tagEdit',
  regularizing: 'nav.reg',
  ready:        'nav.train',
}

export default function PhaseHeaderNav() {
  const { t } = useTranslation()
  const location = useLocation()
  const navigate = useNavigate()
  const { toast } = useToast()
  const ctx = useProjectCtx()

  // URL 匹配 /projects/:pid/v/:vid/:step
  const m = location.pathname.match(/^\/projects\/([^/]+)\/v\/([^/]+)\/([^/?#]+)/)
  if (!m) return null
  const [, pid, vid, stepKey] = m
  const focusPhase = STEP_KEY_TO_PHASE[stepKey]
  if (!focusPhase) return null  // step 不在 phase 集合（如 ready 之后的 monitor 等）

  if (!ctx) return null
  const projectId = Number(pid)
  const versionId = Number(vid)

  const focusIdx = PHASE_ORDER.indexOf(focusPhase)
  const prevPhase: VersionPhase | null = focusIdx > 0 ? PHASE_ORDER[focusIdx - 1] : null
  const nextPhase: VersionPhase | null =
    focusIdx < PHASE_ORDER.length - 1 ? PHASE_ORDER[focusIdx + 1] : null

  const handlePrev = () => {
    if (!prevPhase) return
    navigate(`/projects/${projectId}/v/${versionId}/${PHASE_TO_STEP_KEY[prevPhase]}`)
  }

  const handleNext = async () => {
    if (!nextPhase) return
    // 当前 focus phase 在 ready（最末）就不应到这里（按钮 hidden）
    const isSkippable = PHASE_SKIPPABLE.includes(focusPhase)
    try {
      const res = isSkippable
        ? await api.skipVersionPhase(projectId, versionId)
        : await api.advanceVersionPhase(projectId, versionId)
      if (!res.ok) {
        toast(res.reason || t('phaseNav.advanceFailed'), 'error')
        return
      }
      // 推进成功 → focus 跳下一个 phase 页面（cursor 也已经在 res.new_phase）
      navigate(`/projects/${projectId}/v/${versionId}/${PHASE_TO_STEP_KEY[nextPhase]}`)
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  // 都没按钮就不渲染整条 bar（节省空间）
  if (!prevPhase && (!nextPhase || focusPhase === 'ready')) return null

  return (
    <div className="px-4 py-2 border-b border-subtle bg-sunken flex items-center min-h-[44px]">
      <div className="flex items-center gap-2 ml-auto">
        {prevPhase && (
          <button
            type="button"
            onClick={handlePrev}
            className="btn btn-secondary text-sm"
            title={`${t('phaseNav.prevTitle')} ${PHASE_TO_IDX[prevPhase]} ${t(PHASE_TO_LABEL_KEY[prevPhase])}`}
          >
            ← {PHASE_TO_IDX[prevPhase]} {t(PHASE_TO_LABEL_KEY[prevPhase])}
          </button>
        )}
        {nextPhase && focusPhase !== 'ready' && (
          <button
            type="button"
            onClick={handleNext}
            className="btn btn-primary text-sm"
            title={`${t('phaseNav.nextTitle')} ${PHASE_TO_IDX[nextPhase]} ${t(PHASE_TO_LABEL_KEY[nextPhase])}`}
          >
            {PHASE_TO_IDX[nextPhase]} {t(PHASE_TO_LABEL_KEY[nextPhase])} →
          </button>
        )}
      </div>
    </div>
  )
}
