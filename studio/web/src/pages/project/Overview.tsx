import { useNavigate, useOutletContext } from 'react-router-dom'
import { api, type ProjectDetail, type Version } from '../../api/client'
import { useToast } from '../../components/Toast'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

export default function ProjectOverview() {
  const { project, reload } = useOutletContext<Ctx>()
  const navigate = useNavigate()
  const { toast } = useToast()

  const handleActivate = async (v: Version) => {
    try {
      await api.activateVersion(project.id, v.id)
      await reload()
      // 跳到该版本第一个未完成的 step（PP1 没决策器，统一进 curate）
      navigate(`/projects/${project.id}/v/${v.id}/curate`)
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  return (
    <div className="space-y-4">
      <header>
        <h2 className="text-lg font-semibold">概览</h2>
        <p className="text-sm text-slate-400">
          下载共 {project.download_image_count} 张 · 共 {project.versions.length} 个版本
        </p>
      </header>
      {project.note && (
        <p className="text-sm text-slate-300 border-l-2 border-slate-700 pl-3">
          {project.note}
        </p>
      )}

      <h3 className="text-sm font-semibold text-slate-300 mt-4">版本</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {project.versions.map((v) => (
          <article
            key={v.id}
            className={
              'rounded-xl border p-4 flex flex-col gap-1 ' +
              (v.id === project.active_version_id
                ? 'border-cyan-700 bg-cyan-950/20'
                : 'border-slate-700 bg-slate-800/40 hover:border-slate-600')
            }
          >
            <div className="flex items-center gap-2">
              <span className="text-base font-mono text-slate-100 flex-1 truncate">
                {v.label}
              </span>
              <span className="text-[10px] text-slate-500">{v.stage}</span>
            </div>
            <div className="text-xs text-slate-400 flex gap-3">
              <span>{v.stats?.train_image_count ?? 0} 训练图</span>
              <span>{v.stats?.reg_image_count ?? 0} 正则图</span>
              {v.stats?.has_output && (
                <span className="text-emerald-400">✓ 已训练</span>
              )}
            </div>
            {v.note && (
              <p className="text-xs text-slate-400 line-clamp-2">{v.note}</p>
            )}
            <div className="flex gap-2 mt-2">
              <button
                onClick={() => handleActivate(v)}
                className="text-xs px-2 py-1 rounded bg-cyan-600 hover:bg-cyan-500 text-white"
              >
                {v.id === project.active_version_id ? '打开' : '激活并打开'}
              </button>
            </div>
          </article>
        ))}
      </div>
    </div>
  )
}
