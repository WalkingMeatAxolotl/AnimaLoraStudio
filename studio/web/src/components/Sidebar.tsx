import { NavLink } from 'react-router-dom'

const links: Array<{ to: string; label: string; phase?: string }> = [
  { to: '/', label: '监控' },
  { to: '/configs', label: '配置' },
  { to: '/datasets', label: '数据集', phase: 'P4' },
  { to: '/queue', label: '队列', phase: 'P3' },
]

export default function Sidebar() {
  return (
    <aside className="w-48 shrink-0 flex flex-col py-6 px-3 border-r border-slate-800">
      <div className="px-3 mb-6">
        <div className="text-lg font-bold bg-gradient-to-r from-cyan-400 to-violet-400 bg-clip-text text-transparent">
          AnimaStudio
        </div>
        <div className="text-xs text-slate-500 mt-0.5">v0.1</div>
      </div>
      <nav className="flex-1 space-y-1">
        {links.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end
            className={({ isActive }) =>
              'flex items-center justify-between px-3 py-2 rounded text-sm transition-colors ' +
              (isActive
                ? 'bg-cyan-600/20 text-cyan-300'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60')
            }
          >
            <span>{l.label}</span>
            {l.phase && (
              <span className="text-xs text-slate-600">{l.phase}</span>
            )}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
