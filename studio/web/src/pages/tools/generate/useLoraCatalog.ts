import { useCallback, useMemo, useRef, useState } from 'react'
import { api, type LoraCkpt, type VersionStatus } from '../../../api/client'

/** 测试页 LoRA 选择数据的「懒级联」缓存。
 *
 * 取代旧 `useProjectLoras`（mount 时一把拉 listProjects + getProject×N +
 * listVersionLoraCkpts×M = 几十上百个请求）。改成按用户真正点进去的路径按需拉：
 *   - 打开 picker      → ensureProjects()  拉项目列表（1 次）
 *   - 选中某 project   → ensureVersions(pid) 拉该项目 versions（1 次/项目）
 *   - 选中某 version   → fetchCkpts(pid,vid) 拉该版本 ckpts（1 次/版本）
 * 不开 picker = 0 请求。三层各自缓存 + in-flight 去重，多个 picker 实例 / 反复
 * 切换都不重发。catalog 在 Generate 顶层建一份往下传，实例间共享缓存。
 *
 * 同一份 fetchCkpts 也供历史回填 / XY 轴绑定的「按需解析 ckpt path」复用。
 */

export interface LoraProjectOption {
  id: number
  title: string
}

export interface LoraVersionOption {
  id: number
  label: string
  status: VersionStatus
}

export interface LoraCatalog {
  /** 已加载的项目列表（空 = 还没加载或没有项目）。响应式。 */
  projects: LoraProjectOption[]
  projectsLoading: boolean
  /** 幂等触发加载项目列表（picker mount / 打开时调）。 */
  ensureProjects: () => void
  /** 可 await 版（snapshot 解析用）：已缓存即返，否则拉一次。 */
  loadProjects: () => Promise<LoraProjectOption[]>
  /** 某项目的版本列表（undefined = 还没加载）。响应式。 */
  versionsOf: (pid: number) => LoraVersionOption[] | undefined
  /** 幂等触发加载某项目 versions（picker 选了 project 时调）。 */
  ensureVersions: (pid: number) => void
  /** 拉某版本的 ckpts（缓存 + in-flight 去重；失败 rethrow 让 picker 显错）。 */
  fetchCkpts: (pid: number, vid: number) => Promise<LoraCkpt[]>
}

export function useLoraCatalog(): LoraCatalog {
  const [projects, setProjects] = useState<LoraProjectOption[]>([])
  const [projectsLoading, setProjectsLoading] = useState(false)
  const projectsReq = useRef<Promise<LoraProjectOption[]> | null>(null)

  const [versionsByPid, setVersionsByPid] = useState<Record<number, LoraVersionOption[]>>({})
  const versionReqs = useRef<Record<number, Promise<LoraVersionOption[]>>>({})

  const ckptCache = useRef<Record<string, LoraCkpt[]>>({})
  const ckptReqs = useRef<Record<string, Promise<LoraCkpt[]>>>({})

  const loadProjects = useCallback((): Promise<LoraProjectOption[]> => {
    if (projectsReq.current) return projectsReq.current
    setProjectsLoading(true)
    const req = api.listProjects()
      .then((ps) => {
        const opts = ps.map((p) => ({ id: p.id, title: p.title }))
        setProjects(opts)
        return opts
      })
      .catch(() => {
        projectsReq.current = null  // 失败允许下次重试
        return [] as LoraProjectOption[]
      })
      .finally(() => setProjectsLoading(false))
    projectsReq.current = req
    return req
  }, [])

  const ensureProjects = useCallback(() => { void loadProjects() }, [loadProjects])

  const loadVersions = useCallback((pid: number): Promise<LoraVersionOption[]> => {
    const existing = versionReqs.current[pid]
    if (existing) return existing
    const req = api.getProject(pid)
      .then((d) => {
        const opts = d.versions.map((v) => ({ id: v.id, label: v.label, status: v.status }))
        setVersionsByPid((m) => ({ ...m, [pid]: opts }))
        return opts
      })
      .catch(() => {
        delete versionReqs.current[pid]
        return [] as LoraVersionOption[]
      })
    versionReqs.current[pid] = req
    return req
  }, [])

  const ensureVersions = useCallback((pid: number) => { void loadVersions(pid) }, [loadVersions])

  const versionsOf = useCallback(
    (pid: number): LoraVersionOption[] | undefined => versionsByPid[pid],
    [versionsByPid],
  )

  const fetchCkpts = useCallback((pid: number, vid: number): Promise<LoraCkpt[]> => {
    const key = `${pid}:${vid}`
    const cached = ckptCache.current[key]
    if (cached) return Promise.resolve(cached)
    const inflight = ckptReqs.current[key]
    if (inflight) return inflight
    const req = api.listVersionLoraCkpts(pid, vid)
      .then((items) => {
        ckptCache.current[key] = items
        delete ckptReqs.current[key]
        return items
      })
      .catch((e) => {
        delete ckptReqs.current[key]
        throw e
      })
    ckptReqs.current[key] = req
    return req
  }, [])

  return useMemo<LoraCatalog>(
    () => ({
      projects, projectsLoading, ensureProjects, loadProjects,
      versionsOf, ensureVersions, fetchCkpts,
    }),
    [projects, projectsLoading, ensureProjects, loadProjects, versionsOf, ensureVersions, fetchCkpts],
  )
}
