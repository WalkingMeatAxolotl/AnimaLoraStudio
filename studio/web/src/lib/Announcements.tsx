// Announcements.tsx —— 公告栏全局数据层（announcement-center Phase 1）。
//
// 持有 posts（/api/announcements）+ read 状态（localStorage，按篇）+ 公告栏开关 +
// 「有可用更新」检查（D8：从 Topbar 迁进来）。Topbar 铃铛 与 AnnouncementCenter
// modal 共用本 context，读一篇 → 红点实时同步。
//
// 触发规则（仿 onboarding 的 localStorage 思路）：
// - 全新安装（无 lastVersion 记录）：当前全部 post 标记已读、不自动弹。
// - 版本号变了且有未读 → 自动弹一次；记 lastVersion，同版本不再弹。
// - read 状态纯前端，不进后端。
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { api, type AnnouncementPost } from '../api/client'

const READ_KEY = 'studio.announcements.read'
const LASTVER_KEY = 'studio.announcements.lastVersion'

function loadReadIds(): Set<string> {
  try {
    const raw = JSON.parse(localStorage.getItem(READ_KEY) ?? '[]')
    return new Set(Array.isArray(raw) ? (raw as string[]) : [])
  } catch {
    return new Set()
  }
}
function saveReadIds(s: Set<string>) {
  try { localStorage.setItem(READ_KEY, JSON.stringify([...s])) } catch { /* ignore */ }
}
function getLastVersion(): string | null {
  try { return localStorage.getItem(LASTVER_KEY) } catch { return null }
}
function setLastVersion(v: string) {
  try { localStorage.setItem(LASTVER_KEY, v) } catch { /* ignore */ }
}

interface UpdateInfo {
  has_update: boolean
  latest_tag: string | null
  latest_commit: string
}

interface AnnouncementsCtx {
  posts: AnnouncementPost[]
  readIds: Set<string>
  unreadCount: number
  open: boolean
  openCenter: () => void
  closeCenter: () => void
  markRead: (id: string) => void
  updateInfo: UpdateInfo | null
}

const Ctx = createContext<AnnouncementsCtx | null>(null)

export function AnnouncementsProvider({ children }: { children: ReactNode }) {
  const [posts, setPosts] = useState<AnnouncementPost[]>([])
  const [readIds, setReadIds] = useState<Set<string>>(() => loadReadIds())
  const [open, setOpen] = useState(false)
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null)

  // 拉公告 + 当前版本 → 决定是否自动弹。
  useEffect(() => {
    let cancelled = false
    Promise.all([
      api.getAnnouncements().catch(() => [] as AnnouncementPost[]),
      api.health().then((h) => h.version).catch(() => null),
    ]).then(([fetched, version]) => {
      if (cancelled) return
      setPosts(fetched)
      const ids = fetched.map((p) => p.id)
      const last = getLastVersion()
      if (last === null) {
        // 全新安装：全标已读，不弹（新用户不被历史公告/迁移提示打扰）。
        const all = new Set(ids)
        setReadIds(all)
        saveReadIds(all)
        if (version) setLastVersion(version)
        return
      }
      const cur = loadReadIds()
      const unread = ids.filter((id) => !cur.has(id))
      if (unread.length > 0 && version && version !== last) {
        setOpen(true)
        setLastVersion(version) // 同版本不再自动弹；红点仍按篇保留
      }
    })
    return () => { cancelled = true }
  }, [])

  // 「有可用更新」检查（D8：原 Topbar 的 pill 逻辑迁到这里）。
  useEffect(() => {
    let cancelled = false
    api.checkSystemUpdate('master').then((r) => {
      if (!cancelled && r.has_update) {
        setUpdateInfo({ has_update: true, latest_tag: r.latest_tag, latest_commit: r.latest_commit })
      }
    }).catch(() => { /* silent */ })
    return () => { cancelled = true }
  }, [])

  const markRead = useCallback((id: string) => {
    setReadIds((prev) => {
      if (prev.has(id)) return prev
      const next = new Set(prev)
      next.add(id)
      saveReadIds(next)
      return next
    })
  }, [])

  const unreadCount = useMemo(
    () => posts.filter((p) => !readIds.has(p.id)).length,
    [posts, readIds],
  )
  const openCenter = useCallback(() => setOpen(true), [])
  const closeCenter = useCallback(() => setOpen(false), [])

  const value = useMemo<AnnouncementsCtx>(() => ({
    posts, readIds, unreadCount, open, openCenter, closeCenter, markRead, updateInfo,
  }), [posts, readIds, unreadCount, open, openCenter, closeCenter, markRead, updateInfo])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useAnnouncements(): AnnouncementsCtx {
  const v = useContext(Ctx)
  if (!v) throw new Error('useAnnouncements must be used within AnnouncementsProvider')
  return v
}

/** 版本面板「更新内容」概览的一条：kind（来自所属 `### 分组`）+ summary（要点首句）。 */
export interface ReleaseEntry {
  kind: string
  summary: string
}

// release post 正文的 `### 分组标题`（中/英）→ kind，对齐 Keep a Changelog 类目。
const RELEASE_KIND_BY_HEADING: Record<string, string> = {
  新增: 'added', 变更: 'changed', 改进: 'improved', 修复: 'fixed',
  弃用: 'deprecated', 删除: 'removed', 安全: 'security',
  Added: 'added', Changed: 'changed', Improved: 'improved', Fixed: 'fixed',
  Deprecated: 'deprecated', Removed: 'removed', Security: 'security',
}

/** 从 release post 正文抽「更新内容」概览条目，给版本面板用。
 *  依赖 CONTENT-GUIDE 约定：`### 分组` 决定 kind，组内每个顶层要点 `- **首句**`
 *  是一条 summary（剥尾部 PR 号、跳过缩进子要点）。返回全部条目，截断由调用方做。 */
export function extractReleaseEntries(body: string): ReleaseEntry[] {
  const out: ReleaseEntry[] = []
  let kind = 'changed'
  for (const line of body.split('\n')) {
    const h = line.match(/^###\s+(.+?)\s*$/)
    if (h) { kind = RELEASE_KIND_BY_HEADING[h[1].trim()] ?? 'changed'; continue }
    const m = line.match(/^- \*\*(.+?)\*\*\s*$/)
    if (!m) continue
    out.push({ kind, summary: m[1].replace(/[（(]#[#\d,，\s]+[）)]\s*$/, '').trim() })
  }
  return out
}
