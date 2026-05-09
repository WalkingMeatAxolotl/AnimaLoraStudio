import { useEffect, useState } from 'react'
import { api } from '../../../api/client'
import type { RecentLora } from './types'

/** 启动一次拉取所有项目的最新 LoRA。
 *
 * listProjects → 并行 getProject → 收集 output_lora_path。
 * 用户场景下 project 数 < 20，N+1 调用可接受；不在乎实时性，启动加载一次。
 * 失败不抛 — 用户仍可手敲 / PathPicker 兜底。
 */
export function useRecentLoras(): RecentLora[] {
  const [recentLoras, setRecentLoras] = useState<RecentLora[]>([])
  useEffect(() => {
    void (async () => {
      try {
        const projects = await api.listProjects()
        const details = await Promise.all(
          projects.map((p) => api.getProject(p.id).catch(() => null))
        )
        const items: RecentLora[] = []
        for (const d of details) {
          if (!d) continue
          for (const v of d.versions) {
            if (v.output_lora_path) {
              items.push({
                label: `${d.title} / ${v.label}`,
                path: v.output_lora_path,
                createdAt: v.created_at,
              })
            }
          }
        }
        items.sort((a, b) => b.createdAt - a.createdAt)
        setRecentLoras(items)
      } catch {
        /* 启动失败不阻塞 — 用户仍可手敲 / PathPicker */
      }
    })()
  }, [])
  return recentLoras
}
