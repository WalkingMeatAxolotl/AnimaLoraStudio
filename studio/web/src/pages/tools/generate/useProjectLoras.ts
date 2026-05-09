import { useEffect, useState } from 'react'
import { api } from '../../../api/client'
import type { ProjectLora } from './types'

/** 启动一次拉取所有项目下的 LoRA 版本（含训练中）。
 *
 * 之前的实现过滤 `if (!v.output_lora_path) continue` —— 但 output_lora_path
 * 只在训练完成（status=done）时回填，训练中的 version 该字段是 null。
 * 实际磁盘上 anima_train 已经在 versions/{label}/output/ 写了 step ckpt
 * （_step1500.safetensors 等），picker 该把它们也列出来。
 *
 * 当前策略：
 *   1. v.output_lora_path 优先（已完成训练，path = _final.safetensors）
 *   2. fallback：fetch listVersionLoraCkpts 取 latest（最新 step / epoch ckpt）
 *   3. 都没 → version output/ 没任何 ckpt，跳过
 *
 * 失败不抛 —— 用户走「外部文件…」PathPicker 兜底。N+1 调用：用户场景下
 * project < 20 可接受；启动加载一次，picker 不实时刷新（用户预期）。
 */
export function useProjectLoras(): ProjectLora[] {
  const [items, setItems] = useState<ProjectLora[]>([])
  useEffect(() => {
    void (async () => {
      try {
        const projects = await api.listProjects()
        const details = await Promise.all(
          projects.map((p) => api.getProject(p.id).catch(() => null))
        )

        // 第一阶段：从 v.output_lora_path 直接构造（已完成的）+ 收集需要 fallback 的
        const out: ProjectLora[] = []
        const fallbacks: Array<{ pid: number; vid: number; meta: ProjectLora }> = []

        for (const d of details) {
          if (!d) continue
          for (const v of d.versions) {
            if (v.output_lora_path) {
              out.push({
                projectId: d.id,
                projectTitle: d.title,
                versionId: v.id,
                versionLabel: v.label,
                stage: v.stage,
                path: v.output_lora_path,
                createdAt: v.created_at,
              })
            } else {
              // 训练中或没 final → 尝试用最新 step/epoch ckpt
              fallbacks.push({
                pid: d.id, vid: v.id,
                meta: {
                  projectId: d.id,
                  projectTitle: d.title,
                  versionId: v.id,
                  versionLabel: v.label,
                  stage: v.stage,
                  path: '',  // 由 fallback 填
                  createdAt: v.created_at,
                },
              })
            }
          }
        }

        // 第二阶段：并发拉所有需要 fallback 的 ckpt 列表，取 latest
        const fbResults = await Promise.all(
          fallbacks.map(async (fb) => {
            try {
              const ckpts = await api.listVersionLoraCkpts(fb.pid, fb.vid)
              if (ckpts.length === 0) return null
              return { ...fb.meta, path: ckpts[0].path }
            } catch {
              return null
            }
          })
        )
        for (const r of fbResults) {
          if (r) out.push(r)
        }

        out.sort((a, b) => b.createdAt - a.createdAt)
        setItems(out)
      } catch {
        /* 启动失败不阻塞 */
      }
    })()
  }, [])
  return items
}
