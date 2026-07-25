// 版本级评估页 —— 评估结果的规范位置。
//
// 评估的对象是 output/ 里的一组 checkpoint，不是某一次训练进程：checkpoint 比 task
// 活得久、resume 续训会横跨两个 task、手动丢进 output/ 的 LoRA 根本没有对应 task。
// 所以主入口在这里，不必先猜是哪次训练产生的结果。
//
// 训练页的指标面板**不动**（训练完就地看结果是高频路径），它只是同一个组件多传一个
// taskId 做过滤；这里不传，看的是整个 version 的历次评估。
import { useCallback, useMemo } from 'react'
import { useOutletContext, useSearchParams } from 'react-router-dom'
import type { ProjectDetail, Version } from '../../../api/client'
import { EvalMetricsPanel } from '../../../components/EvalMetricsPanel'
import StepShell from '../../../components/StepShell'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
}

export default function EvaluationPage() {
  const { project, activeVersion } = useOutletContext<Ctx>()
  // `?session=` 让队列作业详情 / 训练页能深链到具体某一次评估
  const [params, setParams] = useSearchParams()
  const sessionParam = params.get('session')
  const initialSessionId = useMemo(() => {
    const n = Number(sessionParam)
    return sessionParam && Number.isFinite(n) ? n : null
  }, [sessionParam])

  const syncSessionToUrl = useCallback((sid: number | null) => {
    setParams((prev) => {
      const next = new URLSearchParams(prev)
      if (sid == null) next.delete('session')
      else next.set('session', String(sid))
      return next
    }, { replace: true })
  }, [setParams])

  if (!activeVersion) {
    return (
      <StepShell idx="" title="评估" subtitle="用验证集给 checkpoint 打分并肉眼对比">
        <div className="card px-4 py-3 text-sm text-fg-tertiary">
          先选一个版本。评估的对象是该版本 output/ 下的 checkpoint。
        </div>
      </StepShell>
    )
  }

  return (
    <StepShell
      idx=""
      title="评估"
      subtitle={`${project.slug} · ${activeVersion.label} —— 用验证集给 checkpoint 打分并肉眼对比`}
    >
      <div className="flex-1 min-h-0 overflow-auto">
        <EvalMetricsPanel
          projectId={project.id}
          versionId={activeVersion.id}
          subtitle={`${project.slug} · ${activeVersion.label} · 该版本的全部评估`}
          // 版本页不挂 monitor SSE；面板自己按 Session 是否在跑决定要不要轮询
          connected={false}
          initialSessionId={initialSessionId}
          onSessionChange={syncSessionToUrl}
        />
      </div>
    </StepShell>
  )
}
