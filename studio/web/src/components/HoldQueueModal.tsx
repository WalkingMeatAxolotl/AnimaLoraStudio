// HoldQueueModal —— ADR 0006 PR-4 §4.4 挂起队列 confirmation modal。
//
// 情形 A：没有 running task → 简单二选一确认
// 情形 B：有 running task → 多 radio 让用户选 "让它跑完" 还是 "同时暂停"
//   主按钮文案随 radio 联动，避免点完不知道自己确认了什么。
//
// 决策 → caller 通过 onConfirm 收到，自己去调 hold + 可选 pause API。
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { Task } from '../api/client'

export type HoldDecision =
  | { kind: 'hold-only' }
  | { kind: 'hold-and-pause'; taskId: number }

export interface HoldQueueModalProps {
  /** 当前 running task；为 null = 情形 A。 */
  runningTask: Task | null
  onCancel: () => void
  onConfirm: (decision: HoldDecision) => void
}

export function HoldQueueModal({ runningTask, onCancel, onConfirm }: HoldQueueModalProps) {
  const { t } = useTranslation()
  // 情形 B 才用：默认 "让它跑完"
  const [pauseToo, setPauseToo] = useState(false)
  const hasRunning = runningTask !== null
  const runningName = runningTask?.name ?? ''
  const runningId = runningTask?.id ?? 0

  const handleConfirm = () => {
    if (hasRunning && pauseToo) {
      onConfirm({ kind: 'hold-and-pause', taskId: runningId })
    } else {
      onConfirm({ kind: 'hold-only' })
    }
  }

  const confirmText = !hasRunning
    ? t('queue.holdModal.confirmSimple')
    : pauseToo
      ? t('queue.holdModal.confirmPauseToo', { id: runningId })
      : t('queue.holdModal.confirmLetRun', { id: runningId })

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="hold-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-md"
      data-testid="hold-queue-modal"
    >
      <div className="w-[90%] max-w-[520px] flex flex-col gap-5 p-7 bg-elevated border border-dim rounded-lg shadow-xl">
        <h2
          id="hold-modal-title"
          className="m-0 text-lg font-semibold text-fg-primary"
        >
          {t('queue.holdModal.title')}
        </h2>

        <p className="m-0 text-sm text-fg-secondary leading-relaxed">
          {hasRunning
            ? t('queue.holdModal.descWithRunning')
            : t('queue.holdModal.descNoRunning')}
        </p>

        {hasRunning && (
          <div className="flex flex-col gap-3">
            <p className="m-0 text-sm text-fg-primary">
              {t('queue.holdModal.currentlyRunning', { id: runningId, name: runningName })}
            </p>
            <div className="flex flex-col gap-2 pl-1">
              <label className="flex items-start gap-2 cursor-pointer text-sm text-fg-secondary">
                <input
                  type="radio"
                  name="hold-option"
                  checked={!pauseToo}
                  onChange={() => setPauseToo(false)}
                  className="mt-1"
                  data-testid="hold-opt-let-run"
                />
                <span>{t('queue.holdModal.optionLetRun', { id: runningId })}</span>
              </label>
              <label className="flex items-start gap-2 cursor-pointer text-sm text-fg-secondary">
                <input
                  type="radio"
                  name="hold-option"
                  checked={pauseToo}
                  onChange={() => setPauseToo(true)}
                  className="mt-1"
                  data-testid="hold-opt-pause-too"
                />
                <span>{t('queue.holdModal.optionPauseToo', { id: runningId })}</span>
              </label>
            </div>
          </div>
        )}

        <div className="flex justify-end gap-2 pt-1">
          <button
            type="button"
            onClick={onCancel}
            className="px-3 py-1.5 text-sm rounded border border-dim bg-surface hover:bg-surface-hover text-fg-primary"
            data-testid="hold-cancel-btn"
          >
            {t('common.cancel', { defaultValue: 'Cancel' })}
          </button>
          <button
            type="button"
            onClick={handleConfirm}
            className="px-3 py-1.5 text-sm rounded bg-accent text-accent-on hover:bg-accent-hover"
            data-testid="hold-confirm-btn"
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  )
}
