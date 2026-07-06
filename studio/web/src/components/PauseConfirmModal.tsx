// PauseConfirmModal —— ADR 0006 Addendum 1 §UI 决策第 5 条 暂停 confirm modal。
//
// 设计动机：方案 Δ 把暂停语义降级为「立即释放 GPU + 丢弃当前轮进度」，跟用户
// 直觉的"保存进度后退出"有差距。每次按下暂停按钮前用 confirm modal 统一告知：
// (1) 部分实验性参数（InfoNoise / Prodigy 类 / cosine LR）可能受暂停影响
// (2) 恢复时会丢失当前轮进度，从上一轮 epoch 结束位置继续
// 用户确认后才真正调 pause API → 进 PauseProgressModal 锁屏。
//
// 文案统一不带任何动态字段（无 epoch N / task name）—— ADR 决策第 5 条明确：
// "用户不需要知道实际逻辑是什么样的"。
import { useTranslation } from 'react-i18next'

export interface PauseConfirmModalProps {
  onCancel: () => void
  onConfirm: () => void
}

export function PauseConfirmModal({ onCancel, onConfirm }: PauseConfirmModalProps) {
  const { t } = useTranslation()

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="pause-confirm-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-md"
      data-testid="pause-confirm-modal"
    >
      <div className="w-[90%] max-w-[480px] flex flex-col gap-5 p-7 bg-elevated border border-dim rounded-lg shadow-xl">
        <h2
          id="pause-confirm-title"
          className="m-0 text-lg font-semibold text-fg-primary"
        >
          {t('queue.pauseConfirm.title')}
        </h2>

        <p className="m-0 text-sm text-fg-secondary leading-relaxed">
          {t('queue.pauseConfirm.adaptiveWarning')}
        </p>

        <p className="m-0 text-sm text-fg-secondary leading-relaxed">
          {t('queue.pauseConfirm.epochLoss')}
        </p>

        <div className="flex justify-end gap-3 mt-2">
          {/* 统一 modal footer 范式：全尺寸 btn-secondary / btn-primary（同 Dialog /
              新建项目 modal；原 ghost+sm 小一号）。 */}
          <button
            type="button"
            onClick={onCancel}
            className="btn btn-secondary"
            data-testid="pause-confirm-cancel"
          >
            {t('queue.pauseConfirm.cancel')}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className="btn btn-primary"
            data-testid="pause-confirm-ok"
          >
            {t('queue.pauseConfirm.confirm')}
          </button>
        </div>
      </div>
    </div>
  )
}
