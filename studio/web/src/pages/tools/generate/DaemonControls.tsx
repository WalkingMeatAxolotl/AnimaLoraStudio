import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type DaemonStatus } from '../../../api/client'
import { useEventStream } from '../../../lib/useEventStream'
import { useToast } from '../../../components/Toast'

/** Header 行尾的「清理显存」+「日志」按钮：单一按钮，状态隐式（busy / 未加载时 disabled）。
 *
 * 之前 sidebar 末尾的"推理 daemon · 状态"卡片合并到这里 —— 用户决策：
 * 不要时刻显示状态文字，需要释放 VRAM 时按按钮就行。
 *
 * 日志按钮：onToggleLog 切换 daemon log 抽屉（位于 Generate 页底部 40vh）。
 */
export default function DaemonControls({ onToggleLog }: { onToggleLog?: () => void }) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [status, setStatus] = useState<DaemonStatus | null>(null)
  const [unloading, setUnloading] = useState(false)

  useEffect(() => {
    void api.getDaemonStatus()
      .then(setStatus)
      .catch(() => { /* 启动一闪 */ })
  }, [])

  useEventStream((evt) => {
    if (evt.type === 'daemon_state_changed') {
      setStatus({
        state: evt.state,
        model_loaded: !!evt.model_loaded,
        busy: !!evt.busy,
        alive: evt.state !== 'stopped',
      } as DaemonStatus)
    }
  })

  const handleUnload = async () => {
    setUnloading(true)
    try {
      const r = await api.unloadDaemon()
      if (r.noop) {
        toast(t('generate.vramAlreadyFree'), 'info')
      } else {
        toast(t('generate.vramUnloadRequested'), 'success')
      }
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setUnloading(false)
    }
  }

  const canUnload = !!(status && status.model_loaded && !status.busy && status.state !== 'unloading')

  return (
    <div className="flex items-center gap-2">
      <button
        className="btn btn-ghost text-sm"
        onClick={handleUnload}
        disabled={!canUnload || unloading}
        title={
          !status ? t('generate.daemonLoading')
            : status.busy ? t('generate.daemonBusy')
              : !status.model_loaded ? t('generate.daemonNoModel')
                : t('generate.daemonUnloadTitle')
        }
      >
        {unloading ? t('generate.unloadingVram') : t('generate.unloadVram')}
      </button>
      {onToggleLog && (
        <button className="btn btn-ghost text-sm" onClick={onToggleLog}>
          {t('generate.logDrawerOpen')}
        </button>
      )}
    </div>
  )
}
