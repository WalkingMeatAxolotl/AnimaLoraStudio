import { useEffect, useState } from 'react'
import { api, type DaemonStatus } from '../../../api/client'
import { useEventStream } from '../../../lib/useEventStream'
import { useToast } from '../../../components/Toast'

/** Header 行尾的「清理显存」按钮：单一按钮，状态隐式（busy / 未加载时 disabled）。
 *
 * 之前 sidebar 末尾的"推理 daemon · 状态"卡片合并到这里 —— 用户决策：
 * 不要时刻显示状态文字，需要释放 VRAM 时按按钮就行。
 */
export default function DaemonControls() {
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
        toast('显存已释放（模型未加载）', 'info')
      } else {
        toast('已请求清理显存', 'success')
      }
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setUnloading(false)
    }
  }

  const canUnload = !!(status && status.model_loaded && !status.busy && status.state !== 'unloading')

  return (
    <button
      className="btn btn-ghost text-sm"
      onClick={handleUnload}
      disabled={!canUnload || unloading}
      title={
        !status ? '加载状态中…'
          : status.busy ? '生成中不可清理'
            : !status.model_loaded ? '模型未加载，无需清理'
              : '释放推理 daemon 占用的 VRAM；下次出图按需重 load'
      }
    >
      {unloading ? '清理中…' : '清理显存'}
    </button>
  )
}
