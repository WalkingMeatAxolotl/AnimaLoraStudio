/**
 * useAdvancedMode — Schema 表单的"简单/高级"开关状态 hook（P1-4）。
 *
 * 行为：
 *   - 持久化到 localStorage（key: `studio:advanced_mode`，加 studio: 命名空间防撞）
 *   - mount 时从 localStorage 读初值
 *   - 监听 window 'storage' 事件：当其他 tab 改这个 key 时本 tab 同步更新
 *
 * Train 页和 Presets 页都用这个 hook，保证两个入口 + 多 tab 状态一致。
 */
import { useCallback, useEffect, useState } from 'react'

const STORAGE_KEY = 'studio:advanced_mode'

function readPersisted(): boolean {
  return localStorage.getItem(STORAGE_KEY) === 'true'
}

export function useAdvancedMode(): [boolean, () => void] {
  const [advancedMode, setAdvancedMode] = useState<boolean>(readPersisted)

  // 跨 tab 同步：监听 storage 事件，当其他 tab 改 advanced_mode 时本 tab 跟进
  useEffect(() => {
    const handler = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) {
        setAdvancedMode(e.newValue === 'true')
      }
    }
    window.addEventListener('storage', handler)
    return () => window.removeEventListener('storage', handler)
  }, [])

  const toggle = useCallback(() => {
    setAdvancedMode(v => {
      const next = !v
      localStorage.setItem(STORAGE_KEY, String(next))
      return next
    })
  }, [])

  return [advancedMode, toggle]
}
