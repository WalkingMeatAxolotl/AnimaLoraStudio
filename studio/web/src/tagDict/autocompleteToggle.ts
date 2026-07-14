/** 全局 toggle：tag 输入框是否启用 autocomplete 候选弹出。默认开。
 *
 * 实现走 localStorage + 模块级 subscribers，跟 showToggle.ts 同风格（裸 KV + try-catch）。
 * 所有 autocomplete 入口（useTagSuggest）订阅此开关，Settings 里关掉即全站生效。
 */
import { useSyncExternalStore } from 'react'

const STORAGE_KEY = 'studio.tag.autocomplete'

const listeners = new Set<() => void>()

function compute(): boolean {
  // 没设过 = 开（默认打开，只有显式 '0' 才关）
  try { return localStorage.getItem(STORAGE_KEY) !== '0' } catch { return true }
}

function subscribe(l: () => void): () => void {
  listeners.add(l)
  return () => { listeners.delete(l) }
}

/** 给 React 组件订阅：返回 [enabled, setEnabled]。 */
export function useTagAutocompleteEnabled(): [boolean, (next: boolean) => void] {
  const value = useSyncExternalStore(subscribe, compute, compute)
  const setter = (next: boolean) => {
    try { localStorage.setItem(STORAGE_KEY, next ? '1' : '0') } catch { /* ignore */ }
    listeners.forEach((l) => l())
  }
  return [value, setter]
}
