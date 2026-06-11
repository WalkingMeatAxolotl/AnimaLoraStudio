/** 全局 toggle：chip 上是否显示中文翻译。
 *
 * 默认值规则（用户拍板）：
 *   - 首次读取：localStorage 没值时，依 i18n lang 写入：lang 以 `zh` 起头 → '1'（开），
 *     否则 '0'（关）。
 *   - 之后用户切 lang 不再覆盖；尊重用户手动设置。
 *
 * 实现走 localStorage + 模块级 subscribers，跟 i18n/index.ts 同风格（裸 KV + try-catch）。
 */
import { useSyncExternalStore } from 'react'

import { getStoredLang } from '../i18n'

const STORAGE_KEY = 'studio.tag.showTranslation'

const listeners = new Set<() => void>()

function readRaw(): string | null {
  try { return localStorage.getItem(STORAGE_KEY) } catch { return null }
}

function writeRaw(v: string): void {
  try { localStorage.setItem(STORAGE_KEY, v) } catch { /* ignore */ }
}

function defaultFromLang(): boolean {
  const lang = (getStoredLang() ?? 'zh').toLowerCase()
  return lang.startsWith('zh')
}

/** 计算"当前是否开"——若 localStorage 没设过就按 lang 默认 + 顺手回写一份。 */
function compute(): boolean {
  const raw = readRaw()
  if (raw === null) {
    const def = defaultFromLang()
    writeRaw(def ? '1' : '0')
    return def
  }
  return raw === '1'
}

function subscribe(l: () => void): () => void {
  listeners.add(l)
  return () => { listeners.delete(l) }
}

/** 给 React 组件订阅：返回 [show, setShow]。 */
export function useShowTagTranslation(): [boolean, (next: boolean) => void] {
  const value = useSyncExternalStore(subscribe, compute, compute)
  const setter = (next: boolean) => {
    writeRaw(next ? '1' : '0')
    listeners.forEach((l) => l())
  }
  return [value, setter]
}

/** 非 hook 形式读（chip 渲染路径深的地方用，比传 prop 链方便）。 */
export function isShowingTranslation(): boolean {
  return compute()
}
