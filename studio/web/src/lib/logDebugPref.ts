/** 全局「默认显示调试日志」（docs/design/logging-target-state.md D1）。
 *
 * 存后端 `secrets.system.log_debug_default`（换浏览器不丢）。它只管**默认值**：
 * run.log / daemon ring 恒记 DEBUG，每个 LogView 有自己的「调试」开关（不持久化），
 * 挂载时取这里的值作初值；之后全局改了不回推已打开的视图（open question 3：不同步）。
 *
 * 模块级缓存 + 订阅：设置页改了，其它已挂载的消费者（之后新挂的视图）立刻拿到新默认。
 */
import { useEffect, useSyncExternalStore } from 'react'

import { api } from '../api/client'

let _value: boolean | null = null       // null = 尚未从后端拿到
let _loading: Promise<void> | null = null
const _listeners = new Set<() => void>()

function _emit(): void {
  for (const l of _listeners) l()
}

function _subscribe(l: () => void): () => void {
  _listeners.add(l)
  return () => { _listeners.delete(l) }
}

function _get(): boolean | null {
  return _value
}

/** 首次需要时拉一次后端；并发调用共享同一个 promise。 */
export function ensureLogDebugDefaultLoaded(): Promise<void> {
  if (_value !== null) return Promise.resolve()
  if (!_loading) {
    _loading = api.getSecrets()
      .then((sec) => { _value = !!sec.system?.log_debug_default })
      .catch(() => { _value = false })  // 拉不到按关处理，不阻塞日志视图
      .finally(() => { _loading = null; _emit() })
  }
  return _loading
}

/** 设置页保存后调用：同步缓存，不再额外请求。 */
export function setLogDebugDefaultCache(v: boolean): void {
  _value = v
  _emit()
}

/** 返回全局默认；未加载完成时为 null（调用方当 false 用，加载完自动更新）。 */
export function useLogDebugDefault(): boolean | null {
  const v = useSyncExternalStore(_subscribe, _get, _get)
  useEffect(() => { void ensureLogDebugDefaultLoaded() }, [])
  return v
}

/** 测试钩子。 */
export function _resetLogDebugPrefForTests(): void {
  _value = null
  _loading = null
  _listeners.clear()
}
