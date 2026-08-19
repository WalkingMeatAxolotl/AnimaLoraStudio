/** Tag 词典全站 UI 偏好（chip 翻译 / 输入补全）—— 后端 `secrets.tag_dictionary` 的前端镜像。
 *
 * 模块级 singleton + useSyncExternalStore（跟 store.ts 同构）：Settings 切开关后
 * TranslatedTag / useTagSuggest 等全站消费点即时生效，不用传 prop 链。
 *
 * 生命周期：
 *   1. 首个消费者 mount → loadTagPrefs()：GET /api/secrets 读 tag_dictionary。
 *      加载前的快照 = 本地推导值（旧 localStorage 值 → 否则默认），首屏不闪。
 *   2. 后端字段为 null（从未设过）→ 一次性 seed 并 PUT 写回：旧 localStorage 值
 *      优先；show_translation 无旧值时按界面语言推导（zh 开、其它关，用户拍板：
 *      seed 后切语言不再覆盖）；autocomplete 无旧值不写（生效默认开）。
 *      后端已是 bool（用户设过 / 别的浏览器 seed 过）→ 以后端为准，不覆盖。
 *      两种情况都清掉旧 localStorage 键（seed 失败时保留，下次加载重试）。
 *   3. set*：乐观更新 + PUT；失败回滚并 rethrow（Settings 侧 toast）。
 *
 * 界面语言本身（i18n lang）仍在 localStorage，不在本模块管。
 */
import { useEffect, useSyncExternalStore } from 'react'

import { api, type TagDictionarySecretsConfig } from '../api/client'
import { getStoredLang } from '../i18n'

/** 迁移前的 localStorage 键（0.25 之前的实现）。只读 + 清理，不再写。 */
export const LEGACY_SHOW_KEY = 'studio.tag.showTranslation'
export const LEGACY_AUTOCOMPLETE_KEY = 'studio.tag.autocomplete'

export interface TagPrefsState {
  /** 后端值已拉到（含 seed 完成）；false 时 show/autocomplete 是本地推导值。 */
  loaded: boolean
  showTranslation: boolean
  autocomplete: boolean
}

function readLegacy(key: string): boolean | null {
  try {
    const raw = localStorage.getItem(key)
    if (raw === '1') return true
    if (raw === '0') return false
    return null
  } catch { return null }
}

function clearLegacy(): void {
  try {
    localStorage.removeItem(LEGACY_SHOW_KEY)
    localStorage.removeItem(LEGACY_AUTOCOMPLETE_KEY)
  } catch { /* ignore */ }
}

/** show_translation 的语言默认：界面语言以 zh 起头 → 开，否则关。 */
export function defaultShowFromLang(): boolean {
  const lang = (getStoredLang() ?? 'zh').toLowerCase()
  return lang.startsWith('zh')
}

function provisionalState(): TagPrefsState {
  return {
    loaded: false,
    showTranslation: readLegacy(LEGACY_SHOW_KEY) ?? defaultShowFromLang(),
    autocomplete: readLegacy(LEGACY_AUTOCOMPLETE_KEY) ?? true,
  }
}

let state: TagPrefsState = provisionalState()
const listeners = new Set<() => void>()
let inFlight: Promise<void> | null = null

function setState(next: Partial<TagPrefsState>): void {
  state = { ...state, ...next }
  listeners.forEach((l) => l())
}

function subscribe(l: () => void): () => void {
  listeners.add(l)
  return () => { listeners.delete(l) }
}

function getSnapshot(): TagPrefsState {
  return state
}

async function doLoad(): Promise<void> {
  let remote: TagDictionarySecretsConfig | undefined
  try {
    remote = (await api.getSecrets()).tag_dictionary
  } catch {
    // 拉不到：保持本地推导值、不标 loaded，下一个消费者 mount 时再试。
    return
  }
  const legacyShow = readLegacy(LEGACY_SHOW_KEY)
  const legacyAc = readLegacy(LEGACY_AUTOCOMPLETE_KEY)
  const patch: Partial<TagDictionarySecretsConfig> = {}

  let show: boolean
  if (remote?.show_translation == null) {
    // 从未设过：旧值优先，否则按语言推导；两者都要写回让它「钉住」。
    show = legacyShow ?? defaultShowFromLang()
    patch.show_translation = show
  } else {
    show = remote.show_translation
  }

  let ac: boolean
  if (remote?.autocomplete == null) {
    ac = legacyAc ?? true
    // 默认开不必写回；只有旧浏览器显式设过才 seed。
    if (legacyAc !== null) patch.autocomplete = ac
  } else {
    ac = remote.autocomplete
  }

  let seeded = true
  if (Object.keys(patch).length > 0) {
    try {
      await api.updateSecrets({ tag_dictionary: patch })
    } catch {
      // seed 失败：本地先用推导值；旧键留着下次重试，不丢用户旧设置。
      seeded = false
    }
  }
  if (seeded) clearLegacy()
  setState({ loaded: true, showTranslation: show, autocomplete: ac })
}

/** 首次加载：idempotent；已加载直接返回，加载中复用 in-flight Promise。 */
export function loadTagPrefs(): Promise<void> {
  if (state.loaded) return Promise.resolve()
  if (inFlight) return inFlight
  inFlight = doLoad().finally(() => { inFlight = null })
  return inFlight
}

async function commit(
  local: Partial<Pick<TagPrefsState, 'showTranslation' | 'autocomplete'>>,
  remote: Partial<TagDictionarySecretsConfig>,
): Promise<void> {
  const prev = state
  setState(local)
  try {
    await api.updateSecrets({ tag_dictionary: remote })
  } catch (e) {
    setState({ showTranslation: prev.showTranslation, autocomplete: prev.autocomplete })
    throw e
  }
}

/** 写入 chip 翻译开关：乐观更新 + PUT，失败回滚并抛错。 */
export function setShowTagTranslation(next: boolean): Promise<void> {
  return commit({ showTranslation: next }, { show_translation: next })
}

/** 写入输入补全开关：乐观更新 + PUT，失败回滚并抛错。 */
export function setTagAutocompleteEnabled(next: boolean): Promise<void> {
  return commit({ autocomplete: next }, { autocomplete: next })
}

function useTagPrefs(): TagPrefsState {
  const snap = useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
  useEffect(() => { void loadTagPrefs() }, [])
  return snap
}

/** 给 React 组件订阅：返回 [show, setShow]。 */
export function useShowTagTranslation(): [boolean, (next: boolean) => Promise<void>] {
  return [useTagPrefs().showTranslation, setShowTagTranslation]
}

/** 给 React 组件订阅：返回 [enabled, setEnabled]。 */
export function useTagAutocompleteEnabled(): [boolean, (next: boolean) => Promise<void>] {
  return [useTagPrefs().autocomplete, setTagAutocompleteEnabled]
}

/** 测试用：直接注入 state，绕过网络（setup.ts 预热为 loaded，组件测试不触发请求）。 */
export function __setTagPrefsForTest(next: Partial<TagPrefsState>): void {
  setState(next)
}

/** 测试用：回到「未加载」的初始态（重新读 localStorage 推导本地值）。 */
export function __resetTagPrefsForTest(): void {
  inFlight = null
  setState(provisionalState())
}
