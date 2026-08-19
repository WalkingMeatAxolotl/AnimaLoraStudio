import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

// 测试环境里默认 locale = zh（i18n/index.ts 读 localStorage 取不到就 fallback 'zh'）。
// 不 import 这个,useTranslation 返回 raw key,所有断言中文字面量全打挂。
import '../i18n'

// jsdom 装的 fetch 会真去打网络。tagDict store / 其他 mount-time 请求在测试态下
// 打 404 是预期分支，但 `network error` 会让 React 报 act() warning。给 fetch 装
// 默认 stub：所有未显式 mock 的请求都返 404。具体测试可在自己的 beforeEach 里
// vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(...) 覆盖。
if (typeof globalThis.fetch === 'function') {
  vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 404 })))
}

// jsdom 没有 ResizeObserver（useAutoGrowTextarea 等会 new 它撑高 textarea）；装个
// no-op，避免组件 mount 时抛错。测试不校验自动撑高，回调不触发即可。
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver
}

// 把 tagDict store 预热到 'empty' 状态：useTagDict 的 useEffect 看到非 idle 就
// 跳过 loadDict，避免组件 mount 时触发异步 fetch + act() warning。需要看 dict
// ready 状态的具体测试可以 __setStateForTest 覆盖。
import { __setStateForTest } from '../tagDict/store'
__setStateForTest({ status: 'empty' })

// 同理预热 tagDict 偏好 store（chip 翻译 / 输入补全开关，后端 secrets 镜像）为
// loaded：useShowTagTranslation / useTagAutocompleteEnabled 的 mount effect 看到
// loaded 就不打 /api/secrets。值保持推导默认（测试态 lang=zh → 翻译开、补全开）；
// 要改开关的测试用 __setTagPrefsForTest，测 seed/加载流程的用 __resetTagPrefsForTest。
import { __setTagPrefsForTest } from '../tagDict/prefs'
__setTagPrefsForTest({ loaded: true })
