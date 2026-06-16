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
