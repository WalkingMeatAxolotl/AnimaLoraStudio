import { describe, expect, it } from 'vitest'
import type { SystemUpdateCheck } from '../api/client'
import {
  formatDevStateText,
  formatMasterStateText,
  isDevSwitchButtonDisabled,
  shouldShowMasterUpdateButton,
} from './versionPanel'

/** 构造一个 check 对象（必填字段填默认值，便于按需 override）。 */
function mkCheck(over: Partial<SystemUpdateCheck>): SystemUpdateCheck {
  return {
    channel: 'master',
    current_commit: '',
    latest_commit: '',
    commits_ahead: 0,
    has_update: false,
    latest_tag: null,
    checked_at: 0,
    error: null,
    state: 'up_to_date',
    installed_version: null,
    latest_version: null,
    behind_count: 0,
    ...over,
  }
}

describe('formatMasterStateText (ADR 0005)', () => {
  it('null check → 未检查', () => {
    expect(formatMasterStateText(null)).toBe('未检查')
  })

  it('up_to_date + latest_version → 已是最新 vX.Y.Z', () => {
    expect(formatMasterStateText(mkCheck({ state: 'up_to_date', latest_version: 'v0.8.0' })))
      .toBe('已是最新 v0.8.0')
  })

  it('up_to_date 无 latest_version → 已是最新', () => {
    expect(formatMasterStateText(mkCheck({ state: 'up_to_date' }))).toBe('已是最新')
  })

  it('update_available 显示目标版本号', () => {
    expect(formatMasterStateText(mkCheck({
      state: 'update_available', latest_version: 'v0.9.0',
    }))).toBe('有新稳定版 v0.9.0')
  })

  it('update_available fallback 到 latest_tag', () => {
    expect(formatMasterStateText(mkCheck({
      state: 'update_available', latest_version: null, latest_tag: 'v0.9.0-rc1',
    }))).toBe('有新稳定版 v0.9.0-rc1')
  })

  it('ahead → 本地领先稳定版（不暴露 commit 词汇）', () => {
    expect(formatMasterStateText(mkCheck({ state: 'ahead' }))).toBe('本地领先稳定版')
  })

  it('detached → 当前 commit 不在稳定版历史上', () => {
    expect(formatMasterStateText(mkCheck({ state: 'detached' })))
      .toBe('当前 commit 不在稳定版历史上')
  })

  it('不出现 git 词汇：commits / sha / branch', () => {
    const samples = [
      formatMasterStateText(mkCheck({ state: 'up_to_date', latest_version: 'v0.8.0' })),
      formatMasterStateText(mkCheck({ state: 'update_available', latest_version: 'v0.9.0' })),
      formatMasterStateText(mkCheck({ state: 'ahead' })),
    ]
    for (const s of samples) {
      expect(s).not.toMatch(/commits?/i)
      expect(s).not.toMatch(/\bsha\b/i)
      expect(s).not.toMatch(/branch/i)
    }
  })
})

describe('formatDevStateText (ADR 0005)', () => {
  it('null check → 未抓取', () => {
    expect(formatDevStateText(null)).toBe('未抓取')
  })

  it('up_to_date → 与 dev HEAD 一致', () => {
    expect(formatDevStateText(mkCheck({ channel: 'dev', state: 'up_to_date' })))
      .toBe('与 dev HEAD 一致')
  })

  it('update_available with behind_count=3 → 有 3 项新更新', () => {
    expect(formatDevStateText(mkCheck({
      channel: 'dev', state: 'update_available', behind_count: 3,
    }))).toBe('有 3 项新更新')
  })

  it('update_available with behind_count=0 → 有新更新', () => {
    expect(formatDevStateText(mkCheck({
      channel: 'dev', state: 'update_available', behind_count: 0,
    }))).toBe('有新更新')
  })

  it('ahead → 本地领先 dev HEAD', () => {
    expect(formatDevStateText(mkCheck({ channel: 'dev', state: 'ahead' })))
      .toBe('本地领先 dev HEAD')
  })

  it('不出现 "commits" 字眼（dev 通道改"项更新"）', () => {
    const s = formatDevStateText(mkCheck({
      channel: 'dev', state: 'update_available', behind_count: 5,
    }))
    expect(s).not.toMatch(/commits?/i)
    expect(s).toContain('项新更新')
  })
})

describe('shouldShowMasterUpdateButton', () => {
  it('null check → false', () => {
    expect(shouldShowMasterUpdateButton(null)).toBe(false)
  })

  it('up_to_date → false（已是最新，不显示更新按钮）', () => {
    expect(shouldShowMasterUpdateButton(mkCheck({
      state: 'up_to_date', latest_version: 'v0.8.0',
    }))).toBe(false)
  })

  it('update_available + latest_version → true', () => {
    expect(shouldShowMasterUpdateButton(mkCheck({
      state: 'update_available', latest_version: 'v0.9.0',
    }))).toBe(true)
  })

  it('update_available 但无目标版本号 → false（防止显示空目标按钮）', () => {
    expect(shouldShowMasterUpdateButton(mkCheck({
      state: 'update_available', latest_version: null, latest_tag: null,
    }))).toBe(false)
  })

  it('ahead / detached → false（不暗示用户能"更新到自己"）', () => {
    expect(shouldShowMasterUpdateButton(mkCheck({ state: 'ahead' }))).toBe(false)
    expect(shouldShowMasterUpdateButton(mkCheck({ state: 'detached' }))).toBe(false)
  })
})

describe('isDevSwitchButtonDisabled (release 直后场景 regression)', () => {
  it('up_to_date → disabled（当前 commit 已等于 dev HEAD，切是 no-op）', () => {
    expect(isDevSwitchButtonDisabled(mkCheck({
      channel: 'dev', state: 'up_to_date',
    }))).toBe(true)
  })

  it('update_available → 可点', () => {
    expect(isDevSwitchButtonDisabled(mkCheck({
      channel: 'dev', state: 'update_available', behind_count: 2,
    }))).toBe(false)
  })

  it('null check → 可点（用户没抓取过，按钮要能触发抓取 + 切换）', () => {
    expect(isDevSwitchButtonDisabled(null)).toBe(false)
  })
})
