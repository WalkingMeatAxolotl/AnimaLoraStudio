/** 版本面板文案 mapping（ADR 0005）。
 *
 * 把"check.state + version/check 数据 → 用户可读文案"的逻辑抽出来，
 * 与 React 组件解耦便于单测。所有文案**只用版本号 / 状态语言**，
 * 不出现 commits / sha / branch 等 git 词汇。
 */
import type { SystemUpdateCheck } from '../api/client'

/** master（稳定版）通道的状态行文案。
 *
 * 优先级：
 * - up_to_date：已是最新 vX.Y.Z（latest_version 有就拼版本号，没就裸"已是最新"）
 * - update_available：有新稳定版 vX.Y.Z
 * - ahead：本地领先稳定版（罕见，回滚 / 抢跑）
 * - detached：当前 commit 不在稳定版历史上（feature branch）
 * - check=null：未检查
 */
export function formatMasterStateText(check: SystemUpdateCheck | null): string {
  if (!check) return '未检查'
  if (check.state === 'up_to_date') {
    return check.latest_version ? `已是最新 ${check.latest_version}` : '已是最新'
  }
  if (check.state === 'update_available') {
    const target = check.latest_version ?? check.latest_tag ?? check.latest_commit?.slice(0, 8) ?? ''
    return target ? `有新稳定版 ${target}` : '有新稳定版'
  }
  if (check.state === 'ahead') return '本地领先稳定版'
  return '当前 commit 不在稳定版历史上'
}

/** dev（开发版）通道的状态行文案。
 *
 * dev 没有版本号语义（滚动），所以"N 项新更新"是合理表达；不暴露
 * commits 字眼给用户。
 */
export function formatDevStateText(check: SystemUpdateCheck | null): string {
  if (!check) return '未抓取'
  if (check.state === 'up_to_date') return '与 dev HEAD 一致'
  if (check.state === 'update_available') {
    return check.behind_count > 0 ? `有 ${check.behind_count} 项新更新` : '有新更新'
  }
  if (check.state === 'ahead') return '本地领先 dev HEAD'
  return '当前 commit 不在 dev 历史上'
}

/** master 通道"更新"按钮是否应该显示（state=update_available 且有目标版本）。 */
export function shouldShowMasterUpdateButton(check: SystemUpdateCheck | null): boolean {
  if (!check || check.state !== 'update_available') return false
  return !!(check.latest_version ?? check.latest_tag)
}

/** dev 通道"切到 dev HEAD"按钮是否应该 disabled。
 *
 * 当 state=up_to_date 时按钮 disabled —— 即便用户的 installed_kind 是
 * stable，只要 commit 已等于 origin/dev HEAD，切操作就是 no-op，
 * 该 disabled 避免点了无反应（release 直后场景）。
 */
export function isDevSwitchButtonDisabled(check: SystemUpdateCheck | null): boolean {
  return check?.state === 'up_to_date'
}
