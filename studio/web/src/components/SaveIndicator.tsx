import { useTranslation } from 'react-i18next'
import type { SaveStatus } from '../lib/SettingsData'

/**
 * 自动保存状态指示(instant-apply / debounce 自动保存页面共用):
 * saving → 「保存中…」;saved → 「✓ 已保存 hh:mm:ss」带 flash 动画;
 * error → 红色错误;idle 不渲染。原 Settings 页私有组件,刀 2 UX 收尾抽出,
 * Settings 顶栏 / Train 页 header / Presets 页 header 三处共用。
 */
export default function SaveIndicator({ status }: { status: SaveStatus }) {
  const { t } = useTranslation()
  if (status.state === 'saving') {
    return <span className="text-xs text-fg-tertiary">{t('settings.saveStatus.saving')}</span>
  }
  if (status.state === 'saved') {
    const time = new Date(status.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    // key={status.at}：每次保存重挂载 → 重播 flash 动画，连续保存也能一眼看出"又存了"。
    return (
      <span key={status.at} className="settings-saved-flash text-xs inline-flex items-center gap-1">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
          <path d="M20 6L9 17l-5-5" />
        </svg>
        {t('settings.saveStatus.saved', { time })}
      </span>
    )
  }
  if (status.state === 'error') {
    return <span className="text-xs text-err">{t('settings.saveStatus.error')}</span>
  }
  return null
}
