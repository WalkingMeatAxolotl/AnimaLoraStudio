// SettingsDrawer.tsx —— Settings 覆盖式抽屉。
//
// Settings 是本地静态配置页，不制造 transient loading shell。首次打开时内容与 Drawer
// 在同一次 commit 中挂载并作为一个整体入场；之后保持挂载，避免再次打开时重建组件树。
import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import Drawer from './Drawer'
import SettingsPage from '../pages/tools/Settings'
import { useSettingsDrawer } from '../lib/SettingsDrawer'

export default function SettingsDrawer() {
  const { isOpen, close, setReady } = useSettingsDrawer()
  const { t } = useTranslation()
  const [contentMounted, setContentMounted] = useState(false)

  useEffect(() => {
    if (isOpen) {
      setContentMounted(true)
      return
    }
    setReady(false)
  }, [isOpen, setReady])

  const handleEntered = useCallback(() => {
    setReady(true)
  }, [setReady])

  // isOpen 直接参与条件，确保首次打开时内容和 opening 壳层在同一次 commit 中出现；
  // state 仅负责在第一次打开后保活内容。
  const shouldMountContent = isOpen || contentMounted

  return (
    <Drawer
      open={isOpen}
      title={t('nav.settings')}
      showTitle={false}
      size="page"
      onClose={() => { void close() }}
      onEntered={handleEntered}
      testId="settings-drawer"
    >
      {shouldMountContent && (
        <div className="flex h-full min-h-0 flex-col">
          <SettingsPage />
        </div>
      )}
    </Drawer>
  )
}
