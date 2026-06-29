import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { AnnouncementCenter } from './components/AnnouncementCenter'
import { DialogProvider } from './components/Dialog'
import { ErrorBoundary } from './components/ErrorBoundary'
import { FirstRunLangModal } from './components/FirstRunLangModal'
import { FirstRunOnboardingModal } from './components/FirstRunOnboardingModal'
import { ToastProvider } from './components/Toast'
import { AnnouncementsProvider } from './lib/Announcements'
import { installGlobalErrorHandlers } from './lib/errors/setup'
import { SettingsDataProvider } from './lib/SettingsData'
import { SettingsDrawerProvider } from './lib/SettingsDrawer'
import { initTheme } from './lib/theme'
import './i18n'
import './index.css'

// ADR-0009 PR-3 C2: window.onerror + unhandledrejection 三路捕获 → /api/client-errors
installGlobalErrorHandlers()

initTheme()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <ToastProvider>
        <DialogProvider>
          <SettingsDataProvider>
            <SettingsDrawerProvider>
              <AnnouncementsProvider>
                <FirstRunLangModal />
                <FirstRunOnboardingModal />
                <AnnouncementCenter />
                <App />
              </AnnouncementsProvider>
            </SettingsDrawerProvider>
          </SettingsDataProvider>
        </DialogProvider>
      </ToastProvider>
    </ErrorBoundary>
  </React.StrictMode>,
)
