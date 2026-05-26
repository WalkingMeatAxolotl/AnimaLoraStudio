import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { DialogProvider } from './components/Dialog'
import { ErrorBoundary } from './components/ErrorBoundary'
import { FirstRunLangModal } from './components/FirstRunLangModal'
import { ToastProvider } from './components/Toast'
import { SettingsDataProvider } from './lib/SettingsData'
import { initTheme } from './lib/theme'
import './i18n'
import './index.css'

initTheme()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <ToastProvider>
        <DialogProvider>
          <SettingsDataProvider>
            <FirstRunLangModal />
            <App />
          </SettingsDataProvider>
        </DialogProvider>
      </ToastProvider>
    </ErrorBoundary>
  </React.StrictMode>,
)
