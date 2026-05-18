// FirstRunLangModal —— localStorage 没存过 studio.lang 时弹一次的语言选择。
// 设计动机：onboarding 第一帧的身份级选择，强制二选一、不可绕过、不预选高亮，
// 替代 PR #76 那个塞在 studio.bat/sh 里的 CLI prompt（触达率低 + ASCII 受限 +
// 无法预览要选的语言长什么样）。
import { useEffect, useRef, useState } from 'react'
import i18n, { getStoredLang, setStoredLang } from '../i18n'

type Lang = 'en' | 'zh'

const STORAGE_KEY = 'studio.lang'

function detectPreferredLang(): Lang {
  try {
    const navLang = (navigator.language || '').toLowerCase()
    if (navLang.startsWith('zh')) return 'zh'
    return 'en'
  } catch {
    return 'zh'
  }
}

export function FirstRunLangModal() {
  // 用 lazy initializer 在 mount 时只读一次 localStorage，避免 SSR / hydration 抖动。
  const [open, setOpen] = useState<boolean>(() => getStoredLang() === null)
  const preferred = useRef<Lang>(detectPreferredLang())
  const focusRef = useRef<HTMLButtonElement | null>(null)

  // 进入时把键盘焦点落到「检测到的」那张卡——A11y 友好（键盘用户敲 Enter 直接确认
  // 检测语言），视觉上两张卡仍然对等。
  useEffect(() => {
    if (open) focusRef.current?.focus()
  }, [open])

  if (!open) return null

  const handlePick = (lang: Lang) => {
    setStoredLang(lang)
    void i18n.changeLanguage(lang)
    setOpen(false)
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="first-run-lang-title"
      // 不可绕过：不监听 Esc，不监听点击遮罩。这是 onboarding 不是 Dialog。
      // 暗色半透明幕布 + 高斯模糊，把底层 App 推到背景；window 仍是普通尺寸卡片。
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-md"
      data-testid="first-run-lang-modal"
    >
      <div className="w-[90%] max-w-[480px] flex flex-col gap-6 p-8 bg-elevated border border-dim rounded-lg shadow-xl">
        <h1
          id="first-run-lang-title"
          className="m-0 text-center text-2xl font-semibold text-fg-primary"
        >
          Welcome · 欢迎
        </h1>

        <div className="grid grid-cols-2 gap-4">
          <button
            ref={preferred.current === 'en' ? focusRef : undefined}
            type="button"
            onClick={() => handlePick('en')}
            className="flex flex-col gap-2 items-center justify-center px-6 py-8 rounded-lg border border-dim bg-surface hover:border-accent hover:bg-accent-soft transition-colors cursor-pointer"
            data-testid="first-run-lang-en"
          >
            <span className="text-xl font-semibold text-fg-primary">English</span>
            <span className="text-xs text-fg-tertiary">Train your LoRA</span>
          </button>

          <button
            ref={preferred.current === 'zh' ? focusRef : undefined}
            type="button"
            onClick={() => handlePick('zh')}
            className="flex flex-col gap-2 items-center justify-center px-6 py-8 rounded-lg border border-dim bg-surface hover:border-accent hover:bg-accent-soft transition-colors cursor-pointer"
            data-testid="first-run-lang-zh"
          >
            <span className="text-xl font-semibold text-fg-primary">中文</span>
            <span className="text-xs text-fg-tertiary">训练你的 LoRA</span>
          </button>
        </div>

        <p className="m-0 text-center text-xs text-fg-tertiary leading-relaxed">
          You can change this anytime in Settings · 之后可以在设置中切换
        </p>
      </div>
    </div>
  )
}

// 测试用：清掉 localStorage，让 modal 在 dev console 里手动复现。
// 生产代码不引用。
export const __STORAGE_KEY = STORAGE_KEY
