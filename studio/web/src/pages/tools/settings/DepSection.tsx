// 统一的「依赖安装 / 修复」卡片外壳 —— 系统 tab 环境区四件套共用
// （PyTorch / Flash Attention / xformers / ONNX Runtime）。
//
// 统一约定（此前四个组件各手抄一份，措辞/配色/⚠ 使用各不相同）：
// - summary 行：▸ + 标题 + ⓘ 用途说明 + 右侧状态词
// - 状态词颜色语义：ok=绿（正常）、warn=黄（可改善/未装）、err=红（坏了要修）、
//   loading=灰；warn/err 由壳统一加 ⚠ 前缀，包组件不要自己拼
// - 展开内容：状态卡（**包事实**：版本/build/EP）→ 警告条 → 动作行
//   （主按钮 + ↻ 刷新 + 右侧高级 toggle）→ 高级区
// - **机器事实**（驱动/平台/Python）一律不进卡片——见「环境」section 概览
import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { InfoButton } from '../../../components/InfoButton'
import { StatusLabel } from './modelCards'

export type DepLevel = 'ok' | 'warn' | 'err' | 'loading'

export interface DepNotice {
  key: string
  tone: 'err' | 'warn' | 'info'
  content: ReactNode
}

const LEVEL_CLASS: Record<DepLevel, string> = {
  ok: 'text-ok',
  warn: 'text-warn',
  err: 'text-err',
  loading: 'text-fg-tertiary',
}

const NOTICE_CLASS: Record<DepNotice['tone'], string> = {
  err: 'border-err bg-err-soft text-err',
  warn: 'border-warn bg-warn-soft text-warn',
  info: 'border-info bg-info-soft text-info',
}

export function DepSection({
  id, title, subtitle, helpTooltip,
  level, statusText, forceOpen,
  loadError, loading,
  infoCard, notices, primary, onRefresh, busy, advanced,
}: {
  id: string
  title: string
  subtitle?: string
  helpTooltip?: ReactNode
  level: DepLevel
  /** 状态词本体（不带 ⚠，壳按 level 统一加）。 */
  statusText: string
  /** 有问题时默认展开（details open）。 */
  forceOpen: boolean
  loadError?: string | null
  loading?: boolean
  /** 状态卡内容（包事实行）；用 DepVersionRow 保持第一行结构一致。 */
  infoCard?: ReactNode
  notices?: DepNotice[]
  primary?: {
    label: string
    onClick: () => void
    disabled?: boolean
    title?: string
    /** 修复场景（误装等）主按钮升格 btn-primary 强调；常规为 secondary。 */
    emphasized?: boolean
  }
  onRefresh?: () => void
  busy?: boolean
  advanced?: { label: string; open: boolean; onToggle: () => void; children: ReactNode }
}) {
  const { t } = useTranslation()
  const decorated = level === 'warn' || level === 'err' ? `⚠ ${statusText}` : statusText
  return (
    <details id={id} open={forceOpen} className="rounded-md border border-subtle bg-surface group scroll-mt-24">
      <summary className="cursor-pointer p-4 list-none flex items-center gap-2">
        <span className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3">▸</span>
        <h2 className="text-sm font-semibold text-fg-primary m-0">{title}</h2>
        {subtitle && <span className="text-xs text-fg-tertiary">{subtitle}</span>}
        {helpTooltip && <InfoButton>{helpTooltip}</InfoButton>}
        <span className={`ml-auto text-xs font-mono ${LEVEL_CLASS[level]}`}>{decorated}</span>
      </summary>

      <div className="px-4 pb-4 flex flex-col gap-3">
        {loadError && <div className="text-err text-xs font-mono">{loadError}</div>}
        {!loadError && loading && <div className="text-xs text-fg-tertiary">{t('settings.loadingStatus')}</div>}
        {!loadError && !loading && (
          <>
            {infoCard && (
              <div className="rounded-sm border border-subtle bg-sunken p-2 flex flex-col gap-1 text-xs">
                {infoCard}
              </div>
            )}
            {(notices ?? []).map((n) => (
              <div key={n.key} className={`rounded-sm border px-2 py-1.5 text-xs ${NOTICE_CLASS[n.tone]}`}>
                {n.content}
              </div>
            ))}
            {(primary || onRefresh || advanced) && (
              <div className="flex gap-1.5 items-center flex-wrap">
                {primary && (
                  <button
                    onClick={primary.onClick}
                    disabled={primary.disabled}
                    title={primary.title}
                    className={primary.emphasized ? 'btn btn-primary btn-sm' : 'btn btn-secondary btn-sm'}
                  >
                    {primary.label}
                  </button>
                )}
                {onRefresh && (
                  <button onClick={onRefresh} disabled={busy} title={t('settings.refreshStatus')}
                    className="px-2 py-0.5 text-fg-tertiary bg-transparent border-none cursor-pointer rounded-sm">↻</button>
                )}
                {advanced && (
                  <button type="button" onClick={advanced.onToggle}
                    className="btn btn-ghost btn-sm text-xs text-fg-tertiary ml-auto">
                    {advanced.open ? '▾' : '▸'} {advanced.label}
                  </button>
                )}
              </div>
            )}
            {advanced?.open && (
              <div className="flex flex-col gap-2 pt-2 border-t border-subtle">
                {advanced.children}
              </div>
            )}
          </>
        )}
      </div>
    </details>
  )
}

/** 状态卡第一行的统一结构：「<包名>: <版本> [徽标]」。 */
export function DepVersionRow({ name, value, badge }: {
  name: string
  value: string
  badge?: { text: string; ok: boolean }
}) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-fg-tertiary shrink-0">{name}:</span>
      <code className="font-mono text-fg-primary">{value}</code>
      {badge && (
        <StatusLabel
          bg={badge.ok ? 'bg-ok-soft' : 'bg-warn-soft'}
          fg={badge.ok ? 'text-ok' : 'text-warn'}
          text={badge.text}
        />
      )}
    </div>
  )
}
