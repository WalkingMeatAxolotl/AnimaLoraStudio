/**
 * LLM tagger 配置工作区 — 按 "LLM Settings redesign.html" 设计稿实现。
 *
 * 布局：preset bar 顶部 + workspace 双栏 grid (360px 左 / 1fr 右) + savebar 底部。
 * 左栏: 连接 (01) / 采样参数 (03) / 图片预处理 (04) 三张独立 card 纵向堆叠
 * 右栏: Prompt 模板 (02) composer 大 card
 *
 * 设计决策：
 * - 不做"预览请求 JSON" / "试跑一张" / token 价格统计（按用户决定）
 * - savebar 只保留「放弃修改」按钮；保存依赖全局 Settings 顶部"保存"按钮
 */
import { useMemo } from 'react'
import type { LLMPreset } from '../api/client'
import LLMMessagesEditor from './LLMMessagesEditor'

const MASK = '***'

interface Props {
  /** 卡片内顶部的 section 标题；与 SettingsSection 的 h2 视觉对齐。 */
  title?: string
  currentPreset: LLMPreset
  serverCurrentPreset?: LLMPreset
  presets: LLMPreset[]
  currentPresetId: string
  onSelectPreset: (id: string) => void
  onUpdatePreset: <K extends keyof LLMPreset>(field: K, value: LLMPreset[K]) => void
  onResetToBuiltin: () => void
  onSaveAs: () => void
  onAddPreset: () => void
  onDeletePreset: () => void
  llmModelsBusy: boolean
  llmTestBusy: boolean
  onRefreshModels: () => void
  onTestConnection: () => void
}

// ── 设计图字段 → 设计 token 直接映射的样式常量 ─────────────────────────
const inputStyle: React.CSSProperties = {
  width: '100%',
  background: 'var(--bg-sunken)',
  border: '1px solid var(--border-subtle)',
  borderRadius: 'var(--r-md)',
  padding: '8px 11px',
  fontSize: 'var(--t-sm)',
  color: 'var(--fg-primary)',
  fontFamily: 'var(--font-mono)',
  outline: 'none',
  transition: 'all 100ms',
}

export default function LLMTaggerWorkspace(props: Props) {
  const {
    title,
    currentPreset,
    serverCurrentPreset,
    presets,
    currentPresetId,
    onSelectPreset,
    onUpdatePreset,
    onResetToBuiltin,
    onSaveAs,
    onAddPreset,
    onDeletePreset,
    llmModelsBusy,
    llmTestBusy,
    onRefreshModels,
    onTestConnection,
  } = props

  // dirty diff: 比较 currentPreset (draft) 与 serverCurrentPreset (落盘)
  const dirtyCount = useMemo(() => {
    if (!serverCurrentPreset) return 0
    const keys: (keyof LLMPreset)[] = [
      'label', 'base_url', 'api_key', 'model', 'model_ids', 'endpoint',
      'messages', 'output_format', 'temperature', 'max_tokens',
      'max_side', 'jpeg_quality', 'max_image_mb', 'timeout', 'max_retries',
    ]
    let n = 0
    for (const k of keys) {
      const a = currentPreset[k]
      const b = serverCurrentPreset[k]
      // api_key 显示为 MASK 时视为未改
      if (k === 'api_key' && a === MASK) continue
      if (JSON.stringify(a) !== JSON.stringify(b)) n += 1
    }
    return n
  }, [currentPreset, serverCurrentPreset])

  return (
    // 整个 LLM 模块外层 — title + preset bar + workspace 5 个 section 包裹在同一个 card 里
    <div
      className="bg-surface border border-subtle"
      style={{ borderRadius: 'var(--r-lg)', overflow: 'hidden' }}
    >
      {/* 标题与 SettingsSection 的 h2 对齐：text-sm font-semibold + p-4 间距 */}
      {title && (
        <h2
          className="text-sm font-semibold text-fg-primary m-0"
          style={{
            padding: '14px 16px',
            borderBottom: '1px solid var(--border-subtle)',
          }}
        >
          {title}
        </h2>
      )}
      <PresetBar
        currentPreset={currentPreset}
        presets={presets}
        currentPresetId={currentPresetId}
        onSelectPreset={onSelectPreset}
        onResetToBuiltin={onResetToBuiltin}
        onSaveAs={onSaveAs}
        onAddPreset={onAddPreset}
        onDeletePreset={onDeletePreset}
        onUpdateLabel={(label) => onUpdatePreset('label', label)}
        dirtyCount={dirtyCount}
      />

      {/* workspace 双栏 grid：左 360px 三个 section 纵向 / 右 1fr composer 撑满 */}
      <div
        className="grid items-stretch"
        style={{ gridTemplateColumns: '360px 1fr' }}
      >
        {/* LEFT column：3 section 用 border-bottom 分隔，整列右边 border 跟右栏分隔 */}
        <div
          className="flex flex-col"
          style={{ borderRight: '1px solid var(--border-subtle)' }}
        >
          <ConnectionSection
            preset={currentPreset}
            serverPreset={serverCurrentPreset}
            onUpdate={onUpdatePreset}
            llmModelsBusy={llmModelsBusy}
            llmTestBusy={llmTestBusy}
            onRefreshModels={onRefreshModels}
            onTestConnection={onTestConnection}
            bottomBorder
          />
          <AdvancedSection preset={currentPreset} onUpdate={onUpdatePreset} />
        </div>

        {/* RIGHT column：composer 撑满高度；messages 区域内部滚动 */}
        <ComposerSection preset={currentPreset} onUpdate={onUpdatePreset} />
      </div>
    </div>
  )
}

// ── Preset bar ──────────────────────────────────────────────────────────
function PresetBar({
  currentPreset,
  presets,
  currentPresetId,
  onSelectPreset,
  onResetToBuiltin,
  onSaveAs,
  onAddPreset,
  onDeletePreset,
  onUpdateLabel,
  dirtyCount,
}: {
  currentPreset: LLMPreset
  presets: LLMPreset[]
  currentPresetId: string
  onSelectPreset: (id: string) => void
  onResetToBuiltin: () => void
  onSaveAs: () => void
  onAddPreset: () => void
  onDeletePreset: () => void
  onUpdateLabel: (s: string) => void
  dirtyCount: number
}) {
  return (
    <div
      style={{
        padding: '10px 12px 10px 16px',
        display: 'grid',
        gridTemplateColumns: '1fr auto',
        alignItems: 'center',
        gap: 12,
        borderBottom: '1px solid var(--border-subtle)',
      }}
    >
      <div className="flex items-center gap-3.5 min-w-0">
        <Caption>Preset</Caption>
        <div
          className="flex items-center gap-2.5 cursor-pointer"
          style={{
            background: 'var(--bg-sunken)',
            border: '1px solid var(--border-default)',
            borderRadius: 'var(--r-md)',
            padding: '7px 14px 7px 12px',
            fontSize: 'var(--t-sm)',
            color: 'var(--fg-primary)',
            fontWeight: 500,
            minWidth: 260,
            position: 'relative',
          }}
        >
          {currentPreset.builtin && (
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 'var(--t-2xs)',
                padding: '1px 6px',
                borderRadius: 'var(--r-sm)',
                background: 'var(--info-soft)',
                color: 'var(--info)',
                letterSpacing: '0.04em',
              }}
            >
              内置
            </span>
          )}
          {/* 用 select 覆盖整个 pick 让用户能切换；select 透明 */}
          <select
            value={currentPresetId}
            onChange={(e) => onSelectPreset(e.target.value)}
            className="absolute inset-0 opacity-0 cursor-pointer"
            aria-label="选择 preset"
          >
            {presets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.builtin ? '内置 · ' : ''}{p.label}
              </option>
            ))}
          </select>
          <span className="truncate">{currentPreset.label}</span>
          <span className="ml-auto" style={{ color: 'var(--fg-tertiary)' }}>▾</span>
        </div>
        {dirtyCount > 0 && (
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 'var(--t-xs)',
              color: 'var(--fg-tertiary)',
              whiteSpace: 'nowrap',
            }}
          >
            已修改 · <b style={{ color: 'var(--fg-secondary)', fontWeight: 500 }}>{dirtyCount} 处</b> 与默认不同
          </span>
        )}
        {/* 编辑当前 preset label */}
        <input
          type="text"
          value={currentPreset.label}
          onChange={(e) => onUpdateLabel(e.target.value)}
          className="hidden"
          aria-hidden
        />
      </div>
      <div className="flex items-center gap-1.5">
        {currentPreset.builtin && (
          <PBtn variant="danger" onClick={onResetToBuiltin} title="把当前内置预设重置为程序默认值">
            ↺ 重置默认
          </PBtn>
        )}
        <PBtn onClick={onSaveAs} title="复制当前 preset 字段为新预设">⎘ 另存为…</PBtn>
        <PBtn onClick={onAddPreset}>+ 新 Preset</PBtn>
        {!currentPreset.builtin && presets.length > 1 && (
          <PBtn variant="danger" onClick={onDeletePreset}>✕ 删除</PBtn>
        )}
      </div>
    </div>
  )
}

// ── Connection section (01) ────────────────────────────────────────────
function ConnectionSection({
  preset, serverPreset, onUpdate,
  llmModelsBusy, llmTestBusy,
  onRefreshModels, onTestConnection,
  bottomBorder,
}: {
  preset: LLMPreset
  serverPreset?: LLMPreset
  onUpdate: <K extends keyof LLMPreset>(field: K, value: LLMPreset[K]) => void
  llmModelsBusy: boolean
  llmTestBusy: boolean
  onRefreshModels: () => void
  onTestConnection: () => void
  bottomBorder?: boolean
}) {
  return (
    <Section bottomBorder={bottomBorder}>
      <SectionHeader step="01" title="连接" hint="openai-compatible" />
      <SectionBody>
        <Field label="Base URL" required help={<>可填 <Code>/v1</Code>、<Code>/chat/completions</Code> 或 <Code>/responses</Code></>}>
          <input
            type="text"
            value={preset.base_url}
            onChange={(e) => onUpdate('base_url', e.target.value)}
            placeholder="https://api.openai.com/v1"
            style={inputStyle}
            onFocus={(e) => e.currentTarget.style.boxShadow = '0 0 0 3px var(--accent-soft)'}
            onBlur={(e) => e.currentTarget.style.boxShadow = 'none'}
          />
        </Field>

        <Field label="API Key" required>
          <InputWithSuffix>
            <SensitiveInput
              value={preset.api_key}
              serverValue={serverPreset?.api_key ?? ''}
              onChange={(v) => onUpdate('api_key', v)}
            />
          </InputWithSuffix>
        </Field>

        <Field label="Model" required>
          <InputWithSuffix
            suffix={
              <ChipButton onClick={onRefreshModels} disabled={llmModelsBusy || !preset.base_url.trim()}>
                {llmModelsBusy ? '读取中…' : '从服务器拉取'}
              </ChipButton>
            }
          >
            {preset.model_ids.length > 0 ? (
              <select
                value={preset.model}
                onChange={(e) => onUpdate('model', e.target.value)}
                style={{ ...inputStyle, paddingRight: 130 }}
              >
                {!preset.model_ids.includes(preset.model) && preset.model && (
                  <option value={preset.model}>{preset.model}</option>
                )}
                {preset.model_ids.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={preset.model}
                onChange={(e) => onUpdate('model', e.target.value)}
                placeholder="gpt-4.1-mini / qwen-vl-max / 本地模型名"
                style={{ ...inputStyle, paddingRight: 130 }}
              />
            )}
          </InputWithSuffix>
        </Field>

        <Field label="Endpoint 风格">
          {/* 测试连接按钮内联在 Segmented 末尾；结果走 toast 通知，不在 UI 常驻。 */}
          <div className="flex items-center gap-2">
            <div className="flex-1 min-w-0">
              <Segmented
                value={preset.endpoint}
                onChange={(v) => onUpdate('endpoint', v)}
                options={[
                  { value: 'chat_completions', label: 'CHAT COMPLETIONS' },
                  { value: 'responses', label: 'RESPONSES' },
                ]}
              />
            </div>
            <ChipButton
              onClick={onTestConnection}
              disabled={llmTestBusy || !preset.base_url.trim() || !preset.model.trim()}
            >
              {llmTestBusy ? '测试中…' : '测试连接'}
            </ChipButton>
          </div>
        </Field>
      </SectionBody>
    </Section>
  )
}

// ── Sampling section (03) ───────────────────────────────────────────────
// ── 高级参数：默认折叠的 details 面板，包住 03 采样 + 04 图片预处理 ──────
// 折叠时显示 summary 行（temp / max / max-side / q）；展开后内部完整渲染两个 sub-section。
function AdvancedSection({ preset, onUpdate }: {
  preset: LLMPreset
  onUpdate: <K extends keyof LLMPreset>(field: K, value: LLMPreset[K]) => void
}) {
  return (
    <details className="group">
      <summary
        className="cursor-pointer list-none flex items-center justify-between gap-2"
        style={{
          padding: '11px 16px 10px',
          // 用 SectionHeader 同款下边框；展开时由内层 SamplingSection 自带 border 继续分隔。
          borderBottom: '1px solid var(--border-subtle)',
        }}
      >
        <h3
          className="m-0 flex items-center gap-2 whitespace-nowrap"
          style={{ fontSize: 'var(--t-md)', fontWeight: 600, letterSpacing: '-0.005em' }}
        >
          <span
            className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3"
            style={{ fontFamily: 'var(--font-mono)' }}
          >
            ▸
          </span>
          <Step>⚙</Step>
          <span>高级参数</span>
        </h3>
        {/* summary 值：折叠时显示当前关键数值（紧凑形式，避免 360px 左栏装不下）。
         * truncate + min-w-0 让超长时优雅省略而不是把标题挤竖。 */}
        <span
          className="group-open:hidden truncate min-w-0"
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 'var(--t-xs)',
            color: 'var(--fg-tertiary)',
          }}
          title={`temperature ${preset.temperature} · max_tokens ${preset.max_tokens} · max_side ${preset.max_side}px · jpeg_quality ${preset.jpeg_quality}`}
        >
          {preset.temperature} · {preset.max_tokens}t · {preset.max_side}px · q{preset.jpeg_quality}
        </span>
      </summary>
      {/* 展开内容：03 采样 + 04 图片预处理 原样堆叠 */}
      <SamplingSection preset={preset} onUpdate={onUpdate} bottomBorder />
      <ImageSection preset={preset} onUpdate={onUpdate} />
    </details>
  )
}

function SamplingSection({ preset, onUpdate, bottomBorder }: {
  preset: LLMPreset
  onUpdate: <K extends keyof LLMPreset>(field: K, value: LLMPreset[K]) => void
  bottomBorder?: boolean
}) {
  return (
    <Section bottomBorder={bottomBorder}>
      <SectionHeader step="03" title="采样参数" hint="model-side" />
      <SectionBody>
        <Field label="Temperature" optional="— 越低越稳定">
          <SliderRow value={preset.temperature} min={0} max={2} step={0.05}
            onChange={(v) => onUpdate('temperature', v)} />
        </Field>
        <Field label="Max tokens">
          <SliderRow value={preset.max_tokens} min={64} max={4096} step={32}
            onChange={(v) => onUpdate('max_tokens', Math.round(v))} integer />
        </Field>
        <Row2>
          <Field label="Timeout" optional="s">
            <input
              type="number" min={5} max={600}
              value={preset.timeout}
              onChange={(e) => onUpdate('timeout', Math.max(5, Number(e.target.value) || 5))}
              style={inputStyle}
            />
          </Field>
          <Field label="Max retries">
            <input
              type="number" min={1} max={10}
              value={preset.max_retries}
              onChange={(e) => onUpdate('max_retries', Math.max(1, Number(e.target.value) || 1))}
              style={inputStyle}
            />
          </Field>
        </Row2>
      </SectionBody>
    </Section>
  )
}

// ── Image preprocessing section (04) ────────────────────────────────────
function ImageSection({ preset, onUpdate, bottomBorder }: {
  preset: LLMPreset
  onUpdate: <K extends keyof LLMPreset>(field: K, value: LLMPreset[K]) => void
  bottomBorder?: boolean
}) {
  return (
    <Section bottomBorder={bottomBorder}>
      <SectionHeader step="04" title="图片预处理" hint="before upload" />
      <SectionBody>
        <Field label="Max side" optional="px · 缩放最长边">
          <SliderRow value={preset.max_side} min={512} max={2048} step={64}
            onChange={(v) => onUpdate('max_side', Math.round(v))} integer />
        </Field>
        <Row2>
          <Field label="JPEG quality">
            <input
              type="number" min={1} max={100}
              value={preset.jpeg_quality}
              onChange={(e) => onUpdate('jpeg_quality',
                Math.max(1, Math.min(100, Number(e.target.value) || 85)))}
              style={inputStyle}
            />
          </Field>
          <Field label="Max size" optional="MB">
            <input
              type="number" min={0.1} max={25} step={0.1}
              value={preset.max_image_mb}
              onChange={(e) => onUpdate('max_image_mb',
                Math.max(0.1, Number(e.target.value) || 5))}
              style={inputStyle}
            />
          </Field>
        </Row2>
        <Help>
          Claude 等服务限制 <b style={{ color: 'var(--fg-secondary)' }}>5 MB</b> / 张。超过会被压缩到此值以下。
        </Help>
      </SectionBody>
    </Section>
  )
}

// ── Composer section (02) — the hero ────────────────────────────────────
// 高度撑满左栏总高（grid items-stretch + h-full）；messages 区域内部滚动。
function ComposerSection({ preset, onUpdate }: {
  preset: LLMPreset
  onUpdate: <K extends keyof LLMPreset>(field: K, value: LLMPreset[K]) => void
}) {
  return (
    <div
      className="flex flex-col"
      style={{ height: '100%', minHeight: 0, overflow: 'hidden' }}
    >
      {/* composer-tabbar — 固定高 */}
      <div
        className="flex items-center justify-between shrink-0"
        style={{ padding: '12px 16px', borderBottom: '1px solid var(--border-subtle)' }}
      >
        <div className="flex items-center gap-4">
          <h3 className="flex items-center gap-2 m-0" style={{
            fontSize: 'var(--t-md)', fontWeight: 600, letterSpacing: '-0.005em',
          }}>
            <Step>02</Step>
            <span>Prompt 模板</span>
          </h3>
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
            color: 'var(--fg-tertiary)', letterSpacing: '0.04em',
          }}>
            {preset.messages.length} 条消息 · 拖动调整顺序
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="flex items-center" style={{
            background: 'var(--bg-sunken)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--r-md)',
            padding: '4px 6px 4px 12px',
            gap: 8,
          }}>
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
              color: 'var(--fg-tertiary)', letterSpacing: '0.06em',
              textTransform: 'uppercase',
            }}>Output</span>
            <select
              value={preset.output_format}
              onChange={(e) => onUpdate('output_format', e.target.value as LLMPreset['output_format'])}
              className="cursor-pointer outline-none border-0"
              style={{
                background: 'transparent', color: 'var(--fg-primary)',
                fontSize: 'var(--t-sm)', padding: '4px 6px',
              }}
            >
              <option value="json">JSON caption</option>
              <option value="text">Text caption</option>
            </select>
          </div>
        </div>
      </div>

      {/* msg-list — flex-1 + 内滚 */}
      <div
        className="flex-1 min-h-0 overflow-y-auto"
        style={{ padding: '14px 16px' }}
      >
        {preset.endpoint === 'responses' && (
          <div
            style={{
              fontSize: 'var(--t-xs)', color: 'var(--warn)',
              fontFamily: 'var(--font-mono)', letterSpacing: '0.04em',
              marginBottom: 10,
            }}
          >
            ⚠️ Responses endpoint 只用 system + 第一条 user；其他 messages 会被忽略
          </div>
        )}
        <LLMMessagesEditor
          messages={preset.messages}
          onChange={(msgs) => onUpdate('messages', msgs)}
        />
      </div>
    </div>
  )
}

// ── Reusable primitives ─────────────────────────────────────────────────

/** 左栏内的一个 section（无独立 border + radius；section 之间用 bottomBorder 分隔）。 */
function Section({ children, bottomBorder }: {
  children: React.ReactNode
  bottomBorder?: boolean
}) {
  return (
    <div style={bottomBorder ? { borderBottom: '1px solid var(--border-subtle)' } : undefined}>
      {children}
    </div>
  )
}

function SectionHeader({ step, title, hint }: { step: string; title: string; hint?: string }) {
  return (
    <div
      className="flex items-center justify-between"
      style={{
        padding: '11px 16px 10px',
        borderBottom: '1px solid var(--border-subtle)',
      }}
    >
      <h3 className="m-0 flex items-center gap-2" style={{
        fontSize: 'var(--t-md)', fontWeight: 600, letterSpacing: '-0.005em',
      }}>
        <Step>{step}</Step>
        <span>{title}</span>
      </h3>
      {hint && (
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: 'var(--t-xs)',
          color: 'var(--fg-tertiary)',
        }}>
          {hint}
        </span>
      )}
    </div>
  )
}

function SectionBody({ children }: { children: React.ReactNode }) {
  return <div style={{ padding: '12px 16px 14px' }}>{children}</div>
}

function Step({ children }: { children: React.ReactNode }) {
  return (
    <span style={{
      fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
      color: 'var(--accent)', background: 'var(--accent-soft)',
      borderRadius: 999, padding: '2px 7px', letterSpacing: '0.04em',
    }}>
      {children}
    </span>
  )
}

function Caption({ children }: { children: React.ReactNode }) {
  return (
    <span style={{
      fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
      color: 'var(--fg-tertiary)', letterSpacing: '0.08em',
      textTransform: 'uppercase', whiteSpace: 'nowrap',
    }}>
      {children}
    </span>
  )
}

function Field({ label, required, optional, help, children }: {
  label: string
  required?: boolean
  optional?: string
  help?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="grid" style={{ gap: 5, marginBottom: 12 }}>
      <label
        className="flex items-center gap-1.5"
        style={{
          fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
          color: 'var(--fg-secondary)', letterSpacing: '0.06em',
          textTransform: 'uppercase',
        }}
      >
        {label}
        {required && <span style={{ color: 'var(--err)' }}>*</span>}
        {optional && (
          <span style={{
            fontWeight: 400, textTransform: 'none', letterSpacing: 0,
            color: 'var(--fg-tertiary)', fontFamily: 'var(--font-sans)',
          }}>
            {optional}
          </span>
        )}
      </label>
      {children}
      {help && <Help>{help}</Help>}
    </div>
  )
}

function Help({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontSize: 'var(--t-xs)', color: 'var(--fg-tertiary)', lineHeight: 1.45,
    }}>
      {children}
    </div>
  )
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <code style={{
      fontFamily: 'var(--font-mono)',
      color: 'var(--fg-secondary)',
    }}>
      {children}
    </code>
  )
}

function Row2({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', gap: 12 }}>
      {children}
    </div>
  )
}

function InputWithSuffix({ children, suffix }: {
  children: React.ReactNode
  suffix?: React.ReactNode
}) {
  return (
    <div style={{ position: 'relative' }}>
      {children}
      {suffix && (
        <div
          className="flex items-center gap-0.5"
          style={{
            position: 'absolute', right: 4, top: '50%',
            transform: 'translateY(-50%)',
          }}
        >
          {suffix}
        </div>
      )}
    </div>
  )
}

function ChipButton({ children, onClick, disabled, active }: {
  children: React.ReactNode
  onClick?: () => void
  disabled?: boolean
  active?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      style={{
        background: 'transparent',
        border: '1px solid var(--border-subtle)',
        color: active ? 'var(--accent)' : 'var(--fg-secondary)',
        borderColor: active ? 'var(--accent)' : 'var(--border-subtle)',
        fontSize: 'var(--t-2xs)',
        padding: '4px 8px',
        borderRadius: 'var(--r-sm)',
        fontFamily: 'var(--font-mono)',
        letterSpacing: '0.04em',
        opacity: disabled ? 0.5 : 1,
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
      onMouseEnter={(e) => {
        if (disabled) return
        e.currentTarget.style.color = 'var(--accent)'
        e.currentTarget.style.borderColor = 'var(--accent)'
      }}
      onMouseLeave={(e) => {
        if (disabled || active) return
        e.currentTarget.style.color = 'var(--fg-secondary)'
        e.currentTarget.style.borderColor = 'var(--border-subtle)'
      }}
    >
      {children}
    </button>
  )
}

function PBtn({ children, onClick, variant, title }: {
  children: React.ReactNode
  onClick?: () => void
  variant?: 'default' | 'primary' | 'danger'
  title?: string
}) {
  const base: React.CSSProperties = {
    background: variant === 'primary' ? 'var(--accent)' : 'transparent',
    border: '1px solid transparent',
    color: variant === 'primary' ? 'var(--accent-fg)' : 'var(--fg-secondary)',
    padding: variant === 'primary' ? '7px 14px' : '6px 10px',
    borderRadius: 'var(--r-md)',
    fontSize: 'var(--t-sm)',
    fontWeight: variant === 'primary' ? 500 : 400,
    display: 'inline-flex', alignItems: 'center', gap: 6,
    cursor: 'pointer',
    whiteSpace: 'nowrap',
  }
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      style={base}
      onMouseEnter={(e) => {
        if (variant === 'primary') {
          e.currentTarget.style.background = 'var(--accent-hover)'
        } else if (variant === 'danger') {
          e.currentTarget.style.background = 'var(--err-soft)'
          e.currentTarget.style.color = 'var(--err)'
        } else {
          e.currentTarget.style.background = 'var(--bg-overlay)'
          e.currentTarget.style.color = 'var(--fg-primary)'
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = variant === 'primary' ? 'var(--accent)' : 'transparent'
        e.currentTarget.style.color = variant === 'primary' ? 'var(--accent-fg)' : 'var(--fg-secondary)'
      }}
    >
      {children}
    </button>
  )
}

function Segmented<T extends string>({ value, onChange, options }: {
  value: T
  onChange: (v: T) => void
  options: { value: T; label: string }[]
}) {
  return (
    <div
      className="inline-flex w-full"
      style={{
        padding: 3, gap: 2,
        background: 'var(--bg-sunken)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--r-md)',
      }}
    >
      {options.map((opt) => {
        const on = value === opt.value
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            style={{
              flex: 1,
              background: on ? 'var(--bg-elevated)' : 'transparent',
              color: on ? 'var(--fg-primary)' : 'var(--fg-tertiary)',
              boxShadow: on ? 'var(--sh-sm)' : 'none',
              border: 0,
              padding: '6px 10px',
              borderRadius: 4,
              fontSize: 'var(--t-xs)',
              fontFamily: 'var(--font-mono)',
              cursor: 'pointer',
              letterSpacing: '0.04em',
            }}
          >
            {opt.label}
          </button>
        )
      })}
    </div>
  )
}

function SliderRow({ value, min, max, step, onChange, integer }: {
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  integer?: boolean
}) {
  const clamp = (v: number) => Math.max(min, Math.min(max, v))
  return (
    <div className="grid items-center" style={{ gridTemplateColumns: '1fr 64px', gap: 10 }}>
      <input
        type="range"
        min={min} max={max} step={step ?? 1}
        value={value}
        onChange={(e) => onChange(clamp(Number(e.target.value)))}
        style={{ width: '100%', accentColor: 'var(--accent)' }}
      />
      <input
        type="number"
        min={min} max={max} step={step ?? 1}
        value={integer ? Math.round(value) : value}
        onChange={(e) => onChange(clamp(Number(e.target.value)))}
        style={{
          background: 'var(--bg-sunken)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--r-sm)',
          padding: '4px 8px',
          fontFamily: 'var(--font-mono)',
          fontSize: 'var(--t-xs)',
          color: 'var(--fg-primary)',
          textAlign: 'right',
          outline: 'none',
          width: '100%',
        }}
      />
    </div>
  )
}

function SensitiveInput({ value, serverValue, onChange }: {
  value: string
  serverValue: string
  onChange: (v: string) => void
}) {
  const masked = value === MASK
  return (
    <input
      type="password"
      value={masked ? '' : value}
      placeholder={serverValue === MASK ? '已保存（不显示），输入新值才覆盖' : ''}
      onChange={(e) => onChange(e.target.value || MASK)}
      style={inputStyle}
    />
  )
}
