import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { SchemaProperty } from '../api/client'
import { useProjectCtx } from '../context/ProjectContext'
import { controlKind, fieldLabel, schemaEnumLabel } from '../lib/schema'
import PathPicker from './PathPicker'
import ResumeFieldPicker from './ResumeFieldPicker'

interface Props {
  name: string
  prop: SchemaProperty
  value: unknown
  onChange: (v: unknown) => void
  /** disabled 状态（自动控制字段灰显 readonly）。 */
  disabled?: boolean
  /** 字段标签后的小徽章（如「自动 · 全局设置」/「自动 · 项目设置」）。
   * 与 disabled 解耦：可以让字段保持可编辑只挂个徽章作信息提示，也可以
   * 配合 disabled 来表达「这字段被自动填且不让你改」。支持 ReactNode 以便
   * 嵌入可点击链接（如跳转到 Settings 对应区段）。 */
  hint?: React.ReactNode
  /** 覆盖 prop.description 的说明文字（用于条件上下文描述）。 */
  descriptionOverride?: string
  /** path 字段右侧额外按钮槽（如「↺ 重置为全局默认」）。仅对 string/path
   * 字段渲染；其他类型字段忽略。 */
  suffix?: React.ReactNode
}

// input 覆盖 .input 默认值（更紧凑；背景用 canvas 而不是 surface）
const inputStyle: React.CSSProperties = {
  width: '100%', padding: '5px 10px',
  background: 'var(--bg-canvas)', border: '1px solid var(--border-default)',
  borderRadius: 'var(--r-sm)', fontSize: 'var(--t-sm)',
  color: 'var(--fg-primary)',
}

const FieldHint = ({ children }: { children: React.ReactNode }) => (
  <span className="ml-2 text-[11px] text-warn align-middle">{children}</span>
)

/** 单个表单字段，按 control kind 分发渲染。 */
export default function Field({
  name, prop, value, onChange, disabled = false, hint, descriptionOverride, suffix,
}: Props) {
  const { t } = useTranslation()
  const kind = controlKind(prop)
  const label = fieldLabel(name)
  const help = descriptionOverride ?? prop.description
  const hintText = hint ?? (disabled ? t('field.autoProject') : null)
  const hintNode = hintText ? <FieldHint>{hintText}</FieldHint> : null
  void name

  // bool ----------------------------------------------------------------
  if (kind === 'bool') {
    return (
      <label className={`flex items-start gap-3 py-1.5 ${disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}`}>
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
          disabled={disabled}
          style={{ marginTop: 4, height: 16, width: 16, borderRadius: 'var(--r-sm)' }}
        />
        <span className="flex-1">
          <div className="text-sm text-fg-primary">
            {label}
            {hintNode}
          </div>
          {help && <div className="text-xs text-fg-tertiary mt-1">{help}</div>}
        </span>
      </label>
    )
  }

  // tristate (Optional[bool]: null / true / false) ----------------------
  if (kind === 'tristate') {
    const triValue = value === true ? 'true' : value === false ? 'false' : ''
    return (
      <div className="py-1.5">
        <div className="text-sm font-medium text-fg-secondary mb-1">
          {label}{hintNode}
        </div>
        <select
          value={triValue}
          onChange={(e) => {
            const v = e.target.value
            onChange(v === 'true' ? true : v === 'false' ? false : null)
          }}
          disabled={disabled}
          className="input" style={inputStyle}
        >
          <option value="">{t('field.useGlobal')}</option>
          <option value="true">{t('field.yes')}</option>
          <option value="false">{t('field.no')}</option>
        </select>
        {help && <div className="text-xs text-fg-tertiary mt-1">{help}</div>}
      </div>
    )
  }

  // select --------------------------------------------------------------
  if (kind === 'select') {
    return (
      <div className="py-1.5">
        <div className="text-sm font-medium text-fg-secondary mb-1">
          {label}{hintNode}
        </div>
        <select
          value={String(value ?? '')}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className="input" style={inputStyle}
        >
          {(prop.enum ?? []).map((opt) => (
            <option key={String(opt)} value={String(opt)}>
              {schemaEnumLabel(name, opt, t)}
            </option>
          ))}
        </select>
        {help && <div className="text-xs text-fg-tertiary mt-1">{help}</div>}
      </div>
    )
  }

  // textarea ------------------------------------------------------------
  if (kind === 'textarea') {
    return (
      <div className="py-1.5">
        <div className="text-sm font-medium text-fg-secondary mb-1">
          {label}{hintNode}
        </div>
        <textarea
          rows={3}
          value={String(value ?? '')}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className="input input-mono" style={inputStyle}
        />
        {help && <div className="text-xs text-fg-tertiary mt-1">{help}</div>}
      </div>
    )
  }

  // string-list ---------------------------------------------------------
  if (kind === 'string-list') {
    const list = Array.isArray(value) ? (value as string[]) : []
    const text = list.join('\n')
    return (
      <div className="py-1.5">
        <div className="text-sm font-medium text-fg-secondary mb-1">
          {label}{t('field.multilineHint')}{hintNode}
        </div>
        <textarea
          rows={Math.max(3, list.length + 1)}
          value={text}
          onChange={(e) => {
            const arr = e.target.value
              .split('\n')
              .map((s) => s.trim())
              .filter((s) => s.length > 0)
            onChange(arr)
          }}
          disabled={disabled}
          className="input input-mono" style={inputStyle}
        />
        {help && <div className="text-xs text-fg-tertiary mt-1">{help}</div>}
      </div>
    )
  }

  // int-list (e.g. resolution: [512, 768, 1024]) -----------------------
  if (kind === 'int-list') {
    return (
      <IntListField
        label={label}
        help={help}
        value={value}
        defaultValue={prop.default}
        onChange={onChange}
        disabled={disabled}
        hintNode={hintNode}
      />
    )
  }

  // code ----------------------------------------------------------------
  if (kind === 'code') {
    return (
      <JsonCodeField
        label={label}
        help={help}
        value={value}
        onChange={onChange}
        disabled={disabled}
        hintNode={hintNode}
      />
    )
  }

  // int / float ---------------------------------------------------------
  if (kind === 'int' || kind === 'float') {
    return (
      <NumberField
        label={label}
        kind={kind}
        help={help}
        value={value}
        defaultValue={prop.default}
        minimum={prop.minimum}
        maximum={prop.maximum}
        onChange={onChange}
        disabled={disabled}
        hintNode={hintNode}
      />
    )
  }

  // string / path -------------------------------------------------------
  return (
    <PathStringField
      name={name}
      label={label}
      kind={kind}
      help={help}
      value={value}
      onChange={onChange}
      disabled={disabled}
      hintNode={hintNode}
      suffix={suffix}
    />
  )
}

interface JsonCodeFieldProps {
  label: string
  help: string | undefined
  value: unknown
  onChange: (v: unknown) => void
  disabled?: boolean
  hintNode?: React.ReactNode
}

function formatJsonCode(v: unknown): string {
  if (v === null || v === undefined || v === '') return ''
  if (typeof v === 'string') return v
  return JSON.stringify(v, null, 2)
}

function JsonCodeField({
  label, help, value, onChange, disabled = false, hintNode,
}: JsonCodeFieldProps) {
  const [raw, setRaw] = useState<string>(() => formatJsonCode(value))
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    if (document.activeElement !== inputRef.current) {
      setRaw(formatJsonCode(value))
      setError(null)
    }
  }, [value])

  const commit = () => {
    const text = raw.trim()
    if (text === '') {
      onChange(null)
      setError(null)
      return
    }
    try {
      const parsed = JSON.parse(text) as unknown
      if (parsed === null || typeof parsed !== 'object') {
        setError('JSON must be an object or array')
        return
      }
      onChange(parsed)
      setRaw(JSON.stringify(parsed, null, 2))
      setError(null)
    } catch {
      setError('Invalid JSON')
    }
  }

  return (
    <div className="py-1.5">
      <div className="text-sm font-medium text-fg-secondary mb-1">
        {label}{hintNode}
      </div>
      <textarea
        ref={inputRef}
        rows={Math.max(3, raw.split('\n').length + 1)}
        value={raw}
        onChange={(e) => {
          setRaw(e.target.value)
          setError(null)
        }}
        onBlur={commit}
        disabled={disabled}
        className="input input-mono"
        style={inputStyle}
      />
      {error && <div className="text-xs text-err mt-1">{error}</div>}
      {help && <div className="text-xs text-fg-tertiary mt-1">{help}</div>}
    </div>
  )
}

interface IntListFieldProps {
  label: string
  help: string | undefined
  value: unknown
  defaultValue: unknown
  onChange: (v: unknown) => void
  disabled?: boolean
  hintNode?: React.ReactNode
}

/** 整数列表输入（如 resolution: [512, 768, 1024]）。逗号或空格分隔；后端 validator
 *  负责 snap/clamp，前端只收集数字。清空后回落到默认值（与 NumberField 一致）。 */
function IntListField({
  label, help, value, defaultValue, onChange, disabled = false, hintNode,
}: IntListFieldProps) {
  const fmt = (v: unknown) =>
    Array.isArray(v) ? (v as number[]).join(', ') : v === null || v === undefined ? '' : String(v)
  const [raw, setRaw] = useState<string>(() => fmt(value))
  const inputRef = useRef<HTMLInputElement | null>(null)
  // placeholder 纯由该字段的 default 派生（通用组件，不写死任何字段专属值）
  const placeholder = fmt(defaultValue)

  useEffect(() => {
    if (document.activeElement !== inputRef.current) setRaw(fmt(value))
  }, [value])

  const commit = () => {
    const nums = raw
      .split(/[,\s]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
      .map((s) => parseInt(s, 10))
      .filter((n) => Number.isFinite(n))
    // 清空 → 回落默认值（避免存空列表）
    if (nums.length === 0 && Array.isArray(defaultValue)) {
      onChange(defaultValue)
      setRaw(fmt(defaultValue))
      return
    }
    onChange(nums)
    setRaw(nums.join(', '))
  }

  return (
    <div className="py-1.5">
      <div className="text-sm font-medium text-fg-secondary mb-1">
        {label}{hintNode}
      </div>
      <input
        ref={inputRef}
        type="text"
        inputMode="numeric"
        value={raw}
        onChange={(e) => setRaw(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault()
            commit()
          }
        }}
        disabled={disabled}
        className="input input-mono" style={inputStyle}
        placeholder={placeholder}
      />
      {help && <div className="text-xs text-fg-tertiary mt-1">{help}</div>}
    </div>
  )
}

interface NumberFieldProps {
  label: string
  kind: 'int' | 'float'
  help: string | undefined
  value: unknown
  defaultValue: unknown
  minimum?: number
  maximum?: number
  onChange: (v: unknown) => void
  disabled?: boolean
  hintNode?: React.ReactNode
}

function NumberField({
  label, kind, help, value, defaultValue, minimum, maximum,
  onChange, disabled = false, hintNode,
}: NumberFieldProps) {
  const formatNum = (v: unknown) =>
    v === null || v === undefined ? '' : String(v)
  const [raw, setRaw] = useState<string>(() => formatNum(value))
  const inputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    if (document.activeElement !== inputRef.current) {
      setRaw(formatNum(value))
    }
  }, [value])

  const commit = () => {
    if (raw === '') {
      onChange(defaultValue)
      setRaw(formatNum(defaultValue))
      return
    }
    const num = kind === 'int' ? parseInt(raw, 10) : parseFloat(raw)
    if (Number.isNaN(num)) {
      setRaw(formatNum(value))
      return
    }
    if (
      (minimum !== undefined && num < minimum) ||
      (maximum !== undefined && num > maximum)
    ) {
      setRaw(formatNum(value))
      return
    }
    onChange(num)
    setRaw(formatNum(num))
  }

  return (
    <div className="py-1.5">
      <div className="text-sm font-medium text-fg-secondary mb-1">
        {label}{hintNode}
      </div>
      <input
        ref={inputRef}
        type="text"
        inputMode={kind === 'int' ? 'numeric' : 'decimal'}
        value={raw}
        onChange={(e) => setRaw(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault()
            commit()
          }
        }}
        disabled={disabled}
        className="input input-mono" style={inputStyle}
      />
      {help && <div className="text-xs text-fg-tertiary mt-1">{help}</div>}
    </div>
  )
}

interface PathFieldProps {
  /** schema 字段名，让 path 字段判定是否走专用 picker（resume_state / resume_lora）。 */
  name: string
  label: string
  kind: 'path' | 'string'
  help: string | undefined
  value: unknown
  onChange: (v: unknown) => void
  disabled?: boolean
  hintNode?: React.ReactNode
  /** 输入行右侧额外按钮槽（如重置按钮）。 */
  suffix?: React.ReactNode
}

function PathStringField({
  name, label, kind, help, value, onChange, disabled = false, hintNode, suffix,
}: PathFieldProps) {
  const { t } = useTranslation()
  const [picking, setPicking] = useState(false)
  const text = value === null || value === undefined ? '' : String(value)
  const browseBtnRef = useRef<HTMLButtonElement | null>(null)
  const projectCtx = useProjectCtx()

  // resume_state / resume_lora：走项目内语义 picker（dropdown），用户看不到深路径。
  // 外部文件用户直接在 input 手填即可。
  const resumeKind: 'state' | 'lora' | null =
    name === 'resume_state' ? 'state' :
    name === 'resume_lora' ? 'lora' : null
  const useResumePicker = kind === 'path' && resumeKind !== null && projectCtx !== null

  return (
    <div className="py-1.5 relative">
      <div className="text-sm font-medium text-fg-secondary mb-1">
        {label}
        {kind === 'path' && (
          <span className="ml-2 text-xs text-fg-tertiary">{t('field.pathHint')}</span>
        )}
        {hintNode}
      </div>
      <div className="flex gap-2">
        <input
          type="text"
          value={text}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className={'input' + (kind === 'path' ? ' input-mono' : '')} style={inputStyle}
        />
        {kind === 'path' && (
          <button
            ref={browseBtnRef}
            type="button"
            onClick={() => setPicking((p) => !p)}
            disabled={disabled}
            className="btn btn-secondary btn-sm shrink-0"
          >
            {useResumePicker ? t('field.browseProject') : t('field.browse')}
          </button>
        )}
        {suffix}
      </div>
      {help && <div className="text-xs text-fg-tertiary mt-1">{help}</div>}
      {/* resume_state / resume_lora：贴字段的 dropdown，按 version 分组列文件 */}
      {useResumePicker && picking && !disabled && (
        <ResumeFieldPicker
          pid={projectCtx!.project.id}
          kind={resumeKind!}
          value={text}
          onChange={onChange as (v: string) => void}
          onClose={() => setPicking(false)}
          anchorRef={browseBtnRef}
        />
      )}
      {/* 其它 path 字段：保留 PathPicker 模态框（外部模型路径等场景） */}
      {!useResumePicker && picking && !disabled && (
        <PathPicker
          initialPath={text || undefined}
          onPick={(p) => {
            onChange(p)
            setPicking(false)
          }}
          onClose={() => setPicking(false)}
        />
      )}
    </div>
  )
}
