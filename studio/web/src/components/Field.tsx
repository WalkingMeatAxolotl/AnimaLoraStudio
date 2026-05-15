import { useEffect, useRef, useState } from 'react'
import type { SchemaProperty } from '../api/client'
import { useProjectCtx } from '../context/ProjectContext'
import { controlKind, fieldLabel } from '../lib/schema'
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
   * 配合 disabled 来表达「这字段被自动填且不让你改」。 */
  hint?: string
  /** 覆盖 prop.description 的说明文字（用于条件上下文描述）。 */
  descriptionOverride?: string
}

// input 覆盖 .input 默认值（更紧凑；背景用 canvas 而不是 surface）
const inputStyle: React.CSSProperties = {
  width: '100%', padding: '5px 10px',
  background: 'var(--bg-canvas)', border: '1px solid var(--border-default)',
  borderRadius: 'var(--r-sm)', fontSize: 'var(--t-sm)',
  color: 'var(--fg-primary)',
}

const FieldHint = ({ text }: { text: string }) => (
  <span className="ml-2 text-[11px] text-warn align-middle">{text}</span>
)

/** 单个表单字段，按 control kind 分发渲染。 */
export default function Field({
  name, prop, value, onChange, disabled = false, hint, descriptionOverride,
}: Props) {
  const kind = controlKind(prop)
  const label = fieldLabel(name)
  const help = descriptionOverride ?? prop.description
  const hintText = hint ?? (disabled ? '自动 · 项目控制' : null)
  const hintNode = hintText ? <FieldHint text={hintText} /> : null
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
              {String(opt)}
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
          {label}（每行一项）{hintNode}
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
    />
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
}

function PathStringField({
  name, label, kind, help, value, onChange, disabled = false, hintNode,
}: PathFieldProps) {
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
          <span className="ml-2 text-xs text-fg-tertiary">(path)</span>
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
            {useResumePicker ? '📁 浏览本项目' : '浏览'}
          </button>
        )}
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
