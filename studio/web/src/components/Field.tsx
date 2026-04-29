import { useState } from 'react'
import type { SchemaProperty } from '../api/client'
import { controlKind, fieldLabel } from '../lib/schema'
import PathPicker from './PathPicker'

interface Props {
  name: string
  prop: SchemaProperty
  value: unknown
  onChange: (v: unknown) => void
  /** disabled 状态（自动控制字段灰显 readonly）。 */
  disabled?: boolean
  /** 自定义 disabled 徽章文字；不传则用默认「自动 · 项目控制」。 */
  disabledHint?: string
}

const labelCls = 'text-sm font-medium text-slate-300 mb-1'
const helpCls = 'text-xs text-slate-500 mt-1'
const inputCls =
  'w-full px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-md ' +
  'focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 ' +
  'text-sm disabled:opacity-50 disabled:cursor-not-allowed'

const renderHint = (text: string) => (
  <span className="ml-2 text-[10px] text-amber-400/80 align-middle">
    {text}
  </span>
)

/** 单个表单字段，按 control kind 分发渲染。 */
export default function Field({
  name, prop, value, onChange, disabled = false, disabledHint,
}: Props) {
  const kind = controlKind(prop)
  const label = fieldLabel(name)
  const help = prop.description
  const hintNode = disabled
    ? renderHint(disabledHint ?? '自动 · 项目控制')
    : null
  void name

  // bool ----------------------------------------------------------------
  if (kind === 'bool') {
    return (
      <label
        className={
          'flex items-start gap-3 py-1.5 ' +
          (disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer')
        }
      >
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
          disabled={disabled}
          className="mt-1 h-4 w-4 rounded border-slate-600 bg-slate-800
            text-cyan-500 focus:ring-cyan-500 disabled:opacity-50"
        />
        <span className="flex-1">
          <div className="text-sm text-slate-200">
            {label}
            {hintNode}
          </div>
          {help && <div className={helpCls}>{help}</div>}
        </span>
      </label>
    )
  }

  // select --------------------------------------------------------------
  if (kind === 'select') {
    return (
      <div className="py-1.5">
        <div className={labelCls}>
          {label}
          {hintNode}
        </div>
        <select
          value={String(value ?? '')}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className={inputCls}
        >
          {(prop.enum ?? []).map((opt) => (
            <option key={String(opt)} value={String(opt)}>
              {String(opt)}
            </option>
          ))}
        </select>
        {help && <div className={helpCls}>{help}</div>}
      </div>
    )
  }

  // textarea ------------------------------------------------------------
  if (kind === 'textarea') {
    return (
      <div className="py-1.5">
        <div className={labelCls}>
          {label}
          {hintNode}
        </div>
        <textarea
          rows={3}
          value={String(value ?? '')}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className={inputCls + ' font-mono'}
        />
        {help && <div className={helpCls}>{help}</div>}
      </div>
    )
  }

  // string-list ---------------------------------------------------------
  if (kind === 'string-list') {
    const list = Array.isArray(value) ? (value as string[]) : []
    const text = list.join('\n')
    return (
      <div className="py-1.5">
        <div className={labelCls}>
          {label}（每行一项）
          {hintNode}
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
          className={inputCls + ' font-mono'}
        />
        {help && <div className={helpCls}>{help}</div>}
      </div>
    )
  }

  // int / float ---------------------------------------------------------
  if (kind === 'int' || kind === 'float') {
    return (
      <div className="py-1.5">
        <div className={labelCls}>
          {label}
          {hintNode}
        </div>
        <input
          type="number"
          step={kind === 'int' ? 1 : 'any'}
          value={value === null || value === undefined ? '' : String(value)}
          min={prop.minimum}
          max={prop.maximum}
          onChange={(e) => {
            const raw = e.target.value
            if (raw === '') {
              onChange(prop.default)
              return
            }
            const num = kind === 'int' ? parseInt(raw, 10) : parseFloat(raw)
            if (!Number.isNaN(num)) onChange(num)
          }}
          disabled={disabled}
          className={inputCls}
        />
        {help && <div className={helpCls}>{help}</div>}
      </div>
    )
  }

  // string / path -------------------------------------------------------
  return (
    <PathStringField
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

interface PathFieldProps {
  label: string
  kind: 'path' | 'string'
  help: string | undefined
  value: unknown
  onChange: (v: unknown) => void
  disabled?: boolean
  hintNode?: React.ReactNode
}

function PathStringField({
  label, kind, help, value, onChange, disabled = false, hintNode,
}: PathFieldProps) {
  const [picking, setPicking] = useState(false)
  const text = value === null || value === undefined ? '' : String(value)
  return (
    <div className="py-1.5">
      <div className={labelCls}>
        {label}
        {kind === 'path' && (
          <span className="ml-2 text-xs text-slate-500">(path)</span>
        )}
        {hintNode}
      </div>
      <div className="flex gap-2">
        <input
          type="text"
          value={text}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className={inputCls + (kind === 'path' ? ' font-mono' : '')}
        />
        {kind === 'path' && (
          <button
            type="button"
            onClick={() => setPicking(true)}
            disabled={disabled}
            className="px-2 py-1 text-xs rounded bg-slate-700 hover:bg-slate-600 shrink-0 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            浏览
          </button>
        )}
      </div>
      {help && <div className={helpCls}>{help}</div>}
      {picking && !disabled && (
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
