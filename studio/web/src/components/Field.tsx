import type { SchemaProperty } from '../api/client'
import { controlKind, fieldLabel } from '../lib/schema'

interface Props {
  name: string
  prop: SchemaProperty
  value: unknown
  onChange: (v: unknown) => void
}

const labelCls = 'text-sm font-medium text-slate-300 mb-1'
const helpCls = 'text-xs text-slate-500 mt-1'
const inputCls =
  'w-full px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-md ' +
  'focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 ' +
  'text-sm'

/** 单个表单字段，按 control kind 分发渲染。 */
export default function Field({ name, prop, value, onChange }: Props) {
  const kind = controlKind(prop)
  const label = fieldLabel(name)
  const help = prop.description

  // bool ----------------------------------------------------------------
  if (kind === 'bool') {
    return (
      <label className="flex items-start gap-3 py-1.5 cursor-pointer">
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
          className="mt-1 h-4 w-4 rounded border-slate-600 bg-slate-800
            text-cyan-500 focus:ring-cyan-500"
        />
        <span className="flex-1">
          <div className="text-sm text-slate-200">{label}</div>
          {help && <div className={helpCls}>{help}</div>}
        </span>
      </label>
    )
  }

  // select --------------------------------------------------------------
  if (kind === 'select') {
    return (
      <div className="py-1.5">
        <div className={labelCls}>{label}</div>
        <select
          value={String(value ?? '')}
          onChange={(e) => onChange(e.target.value)}
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
        <div className={labelCls}>{label}</div>
        <textarea
          rows={3}
          value={String(value ?? '')}
          onChange={(e) => onChange(e.target.value)}
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
        <div className={labelCls}>{label}（每行一项）</div>
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
        <div className={labelCls}>{label}</div>
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
          className={inputCls}
        />
        {help && <div className={helpCls}>{help}</div>}
      </div>
    )
  }

  // string / path -------------------------------------------------------
  return (
    <div className="py-1.5">
      <div className={labelCls}>
        {label}
        {kind === 'path' && (
          <span className="ml-2 text-xs text-slate-500">(path)</span>
        )}
      </div>
      <input
        type="text"
        value={value === null || value === undefined ? '' : String(value)}
        onChange={(e) => onChange(e.target.value)}
        className={inputCls + (kind === 'path' ? ' font-mono' : '')}
      />
      {help && <div className={helpCls}>{help}</div>}
    </div>
  )
}
