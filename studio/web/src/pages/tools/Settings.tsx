import { useEffect, useMemo, useState } from 'react'
import {
  api,
  DEFAULT_WD14_MODELS,
  type Secrets,
  type SecretsPatch,
} from '../../api/client'
import { useToast } from '../../components/Toast'

const MASK = '***'

type Section =
  | 'gelbooru'
  | 'danbooru'
  | 'download'
  | 'huggingface'
  | 'joycaption'
  | 'wd14'

const EMPTY: Secrets = {
  gelbooru: {
    user_id: '',
    api_key: '',
    save_tags: false,
    convert_to_png: true,
    remove_alpha_channel: false,
  },
  danbooru: { username: '', api_key: '', account_type: 'free' },
  download: { exclude_tags: [] },
  huggingface: { token: '' },
  joycaption: {
    base_url: 'http://localhost:8000/v1',
    model: 'fancyfeast/llama-joycaption-beta-one-hf-llava',
    prompt_template: 'Descriptive Caption',
  },
  wd14: {
    model_id: 'SmilingWolf/wd-eva02-large-tagger-v3',
    model_ids: [...DEFAULT_WD14_MODELS],
    local_dir: null,
    threshold_general: 0.35,
    threshold_character: 0.85,
    blacklist_tags: [],
  },
}

export default function SettingsPage() {
  const [server, setServer] = useState<Secrets | null>(null)
  const [draft, setDraft] = useState<Secrets>(EMPTY)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const { toast } = useToast()

  useEffect(() => {
    api
      .getSecrets()
      .then((s) => {
        setServer(s)
        setDraft(s)
      })
      .catch((e) => setError(String(e)))
  }, [])

  const dirty = useMemo(
    () => server !== null && JSON.stringify(server) !== JSON.stringify(draft),
    [server, draft]
  )

  const update = <S extends Section, K extends keyof Secrets[S]>(
    section: S,
    key: K,
    value: Secrets[S][K]
  ) => {
    setDraft((prev) => ({
      ...prev,
      [section]: { ...prev[section], [key]: value },
    }))
  }

  const save = async () => {
    if (!server) return
    // 把和 server 相同的 leaf 全部省掉，把 MASK 占位的字段也省掉（不变）
    const patch = buildPatch(draft, server)
    setSaving(true)
    setError(null)
    try {
      const next = await api.updateSecrets(patch)
      setServer(next)
      setDraft(next)
      toast('已保存', 'success')
    } catch (e) {
      setError(String(e))
      toast('保存失败', 'error')
    } finally {
      setSaving(false)
    }
  }

  if (error && !server) {
    return (
      <div className="text-red-300 font-mono text-sm p-4 bg-red-900/20 rounded">
        {error}
      </div>
    )
  }

  return (
    <div className="max-w-3xl space-y-8 pb-12">
      <header className="flex items-center gap-3">
        <h1 className="text-2xl font-semibold flex-1">设置</h1>
        <button
          onClick={save}
          disabled={!dirty || saving}
          className="px-3 py-1.5 rounded text-sm bg-cyan-600 hover:bg-cyan-500
            disabled:bg-slate-700 disabled:text-slate-500"
        >
          {saving ? '保存中...' : '保存'}
        </button>
      </header>

      {error && (
        <div className="p-3 rounded bg-red-900/40 border border-red-700 text-red-300 text-sm font-mono">
          {error}
        </div>
      )}

      <Section title="Gelbooru">
        <Field label="user_id">
          <input
            type="text"
            value={draft.gelbooru.user_id}
            onChange={(e) => update('gelbooru', 'user_id', e.target.value)}
            className={textInput}
          />
        </Field>
        <Field label="api_key">
          <SensitiveInput
            value={draft.gelbooru.api_key}
            serverValue={server?.gelbooru.api_key ?? ''}
            onChange={(v) => update('gelbooru', 'api_key', v)}
          />
        </Field>
        <Field label="save_tags">
          <Bool
            value={draft.gelbooru.save_tags}
            onChange={(v) => update('gelbooru', 'save_tags', v)}
          />
        </Field>
        <Field label="convert_to_png">
          <Bool
            value={draft.gelbooru.convert_to_png}
            onChange={(v) => update('gelbooru', 'convert_to_png', v)}
          />
        </Field>
        <Field label="remove_alpha_channel">
          <Bool
            value={draft.gelbooru.remove_alpha_channel}
            onChange={(v) => update('gelbooru', 'remove_alpha_channel', v)}
          />
        </Field>
      </Section>

      <Section title="Danbooru">
        <Field label="username">
          <input
            type="text"
            value={draft.danbooru.username}
            onChange={(e) => update('danbooru', 'username', e.target.value)}
            placeholder="可选；匿名也能跑（仅速率受限）"
            className={textInput}
          />
        </Field>
        <Field label="api_key">
          <SensitiveInput
            value={draft.danbooru.api_key}
            serverValue={server?.danbooru.api_key ?? ''}
            onChange={(v) => update('danbooru', 'api_key', v)}
          />
        </Field>
        <Field label="account_type">
          <select
            value={draft.danbooru.account_type}
            onChange={(e) =>
              update(
                'danbooru',
                'account_type',
                e.target.value as 'free' | 'gold' | 'platinum'
              )
            }
            className={textInput}
          >
            <option value="free">free（max 2 tag）</option>
            <option value="gold">gold（max 6 tag）</option>
            <option value="platinum">platinum（max 12 tag）</option>
          </select>
        </Field>
      </Section>

      <Section title="下载（全局）">
        <Field label="exclude_tags (逗号分隔)">
          <input
            type="text"
            value={draft.download.exclude_tags.join(', ')}
            onChange={(e) =>
              update(
                'download',
                'exclude_tags',
                e.target.value
                  .split(',')
                  .map((t) => t.trim().replace(/^-+/, ''))
                  .filter(Boolean)
              )
            }
            placeholder="例：comic, monochrome, lowres"
            className={textInput}
          />
        </Field>
        <p className="text-[11px] text-slate-500 px-1">
          搜索时自动追加 <code>-tag</code>，对 Gelbooru 与 Danbooru 同样生效。
        </p>
      </Section>

      <Section title="HuggingFace">
        <Field label="token">
          <SensitiveInput
            value={draft.huggingface.token}
            serverValue={server?.huggingface.token ?? ''}
            onChange={(v) => update('huggingface', 'token', v)}
          />
        </Field>
      </Section>

      <Section title="JoyCaption (vLLM)">
        <Field label="base_url">
          <input
            type="text"
            value={draft.joycaption.base_url}
            onChange={(e) => update('joycaption', 'base_url', e.target.value)}
            className={textInput}
          />
        </Field>
        <Field label="model">
          <input
            type="text"
            value={draft.joycaption.model}
            onChange={(e) => update('joycaption', 'model', e.target.value)}
            className={textInput}
          />
        </Field>
        <Field label="prompt_template">
          <input
            type="text"
            value={draft.joycaption.prompt_template}
            onChange={(e) =>
              update('joycaption', 'prompt_template', e.target.value)
            }
            className={textInput}
          />
        </Field>
      </Section>

      <Section title="WD14">
        <Field label="model_id (当前选用)">
          <select
            value={draft.wd14.model_id}
            onChange={(e) => update('wd14', 'model_id', e.target.value)}
            className={textInput}
          >
            {/* 候选列表渲染。后端 validator 会保证 model_id ∈ model_ids，
             * 所以这里 dropdown 一定能命中当前值。 */}
            {(draft.wd14.model_ids.length > 0
              ? draft.wd14.model_ids
              : [...DEFAULT_WD14_MODELS]
            ).map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </Field>
        <Field label="候选模型 (model_ids)">
          <ModelIdsEditor
            ids={draft.wd14.model_ids}
            currentId={draft.wd14.model_id}
            onChange={(next) => update('wd14', 'model_ids', next)}
          />
        </Field>
        <Field label="local_dir (留空 = 自动 HF 下载)">
          <input
            type="text"
            value={draft.wd14.local_dir ?? ''}
            onChange={(e) =>
              update('wd14', 'local_dir', e.target.value || null)
            }
            className={textInput}
          />
        </Field>
        <Field label="threshold_general">
          <input
            type="number"
            step="0.01"
            min={0}
            max={1}
            value={draft.wd14.threshold_general}
            onChange={(e) =>
              update('wd14', 'threshold_general', Number(e.target.value))
            }
            className={textInput}
          />
        </Field>
        <Field label="threshold_character">
          <input
            type="number"
            step="0.01"
            min={0}
            max={1}
            value={draft.wd14.threshold_character}
            onChange={(e) =>
              update('wd14', 'threshold_character', Number(e.target.value))
            }
            className={textInput}
          />
        </Field>
        <Field label="blacklist_tags (逗号分隔)">
          <input
            type="text"
            value={draft.wd14.blacklist_tags.join(', ')}
            onChange={(e) =>
              update(
                'wd14',
                'blacklist_tags',
                e.target.value
                  .split(',')
                  .map((t) => t.trim())
                  .filter(Boolean)
              )
            }
            className={textInput}
          />
        </Field>
      </Section>
    </div>
  )
}

const textInput =
  'w-full px-2 py-1.5 rounded bg-slate-900 border border-slate-700 text-sm text-slate-200 focus:outline-none focus:border-cyan-500'

function Section({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}) {
  return (
    <section className="border border-slate-700 rounded-lg bg-slate-800/40 p-4 space-y-3">
      <h2 className="text-sm font-semibold text-slate-200 mb-1">{title}</h2>
      {children}
    </section>
  )
}

function Field({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="grid grid-cols-[180px_1fr] gap-3 items-center">
      <label className="text-xs text-slate-400 font-mono">{label}</label>
      {children}
    </div>
  )
}

function Bool({
  value,
  onChange,
}: {
  value: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <input
      type="checkbox"
      checked={value}
      onChange={(e) => onChange(e.target.checked)}
      className="h-4 w-4 accent-cyan-500"
    />
  )
}

function SensitiveInput({
  value,
  serverValue,
  onChange,
}: {
  value: string
  serverValue: string
  onChange: (v: string) => void
}) {
  // 当 server 已有值时，input 显示空 + placeholder「已保存（不显示）」
  // 用户输入任何内容才覆盖。把 MASK 当作「保持不变」的哨兵传回去。
  const masked = value === MASK
  return (
    <input
      type="password"
      value={masked ? '' : value}
      placeholder={
        serverValue === MASK ? '已保存（不显示），输入新值才覆盖' : ''
      }
      onChange={(e) => onChange(e.target.value || MASK)}
      className={textInput}
    />
  )
}

function ModelIdsEditor({
  ids,
  currentId,
  onChange,
}: {
  ids: string[]
  currentId: string
  onChange: (next: string[]) => void
}) {
  const [draft, setDraft] = useState('')
  const seen = new Set(ids)

  const add = () => {
    const v = draft.trim()
    if (!v) return
    if (seen.has(v)) {
      setDraft('')
      return
    }
    onChange([...ids, v])
    setDraft('')
  }
  const remove = (m: string) => {
    if (m === currentId) return // 后端 validator 会兜底，但前端先拦一道，提示更清晰
    onChange(ids.filter((x) => x !== m))
  }

  return (
    <div className="space-y-1.5">
      <ul className="space-y-1">
        {ids.map((m) => {
          const isCurrent = m === currentId
          return (
            <li
              key={m}
              className={
                'flex items-center gap-2 px-2 py-1 rounded border text-xs ' +
                (isCurrent
                  ? 'border-cyan-700 bg-cyan-950/30'
                  : 'border-slate-700 bg-slate-900/40')
              }
            >
              <code className="font-mono text-slate-200 truncate flex-1 min-w-0">
                {m}
              </code>
              {isCurrent ? (
                <span className="text-[10px] text-cyan-300">当前</span>
              ) : (
                <button
                  onClick={() => remove(m)}
                  className="text-[11px] text-slate-500 hover:text-red-400 px-1"
                  title="删除"
                >
                  ×
                </button>
              )}
            </li>
          )
        })}
      </ul>
      <div className="flex gap-1.5">
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              add()
            }
          }}
          placeholder="添加 HuggingFace 模型 ID，如 SmilingWolf/wd-vit-tagger-v3"
          className={textInput + ' flex-1'}
        />
        <button
          onClick={add}
          disabled={!draft.trim() || seen.has(draft.trim())}
          className="px-2.5 py-1 rounded bg-slate-700 hover:bg-slate-600 text-xs text-slate-200 disabled:bg-slate-800 disabled:text-slate-500"
        >
          + 添加
        </button>
      </div>
      <p className="text-[10px] text-slate-500">
        当前选用的模型不能在此删除，需先在上方下拉切到另一个再删。
      </p>
    </div>
  )
}

function buildPatch(draft: Secrets, server: Secrets): SecretsPatch {
  // 只保留与 server 不一致 的 leaf；MASK 占位则跳过（保持不变）。
  const out: Record<string, Record<string, unknown>> = {}
  for (const key of Object.keys(draft) as Section[]) {
    const sub: Record<string, unknown> = {}
    const d = draft[key] as unknown as Record<string, unknown>
    const s = server[key] as unknown as Record<string, unknown>
    for (const k of Object.keys(d)) {
      const dv = d[k]
      const sv = s[k]
      if (dv === MASK) continue
      if (JSON.stringify(dv) !== JSON.stringify(sv)) sub[k] = dv
    }
    if (Object.keys(sub).length) out[key] = sub
  }
  return out as SecretsPatch
}
