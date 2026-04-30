import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  api,
  DEFAULT_WD14_MODELS,
  type ModelDownloadStatus,
  type ModelsCatalog,
  type Secrets,
  type SecretsPatch,
  type WD14Runtime,
} from '../../api/client'
import { useToast } from '../../components/Toast'
import { useEventStream } from '../../lib/useEventStream'

const MASK = '***'

type Section =
  | 'gelbooru'
  | 'danbooru'
  | 'download'
  | 'huggingface'
  | 'joycaption'
  | 'wd14'
  | 'models'

const EMPTY: Secrets = {
  gelbooru: {
    user_id: '',
    api_key: '',
    save_tags: false,
    convert_to_png: true,
    remove_alpha_channel: false,
  },
  danbooru: { username: '', api_key: '', account_type: 'free' },
  download: {
    exclude_tags: [],
    parallel_workers: 4,
    api_rate_per_sec: 2,
    cdn_rate_per_sec: 5,
  },
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
    batch_size: 8,
  },
  models: { root: null, selected_anima: 'preview3-base' },
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

        <div className="grid grid-cols-3 gap-2 pt-2 border-t border-slate-800">
          <Field label="parallel_workers">
            <input
              type="number"
              min={1}
              max={16}
              value={draft.download.parallel_workers}
              onChange={(e) =>
                update('download', 'parallel_workers', Math.max(1, Number(e.target.value) || 1))
              }
              className={textInput}
            />
          </Field>
          <Field label="api_rate_per_sec">
            <input
              type="number"
              step="0.5"
              min={0.5}
              max={10}
              value={draft.download.api_rate_per_sec}
              onChange={(e) =>
                update('download', 'api_rate_per_sec', Math.max(0.5, Number(e.target.value) || 0.5))
              }
              className={textInput}
            />
          </Field>
          <Field label="cdn_rate_per_sec">
            <input
              type="number"
              step="1"
              min={1}
              max={20}
              value={draft.download.cdn_rate_per_sec}
              onChange={(e) =>
                update('download', 'cdn_rate_per_sec', Math.max(1, Number(e.target.value) || 1))
              }
              className={textInput}
            />
          </Field>
        </div>
        <p className="text-[11px] text-slate-500 px-1 leading-relaxed">
          API host (<code>gelbooru.com</code> / <code>danbooru.donmai.us</code>) 与 CDN host (
          <code>img*.gelbooru.com</code> / <code>cdn.donmai.us</code>) 各自独立 token bucket。
          带 api_key 时 gelbooru 实测 5 req/s 拉图很安全；遇到 429 会自动退避 60s + 速率减半。
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
        <Field label="batch_size (GPU 推理一批塞几张；CPU 自动降到 1)">
          <input
            type="number"
            min={1}
            max={64}
            value={draft.wd14.batch_size}
            onChange={(e) =>
              update('wd14', 'batch_size', Math.max(1, Number(e.target.value) || 1))
            }
            className={textInput}
          />
        </Field>
        <WD14RuntimePanel />
      </Section>

      <ModelsSection />
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

// ---------------------------------------------------------------------------
// Models 区块 — Anima 主模型 / VAE / Qwen3 / T5 tokenizer 一键下载（PP7）
// ---------------------------------------------------------------------------

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}

function ModelsSection() {
  const { toast } = useToast()
  const [catalog, setCatalog] = useState<ModelsCatalog | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<Set<string>>(new Set())
  const [rootDraft, setRootDraft] = useState<string>('')
  const [serverRoot, setServerRoot] = useState<string | null>(null)
  const [savingRoot, setSavingRoot] = useState(false)
  const [selectedAnima, setSelectedAnima] = useState<string>('preview3-base')

  const reload = useCallback(async () => {
    try {
      const [c, sec] = await Promise.all([
        api.getModelsCatalog(),
        api.getSecrets(),
      ])
      setCatalog(c)
      const root = sec.models?.root ?? null
      setServerRoot(root)
      // 仅在用户没编辑时同步；用户已经在 input 里改东西时不覆盖
      setRootDraft((prev) => (prev === '' || prev === (serverRoot ?? '') ? root ?? '' : prev))
      setSelectedAnima(sec.models?.selected_anima ?? 'preview3-base')
      setError(null)
    } catch (e) {
      setError(String(e))
    }
    // serverRoot 故意不进 deps：只在「用户没编辑过」的情况同步
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const pickAnima = async (variant: string) => {
    if (variant === selectedAnima) return
    setSelectedAnima(variant)  // 乐观更新
    try {
      await api.updateSecrets({ models: { selected_anima: variant } })
      toast(`默认主模型已切到 ${variant}`, 'success')
      await reload()
    } catch (e) {
      toast(String(e), 'error')
      void reload()  // 回滚到 server 真实值
    }
  }

  useEffect(() => {
    void reload()
  }, [reload])

  const saveRoot = async () => {
    const v = rootDraft.trim()
    setSavingRoot(true)
    try {
      await api.updateSecrets({
        models: { root: v ? v : null },
      })
      toast(v ? `已保存模型根目录: ${v}` : '已恢复默认模型根目录', 'success')
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setSavingRoot(false)
    }
  }

  const rootDirty = rootDraft.trim() !== (serverRoot ?? '')

  // SSE：下载完成 / 失败时刷新 catalog（拿新文件大小）
  useEventStream((evt) => {
    if (evt.type === 'model_download_changed') {
      void reload()
    }
  })

  const start = async (model_id: string, variant?: string) => {
    const key = variant ? `${model_id}:${variant}` : model_id
    setBusy((s) => new Set(s).add(key))
    try {
      await api.startModelDownload({ model_id, variant })
      toast(`开始下载 ${key}`, 'success')
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy((s) => {
        const n = new Set(s)
        n.delete(key)
        return n
      })
    }
  }

  return (
    <Section title="Models（一键下载训练所需模型）">
      <p className="text-[11px] text-slate-500 px-1">
        默认走 hf-mirror.com 镜像（可在 server 启动前设 <code>HF_ENDPOINT</code>{' '}
        覆盖）。新版本发布时改{' '}
        <code>studio/services/model_downloader.py</code> 两行常量。
      </p>

      {/* 模型根目录配置（PP7） */}
      <Field label="模型根目录 (models_root)">
        <div className="flex items-center gap-1.5">
          <input
            type="text"
            value={rootDraft}
            onChange={(e) => setRootDraft(e.target.value)}
            placeholder="留空 = 默认 REPO_ROOT/anima/，云端可填如 /data/anima"
            className={textInput + ' flex-1'}
          />
          <button
            onClick={saveRoot}
            disabled={!rootDirty || savingRoot}
            className="px-2.5 py-1 rounded text-xs bg-cyan-600 hover:bg-cyan-500 text-white disabled:bg-slate-700 disabled:text-slate-500"
            title={rootDirty ? '保存路径配置' : '未修改'}
          >
            {savingRoot ? '保存中...' : '保存路径'}
          </button>
          <button
            onClick={() => setRootDraft(serverRoot ?? '')}
            disabled={!rootDirty || savingRoot}
            className="px-2 py-1 rounded text-xs text-slate-400 hover:text-slate-200 disabled:opacity-30"
            title="还原成 server 当前值"
          >
            ↻
          </button>
        </div>
      </Field>
      <p className="text-[10px] text-slate-500 px-1 -mt-1">
        当前生效路径：
        <code className="text-slate-300">
          {catalog?.models_root ?? '(loading...)'}
        </code>
        {serverRoot && rootDraft.trim() !== serverRoot && (
          <span className="ml-2 text-amber-300">
            ⚠ 输入框未保存（路径修改不会同步到上方主「保存」按钮，需点旁边的独立「保存」）
          </span>
        )}
      </p>
      {error && (
        <div className="text-red-300 text-xs font-mono">{error}</div>
      )}
      {!catalog ? (
        <p className="text-slate-500 text-xs">加载...</p>
      ) : (
        <div className="space-y-2">
          {/* Anima 主模型（多版本 + radio 选默认） */}
          <ModelGroupCard title={catalog.anima_main.name}>
            <p className="text-[11px] text-slate-500">
              {catalog.anima_main.description} ·{' '}
              <code>{catalog.anima_main.repo}</code>
              <br />
              选中的版本会作为<strong className="text-slate-300">新建 version</strong>
              的默认 transformer（写入 yaml 时已展开为绝对路径，已存在 version 不会被改动）。
            </p>
            <ul className="space-y-1 mt-1">
              {catalog.anima_main.variants.map((v) => {
                const key = `anima_main:${v.variant}`
                const dl = catalog.downloads[key]
                const isSel = v.variant === selectedAnima
                const canSelect = v.exists && dl?.status !== 'running'
                return (
                  <li
                    key={v.variant}
                    className={
                      'flex items-center gap-2 text-xs px-1.5 py-1 rounded ' +
                      (isSel ? 'bg-cyan-950/40 border border-cyan-800' : '')
                    }
                  >
                    <input
                      type="radio"
                      name="anima_variant"
                      checked={isSel}
                      disabled={!canSelect}
                      onChange={() => void pickAnima(v.variant)}
                      className="accent-cyan-500 shrink-0"
                      title={
                        canSelect
                          ? '选作默认主模型'
                          : v.exists
                          ? '下载中...'
                          : '未下载，请先下载'
                      }
                    />
                    <code className="font-mono text-slate-200 w-32 shrink-0">
                      {v.variant}
                      {v.is_latest && (
                        <span className="ml-1 text-[9px] text-cyan-300">
                          latest
                        </span>
                      )}
                    </code>
                    <ModelStatusBadge
                      exists={v.exists}
                      size={v.size}
                      status={dl?.status}
                    />
                    <span className="flex-1" />
                    <DownloadButton
                      exists={v.exists}
                      status={dl?.status}
                      busy={busy.has(key)}
                      onClick={() =>
                        void start('anima_main', v.variant)
                      }
                    />
                  </li>
                )
              })}
            </ul>
          </ModelGroupCard>

          {/* VAE */}
          <ModelGroupCard title={catalog.anima_vae.name}>
            <div className="flex items-center gap-2 text-xs">
              <span className="text-slate-500">
                {catalog.anima_vae.description} ·{' '}
                <code>{catalog.anima_vae.repo}</code>
              </span>
              <span className="flex-1" />
              <ModelStatusBadge
                exists={catalog.anima_vae.exists}
                size={catalog.anima_vae.size}
                status={catalog.downloads.anima_vae?.status}
              />
              <DownloadButton
                exists={catalog.anima_vae.exists}
                status={catalog.downloads.anima_vae?.status}
                busy={busy.has('anima_vae')}
                onClick={() => void start('anima_vae')}
              />
            </div>
          </ModelGroupCard>

          {/* Qwen3 + T5 共用渲染 */}
          {(['qwen3', 't5_tokenizer'] as const).map((id) => {
            const m = catalog[id]
            const dl = catalog.downloads[id]
            const allExist = m.files.every((f) => f.exists)
            const totalSize = m.files.reduce((s, f) => s + f.size, 0)
            return (
              <ModelGroupCard key={id} title={m.name}>
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-slate-500">
                    {m.description} · <code>{m.repo}</code>
                  </span>
                  <span className="flex-1" />
                  <ModelStatusBadge
                    exists={allExist}
                    size={totalSize}
                    status={dl?.status}
                    fileCount={m.files.length}
                    existsCount={m.files.filter((f) => f.exists).length}
                  />
                  <DownloadButton
                    exists={allExist}
                    status={dl?.status}
                    busy={busy.has(id)}
                    onClick={() => void start(id)}
                  />
                </div>
              </ModelGroupCard>
            )
          })}

          {/* 当前活跃下载 log_tail（紧凑） */}
          {Object.values(catalog.downloads).filter(
            (d) => d.status === 'running' || d.status === 'failed'
          ).length > 0 && (
            <details className="text-xs mt-2">
              <summary className="cursor-pointer text-slate-400 hover:text-slate-200">
                下载日志 (
                {
                  Object.values(catalog.downloads).filter(
                    (d) => d.status === 'running' || d.status === 'failed'
                  ).length
                }
                )
              </summary>
              <div className="mt-1 space-y-2">
                {Object.values(catalog.downloads).map((d) => (
                  <div
                    key={d.key}
                    className="rounded border border-slate-700 bg-slate-950/40 p-2"
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <code className="font-mono text-slate-300">{d.key}</code>
                      <span
                        className={
                          'text-[10px] px-1.5 py-0.5 rounded ' +
                          (d.status === 'running'
                            ? 'bg-amber-700/50 text-amber-200 animate-pulse'
                            : d.status === 'done'
                            ? 'bg-emerald-700/40 text-emerald-200'
                            : d.status === 'failed'
                            ? 'bg-red-700/50 text-red-200'
                            : 'bg-slate-700/40 text-slate-300')
                        }
                      >
                        {d.status}
                      </span>
                      {d.message && (
                        <span className="text-red-300 truncate">
                          {d.message}
                        </span>
                      )}
                    </div>
                    <pre className="text-[10px] font-mono text-slate-400 max-h-32 overflow-y-auto whitespace-pre-wrap">
                      {d.log_tail.join('\n') || '(等待日志...)'}
                    </pre>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
    </Section>
  )
}

function ModelGroupCard({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}) {
  return (
    <div className="rounded border border-slate-700 bg-slate-900/40 p-2.5">
      <h4 className="text-xs font-semibold text-slate-200 mb-1">{title}</h4>
      {children}
    </div>
  )
}

function ModelStatusBadge({
  exists,
  size,
  status,
  fileCount,
  existsCount,
}: {
  exists: boolean
  size: number
  status?: ModelDownloadStatus['status']
  fileCount?: number
  existsCount?: number
}) {
  if (status === 'running') {
    return (
      <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-700/40 text-amber-200 animate-pulse">
        下载中...
      </span>
    )
  }
  if (status === 'failed') {
    return (
      <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-700/40 text-red-200">
        失败
      </span>
    )
  }
  if (exists) {
    return (
      <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-700/40 text-emerald-200 font-mono">
        ✓ {fmtBytes(size)}
        {fileCount !== undefined && ` (${existsCount}/${fileCount})`}
      </span>
    )
  }
  if (fileCount !== undefined && existsCount! > 0) {
    return (
      <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-700/40 text-amber-200 font-mono">
        部分 ({existsCount}/{fileCount})
      </span>
    )
  }
  return (
    <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700/60 text-slate-400">
      未下载
    </span>
  )
}

function DownloadButton({
  exists,
  status,
  busy,
  onClick,
}: {
  exists: boolean
  status?: ModelDownloadStatus['status']
  busy: boolean
  onClick: () => void
}) {
  const running = status === 'running' || busy
  if (running) {
    return (
      <button
        disabled
        className="text-xs px-2 py-1 rounded bg-slate-700 text-slate-400"
      >
        ...
      </button>
    )
  }
  return (
    <button
      onClick={onClick}
      className={
        'text-xs px-2 py-1 rounded ' +
        (exists
          ? 'bg-slate-700 hover:bg-slate-600 text-slate-300'
          : 'bg-cyan-600 hover:bg-cyan-500 text-white')
      }
      title={exists ? '已下载，点击重新下载（会跳过已存在文件）' : '下载'}
    >
      {exists ? '↻ 重下' : '⤓ 下载'}
    </button>
  )
}


// ---------------------------------------------------------------------------
// PP8 — onnxruntime 运行时面板（CUDA / CPU 显示 + 一键切换）
// ---------------------------------------------------------------------------

function WD14RuntimePanel() {
  const [rt, setRt] = useState<WD14Runtime | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<null | 'auto' | 'gpu' | 'cpu'>(null)
  const { toast } = useToast()

  const refresh = useCallback(async () => {
    try {
      const r = await api.getWD14Runtime()
      setRt(r)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const install = async (target: 'auto' | 'gpu' | 'cpu') => {
    const detail =
      target === 'auto'
        ? '将按 nvidia-smi 检测自动选 GPU/CPU 包'
        : target === 'gpu'
        ? '将卸载现有 onnxruntime 并安装 onnxruntime-gpu'
        : '将卸载现有 onnxruntime-gpu 并安装 onnxruntime（CPU）'
    if (
      !confirm(
        `${detail}。装包需要几分钟。\n\n⚠️ 注意：装完后必须**重启 Studio** 才能生效（onnxruntime 是 C 扩展，进程内不能热替换）。继续？`
      )
    )
      return
    setBusy(target)
    try {
      const result = await api.installWD14Runtime(target)
      setRt({
        installed: result.installed,
        version: result.version,
        providers: result.providers,
        cuda_available: result.cuda_available,
        restart_required: result.restart_required,
        cuda_load_error: result.cuda_load_error,
        preload: result.preload,
        cuda_detect: result.cuda_detect,
      })
      const newPkg = result.installed_pkg ?? result.installed ?? '?'
      const newVer = result.installed_version ?? result.version ?? '?'
      toast(`已装 ${newPkg}==${newVer}，请重启 Studio 让 EP 生效`, 'success')
    } catch (e) {
      toast(`装包失败: ${e}`, 'error')
    } finally {
      setBusy(null)
    }
  }

  if (error) {
    return (
      <div className="text-xs text-red-300 font-mono">{error}</div>
    )
  }
  if (!rt) {
    return <div className="text-xs text-slate-500">加载 runtime 状态...</div>
  }

  const epLabel = (rt.providers ?? [])
    .map((p) => p.replace('ExecutionProvider', ''))
    .join(' / ') || '(none)'
  const cuda = rt.cuda_detect ?? { available: false, driver_version: null, gpu_name: null }
  const cudaInfo = cuda.available
    ? `${cuda.gpu_name ?? '?'} (driver ${cuda.driver_version ?? '?'})`
    : '未检测到 NVIDIA GPU'
  const mismatched = cuda.available && !rt.cuda_available
  const restartRequired = !!rt.restart_required

  return (
    <div className="rounded border border-slate-700 bg-slate-950/40 p-2 space-y-1.5 text-xs">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-slate-500 shrink-0">runtime:</span>
        <code className="font-mono text-slate-200">
          {rt.installed ?? '(未安装)'}
          {rt.version ? `==${rt.version}` : ''}
        </code>
        <span
          className={
            'text-[10px] px-1.5 py-0.5 rounded font-mono shrink-0 ' +
            (rt.cuda_available
              ? 'bg-emerald-700/40 text-emerald-200'
              : 'bg-amber-700/40 text-amber-200')
          }
        >
          {rt.cuda_available ? 'CUDA' : 'CPU only'}
        </span>
      </div>
      <div className="text-slate-500">EP: <code className="text-slate-300 font-mono">{epLabel}</code></div>
      <div className="text-slate-500">GPU 检测: <span className="text-slate-300">{cudaInfo}</span></div>
      {restartRequired && (
        <div className="rounded border border-red-700/60 bg-red-900/30 px-2 py-1.5 text-red-200 text-[11px] leading-relaxed">
          🔁 已装新 onnxruntime 包，但当前进程仍在用旧的（C 扩展不能热替换）。
          <strong className="text-red-100">请重启 Studio</strong> 让 EP 切换生效。
        </div>
      )}
      {!restartRequired && mismatched && (
        <div className="text-amber-300 text-[11px] leading-relaxed">
          ⚠️ 检测到 NVIDIA GPU 但 onnxruntime 只有 CPU EP — WD14 会跑得很慢。点下方「重装为 GPU 版」修复。
        </div>
      )}
      {rt.cuda_load_error && (
        <div className="rounded border border-red-700/60 bg-red-900/30 px-2 py-1.5 text-red-200 text-[11px] leading-relaxed space-y-1">
          <div>
            ⚠️ CUDA EP 加载失败，本次打标已降级到 CPU。常见原因：缺系统 CUDA runtime
            （onnxruntime-gpu wheel 不带）。
          </div>
          <code className="block font-mono text-[10px] text-red-100/90 break-all whitespace-pre-wrap">
            {rt.cuda_load_error}
          </code>
          {rt.preload && rt.preload.applied && rt.preload.candidates === 0 && (
            <div className="text-[10px] text-red-100/80">
              提示：venv 里没装 torch CUDA wheel，无法借 PyTorch 自带 nvidia/* 库
              兜底。可 `pip install torch --index-url https://download.pytorch.org/whl/cu121`
              或系统装 CUDA 12 + cuDNN 9。
            </div>
          )}
          {rt.preload && rt.preload.preloaded.length > 0 && (
            <div className="text-[10px] text-red-100/80">
              已预加载 {rt.preload.preloaded.length} 个 torch CUDA so 但仍失败 —
              可能 onnxruntime 与 CUDA minor 版本不匹配。
            </div>
          )}
        </div>
      )}
      <div className="flex gap-2 flex-wrap pt-1">
        <button
          onClick={() => install('auto')}
          disabled={busy !== null}
          className="px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 text-slate-200 disabled:opacity-50"
          title="按 nvidia-smi 检测自动选 GPU / CPU 包"
        >
          {busy === 'auto' ? '⏳ 装包中...' : '🔁 自动检测'}
        </button>
        <button
          onClick={() => install('gpu')}
          disabled={busy !== null}
          className="px-2 py-1 rounded bg-cyan-700 hover:bg-cyan-600 text-white disabled:opacity-50"
          title="重装 onnxruntime-gpu (CUDA 12.x)"
        >
          {busy === 'gpu' ? '⏳ 装包中...' : '⚡ 重装为 GPU'}
        </button>
        <button
          onClick={() => install('cpu')}
          disabled={busy !== null}
          className="px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 text-slate-300 disabled:opacity-50"
          title="重装 onnxruntime (CPU only)"
        >
          {busy === 'cpu' ? '⏳ 装包中...' : '🐢 重装为 CPU'}
        </button>
        <button
          onClick={() => void refresh()}
          disabled={busy !== null}
          className="px-2 py-1 rounded text-slate-400 hover:text-slate-200 disabled:opacity-50"
          title="重新读取 runtime 状态"
        >
          ↻
        </button>
      </div>
    </div>
  )
}
