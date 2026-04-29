import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useOutletContext } from 'react-router-dom'
import {
  api,
  type ConfigData,
  type PresetSummary,
  type ProjectDetail,
  type SchemaResponse,
  type Version,
  type VersionConfigResponse,
} from '../../../api/client'
import SchemaForm from '../../../components/SchemaForm'
import { useToast } from '../../../components/Toast'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

export default function TrainPage() {
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const navigate = useNavigate()

  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [presets, setPresets] = useState<PresetSummary[]>([])
  const [configResp, setConfigResp] = useState<VersionConfigResponse | null>(null)
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [busy, setBusy] = useState(false)

  const vid = activeVersion?.id ?? null

  const refreshConfig = useCallback(async () => {
    if (!vid) return
    try {
      const r = await api.getVersionConfig(project.id, vid)
      setConfigResp(r)
      setConfig(r.config)
    } catch (e) {
      toast(`加载训练配置失败: ${e}`, 'error')
    }
  }, [project.id, vid, toast])

  useEffect(() => {
    api.schema().then(setSchema).catch((e) => toast(`schema 加载失败: ${e}`, 'error'))
    api.listPresets().then(setPresets).catch(() => setPresets([]))
  }, [toast])

  useEffect(() => {
    void refreshConfig()
  }, [refreshConfig])

  // 项目特定字段（data_dir 等）+ 全局模型路径都灰显 readonly。前者来自后端
  // project_specific_fields；后者前端 hardcode（统一来自 settings.models 配置）。
  const GLOBAL_MODEL_FIELDS = [
    'transformer_path',
    'vae_path',
    'text_encoder_path',
    't5_tokenizer_path',
  ]
  const disabledFields = useMemo(
    () => [
      ...(configResp?.project_specific_fields ?? []),
      ...GLOBAL_MODEL_FIELDS,
    ],
    [configResp]
  )
  const disabledHints = useMemo(() => {
    const h: Record<string, string> = {}
    for (const f of GLOBAL_MODEL_FIELDS) h[f] = '自动 · 全局设置'
    return h
  }, [])

  const dirty = useMemo(() => {
    if (!config || !configResp?.config) return false
    return JSON.stringify(config) !== JSON.stringify(configResp.config)
  }, [config, configResp])

  if (!activeVersion || !vid) {
    return <p className="text-slate-500">请先选择 / 创建一个版本</p>
  }

  const onForkPreset = async (name: string) => {
    if (!name) return
    if (
      configResp?.has_config &&
      !window.confirm(
        `换预设会覆盖当前 version 的配置（已保存的内容会丢失）。继续？`
      )
    ) {
      return
    }
    setBusy(true)
    try {
      await api.forkPresetForVersion(project.id, vid, name)
      await refreshConfig()
      toast(`已从预设 ${name} 复制`, 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const onSaveConfig = async () => {
    if (!config) return
    setBusy(true)
    try {
      const r = await api.putVersionConfig(project.id, vid, config)
      setConfigResp({
        has_config: true,
        config: r.config,
        project_specific_fields: configResp?.project_specific_fields ?? [],
      })
      setConfig(r.config)
      toast('已保存', 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const onSaveAsPreset = async () => {
    const name = window.prompt(
      '保存为新预设。预设名（字母 / 数字 / _ / -，会清掉项目特定字段）：',
      ''
    )
    if (!name) return
    setBusy(true)
    try {
      await api.saveVersionConfigAsPreset(project.id, vid, name, false)
      const list = await api.listPresets()
      setPresets(list)
      toast(`已保存为预设 ${name}`, 'success')
    } catch (e) {
      const msg = String(e)
      if (msg.includes('已存在')) {
        if (window.confirm(`预设 ${name} 已存在，覆盖？`)) {
          try {
            await api.saveVersionConfigAsPreset(project.id, vid, name, true)
            const list = await api.listPresets()
            setPresets(list)
            toast(`已覆盖预设 ${name}`, 'success')
          } catch (e2) {
            toast(String(e2), 'error')
          }
        }
      } else {
        toast(msg, 'error')
      }
    } finally {
      setBusy(false)
    }
  }

  const onEnqueue = async () => {
    if (!configResp?.has_config) {
      toast('先选预设并保存配置', 'error')
      return
    }
    if (dirty) {
      if (!window.confirm('当前有未保存的改动，确定按上次保存的配置入队？')) return
    }
    setBusy(true)
    try {
      const t = await api.enqueueVersionTraining(project.id, vid)
      toast(`已入队 #${t.id}，去 /queue 查看进度`, 'success')
      void reload()
      navigate('/queue')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-col h-full w-full gap-3">
      <header className="flex items-baseline gap-2 flex-wrap shrink-0">
        <h2 className="text-base font-semibold">⑥ 训练</h2>
        <span className="text-xs text-slate-500">
          选预设 → 编辑 config（项目特定字段自动填）→ 入队
        </span>
      </header>

      <section className="rounded-lg border border-slate-700 bg-slate-800/40 p-3 flex flex-wrap items-center gap-3 text-xs shrink-0">
        <span className="text-slate-400">当前预设来源</span>
        <span className="font-mono text-cyan-300">
          {activeVersion.config_name ?? '(未选)'}
        </span>
        <span className="text-slate-700">|</span>
        <span className="text-slate-400">换一个预设</span>
        <select
          value=""
          onChange={(e) => void onForkPreset(e.target.value)}
          disabled={busy || presets.length === 0}
          className="px-2 py-1 rounded bg-slate-950 border border-slate-700"
        >
          <option value="">— 选预设 —</option>
          {presets.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name}
            </option>
          ))}
        </select>
        <button
          onClick={() => void onSaveAsPreset()}
          disabled={busy || !configResp?.has_config}
          className="px-2 py-1 rounded text-xs bg-slate-700 hover:bg-slate-600 disabled:opacity-40"
        >
          保存为新预设
        </button>
        <span className="flex-1" />
        <button
          onClick={() => void onSaveConfig()}
          disabled={busy || !dirty}
          className="px-3 py-1 rounded text-xs bg-slate-700 hover:bg-slate-600 disabled:opacity-40"
        >
          {dirty ? '保存配置' : '已保存'}
        </button>
        <button
          onClick={() => void onEnqueue()}
          disabled={busy || !configResp?.has_config}
          className="px-3 py-1 rounded bg-cyan-600 hover:bg-cyan-500 text-white disabled:bg-slate-700 disabled:text-slate-500"
        >
          开始训练
        </button>
      </section>

      {!configResp?.has_config ? (
        <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
          请先从下拉「换一个预设」选一个，复制进当前 version 后即可编辑配置。
        </div>
      ) : config && schema ? (
        <section className="flex-1 min-h-0 overflow-y-auto pr-1">
          <SchemaForm
            schema={schema}
            values={config}
            onChange={setConfig}
            disabledFields={disabledFields}
            disabledHints={disabledHints}
          />
        </section>
      ) : (
        <p className="text-slate-500 text-sm">加载中...</p>
      )}
    </div>
  )
}
