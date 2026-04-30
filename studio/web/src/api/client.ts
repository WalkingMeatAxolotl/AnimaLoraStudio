// 与 FastAPI 守护进程交互的薄封装。
// 开发时由 Vite proxy 转发到 127.0.0.1:8765；生产部署时与 API 同源。

export interface HealthResponse {
  status: string
  version: string
}

export interface SchemaProperty {
  type?: string | string[]
  default?: unknown
  description?: string
  enum?: unknown[]
  minimum?: number
  maximum?: number
  exclusiveMinimum?: number
  exclusiveMaximum?: number
  group?: string
  control?: string
  cli_alias?: string
  show_when?: string
  anyOf?: Array<{ type?: string }>
  items?: SchemaProperty
}

export interface JsonSchema {
  properties: Record<string, SchemaProperty>
  required?: string[]
}

export interface SchemaResponse {
  schema: JsonSchema
  groups: Array<{ key: string; label: string; default_collapsed?: boolean }>
}

export interface PresetSummary {
  name: string
  path: string
  updated_at: number
}

/** PP0 之前叫 ConfigSummary —— 保留别名一段时间，避免外部代码炸掉。 */
export type ConfigSummary = PresetSummary

export type ConfigData = Record<string, unknown>

// ---- secrets (settings) ---------------------------------------------------

export interface GelbooruConfig {
  user_id: string
  api_key: string
  save_tags: boolean
  convert_to_png: boolean
  remove_alpha_channel: boolean
}

export interface DanbooruConfig {
  username: string
  api_key: string
  account_type: 'free' | 'gold' | 'platinum'
}

export interface DownloadGlobalConfig {
  exclude_tags: string[]
  /** PP9 — Booru 并发池：worker 数量。 */
  parallel_workers: number
  /** PP9 — API host (gelbooru.com / danbooru.donmai.us) 限速。 */
  api_rate_per_sec: number
  /** PP9 — CDN host (img*.gelbooru.com / cdn.donmai.us) 限速。 */
  cdn_rate_per_sec: number
}

export interface HuggingFaceConfig {
  token: string
}

export interface JoyCaptionConfig {
  base_url: string
  model: string
  prompt_template: string
}

export interface WD14Config {
  model_id: string
  /** 候选模型列表；用户在「设置 → WD14」里维护，model_id 必属于该列表。 */
  model_ids: string[]
  local_dir: string | null
  threshold_general: number
  threshold_character: number
  blacklist_tags: string[]
  /** PP8 — batch 推理大小；CPU EP 时强制 1。 */
  batch_size: number
}

/** PP8 — onnxruntime 装包状态 + nvidia-smi 检测结果。 */
export interface WD14Runtime {
  installed: 'onnxruntime' | 'onnxruntime-gpu' | null
  version: string | null
  providers: string[]
  cuda_available: boolean
  /** 装的包（dist-info）与当前进程已 import 的 .pyd 不一致 → 需重启 Studio。 */
  restart_required: boolean
  /** PP9.5 — InferenceSession 创建时实际 dlopen 报的错（如缺 libcurand.so.10）；
   *  非 null 表示已自动降级到 CPU EP，UI 应提示用户装 CUDA 库。 */
  cuda_load_error: string | null
  /** PP9.5 — torch 自带 CUDA so 预加载结果（Linux 才会 applied=true）。 */
  preload?: {
    applied: boolean
    platform_skip: boolean
    preloaded: string[]
    errors: [string, string][]
    candidates: number
  } | null
  cuda_detect: {
    available: boolean
    driver_version: string | null
    gpu_name: string | null
  }
}

export interface WD14InstallResult extends WD14Runtime {
  target: string
  installed_pkg: string | null
  installed_version: string | null
  stdout_tail: string
}

export const DEFAULT_WD14_MODELS: readonly string[] = [
  'SmilingWolf/wd-eva02-large-tagger-v3',
  'SmilingWolf/wd-vit-tagger-v3',
  'SmilingWolf/wd-vit-large-tagger-v3',
  'SmilingWolf/wd-v1-4-convnext-tagger-v2',
]

export interface ModelsConfig {
  /** 训练模型根目录；null/空 → 回退 REPO_ROOT/models/（云端机改这里） */
  root: string | null
  /** 当前默认主模型 variant（preview3-base / preview2 / preview）。
   * Studio 创建新 version 时把它展开成绝对路径写到 yaml.transformer_path；
   * 已存在 version 不动（保证训练重现性）。 */
  selected_anima: string
}

export interface Secrets {
  gelbooru: GelbooruConfig
  danbooru: DanbooruConfig
  download: DownloadGlobalConfig
  huggingface: HuggingFaceConfig
  joycaption: JoyCaptionConfig
  wd14: WD14Config
  models: ModelsConfig
}

/** PUT /api/secrets 的 body：嵌套的 partial dict；MASK ("***") 表示「保持不变」。 */
export type SecretsPatch = Partial<{
  [K in keyof Secrets]: Partial<Secrets[K]>
}>

// ---- models management (PP7) ---------------------------------------------

export interface ModelFileStatus {
  exists: boolean
  size: number
  mtime: number
}

export interface AnimaVariantInfo extends ModelFileStatus {
  variant: string
  is_latest: boolean
  target_path: string
}

export interface AnimaMainCatalog {
  id: 'anima_main'
  name: string
  description: string
  repo: string
  variants: AnimaVariantInfo[]
  latest: string
}

export interface AnimaVaeCatalog extends ModelFileStatus {
  id: 'anima_vae'
  name: string
  description: string
  repo: string
  target_path: string
}

export interface ModelDirCatalog {
  id: 'qwen3' | 't5_tokenizer'
  name: string
  description: string
  repo: string
  target_dir: string
  files: Array<{ name: string; exists: boolean; size: number; mtime: number }>
}

export interface ModelDownloadStatus {
  key: string
  status: 'pending' | 'running' | 'done' | 'failed'
  started_at: number
  finished_at: number | null
  message: string
  log_tail: string[]
}

export interface ModelsCatalog {
  models_root: string
  anima_main: AnimaMainCatalog
  anima_vae: AnimaVaeCatalog
  qwen3: ModelDirCatalog
  t5_tokenizer: ModelDirCatalog
  downloads: Record<string, ModelDownloadStatus>
}

// ---- projects / versions (PP1) -------------------------------------------

export type ProjectStage =
  | 'created'
  | 'downloading'
  | 'curating'
  | 'tagging'
  | 'regularizing'
  | 'configured'
  | 'training'
  | 'done'

export type VersionStage =
  | 'curating'
  | 'tagging'
  | 'regularizing'
  | 'ready'
  | 'training'
  | 'done'

export interface VersionStats {
  train_image_count: number
  tagged_image_count: number
  train_folders: Array<{ name: string; image_count: number }>
  reg_image_count: number
  reg_meta_exists: boolean
  has_output: boolean
}

export interface Version {
  id: number
  project_id: number
  label: string
  config_name: string | null
  stage: VersionStage
  created_at: number
  output_lora_path: string | null
  note: string | null
  stats?: VersionStats
}

export interface ProjectSummary {
  id: number
  slug: string
  title: string
  stage: ProjectStage
  active_version_id: number | null
  created_at: number
  updated_at: number
  note: string | null
  download_image_count?: number
}

export interface ProjectDetail extends ProjectSummary {
  versions: Version[]
  download_image_count: number
}

// ---- jobs (PP2) -----------------------------------------------------------

export type JobStatus = 'pending' | 'running' | 'done' | 'failed' | 'canceled'
export type JobKind = 'download' | 'tag' | 'reg_build'

export interface Job {
  id: number
  project_id: number
  version_id: number | null
  kind: JobKind
  params: string
  params_decoded?: Record<string, unknown> | null
  status: JobStatus
  started_at: number | null
  finished_at: number | null
  pid: number | null
  log_path: string | null
  error_msg: string | null
}

export interface DownloadFile {
  name: string
  size: number
  has_meta: boolean
}

export interface UploadResult {
  added: string[]
  skipped: { name: string; reason: string }[]
}

// ---- curation (PP3) -------------------------------------------------------

/**
 * Curation 列表里的一项：文件名 + 磁盘 mtime（unix 秒）。
 * mtime 用于支持「按下载时间」排序；后端不做排序保证（除按 name 字典序的稳定输出），
 * 排序由前端按用户偏好决定。
 */
export interface CurationItem {
  name: string
  mtime: number
}

export interface CurationView {
  left: CurationItem[] // download − train
  right: Record<string, CurationItem[]> // folder → items
  download_total: number
  train_total: number
  folders: string[]
}

export interface CopyResult {
  copied: string[]
  skipped: string[]
  missing: string[]
}

// ---- tagging (PP4) --------------------------------------------------------

export type TaggerName = 'wd14' | 'joycaption'

export interface TaggerStatus {
  name: TaggerName
  ok: boolean
  msg: string
  requires_service: boolean
}

export interface CaptionPreview {
  name: string
  folder: string
  tag_count: number
  tags_preview: string[]
  has_caption: boolean
}

/** full=1 时返回的 caption 列表项；含完整 tags + format。 */
export interface CaptionEntry extends CaptionPreview {
  tags: string[]
  format: 'txt' | 'json' | 'none'
}

export interface CommitItem {
  folder: string
  name: string
  tags: string[]
}

export interface CommitResult {
  snapshot: CaptionSnapshot
  written: number
  skipped: string[]
}

export interface CaptionFull {
  name: string
  tags: string[]
  format: 'txt' | 'json' | 'none'
}

export type BatchScope =
  | { kind: 'all' }
  | { kind: 'folder'; name: string }
  | { kind: 'files'; items: Array<{ folder: string; name: string }> }

export interface BatchOpRequest {
  op: 'add' | 'remove' | 'replace' | 'dedupe' | 'stats'
  scope: BatchScope
  tags?: string[]
  old?: string
  new?: string
  position?: 'front' | 'back'
  top?: number
}

export interface BatchOpResult {
  op: string
  affected?: number
  items?: Array<[string, number]>
}

export interface CaptionSnapshot {
  id: string
  created_at: number
  size: number
  file_count: number
}

// PP5 ----------------------------------------------------------------

export interface RegMeta {
  generated_at: number
  based_on_version: string
  api_source: string
  target_count: number
  actual_count: number
  source_tags: string[]
  excluded_tags: string[]
  blacklist_tags: string[]
  failed_tags: string[]
  train_tag_distribution: Record<string, number>
  auto_tagged: boolean
  incremental_runs: number
  // PP5.5 — 后处理摘要（postprocessed_at 为 null 表示未跑或 K 找不到）
  postprocessed_at: number | null
  postprocess_clusters: number | null
  postprocess_method: string | null
  postprocess_max_crop_ratio: number | null
}

export interface RegStatus {
  exists: boolean
  meta: RegMeta | null
  image_count: number
  files: string[]
}

export interface RegTagCount {
  tag: string
  count: number
}

// PP6.2 — Train config (version 私有，独立于全局 preset 池)
export interface VersionConfigResponse {
  has_config: boolean
  config: ConfigData | null
  /** 服务端强制覆盖的项目特定字段（前端表单应 disabled 这些） */
  project_specific_fields: string[]
}

export interface RegBuildRequest {
  excluded_tags?: string[]
  auto_tag?: boolean
  api_source?: 'gelbooru' | 'danbooru'
  incremental?: boolean
  // PP5.5 进阶
  skip_similar?: boolean
  aspect_ratio_filter_enabled?: boolean
  min_aspect_ratio?: number
  max_aspect_ratio?: number
  postprocess_method?: 'smart' | 'stretch' | 'crop'
  postprocess_max_crop_ratio?: number
}

export type TaskStatus = 'pending' | 'running' | 'done' | 'failed' | 'canceled'

export interface Task {
  id: number
  name: string
  config_name: string
  status: TaskStatus
  priority: number
  created_at: number
  started_at: number | null
  finished_at: number | null
  pid: number | null
  exit_code: number | null
  output_dir: string | null
  error_msg: string | null
  /** PP1 加；老任务为 null。 */
  project_id?: number | null
  /** PP1 加；老任务为 null。 */
  version_id?: number | null
  /** PP6.3 — version 私有 config 路径（旧任务 null，走 _configs_dir 兜底）。 */
  config_path?: string | null
  /** PP6.1 — per-task monitor state.json 路径。 */
  monitor_state_path?: string | null
}

export interface LogResponse {
  task_id: number
  content: string
  size: number
}

export interface TaskOutputFile {
  name: string
  size: number
  mtime: number
  is_lora: boolean
}

export interface TaskOutputs {
  task_id: number
  output_dir: string | null
  exists: boolean
  /** 仅 loopback 请求为 true；云端永远 false。前端按此控制「打开文件夹」按钮可见性。 */
  supports_open_folder: boolean
  files: TaskOutputFile[]
}

export interface DatasetFolder {
  name: string
  label: string
  repeat: number
  image_count: number
  caption_types: { json: number; txt: number; none: number }
  samples: string[]
  path: string
}

export interface DatasetScan {
  root: string
  exists: boolean
  folders: DatasetFolder[]
  total_images?: number
  weighted_steps_per_epoch?: number
}

export interface QueueExport {
  version: number
  exported_at: number
  tasks: Array<{
    name: string
    config_name: string
    priority: number
    config: Record<string, unknown> | null
  }>
}

export interface ImportResult {
  imported_count: number
  task_ids: number[]
  renamed: Record<string, string>
}

async function req<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const resp = await fetch(path, {
    headers: {
      Accept: 'application/json',
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
    },
    ...init,
  })
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`
    try {
      const body = await resp.json()
      if (body?.detail) detail = body.detail
    } catch {
      // ignore
    }
    throw new Error(detail)
  }
  if (resp.status === 204) return undefined as T
  return (await resp.json()) as T
}

/**
 * 下载二进制为浏览器附件。fetch + blob，让调用方能用 setLoading 包起来显示进度。
 *
 * 用 `<a href download>` 直链虽然简单但点击瞬间就让浏览器接管，前端无法
 * 显示「打 zip 中...」之类的 loading 状态 —— 训练集 / output 几百 MB 时
 * 后端打 zip 要几秒到几十秒，loading 反馈很有必要。
 */
export async function downloadBlob(url: string, filename: string): Promise<void> {
  const resp = await fetch(url, { headers: { Accept: 'application/zip,application/octet-stream' } })
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`
    try {
      const body = await resp.json()
      if (body?.detail) detail = body.detail
    } catch {
      // ignore
    }
    throw new Error(detail)
  }
  const blob = await resp.blob()
  const objectUrl = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = objectUrl
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  // 让浏览器有机会发起下载后再 revoke
  setTimeout(() => URL.revokeObjectURL(objectUrl), 1000)
}

export const api = {
  health: () => req<HealthResponse>('/api/health'),
  state: () => req<Record<string, unknown>>('/api/state'),

  schema: () => req<SchemaResponse>('/api/schema'),

  // Presets (PP0+) -----------------------------------------------------
  listPresets: () =>
    req<{ items: PresetSummary[] }>('/api/presets').then((r) => r.items),
  getPreset: (name: string) => req<ConfigData>(`/api/presets/${name}`),
  savePreset: (name: string, data: ConfigData) =>
    req<{ name: string; path: string }>(`/api/presets/${name}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  deletePreset: (name: string) =>
    req<{ deleted: string }>(`/api/presets/${name}`, { method: 'DELETE' }),
  duplicatePreset: (src: string, newName: string) =>
    req<{ name: string; path: string }>(`/api/presets/${src}/duplicate`, {
      method: 'POST',
      body: JSON.stringify({ new_name: newName }),
    }),

  // 兼容别名：PP0 之前叫 listConfigs / getConfig / ...。保留一段时间。
  listConfigs: () =>
    req<{ items: PresetSummary[] }>('/api/presets').then((r) => r.items),
  getConfig: (name: string) => req<ConfigData>(`/api/presets/${name}`),
  saveConfig: (name: string, data: ConfigData) =>
    req<{ name: string; path: string }>(`/api/presets/${name}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  deleteConfig: (name: string) =>
    req<{ deleted: string }>(`/api/presets/${name}`, { method: 'DELETE' }),
  duplicateConfig: (src: string, newName: string) =>
    req<{ name: string; path: string }>(`/api/presets/${src}/duplicate`, {
      method: 'POST',
      body: JSON.stringify({ new_name: newName }),
    }),

  // Secrets ------------------------------------------------------------
  getSecrets: () => req<Secrets>('/api/secrets'),

  // Models management (PP7) ------------------------------------------------
  getModelsCatalog: () => req<ModelsCatalog>('/api/models/catalog'),
  startModelDownload: (body: { model_id: string; variant?: string }) =>
    req<{ key: string; status: string }>('/api/models/download', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  updateSecrets: (patch: SecretsPatch) =>
    req<Secrets>('/api/secrets', {
      method: 'PUT',
      body: JSON.stringify(patch),
    }),

  // Projects / Versions (PP1) -------------------------------------------
  listProjects: () =>
    req<{ items: ProjectSummary[] }>('/api/projects').then((r) => r.items),
  getProject: (pid: number) =>
    req<ProjectDetail>(`/api/projects/${pid}`),
  createProject: (body: {
    title: string
    slug?: string
    note?: string
    initial_version_label?: string
  }) =>
    req<ProjectDetail>('/api/projects', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  updateProject: (
    pid: number,
    body: Partial<{
      title: string
      note: string
      stage: ProjectStage
      active_version_id: number | null
    }>
  ) =>
    req<ProjectDetail>(`/api/projects/${pid}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  deleteProject: (pid: number) =>
    req<{ deleted: number }>(`/api/projects/${pid}`, { method: 'DELETE' }),
  emptyTrash: () =>
    req<{ removed: number }>('/api/projects/_trash/empty', { method: 'POST' }),

  listVersions: (pid: number) =>
    req<{ items: Version[] }>(`/api/projects/${pid}/versions`).then(
      (r) => r.items
    ),
  getVersion: (pid: number, vid: number) =>
    req<Version>(`/api/projects/${pid}/versions/${vid}`),
  createVersion: (
    pid: number,
    body: {
      label: string
      fork_from_version_id?: number
      note?: string
    }
  ) =>
    req<Version>(`/api/projects/${pid}/versions`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  updateVersion: (
    pid: number,
    vid: number,
    body: Partial<{
      note: string
      stage: VersionStage
      config_name: string | null
    }>
  ) =>
    req<Version>(`/api/projects/${pid}/versions/${vid}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  deleteVersion: (pid: number, vid: number) =>
    req<{ deleted: number }>(`/api/projects/${pid}/versions/${vid}`, {
      method: 'DELETE',
    }),
  activateVersion: (pid: number, vid: number) =>
    req<ProjectDetail>(
      `/api/projects/${pid}/versions/${vid}/activate`,
      { method: 'POST' }
    ),

  // Download / jobs (PP2) ------------------------------------------------
  estimateDownload: (
    pid: number,
    body: { tag: string; api_source?: 'gelbooru' | 'danbooru' }
  ) =>
    req<{
      tag: string
      api_source: 'gelbooru' | 'danbooru'
      exclude_tags: string[]
      effective_query: string
      count: number
    }>(`/api/projects/${pid}/download/estimate`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  startDownload: (
    pid: number,
    body: { tag: string; count: number; api_source?: 'gelbooru' | 'danbooru' }
  ) =>
    req<Job>(`/api/projects/${pid}/download`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  getDownloadStatus: (pid: number) =>
    req<{ job: Job | null; log_tail: string }>(
      `/api/projects/${pid}/download/status`
    ),
  /**
   * 本地上传：单图（jpg/png）或 zip 包。绕过 `req` 的 JSON header，
   * 让浏览器自己加 multipart boundary。后端同步处理并返回结果。
   */
  uploadProjectFiles: async (
    pid: number,
    files: File[]
  ): Promise<UploadResult> => {
    const fd = new FormData()
    for (const f of files) fd.append('files', f, f.name)
    const resp = await fetch(`/api/projects/${pid}/upload`, {
      method: 'POST',
      body: fd,
    })
    if (!resp.ok) {
      let detail = `${resp.status} ${resp.statusText}`
      try {
        const body = await resp.json()
        if (body?.detail) detail = body.detail
      } catch {
        /* ignore */
      }
      throw new Error(detail)
    }
    return (await resp.json()) as UploadResult
  },
  listFiles: (pid: number, bucket = 'download') =>
    req<{ items: DownloadFile[]; count: number }>(
      `/api/projects/${pid}/files?bucket=${encodeURIComponent(bucket)}`
    ),
  /** 从 project 的 download/ 删除指定图片 + 同名 metadata（.booru.txt/.txt/.json）。 */
  deleteProjectFiles: (pid: number, names: string[]) =>
    req<{ deleted: string[]; missing: string[] }>(
      `/api/projects/${pid}/files/delete`,
      {
        method: 'POST',
        body: JSON.stringify({ names }),
      }
    ),
  projectThumbUrl: (pid: number, name: string, bucket = 'download', size = 256) =>
    `/api/projects/${pid}/thumb?bucket=${encodeURIComponent(bucket)}&name=${encodeURIComponent(name)}&size=${size}`,
  getJob: (jid: number) => req<Job>(`/api/jobs/${jid}`),
  getJobLog: (jid: number, tail?: number) => {
    const qs = tail ? `?tail=${tail}` : ''
    return req<{ job_id: number; content: string; size: number }>(
      `/api/jobs/${jid}/log${qs}`
    )
  },
  cancelJob: (jid: number) =>
    req<{ job_id: number; canceled: boolean }>(`/api/jobs/${jid}/cancel`, {
      method: 'POST',
    }),

  // Tagging (PP4) --------------------------------------------------------
  checkTagger: (name: TaggerName) =>
    req<TaggerStatus>(`/api/tagger/${name}/check`),
  startTag: (
    pid: number,
    vid: number,
    body: {
      tagger: TaggerName
      output_format?: 'txt' | 'json'
      /**
       * wd14 本次任务的临时覆盖；仅在 worker 进程生效，不写回 settings。
       * 字段为 undefined / null 时沿用全局 settings。
       */
      wd14_overrides?: {
        threshold_general?: number | null
        threshold_character?: number | null
        model_id?: string | null
        local_dir?: string | null
        blacklist_tags?: string[] | null
      }
    }
  ) =>
    req<Job>(`/api/projects/${pid}/versions/${vid}/tag`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  listCaptions: (pid: number, vid: number, folder?: string) => {
    const qs = folder ? `?folder=${encodeURIComponent(folder)}` : ''
    return req<{ folder: string | null; items: CaptionPreview[] }>(
      `/api/projects/${pid}/versions/${vid}/captions${qs}`
    )
  },
  listCaptionsFull: (pid: number, vid: number) =>
    req<{ folder: null; items: CaptionEntry[] }>(
      `/api/projects/${pid}/versions/${vid}/captions?full=1`
    ),
  commitCaptions: (pid: number, vid: number, items: CommitItem[]) =>
    req<CommitResult>(
      `/api/projects/${pid}/versions/${vid}/captions/commit`,
      { method: 'POST', body: JSON.stringify({ items }) }
    ),
  getCaption: (pid: number, vid: number, folder: string, filename: string) =>
    req<CaptionFull>(
      `/api/projects/${pid}/versions/${vid}/captions/${encodeURIComponent(folder)}/${encodeURIComponent(filename)}`
    ),
  putCaption: (
    pid: number,
    vid: number,
    folder: string,
    filename: string,
    tags: string[]
  ) =>
    req<CaptionFull>(
      `/api/projects/${pid}/versions/${vid}/captions/${encodeURIComponent(folder)}/${encodeURIComponent(filename)}`,
      { method: 'PUT', body: JSON.stringify({ tags }) }
    ),
  batchTag: (pid: number, vid: number, body: BatchOpRequest) =>
    req<BatchOpResult>(
      `/api/projects/${pid}/versions/${vid}/captions/batch`,
      { method: 'POST', body: JSON.stringify(body) }
    ),
  createCaptionSnapshot: (pid: number, vid: number) =>
    req<CaptionSnapshot>(
      `/api/projects/${pid}/versions/${vid}/captions/snapshot`,
      { method: 'POST' }
    ),
  listCaptionSnapshots: (pid: number, vid: number) =>
    req<{ items: CaptionSnapshot[] }>(
      `/api/projects/${pid}/versions/${vid}/captions/snapshots`
    ).then((r) => r.items),
  restoreCaptionSnapshot: (pid: number, vid: number, sid: string) =>
    req<{ id: string; written: number; removed_old: number; skipped: string[] }>(
      `/api/projects/${pid}/versions/${vid}/captions/snapshots/${sid}/restore`,
      { method: 'POST' }
    ),
  deleteCaptionSnapshot: (pid: number, vid: number, sid: string) =>
    req<{ deleted: string }>(
      `/api/projects/${pid}/versions/${vid}/captions/snapshots/${sid}`,
      { method: 'DELETE' }
    ),

  // Regularization (PP5) ------------------------------------------------
  getRegStatus: (pid: number, vid: number) =>
    req<RegStatus>(`/api/projects/${pid}/versions/${vid}/reg`),
  previewRegTags: (pid: number, vid: number, top = 20) =>
    req<{ items: RegTagCount[] }>(
      `/api/projects/${pid}/versions/${vid}/reg/preview-tags?top=${top}`
    ).then((r) => r.items),
  startRegBuild: (pid: number, vid: number, body: RegBuildRequest) =>
    req<Job>(`/api/projects/${pid}/versions/${vid}/reg/build`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  deleteReg: (pid: number, vid: number) =>
    req<{ deleted: boolean; reason?: string }>(
      `/api/projects/${pid}/versions/${vid}/reg`,
      { method: 'DELETE' }
    ),
  getRegCaption: (pid: number, vid: number, path: string) =>
    req<{ path: string; tags: string[] }>(
      `/api/projects/${pid}/versions/${vid}/reg/caption?path=${encodeURIComponent(path)}`
    ),

  // Train config (PP6.2) -------------------------------------------------
  getVersionConfig: (pid: number, vid: number) =>
    req<VersionConfigResponse>(`/api/projects/${pid}/versions/${vid}/config`),
  putVersionConfig: (pid: number, vid: number, data: ConfigData) =>
    req<{ has_config: true; config: ConfigData }>(
      `/api/projects/${pid}/versions/${vid}/config`,
      { method: 'PUT', body: JSON.stringify(data) }
    ),
  forkPresetForVersion: (pid: number, vid: number, name: string) =>
    req<{ has_config: true; config: ConfigData; from_preset: string }>(
      `/api/projects/${pid}/versions/${vid}/config/from_preset`,
      { method: 'POST', body: JSON.stringify({ name }) }
    ),
  saveVersionConfigAsPreset: (
    pid: number,
    vid: number,
    name: string,
    overwrite = false
  ) =>
    req<{ saved_preset: string; config: ConfigData }>(
      `/api/projects/${pid}/versions/${vid}/config/save_as_preset`,
      { method: 'POST', body: JSON.stringify({ name, overwrite }) }
    ),
  enqueueVersionTraining: (pid: number, vid: number) =>
    req<Task>(
      `/api/projects/${pid}/versions/${vid}/queue`,
      { method: 'POST' }
    ),

  // Curation (PP3) -------------------------------------------------------
  getCuration: (pid: number, vid: number) =>
    req<CurationView>(`/api/projects/${pid}/versions/${vid}/curation`),
  copyToTrain: (
    pid: number,
    vid: number,
    body: { files: string[]; dest_folder: string }
  ) =>
    req<CopyResult>(`/api/projects/${pid}/versions/${vid}/curation/copy`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  removeFromTrain: (
    pid: number,
    vid: number,
    body: { folder: string; files: string[] }
  ) =>
    req<{ removed: string[]; missing: string[] }>(
      `/api/projects/${pid}/versions/${vid}/curation/remove`,
      { method: 'POST', body: JSON.stringify(body) }
    ),
  folderOp: (
    pid: number,
    vid: number,
    body: { op: 'create' | 'rename' | 'delete'; name: string; new_name?: string }
  ) =>
    req<Record<string, unknown>>(
      `/api/projects/${pid}/versions/${vid}/curation/folder`,
      { method: 'POST', body: JSON.stringify(body) }
    ),
  versionThumbUrl: (
    pid: number,
    vid: number,
    bucket: 'train' | 'reg' | 'samples',
    name: string,
    folder?: string,
    size: number = 256
  ) => {
    const qs = new URLSearchParams({ bucket, name, size: String(size) })
    if (folder) qs.set('folder', folder)
    return `/api/projects/${pid}/versions/${vid}/thumb?${qs.toString()}`
  },

  // Queue --------------------------------------------------------------
  listQueue: (status?: TaskStatus) => {
    const qs = status ? `?status=${status}` : ''
    return req<{ items: Task[] }>(`/api/queue${qs}`).then((r) => r.items)
  },
  getTask: (id: number) => req<Task>(`/api/queue/${id}`),
  enqueue: (payload: { config_name: string; name?: string; priority?: number }) =>
    req<Task>('/api/queue', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  cancelTask: (id: number) =>
    req<{ task_id: number; canceled: boolean }>(`/api/queue/${id}/cancel`, {
      method: 'POST',
    }),
  retryTask: (id: number) =>
    req<Task>(`/api/queue/${id}/retry`, { method: 'POST' }),
  deleteTask: (id: number) =>
    req<{ deleted: number }>(`/api/queue/${id}`, { method: 'DELETE' }),
  /** 列 task 关联的 output 目录里所有文件（含 size/mtime/是否 lora）。
   * `supports_open_folder` 仅在请求来自 loopback 时为 true，云端为 false。 */
  getTaskOutputs: (id: number) =>
    req<TaskOutputs>(`/api/queue/${id}/outputs`),
  /** 下载单个 output 文件的直链，不发请求。<a href={...} download> 即可。 */
  taskOutputDownloadUrl: (id: number, filename: string) =>
    `/api/queue/${id}/output/${encodeURIComponent(filename)}`,
  /** 把 output 目录全部文件打包成 zip 下载的直链。
   * 推荐用 downloadBlob() 调它，能显示 loading（后端打 zip 要时间）。 */
  taskOutputsZipUrl: (id: number) => `/api/queue/${id}/outputs.zip`,

  // PP8 — WD14 运行时 / GPU 装包 ------------------------------------------
  /** 当前 onnxruntime 状态：包名 / 版本 / providers / nvidia-smi 检测结果。 */
  getWD14Runtime: () => req<WD14Runtime>('/api/wd14/runtime'),
  /** 切换 onnxruntime（同步 pip，几分钟级；UI 必须带 loading）。 */
  installWD14Runtime: (target: 'auto' | 'gpu' | 'cpu') =>
    req<WD14InstallResult>('/api/wd14/install', {
      method: 'POST',
      body: JSON.stringify({ target }),
    }),

  // PP7 — 训练集导出 / 导入 -----------------------------------------------
  /** 当前 version 的 train/ 打包 zip 直链。用 downloadBlob() 调它显示 loading。 */
  versionTrainZipUrl: (pid: number, vid: number) =>
    `/api/projects/${pid}/versions/${vid}/train.zip`,
  /** 上传训练集 zip → 新建 project + v1，返回新项目。 */
  importTrainProject: async (file: File): Promise<{
    project: ProjectDetail
    version: Version
    stats: { image_count: number; tagged_count: number; untagged_count: number; concepts: string[] }
  }> => {
    const fd = new FormData()
    fd.append('file', file)
    const resp = await fetch('/api/projects/import-train', { method: 'POST', body: fd })
    if (!resp.ok) {
      let detail = `${resp.status} ${resp.statusText}`
      try {
        const body = await resp.json()
        if (body?.detail) detail = body.detail
      } catch {
        // ignore
      }
      throw new Error(detail)
    }
    return resp.json()
  },
  /** 在 server 主机的 OS 文件管理器里打开 output 目录（仅 loopback 可用）。 */
  openTaskFolder: (id: number) =>
    req<{ opened: string }>(`/api/queue/${id}/open-folder`, {
      method: 'POST',
    }),
  reorderQueue: (orderedIds: number[]) =>
    req<{ reordered: number }>('/api/queue/reorder', {
      method: 'POST',
      body: JSON.stringify({ ordered_ids: orderedIds }),
    }),
  getLog: (id: number) => req<LogResponse>(`/api/logs/${id}`),

  // Queue import / export ---------------------------------------------
  exportQueue: (ids?: number[]) => {
    const qs = ids && ids.length ? `?ids=${ids.join(',')}` : ''
    return req<QueueExport>(`/api/queue/export${qs}`)
  },
  importQueue: (payload: unknown) =>
    req<ImportResult>('/api/queue/import', {
      method: 'POST',
      body: JSON.stringify({ payload }),
    }),

  // Datasets -----------------------------------------------------------
  listDatasets: (path?: string) => {
    const qs = path ? `?path=${encodeURIComponent(path)}` : ''
    return req<DatasetScan>(`/api/datasets${qs}`)
  },
  thumbnailUrl: (folder: string, name: string) =>
    `/api/datasets/thumbnail?folder=${encodeURIComponent(folder)}&name=${encodeURIComponent(name)}`,

  // Browse -------------------------------------------------------------
  browse: (path?: string) => {
    const qs = path ? `?path=${encodeURIComponent(path)}` : ''
    return req<BrowseResult>(`/api/browse${qs}`)
  },
}

export interface BrowseEntry {
  name: string
  type: 'dir' | 'file'
}

export interface BrowseResult {
  path: string
  parent: string | null
  entries: BrowseEntry[]
}
