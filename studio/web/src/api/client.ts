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
  /** 后端打了 hidden=True 的字段：值仍随 ConfigData 透传 / 保存，但 SchemaForm
   * 不渲染。用于「该字段对当前用户群无意义但 schema 必须保留」的兜底场景。 */
  hidden?: boolean
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
  /** PR-S3 — HF 模型下载端点 endpoint。
   *  `""` → huggingface_hub 默认（直连 huggingface.co）；海外用户推荐
   *  `"https://hf-mirror.com"` → 国内默认（项目主战场国内）
   *  其它 URL → 自定义反代 / 自建镜像 */
  endpoint: string
}

export interface ModelScopeConfig {
  /** 魔搭社区 token。公开模型可不填；私有 / 限速时需要。 */
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

export interface CLTaggerConfig {
  model_id: string
  model_path: string
  tag_mapping_path: string
  local_dir: string | null
  threshold_general: number
  threshold_character: number
  add_rating_tag: boolean
  add_model_tag: boolean
  blacklist_tags: string[]
  batch_size: number
}

/** PR-S2 — PyTorch 安装状态 + 驱动检测 + 推荐 cu tag。 */
export type TorchCuTag = 'cu128' | 'cu126' | 'cu124' | 'cu118' | 'cpu'
export interface TorchStatus {
  installed: boolean
  version: string | null              // "2.5.0+cu128"
  cuda_build: TorchCuTag | null       // 解析自 +suffix
  cuda_available: boolean             // torch.cuda.is_available()
  device_name: string | null          // "NVIDIA GeForce RTX 5090"
  cuda_detect: {
    available: boolean
    driver_version: string | null
    gpu_name: string | null
  }
  recommended_cu_tag: TorchCuTag      // 按驱动版本推荐
  /** 装了 CPU wheel 但有 NVIDIA GPU → 误装，UI 显示「重装为 CUDA 版」红色提示。 */
  is_cpu_with_gpu: boolean
  /** 装了 CUDA wheel 但 cuda.is_available()=False → 驱动 / WSL 问题，pip 修不了。 */
  is_cuda_build_unavailable: boolean
}
/** torch reinstall 总是 deferred：server 写 marker，下次 launcher 启动时跑 pip。
 *  这样避开 Windows 上 torch .pyd 已被 server 进程加载、pip 无法 replace 的死锁。 */
export interface TorchReinstallResult {
  pending: true                       // 永远 true，提示 UI 走「请重启」分支
  target: string                      // 用户传的（"auto" 等）
  tag: TorchCuTag                     // 实际选定（auto 已被 server 解析）
  message: string                     // 中文人话提示，UI 直接显示
}

/** PR-7b — Flash Attention 安装状态 + 环境检测 + GitHub 候选 wheel。 */
export interface FlashAttnEnv {
  python_tag: string                 // cp311
  cuda_tag: string | null            // cu128 / null = 没 nvidia-smi 也没 torch
  cuda_ver: string | null            // 12.8（PyTorch 编译时绑定，flash_attn ABI 跟它走）
  /** nvidia-smi 报告的驱动支持的最高 CUDA；与 cuda_ver 可能不同。
   * 排错时给用户看："驱动支持 cu130，PyTorch 是 cu128，应装 cu128 wheel"。 */
  driver_cuda_ver: string | null
  torch_tag: string | null           // torch2.5
  torch_ver: string | null
  platform: 'linux_x86_64' | 'win_amd64' | null
}
export interface FlashAttnCandidate {
  url: string
  name: string                       // flash_attn-2.8.3+cu128torch2.5-cp311-cp311-win_amd64.whl
  notes: string[]                    // 兼容性说明（CUDA 大版本不同 / Python 不兼容）
  usable: boolean                    // false = Python ABI 不匹配，UI 灰显但允许强装
}
export interface FlashAttnStatus {
  installed: boolean
  version: string | null
  env: FlashAttnEnv
  candidates: FlashAttnCandidate[]   // 按 score 降序，最多 20
  fetch_error: string | null         // GitHub API 限流 / 网络异常
}
export interface FlashAttnInstallResult {
  installed: boolean
  version: string | null
  url: string
  stdout_tail: string                // pip 输出末 40 行
  restart_required: boolean
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
  /** PP9.6 — GPU 路径连同装的 nvidia-*-cu12 wheels 报告；CPU 路径或非 Linux 为 null。
   *  含 `error` 字段表示 onnxruntime-gpu 装好但 CUDA wheels 装失败（不致命）。 */
  cuda_runtime: {
    installed: string[]
    skipped: string[]
    platform_skip: boolean
    stdout?: string
    error?: string
  } | null
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

export interface QueueConfig {
  /** PP10.2：默认 false，训练时推迟 tag/reg_build job 避免 GPU OOM。
   * 用户开后允许 GPU job 与训练并行（自己确认显存够）。 */
  allow_gpu_during_train: boolean
}

/** Phase 2 commit 14 — 测试出图 daemon 行为。 */
export interface GenerateSecretsConfig {
  /** TAEFlux 中间步预览节流。0=关；>0 → daemon 每 N 步推 256px JPEG。
   * 模型缺失时 daemon 静默回退（无预览不影响出图）。 */
  preview_every_n_steps: number
  /** 注意力后端默认值（design 决策：用户配置一次，不每次出图都改）。
   * Generate 页 enqueue 自动注入；Settings 训练 tab 切换。 */
  attention_backend: AttentionBackend
}

export interface Secrets {
  gelbooru: GelbooruConfig
  danbooru: DanbooruConfig
  download: DownloadGlobalConfig
  huggingface: HuggingFaceConfig
  modelscope: ModelScopeConfig
  /** 模型下载源：'huggingface'（默认）或 'modelscope'。
   *  选 modelscope 时，有映射的模型走魔搭 CLI 下载；无映射的自动回退 HF。 */
  download_source: string
  joycaption: JoyCaptionConfig
  wd14: WD14Config
  cltagger: CLTaggerConfig
  models: ModelsConfig
  queue: QueueConfig
  generate: GenerateSecretsConfig
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

export interface WD14VariantInfo {
  model_id: string
  is_current: boolean
  target_path: string
  exists: boolean
  size: number
  files: Array<{ name: string; exists: boolean; size: number; mtime: number }>
}

export interface WD14Catalog {
  id: 'wd14'
  name: string
  description: string
  repo: string
  current_model_id: string
  variants: WD14VariantInfo[]
}

export interface CLTaggerVariantInfo {
  label: string
  model_path: string
  tag_mapping_path: string
  is_current: boolean
  exists: boolean
  size: number
  files: Array<{ name: string; exists: boolean; size: number; mtime: number }>
}

export interface CLTaggerCatalog {
  id: 'cltagger'
  name: string
  description: string
  repo: string
  target_dir: string
  current_model_path: string
  current_tag_mapping_path: string
  variants: CLTaggerVariantInfo[]
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
  wd14: WD14Catalog
  cltagger: CLTaggerCatalog
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

export type TaggerName = 'wd14' | 'cltagger' | 'joycaption'

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
  // "scrape" = booru 拉取，"ai_base" = base 模型先验生成；缺省按 "scrape" 处理（旧 meta 兼容）
  generation_method?: 'scrape' | 'ai_base'
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

/** Attention backend 三选一 — 替代原 xformers/flash_attn 双 bool。 */
/** secrets.generate.attention_backend：'auto' = 按装了什么用（默认）；
 *  显式值（flash_attn/xformers/none）则强制。GenerateRequest 也接此 type
 *  作为 per-request 覆盖（前端不再发；server 自动从 secrets 读 + auto 解析）。 */
export type AttentionBackend = 'auto' | 'none' | 'xformers' | 'flash_attn'

/** PR-9 — 先验生成（base 模型反向出 reg 集，无 LoRA）。 */
export interface RegAiRequest {
  excluded_tags?: string[]
  negative_prompt?: string
  width?: number
  height?: number
  steps?: number
  cfg_scale?: number
  sampler_name?: string
  scheduler?: string
  seed?: number
  incremental?: boolean
  mixed_precision?: string
  attention_backend?: AttentionBackend
}

/** PR-9 — 测试出图（独立工具页，多 LoRA + multi-prompt）。 */
export interface LoraEntry {
  path: string
  scale: number
  /** 来自 picker 的项目 / 版本绑定；外部文件无 */
  project_id?: number | null
  version_id?: number | null
}

/** XY 矩阵：单 task 内循环全图，前端按 (yi, xi) 排成 grid。
 *  设了 xy_matrix 时后端强制 prompts 单条 + count=1（避免排列爆炸）。
 *  v1 不支持 lora_path 轴（缺 unhook 接口，留 v2）。 */
export type XYAxisType =
  | 'lora_scale'
  | 'steps'
  | 'cfg_scale'
  | 'lora_ckpt'  // 同一 LoRA 的不同 step/epoch ckpt（找过拟合拐点）

export interface XYAxisSpec {
  axis: XYAxisType
  /** 类型按 axis 派生：steps→int；lora_scale/cfg_scale→number；lora_ckpt→string(path) */
  values: Array<number | string>
  /** axis=lora_scale / lora_ckpt 时必填 —— 绑定到 lora_configs 哪一项 */
  lora_index?: number | null
}

export interface XYMatrixSpec {
  x: XYAxisSpec
  y?: XYAxisSpec | null
}

export interface GenerateRequest {
  prompts: string[]
  negative_prompt?: string
  width?: number
  height?: number
  steps?: number
  cfg_scale?: number
  sampler_name?: string
  scheduler?: string
  count?: number
  seed?: number
  lora_configs?: LoraEntry[]
  mixed_precision?: string
  attention_backend?: AttentionBackend
  /** 设值时 prompts 限单条 + count=1（schema 校验） */
  xy_matrix?: XYMatrixSpec | null
}

/** version output/ 下扫到的 training_state_step*.pt（断点续训用）。 */
export interface StateCkpt {
  /** global_step 数 */
  step: number
  /** 显示用："step 2476" */
  label: string
  /** 绝对路径 */
  path: string
  /** 文件 mtime 时间戳 */
  mtime: number
}

/** 项目级按 version 分组的 ckpt 列表（resume_state / resume_lora picker 用）。 */
export interface VersionCkptGroup<T> {
  version_id: number
  /** version label，如 "baseline" / "high-lr" */
  label: string
  items: T[]
}

/** version output/ 下扫到的 LoRA ckpt 文件（GET .../lora_ckpts）。 */
export interface LoraCkpt {
  /** 'final' / 'step' / 'epoch' / 'other' */
  kind: 'final' | 'step' | 'epoch' | 'other'
  /** step / epoch 数；final / other 为 0 */
  value: number
  /** 显示用：'final' / 'step 2476' / 'epoch 5' / 文件名 */
  label: string
  /** 绝对路径 */
  path: string
  /** 文件 mtime 时间戳 */
  mtime: number
}

/** Phase 2 commit 14 — TAEFlux 模型状态（GET /api/generate/taeflux/status）。 */
export interface TaeFluxStatus {
  available: boolean
  dir: string
  files: string[]
}

/** Phase 2 — Inference daemon 当前状态（GET /api/generate/daemon/status）。 */
export interface DaemonStatus {
  state: 'stopped' | 'starting' | 'idle' | 'busy' | 'unloading'
  model_loaded: boolean
  busy: boolean
  alive: boolean
}

/** xformers 安装状态 / 安装结果（简化版，对照 FlashAttnStatus）。 */
export interface XformersStatus {
  installed: boolean
  version: string | null
}

export interface XformersInstallResult {
  installed: boolean
  version: string | null
  stdout_tail: string
  restart_required: boolean
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

/** /api/state — per-task monitor state written by the training process */
export interface MonitorState {
  step?: number
  total_steps?: number
  epoch?: number
  total_epochs?: number
  speed?: number          // it/s
  start_time?: number     // unix seconds
  losses?: Array<{ step: number; loss: number }>
  lr_history?: Array<{ step: number; lr: number }>
  samples?: Array<{
    path: string
    step?: number
    /** XY 模式时携带 cell 元数据（generate task 才有；训练 task 为空）。 */
    xy?: { xi: number; yi: number; xv: number | string; yv: number | string | null }
  }>
  config?: Record<string, string | number | boolean>
  vram_used_gb?: number
  vram_total_gb?: number
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
  getLatestVersionJob: (
    pid: number,
    vid: number,
    kind: 'download' | 'tag' | 'reg_build',
  ) =>
    req<{ job: Job | null; log: string }>(
      `/api/projects/${pid}/versions/${vid}/jobs/latest?kind=${kind}`,
    ),

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
      cltagger_overrides?: {
        threshold_general?: number | null
        threshold_character?: number | null
        model_id?: string | null
        model_path?: string | null
        tag_mapping_path?: string | null
        local_dir?: string | null
        add_rating_tag?: boolean | null
        add_model_tag?: boolean | null
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
  /** PR-9 — 启动先验生成 task（base 模型对每张 train 图反向出对照图）。 */
  enqueueRegPrior: (pid: number, vid: number, body: RegAiRequest) =>
    req<Task>(`/api/projects/${pid}/versions/${vid}/reg/generate-prior`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  /** 查询先验生成 task 状态。 */
  getRegPriorTask: (pid: number, vid: number, taskId: number) =>
    req<Task>(`/api/projects/${pid}/versions/${vid}/reg/generate-prior/${taskId}`),

  /** 列出 version output/ 下所有 LoRA ckpt 文件（XY ckpt 轴 + 单图模式切 ckpt）。 */
  listVersionLoraCkpts: (pid: number, vid: number) =>
    req<{ items: LoraCkpt[] }>(`/api/projects/${pid}/versions/${vid}/lora_ckpts`)
      .then((r) => r.items),

  /** 列出项目所有 versions 的 state.pt，按 version 分组（Train 页 resume_state picker）。 */
  listProjectStateCkpts: (pid: number) =>
    req<{ groups: VersionCkptGroup<StateCkpt>[] }>(`/api/projects/${pid}/state_ckpts`)
      .then((r) => r.groups),

  /** 列出项目所有 versions 的 LoRA ckpt，按 version 分组（Train 页 resume_lora picker）。 */
  listProjectLoraCkpts: (pid: number) =>
    req<{ groups: VersionCkptGroup<LoraCkpt>[] }>(`/api/projects/${pid}/lora_ckpts`)
      .then((r) => r.groups),

  /** PR-9 — 启动测试出图 task。Phase 2 起：图走 server 内存 cache，关页面即丢。 */
  enqueueGenerate: (body: GenerateRequest) =>
    req<Task>('/api/generate', { method: 'POST', body: JSON.stringify(body) }),
  /** 查询测试 task 状态。 */
  getGenerateTask: (id: number) => req<Task>(`/api/generate/${id}`),
  /** 测试出图单张 URL（task 跑中或刚完成时拉；客户端断连 30s + LRU 后 404）。 */
  generateSampleUrl: (taskId: number, filename: string) =>
    `/api/generate/${taskId}/sample/${encodeURIComponent(filename)}`,
  /** Phase 2 — daemon 状态查询（前端 DaemonControls）。 */
  getDaemonStatus: () => req<DaemonStatus>('/api/generate/daemon/status'),
  /** Phase 2 — 手动卸载 daemon 模型（busy 时 409）。 */
  unloadDaemon: () => req<{ ok: boolean; noop?: boolean }>(
    '/api/generate/daemon/unload', { method: 'POST' }
  ),
  /** Phase 2 commit 14 — TAEFlux 状态。 */
  getTaeFluxStatus: () => req<TaeFluxStatus>('/api/generate/taeflux/status'),
  /** Phase 2 commit 14 — 同步下载 TAEFlux（~1.6MB，秒级）。已存在 noop。 */
  installTaeFlux: () => req<{ ok: boolean; noop?: boolean }>(
    '/api/generate/taeflux/install', { method: 'POST' }
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

  // PR-S2 — PyTorch 运行时 / 一键重装 ---------------------------------------
  /** 当前 torch 状态：版本 / CUDA build / cuda.is_available / 驱动检测 / 推荐 cu tag。 */
  getTorchStatus: () => req<TorchStatus>('/api/torch/status'),
  /** 卸装重装 torch + torchvision；同步 pip，可能 5-30 分钟，UI 必须带 loading。
   *  装完必须重启 Studio（C extension 不能热替换）。 */
  reinstallTorch: (target: 'auto' | TorchCuTag) =>
    req<TorchReinstallResult>('/api/torch/reinstall', {
      method: 'POST',
      body: JSON.stringify({ target }),
    }),

  // PR-7b — Flash Attention 运行时 / wheel 安装 ----------------------------
  /** 当前 flash_attn 状态 + 环境检测 + GitHub 候选 wheel 列表（前 20）。
   *  fetch_error 非 null 时 candidates=[]，UI 应提示用户改用手动 URL。 */
  getFlashAttnStatus: () => req<FlashAttnStatus>('/api/flash-attention/status'),
  /** 安装 flash_attn wheel；url=null 走 service 自动匹配。
   *  同步 pip install（远端 wheel ~150MB），可能几分钟；UI 按钮必须带 loading。
   *  装完必须重启 Studio 才能切换（C extension 不能热替换）。 */
  installFlashAttn: (url: string | null) =>
    req<FlashAttnInstallResult>('/api/flash-attention/install', {
      method: 'POST',
      body: JSON.stringify({ url }),
    }),

  // xformers 运行时（attention_backend=xformers 用） -----------------------
  /** xformers 安装状态。比 flash_attn 简洁：xformers 走 PyPI 直装，
   *  没有 GitHub 候选 wheel 列表的复杂选择逻辑。 */
  getXformersStatus: () => req<XformersStatus>('/api/xformers/status'),
  /** pip install xformers --index-url <torch-cu-index>。同步 pip，几分钟级。
   *  装失败时后端把 stderr 末尾透传到 message，多数失败 = 上游 wheel 没覆盖
   *  当前 torch+cu 组合。装完必须重启 Studio（C extension 不能热替换）。 */
  installXformers: () =>
    req<XformersInstallResult>('/api/xformers/install', { method: 'POST' }),

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
  getMonitorState: (taskId: number, maxPoints = 1500) =>
    req<MonitorState>(`/api/state?task_id=${taskId}&max_points=${maxPoints}&_=${Date.now()}`),
  sampleImageUrl: (filename: string, taskId: number, w?: number) =>
    `/samples/${filename}?task_id=${taskId}${w ? `&w=${w}` : ''}`,

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
