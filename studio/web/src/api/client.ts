// 与 FastAPI 守护进程交互的薄封装。
// 开发时由 Vite proxy 转发到 127.0.0.1:8765；生产部署时与 API 同源。

// ADR-0009 PR-3 C3: 把后端回的 X-Trace-Id 写到 atom，给 ErrorBoundary /
// window.onerror 上报时带上（让开发者在 server log 能 join "前端崩前最后一次
// API 失败" 跟 "用户实际看到的 toast"）。
import { setLastApiTraceId } from '../lib/errors/report'
import i18n from '../i18n'

export interface HealthResponse {
  status: string
  version: string
}

export interface GpuStats {
  index: number
  name: string
  util_pct: number
  vram_used_gb: number
  vram_total_gb: number
  temp_c: number | null
}

export interface SystemStats {
  cpu_pct: number
  ram_used_gb: number
  ram_total_gb: number
  /** null = NVML 不可用 (无 NVIDIA / 驱动缺失)；[] = NVML 可用但 0 卡。两种都不显示 GPU pill。 */
  gpu: GpuStats[] | null
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
  /** 当此表达式为真时字段在 UI 上 disabled（值由 SchemaForm 自动回退到 default）。
   * 表达式语法与 show_when 一致：`key==value` / `key!=value`。
   * 例：lr_scheduler 在 optimizer_type=prodigy_plus_schedulefree 时被 disable。 */
  disable_when?: string
  /** disable_when 触发时写回的值；缺省回退到 default。 */
  disable_value?: unknown
  /** disable_when 触发时显示的提示徽章文本。 */
  disable_hint?: string
  /** 条件说明文字：当 alt_description_when 表达式为真时，替换 description 显示。 */
  alt_description?: string
  /** 触发 alt_description 的条件表达式，语法同 show_when。 */
  alt_description_when?: string
  /** 高级模式专属字段，简单模式下隐藏。 */
  advanced?: boolean
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
  /** 图片入库处理（booru 下载 / reg / 本地上传共用）。 */
  save_tags: boolean
  convert_to_png: boolean
  remove_alpha_channel: boolean
}

export interface RegConfig {
  /** 正则集生成全局默认排除 tag；进入某个 build 且无本地选择时作种子填充。 */
  default_excluded_tags: string[]
}

export interface HuggingFaceConfig {
  token: string
  /** PR-S3 — HF 模型下载端点 endpoint。
   *  `""` → huggingface_hub 默认（直连 huggingface.co）；海外用户推荐
   *  `"https://hf-mirror.com"` → 国内默认（项目主战场国内）
   *  其它 URL → 自定义反代 / 自建镜像 */
  endpoint: string
}

/** 一套 WandB 账号 + 上传策略预设（0.18 预设化，对齐 LLMPreset 模式）。 */
export interface WandBPreset {
  id: string
  label: string
  api_key: string
  project: string
  entity: string
  base_url: string
  mode: 'online' | 'offline' | 'disabled'
  /** 是否把训练采样图上传到 wandb.ai，默认开；私有 / NSFW 数据集请关掉。 */
  log_samples: boolean
  /** 上传前缩到最长边像素 */
  sample_max_side: number
  /** step 节流：>0 时只在 global_step % N == 0 上传，0 = 不额外节流 */
  sample_every_n_steps: number
  /** 上传模型 artifact 到 wandb */
  upload_model: boolean
  /** 模型 artifact 保留策略：all=全部版本 / last=仅最新 */
  upload_model_policy: 'all' | 'last'
  /** 上传手动保存的训练状态 artifact */
  upload_state_manual: boolean
  /** 手动状态 artifact 保留策略 */
  upload_state_manual_policy: 'all' | 'last'
  /** 上传自动保存的训练状态 artifact */
  upload_state_auto: boolean
  /** 自动状态 artifact 保留策略 */
  upload_state_auto_policy: 'all' | 'last'
}

/** 全局 WandB：顶层只留总开关 + 预设切换，字段全在 preset 里。 */
export interface WandBConfig {
  enabled: boolean
  current_preset: string
  presets: WandBPreset[]
}

export interface ModelScopeConfig {
  /** 魔搭社区 token。公开模型可不填；私有 / 限速时需要。 */
  token: string
}

export interface EvalMetricModelsConfig {
  /** CLIP-T / CLIP-I 默认模型名或本地目录。 */
  clip_model_name: string
  /** DINO-I 默认模型名或本地目录。 */
  dino_model_name: string
  /** CCIP（anime 角色身份）默认 ONNX 变体名。 */
  ccip_model_name: string
  /** 启用哪些评估指标（Settings 复选框）；eval 只算勾选的。 */
  enabled_metrics: string[]
  /** 训练后评估额外出一组纯底模(scale=0)对照，各指标给 Δ = checkpoint − baseline。 */
  eval_baseline_enabled: boolean
}

/** 评估指标 registry 条目（catalog.eval_metric_catalog）：Settings 复选框列表用。 */
export interface EvalMetricCatalogItem {
  key: string
  label: string
  runner: string
  models: string[]
  default: boolean
  desc: string
  note: string
}

export interface EvalMetricSpec {
  key: string
  label: string
  question: string
  requires: string[]
  higher_is_better: boolean
}

export interface EvalMetricState {
  key: string
  label?: string
  status: 'not_run' | 'pending' | 'running' | 'done' | 'failed' | 'unavailable' | string
  value: number | null
  reason?: string
  question?: string
  requires?: string[]
  higher_is_better?: boolean
  count?: number
  model_name?: string
  job_id?: number
}

export interface EvalMetricResult {
  schema_version: number
  has_metrics: boolean
  status: string
  run_id: string
  project_id?: number
  project_slug?: string
  version_id?: number
  version_label?: string
  created_at?: number | null
  updated_at?: number | null
  manifest_digest?: string
  checkpoint?: {
    kind?: string
    label?: string
    path?: string
    value?: number
    mtime?: number
  }
  metrics: Record<string, unknown>
  metric_states: Record<string, EvalMetricState>
  summary?: Record<string, number>
  /** 纯底模(lora_scale=0)对照 run；不作为 checkpoint 展示，只供算 Δ。 */
  baseline?: boolean
  /** 各指标相对 baseline 的净增益 Δ = checkpoint 值 − baseline 值。 */
  delta?: Record<string, number>
  /** baseline 各指标值（参考）。 */
  baseline_metrics?: Record<string, number>
  /** 出图阶段（eval_samples run.json）的状态 + 逐图汇总 {total, pending, running,
   *  done, failed}。出图是评估里最耗时的部分；用它显示「出图 done/total」子进度。 */
  sample_run?: {
    run_id: string
    path?: string
    status: string
    summary: Record<string, number>
    created_at?: number | null
    updated_at?: number | null
  }
}

/** 训练后 / 手动评估的一条 job（inline 训练时评估无 job）。按 run_id 关联到某个
 *  checkpoint 行，用来取原始日志（含报错）。 */
export interface EvalJobInfo {
  id: number
  kind: 'eval_samples' | 'eval_clip' | 'eval_dino' | string
  status: string
  run_id?: string | null
  checkpoint_path?: string | null
}

export interface EvalMetricsListResponse {
  metric_specs: EvalMetricSpec[]
  cache: {
    embeddings_dir: string
    entries: Array<{ key: string; path: string; file_count: number; size_bytes: number }>
  }
  results: EvalMetricResult[]
}

/** Preset messages 序列里的单条 item。
 *  - type='text'：普通文本，需指定 role；content 是 prompt 内容
 *  - type='image'：图片占位 item，打标时后端塞入当前图片；UI 不可编辑 content，但可拖动位置
 */
export interface LLMMessage {
  type: 'text' | 'image'
  role: 'system' | 'user' | 'assistant'
  content: string
}

/** 单个 LLM tagger preset = 一整套 endpoint + messages + 生成参数。
 *  builtin 仅标识 id 在内置列表（用于 UI 显示 "重置为默认"），不锁字段。
 */
export interface LLMPreset {
  id: string
  label: string
  builtin: boolean
  base_url: string
  api_key: string
  model: string
  model_ids: string[]
  endpoint: 'chat_completions' | 'responses'
  messages: LLMMessage[]
  output_format: 'json' | 'text'
  /** Local ONNX tagger used to pre-tag images and inject {{tags}} into messages ('' = off). */
  assist_tagger: string
  temperature: number
  max_tokens: number
  max_side: number
  jpeg_quality: number
  max_image_mb: number
  timeout: number
  max_retries: number
  concurrency: number
  requests_per_second: number
  max_requests_per_minute: number
}

export interface LLMTaggerConfig {
  current_preset: string
  presets: LLMPreset[]
}

export interface LLMConnectionTestResult {
  ok: boolean
  endpoint: LLMPreset['endpoint']
  endpoint_url: string
  model: string
  elapsed_ms: number
  status_code: number | null
  response_preview: string
  error: string
  request_shape: string
}

export interface WD14Config {
  model_id: string
  /** 候选模型列表；用户在「设置 → WD14」里维护，model_id 必属于该列表。 */
  model_ids: string[]
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
  threshold_general: number
  threshold_character: number
  add_copyright_tag: boolean
  add_artist_tag: boolean
  add_meta_tag: boolean
  add_model_tag: boolean
  add_rating_tag: boolean
  add_quality_tag: boolean
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
  /** 'cu128' / 'cu130' = CUDA 版 torch；'cpu' = CPU 版（装不了 flash_attn）；
   *  null = torch 未装 / 检测失败。UI 用 'cpu' 触发「先重装 CUDA 版」提示。 */
  torch_cuda_build: string | null
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

/** onnxruntime 装包状态 + nvidia-smi 检测 + 平台标识（前端用来按平台 disable 按钮）。 */
export interface WD14Runtime {
  installed: 'onnxruntime' | 'onnxruntime-gpu' | 'onnxruntime-directml' | null
  version: string | null
  providers: string[]
  cuda_available: boolean
  /** DirectML EP 可用（Windows + 装了 onnxruntime-directml 时为 true）。 */
  directml_available: boolean
  /** 后端 sys.platform：'win32' / 'linux' / 'darwin' 等。Settings UI 据此 disable
   *  跨平台不可用的按钮（DirectML 仅 Windows；GPU + nvidia-* wheel 仅 Linux 最优）。 */
  platform: string
  /** 装的包（dist-info）与当前进程已 import 的 .pyd 不一致 → 需重启 Studio。 */
  restart_required: boolean
  /** PP9.5 — InferenceSession 创建时实际 dlopen 报的错（如缺 libcurand.so.10）；
   *  非 null 表示已自动降级到 CPU EP，UI 应提示用户装 CUDA 库或换 DirectML。 */
  cuda_load_error: string | null
  /** torch 的 CUDA 大版本（onnxruntime-gpu build 锚点）：12 / 13 / null。
   *  装的 ORT build 必须同 major，否则 import 期 dlopen 挂（cu128 torch → 12）。 */
  torch_cuda_major?: number | null
  /** 已装 ORT 的 CUDA 大版本与 torch 不一致（如装成 cu13 但 torch 是 cu12）。 */
  ort_cuda_major_mismatch?: boolean
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
  /** fork 预设到 version 时是否自动用全局模型路径覆盖 4 个模型字段。
   * ON（默认）：多数用户场景，4 字段在 UI 上 disabled；fork 始终用 Settings 全局。
   * OFF：独立模型用户，fork 尊重预设值，4 字段可编辑 + picker。 */
  auto_sync_paths: boolean
  /** 训练模型根目录；null/空 → 回退 REPO_ROOT/models/（云端机改这里） */
  root: string | null
  /** 当前默认主模型：官方 variant key（1.0 / preview3-base / ...）或
   * custom_anima_paths 里的某个本地 .safetensors 路径。
   * Studio 创建新 version 时把它展开成绝对路径写到 yaml.transformer_path；
   * 已存在 version 不动（保证训练重现性）。 */
  selected_anima: string
  /** 用户注册的本地 custom 主模型（.safetensors 绝对路径）。微调训练 /
   * 在微调权重上测试出图用；仅登记路径，不下载不复制。 */
  custom_anima_paths: string[]
  /** 预处理默认放大器：预设 label（"4x-AnimeSharp" 等）或 custom 文件名
   * （"my-anime.pth"）。Preprocess 页和 worker 用它定权重路径。 */
  selected_upscaler: string
}

export interface QueueConfig {
  /** R-1 资源档位：exclusive（训练/正则 AI/出图/评估出图）运行时是否放行
   *  light 档（打标/超分/正则构建/评估指标，小模型）。默认 true。独占档
   *  永不并行，不受此开关影响。 */
  light_tasks_during_train: boolean
}

/** Phase 2 commit 14 — 测试出图 daemon 行为。 */
export interface GenerateSecretsConfig {
  /** TAEFlux 中间步预览节流。0=关；>0 → daemon 每 N 步推 256px JPEG。
   * 模型缺失时 daemon 静默回退（无预览不影响出图）。 */
  preview_every_n_steps: number
  /** 注意力后端默认值（design 决策：用户配置一次，不每次出图都改）。
   * Generate 页 enqueue 自动注入；Settings 训练 tab 切换。 */
  attention_backend: AttentionBackend
  /** 测试出图 VAE decode 精度。bf16（默认）对齐 ComfyUI 现代 GPU 的 auto
   * VAE dtype；fp32 全精度（decode 前 daemon 临时 offload DiT/Qwen 腾显存）。 */
  vae_precision: 'bf16' | 'fp32'
  /** 测试出图 daemon 闲置 N 分钟自动卸载模型释放 VRAM。0 = 关闭，模型常驻
   * 直到手动点"清理显存"。计时只在 idle + 模型 loaded 时跑。 */
  idle_timeout_minutes: number
  /** 开后每次出图自动落盘到 studio_data/test/<date>/{single,xy}/image_N.png。
   * 默认关；compare 模式始终不落盘。 */
  save_test_images: boolean
}

/** 系统级偏好（ADR 0002 / 0005）。update_channel 是用户视图偏好（"stable" /
 *  "dev"），与 git 工作树状态解耦：toggle 切换不触发 git 操作，仅改 UI 展示
 *  的通道；真正"切到 dev HEAD" / "更新到 vX.Y.Z" 是单独按钮。
 *  show_dev_channel 是 deprecated 字段（pydantic 兼容），新代码用 update_channel。 */
export interface SystemPrefsConfig {
  update_channel: 'stable' | 'dev'
  /** @deprecated use update_channel */
  show_dev_channel: boolean
}

export interface ProxyConfig {
    enabled: boolean;
    http_proxy: string;
    https_proxy: string;
    no_proxy: string;
}

/** Tag 翻译词典 — meta 字段。kind=default：来自首启自动下载或用户点 "恢复默认"；
 *  kind=user：用户手动上传。前端 Settings UI 用 source_name / entry_count 显示。 */
export interface TagDictionaryMeta {
  source_name: string
  source_url: string
  entry_count: number
  downloaded_at: number
  kind: 'default' | 'user'
}

export interface TagDictionaryMetaResponse {
  loaded: boolean
  meta: TagDictionaryMeta | null
}

export interface TagDictionaryPayload {
  entries: Record<string, string[]>
  meta: TagDictionaryMeta
}

export interface Secrets {
  gelbooru: GelbooruConfig
  danbooru: DanbooruConfig
  download: DownloadGlobalConfig
  reg: RegConfig
  huggingface: HuggingFaceConfig
  wandb: WandBConfig
  modelscope: ModelScopeConfig
  eval_metrics: EvalMetricModelsConfig
  /** 旧的全局下载源（已退役为迁移种子，无 UI）。新模型按类型在 download_sources 里各自选。 */
  download_source: string
  /** 按类型下载源：{training|wd14|upscaler: 'huggingface'|'modelscope'}。固定 HF 的类型不在内。 */
  download_sources: Record<string, string>
  // JoyCaption 已合并为 llm_tagger 的 builtin preset
  llm_tagger: LLMTaggerConfig
  wd14: WD14Config
  cltagger: CLTaggerConfig
  models: ModelsConfig
  queue: QueueConfig
  generate: GenerateSecretsConfig
  system: SystemPrefsConfig
  proxy: ProxyConfig
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

/** 用户注册的本地 custom 主模型（PathPicker 选盘上已有的 .safetensors）。 */
export interface CustomAnimaInfo extends ModelFileStatus {
  /** 注册的绝对路径（也是选中时写入 selected_anima 的值）。 */
  path: string
  /** 文件名，列表展示用。 */
  name: string
}

export interface AnimaMainCatalog {
  id: 'anima_main'
  name: string
  description: string
  repo: string
  variants: AnimaVariantInfo[]
  /** 本地注册的 custom 主模型列表。 */
  custom: CustomAnimaInfo[]
  /** 当前选中的主模型：variant key 或 custom 路径。 */
  selected: string
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
  model_id: string
  model_path: string
  tag_mapping_path: string
  description?: string
  is_current: boolean
  target_path?: string
  version_dir?: string
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

export interface EvalVariantInfo {
  kind: 'clip' | 'dino'
  model_id: string
  target_path: string
  exists: boolean
  size: number
  /** 下载前的预估大小（bytes）；未知 model_id 为 0。 */
  size_estimate: number
}

export interface EvalMetricsCatalog {
  id: 'eval_metrics'
  name: string
  description: string
  variants: EvalVariantInfo[]
}

export interface ModelDownloadStatus {
  key: string
  status: 'pending' | 'running' | 'done' | 'failed'
  started_at: number
  finished_at: number | null
  message: string
  log_tail: string[]
}

export interface UpscalerVariant {
  label: string
  filename: string
  kind: 'preset' | 'custom'
  hf_repo: string | null
  ms_repo: string | null
  size_mb: number | null
  description: string
  target_path: string
  is_current: boolean
  exists: boolean
  size: number
  mtime: number
  /** @deprecated 兼容老 build，新代码用 hf_repo/ms_repo */
  repo?: string
}
export interface UpscalersCatalog {
  id: 'upscalers'
  name: string
  description: string
  default: string
  /** 当前选中的放大器（来自 secrets.models.selected_upscaler，回退 default） */
  current: string
  target_dir: string
  variants: UpscalerVariant[]
}

export interface ModelsCatalog {
  models_root: string
  anima_main: AnimaMainCatalog
  anima_vae: AnimaVaeCatalog
  qwen3: ModelDirCatalog
  t5_tokenizer: ModelDirCatalog
  wd14: WD14Catalog
  cltagger: CLTaggerCatalog
  eval_metrics?: EvalMetricsCatalog
  /** 评估指标 registry（Settings 复选框列表）。 */
  eval_metric_catalog?: EvalMetricCatalogItem[]
  upscalers?: UpscalersCatalog
  /** 按类型的下载源选项：current = 当前选中，available = 可选源（长度 1 = 固定单源）。 */
  download_source_options: Record<string, { current: string; available: string[] }>
  downloads: Record<string, ModelDownloadStatus>
}

// ---- projects / versions (PP1) -------------------------------------------

// ADR-0007 PR-5: 老 ProjectStage / VersionStage 已删（DB 列也由 v9 destructive 删）。
// 用 VersionStatus + VersionPhase 替代。

/** ADR-0007 §11.3-B 新模型：version 运行态状态机（5 enum）。 */
export type VersionStatus =
  | 'preparing'
  | 'training'
  | 'completed'
  | 'failed'
  | 'canceled'

/** ADR-0007 §11.3-B 新模型：version 准备 cursor（仅 status=preparing 时有意义）。
 *  按 PHASE_ORDER 顺序：curating → preprocessing → tagging → editing →
 *  regularizing → ready（ADR 0010 amendment 加 preprocessing）。 */
export type VersionPhase =
  | 'curating'
  | 'preprocessing'
  | 'tagging'
  | 'editing'
  | 'regularizing'
  | 'ready'

export const PHASE_ORDER: VersionPhase[] = [
  'curating', 'preprocessing', 'tagging', 'editing', 'regularizing', 'ready',
]

export const PHASE_SKIPPABLE: VersionPhase[] = ['preprocessing', 'regularizing']

/** ADR-0007 §11.5-A: advance / skip phase endpoint response。 */
export interface PhaseAdvanceResult {
  advanced: boolean
  ok: boolean
  reason: string
  new_phase: VersionPhase | null
  version: Version | null
}

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
  /** ADR-0007 §11.3-B: 运行态主状态机（5 enum）。 */
  status: VersionStatus
  /** ADR-0007 §11.3-B: phase cursor，仅 status=preparing 时有意义。 */
  phase: VersionPhase
  last_failure_reason: string | null
  created_at: number
  output_lora_path: string | null
  note: string | null
  /** 触发词；由 Step 4 (Tagging) 写入，打标时 prepend 到每张 caption；空串=未启用。 */
  trigger_word: string
  stats?: VersionStats
}

export interface ProjectSummary {
  id: number
  slug: string
  title: string
  active_version_id: number | null
  /** ADR-0007 §11.8-E: 项目卡片右上角 status badge / 卡片显 version 名（list 端点 enrich）。 */
  active_version_label: string | null
  active_version_status: VersionStatus | null
  /** v12: preparing 时的 phase cursor（badge 显示"准备中 · 打标"）；无 active version 为 null。 */
  active_version_phase: VersionPhase | null
  created_at: number
  updated_at: number
  /** v12: 非 null = 已归档（软隐藏）。list 归档/活跃都返回，切分在前端。 */
  archived_at: number | null
  note: string | null
  download_image_count?: number
  preprocess_image_count?: number
}

export interface ProjectDetail extends ProjectSummary {
  versions: Version[]
  download_image_count: number
  preprocess_image_count: number
}

// ---- jobs (PP2) -----------------------------------------------------------

export type JobStatus = 'pending' | 'running' | 'done' | 'failed' | 'canceled'
export type JobKind =
  | 'download' | 'preprocess' | 'tag' | 'reg_build'
  | 'eval_samples' | 'eval_clip' | 'eval_dino' | 'eval_tag' | 'eval_ccip'

export interface Job {
  id: number
  project_id: number
  version_id: number | null
  kind: JobKind
  params: string
  params_decoded?: Record<string, unknown> | null
  status: JobStatus
  /** v16 — 入队时间；老作业 NULL（入队时刻未记录，UI 显示 —）。 */
  created_at?: number | null
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

export interface DataExportItem {
  filename: string
  path: string
  size: number
  mtime: number
}

export interface BundleImportResult {
  project: ProjectDetail
  version: Version
  stats: {
    train_image_count: number
    train_tagged_count: number
    reg_image_count: number
    preset_count: number
  }
}

// ---- preprocess (ADR 0010 train scope) -----------------------------------

/** 裁剪页工作集一项（train scope，rel path 形式）：name + 像素尺寸 + 是否已处理。 */
export interface CropWorkspaceItem {
  name: string
  /** download/ 下原图名（origin）；下游还原走这个名。 */
  source: string
  w: number
  h: number
  mtime: number
  size: number
  processed: boolean
}

/** 总览页「已删除」tab 一项：被去重审核标记的 entry。物理图仍在 download/{source}。 */
export interface DuplicateRemovedItem {
  /** manifest entry 的 key（一般 == source）。restore 时按这个名传。 */
  name: string
  /** download/ 下原图名（origin）。缩略图按 source + bucket=download 取。 */
  source: string
  /** 像素尺寸 — origin 文件不存在时 null。 */
  w: number | null
  h: number | null
  mtime: number
  size: number
}

// ---- ADR 0010 train-scope types -----------------------------------------

/** ADR 0010 train scope: 列 versions/{label}/train/ 全部图 + manifest 元数据。
 *  替代老 `{processed, pending}` 双 list 概念——新模型下 train/ 即"训练集 grid"，
 *  状态从字段差异隐含推断（详 ADR 0010 §Manifest schema v2 + backend
 *  `_is_processed`：扩展名变 / `_cN` 后缀 / train size != download size）。 */
export interface TrainImage {
  /** POSIX rel path "{N_label}/{image}"（如 "1_data/X.png"）。 */
  name: string
  mtime: number
  size: number
  /** PIL 读图头；损坏 / 物理不存在 null。 */
  w: number | null
  h: number | null
  /** download/ 下原图名（无 sub-folder 结构）；restore 反查走这个。 */
  origin: string | null
  /** @deprecated 兼容字段；后端两个字段值相同。 */
  source: string | null
  /** download/{origin} 物理缺失（restore 会落 no_origin）。 */
  orphan: boolean
  /** 人工去重审核标记。UI 区分"训练参与" vs "审核跳过"。 */
  duplicate_removed: boolean
  /** ADR 0010 状态推断（backend `_is_processed`）：upscale / crop / 转码过的 train
   *  文件 → true；curate 时复制的原样副本 → false。UI 用这个画"已处理"徽章。 */
  processed: boolean
  /** 老 schema 透传字段（新 entry 一律 null；前端容忍）。 */
  model: string | null
  scale: number | null
  action: string | null
  target_area: number | null
  src_size: [number, number] | null
  dst_size: [number, number] | null
  elapsed_seconds: number | null
}

/** ADR 0010 §Restore 语义：restore 返三组：成功 / manifest 无 entry / download
 *  缺失。`no_origin` 给 UI 三选项 [拖入替换 / 保留 / 移除] 用。 */
export interface TrainRestoreResult {
  restored: string[]
  missing: string[]
  no_origin: string[]
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
  /** ADR 0010 fixup（2026-06-04）：train 区项目带 download 原图文件名（按
   *  train manifest entry.origin 反查；老项目无 manifest → fallback 用 name
   *  自身）。Curation 右侧 thumb 走 `download` bucket + 这个 origin，显示
   *  **预处理前的样子**——避免 multi-crop fan-out / 去重 / upscale 改字节
   *  让筛选页缩略图"位置移动"。预处理结果用 Preprocess Overview 看。
   *  left 区项目（download 候选）这个字段缺失/无意义。 */
  origin?: string
}

export interface CurationView {
  left: CurationItem[] // download − train − validation
  right: Record<string, CurationItem[]> // folder → items
  download_total: number
  train_total: number
  folders: string[]
}

/** held-out 验证集里的一张图：扁平列表（无文件夹概念），但带物理 `folder`
 *  供缩略图寻址（version thumb 的 validation bucket 需要）与精确删除。 */
export interface ValidationItem {
  name: string
  mtime: number
  folder: string
}

export interface CurationValidationView {
  left: CurationItem[] // download − train − validation（与训练集同候选池）
  right: ValidationItem[] // validation 全量扁平
  download_total: number
  val_total: number
}

export interface CopyResult {
  copied: string[]
  skipped: string[]
  missing: string[]
}

/** 去重扫描请求体。算法内部还有一批阈值/性能参数，但都已固化为后端常量，
 *  UI 只暴露这两项：
 *   - match_scope：只查全图重复，还是连同分镜差分/裁剪一起（both 才开裁剪检测）
 *   - sensitivity：差分/裁剪判定的松紧（驱动后端 variant_score + crop_score） */
export interface DuplicateScanOptions {
  match_scope: 'strict' | 'both'
  sensitivity: 'loose' | 'standard' | 'strict'
}

export interface DuplicateMetrics {
  score: number
  match_type: 'keep' | 'strict-duplicate' | 'same-scene-variant' | 'linked-indirectly' | string
  structure_diff: number
  phash_diff: number
  soft_phash_diff: number
  dhash_diff: number
  ahash_diff: number
  edge_diff: number
  color_diff: number
  tile_median: number
  tile_mean: number
  tile_close_ratio: number
  gray_diff: number
  gray_close_ratio: number
  aspect_delta: number
  note: string
}

export interface DuplicateItem {
  name: string
  keep: boolean
  width: number
  height: number
  filesize_kb: number
  metrics: DuplicateMetrics | null
}

export interface DuplicateGroup {
  group_id: number
  keep: string
  items: DuplicateItem[]
  best: DuplicateMetrics | null
}

export interface DuplicateScanResult {
  target: 'preprocess' | 'download'
  match_scope: DuplicateScanOptions['match_scope']
  total_images: number
  readable_images: number
  group_count: number
  candidate_count: number
  crop_relation_count: number
  elapsed_seconds: number
  stats: {
    total_pairs: number
    aspect_skipped_pairs: number
    prefiltered_pairs: number
    compared_pairs: number
  }
  groups: DuplicateGroup[]
}

export interface DuplicateApplyResult {
  removed: string[]
  missing: string[]
  skipped: string[]
}

// ---- tagging (PP4) --------------------------------------------------------

export type TaggerName = 'wd14' | 'cltagger' | 'joycaption' | 'llm'

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
  /** A3 — 实际跑过 auto_tag 的 tagger 名（"wd14" / "cltagger" / ...）；
   * null = 没跑 / 旧 meta 未带此字段。auto_tagged=true 但此字段为 null
   * 视作旧版本数据（未知 tagger）。 */
  auto_tag_kind?: string | null
  /** B1（PR-2）—— 该 reg 集生成时的 build_mode；老 meta 无此字段 → 后端
   * 默认填 'mirror'。前端 mode 切换拦截优先看这个；fallback 才靠 reg.files
   * 路径前缀推断。 */
  build_mode?: string
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
  /** fork preset 时后端将注入的项目预填值（项目路径 + 全局模型路径 + reg
   * 检测）。新建预设预览表单用它显示「保存后会得到的值」。无论 has_config
   * 与否都返回 —— 新建预设可以在 version 已有 config 的状态下被点（覆盖
   * 当前预设），所以这个 hint 跟 has_config 状态无关。 */
  project_specific_defaults?: ConfigData
  dropped_fields?: string[]
  defaulted_fields?: string[]
}

/** 训练集 ARB 桶分布（后端用真 BucketManager 算）。count = 有效样本数（含 repeat × fan-out）。 */
export interface BucketDistribution {
  resolutions: number[]
  aspect_ratio_limit: number
  groups: Array<{
    reso: number
    buckets: Array<{ w: number; h: number; count: number }>
  }>
}

export interface RegBuildRequest {
  excluded_tags?: string[]
  auto_tag?: boolean
  /** A3 — auto-tag 用的 tagger。当前 UI 只暴露 wd14 / cltagger；
   * 后端 422 校验同样收紧到这两个。 */
  auto_tag_kind?: 'wd14' | 'cltagger'
  api_source?: 'gelbooru' | 'danbooru'
  /** 默认 true（增量）—— 用户决策：避免开始生成时清掉昨天好不容易拉的图。
   * false = full：worker 入口先清 reg/（含 .deleted_ids.json）。 */
  incremental?: boolean
  /** A4 v2 — build 完后 worker 自动跑 dedup + 不够 incremental 补足循环，
   * 最多 3 轮，在分辨率聚类前。默认 true。 */
  auto_dedup?: boolean
  /** B1（PR-2）—— 构建模式：
   * - mirror：镜像 train 子文件夹（5_concept/、1_general/ ...），target_count 忽略
   * - flat：所有图进 1_data/ 单桶，target_count 决定总图数（null = train 总数）
   * 默认 flat；切换前提是 reg 集已清空（前端拦截）。 */
  build_mode?: 'mirror' | 'flat'
  /** B1（PR-2）—— flat 模式下目标图数；null = 用 train 总图数。 */
  target_count?: number | null
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
  /** 本次先验生成临时选用的底模（官方 variant key 或本地 custom 路径）；
   *  省略 → server 用 Settings 里的 selected_anima。 */
  base_model?: string
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
}

/** PR-9 — 测试出图（独立工具页，多 LoRA + multi-prompt）。 */
export interface LoraEntry {
  path: string
  scale: number
  /** 来自 picker 的项目 / 版本绑定；外部文件无 */
  project_id?: number | null
  version_id?: number | null
  /** 仅 placeholder 状态用：历史回填时 resolve 失败保留原 basename
   *  （如 "my-lora.safetensors"），让 SidebarLoras 渲染 ⚠ placeholder 卡片
   *  提示用户重选。`path` 非空时此字段被忽略；submit 时 path='' 的 entry
   *  会被 `.filter(l => l.path.trim())` 跳过，不影响 daemon。 */
  name?: string | null
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
  /** 本次出图临时选用的底模（官方 variant key 或本地 custom 路径）；
   *  省略 → server 用 Settings 里的 selected_anima。 */
  base_model?: string
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
  /** 前端构造的 GenerateParamsSnapshot dict，server 不解释结构、透传到
   *  daemon → image_done 时塞进加密 cache payload header。
   *  /api/generate/cache/index 时返还作为 CacheEntry.params 回填用。 */
  params_snapshot?: Record<string, unknown> | null
}

/** GET /api/generate/cache/index — 当前 session 加密磁盘 cache 索引。
 *  server 端 SessionCache 按 task_id 聚合返回；前端转成 CacheEntry。 */
export interface CacheGenerateHistoryEntry {
  /** "cache:<task_id>" */
  id: string
  taskId: number
  mode: 'single' | 'xy'
  /** Unix timestamp ms */
  createdAt: number
  /** 该 task 的所有文件名（XY 时按文件名排序） */
  filenames: string[]
  /** GenerateParamsSnapshot dict */
  params: Record<string, unknown>
  /** 仅 mode=xy 存在；列每张图的 xy 位置，PreviewXYGrid 重建网格用 */
  samples?: Array<{
    filename: string
    xy: { xi: number; yi: number; xv: string | number; yv: string | number | null }
  }>
}

/** 落盘测试图历史 entry（GET /api/generate/disk-history）。
 *  params 是 GenerateParamsSnapshot（前端用 paramsSnapshot.ts 的类型解读），
 *  这里用 unknown 让 api/client.ts 不依赖 pages 层类型。 */
export interface DiskGenerateHistoryEntry {
  /** 稳定 ID："disk:<date>:<mode>:image_<N>"；前端按此 dedup */
  id: string
  /** YYYY-MM-DD */
  date: string
  mode: 'single' | 'xy'
  filename: string
  /** 服务端绝对路径，用于和 IDB entry.diskPath 做 dedup */
  path: string
  /** /api/generate/disk-image/<date>/<mode>/<filename> */
  url: string
  /** Unix timestamp（sidecar 写入或 fallback 文件 mtime） */
  created_at: number
  schema_version: number
  /** sidecar 里的 params object（前端按 GenerateParamsSnapshot 解读） */
  params: Record<string, unknown>
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

export type TaskStatus =
  'pending' | 'running' | 'done' | 'failed' | 'canceled' | 'paused' | 'scheduled'

/** tasks.task_type 的合法值。R-3 台账合并起含九类数据作业 kind。
 *  档位：exclusive = train/reg_ai/generate/eval_samples；light = 其余；io = download。 */
export type TaskType =
  | 'train' | 'reg_ai' | 'generate'
  | 'download' | 'preprocess' | 'tag' | 'reg_build'
  | 'eval_samples' | 'eval_clip' | 'eval_dino' | 'eval_tag' | 'eval_ccip'

/** R-5 档位视图参数：GPU 视图 = exclusive，数据视图 = data（light+io）。 */
export type QueueResourceClass = 'exclusive' | 'data'

/** Terminal task statuses — UI 一般禁用这些上的操作按钮（cancel / pause 等）。
 *  `paused` **不**进 terminal — 它可被 resume 复活。 */
export const TERMINAL_TASK_STATUSES: ReadonlyArray<TaskStatus> = [
  'done', 'failed', 'canceled',
]

export interface Task {
  id: number
  name: string
  config_name: string
  /** 0.17 P-D — 后端权威任务类型（_v5 migration 加，值 train/reg_ai/generate）。
   *  老行经 `NOT NULL DEFAULT 'train'` 的 ALTER 自动 backfill；此处可选仅为兼容
   *  未带该字段的测试 mock，运行时恒有值。 */
  task_type?: TaskType
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
  /** ADR 0006 PR-2 — paused task 的 .pt 文件路径（pause_step_<N>.pt）。 */
  paused_state_path?: string | null
  /** ADR 0006 PR-2 — paused task 的 config snapshot 路径（pause_step_<N>.config.json）。 */
  paused_config_path?: string | null
  /** ADR 0006 PR-2 — paused 时的 global_step（UI "在 step N 暂停于 …" 显示）。 */
  paused_step?: number | null
  /** ADR 0006 PR-2 — paused 时间（unix 秒）。 */
  paused_at?: number | null
  /** 0.17 P-B — 计划开始时间（unix 秒）。status='scheduled' 时有值；到点提升为
   *  pending 后保留作记录。非计划任务恒 null。 */
  scheduled_at?: number | null
  /** R-2/_v17 — 数据作业类 task 的 kind 专属参数 JSON；train/reg_ai 恒 null。 */
  params?: string | null
  /** 后端读路径附带解码（同旧 jobs DAO 约定）。 */
  params_decoded?: Record<string, unknown> | null
  /** ADR 0006 PR-4 — is_pausable 信号（§8.1）：UI 用来决定是否显示暂停
   *  按钮。supervisor 跑得起来时由 server enrich；空载默认 false。 */
  is_pausable?: boolean
  /** ADR 0006 Addendum 2 — 最近一次 epoch 末 auto backup 的 .pt 路径
   *  （auto_epoch_state.pt，覆盖式单文件）。failed/canceled resume 的恢复点。 */
  last_state_path?: string | null
  /** ADR 0006 Addendum 2 — auto backup 配套 config snapshot 路径。 */
  last_config_path?: string | null
  /** ADR 0006 Addendum 2 — 备份点 epoch（UI "从 epoch N 继续" 提示）。 */
  last_state_epoch?: number | null
  /** ADR 0006 Addendum 2 — 备份点 global_step。 */
  last_state_step?: number | null
  /** ADR 0006 Addendum 2 — is_resumable 信号：status ∈ paused/failed/canceled
   *  且恢复点文件在盘上。UI 用来决定是否显示"继续训练"按钮。 */
  is_resumable?: boolean
}

/** 0.17 P-E — /api/queue?group=history 的分页响应。 */
export interface QueueHistoryPage {
  items: Task[]
  total: number
  page: number
  page_size: number
}

/** ADR 0006 PR-2 — GET /api/queue/hold 返回。`held=true` 时 UI 顶部
 *  banner sticky 显示；`pending_waiting` 是当前 pending 队列长度（提示用）。 */
export interface QueueHoldState {
  held: boolean
  pending_waiting: number
}

export interface LogResponse {
  task_id: number
  content: string
  size: number
}

/** /api/state — per-task monitor state written by the training process */
export interface MonitorState {
  task_id?: number
  project_id?: number
  project_slug?: string
  version_id?: number
  version_label?: string
  step?: number
  total_steps?: number
  epoch?: number
  total_epochs?: number
  speed?: number          // it/s
  start_time?: number     // unix seconds
  losses?: Array<{ step: number; loss: number }>
  lr_history?: Array<{ step: number; lr: number }>
  optimizer_metrics_history?: Array<{
    step: number
    lr?: number
    actual_lr?: number
    base_lr?: number
    effective_lr?: number
    d?: number
    d_min?: number
    d_max?: number
    actual_lr_min?: number
    actual_lr_max?: number
  }>
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
  path: string
  size: number
  mtime: number
  kind: 'lora' | 'training_state' | 'pause_state' | 'auto_epoch_state' | 'other'
  is_lora: boolean
}

export interface TaskOutputs {
  task_id: number
  output_dir: string | null
  exists: boolean
  /** 仅 loopback 请求为 true；云端永远 false。前端按此控制「打开文件夹」按钮可见性。 */
  supports_open_folder: boolean
  files: TaskOutputFile[]
  /** "{slug}-{label}"，用作打包下载的 zip 文件名前缀（和 train.zip 命名风格一致）。
   * 老任务没绑 project / version → null，调用方 fallback 到 task_{id}。 */
  archive_basename: string | null
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

export interface ImportResult {
  imported_count: number
  task_ids: number[]
  renamed: Record<string, string>
}

/**
 * API 错误：除了 `message`（用于直接 toast 的字符串），额外保留 `status` 和
 * `detail`（FastAPI 端 raise HTTPException(status, detail=dict(...)) 时
 * detail 是结构化对象，调用方可以 `e.detail.error` 区分类型）。
 *
 * 用 Error 而非自定义 class 是因为不少现有 callsite 是 `catch (e) { toast(String(e)) }`
 * 这种通用写法；保留 `Error.prototype.toString()` 行为不破坏它们。需要结构化
 * 处理的新 callsite 强制 cast：`(e as ApiError).detail`。
 *
 * ADR-0009 PR-3 C3: 新加 `traceId` 字段 — 后端 dual-write envelope 的
 * `body.error.trace_id` 或 X-Trace-Id response header。toast 显示 "trace ab12cd34"
 * 后缀让用户截图给开发；ErrorBoundary 上报时也带，串起前端崩前最后一次失败。
 */
export type ApiError = Error & {
  status?: number
  /** ADR-0009 Phase 2: 后端 body.error.code（语义错误码），前端按它查 errors.* i18n。 */
  code?: string
  detail?: unknown
  traceId?: string
}

/**
 * ADR-0009 Phase 2 统一错误解析：所有 fetch / XHR 失败路径共用，保证 toast 文案
 * 一致且可本地化。
 *
 * 优先 `body.error`：用 `error.code` 查 `errors.<code>` i18n（带 `error.details`
 * 插值，缺词条则回退 `error.message` 英文）。`body.detail` 退为 fallback —— 结构化
 * detail（如 409 冲突的 config/suggested_name）仍挂到 `err.detail` 给 callsite；
 * 没有 error 信封时（RequestValidationError 422 list / 极老路径）才用 detail 取文案。
 */
export function makeApiError(
  status: number,
  statusText: string,
  body: unknown,
  headerTraceId?: string | null,
): ApiError {
  let message = `${status} ${statusText}`
  let code: string | undefined
  let detail: unknown = null
  let traceId: string | undefined
  const b = body as {
    detail?: unknown
    error?: { code?: unknown; message?: unknown; trace_id?: unknown; details?: unknown }
  } | null | undefined
  const err = b?.error
  if (err && typeof err === 'object') {
    code = typeof err.code === 'string' ? err.code : undefined
    const enMsg =
      typeof err.message === 'string' && err.message ? err.message : message
    const params =
      err.details && typeof err.details === 'object'
        ? (err.details as Record<string, unknown>)
        : {}
    message = code ? i18n.t(`errors.${code}`, { ...params, defaultValue: enMsg }) : enMsg
    if (typeof err.trace_id === 'string') traceId = err.trace_id
    // 结构化数据现在挂在 error.details（如 409 冲突的 config/suggested_name、
    // running_tasks 列表），callsite 经 err.detail 读到。
    if (err.details && typeof err.details === 'object') detail = err.details
  }
  if (b && b.detail !== undefined) {
    if (!err) {
      if (typeof b.detail === 'string') {
        message = b.detail
      } else if (b.detail && typeof b.detail === 'object') {
        detail = b.detail
        const dm = (b.detail as { message?: unknown }).message
        if (typeof dm === 'string') message = dm
      }
    } else if (detail === null && b.detail && typeof b.detail === 'object') {
      detail = b.detail
    }
  }
  if (!traceId && headerTraceId) traceId = headerTraceId
  if (traceId) setLastApiTraceId(traceId)
  const e = new Error(message) as ApiError
  e.status = status
  e.code = code
  e.detail = detail
  e.traceId = traceId
  return e
}

/**
 * ADR-0009 PR-3 C3: 把 ApiError.traceId 末 8 字符格式化成 toast 后缀。
 *
 * 用户报问题时把 toast 截图给开发；开发拿这 8 字符 `jq 'select(.trace_id |
 * endswith("..."))' studio.log` 一行还原完整链路。
 *
 * 调用模式（callsite 自愿用，不强制 — 现有 toast(e.message,'error') 不破）：
 *     toast(`${e.message}${formatErrorTraceSuffix(e)}`, 'error')
 */
export function formatErrorTraceSuffix(err: unknown): string {
  const traceId = (err as ApiError | undefined)?.traceId
  if (!traceId) return ''
  return `  ·  trace ${traceId.slice(-8)}`
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
    const body = await resp.json().catch(() => null)
    throw makeApiError(resp.status, resp.statusText, body, resp.headers.get('X-Trace-Id'))
  }
  if (resp.status === 204) return undefined as T
  return (await resp.json()) as T
}

/** 上传进度事件 — 与 XMLHttpRequestEventTarget#progress 字段一一对应。 */
export interface UploadProgressEvent {
  loaded: number
  total: number
  /** total === 0 时为 false（服务端没回 Content-Length 或 chunked），ETA 无法计算。 */
  lengthComputable: boolean
}

/**
 * XHR-based multipart upload；fetch() 没有 request body progress 事件，所以
 * 上传进度必须走 XHR。错误格式跟 `req` 对齐（ApiError + 解析 detail）。
 */
async function xhrUpload<T>(
  url: string,
  body: FormData,
  onProgress?: (e: UploadProgressEvent) => void,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', url, true)
    xhr.responseType = 'text'
    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        onProgress({
          loaded: e.loaded,
          total: e.total,
          lengthComputable: e.lengthComputable,
        })
      }
    }
    xhr.onload = () => {
      const text = xhr.responseText
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(text ? (JSON.parse(text) as T) : (undefined as T))
        } catch {
          reject(new Error('invalid JSON response'))
        }
        return
      }
      let parsed: unknown = null
      try {
        parsed = JSON.parse(text)
      } catch {
        /* body 不是 JSON：makeApiError 用 statusText 兜底 */
      }
      reject(
        makeApiError(xhr.status, xhr.statusText, parsed, xhr.getResponseHeader('X-Trace-Id')),
      )
    }
    xhr.onerror = () => reject(new Error('network error'))
    xhr.send(body)
  })
}

/** studio_data 存储位置：当前/默认 + 全量扫描（迁移确认 modal 显示用）。 */
export interface StudioDataScanEntry {
  name: string
  is_dir: boolean
  files: number
  bytes: number
}

export interface StudioDataInfo {
  current: string
  default: string
  is_custom: boolean
  /** 请求带 scan=false 时为 null（Settings 页仅显示路径，免扫盘） */
  scan: {
    total_files: number
    total_bytes: number
    entries: StudioDataScanEntry[]
  } | null
}

/** 迁移状态快照（modal 重开 / SSE 漏事件兜底；实时进度走 SSE
 *  `studio_data_migrate_progress` / `_done` 事件）。 */
export interface StudioDataMigrateStatus {
  state: 'idle' | 'running' | 'done' | 'error'
  target: string
  total_files: number
  total_bytes: number
  done_files: number
  done_bytes: number
  current_file: string
  error: string
}

/** 模型根目录存储位置：和 studio_data 同结构（迁移确认 modal 复用展示）。 */
export interface ModelsRootInfo {
  current: string
  default: string
  is_custom: boolean
  /** 请求带 scan=false 时为 null（Settings 页仅显示路径，免扫盘） */
  scan: {
    total_files: number
    total_bytes: number
    entries: StudioDataScanEntry[]
  } | null
}

/** 模型根目录迁移状态快照（实时进度走 SSE `models_root_migrate_progress` / `_done`）。 */
export interface ModelsRootMigrateStatus {
  state: 'idle' | 'running' | 'done' | 'error'
  target: string
  total_files: number
  total_bytes: number
  done_files: number
  done_bytes: number
  current_file: string
  error: string
}

export interface AnnouncementPost {
  id: string
  date: string
  tag: 'release' | 'notice' | 'migration'
  title: { zh: string; en: string }
  body: { zh: string; en: string }
  pin: boolean
  version: string | null
}

export const api = {
  health: () => req<HealthResponse>('/api/health'),
  systemStats: () => req<SystemStats>('/api/system/stats'),
  state: () => req<Record<string, unknown>>('/api/state'),

  schema: () => req<SchemaResponse>('/api/schema'),

  // Presets (PP0+) -----------------------------------------------------
  listPresets: () =>
    req<{ items: PresetSummary[] }>('/api/presets').then((r) => r.items),
  getPreset: (name: string) => req<ConfigData>(`/api/presets/${name}`),
  getPresetWithWarnings: (name: string) =>
    req<{ config: ConfigData; dropped_fields: string[]; defaulted_fields: string[] }>(
      `/api/presets/${name}?warnings=true`,
    ),
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
  exportPresetToDataExports: (name: string, config: ConfigData) =>
    req<DataExportItem>(`/api/presets/${encodeURIComponent(name)}/export`, {
      method: 'POST',
      body: JSON.stringify({ config }),
    }),
  /** 端到端 yaml 文件下载直链，server FileResponse 已设 Content-Disposition。
   *  <a href={...} download> 触发即可，不发 fetch。 */
  presetDownloadUrl: (name: string) =>
    `/api/presets/${encodeURIComponent(name)}/download`,
  importPresetFromPath: (path: string) =>
    req<{ name: string; path: string }>('/api/presets/import-from-path', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),
  /** 端到端文件上传：把 .yaml/.yml/.json 文件给后端解析 + schema 校验 + 直接落盘,
   *  返回 {name, path}。前端拿到 name 直接 refreshList + setSelected(name) 即可。
   *
   *  冲突(同名 preset 已存在)→ 抛 ApiError(status=409),err.detail =
   *  {message, config, suggested_name},call site 据此弹 ImportConflictDialog
   *  让用户选覆盖 / 另存为,再走 PUT /api/presets/{name}。
   *  绕过 req() 的 JSON header,让浏览器自加 multipart boundary。 */
  importPreset: async (file: File): Promise<{ name: string; path: string }> => {
    const fd = new FormData()
    fd.append('file', file, file.name)
    const resp = await fetch('/api/presets/import', { method: 'POST', body: fd })
    if (!resp.ok) {
      const body = await resp.json().catch(() => null)
      throw makeApiError(resp.status, resp.statusText, body, resp.headers.get('X-Trace-Id'))
    }
    return (await resp.json()) as { name: string; path: string }
  },

  /** WandB preset yaml 下载直链（**含真实 api_key**，服务端显式导出端点）。
   *  <a href={...} download> 触发即可，不发 fetch。 */
  wandbPresetExportUrl: (id: string) =>
    `/api/secrets/wandb/presets/${encodeURIComponent(id)}/export`,
  /** 上传 yaml/json 导入 wandb preset；返回新 preset 标识 + 最新 masked secrets。 */
  importWandbPreset: async (
    file: File,
  ): Promise<{ id: string; label: string; secrets: Secrets }> => {
    const fd = new FormData()
    fd.append('file', file, file.name)
    const resp = await fetch('/api/secrets/wandb/presets/import', { method: 'POST', body: fd })
    if (!resp.ok) {
      const body = await resp.json().catch(() => null)
      throw makeApiError(resp.status, resp.statusText, body, resp.headers.get('X-Trace-Id'))
    }
    return (await resp.json()) as { id: string; label: string; secrets: Secrets }
  },

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

  // Tag dictionary -----------------------------------------------------
  /** 当前词典 meta + 是否已加载。Settings UI 启动时 ping，决定显示"未初始化"还是详情。 */
  getTagDictionaryMeta: () =>
    req<TagDictionaryMetaResponse>('/api/tag-dictionary/meta'),
  /** 完整 dict JSON (~600KB gzip)。store.ts 启动拉一次后缓存内存。 */
  getTagDictionaryData: () =>
    req<TagDictionaryPayload>('/api/tag-dictionary/data'),
  /** 上传 csv/txt 替换当前词典。返回新 meta。 */
  uploadTagDictionary: async (file: File): Promise<TagDictionaryMetaResponse> => {
    const fd = new FormData()
    fd.append('file', file, file.name)
    const resp = await fetch('/api/tag-dictionary/upload', { method: 'POST', body: fd })
    if (!resp.ok) {
      const body = await resp.json().catch(() => null)
      throw makeApiError(resp.status, resp.statusText, body, resp.headers.get('X-Trace-Id'))
    }
    return (await resp.json()) as TagDictionaryMetaResponse
  },
  /** 重新从 GitHub 拉默认词典（首次失败 / 用户想重置都走这个）。 */
  resetTagDictionary: () =>
    req<TagDictionaryMetaResponse>('/api/tag-dictionary/reset', { method: 'POST' }),

  // Models management (PP7) ------------------------------------------------
  getModelsCatalog: () => req<ModelsCatalog>('/api/models/catalog'),
  /** 当前 Settings 算出的 4 个模型字段绝对路径。预设页 reset / 新建用。 */
  getModelPathDefaults: () => req<Record<string, string>>('/api/models/path-defaults'),
  startModelDownload: (body: { model_id: string; variant?: string }) =>
    req<{ key: string; status: string }>('/api/models/download', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  /** 注册一个本地 .safetensors 主模型（微调训练 / 微调上测试）。返回新 catalog。 */
  addCustomAnima: (path: string) =>
    req<ModelsCatalog>('/api/models/anima/custom', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),
  /** 注销一个本地 custom 主模型。返回新 catalog。 */
  removeCustomAnima: (path: string) =>
    req<ModelsCatalog>('/api/models/anima/custom', {
      method: 'DELETE',
      body: JSON.stringify({ path }),
    }),
  startUpscalerCustomDownload: (body: {
    source: 'hf' | 'ms'
    repo_id: string
    filename: string
  }) =>
    req<{ key: string; status: string }>('/api/upscalers/download_custom', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  selectUpscaler: (label: string) =>
    req<{ selected: string }>('/api/upscalers/select', {
      method: 'POST',
      body: JSON.stringify({ label }),
    }),
  refreshLLMModels: (body: {
    preset_id?: string
    base_url?: string
    api_key?: string
    timeout?: number
  }) =>
    req<{ items: string[]; preset_id: string; secrets: Secrets }>('/api/llm-tagger/models/refresh', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  testLLMConnection: (
    body:
      & { preset_id?: string }
      & Partial<Pick<LLMPreset, 'base_url' | 'api_key' | 'model' | 'endpoint' | 'timeout' | 'max_tokens' | 'temperature'>>,
  ) =>
    req<LLMConnectionTestResult>('/api/llm-tagger/test', {
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
      active_version_id: number | null
    }>
  ) =>
    req<ProjectDetail>(`/api/projects/${pid}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  deleteProject: (pid: number) =>
    req<{ deleted: number }>(`/api/projects/${pid}`, { method: 'DELETE' }),
  /** 归档（软隐藏，可逆）：目录 / versions / 任务全部原样。 */
  archiveProject: (pid: number) =>
    req<ProjectDetail>(`/api/projects/${pid}/archive`, { method: 'POST' }),
  unarchiveProject: (pid: number) =>
    req<ProjectDetail>(`/api/projects/${pid}/unarchive`, { method: 'POST' }),

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
      status: VersionStatus
      phase: VersionPhase
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
    req<{ active_version_id: number }>(
      `/api/projects/${pid}/versions/${vid}/activate`,
      { method: 'POST' }
    ),

  // Phase cursor 推进 / 跳过 (ADR-0007 §11.5-A) --------------------------
  advanceVersionPhase: (pid: number, vid: number) =>
    req<PhaseAdvanceResult>(
      `/api/projects/${pid}/versions/${vid}/advance-phase`,
      { method: 'POST' }
    ),

  skipVersionPhase: (pid: number, vid: number) =>
    req<PhaseAdvanceResult>(
      `/api/projects/${pid}/versions/${vid}/skip-phase`,
      { method: 'POST' }
    ),

  // Task config snapshot (ADR-0007 §11.7) --------------------------------
  getTaskSnapshotConfig: (taskId: number) =>
    req<{ yaml: string; config: Record<string, unknown> }>(
      `/api/queue/${taskId}/snapshot/config`
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
   * 本地上传：单图（jpg/png）或 zip 包。走 XHR 以拿到 upload progress 事件
   * （fetch 没有 request body progress）。后端同步解 zip / 落盘，所以
   * progress 到 100% 后还有一段 server processing 时间。
   */
  uploadProjectFiles: (
    pid: number,
    files: File[],
    onProgress?: (e: UploadProgressEvent) => void,
  ): Promise<UploadResult> => {
    const fd = new FormData()
    for (const f of files) fd.append('files', f, f.name)
    return xhrUpload<UploadResult>(`/api/projects/${pid}/upload`, fd, onProgress)
  },
  uploadProjectFileFromPath: (pid: number, path: string) =>
    req<UploadResult>(`/api/projects/${pid}/upload-from-path`, {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),
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
  /** `v`：文件 mtime（unix s），仅用作浏览器端 cache-buster。**服务端忽略**该参数
   *  （后端 cache key 仍按 src+mtime+size 计算）；目的是让 in-place 覆盖后的图
   *  （裁剪 / 放大同名输出）URL 变化，浏览器不再命中 memory image cache 复用旧
   *  decoded 像素。`Cache-Control: no-cache` 对 disk cache 强制 revalidate，
   *  但 CSS `background-image` 的 in-memory decoded image 不受其约束，必须
   *  靠 URL 唯一性来失效 — 见 PreprocessCrop bug 修复。 */
  projectThumbUrl: (
    pid: number,
    name: string,
    bucket = 'download',
    size = 256,
    v?: number,
    /** raw=true（仅 bucket=download 有效）：跳过 resolve_origin，强制 download/{name}
     *  原始字节。给「对比预览」左 pane 用 —— 不能被 preprocess 派生 hijack。 */
    raw?: boolean,
  ) =>
    `/api/projects/${pid}/thumb?bucket=${encodeURIComponent(bucket)}&name=${encodeURIComponent(name)}&size=${size}`
    + (v ? `&v=${v}` : '')
    + (raw ? '&raw=1' : ''),

  // ---- ADR 0010 train-scope endpoints -----------------------------------
  // PR-3 加；PR-4 前端切到这套；后续 PR-5 删老的 (`/preprocess/*` without vid)。
  startPreprocessTrain: (
    pid: number,
    vid: number,
    body: {
      mode: 'all' | 'selected' | 'all_force'
      names?: string[]
      model?: string
      tile_size?: number
      tile_pad?: number
      device?: 'auto' | 'cuda' | 'cpu'
      target_area?: number | null
    },
  ) =>
    req<Job>(`/api/projects/${pid}/versions/${vid}/preprocess/start`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  getPreprocessStatusTrain: (pid: number, vid: number) =>
    req<{
      job: Job | null
      log_tail: string
      summary: { image_count: number }
    }>(`/api/projects/${pid}/versions/${vid}/preprocess/status`),
  listPreprocessFilesTrain: (pid: number, vid: number) =>
    req<{
      images: TrainImage[]
      summary: { image_count: number }
    }>(`/api/projects/${pid}/versions/${vid}/preprocess/files`),
  /** ADR 0010 §Restore: 从 download/{entry.origin} 复制覆盖回 train/{name}；
   *  download 缺失返 `no_origin` 列表（UI 给三选项 [拖入替换 / 保留 / 移除]）。 */
  restorePreprocessFilesTrain: (pid: number, vid: number, names: string[]) =>
    req<TrainRestoreResult>(
      `/api/projects/${pid}/versions/${vid}/preprocess/files/restore`,
      { method: 'POST', body: JSON.stringify({ names }) },
    ),
  /** 只清 train manifest，**不动** train/ 物理文件（train 是训练数据本身）。 */
  resetPreprocessFilesTrain: (pid: number, vid: number) =>
    req<{ ok: boolean }>(
      `/api/projects/${pid}/versions/${vid}/preprocess/files/reset`,
      { method: 'POST' },
    ),
  listCropWorkspaceTrain: (pid: number, vid: number) =>
    req<{ images: CropWorkspaceItem[] }>(
      `/api/projects/${pid}/versions/${vid}/preprocess/crop/workspace`,
    ),
  listPreprocessDuplicatesRemovedTrain: (pid: number, vid: number) =>
    req<{ images: DuplicateRemovedItem[] }>(
      `/api/projects/${pid}/versions/${vid}/preprocess/duplicates/removed`,
    ),
  startPreprocessCropTrain: (
    pid: number,
    vid: number,
    crops: Record<string, { x: number; y: number; w: number; h: number; label?: string }[]>,
  ) =>
    req<Job>(`/api/projects/${pid}/versions/${vid}/preprocess/crop`, {
      method: 'POST',
      body: JSON.stringify({ crops }),
    }),

  // R-5 台账合并：/api/jobs* 已删，作业与任务同源 /api/queue（单一 ID 空间）。
  // getJob / cancelJob 保留函数名给步骤页（Download/Tagging/Reg/Preprocess），
  // 内部改指 /api/queue；kind 由 task_type 派生。
  getJob: (jid: number) =>
    req<Task & { kind?: JobKind }>(`/api/queue/${jid}`).then(
      (t) => ({ ...t, kind: (t.task_type ?? 'train') as JobKind }) as unknown as Job,
    ),
  cancelJob: (jid: number) =>
    req<{ task_id: number; canceled: boolean }>(`/api/queue/${jid}/cancel`, {
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
      /**
       * 已有 caption 文件时的策略：overwrite（默认覆盖）/ skip（保留原文件）
       * / append（tag 级 merge + dedupe 后写回原格式）。
       * 落盘格式跟着产物走（LLM json preset → .json，其余 → .txt），不再由请求指定。
       */
      on_existing?: 'overwrite' | 'skip' | 'append'
      /**
       * wd14 本次任务的临时覆盖；仅在 worker 进程生效，不写回 settings。
       * 字段为 undefined / null 时沿用全局 settings。
       */
      wd14_overrides?: {
        threshold_general?: number | null
        threshold_character?: number | null
        model_id?: string | null
        blacklist_tags?: string[] | null
      }
      cltagger_overrides?: {
        threshold_general?: number | null
        threshold_character?: number | null
        model_id?: string | null
        model_path?: string | null
        tag_mapping_path?: string | null
        add_copyright_tag?: boolean | null
        add_meta_tag?: boolean | null
        add_model_tag?: boolean | null
        add_rating_tag?: boolean | null
        add_quality_tag?: boolean | null
        blacklist_tags?: string[] | null
      }
      // current_preset 切换 active preset；其他字段覆盖 preset 同名字段。
      // api_key / model_ids / id / label / builtin 不允许 override。
      // PR #34 (P0-2) 的 `_output_format` 被本次重构吸收 — preset 自己有 output_format 字段。
      llm_overrides?:
        & { current_preset?: string }
        & Partial<Omit<LLMPreset, 'id' | 'label' | 'builtin' | 'api_key' | 'model_ids'>>
      /**
       * 触发词；空串 / undefined = 不启用。worker 端写 caption 时 prepend 为
       * 第一个 tag，并同步落库到 version.trigger_word，后续 train 读出。
       */
      trigger_word?: string
      /**
       * 打标范围：'all'（默认，train 全部 + validation）/ 'validation'（只打
       * held-out 验证集）/ 某个 train 子文件夹名（只打那一个）。
       */
      scope?: string
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
  /** A1 — 批量删 reg 集中的指定图片（含同名 .txt）。
   * `relative_paths` 是相对 reg/ 的路径列表，跨子文件夹可。
   * 后端自动把删除的 booru ID 追加到 reg/.deleted_ids.json，
   * 下次 incremental build 时自动排除。 */
  deleteRegFiles: (pid: number, vid: number, relative_paths: string[]) =>
    req<{ deleted: string[]; count: number }>(
      `/api/projects/${pid}/versions/${vid}/reg/delete-files`,
      { method: 'POST', body: JSON.stringify({ relative_paths }) }
    ),
  /** A4 — 用 preprocess dedup 默认参数扫一遍 reg 集，自动删除每组建议删除项
   * （不弹 review panel，"推荐删除"直接删；reg 集 quality bar 比 train 低）。
   * 同步返回 — 大集会慢；前端要 disable 按钮 + spinner。 */
  dedupPurgeReg: (pid: number, vid: number) =>
    req<{ scanned: number; groups: number; deleted: string[]; count: number }>(
      `/api/projects/${pid}/versions/${vid}/reg/dedup-purge`,
      { method: 'POST' }
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
  /** 回放最近一次先验生成 task + 日志，用于切页面/刷新后的日志恢复。 */
  getLatestRegPriorTask: (pid: number, vid: number) =>
    req<{ task: Task | null; log: string }>(
      `/api/projects/${pid}/versions/${vid}/reg/generate-prior/latest`,
    ),
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
  /** 落盘历史：扫 studio_data/test/&lt;date&gt;/{single,xy}/image_N.json sidecar，
   *  按 created_at desc 返回；用于历史栏跨会话回看已落盘的测试图。
   *  注意路径用 disk/ 子前缀避开 `/api/generate/{task_id}` 的单段 catch-all。 */
  listDiskGenerateHistory: (limit = 500) =>
    req<{ entries: DiskGenerateHistoryEntry[] }>(`/api/generate/disk/history?limit=${limit}`),
  /** 当前 session 加密磁盘 cache 历史（save_test_images=false 时唯一来源）。
   *  server 重启 / SSE 断连 30s + LRU 后 entry 消失；刷新 / 切路由都拉这里。 */
  listCacheGenerateHistory: () =>
    req<{ entries: CacheGenerateHistoryEntry[] }>('/api/generate/cache/index'),
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
  /** daemon stderr ring buffer。since_seq>0 时只返增量。 */
  getDaemonLogs: (sinceSeq = 0, limit = 2000) =>
    req<{ entries: Array<{ ts: number; seq: number; line: string }>; next_seq: number }>(
      `/api/generate/daemon/logs?since_seq=${sinceSeq}&limit=${limit}`,
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
  getBucketDistribution: (pid: number, vid: number) =>
    req<BucketDistribution>(
      `/api/projects/${pid}/versions/${vid}/bucket-distribution`
    ),
  putVersionConfig: (pid: number, vid: number, data: ConfigData) =>
    req<{ has_config: true; config: ConfigData }>(
      `/api/projects/${pid}/versions/${vid}/config`,
      { method: 'PUT', body: JSON.stringify(data) }
    ),
  forkPresetForVersion: (pid: number, vid: number, name: string) =>
    req<{
      has_config: true
      config: ConfigData
      from_preset: string
      dropped_fields: string[]
      defaulted_fields: string[]
    }>(
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
  /** 0.17 P-B — scheduledAt（unix 秒）给了则建成 scheduled（计划任务），到点
   *  由 supervisor 提升为 pending；不给立即入队（原行为）。 */
  enqueueVersionTraining: (pid: number, vid: number, opts?: { scheduledAt?: number }) =>
    req<Task>(
      `/api/projects/${pid}/versions/${vid}/queue`,
      {
        method: 'POST',
        ...(opts?.scheduledAt != null
          ? { body: JSON.stringify({ scheduled_at: opts.scheduledAt }) }
          : {}),
      }
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
  // 验证集（held-out）手动维护——与 train curation 对称，右栏扁平无文件夹
  getCurationValidation: (pid: number, vid: number) =>
    req<CurationValidationView>(
      `/api/projects/${pid}/versions/${vid}/curation/validation`
    ),
  copyToValidation: (pid: number, vid: number, body: { files: string[] }) =>
    req<CopyResult>(
      `/api/projects/${pid}/versions/${vid}/curation/validation/copy`,
      { method: 'POST', body: JSON.stringify(body) }
    ),
  removeFromValidation: (
    pid: number,
    vid: number,
    body: { items: { folder: string; name: string }[] }
  ) =>
    req<{ removed: string[]; missing: string[] }>(
      `/api/projects/${pid}/versions/${vid}/curation/validation/remove`,
      { method: 'POST', body: JSON.stringify(body) }
    ),
  // ADR 0010 train scope duplicates
  scanDuplicatesTrain: (
    pid: number,
    vid: number,
    body: DuplicateScanOptions,
  ) =>
    req<DuplicateScanResult>(
      `/api/projects/${pid}/versions/${vid}/preprocess/duplicates/scan`,
      { method: 'POST', body: JSON.stringify(body) }
    ),
  applyDuplicateActionTrain: (
    pid: number,
    vid: number,
    body: { names: string[] },
  ) =>
    req<DuplicateApplyResult>(
      `/api/projects/${pid}/versions/${vid}/preprocess/duplicates/apply`,
      { method: 'POST', body: JSON.stringify(body) }
    ),
  versionThumbUrl: (
    pid: number,
    vid: number,
    bucket: 'train' | 'reg' | 'samples' | 'validation',
    name: string,
    folder?: string,
    size: number = 256
  ) => {
    const qs = new URLSearchParams({ bucket, name, size: String(size) })
    if (folder) qs.set('folder', folder)
    return `/api/projects/${pid}/versions/${vid}/thumb?${qs.toString()}`
  },

  // Queue --------------------------------------------------------------
  listQueue: (status?: TaskStatus, opts?: { includeGenerate?: boolean }) => {
    const params: string[] = []
    if (status) params.push(`status=${status}`)
    // /api/queue 默认隐藏 generate（测试出图）task，列表里不混淆 train slot；
    // 想看 generate 任务（如 Overview 的 "查看输出"）显式开关。
    if (opts?.includeGenerate) params.push('include_generate=true')
    const qs = params.length ? `?${params.join('&')}` : ''
    return req<{ items: Task[] }>(`/api/queue${qs}`).then((r) => r.items)
  },
  // 0.17 P-A/P-C —— 队列页分区数据源。live = 进行中 + 等待（running/paused/pending），
  // 不分页；q 搜 name/config_name。
  // 不分页；q 搜 name/config_name；type 按 task_type 过滤（0.17 P-F）。
  listQueueLive: (q?: string, type?: TaskType, resourceClass?: QueueResourceClass) => {
    const params = new URLSearchParams({ group: 'live' })
    if (q) params.set('q', q)
    if (type) params.set('types', type)
    if (resourceClass) params.set('resource_class', resourceClass)
    return req<{ items: Task[] }>(`/api/queue?${params}`).then((r) => r.items)
  },
  // 0.17 P-E —— history = 已结束（done/failed/canceled），后端分页。status 传终态
  // 做子过滤，q 搜 name/config_name，type 按 task_type 过滤（P-F）。返回
  // { items, total, page, page_size }。
  listQueueHistory: (opts: {
    page: number; pageSize: number; q?: string; status?: TaskStatus;
    type?: TaskType; resourceClass?: QueueResourceClass
  }) => {
    const params = new URLSearchParams({
      group: 'history',
      page: String(opts.page),
      page_size: String(opts.pageSize),
    })
    if (opts.q) params.set('q', opts.q)
    if (opts.status) params.set('status', opts.status)
    if (opts.type) params.set('types', opts.type)
    if (opts.resourceClass) params.set('resource_class', opts.resourceClass)
    return req<QueueHistoryPage>(`/api/queue?${params}`)
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
  /** 0.17 P-B — scheduled task 手动提前：立即转 pending 参与调度。非 scheduled 409。 */
  startTaskNow: (id: number) =>
    req<{ task_id: number; status: string }>(`/api/queue/${id}/start_now`, {
      method: 'POST',
    }),
  retryTask: (id: number) =>
    req<Task>(`/api/queue/${id}/retry`, { method: 'POST' }),
  /** ADR 0006 — 暂停 running task。返回时 task 还在 running，需订阅 SSE
   *  task_state_changed 看 status 转 paused。状态不对（非 running / train_loop
   *  未启动）抛 409。 */
  pauseTask: (id: number) =>
    req<{ task_id: number; pause_pending: boolean }>(
      `/api/queue/${id}/pause`,
      { method: 'POST' },
    ),
  /** ADR 0006 PR-3 + Addendum 2 — 恢复 paused / failed / canceled task
   *  （从最近的 epoch 末 auto backup 续训）。恢复点文件缺失返 409 引导走
   *  ResumeFieldPicker 起新 task。done 不可恢复（走 retry）。 */
  resumeTask: (id: number) =>
    req<{ task_id: number; status: string }>(
      `/api/queue/${id}/resume`,
      { method: 'POST' },
    ),
  /** ADR 0006 PR-2 — 查队列挂起状态 + 等待恢复调度的 pending 数。 */
  getQueueHold: () => req<QueueHoldState>('/api/queue/hold'),
  /** 挂起队列：dispatcher 不拉新 task，已 running 的不受影响。 */
  holdQueue: () =>
    req<{ held: boolean }>('/api/queue/hold', { method: 'POST' }),
  /** 恢复调度：dispatcher 重新按优先级拉 pending。 */
  releaseQueue: () =>
    req<{ held: boolean }>('/api/queue/release', { method: 'POST' }),
  deleteTask: (id: number) =>
    req<{ deleted: number }>(`/api/queue/${id}`, { method: 'DELETE' }),
  /** 列 task 关联的 output 目录里所有文件（含 size/mtime/是否 lora）。
   * `supports_open_folder` 仅在请求来自 loopback 时为 true，云端为 false。 */
  getTaskOutputs: (id: number) =>
    req<TaskOutputs>(`/api/queue/${id}/outputs`),
  /** 下载单个 output 文件的直链，不发请求。<a href={...} download> 即可。 */
  taskOutputDownloadUrl: (id: number, path: string) =>
    `/api/queue/${id}/output/${path.split('/').map(encodeURIComponent).join('/')}`,
  /** output 目录打包 zip 下载直链。
   * 不传 files → 全量；传相对路径数组 → 仅打包这些（后端 whitelist 校验）。
   * 配合 <a href download> 触发，浏览器原生接管下载条；后端 zip 写完会
   * publish task_outputs_zip_ready / task_outputs_zip_failed 事件供前端清 loading。 */
  taskOutputsZipUrl: (id: number, files?: ReadonlyArray<string>) => {
    if (!files || files.length === 0) return `/api/queue/${id}/outputs.zip`
    const q = files.map((n) => encodeURIComponent(n)).join(',')
    return `/api/queue/${id}/outputs.zip?files=${q}`
  },
  exportTaskOutputs: (id: number, files?: ReadonlyArray<string>) =>
    req<DataExportItem>(`/api/queue/${id}/export-outputs`, {
      method: 'POST',
      body: JSON.stringify({ files: files && files.length > 0 ? Array.from(files) : null }),
    }),
  /** 删除 output 目录下选中的文件（批量）。relative_paths 相对 output/。
   *  任一不存在 → 后端 404 整批拒绝，前端 toast 错误后调用方 caller 应自行刷新。 */
  deleteTaskOutputs: (id: number, files: ReadonlyArray<string>) =>
    req<{ deleted: string[] }>(`/api/queue/${id}/outputs`, {
      method: 'DELETE',
      body: JSON.stringify({ files: Array.from(files) }),
    }),

  // PP8 — WD14 运行时 / GPU 装包 ------------------------------------------
  /** 当前 onnxruntime 状态：包名 / 版本 / providers / nvidia-smi 检测结果。 */
  getWD14Runtime: () => req<WD14Runtime>('/api/wd14/runtime'),
  /** 切换 onnxruntime（同步 pip，几分钟级；UI 必须带 loading）。 */
  installWD14Runtime: (target: 'auto' | 'gpu' | 'cpu' | 'directml') =>
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
  /** 当前 version 的 train/ 打包 zip 直链。<a href download> 触发即可,
   * 后端 publish version_train_zip_ready/_failed SSE 供前端清 "打包中..." 状态。 */
  versionTrainZipUrl: (pid: number, vid: number) =>
    `/api/projects/${pid}/versions/${vid}/train.zip`,

  /** 当前 version 的 bundle.zip 直链。<a href download> 触发浏览器下载。 */
  versionBundleZipUrl: (
    pid: number,
    vid: number,
    opts: {
      train?: boolean
      trainCaptions?: boolean
      reg?: boolean
      regCaptions?: boolean
      includeConfig?: boolean
      latentCache?: boolean
    },
  ): string => {
    const p = new URLSearchParams()
    p.set('train', opts.train !== false ? '1' : '0')
    p.set('train_captions', opts.trainCaptions !== false ? '1' : '0')
    p.set('reg', opts.reg ? '1' : '0')
    p.set('reg_captions', opts.regCaptions ? '1' : '0')
    p.set('include_config', opts.includeConfig ? '1' : '0')
    p.set('latent_cache', opts.latentCache ? '1' : '0')
    return `/api/projects/${pid}/versions/${vid}/bundle.zip?${p.toString()}`
  },
  exportBundleToDataExports: (
    pid: number,
    vid: number,
    opts: {
      train?: boolean
      trainCaptions?: boolean
      reg?: boolean
      regCaptions?: boolean
      includeConfig?: boolean
      latentCache?: boolean
    },
  ) =>
    req<DataExportItem>(`/api/projects/${pid}/versions/${vid}/export-bundle`, {
      method: 'POST',
      body: JSON.stringify({
        train: opts.train !== false,
        train_captions: opts.trainCaptions !== false,
        reg: opts.reg === true,
        reg_captions: opts.regCaptions === true,
        include_config: opts.includeConfig === true,
        latent_cache: opts.latentCache === true,
      }),
    }),
  listDataExports: () => req<DataExportItem[]>('/api/data-exports'),

  /** 从 PathPicker 选中的 zip 路径导入 bundle（v1/v2 均支持）→ 新建 project + v1。 */
  importBundleFromPath: (path: string) =>
    req<BundleImportResult>('/api/projects/import-bundle', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),
  importBundleFromDataExports: (filename: string) =>
    req<BundleImportResult>('/api/projects/import-bundle', {
      method: 'POST',
      body: JSON.stringify({ filename }),
    }),
  importBundleUpload: (
    file: File,
    onProgress?: (e: UploadProgressEvent) => void,
  ): Promise<BundleImportResult> => {
    const fd = new FormData()
    fd.append('file', file, file.name)
    return xhrUpload<BundleImportResult>('/api/projects/import-bundle/upload', fd, onProgress)
  },
  /** 上传训练集 zip → 新建 project + v1，返回新项目。 */
  importTrainProject: (
    file: File,
    onProgress?: (e: UploadProgressEvent) => void,
  ): Promise<{
    project: ProjectDetail
    version: Version
    stats: { image_count: number; tagged_count: number; untagged_count: number; concepts: string[] }
  }> => {
    const fd = new FormData()
    fd.append('file', file)
    return xhrUpload('/api/projects/import-train', fd, onProgress)
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
  /** 默认拉全量历史（max_points=0，server 跳过降采样）；想要降采样预览
   *  传具体数字。cold start 是一次性 HTTP，长训练（10k+ 步）下也只是 ~500KB
   *  payload，不值得为视觉损耗换网络节省。 */
  getMonitorState: (taskId: number, maxPoints?: number) =>
    req<MonitorState>(
      `/api/state?task_id=${taskId}` +
      (maxPoints != null ? `&max_points=${maxPoints}` : '') +
      `&_=${Date.now()}`,
    ),
  sampleImageUrl: (filename: string, taskId: number, w?: number) =>
    `/samples/${filename}?task_id=${taskId}${w ? `&w=${w}` : ''}`,
  listEvalMetrics: (pid: number, vid: number, taskId?: number) =>
    req<EvalMetricsListResponse>(
      `/api/projects/${pid}/versions/${vid}/eval/metrics?` +
      (taskId ? `task_id=${taskId}&` : '') +
      `_=${Date.now()}`,
    ),
  /** 列某 task 的训练后/手动评估 job（按 run_id 关联 checkpoint 行 + 取原始日志）。 */
  listTaskEvalJobs: (pid: number, vid: number, taskId: number) =>
    req<{ jobs: EvalJobInfo[] }>(
      `/api/projects/${pid}/versions/${vid}/eval/jobs?task_id=${taskId}`,
    ),
  /** 手动评估完成任务的选定 checkpoint（task-scoped，绕过自动评估开关）。 */
  runTaskEval: (
    pid: number,
    vid: number,
    body: { task_id: number; checkpoints: string[] },
  ) =>
    req<{ queued: number }>(
      `/api/projects/${pid}/versions/${vid}/eval/run`,
      { method: 'POST', body: JSON.stringify(body) },
    ),

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

  // studio_data 存储位置 -------------------------------------------------
  // withScan=true 含全量扫描，大目录可能要数秒 —— 调用方给加载态。
  getStudioDataInfo: (withScan = true) =>
    req<StudioDataInfo>(`/api/studio-data/info?scan=${withScan}`),
  // 422 = 目标不合法 / 有 running task；409 = 已有迁移在跑。
  startStudioDataMigrate: (target: string) =>
    req<{ ok: boolean }>('/api/studio-data/migrate', {
      method: 'POST',
      body: JSON.stringify({ target }),
    }),
  getStudioDataMigrateStatus: () =>
    req<StudioDataMigrateStatus>('/api/studio-data/migrate_status'),

  // 模型根目录存储位置（镜像 studio_data，但迁移完无需重启，立即生效）----------
  getModelsRootInfo: (withScan = true) =>
    req<ModelsRootInfo>(`/api/models-root/info?scan=${withScan}`),
  // 422 = 目标不合法 / 有 running task；409 code models_root.migration_busy = 已有
  // 迁移在跑；409 code models_root.target_conflict = 目标已有 models 数据（detail 带
  // existing_files/existing_bytes/same_name_files，modal 弹「跳过/覆盖/取消」后带
  // onConflict 重发）。
  startModelsRootMigrate: (target: string, onConflict?: 'skip' | 'overwrite') =>
    req<{ ok: boolean }>('/api/models-root/migrate', {
      method: 'POST',
      body: JSON.stringify({ target, on_conflict: onConflict ?? null }),
    }),
  getModelsRootMigrateStatus: () =>
    req<ModelsRootMigrateStatus>('/api/models-root/migrate_status'),

  // System lifecycle (ADR 0002) ----------------------------------------
  // 重启 server。后端写 tmp/restart + 给自己发 SIGINT 触发 uvicorn graceful
  // shutdown；cli.py 的 loop 拾起并重启。前端调完后应进入"重启中"等待状态，
  // 轮询 /api/health 直到服务回来。
  restartServer: () =>
    req<{ ok: boolean; message: string }>('/api/system/restart', {
      method: 'POST',
    }),

  // 当前仓库 git 状态：__version__ / commit / tag / branch / dirty
  getSystemVersion: () => req<SystemVersion>('/api/system/version'),

  // git fetch + 比对。master 通道 24h cache；force=true 强制重 fetch。
  // dev 通道（PR-D）每次都 fetch，不缓存。
  checkSystemUpdate: (channel: 'master' | 'dev' = 'master', force = false) => {
    const qs = new URLSearchParams({ channel, force: String(force) })
    return req<SystemUpdateCheck>(`/api/system/update_check?${qs.toString()}`)
  },
  getAnnouncements: () =>
    req<{ posts: AnnouncementPost[] }>('/api/announcements').then((r) => r.posts),

  // 请求 update：写 .update_pending + 触发 SIGINT 重启。
  // 422 = running task 或 dirty working tree（force=true 时跳过 dirty 闸，
  // reset --hard 覆盖本地未提交改动；用户需先在 UI 确认强制覆盖）。
  performSystemUpdate: (target: string = 'origin/master', force = false) =>
    req<{ ok: boolean; message: string }>('/api/system/update', {
      method: 'POST',
      body: JSON.stringify({ target, force }),
    }),

  // 回滚到 .last_version 记录的上一版本（PR-C）。
  // 422 = running task / dirty；409 = 没有 .last_version 或 commit 已 GC。
  rollbackSystem: () =>
    req<{ ok: boolean; message: string; target: string }>('/api/system/rollback', {
      method: 'POST',
    }),

  // 最近一次 update 的结构化结果（PR-C）。status: null = 从未 update 过。
  getSystemUpdateStatus: () => req<SystemUpdateStatus>('/api/system/update_status'),

  // 完整 .update_log 文本（PR-C，失败时 UI 弹 modal 用）。
  getSystemUpdateLog: () => req<{ content: string }>('/api/system/update_log'),

  // chunk 3 — git fetch + log origin/dev，返回最近 N 个 commit
  // （DevCard 时间线 + 任意 commit 切换用）。limit 默认 10，clamp 1-50。
  getDevCommits: (limit = 10) =>
    req<DevCommitsResult>(`/api/system/dev_commits?limit=${limit}`),

  // chunk 4 — 更新前置检查。VersionSection preview 状态展开时拉取，渲染
  // pre-flight 行；任一 level=err → blocking=true 禁用确认按钮。
  // target 接受任意 git ref（tag / branch / commit sha）。
  getPreflight: (target: string) =>
    req<PreflightResult>(`/api/system/preflight?target=${encodeURIComponent(target)}`),

  // 0.8.1 hotfix — zip 安装用户一键初始化 git 仓库。幂等：已是 git 仓库
  // 直接返 ok=true + already_initialized=true。失败 500 + detail.error。
  initGitRepo: () =>
    req<{ ok: boolean; already_initialized: boolean; anchor?: string; anchor_kind?: string }>(
      '/api/system/init_git',
      { method: 'POST' },
    ),
}

export interface SystemVersion {
  version: string
  commit: string
  commit_short: string
  commit_time_iso: string
  /** @deprecated UI 用 installed_kind / installed_label；branch 仅 debug */
  branch: string
  tag: string | null
  is_dirty: boolean
  /** 产品视角的"装了什么"分类（ADR 0005）。
   *  - stable：HEAD 命中 vX.Y.Z release tag，或 __version__ 匹某 release tag 且 tree 一致
   *  - dev：commit == origin/dev HEAD
   *  - custom：feature branch / detached / 未识别 commit
   *  - zip：REPO_ROOT/.git 缺失（zip 解压用户，0.8.1 hotfix） */
  installed_kind: 'stable' | 'dev' | 'custom' | 'zip'
  /** 用户可读 label，如 "v0.8.0" / "dev @ f6f202b · 2026-05-16" / "自定义（feat/foo @ a1b2c3d）"。
   *  dirty 时追加 "· 未提交修改" */
  installed_label: string
  /** "vX.Y.Z" 形式，仅 installed_kind=stable 时填；前端做版本号比对用 */
  stable_version: string | null
  /** False = zip 安装 / 没有 origin remote。前端显示 init banner 时用（0.8.1 hotfix） */
  is_git_repo: boolean
  /** False = git binary 不在 PATH。zip 用户 + 没装 git → 显示"先装 git"提示而非 init 按钮 */
  git_available: boolean
}

export interface SystemUpdateCheck {
  channel: 'master' | 'dev'
  current_commit: string
  latest_commit: string
  /** @deprecated 前端用 behind_count；commits_ahead 是 git 词汇 */
  commits_ahead: number
  /** @deprecated 前端用 state；has_update = (state === 'update_available') */
  has_update: boolean
  latest_tag: string | null
  checked_at: number
  error: string | null
  /** 状态机（ADR 0005）。
   *  - up_to_date：已是最新（版本号 / commit 一致）
   *  - update_available：远端有更新
   *  - ahead：本地领先远端（罕见，常见于回滚后又抢跑）
   *  - detached：当前 commit 不在 channel 历史上（feature branch / 离群） */
  state: 'up_to_date' | 'update_available' | 'ahead' | 'detached'
  /** 当前装的稳定版（master 通道）："vX.Y.Z" / null（没装 stable） */
  installed_version: string | null
  /** 远端最新稳定版（master 通道）："vX.Y.Z" / null（远端没 tag / dev 通道） */
  latest_version: string | null
  /** 前端文案"N 项更新"用（= commits_ahead，但语义更清楚） */
  behind_count: number
}

/** PR-C — 最近一次 update 的结构化结果。
 *  - status=null：从未 update 过，UI 不展示 banner
 *  - status='ok'：可选展示"已更新到 X"
 *  - status='aborted' / 'failed' / 'partial'：红色 banner + reason + "查看日志"
 *  - rollback_target：.last_version 内容（commit sha），UI 用它判断是否显示回滚按钮
 */
export interface SystemUpdateStatus {
  status: 'ok' | 'aborted' | 'failed' | 'partial' | null
  reason?: string
  target?: string
  from_commit?: string
  to_commit?: string
  started_at?: number
  finished_at?: number
  deps_changed?: boolean
  log_excerpt?: string
  rollback_target?: string | null
  /** rollback target commit 的 exact tag（如 v0.6.0）。后端 git describe
   *  --tags --exact-match 拿；commit 没打 tag → null。UI 优先显示 tag，
   *  fallback 到 sha 前 8 位 */
  rollback_target_tag?: string | null
}

/** chunk 3 — dev 通道最近 commit 摘要。fetched=false 时表示 git fetch 失败
 *  （离线 / 网络问题），commits 是本地 origin/dev 缓存。error 文案给 UI 提示。 */
export interface DevCommit {
  sha: string           // full sha，作为 performSystemUpdate target
  short_sha: string     // 前 8 位
  msg: string           // commit subject
  time_iso: string      // ISO8601
  author: string
}
export interface DevCommitsResult {
  commits: DevCommit[]
  fetched: boolean
  error: string | null
}

/** chunk 4 — 更新前置检查。任一 level=err → blocking=true 禁用确认按钮。 */
export interface PreflightCheck {
  key: 'dirty' | 'running_tasks' | 'requirements_diff' | 'last_version'
  level: 'ok' | 'warn' | 'err'
  label: string
}
export interface PreflightRequirementsDiff {
  added: string[]
  removed: string[]
  changed: { name: string; from: string; to: string }[]
}
export interface PreflightResult {
  target: string
  target_resolved: string | null
  checks: PreflightCheck[]
  blocking: boolean
  /** 工作树有未提交改动（自动 churn 已剔除）。true → 确认时弹"强制覆盖" modal。
   *  dirty 是 warn 不进 blocking，但仍需用户显式确认才会带 force 覆盖。 */
  working_tree_dirty: boolean
  requirements_diff: PreflightRequirementsDiff
}

export interface BrowseEntry {
  name: string
  type: 'dir' | 'file'
}

export interface BrowseResult {
  path: string
  parent: string | null
  entries: BrowseEntry[]
  /** 若传入的是文件路径，后端会回退到父目录，并把文件名放在这里供 picker 高亮。 */
  selected?: string | null
}
