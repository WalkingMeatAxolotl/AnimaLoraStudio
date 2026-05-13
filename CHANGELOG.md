# Changelog

> **本文件由 [`tools/bump_version.py render-changelog`](tools/bump_version.py)
> 从 [`release_notes.yaml`](release_notes.yaml) 自动派生 —— 请改 yaml，不要改本文件。
> 编写规范见 [`docs/release-notes-spec.md`](docs/release-notes-spec.md)。

---

## [0.6.0] — 2026-05-12

LLM tagger + 训练监控可观测性 + Settings 页面体系重排

### 新增

- **LLM tagger 第二打标器：OpenAI 兼容 API 长 caption（#18, #34, #35）**
  - 支持 OpenRouter / vLLM / Ollama 等任何 OpenAI Chat Completions 兼容端点
  - 训练 WandB 集成：`tracker_project` / `tracker_run_name` / `wandb_api_key`
    串到 sd-scripts，run url + 关键 metric 同步贴回项目页
  - 默认 opt-in：不配 endpoint 不调 LLM；wandb 默认关（用户显式开才上传）
  - JoyCaption 退化为 builtin preset，`JoyCaptionConfig` 删除
  - Prompt → messages 序列：`LLMMessage` 类型 + dnd-kit 拖拽编辑，支持
    multi-turn / few-shot 对话格式
  - Settings UI 双栏 grid + 4 张 section 合并大 card + composer 高度撑满

- **训练监控加 Topbar 系统资源 pill + SSE 增量协议（#37, #42）**
  - Topbar 4 个等宽 pill（CPU / GPU / MEM / VRAM，min-w 96px）+ 两端对齐
  - 从 `nvidia-ml-py`（pynvml 已停维护）拉，backend `_StatsThread` 2.5s
    间隔通过 SSE `system_stats_updated` 推到前端
  - Monitor 改增量协议：步进式 delta 取代每秒 full snapshot，10k 步训练
    payload O(N) → O(1)
  - Cold-start 拉全量历史：`/api/state` 默认 `max_points=0` 不降采样；
    前端 `MAX_LOSSES / MAX_LR` 5000 → 50000 对齐 backend `train_monitor`
  - `_SelectiveGZipMiddleware`：10k 步 `/api/state` ~500KB → ~100KB（5x），
    `/api/events` SSE + `/samples/*` 图片白名单跳过

- **新增 ModelScope（魔搭社区）作为模型下载源（#25）**
  - HF 拉不下来时切 ModelScope；Settings 加 `download_source` 选项，默认
    仍 `huggingface`
  - `_get_download_source()` 优先级：`MODELSCOPE_SOURCE` env > secrets >
    `'huggingface'`
  - 有映射的模型走魔搭 CLI 下载；无映射自动回退 HF
  - onnxruntime 安装失败时自动 fallback 到腾讯 pypi 镜像（仅 fallback，
    不改默认源）

### 变更

- **Settings 页面结构重排：新增监控 tab + 面包屑跳转 + sticky 索引（#36）**
  - 新增「监控」tab，WandB 从「训练」搬过去
  - HF / ModelScope 在「训练」合并成「模型下载源」section，按 `download_source`
    条件渲染
  - PageHeader 加 `tabs` slot；全局移除 eyebrow（与 Topbar 面包屑功能重复）
  - Topbar 面包屑改 React-Router `Link`，可点击跳转
  - 右侧 sticky section index：`IntersectionObserver(rootMargin: -20% 0px -70% 0px)`
    高亮 + 平滑滚动
  - LLM tagger 采样参数 + 图片预处理合并成默认折叠的「高级参数」面板
  - URL routing 不变（`/tools/settings`），tab 走 React state，旧浏览器
    书签照常工作

### 改进

- **队列输出页面改直链下载 + 批量 zip + 按 step/seed 排序（#33）**
  - Queue 详情页：直链下载 + 批量 zip + 按 step / seed 排序 + 文件名命名对齐
  - 之前必须从 `studio_data/projects/.../output/` 深挖才能拿到训练产物

### 修复

- **LLM tagger 后续修：caption 去重 + WandB 默认关 + 采样图缩图（#34）**
  PR #18 P0 followups 4 项：

  - **caption 重复**：`utils/caption_utils.py` 与 `studio/services/caption_format.py`
    各有一份 `normalize_caption_json`，merge 逻辑微妙不同（lowercase 去重
    vs 简单 extend）。改为单源 re-export，避免 schema 调整时双份漂移
  - **WandB 默认关**：PR #18 默认 `log_with_wandb=True` 导致用户必装 wandb
    才能跑；改 opt-in
  - **训练采样图缩图**：原 1024×1024 PNG 直塞前端 → 后端缩到 256 thumbnail
  - **自定义 `output_format`**：之前硬编码 PNG，加用户可选字段

- **Danbooru estimate API 403 漏修（0.5.2 hotfix 当时只修了 search）（#41）**
  - 0.5.2 当时只修了 search，estimate 走单独路径仍裸 UA；这次一并加上
    `AnimaLoraStudio/<version>` UA + `Accept: application/json`
  - 配套 `tests/test_downloader.py` 加 estimate 回归用例

- **先验生成 500 `NameError: STUDIO_DATA`（#42 内）**
  `reg_generate_prior` 写 cfg 用 `STUDIO_DATA / "reg_ai_configs"`，但
  `server.py` 顶部 `from .paths import (...)` 漏掉 `STUDIO_DATA`，路由
  一调即崩。一行 import 修复。

---

## [0.5.2] — 2026-05-12

Danbooru 挂 Cloudflare 后 search API 403 hotfix

### 改进

- **UA 带 `(by username)` + Danbooru 强制账号绑定（不再支持匿名）**
  - UA 带 `(by username)`：符合 danbooru TOS 推荐格式；CF 收紧时按账户
    白名单比按匿名 UA 更安全
  - `secrets.has_credentials_for("danbooru")` 现在校 `username + api_key`；
    与 gelbooru 行为一致
  - 之前注释说"匿名也能跑"已不再属实（CF 时代匿名 = 0），改为明确强制
  - Settings UI placeholder 改为"必填 — danbooru 挂了 Cloudflare 后不再支持匿名"

### 修复

- **Danbooru search API 403 — 加应用 UA 让 Cloudflare 放行**
  Danbooru 挂 Cloudflare 后 search API 全部 403（`Just a moment...` 挑战页）。

  - `services/booru_api.search_posts` 之前没设 `headers`，requests 默认 UA
    `python-requests/X.Y.Z` 被 CF 直接拦
  - 用应用名 UA `AnimaLoraStudio/0.5.2` 而不是浏览器伪装（实测 Chrome UA
    也照 403，CF 把它当作"浏览器但不跑 JS"的爬虫）
  - 加 `Accept: application/json` 让中间件路由更确定
  - `pynvml` → `nvidia-ml-py`（PR #37 已加过，这版统一）

---

## [0.5.1] — 2026-05-10

UI 体验小改进 + onnxruntime-gpu 跨平台修复

### 改进

- **打标 curation 工作流：全屏 preview + 键盘 accept/remove 快捷键（#27）**
  - 全屏 preview 取代弹窗预览
  - 键盘 accept / remove 快捷键，过单张图更快
  - tag 保存后明确的 CNB export 入口

### 修复

- **onnxruntime-gpu 在 Windows / Linux 静默降级 CPU（#29, #30）**
  根因：onnxruntime 在 CUDA EP dlopen 失败时**不抛异常**，会内部 silently
  fallback 到 CPU；`ort.get_available_providers()` 仍报 CUDA 可用，UI 显示
  一切正常，用户只看到 CPU 占用飙升。

  - 加监控：`_create_session` 比对实际 `session.get_providers()`，请求
    过 CUDA 但实际不在 → 上报 `cuda_load_error` 让 UI 可见
  - Windows：Python 3.8+ 废除 PATH 自动加载 DLL，新增 `os.add_dll_directory(torch/lib)`
    让 onnxruntime 找得到 torch 自带的 `cublasLt64_12.dll` / `cudnn_*.dll`
  - Linux：worker subprocess 顶层显式 `import onnxruntime_setup` 触发 preload；
    修 `_has_system_cuda_libs()` 误判（云镜像装 CUDA Toolkit 但没装 cuDNN
    → 之前被误判为完整系统 CUDA → 跳过 preload）
  - 新增 `tools/diagnose_onnx_gpu.py` 诊断脚本

---

## [0.5.0] — 2026-05-09

测试出图 + 先验生成 + Setup 重写 + Settings 拆分 + CLTagger（49 commits / 132 files）

### 新增

- **测试出图（Generate）：侧栏入口 + 推理 daemon + XY 矩阵评测（#19, #22）**
  - 侧栏「测试」入口；`/api/generate` + `runtime/anima_generate.py`
  - 推理 daemon（常驻 GPU，避免每次重载）
  - XY 矩阵评测（参数扫）
  - `inference_core` 抽出，修多 LoRA 加载 P0 bug
  - SSE 改共享一条 EventSource，解 outputs / 刷页面挂死
  - favicon 随机轮换（noal_*.png）

- **先验生成（无 LoRA）：Step 4 加先验 tab + /reg/generate-prior 端点**
  - Step 4 加「先验生成」tab + explainer
  - `/api/projects/.../reg/generate-prior` + `runtime/anima_reg_ai.py`
  - `RegMeta.generation_method` 区分手工 / AI 生成

- **断点续训：resume_state / resume_lora 加项目内文件 picker**
  - `resume_state` / `resume_lora` 字段旁边的「📁 浏览本项目」按钮：弹出
    dropdown 贴字段，按 version 分组列出项目所有可用文件，用户看的是
    「baseline / step 2476」这种语义 label，不暴露
    `studio_data/projects/.../output/...` 深路径
  - 选中后写绝对路径回字段（schema 字段值仍是真路径，后端协议不变）
  - 外部文件 / 别项目的 ckpt 用户直接在字段 input 手填即可（不弹 picker，
    留空白逃生口）
  - 后端：`versions.list_project_state_ckpts()` / `list_project_lora_ckpts()`
    + `/api/projects/{pid}/state_ckpts` / `/lora_ckpts` 端点
  - 解决 UX 根因：之前用户必须从 REPO_ROOT 5 层深挖到
    `output/training_state_step*.pt` 才能续训

- **Setup 重写：GPU-aware torch 首装 + venv stale check + --reinstall 救命**
  - `studio.bat` 纯 ASCII 守护（cp936 cmd.exe 不再炸）+ 单测兜底
  - bootstrap：Windows 优先 `py -3`，Linux 迭代版本检查
  - venv stale check + `--reinstall` flag（环境救命）
  - 首装 GPU-aware torch；CPU-only 误装大警告
  - defer torch reinstall 到 launcher 进程，解 Windows 锁文件 + 自愈僵尸目录
  - Settings 加 PyTorch section，一键重装为 CUDA 版
  - `studio.sh --mirror` flag + HF 镜像端点可配置（Settings UI toggle）
  - ONNX CUDA 错误推理期自动降 CPU；系统 CUDA 时跳过 torch wheel preload

- **Attention Backend 单字段（xformers / flash_attn / none）三选一（#21）**
  - `attention_backend` 单字段替代 `xformers` / `flash_attn` 双 bool
  - `/api/xformers/{status,install}` + Settings xformers 卡片
  - flash_attn 一键装 wheel + 模型层 fast path + CLI 入口
  - `detect_env` 改用 torch ABI 拿 cuda_tag，不依赖 nvidia-smi

- **新 CLTagger 打标器（外部贡献）+ tagger registry（#14）**
  - 新 CLTagger（外部贡献）
  - 抽 `OnnxTaggerBase`，CLTagger 自动获得 PP10 线程池
  - tagger registry + 统一 `<name>_overrides` 持久化键

- **版本号集中到 studio/__init__.py + 新建 CHANGELOG.md（后续被 yaml 替代）**
  - 版本号集中到 `studio/__init__.py:__version__`，FastAPI / Sidebar 都从这派生
  - 新建 `CHANGELOG.md`（后续被 `release_notes.yaml` 替代为 source of truth）

- **docs/ 重构：拆 user-guide / architecture / adr 三块**
  - 拆 `docs/` 为三块：`user-guide/`（用户向）、`architecture/`（开发者向）、
    `adr/`（决策记录）
  - 新建 `docs/README.md` 总入口 + `docs/adr/README.md` 含 ADR 模板
  - 三篇互斥方案文档合并为 ADR 0001 — LoKr 走 lycoris-lora 而不切 sd-scripts
  - 删除已落地的 11 篇 PP 阶段 plan，保留 overview 改写为
    `architecture/studio-pipeline.md`
  - 删除过期的 `trainer-optimization-analysis.md`（2025-02 快照，建议项已落地）
  - `docs/_local/` 进 `.gitignore` 收个人草稿

### 变更

- **目录重组：scripts/ + tools/anima_* 搬到 runtime/**
  - 新目录 `runtime/` 容纳所有 Anima 运行时核心（独立进程 / Studio
    subprocess 调起 / 可单独 CLI 跑）：`anima_train` / `anima_generate` /
    `anima_daemon` / `anima_reg_ai` / `train_monitor`
  - `tools/` 收敛为纯用户 CLI + setup helper（download_models /
    install_flash_attn / select_torch_index / validate_local_models /
    check_requirements_changed / bench_*）
  - 删除 ADR 0001 烟测遗物：`probe_lycoris_anima.py` + 5 个 `stage*.yaml`
    + `.gitignore` 4 行 `scripts/stage*_output/` 排除
  - 依赖方向单向：`models → utils → runtime → studio → tools`

- **Settings 拆 4 个 tab（数据集 / 打标 / 训练 / 页面）+ ONNX 独立 section**
  - 拆 4 个 tab：数据集 / 打标 / 训练 / 页面
  - ONNX Runtime 拆独立 section
  - WD14 / CLTagger 改 anima 主模型样式（radio + 行内下载）
  - 字段对齐 + 2K 屏留白修复

- **训练监控简化：no_progress 默认 True + 隐藏「监控与进度」组**
  - 训练脚本搬到 `scripts/` + `tools/`，淘汰 `monitor_smooth.html`
  - `LoraEntry` 抽到 `schema.py`（收尾 PR-9）
  - 隐藏「监控与进度」组，`no_progress` 默认改 True

### 修复

- **patch lycoris-lora 3.4.0 LokrModule.get_weight rank_dropout device bug**
  临时 patch lycoris-lora 3.4.0 的 `LokrModule.get_weight` 在 rank_dropout
  路径上的 device mismatch bug（CUDA vs CPU 张量混用）。upstream 修了之后
  删 patch。

- **stale 检测 mtime 改回并联，本地未 commit 编辑也触发重建**
  `_web_dist_is_stale()` 之前把 mtime 降级成 fallback（HEAD 一致就跳过），
  导致本地编辑后 `studio run` 看不到变化。改成并联（HEAD 比对 || mtime
  比对，任一 stale 就重建）。

- **折叠态干掉单独的「导出训练集」按钮，避免误触**

- **修补 PR #14 (CLTagger) 遗留的 UX 与测试漏洞**

---
