# Changelog

仓库版本号唯一来源是 `studio/__init__.py` 的 `__version__`。FastAPI（`/api/health`）和前端 Sidebar 都从它派生。`studio/web/package.json` 的 `version` 字段需手动同步保持一致。

每次 release 改 `__version__` + 同步 `package.json` + 在本文件加一段。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本规则按语义化版本（0.x 阶段 MINOR 视为破坏性升级）。

---

## [0.5.1] — 2026-05-10

UI 体验小改进 + onnxruntime-gpu 跨平台修复（Windows / Linux 都踩到「装了 GPU 包但实际跑 CPU」）。

### 改进

- **打标 curation 工作流**（#27）
  - 全屏 preview 取代弹窗预览
  - 键盘 accept / remove 快捷键，过单张图更快
  - tag 保存后明确的 CNB export 入口

### 修复

- **onnxruntime-gpu 静默降级 CPU**（#29 Windows / #30 Linux）
  - 根因：onnxruntime 在 CUDA EP dlopen 失败时**不抛异常**，会内部 silently fallback 到 CPU；`ort.get_available_providers()` 仍报 CUDA 可用，UI 显示一切正常，用户只看到 CPU 占用飙升
  - 加监控：`_create_session` 比对实际 `session.get_providers()`，请求过 CUDA 但实际不在 → 上报 `cuda_load_error` 让 UI 可见
  - Windows：Python 3.8+ 废除 PATH 自动加载 DLL，新增 `os.add_dll_directory(torch/lib)` 让 onnxruntime 找得到 torch 自带的 `cublasLt64_12.dll` / `cudnn_*.dll`
  - Linux：worker subprocess 顶层显式 `import onnxruntime_setup` 触发 preload；修 `_has_system_cuda_libs()` 误判（云镜像装 CUDA Toolkit 但没装 cuDNN → 之前被误判为完整系统 CUDA → 跳过 preload）
  - 新增 `tools/diagnose_onnx_gpu.py` 诊断脚本

---

## [0.5.0] — 2026-05-09

累计 49 commits / 132 files (+17k / -1.6k)。集中在 4 块：测试出图、先验生成、Setup 重写、Settings 拆分 + 新 tagger（CLTagger）。

### 新增

- **断点续训 / resume_lora 字段内语义 picker**
  - `resume_state` / `resume_lora` 字段旁边的「📁 浏览本项目」按钮：弹出 dropdown 贴字段，按 version 分组列出项目所有可用文件，用户看的是「baseline / step 2476」这种语义 label，不暴露 `studio_data/projects/.../output/...` 深路径
  - 选中后写绝对路径回字段（schema 字段值仍是真路径，后端协议不变）
  - 外部文件 / 别项目的 ckpt 用户直接在字段 input 手填即可（不弹 picker，留空白逃生口）
  - 后端：`versions.list_project_state_ckpts()` / `list_project_lora_ckpts()` 项目级 + `/api/projects/{pid}/state_ckpts` / `/lora_ckpts` 端点
  - 前端：`ResumeFieldPicker` 组件 + `Field.tsx` 按字段名 dispatch（resume_state / resume_lora 走专用 picker，其它 path 字段保留 PathPicker）
  - 解决 UX 根因：之前用户必须从 REPO_ROOT 5 层深挖到 `output/training_state_step*.pt` 才能续训
- **测试出图（Generate）**
  - 侧栏「测试」入口；`/api/generate` + `runtime/anima_generate.py`（#19）
  - 推理 daemon（常驻 GPU，避免每次重载）+ XY 矩阵评测（参数扫）（#22）
  - `inference_core` 抽出，修多 LoRA 加载 P0 bug（#19）
  - SSE 改共享一条 EventSource，解 outputs/刷页面挂死
  - favicon 随机轮换（noal_*.png）
- **先验生成（无 LoRA）**
  - Step 4 加「先验生成」tab + explainer
  - `/api/projects/.../reg/generate-prior` + `runtime/anima_reg_ai.py`
  - `RegMeta.generation_method` 区分手工 / AI 生成
- **Setup & 环境**
  - `studio.bat` 纯 ASCII 守护（cp936 cmd.exe 不再炸）+ 单测兜底
  - bootstrap：Windows 优先 `py -3`，Linux 迭代版本检查
  - venv stale check + `--reinstall` flag（环境救命）
  - 首装 GPU-aware torch；CPU-only 误装大警告
  - defer torch reinstall 到 launcher 进程，解 Windows 锁文件 + 自愈僵尸目录
  - Settings 加 PyTorch section，一键重装为 CUDA 版
  - `studio.sh --mirror` flag + HF 镜像端点可配置（Settings UI toggle）
  - ONNX CUDA 错误推理期自动降 CPU；系统 CUDA 时跳过 torch wheel preload
- **Attention Backend（#21）**
  - `attention_backend` 单字段替代 `xformers` / `flash_attn` 双 bool
  - `/api/xformers/{status,install}` + Settings xformers 卡片
  - 加速三选一下拉
  - flash_attn 一键装 wheel + 模型层 fast path + CLI 入口
  - `detect_env` 改用 torch ABI 拿 cuda_tag，不依赖 nvidia-smi
- **Tagger**
  - 新 CLTagger（外部贡献，#14）
  - 抽 `OnnxTaggerBase`，CLTagger 自动获得 PP10 线程池
  - tagger registry + 统一 `<name>_overrides` 持久化键
- **版本控制**
  - 版本号集中到 `studio/__init__.py:__version__`，FastAPI / Sidebar 都从这派生
  - 新建本 `CHANGELOG.md`
- **文档结构重构**
  - 拆 `docs/` 为三块：`user-guide/`（用户向）、`architecture/`（开发者向）、`adr/`（决策记录）
  - 新建 `docs/README.md` 总入口 + `docs/adr/README.md` 含 ADR 模板
  - 三篇互斥方案文档合并为 [ADR 0001 — LoKr 走 lycoris-lora 而不切 sd-scripts](docs/adr/0001-lokr-via-lycoris-lora.md)
  - 删除已落地的 11 篇 PP 阶段 plan（`studio-pipeline/PP0–PP10`），保留 overview 改写为 `architecture/studio-pipeline.md`
  - 删除过期的 `trainer-optimization-analysis.md`（2025-02 快照，建议项已落地）
  - `docs/_local/` 进 `.gitignore` 收个人草稿
- **目录重组：`scripts/` + `tools/anima_*` → `runtime/`**
  - 新目录 `runtime/` 容纳所有 Anima 运行时核心（独立进程 / Studio subprocess 调起 / 可单独 CLI 跑）：`anima_train` / `anima_generate` / `anima_daemon` / `anima_reg_ai` / `train_monitor`
  - `tools/` 收敛为纯用户 CLI + setup helper（download_models / install_flash_attn / select_torch_index / validate_local_models / check_requirements_changed / bench_*）
  - 删除 ADR 0001 烟测遗物：`probe_lycoris_anima.py` + 5 个 `stage*.yaml` + `.gitignore` 4 行 `scripts/stage*_output/` 排除
  - 同步更新所有 subprocess 命令构造、sys.path 注入、test 路径断言、文档引用
  - 依赖方向单向：`models → utils → runtime → studio → tools`

### 变更

- **Settings 页**
  - 拆 4 个 tab：数据集 / 打标 / 训练 / 页面
  - ONNX Runtime 拆独立 section
  - WD14 / CLTagger 改 anima 主模型样式（radio + 行内下载）
  - 字段对齐 + 2K 屏留白修复
- 训练脚本搬到 `scripts/` + `tools/`，淘汰 `monitor_smooth.html`
- `LoraEntry` 抽到 `schema.py`（收尾 PR-9）
- 隐藏「监控与进度」组，`no_progress` 默认改 True

### 修复

- patch lycoris-lora 3.4.0 `LokrModule.get_weight` rank_dropout device bug
- stale 检测 mtime 改回并联，本地未 commit 编辑也触发重建
- 折叠态干掉单独的「导出训练集」按钮，避免误触
- 修补 PR #14 遗留的 UX 与测试漏洞

### 子 PR（已合到 dev）

- #14 CLTagger 支持（外部贡献）
- #19 PR-17 borrowed（Generate Phase 1）
- #20 PR-17 part 2（reg / inference_core 收尾）
- #21 attention_backend 整合
- #22 测试页面重设计 Phase 2（XY / daemon / 评测）

---

## [0.1.0] — 初始版本

`__version__` 字段诞生时的占位版本号（FastAPI app version 与 package.json 同此）；当时 Sidebar 显示的是手写的 `0.4`，未与代码版本号对齐。本次 0.5.0 起统一治理。
