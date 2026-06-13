# AnimaLoraStudio

[![中文](https://img.shields.io/badge/lang-%E4%B8%AD%E6%96%87-blue)](README.md) [![English](https://img.shields.io/badge/lang-English-lightgrey)](README.en.md) [![Version](https://img.shields.io/badge/version-0.13.0-blue)](CHANGELOG.md)

**端到端流水线**：从 Booru 抓图 → 筛选 → 打标 → 正则集 → 训练 → 出图测试，全流程在一个浏览器面板里推进。专为 [Anima](https://huggingface.co/circlestone-labs/Anima)（Cosmos DiT 二次元特调）训练优化。

## 特性

### 数据准备

- **Booru 抓图集成**：原生支持 Gelbooru / Danbooru，包含 Cloudflare 兼容 UA、双 token bucket 速率限制、Danbooru 账号认证
- **正则集自动生成**：基于训练集 tag 分布的 Booru 反向搜索 + 长宽比聚类；或使用底模直接生成（无需 LoRA）
- **三种打标器**：WD14（本地 ONNX）、CLTagger（外部贡献，本地 ONNX）、LLM（OpenAI 兼容 API，支持长 caption）
- **触发词自动注入**：在打标步骤填写一次，自动写入每张 caption 与训练采样图 prompt
- **训练前去重审核**：基于 perceptual hash 扫描相似 / 差分图，按组人工审核保留 / 移除，软删图可恢复

### 训练与实验管理

- **Project / Version 双层模型**：单项目可包含多个 version，共享数据下载，独立维护配置 / 输出 / 正则集
- **预设双向流**：训练配置可在 version 私有 config 与全局预设池之间互相 fork
- **多任务队列**：支持任务排队、暂停（从最近 epoch 末保存进度）、恢复、队列调度挂起
- **内置出图测试**：训练完成后直接在 Studio 内进行单图测试或 XY 矩阵评测，常驻推理 daemon 减少模型加载开销
- **训练集打包**：bundle 一键导出 / 导入（本地下载或服务器路径），便于跨机器迁移项目

### 训练算法

- **Loss 函数**：MSE / Huber，可配置权重曲线（min_snr / cosmap / detail_inv_t 等）
- **Timestep 采样**：uniform / logit_normal / mode / mixed_uniform 等，含可配置 schedule shift
- **InfoNoise 自适应采样（可选）**：基于 I-MMSE 的反 CDF 时间步采样器
- **优化器**：AdamW / Lion / Automagic / Prodigy / Prodigy+ScheduleFree（推荐起点 / 切换换算见 [`docs/user-guide/optimizers.md`](docs/user-guide/optimizers.md)）
- **Adapter**：LoRA + LyCORIS LoKr（走 [lycoris-lora](https://github.com/KohakuBlueleaf/LyCORIS) 官方库，含 DoRA / rs-LoRA / dropout）
- **分层 rank**：`lora_rank_rules` 按层名正则配不同 rank，便于按模块重要性差异化分配参数预算
- **Attention backend**：xformers / flash_attn / PyTorch SDPA

### 工程体验

- **环境自愈**：首装自动选择 GPU 兼容 torch（cu118 至 cu130）、venv 与 requirements.txt 哈希比对自动同步、Windows 锁文件处理
- **Web 界面自更新**：Settings 内支持 git pull、重启、回滚；master 稳定通道与 dev 滚动通道
- **加速后端切换**：Settings 内一键安装 xformers / flash_attn wheel；ONNX Runtime 三档（DirectML / CUDA / CPU）按平台一键装
- **国际化**：内置中英双语界面，首次启动选择语言，Settings 内可切换

![Studio 训练页](docs/images/studio-train.png)

### 架构

- **训练核心** (`runtime/anima_train.py`) 与 Studio 后端解耦，支持独立 CLI 调用或由 Studio 作为 subprocess 拉起
- **可扩展插件**：adapter / optimizer / scheduler / loss / timestep_sampler 五个 plugin registry，自定义变体需新增 builder 函数、字典注册与 schema Literal（详见 [`runtime/training/README.md`](runtime/training/README.md) 与 [ADR 0003](docs/adr/0003-anima-train-refactor.md)）

### Studio Web 工作台 (`studio/`)

流水线 8 步 + 工具页：

1. **下载** — Booru 抓取 / 本地 jpg / png / zip 上传
2. **筛选** — download / train 双面板，多选复制 / 移除，子文件夹管理
3. **预处理**（可选）— 总览（多选 + 一键撤销）+ 去重审核 + 放大（ESRGAN / Real-ESRGAN 多预设）+ 裁剪（手动框选 + 智能 AR 聚类预填）+ 涂抹
4. **打标** — WD14 / CLTagger / LLM 三选，GPU EP 自动 fallback；顶部 trigger_word 输入
5. **标签编辑** — 缓存模式 + 还原点，批量加 / 删 / 替换
6. **正则集**（可选）— AI 先验生成（默认）/ Booru 反向搜；mirror + flat 结构，可编辑 / 删图 / 自动去重 / 双 tagger 可选
7. **训练** — 预设双向流，入队即开始；config 编辑自动落盘；Simple / Advanced 模式
8. **测试出图** — 单图 / XY 矩阵 / 推理 daemon

通用面板：

- 队列 / 任务详情（日志 / 监控 / 输出下载 / 全量 zip）
- 实时训练监控（loss / lr 曲线 + 采样图按 step 切换）
- Topbar 系统资源（CPU / GPU / 内存 / VRAM）
- Settings（凭据 / 模型管理 / 加速后端 / WandB / 自更新 / 显示）
- 暗色 / 日间模式与字号密度切换

---

---

## 快速开始

### 0. 系统先决条件（需自行安装）

下面这些**不是** Studio 自动装的，得先准备好：

- **NVIDIA GPU 驱动 + CUDA runtime**（**16 GB+ 显存推荐，8 GB 极限可跑**；A 卡 / Apple Silicon 不支持）
- **Python 3.10+**（PATH 上能直接 `python` 调到）
- **Node.js 18+**（前端构建用，PATH 上能 `npm`）
- **Git**

### 1. 拉代码 + 启动 Studio

```bash
git clone https://github.com/WalkingMeatAxolotl/AnimaLoraStudio
cd AnimaLoraStudio

# Windows
studio.bat

# Linux / macOS
./studio.sh
```

首次运行会自动：建 `venv/` → 按 GPU 驱动检测装对应 CUDA torch（cu118 至 cu130）→ 装 `requirements.txt` → 构建前端 → 起后端 → 自动开浏览器到 <http://127.0.0.1:8765/studio/>。首次启动会弹引导 modal，按 checklist 一键安装底模 + ONNX Runtime + 训练加速包。

> 如果驱动检测失败导致装了 CPU 版 torch，可在 Settings → 系统 → PyTorch 一键重装 CUDA 版；也可通过 `studio.bat --torch cu128`（或 `studio.sh --torch cu128`）显式指定。

其它启动方式（等价于上面，便于直接 `python` 调）：

```bash
python -m studio              # 构建前端（如缺）+ 起后端
python -m studio dev          # 前后端 watch：vite 5173 + uvicorn 8765 --reload
python -m studio build        # 仅构建前端
python -m studio test         # pytest + vitest
```

### 2. 在 Studio 里下载模型

打开后先去 **设置（Settings）→ Models**，点按钮一键下载训练所需的全部权重 + tokenizer。

下载源默认走 `huggingface.co` 官方。国内用户如果直连慢，可以去 **Settings → 训练 → HuggingFace → endpoint** 切到「自定义 URL」粘贴自建反代，或者切到 **Settings → 训练 → 下载源 → ModelScope**（魔搭社区直连，需 `pip install modelscope`）。CLI 用户用 `python tools/download_models.py --endpoint URL` 或 `--modelscope` 覆盖。

下载内容（默认落到 `./models/`）：

| 项 | 来源 | 路径 | 大小 |
|---|---|---|---|
| Anima 主模型（latest = 1.0）| [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | `models/diffusion_models/` | ~4 GB |
| Anima VAE | 同上 | `models/vae/` | ~250 MB |
| Qwen3-0.6B-Base 文本编码器 | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | `models/text_encoders/` | ~1.2 GB |
| T5 tokenizer（仅 3 文件，不下权重）| [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) | `models/t5_tokenizer/` | <1 MB |

也可以走 CLI（与 UI 共用同一份代码）：

```bash
python tools/download_models.py                   # 全量下（HF 官方源）
python tools/download_models.py --endpoint URL    # 走自建反代
python tools/download_models.py --modelscope      # 走魔搭社区
python tools/download_models.py --variant preview3-base
python tools/download_models.py --skip-main --skip-vae
python tools/download_models.py --output /data/anima
```

WD14 打标模型不在这里——首次进 ③ 打标时自动从 HF 拉到 `models/wd14/`。

### 3. 跟着 Stepper 走

打开 <http://127.0.0.1:8765/studio/>：

1. 项目页「+ 新建项目」
2. **① 下载**：Booru 抓图（先在设置填 Gelbooru / Danbooru 凭据）或本地上传 zip
3. **② 筛选**：双 grid，选要训的图复制到 train/
4. **③ 预处理（可选）**：去重审核 / 放大 / 裁剪；不需要可直接跳过
5. **④ 打标**：选 WD14 / CLTagger / LLM（OpenAI compatible，含 JoyCaption preset）+ 阈值，一键自动打标
6. **⑤ 标签编辑**：批量加 / 删 / 替换；单图修；自动还原点
7. **⑥ 正则集（可选）**：两种生成方式可选 ——
   - **Booru 反向搜**：基于 tag 分布反向搜 booru，自动 WD14 打标 + 分辨率 AR 聚类
   - **AI 先验生成**：无 LoRA 直接用底模出图当 reg 集
8. **⑦ 训练**：选 preset 复制进 version 私有 config，改参数（debounce 600ms 自动落盘，无需点保存），入队即开始训练。Picker 标签会显示「· 已自定义」表示和原预设已分叉，预设池不会被改
9. 「队列」页查看任务，进**任务详情**看日志 / 监控 / 输出（含一键全量 zip 下载）

训完后侧栏 **测试**：跑单图 / XY 矩阵 / 推理 daemon 评测 LoRA，prompt 可从训练集直接拉，不用切 ComfyUI 反复测。

输出的 LoRA 权重已经是 `lora_unet_*` 格式，**直接拖进 ComfyUI 即可**，不需要任何转换。

---

## 项目结构

```
AnimaLoraStudio/
├── runtime/                       # Anima 运行时核心（独立进程；Studio 通过 subprocess 拉起，也可单独 CLI 跑）
│   ├── anima_train.py             # 训练入口
│   ├── training/                  # 训练栈子包：context / phases / loop / sample_runner
│   │   ├── adapters/              # plugin: lokr / loha / lora
│   │   ├── optimizers/            # plugin: adamw / lion / automagic / prodigy / prodigy_plus_schedulefree
│   │   ├── schedulers/            # plugin: cosine / cosine_with_restart / cosine_with_warmup / none
│   │   ├── inference_samplers/    # plugin: er_sde 等
│   │   └── phases/                # bootstrap / models / dataset / optimizer / resume / finalize
│   ├── anima_generate.py          # 出图：单图 / XY 矩阵
│   ├── anima_daemon.py            # 推理 daemon：常驻 GPU 加载 LoRA 和底模
│   ├── anima_reg_ai.py            # AI 先验生成：无 LoRA 直接用底模出 reg 集
│   └── train_monitor.py           # 训练状态写入器
├── studio/                        # AnimaStudio Web 工作台（FastAPI + React）— 4 层架构（ADR 0008）
│   ├── api/                       # HTTP 表面：FastAPI app + 27 router + schemas + deps + exception_handlers
│   ├── services/                  # 业务服务 11 子包：tagging / booru / reg / inference / models /
│   │                              #   preprocess / projects / dataset / presets / runtime / data_io
│   ├── domain/                    # pydantic 模型：TrainingConfig / LoRA / XY / Generate / RegAi + migrations
│   ├── infrastructure/            # 路径 / 数据库 / event bus / secrets / 日志 / argparse 桥接
│   ├── supervisor/                # 任务调度守护线程
│   ├── workers/                   # 4 个后台子进程入口（download / tag / reg_build / preprocess）
│   ├── server.py                  # 51 行兼容 shim，re-export `app` / `main`（真实入口在 api/app.py / api/main.py）
│   └── web/                       # React + Vite 前端
├── tools/                         # 用户 CLI / 启动期 setup helper
│   ├── download_models.py         # 一键下载主模型 / VAE / Qwen3 / T5 tokenizer
│   ├── install_flash_attn.py      # flash_attn wheel 一键装
│   ├── select_torch_index.py      # GPU-aware torch CUDA index 选择（启动期自动调）
│   ├── check_requirements_changed.py  # venv stale 检测（启动期自动调）
│   └── validate_local_models.py   # 验证本地 Qwen / T5 是否可离线加载
├── docs/                          # 三块：user-guide / architecture / adr（见 docs/README.md）
├── utils/                         # anima_train 共享 utility（model loader / optimizer / lycoris_adapter / ...）
└── models/                        # 模型代码 + tokenizer 预置文件 + 大权重落点（混合）
    ├── anima_modeling*.py         # tracked：Anima Cosmos transformer 的 PyTorch 实现
    ├── cosmos_predict2_modeling.py
    ├── wan/vae2_1.py              # tracked：Wan2.1 VAE 实现
    ├── text_encoders/             # tracked: Qwen tokenizer 小文件 + 用户下载的 model.safetensors
    ├── t5_tokenizer/              # tracked: T5 tokenizer 文件（无权重）
    ├── diffusion_models/          # 用户下载的 Anima 主模型（gitignored）
    ├── vae/                       # 用户下载的 VAE 权重（gitignored）
    ├── wd14/                      # WD14 ONNX 模型（HF 自动下载，gitignored）
    └── taeflux/                   # TAEFlux 中间步预览权重（gitignored）
```

运行时数据（gitignored）:
- `studio_data/` — SQLite + 用户 preset
- `studio_data/tasks/{id}/` — 每个训练 task 的 config snapshot + monitor state + 采样图 + run.log（删 version 不丢历史）
- `studio_data/projects/{id}-{slug}/versions/{label}/output/` — 训练产物 LoRA
- `studio_data/projects/{id}-{slug}/versions/{label}/reg/` — 正则集（多 task 复用）
- `models/diffusion_models/`, `models/vae/`, `models/wd14/` — 大权重文件

---

## 工具脚本

`tools/` 下的 CLI 与 Studio UI 共用同一份 `services/` 代码，方便无头环境用：

| 脚本 | 用途 |
|---|---|
| `tools/download_models.py` | 一键下载主模型 / VAE / Qwen3 / T5 tokenizer。多版本可选，支持 `--no-mirror` / `--endpoint URL` |
| `tools/install_flash_attn.py` | 按 torch ABI 自动选 flash_attn wheel 装上 |
| `tools/select_torch_index.py` | 探测 GPU + 推荐 PyTorch CUDA index URL（cu130 / cu128 / ...） |
| `tools/validate_local_models.py` | 验证本地 Qwen / T5 是否可离线加载 |

`runtime/` 下的运行时脚本（`anima_train` / `anima_generate` / `anima_daemon` / `anima_reg_ai`）也可以脱离 Studio 直接 CLI 跑——详见各脚本顶部 docstring。

---

## 文档

文档总入口：[docs/README.md](docs/README.md)。分三块：

**用户向**（[`docs/user-guide/`](docs/user-guide/)）
- [tagging-guide.md](docs/user-guide/tagging-guide.md) — Anima 标签格式与最佳实践
- [training-tips.md](docs/user-guide/training-tips.md) — 训练参数 / 显存配置矩阵 / 常见问题
- [regularization.md](docs/user-guide/regularization.md) — 正则集生成原理
- [caption-format.md](docs/user-guide/caption-format.md) — JSON 标签格式 + 分类 shuffle

**开发者向**
- [docs/architecture/studio-pipeline.md](docs/architecture/studio-pipeline.md) — Studio 跨步骤架构总览
- [studio/README.md](studio/README.md) — Studio 内部模块结构

**协作公约**
- [CONTRIBUTING.md](CONTRIBUTING.md) — 流程 / 分支 / commit / PR / release
- [docs/AGENTS.md](docs/AGENTS.md) — 代码质量约定与 AI agent 协作

**历史决策**（[`docs/adr/`](docs/adr/)）
- 记录「为什么选 X 而不选 Y」，已落地的不删

**版本变更**
- [CHANGELOG.md](CHANGELOG.md)

---

## 版本

当前版本 **0.13.0**。完整变更历史见 [CHANGELOG.md](CHANGELOG.md)。Studio 内 Settings → 系统 → 版本卡片可一键升级到最新版本。

---

## 硬件要求

- **GPU**：NVIDIA，**16 GB+ 显存推荐**（RTX 4060Ti 16G / 4070Ti / 4080 / 5070+ / 3090 / 4090 / 5090 等）；**8 GB 极限可跑**（部分笔记本 GPU 实测可行，需关 sample 输出 + 减小 batch / 分辨率，且训练速度明显下降）。系统 GPU 占用低，VRAM 主要给训练；A 卡 / Apple Silicon 不支持
- **RAM**：16 GB+
- **存储**：SSD 强烈推荐（latent cache + sample 输出 IO 频繁）

---

## 上游与致谢

- 核心训练脚本派生自 [**Moeblack/AnimaLoraToolkit**](https://github.com/Moeblack/AnimaLoraToolkit)
- 主模型 / VAE：[circlestone-labs / Anima](https://huggingface.co/circlestone-labs/Anima)
- OrthoLoRA / T-LoRA 适配器实现派生自 [**sorryhyun/anima_lora**](https://github.com/sorryhyun/anima_lora)（MIT），算法出自 [ControlGenAI/T-LoRA](https://github.com/ControlGenAI/T-LoRA) 论文与官方实现
- Automagic 优化器移植自 [**ostris/ai-toolkit**](https://github.com/ostris/ai-toolkit)（MIT），bf16 Kahan 路径参考 [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)
- 测试出图 / 采样链路对齐并派生自 [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI)（GPL-3.0）

完整的第三方算法 / 代码 / 论文出处见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

---

## License

仓库整体以 **GPL-3.0** 发布（包含 / 派生自 ComfyUI 的 GPL-3.0 代码实现）。

仓库内同时包含部分 Apache-2.0 第三方实现（NVIDIA Cosmos / Wan2.1 等），请保留原文件头声明。详见：

- `LICENSE`（GPL-3.0）
- `LICENSE-APACHE`（Apache-2.0 文本，用于仓库内 Apache-2.0 组件）
- `THIRD_PARTY_NOTICES.md`

**模型权重**（Anima / Qwen / VAE）有各自的条款（含 Non-Commercial 等限制），请以对应模型卡 / HF repo 协议为准。
