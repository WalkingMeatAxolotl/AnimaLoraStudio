# AnimaLoraStudio

[![Version](https://img.shields.io/badge/version-0.8.2-blue)](CHANGELOG.md)

**端到端流水线**：从 Booru 抓图 → 筛选 → 打标 → 正则集 → 训练 → 出图测试，全流程在一个浏览器面板里推进。专为 [Anima](https://huggingface.co/circlestone-labs/Anima)（Cosmos DiT 二次元特调）训练优化。

## 特性

- **🧬 Project / Version 双层数据模型** — 每次训练对应一个 `Project` + 一个 `Version`；version 可 fork（共享 download，独立 train/reg/output），方便 A/B 调参不重抓数据
- **📦 正则集自动生成（Booru 反向搜）** — 基于 train 集 tag 分布贪心搜 booru，按长宽比聚类拼出**匹配画风**的 reg 集；或 **AI 先验生成**：无 LoRA 直接用底模出图当 reg 集
- **🧪 内置出图测试 + XY 矩阵评测**（v0.5 新）— 训完直接在 Studio 里扫 LoRA 权重 / step / sampler 等参数出对比图，不用切 ComfyUI
- **🌐 Booru 池** — 统一双 token bucket（API 2 / CDN 5 req/s）+ 并发 worker + 429 sticky 退避，download 和 reg 共用
- **🛠️ 环境自愈系统** — venv stale 检测 / `--reinstall` 救命 flag / 首装 GPU-aware torch / Windows 锁文件 defer / ONNX CUDA 失败自动降 CPU；少有同类工具做这一层
- **🚀 一键加速后端切换**（v0.5 新）— Settings 里一键装 xformers / flash_attn wheel，三选一切 attention backend，训练 / 出图共用
- **🔁 Preset 双向流** — version 私有 config 和全局 preset 池可 fork / save_as_preset，参数实验不污染基线
- **🔄 webui 一键自更新**（v0.7 新）— Settings → 系统 → 版本卡片里 git pull + 重启 + 回滚全套，不用回命令行；master / dev 双通道并排，CHANGELOG 嵌入 + 4 项 pre-flight 检查（详见 [ADR 0002](docs/adr/0002-webui-self-update.md)）
- **🧩 训练栈可扩展**（v0.7 新）— `runtime/training/` 子包 + 4 个 plugin registry（adapters / optimizers / schedulers / inference_samplers）+ `AdapterProtocol` hook（on_step_begin / regularization_loss / excludes_weight_decay）；加新变体走 3 步（写 build 函数 + 字典加行 + schema Literal）不动 phases / loop（详见 [ADR 0003](docs/adr/0003-anima-train-refactor.md) + [`runtime/training/README.md`](runtime/training/README.md)）
- **🖼️ 图片预处理流水线**（v0.8 新）— 流水线插入「② 预处理」step：ESRGAN / Real-ESRGAN 等多放大器预设 + ModelScope/HF 双源、智能流水（大图直接 resize 跳过放大模型）、SSE 实时进度。单 manifest 单 grid + 状态徽章（详见 [ADR 0004](docs/adr/0004-preprocess-manifest.md)）
- **🎲 InfoNoise 自适应训练**（v0.8 新）— 基于 I-MMSE 等价的反 CDF 时间步采样器，把抽样集中在信息量大的噪声窗口；走 `timestep_samplers/` plugin registry，默认关，存量训练零侵入
- **⏸️ 暂停 / 恢复训练 + 队列挂起**（v0.9 新）— UI 上一键暂停 running task（保 optimizer / step / monitor state + config snapshot），日后点恢复从同一步接续训练；队列挂起开关让 dispatcher 不拉新 pending，过夜 / 维护场景顺手用（详见 [ADR 0006](docs/adr/0006-queue-pause-resume.md)）

![Studio 训练页](docs/images/studio-train.png)

### 训练核心 (`runtime/anima_train.py`)

- LoRA + LyCORIS LoKr 双模式（走 [lycoris-lora](https://github.com/KohakuBlueleaf/LyCORIS) 官方库，含 DoRA / rs-LoRA / dropout；详见 [ADR 0001](docs/adr/0001-lokr-via-lycoris-lora.md)）
- 三种 attention 后端：xformers / flash_attn / PyTorch SDPA，UI 切换或 CLI 指定

### Studio Web 工作台 (`studio/`)

八步流水线 + 工具页：

1. **下载** — Booru 抓取（Gelbooru / Danbooru，凭据进 Settings）+ 本地 jpg/png/zip 上传
2. **预处理**（v0.8 新，可选）— 图片放大流水线：ESRGAN / Real-ESRGAN 等预设 + ModelScope/HF 双源 + SSE 实时进度
3. **筛选** — download / train 双面板，多选复制 / 移除，子文件夹管理
4. **打标** — WD14 / **CLTagger**（v0.5 新，本地 ONNX）/ LLM（OpenAI compatible，含 JoyCaption / OpenAI / Anthropic 等 preset）三选；GPU EP 自动 fallback
5. **标签编辑** — 缓存模式 + 还原点，批量加 / 删 / 替换；批量范围支持「当前筛选」（v0.8 新）
6. **正则集**（可选）— Booru 反向搜（自动 WD14 打标 + AR 聚类）/ **AI 先验生成**（v0.5 新，无 LoRA）
7. **训练** — preset 双向流，入队即开始；config 编辑 600ms debounce 自动落盘；Simple / Advanced 模式（v0.8 新）
8. **测试出图**（v0.5 新）— 单图 / XY 矩阵 / 推理 daemon；prompt 可从训练集拉

通用组件：
- 队列 / 任务详情（日志 / 监控 / 输出下载 / 全量 zip）
- 监控页（React 原生 loss / lr 曲线 + 采样图条按 step 切换）
- **Topbar 系统资源 pill**（v0.6 新）— CPU / GPU / MEM / VRAM 等宽 4 项实时刷新（pynvml → nvidia-ml-py，SSE 2.5s 增量推送）
- Settings 7 tab（数据集 / 打标 / 训练 / 监控 / 测试 / 页面 / 系统）：凭据 / 模型一键下载 / PyTorch 一键重装 CUDA 版 / xformers / flash_attn 一键装 / HF / ModelScope 双源 / WandB / **webui 一键自更新**（v0.7 新）
- 暗色 / 日间模式 + 字号密度切换

---

## 上游与致谢

- 核心训练脚本派生自 [**Moeblack/AnimaLoraToolkit**](https://github.com/Moeblack/AnimaLoraToolkit)。
- 主模型 / VAE：[circlestone-labs / Anima](https://huggingface.co/circlestone-labs/Anima)

---

## 快速开始

### 0. 系统先决条件（需自行安装）

下面这些**不是** Studio 自动装的，得先准备好：

- **NVIDIA GPU 驱动 + CUDA runtime**（**16 GB+ 显存推荐，12 GB 极限可跑**；A 卡 / Apple Silicon 不支持）
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

首次运行会自动：建 `venv/` → 装 `requirements.txt` → 按 GPU 检测装 onnxruntime → 构建前端 → 起后端 → 自动开浏览器到 <http://127.0.0.1:8765/studio/>。

> ⚠️ `requirements.txt` 里 torch 没指 CUDA index，自举装的是 CPU torch。**首次跑完后**激活 venv 装 CUDA 版覆盖一遍：
>
> ```bash
> # Windows
> .\venv\Scripts\activate
> # Linux / macOS: source venv/bin/activate
>
> pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130
> ```
>
> （cu130 = CUDA 13.0；旧驱动按需换 cu128 / cu126 等。）

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

> 注：`hf-mirror.com` preset 暂从 UI 隐藏 —— 该社区反代服务端最近的改动让所有 `huggingface_hub` 版本都拿不到 `commit_hash`，导致下载失败（细节见 `docs/todo/hf-mirror-recheck.md`）。endpoint 字段本身仍接受任意 URL，恢复后我们会把 preset 加回来。

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
4. **③ 打标**：选 WD14 / CLTagger / LLM（OpenAI compatible，含 JoyCaption preset）+ 阈值，一键自动打标
5. **④ 标签编辑**：批量加 / 删 / 替换；单图修；自动还原点
6. **⑤ 正则集**：两种生成方式可选 ——
   - **Booru 反向搜**：基于 tag 分布反向搜 booru，自动 WD14 打标 + 分辨率 AR 聚类
   - **AI 先验生成**：无 LoRA 直接用底模出图当 reg 集（v0.5 新）
7. **⑥ 训练**：选 preset 复制进 version 私有 config，改参数（debounce 600ms 自动落盘，无需点保存），入队即开始训练。Picker 标签会显示「· 已自定义」表示和原预设已分叉，预设池不会被改
8. 「队列」页查看任务，进**任务详情**看日志 / 监控 / 输出（含一键全量 zip 下载）

训完后侧栏 **测试**（v0.5 新）：跑单图 / XY 矩阵 / 推理 daemon 评测 LoRA，prompt 可从训练集直接拉，不用切 ComfyUI 反复测。

输出的 LoRA 权重已经是 `lora_unet_*` 格式，**直接拖进 ComfyUI 即可**，不需要任何转换。

---

## 项目结构

```
AnimaLoraStudio/
├── runtime/                       # Anima 运行时核心（独立进程；Studio 通过 subprocess 拉起，也可单独 CLI 跑）
│   ├── anima_train.py             # 训练入口（128 行 thin entry，编排 6 phase）
│   ├── training/                  # 训练栈子包（ADR 0003）：context / phases / loop / sample_runner
│   │   ├── adapters/              # plugin: lokr / loha / lora（AdapterProtocol 接入点）
│   │   ├── optimizers/            # plugin: adamw / prodigy / prodigy_plus_schedulefree
│   │   ├── schedulers/            # plugin: cosine / cosine_with_restart / none
│   │   ├── inference_samplers/    # plugin: er_sde（未注册名走 Euler 兜底）
│   │   └── phases/                # bootstrap / models / dataset / optimizer / resume / finalize
│   ├── anima_generate.py          # 出图：单图 / XY 矩阵
│   ├── anima_daemon.py            # 推理 daemon：常驻 GPU 加载 LoRA 和底模
│   ├── anima_reg_ai.py            # AI 先验生成：无 LoRA 直接用底模出 reg 集
│   └── train_monitor.py           # 训练状态写入器（被 anima_train import 调）
├── studio/                        # AnimaStudio Web 工作台（FastAPI + React）
│   ├── server.py                  # 守护进程入口
│   ├── services/                  # 业务逻辑（uploads / 打标 / 正则集 / inference_core /
│   │                              #   torch_setup / xformers_setup / flash_attention_setup 等）
│   ├── workers/                   # 后台任务子进程（download / tag / reg_build）
│   └── web/                       # React + Vite 前端
├── tools/                         # 用户 CLI / 启动期 setup helper
│   ├── download_models.py         # 一键下载主模型 / VAE / Qwen3 / T5 tokenizer
│   ├── install_flash_attn.py      # flash_attn wheel 一键装
│   ├── select_torch_index.py      # GPU-aware torch CUDA index 选择
│   ├── check_requirements_changed.py  # venv stale 检测（被 studio.bat / studio.sh 调）
│   ├── validate_local_models.py   # 验证本地 Qwen / T5 是否可离线加载
│   └── bench_*.py                 # 性能诊断（dev only）
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
    └── taeflux/                   # TAEFlux 中间步预览权重（v0.5，启动后台下载，gitignored）
```

运行时数据（gitignored）:
- `studio_data/` — SQLite + 用户 preset + 任务日志 + per-task monitor state + samples
- `models/diffusion_models/`, `models/vae/`, `models/wd14/` — 大权重文件
- `studio_data/projects/{id}-{slug}/versions/{label}/output/` — 训练产物 LoRA

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

当前版本 **0.8.2**（见 [CHANGELOG.md](CHANGELOG.md)）。

版本号唯一来源是 `studio/__init__.py:__version__`：

- 后端：FastAPI app `version=__version__`，`/api/health` 返回
- 前端：Sidebar 通过 `/api/health` 拉取，不再硬编码
- `studio/web/package.json` 的 `version` 字段由发版工具同步

发布新版本的 source of truth 是 [`release_notes.yaml`](release_notes.yaml)（不是 `CHANGELOG.md`，后者由工具派生）。流程：改 yaml → 跑 `python tools/bump_version.py bump --version X.Y.Z` 一键同步 `__init__.py` / `package.json` / `CHANGELOG.md` → 改 README badge + 当前版本句 → release PR。完整步骤见 [CONTRIBUTING.md](CONTRIBUTING.md) 的「Release 流程（Maintainer）」段。

---

## 硬件要求

- **GPU**：NVIDIA，**16 GB+ 显存推荐**（RTX 4060Ti 16G / 4070Ti / 4080 / 5070+ / 3090 / 4090 / 5090 等）；**12 GB 极限可跑**（4070 / 3060 12G 等，需关 sample 输出或减小 batch / 分辨率）。系统 GPU 占用低，VRAM 主要给训练；A 卡 / Apple Silicon 不支持
- **RAM**：16 GB+
- **存储**：SSD 强烈推荐（latent cache + sample 输出 IO 频繁）

---

## License

仓库整体以 **GPL-3.0** 发布（包含 / 派生自 ComfyUI 的 GPL-3.0 代码实现）。

仓库内同时包含部分 Apache-2.0 第三方实现（NVIDIA Cosmos / Wan2.1 等），请保留原文件头声明。详见：

- `LICENSE`（GPL-3.0）
- `LICENSE-APACHE`（Apache-2.0 文本，用于仓库内 Apache-2.0 组件）
- `THIRD_PARTY_NOTICES.md`

**模型权重**（Anima / Qwen / VAE）有各自的条款（含 Non-Commercial 等限制），请以对应模型卡 / HF repo 协议为准。
