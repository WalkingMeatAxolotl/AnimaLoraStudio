# AnimaLoraToolkit

Anima LoRA / LoKr 训练工具集，**附带完整 Web 工作台 (AnimaStudio)**。

从「准备数据 → 打标 → 正则集 → 训练 → 监控 → 下载 LoRA」一条流水线，在浏览器里点完。也支持纯 CLI 跑训练。

输出的 LoRA 权重直接 ComfyUI 可用（`lora_unet_*` 格式，无需任何转换）。

---

## 上游与致谢

本仓库的核心训练脚本最初派生自 [**FHfanshu/Anima_Trainer**](https://github.com/FHfanshu/Anima_Trainer)；之后做了大量重构、改造与扩展，已经与上游完全分歧（独立仓库）。仍感谢原作者打的底子。

- 主模型 / VAE：[circlestone-labs / Anima](https://huggingface.co/circlestone-labs/Anima)
- ComfyUI 兼容格式：[comfyanonymous / ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- 训练监控前端 HTML：单文件 `monitor_smooth.html`

---

## 主要特性

**核心训练 (`anima_train.py`)**
- LoRA + LyCORIS LoKr 双模式，输出原生 ComfyUI 格式
- Flow Matching + ARB 分桶 + 梯度检查点
- 断点续训（state.pt 含 optimizer / RNG / loss 历史）
- 多优化器：AdamW / AdamW8bit / Prodigy
- bf16 / fp16 训练
- 训练时 sample 出图 + 实时 loss 曲线

**AnimaStudio Web 工作台 (`studio/`)**
- 项目 / 版本 数据模型，每次训练对应一个 `Project` + 一个 `Version`
- ① 下载（Booru 抓取 + 本地 jpg/png/zip 上传）
- ② 筛选（download / train 双面板，多选复制 / 移除）
- ③ 打标（WD14 ONNX 本地 / JoyCaption vLLM 远程；多模型选）
- ④ 标签编辑（缓存模式 + 还原点）
- ⑤ 正则集（基于 train tag 分布贪心搜索 + AR 聚类）
- ⑥ 训练（preset 双向流，version 私有 config + 全局 preset 池）
- 队列 / 任务详情（日志 / 监控 / 输出下载 / 全量 zip）
- 设置（凭据 / WD14 多模型 / 模型一键下载 / 路径自定义）

---

## 快速开始

### 1. 安装

```bash
git clone https://github.com/<your-name>/AnimaLoraToolkit.git
cd AnimaLoraToolkit

python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux / macOS
# source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

需要 Node 18+（前端构建用）。

### 2. 下载模型

一条命令下完所有训练所需：

```bash
python tools/download_models.py
```

默认走 [hf-mirror.com](https://hf-mirror.com)（国内可直接下）。要用官方源加 `--no-mirror`。

下载内容（默认落到 `./models/`）：

| 项 | 来源 | 路径 | 大小 |
|---|---|---|---|
| Anima 主模型（latest = preview3-base）| [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | `models/diffusion_models/` | ~4 GB |
| Anima VAE | 同上 | `models/vae/` | ~250 MB |
| Qwen3-0.6B-Base 文本编码器 | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | `models/text_encoders/` | ~1.2 GB |
| T5 tokenizer（仅 3 文件，不下权重）| [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) | `models/t5_tokenizer/` | <1 MB |

更多选项：`python tools/download_models.py --help`，包含：
- `--variant {latest, preview3-base, preview2, preview}` 选 Anima 主模型版本
- `--skip-{main,vae,qwen,t5}` 单独跳过某项
- `--output PATH` 自定义目标根目录

也可以**直接进 Studio 设置页里点按钮下载**（与 CLI 共享同一份代码，可选根目录、状态实时刷新）。

### 3. 启动 Studio

```bash
python -m studio              # 构建前端（如缺）+ 起后端，自动开浏览器
```

或开发模式（前后端 watch）：

```bash
python -m studio dev          # vite 5173 + uvicorn 8765 --reload
```

Windows 上也可以双击 `studio.bat`。

打开 http://127.0.0.1:8765/studio/，跟着 Stepper 走：

1. 项目页「+ 新建项目」
2. **① 下载**：Booru 抓图（先在设置填 Gelbooru / Danbooru 凭据）或本地上传 zip
3. **② 筛选**：双 grid，选要训的图复制到 train/
4. **③ 打标**：选 WD14 模型 + 阈值，一键自动打标
5. **④ 标签编辑**：批量加 / 删 / 替换；单图修；自动还原点
6. **⑤ 正则集**：基于 tag 分布反向搜 booru，自动 WD14 打标 + 分辨率 AR 聚类
7. **⑥ 训练**：选 preset 复制进 version 私有 config，改参数 → 入队
8. 「队列」页查看任务，进**任务详情**看日志 / 监控 / 输出（含一键全量 zip 下载）

### 4. 用 LoRA

LoRA 权重直接 ComfyUI 可加载（**已经是 `lora_unet_*` 格式**），不需要任何转换。

---

## 项目结构

```
AnimaLoraToolkit/
├── anima_train.py            # 训练核心（被 Studio worker 通过 subprocess 拉起）
├── train_monitor.py          # 训练状态写入器（被 anima_train 调）
├── monitor_smooth.html       # 监控 UI（HTML，Studio iframe 嵌入）
├── studio/                   # AnimaStudio Web 工作台（FastAPI + React）
│   ├── server.py             # 守护进程入口
│   ├── services/             # 业务逻辑（uploads / 打标 / 正则集 / model_downloader 等）
│   ├── workers/              # 后台任务子进程（download / tag / reg_build）
│   └── web/                  # React + Vite 前端
├── tools/
│   └── download_models.py    # 一键下载训练所需模型（CLI 薄壳，与 Studio UI 共用 services）
├── docs/                     # 详细文档（标签格式 / 正则集原理 / Studio 设计等）
├── utils/                    # anima_train.py import 的 utility（model loader / optimizer 等）
└── models/                   # 模型代码 + tokenizer 预置文件 + 大权重落点（混合）
    ├── anima_modeling*.py    # tracked：Anima Cosmos transformer 的 PyTorch 实现
    ├── cosmos_predict2_modeling.py
    ├── wan/vae2_1.py         # tracked：Wan2.1 VAE 实现
    ├── text_encoders/        # tracked: Qwen tokenizer 小文件 + 用户下载的 model.safetensors
    ├── t5_tokenizer/         # tracked: T5 tokenizer 文件（无权重）
    ├── diffusion_models/     # 用户下载的 Anima 主模型（gitignored）
    ├── vae/                  # 用户下载的 VAE 权重（gitignored）
    └── wd14/                 # WD14 ONNX 模型（HF 自动下载，gitignored）
```

运行时数据（gitignored）:
- `studio_data/` — SQLite + 用户 preset + 任务日志 + per-task monitor state + samples
- `models/diffusion_models/`, `models/vae/`, `models/wd14/` — 大权重文件
- `studio_data/projects/{id}-{slug}/versions/{label}/output/` — 训练产物 LoRA

---

## 工具脚本

| 脚本 | 用途 |
|---|---|
| `tools/download_models.py` | 一键下载所有训练所需的主模型 / VAE / Qwen3 / T5 tokenizer。多版本可选 |
| `tools/validate_local_models.py` | 验证本地 Qwen / T5 是否可离线加载 |

也可以直接进 Studio「设置 → Models」UI 一键下载（与 CLI 共用同一份代码）。

---

## 文档

- [docs/json-caption-format.md](docs/json-caption-format.md) — JSON 标签格式 + 分类 shuffle
- [docs/tagging-guide.md](docs/tagging-guide.md) — Anima 标签格式与最佳实践
- [docs/training-tips.md](docs/training-tips.md) — 训练参数 / 断点续训 / 常见问题
- [docs/regularization-analysis.md](docs/regularization-analysis.md) — 正则集生成原理
- [docs/trainer-optimization-analysis.md](docs/trainer-optimization-analysis.md) — 训练性能调优
- [docs/studio-pipeline/](docs/studio-pipeline/) — Studio 七步改造的设计文档（开发者向）
- [studio/README.md](studio/README.md) — Studio 内部架构

---

## 硬件要求

- **GPU**：24 GB+ 显存（RTX 3090 / 4090 / 5090；Apple Silicon 暂不支持）
- **RAM**：32 GB+
- **存储**：SSD 强烈推荐（latent cache + sample 输出 IO 频繁）

---

## License

仓库整体以 **GPL-3.0** 发布（包含 / 派生自 ComfyUI 的 GPL-3.0 代码实现）。

仓库内同时包含部分 Apache-2.0 第三方实现（NVIDIA Cosmos / Wan2.1 等），请保留原文件头声明。详见：

- `LICENSE`（GPL-3.0）
- `LICENSE-APACHE`（Apache-2.0 文本，用于仓库内 Apache-2.0 组件）
- `THIRD_PARTY_NOTICES.md`

**模型权重**（Anima / Qwen / VAE）有各自的条款（含 Non-Commercial 等限制），请以对应模型卡 / HF repo 协议为准。
