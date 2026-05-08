# AnimaLoraStudio

Anima LoRA / LoKr 训练工具集，**附带完整 Web 工作台 (AnimaLoraStudio)**。

从「准备数据 → 打标 → 正则集 → 训练 → 监控 → 下载 LoRA」一条流水线，在浏览器里点完。也支持纯 CLI 跑训练。

输出的 LoRA 权重直接 ComfyUI 可用（`lora_unet_*` 格式，无需任何转换）。

![Studio 训练页](docs/images/studio-train.png)

---

## 上游与致谢

本仓库的核心训练脚本派生自 [**Moeblack/AnimaLoraToolkit**](https://github.com/Moeblack/AnimaLoraToolkit)。

- 主模型 / VAE：[circlestone-labs / Anima](https://huggingface.co/circlestone-labs/Anima)

---

## 主要特性

**核心训练 (`scripts/anima_train.py`)**
- LoRA + LyCORIS LoKr 双模式，输出原生 ComfyUI 格式
- Flow Matching + ARB 分桶 + 梯度检查点
- 断点续训（state.pt 含 optimizer / RNG / loss 历史）
- 多优化器：AdamW / AdamW8bit / Prodigy
- bf16 / fp16 训练
- 训练时 sample 出图 + 实时 loss 曲线

**AnimaLoraStudio Web 工作台 (`studio/`)**
- 项目 / 版本 数据模型，每次训练对应一个 `Project` + 一个 `Version`
- ① 下载（Booru 抓取 + 本地 jpg/png/zip 上传）
- ② 筛选（download / train 双面板，多选复制 / 移除）
- ③ 打标（WD14 ONNX 本地 / JoyCaption vLLM 远程；多模型选）
- ④ 标签编辑（缓存模式 + 还原点）
- ⑤ 正则集（基于 train tag 分布贪心搜索 + AR 聚类）
- ⑥ 训练（preset 双向流，version 私有 config + 全局 preset 池）
- 队列 / 任务详情（日志 / 监控 / 输出下载 / 全量 zip）
- 设置（凭据 / WD14 多模型 / 模型一键下载 / 路径自定义）
- 监控页 React 原生（loss / lr 曲线 + 采样图缩略图条，按 step 切换）
- 暗色 / 日间模式 + 字号密度切换；config 编辑自动落盘（无需点保存）

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

下载源默认走 `hf-mirror.com`（国内反代，社区维护）—— 国内用户开箱即用。海外用户去 **Settings → 训练 → HuggingFace → endpoint** 切换到 `huggingface.co` 官方源（更快直连），或粘贴自建反代 URL。CLI 用户可以在 `python tools/download_models.py` 加 `--no-mirror` / `--endpoint URL` 显式覆盖。

下载内容（默认落到 `./models/`）：

| 项 | 来源 | 路径 | 大小 |
|---|---|---|---|
| Anima 主模型（latest = preview3-base）| [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | `models/diffusion_models/` | ~4 GB |
| Anima VAE | 同上 | `models/vae/` | ~250 MB |
| Qwen3-0.6B-Base 文本编码器 | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | `models/text_encoders/` | ~1.2 GB |
| T5 tokenizer（仅 3 文件，不下权重）| [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) | `models/t5_tokenizer/` | <1 MB |

也可以走 CLI（与 UI 共用同一份代码）：

```bash
python tools/download_models.py                   # 全量下
python tools/download_models.py --no-mirror       # 走 HF 官方源
python tools/download_models.py --variant preview2
python tools/download_models.py --skip-main --skip-vae
python tools/download_models.py --output /data/anima
```

WD14 打标模型不在这里——首次进 ③ 打标时自动从 HF 拉到 `models/wd14/`。

### 3. 跟着 Stepper 走

打开 <http://127.0.0.1:8765/studio/>：

1. 项目页「+ 新建项目」
2. **① 下载**：Booru 抓图（先在设置填 Gelbooru / Danbooru 凭据）或本地上传 zip
3. **② 筛选**：双 grid，选要训的图复制到 train/
4. **③ 打标**：选 WD14 模型 + 阈值，一键自动打标
5. **④ 标签编辑**：批量加 / 删 / 替换；单图修；自动还原点
6. **⑤ 正则集**：基于 tag 分布反向搜 booru，自动 WD14 打标 + 分辨率 AR 聚类
7. **⑥ 训练**：选 preset 复制进 version 私有 config，改参数（debounce 600ms 自动落盘，无需点保存），入队即开始训练。Picker 标签会显示「· 已自定义」表示和原预设已分叉，预设池不会被改
8. 「队列」页查看任务，进**任务详情**看日志 / 监控 / 输出（含一键全量 zip 下载）

输出的 LoRA 权重已经是 `lora_unet_*` 格式，**直接拖进 ComfyUI 即可**，不需要任何转换。

---

## 项目结构

```
AnimaLoraStudio/
├── scripts/
│   └── anima_train.py        # 训练核心（被 Studio worker 通过 subprocess 拉起）
├── studio/                   # AnimaStudio Web 工作台（FastAPI + React）
│   ├── server.py             # 守护进程入口
│   ├── services/             # 业务逻辑（uploads / 打标 / 正则集 / model_downloader 等）
│   ├── workers/              # 后台任务子进程（download / tag / reg_build）
│   └── web/                  # React + Vite 前端
├── tools/
│   ├── train_monitor.py      # 训练状态写入器（被 anima_train 调）
│   └── download_models.py    # 一键下载训练所需模型（CLI 薄壳，与 Studio UI 共用 services）
├── docs/                     # 详细文档（标签格式 / 正则集原理 / Studio 设计等）
├── utils/                    # anima_train 共享 utility（model loader / optimizer 等）
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
