# 上手教程（Getting Started）

从零跑通一条 LoRA 训练流水线。本文是 [README](../../README.md) 「快速开始」的完整版。

## 先决条件

下面这些**不是** Studio 自动装的，得先准备好：

- **NVIDIA GPU 驱动 + CUDA runtime**（16 GB+ 显存推荐，8 GB 极限可跑；A 卡 / Apple Silicon 不支持）
- **Python 3.10+**（PATH 上能直接 `python` 调到）
- **Node.js 18+**（前端构建用，PATH 上能 `npm`）
- **Git**

硬件细节见 [README → 硬件要求](../../README.md#硬件要求)。

## 启动 Studio

```bash
git clone https://github.com/WalkingMeatAxolotl/AnimaLoraStudio
cd AnimaLoraStudio

# Windows
studio.bat

# Linux / macOS
./studio.sh
```

首次运行会自动：建 `venv/` → 按 GPU 驱动检测装对应 CUDA torch（cu118 至 cu130）→ 装 `requirements.txt` → 构建前端 → 起后端 → 自动开浏览器到 <http://127.0.0.1:8765/>。首次启动会弹引导 modal，按 checklist 一键安装底模 + ONNX Runtime + 训练加速包。

> 如果驱动检测失败导致装了 CPU 版 torch，可在 Settings → 系统 → PyTorch 一键重装 CUDA 版；也可通过 `studio.bat --torch cu128`（或 `studio.sh --torch cu128`）显式指定。

### 其它启动方式

等价于上面，便于直接 `python` 调：

```bash
python -m studio              # 构建前端（如缺）+ 起后端
python -m studio dev          # 前后端 watch：vite 5173 + uvicorn 8765 --reload
python -m studio build        # 仅构建前端
python -m studio test         # pytest + vitest
```

## 下载模型

打开后先去 **设置（Settings）→ Models**，点按钮一键下载训练所需的全部权重 + tokenizer（默认落到 `./models/`）：

| 项 | 来源 | 路径 | 大小 |
|---|---|---|---|
| Anima 主模型（latest = 1.0）| [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | `models/diffusion_models/` | ~4 GB |
| Qwen-Image VAE（Anima / Krea 2 共享） | 同上 | `models/vae/` | ~250 MB |
| Qwen3-0.6B-Base 文本编码器 | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | `models/text_encoders/` | ~1.2 GB |
| T5 tokenizer（仅 3 文件，不下权重）| [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) | `models/t5_tokenizer/` | <1 MB |
| Krea 2 Raw（LoRA 训练 / 训练中采样） | [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) | `models/diffusion_models/krea2-raw-bf16.safetensors` | ~26.3 GB |
| Krea 2 Turbo（测试推理） | [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) | `models/diffusion_models/krea2-turbo-bf16.safetensors` | ~26.3 GB |
| Krea 2 文本编码器 | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | `models/text_encoders/Qwen_Qwen3-VL-4B-Instruct/` | ~8.89 GB |

Krea 2 权重受 [Krea 2 Community License](https://huggingface.co/krea/Krea-2-Raw/blob/main/LICENSE.pdf) 约束。训练和训练中采样直接复用 Raw；Turbo 作为测试推理模型。在 Raw 上训练的 LoRA 可直接加载到 Turbo。Krea 2 与 Anima 共享现有 VAE，无需重复下载。选择 ModelScope 时，Raw / Turbo 从 [Comfy-Org/Krea-2](https://www.modelscope.cn/models/Comfy-Org/Krea-2) 下载。

WD14 打标模型不在这里——首次进 ④ 打标时自动从 HF 拉到 `models/wd14/`。

**国内加速**：直连 `huggingface.co` 慢，可去 Settings → 训练 → HuggingFace → endpoint 切到「自定义 URL」粘贴自建反代，或切到 Settings → 训练 → 下载源 → ModelScope（魔搭社区直连，需 `pip install modelscope`）。

也可走 CLI（与 UI 共用同一份代码，全部 flag 见 [tools/README.md](../../tools/README.md)）：

```bash
python tools/download_models.py                   # Anima（默认，HF 官方源）
python tools/download_models.py --family krea2    # Krea 2 Raw + 共享 VAE + Qwen3-VL
python tools/download_models.py --family krea2 --variant turbo
python tools/download_models.py --endpoint URL    # 走自建反代
python tools/download_models.py --modelscope      # 走魔搭社区
```

## 流水线：跟着 Stepper 走

打开 <http://127.0.0.1:8765/>，项目页「+ 新建项目」，侧栏 Stepper 引导走 8 步（标 ✱ 的可跳过）：

1. **下载** — Booru 抓图（先在 Settings 填 Gelbooru / Danbooru 凭据）或本地 jpg / png / zip 上传。
2. **筛选** — download / train 双面板，多选复制要训的图到 train/，子文件夹管理。
3. **预处理** ✱ — 总览（多选 + 一键撤销）+ 去重审核 + 放大（ESRGAN / Real-ESRGAN 多预设）+ 裁剪（手动框选 + 自动 AR 聚类预填）+ 涂抹。不需要可直接跳过。
4. **打标** — WD14 / CLTagger / LLM（OpenAI 兼容，含 JoyCaption preset）三选一 + 阈值，GPU EP 自动 fallback；顶部填 trigger_word 自动注入每张 caption 与采样图 prompt。
5. **标签编辑** — 缓存模式 + 还原点，批量加 / 删 / 替换，单图修。
6. **正则集** ✱ — 两种生成方式：**AI 先验生成**（默认，无 LoRA 直接用底模出图当 reg 集）或 **Booru 反向搜**（按 tag 分布反搜 booru + 自动 WD14 打标 + 分辨率 AR 聚类）。mirror / flat 结构，可编辑 / 删图 / 自动去重 / 双 tagger 可选。
7. **训练** — 选 preset 复制进 version 私有 config，改参数（debounce 600ms 自动落盘，无需点保存），入队即开始训练。Picker 标签显示「· 已自定义」表示和原预设已分叉，预设池不会被改。Simple / Advanced 模式。
8. **测试出图** — 单图 / XY 矩阵 / 推理 daemon。

「队列」页查看任务，进**任务详情**看日志 / 监控 / 输出（含一键全量 zip 下载）。

## 测试 LoRA + 用到 ComfyUI

训完后侧栏 **测试**：跑单图 / XY 矩阵 / 推理 daemon 评测 LoRA，prompt 可从训练集直接拉，不用切 ComfyUI 反复测。

输出的 LoRA 权重已经是 `lora_unet_*` 格式，**直接拖进 ComfyUI 即可**，不需要任何转换。

## 进一步

- 训练参数 / 显存配置 / 算法选项 → [training-tips.md](training-tips.md)
- 标签格式与最佳实践 → [tagging-guide.md](tagging-guide.md)
- 各优化器起步参数 → [optimizers.md](optimizers.md)
