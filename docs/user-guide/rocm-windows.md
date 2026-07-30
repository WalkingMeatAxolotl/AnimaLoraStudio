# Windows ROCm 使用指南

本分支在不改变 PyTorch 设备 API 的前提下支持 Windows ROCm。PyTorch 的 ROCm
wheel 仍使用 `torch.cuda`、`cuda` 设备字符串和 `torch.cuda.amp` 命名；是否为 AMD
后端必须读取 `torch.version.hip`，不能用 `torch.version.cuda is None` 判断为 CPU。

## 已验证组合

- Python 3.12.10（`E:\aiwork\python_embeded\python.exe`）
- PyTorch `2.9.1+rocm7.14.0a20260624`
- HIP 7.14 / Radeon RX 7900 XTX（gfx1100）
- Anima 2B 单文件 UNet、Qwen-Image VAE、ComfyUI 单文件 Qwen3-0.6B 文本编码器
- bf16 + PyTorch SDPA

CUDA 专用的 xformers 和 flash-attn 不用于该路径。训练配置必须设置：

```yaml
mixed_precision: bf16
attention_backend: none  # UI 中显示为 SDPA
num_workers: 0           # Windows
```

## 使用现有 Python 环境

默认启动器使用 `E:\aiwork\python_embeded\python.exe`。其他机器可先覆盖环境变量：

```bat
set ANIMA_ROCM_PYTHON=E:\path\to\python.exe
studio_rocm.bat --check
```

`studio_rocm.bat` 不会创建 venv，也不会重装 torch。若 Studio 后端依赖不全：

```bat
E:\aiwork\python_embeded\python.exe -m pip install -r requirements-rocm.txt
```

该命令复用 `requirements.txt`；已安装的 ROCm torch 满足其版本下限，pip 只补缺包。
安装后可用 `studio_rocm.bat` 启动 Web 工作台。

两个 ROCm 启动器会把 MIOpen 性能数据库和编译内核缓存写到仓库内已忽略的
`.cache\miopen`，避免嵌入式 Python 无权创建用户目录时出现
`miopenStatusUnknownError`。可用 `ANIMA_ROCM_CACHE_DIR` 改到其他可写位置；直接运行
`runtime\anima_train.py` 时也会在导入 torch 前应用同一默认值。

## 单文件 ComfyUI 文本编码器

`models\clip\anima_baseV10_txt.safetensors` 是 Qwen 文本编码器，不是 VAE。本分支在
收到单文件权重时自动启用 Comfy 兼容 Qwen loader，并从同一 ComfyUI 根目录查找：

- `comfy\text_encoders\qwen25_tokenizer`
- `comfy\text_encoders\t5_tokenizer`

因此无需复制 tokenizer 或联网下载。若权重离开标准 ComfyUI 目录，请改用包含
`tokenizer_config.json` 的完整 Qwen3-0.6B 模型目录，并显式配置 T5 tokenizer。

## CLI 训练

复制 [`examples/rocm/anima-windows-smoke.example.yaml`](../../examples/rocm/anima-windows-smoke.example.yaml)
并按机器修改路径，然后运行：

```bat
train_rocm.bat D:\path\to\anima-rocm.yaml
```

启动器先检查 ROCm、bf16 SDPA、模型结构和所有图片的 `.txt` caption，再进入训练。
示例的 `max_steps: 1` 是冒烟测试；正式训练时设为 `0`，并调整 epoch、rank、分辨率和
保存策略。`latent_cache_dir` 可把 `.npz` 放到独立可写目录，适合只读训练集；留空则
保持上游行为，写在图片旁。模型、数据集、latent cache、训练输出均不应提交到 GitHub。

示例从 256 分辨率开始，只用于确认一次完整 forward/backward/保存链路；验证通过后再
逐步提升到 512/1024。Windows ROCm 的 SDPA 回退在高分辨率上可能明显慢于 CUDA
flash-attn，不能用 1 步冒烟耗时估算正式训练速度。示例启用 `kv_trim`，把短 caption
的 cross-attention padding 从 512 tokens 收缩到有效 bucket；并仅在 256 冒烟时关闭
gradient checkpoint。提高分辨率后若显存不足，应重新打开 checkpoint。

## 限制

- 当前实机验收覆盖 Anima；Krea 2 的 Windows ROCm 训练尚未声明完成。
- xformers、flash-attn 及 CUDA wheel 安装入口在 ROCm 环境中不可用。
- NVIDIA NVML 不存在时，显存护栏回退到 `torch.cuda.mem_get_info()`；它能保护当前
  训练进程，但在 WDDM 下不保证精确看到其他进程的全部显存占用。
