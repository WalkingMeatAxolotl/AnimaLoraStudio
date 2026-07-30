# 0016 — 以外部 Python 运行时接入 Windows ROCm

**状态**：Accepted
**日期**：2026-07-30
**决策者**：ROCm 适配维护者

## 背景

原启动器只通过 `nvidia-smi` 选择 CUDA wheel，并把 `torch.version.cuda is None` 的
build 归类为 CPU。Windows ROCm wheel 却仍通过 `torch.cuda` API 暴露 GPU，同时以
`torch.version.hip` 标识后端；让原启动器管理它会误报 CPU，设置页的一键 CUDA 重装
还可能覆盖能工作的 ROCm torch。

Anima 的 ComfyUI 分发也常把 Qwen3-0.6B 保存成单个 safetensors，而原训练 loader
要求 Hugging Face 目录，无法直接使用用户已有权重。

## 候选方案

1. 修改通用 `studio.bat`，自动下载和维护 Windows ROCm wheel。不同 ROCm Windows
   发行版没有统一官方 index，自动覆盖现有环境风险高。
2. 建第二套训练实现，把所有 `torch.cuda` 调用改成 `torch.hip`。PyTorch 并不存在
   对等的公开 `torch.hip` 设备 API，这会制造错误抽象和大量分叉。
3. 保留 PyTorch 的 CUDA 设备命名，增加后端探针和独立外部环境启动器；ROCm 固定走
   SDPA，并支持 ComfyUI 单文件文本编码器及随应用发布的 tokenizer。

## 决策

采用方案 3：

- `utils.accelerator` 以 `torch.version.hip` / `torch.version.cuda` 区分 ROCm、CUDA、CPU。
- ROCm 继续使用 `cuda` device 字符串，但训练启动期只接受
  `attention_backend=none`（PyTorch SDPA）。
- `studio_rocm.bat` / `train_rocm.bat` 使用已有 Python，不创建 venv、不重装 torch；
  `tools/rocm_check.py` 在启动训练前验证运行时、SDPA、模型结构和数据集。
- MIOpen 用户数据库与内核缓存默认重定向到仓库内已忽略的 `.cache/miopen`，并允许
  用户通过环境变量覆盖。
- Studio 状态 API/UI 显示 ROCm，并隐藏 CUDA 重装控制；xformers/flash-attn 安装服务
  对 ROCm fail-fast。
- Anima loader 对单文件 Qwen 自动选择 Comfy 兼容实现，并从 ComfyUI 根目录发现
  qwen25 与 T5 tokenizer。

## 理由

这条路径把不可移植的部分限制在启动和后端选择层，训练算法、模型族派发、adapter
registry 与 sister script 契约均不分叉。外部环境所有权清晰，也避免通用 installer
对非官方 Windows ROCm wheel 做未经验证的升级或降级。

## 后果

- Windows ROCm 用户需先自备能运行的 torch 环境；应用只补普通 Python 依赖。
- 默认 `flash_attn` 没有全局改动，CUDA 用户行为保持不变；ROCm 示例显式写 SDPA。
- 单文件权重依赖标准 ComfyUI 目录结构来自动发现 tokenizer，离开该结构时需提供完整
  HF 模型/tokenizer 目录。
- NVML 不可用时显存水位使用 PyTorch fallback，跨进程可见性弱于 NVIDIA 路径。

## 验证

- 单测覆盖 ROCm/ CUDA/CPU 后端识别、Studio 状态和单文件 tokenizer 发现。
- `tools/rocm_check.py` 在 gfx1100 上执行 bf16 SDPA forward/backward。
- 以真实 Anima/VAE/Qwen 权重和带 caption 的训练集运行 `max_steps=1` 冒烟训练，确认
  完成模型加载、31 图独立 latent cache、文本编码、反向、AdamW step 与 LoRA 保存。
  256px + rank 8 + `kv_trim` 实测 loss `0.258150`、约 `0.29 it/s`；最终 safetensors
  为 23,037,208 bytes / 840 tensors，含 LyCORIS 与 Anima metadata。
- 生产前端通过 ESLint、TypeScript 与 Vite build；浏览器确认 ROCm/HIP/GPU 状态卡正确
  渲染且 CUDA 重装按钮隐藏。真实 Studio `/api/health` 与 `/api/torch/status` 通过。
