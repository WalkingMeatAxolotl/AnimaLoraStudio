# Third-Party Notices

本仓库包含/改写/派生了部分第三方代码与实现片段，以及若干基于公开论文/工程实现的算法移植。
请在分发时遵守其许可并保留必要的版权与许可声明。

---

## 仓库源头

### Moeblack / AnimaLoraToolkit

- **来源**：[`Moeblack/AnimaLoraToolkit`](https://github.com/Moeblack/AnimaLoraToolkit)
- **关系**：本仓库 fork 起点，核心训练脚本与早期 anima_train 入口派生自该项目，后续大幅
  重构。CLAUDE.md / README "上游与致谢" 节作高层提示，工程内部不再保留逐文件标注。

---

## 模型与权重

### circlestone-labs / Anima

- **来源**：[`circlestone-labs/Anima`](https://huggingface.co/circlestone-labs/Anima)
- **关系**：主扩散模型 + VAE。**模型权重许可独立**（含 Non-Commercial 等限制），以
  HuggingFace 模型卡协议为准；本仓库 NOTICES 不复述。

---

## 代码移植 / 算法实现

### ComfyUI (GPL-3.0)

- **来源**：[`comfyanonymous/ComfyUI`](https://github.com/comfyanonymous/ComfyUI)（现由 Comfy-Org 维护）
- **许可**：GPL-3.0
- **涉及文件**：
  - `modeling/anima/anima_modeling.py` — Anima DiT / LLMAdapter 结构与 ComfyUI
    `comfy/ldm/anima/model.py` 高度相关
  - `modeling/krea2/krea2_modeling.py` — Krea2 single-stream MMDiT 结构与参数模块
    命名派生自 ComfyUI `comfy/ldm/krea2/model.py`，固定参考 commit
    `87d23b81765161624889febfb3b81f19f3c8435b`；保留 `blocks.N.attn.*`、
    `blocks.N.mlp.*`、`txtfusion.*`、`txtmlp.*` 路径以兼容 ComfyUI 的 kohya LoRA 映射
  - `runtime/training/comfy_qwen.py` — Comfy-style Anima Qwen3 0.6B text encoder
    路径，对应 ComfyUI `comfy/text_encoders/anima.py` 的 Qwen3 encoder 行为
  - `runtime/training/text_encoding.py` — Anima prompt / tag 权重 / T5 token weights
    行为对齐 ComfyUI Anima text encoder 与 prompt weighting 规则
  - `runtime/training/sampling.py` — `_time_snr_shift` / `_flow_sigmas_simple` /
    `_flow_sigmas_sgm_uniform` / empty latent channel fix / CPU-seeded txt2img noise /
    batched CFG / sample helper 对齐 ComfyUI `ModelSamplingDiscreteFlow` + KSampler 行为
  - `runtime/training/inference_samplers/er_sde.py` — `sample_er_sde` +
    `default_noise_sampler` 参考 ComfyUI `k_diffusion_sampling`
    （删去 model_patcher 依赖）
  - `runtime/training/inference_samplers/dpmpp_3m_sde.py` —
    DPM-Solver++(3M) SDE / BrownianTree noise / first-sigma offset / log-SNR 细节
    对齐 ComfyUI `comfy/k_diffusion/sampling.py` + `comfy/model_sampling.py`
  - `studio/domain/comfy_parity.py`、`runtime/anima_generate.py`、
    `runtime/anima_daemon.py`、`runtime/training/sample_runner.py`、
    `studio/api/routers/generate.py` — 将测试出图与训练 sample 接到 Comfy-style
    parity runtime 的本项目 glue/config 代码
  - `runtime/anima_daemon.py` — 中间步预览的 `_WAN21_LATENT_RGB_FACTORS` /
    `_WAN21_LATENT_RGB_BIAS` 常量取自 ComfyUI `comfy/latent_formats.py` 的 `Wan21`
    （Qwen-Image VAE 复用 Wan2.1 latent 空间：`comfy/supported_models.py`
    `QwenImage.latent_format = Wan21`）；`_decode_latent2rgb_preview` 的投影 + 范围
    映射对齐 ComfyUI `latent_preview.py` `Latent2RGBPreviewer`

> 由于包含/派生自 GPL-3.0 代码，本项目整体以 GPL-3.0 发布（见 `LICENSE`）。

### NVIDIA Cosmos (Apache-2.0)

- **来源**：NVIDIA 相关实现（文件内含 SPDX 头）
- **许可**：Apache-2.0（见文件头 `SPDX-License-Identifier: Apache-2.0`）
- **涉及文件**：
  - `modeling/cosmos_predict2_modeling.py`

本仓库额外提供 `LICENSE-APACHE` 以便分发 Apache-2.0 许可文本。

### Alibaba Wan2.1 VAE（请再次确认上游许可）

- **来源**：[`Wan-Video/Wan2.1`](https://github.com/Wan-Video/Wan2.1) 的 VAE 实现（与
  `wan/modules/vae.py` 对应）
- **涉及文件**：
  - `modeling/wan/vae2_1.py`

该文件头目前仅包含版权声明（未显式 SPDX）。上游仓库通常宣称 Apache-2.0，但建议你在开源前
**再次核对上游仓库的 LICENSE/NOTICE**，确保分发合规。

### sorryhyun / anima_lora — OrthoLoRA / T-LoRA 实现 (MIT)

- **来源**：[`sorryhyun/anima_lora`](https://github.com/sorryhyun/anima_lora) — Seunghyun Ji
- **许可**：MIT — Copyright (c) 2026 Seunghyun Ji
- **涉及文件**：
  - `utils/ortho_adapter.py` — OrthoLoRA 层（冻结 SVD 基 + Cayley 旋转 + λ 零初始化）
    与"蒸馏保存为标准 LoRA"（闭式折叠旋转 + √λ 拆分）派生自上游
    `networks/lora_modules/ortho.py` `OrthoLoRAModule`；本仓库改动见该文件头
  - `utils/lycoris_adapter.py` — T-LoRA timestep rank mask 调度公式
    （`(1-t)^α`、batch 均值聚合、训练期专属）取自上游；mask 注入机制为本仓库实现
  - `runtime/training/adapters/tlora.py` — 接线层（本仓库代码，按上游方案组装）
  - 实验分支 `experimental/pr49-adapters` 中的 OrthoGrad / Ortho-Hydra 等
    亦来自同一上游，若日后合入主线需补对应文件头声明
- **研究归属**：T-LoRA 时间步自适应 rank 思想出自
  [`ControlGenAI/T-LoRA`](https://github.com/ControlGenAI/T-LoRA)（MIT，官方实现）；
  正交参数化族系参考 PSOFT（Wu et al.）。anima_lora 的实现是对官方的独立
  重新参数化（λ 零初始化 + 构造性正交），非官方代码搬运；本仓库训练/保存行为
  对齐 anima_lora 方案，推理语义与官方等价（详见 `utils/ortho_adapter.py` 文件头）

### ostris / ai-toolkit — Automagic optimizer 与 8-bit lr_mask (MIT)

- **来源**：[`ostris/ai-toolkit`](https://github.com/ostris/ai-toolkit) — Ostris (Jaret Burkett)
- **许可**：MIT — Copyright (c) 2024 Ostris, LLC
- **涉及文件**：
  - `utils/optimizer_utils.py`
    - `class Auto8bitTensor` — 8-bit 量化张量包装（per-tensor int8 + scale）
    - `class Automagic` — sign-agreement → `lr_bump` per-parameter 调度 + Adafactor
      factored 2nd moment + RMS clip
    - `_copy_stochastic` / `_copy_stochastic_bf16` / `_stochastic_grad_accumulation`
      — stochastic rounding 辅助函数（grad-accum hook 默认 disable，对齐上游已注释
      行为；详见 `class Automagic.__init__` 上方 comment）
- **修改点**：
  - bf16 路径采用 Kahan compensated summation（state['shift']），借鉴自下游
    `tdrussell/diffusion-pipe` 的同名移植，而非上游原版 stochastic rounding
  - `paramiter_swapping` feature 未移植

原文件头部 MIT license block 已贴在 `utils/optimizer_utils.py` `class Auto8bitTensor` /
`class Automagic` 上方，请勿删除。

### tdrussell / diffusion-pipe — Automagic bf16 Kahan path

- **来源**：[`tdrussell/diffusion-pipe`](https://github.com/tdrussell/diffusion-pipe)
  `optimizers/automagic.py`（基于 ostris/ai-toolkit 的同名移植 + bf16 Kahan 改进）
- **关系**：本仓库 Automagic 实现的 bf16 Kahan compensated summation 路径
  （`state['shift']` 累加 + `p.add_(shift)` + `shift.add_(grad.sub_(p))` 经典 Kahan 序列）
  与 diffusion-pipe 一致；其余算法核心来自上游 ai-toolkit。

### Lion optimizer (research attribution — 自实现)

- **论文**：Chen et al. 2023, *Symbolic Discovery of Optimization Algorithms*,
  [arXiv:2302.06675](https://arxiv.org/abs/2302.06675) (Google Brain)
- **Reference 实现对照**（仅用于校对，未直接复制代码）：
  - [`google/automl/lion`](https://github.com/google/automl/tree/master/lion) (Apache 2.0)
  - [`lucidrains/lion-pytorch`](https://github.com/lucidrains/lion-pytorch) (MIT)
- **涉及文件**：
  - `utils/optimizer_utils.py` `class Lion` / `create_lion`

`class Lion` 是自实现（~50 行），按论文 Algorithm 1 重写，不直接复制 reference 代码，
故 license 不强制 attribution；论文引用 + reference URL 作为学术礼貌已在 docstring 标注。

### nikhilvyas / SOAP — SOAP optimizer (MIT)

- **来源**：[`nikhilvyas/SOAP`](https://github.com/nikhilvyas/SOAP) — Nikhil Vyas
- **论文**：Vyas et al. 2024, *SOAP: Improving and Stabilizing Shampoo using Adam*,
  [arXiv:2409.11321](https://arxiv.org/abs/2409.11321)
- **许可**：MIT — Copyright (c) 2024 Nikhil Vyas
- **涉及文件**：
  - `utils/soap_optimizer.py` `class SOAP` — Adam-in-Shampoo-eigenbasis update
    （`_project` / `_project_back` / `_orthogonal_matrix(_qr)` / `_update_preconditioner`）
    派生自官方参考实现
  - `runtime/training/optimizers/soap.py` — registry 接线（本仓库代码）
  - `utils/optimizer_utils.py` `create_soap` — 工厂壳（本仓库代码）
- **修改点**：optimizer state 固定 fp32（bf16 LoRA/LoKr 训练数值稳定）；新增
  `precond_in_state=False` 把可重算的 GG/Q 剔出 state_dict 保持 ckpt 小 + resume
  冷重建。原文件头 MIT license block 已贴在 `utils/soap_optimizer.py` 顶部，请勿删除。

### yangluo7 / CAME — CAME optimizer (MIT)

- **来源**：[`yangluo7/CAME`](https://github.com/yangluo7/CAME) — Yang Luo
- **论文**：Luo et al. 2023, *CAME: Confidence-guided Adaptive Memory Efficient Optimization*,
  [arXiv:2307.02047](https://arxiv.org/abs/2307.02047)（ACL 2023 Outstanding Paper）
- **许可**：MIT — Copyright (c) 2023 Yang Luo
- **涉及文件**：
  - `utils/optimizer_utils.py` `class CAME` / `create_came` — Adafactor 式分解二阶矩
    + 置信度引导（instability EMA）update，step 公式派生自官方实现
  - `runtime/training/optimizers/came.py` — registry 接线（本仓库代码）
- **修改点**：optimizer state 固定 fp32（bf16 LoRA/LoKr 训练数值稳定，同 SOAP）；
  bf16 参数写回走 stochastic rounding（`_copy_stochastic`；fp16 为普通 cast，
  bit-trick 仅适用 bf16）；`load_state_dict` 后恢复 fp32 state（resume fixup）。
  算法公式未改动。

### facebookresearch / schedule-free — Schedule-Free 机制 (Apache-2.0, research attribution)

- **来源**：[`facebookresearch/schedule-free`](https://github.com/facebookresearch/schedule-free)
  `AdamWScheduleFree`
- **论文**：Defazio et al. 2024, *The Road Less Scheduled*,
  [arXiv:2405.15682](https://arxiv.org/abs/2405.15682)
- **许可**：Apache-2.0
- **涉及文件**：
  - `utils/soap_optimizer.py` `class SOAPScheduleFree` — 在 SOAP 预条件外套 Schedule-Free
    轨迹（丢一阶动量、z/x Polyak 平均、`train()`/`eval()` 权重 swap）。SF 机制 base-optimizer
    无关，按论文 + 参考 `AdamWScheduleFree` 的 in-place y/z 更新与 train/eval swap 自实现，
    非直接复制代码；SOAP 预条件部分见上条 MIT 归属
  - `runtime/training/optimizers/soap_sf.py` — registry 接线 + lr_scheduler=none 校验（本仓库代码）
  - `utils/optimizer_utils.py` `create_soap_sf` — 工厂壳（本仓库代码）

### InfoNoise timestep sampler (research attribution — 自实现)

- **论文**：*Information-Guided Noise Allocation for Efficient Diffusion Training*,
  [arXiv:2602.18647](https://arxiv.org/abs/2602.18647)
- **涉及文件**：
  - `runtime/training/timestep_samplers/infonoise.py` `class InfoNoiseScheduler`
- **关系**：基于论文 Algorithm 1（I-MMSE 恒等式 + log-σ bin EMA）自实现，未参考特定
  reference 代码。Anima 在 Flow Matching `t ∈ (0,1)` 空间内做了 `σ = t/(1-t)` 适配，
  与论文 σ-空间设计保持一致。

### kohya-ss / musubi-tuner — Krea2 training references (Apache-2.0)

- **来源**：[`kohya-ss/musubi-tuner`](https://github.com/kohya-ss/musubi-tuner)；
  固定参考 commit `8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6`
- **许可**：Apache-2.0（上游 README `License` 段声明 other code 适用 Apache-2.0）
- **涉及文件**：
  - `runtime/training/timestep_samplers/krea2_shift.py` — Krea2 `krea2_shift` 的 image-sequence
    length 线性 `mu`、`exp(mu)` 与 Möbius shift 公式；端点同时与 Hugging Face diffusers
    Krea2 inference pipeline（Apache-2.0，commit
    `bc529a5f677db9c4b3fc72c76962c4e2f61567e1`）交叉核对
  - `modeling/krea2/krea2_modeling.py` — 训练 tensor 布局、三轴 RoPE 与逐 block
    gradient checkpointing 参考 `src/musubi_tuner/krea2/krea2_mmdit.py`；模型配置与结构同时
    对照 Hugging Face diffusers `transformer_krea2.py`（Apache-2.0，同上固定 commit）
  - `runtime/training/families/krea2/loader.py` — meta-device + `assign=True` 的大权重
    加载策略参考 `src/musubi_tuner/krea2/krea2_utils.py`；结构指纹、前缀归一和错误诊断为本仓库实现
  - `runtime/training/families/krea2/preset.py` — 全部 Linear target 与统一
    `lora_unet` 前缀参考 `src/musubi_tuner/networks/lora_krea2.py`

实现按本项目 timestep sampler protocol / 纯 torch modeling 边界适配；具体派生关系见各文件头。

---

## Pip 依赖（许可随各自 wheel 分发）

下列依赖仅通过 `pip install` 引入，不在本仓库源码内 copy，本 NOTICES 不复述其 license。
但提及关键算法出处方便审计：

| 包 | 许可（参考） | 用途 |
|---|---|---|
| `lycoris-lora` | Apache-2.0 | LoRA / LoKr / LoHa / DoRA / rs-LoRA 适配器后端（`utils/lycoris_adapter.py` 封装） |
| `prodigyopt` | MIT | Prodigy 优化器（`utils/optimizer_utils.py` create_prodigy） |
| `prodigy-plus-schedulefree` | MIT | PPSF 优化器（同上 create_prodigy_plus_schedulefree） |
| `transformers` / `diffusers` | Apache-2.0 | 文本编码 / 推理 helper / scheduler 形式参考（cosine_with_warmup 与 transformers `get_cosine_with_min_lr_schedule_with_warmup` 数学等价） |
| `optimum-quanto` | Apache-2.0 | Automagic `QBytesTensor` 量化基模兼容路径 |
| `safetensors` / `bitsandbytes` / `wandb` 等 | 各自许可 | — |

---

如你希望把项目改为更宽松的许可（例如 MIT），需要先移除/替换所有 GPL-3.0 派生部分
（ComfyUI 相关），并重新梳理第三方依赖的许可兼容性。

