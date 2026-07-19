# Krea2 推理侧 fp8 支持方案（ComfyUI 逐位 parity 版）

- 状态：已定稿（用户批准 2026-07-16；随 fp8 实现 PR 入库）
- 日期：2026-07-16
- Parity oracle：`G:\ComfyUI-aki-v1.6\ComfyUI`（v0.27.1 + 内嵌 python 3.11.9 +
  torch 2.7.0/cu128 + comfy_kitchen 0.2.16）。**验收 = 同参数出图与它逐位一致**
  （Anima parity 同款要求）。
- 范围（用户裁定）：只做 fp8 两种（scaled + 纯 cast）。int8/nvfp4/GGUF/ConvRot 不做。
- 训练路径不变：C13 拒绝保留，文案改「训练不支持量化权重；推理支持 fp8」。

---

## 0. 用户文件实测（方案的直接对象）

`krea2TurboOfficialComfy_krea2TurboFp8.safetensors`（Comfy-Org 官方 Turbo fp8_scaled）：

- 686 张量 = 256 fp8(E4M3) block Linear 权重 + 256 个 F32 **标量** `*.weight_scale`
  + 174 bf16（RMSNorm scale / modulation / embed，永不量化）
- 量化声明在 safetensors `__metadata__._quantization_metadata` JSON：per-layer
  `{"format": "float8_e4m3fn"}`，**敏感层（attn.gate / attn.wo）额外标
  `full_precision_matrix_mult: true`**
- 无 input_scale、无老式 `scaled_fp8` marker
- **它是 Turbo** → 与 P4-4 的 distilled 检测联动（见 §5 检测层）

## 1. 本地 ComfyUI 对这个文件的精确路径（逐行核实）

1. `convert_old_quants`（utils.py:1403-1461）读 `_quantization_metadata` → 为每层
   注入 `.comfy_quant` uint8 键 → `detect_layer_quantization` 命中 →
   `quant_config={"mixed_ops":True}` → `pick_operations` 返回 `mixed_precision_ops`
2. **运行时门槛决定实际数值路径**（对 parity 是好事，路径确定）：
   - torch 2.7 < 2.8 → aimdo/DynamicVRAM 关（legacy ModelPatcher）
   - cu128 < cu130 → comfy_kitchen **CUDA kernel 禁用**，registry 落 **eager 纯
     torch 实现**（可逐行复刻）
   - `--fast` 默认空 → fp8 原生 matmul（`_scaled_mm`）不走
3. **无 LoRA 的每层前向**（确定性、零 RNG）：
   ```
   W_bf16 = W_fp8.to(bf16) * scale.to(bf16)      # ck eager quantization.py:59-63
   y = F.linear(x_bf16, W_bf16, bias)            # ops.py:492-496
   ```
   compute dtype = bf16（`unet_manual_cast`，Krea2 supported_dtypes 首项）；
   cast 顺序 = 先 `.to(device)`（fp8 字节过 PCIe）再 `.to(dtype)`（ops.py:360,376）
4. **纯 fp8 cast 文件**（无 metadata/scale）：`weight_dtype` 按 numel 多数派
   （utils.py:183-193）→ fp8 透传 → `manual_cast` ops → 同上但无 `* scale`
5. **LoRA × fp8**（ComfyUI 对我们 v1 open question 的回答 = **merge 回写**）：
   ```
   temp = dequant(W).to(lora_compute_dtype)       # fp16(Ada+) 或 fp32
   W' = calculate_weight(patches, temp, key)      # lora.py:438，intermediate fp32
   W_fp8' = requantize(W', scale="recalculate",   # scale = amax|W'|/448
                        stochastic_rounding, seed=CRC32(key))   # utils.py:1463 CRC32
   ```
   - **seed = 层名 CRC32，固定可复现** → LoRA+fp8 出图仍是确定性的
   - SR 实现在 cu128 环境走 ck **eager** 纯 torch 版（calc_mantissa，
     quantization.py:66+）→ 可逐行复刻
   - lowvram 才走"cast 后叠 delta 不回写"路径；32GB 全载 fp8 DiT（~13GB）走
     正常 merge 路径 → 我们复刻 merge 路径
6. **TE 编排**（回答 v1 的 offload 问题）：Qwen3-VL 以 **fp16 存储 + fp32 强制
   compute**（sd.py:258）在 GPU 编码；采样前 `free_memory` 把 CLIP LRU 卸载到
   **CPU**（model_management.py:829），DiT 独占 GPU；下个 prompt 再搬回

## 2. ⚠️ Step 0 —— fp8 之外的两笔 parity 债（先确认口径再动 fp8）

对照中发现 **bf16 基线本身就与 ComfyUI 不一致**，不先解决就无从归因 fp8 差异：

| # | 分歧 | ComfyUI | 我们 | 影响 |
|---|---|---|---|---|
| P-1 | **krea2 sigma 时刻表** | `sampling_settings={shift:1.15}` **固定 shift**（supported_models.py:1823，Raw/Turbo 同）| Raw 走 resolution-aware 动态 mu（diffusers 口径），仅 distilled 固定 1.15 | **同参数 sigma 表不同 → 永远无法逐位一致**。恰好我们的 distilled 公式 = Comfy 的全局公式 |
| P-2 | **TE 数值** | Qwen3-VL fp16 存储 + fp32 compute | bf16 | conditioning 数值不同 → 逐位差异 |

**P-1 需要拍板**：Generate 的 krea2 调度加一个 Comfy 口径（固定 shift 1.15，等价
于把 distilled sigma 公式变成 krea2 生成默认）？还是保留 diffusers 动态 mu 双轨？
建议：**Generate 页默认 Comfy 口径**（这页的存在意义就是 Comfy parity 验证），
训练中预览维持现状（那是训练侧语义，无 parity 要求）。
**P-2**：krea2 文本栈加 generate 场景的 fp16+fp32-compute 模式（训练侧 bf16 不动）。

## 3. 实施切分

- **PR-QP0（parity 基线）**：P-1 调度口径 + P-2 TE 数值 + TE offload 编排
  （encode → TE→CPU → 采样，复刻 comfy free_memory 语义）。验收：**bf16 Raw 权重
  同参数出图与本地 ComfyUI 逐位一致**（沿用 anima parity 的 pinned-oracle 方法）。
- **PR-Q1（fp8 主刀）**：
  - 检测层：safetensors `_quantization_metadata` 解析 + `weight_scale` 键收集 +
    纯 cast dtype 扫描；`is_distilled_path` 之外补「文件内容也能判 Turbo」不做——
    distilled 仍按 catalog variant / 显式选择（fp8 文件按其本名注册为 custom）
  - 存储层：fp8 张量 + scale 标量原样进模型（inference 专用加载路径；训练
    loader 不动）
  - 计算层：`QuantLinear`（或 forward patch）复刻 §1.3 公式，含
    `full_precision_matrix_mult` 层标记的尊重
  - LoRA：复刻 comfy merge 语义（dequant → calculate_weight 等价 → recalculate
    scale + eager SR，seed=CRC32(key)）；lora_merge POC（tools/lora_merge.py）的
    LoKr 精确 merge 数学直接复用。XY lora_scale 轴 = 从原始 fp8 备份重 merge
    （comfy weight_backup 同款）
  - C13 文案更新 + THIRD_PARTY_NOTICES（ComfyUI ops/utils/float + comfy_kitchen
    eager 实现的派生署名，GPL-3.0/相应许可核对）
  - 验收：**用户的官方 Turbo fp8 文件，同参数与本地 ComfyUI 逐位一致**（无 LoRA
    与挂 LoRA 各一组）
- **PR-Q2（纯 cast 补全）**：无 scale 文件的 manual_cast 等价路径（`.to(bf16)`
  无乘 scale）+ 混合 dtype 文件的多数派判定。小刀。

## 4. 已消解的 v1 open questions

1. LoRA 策略 → ComfyUI 答案 = **merge 回写 + SR(CRC32 seed)**（非 bypass）。
2. 只接 krea2 → 是（anima 无需求，接口族无关）。
3. TE offload → 复刻 comfy「编码后卸 CPU」，固定行为无旋钮。
4. dequant 目标 → bf16，且 **scale 也先转 bf16 再乘**（eager 公式逐位语义）。

## 5. 残留待实机确认（不阻塞 PR-QP0）

- 绘世启动器是否给 ComfyUI 传 `--fast` / `--enable-dynamic-vram`（会改变 oracle
  基线；请用户看一眼启动器设置或启动日志首屏）
- Comfy 侧生成 parity 样张时用与我们相同的 sampler/scheduler 组合（euler +
  固定 shift）
- 用户 ComfyUI workflow 实际加载的 TE 文件 dtype（P-2 假定 fp16 存储；若是
  fp8 版 TE 则口径另议）；以及其 fp16 权重与我们「HF bf16 目录 cast fp16」
  是否同源逐位一致（两者都应是官方 bf16 的 RTN cast，真卡时抽查张量校验）
