# torch.compile 训练加速

## 概述

`torch_compile: true` 启用 PyTorch Inductor 编译加速。通过 constant-token bucketing +
per-block 编译，实测可获得 30%~2× 的训练速度提升（取决于 GPU 架构和 batch size）。

这是一个**实验性功能**——默认关闭，不影响现有训练行为。

## 工作原理

### 为什么普通训练无法直接 compile

torch.compile 依赖 Dynamo 追踪（trace）计算图。当输入 tensor 的 shape 变化时，
Dynamo 必须重新追踪（recompile），导致：
- 每种新 shape 触发一次 ~30s 的编译
- ARB 分桶产生数十种不同的 (H, W)，每种都触发 recompile
- 结果是 compile 反而比不 compile 更慢

### 解决方案：Constant-token Bucketing + Native Flatten

核心思想：**保证所有 bucket 的 patch token 数恒定，再把 5D 形状展平为 fake-5D**。

Anima DiT 的 patch_spatial=16，所以一张 (W, H) 的图产生 `(W/16) × (H/16)` 个 patch token。
本实现使用两个 token-count 家族（24 种分辨率，每家族 12 种）：

| 家族 | Token 数 | 宽高比覆盖 |
|------|---------|-----------|
| 4032 | 63×64, 56×72, 48×84, 42×96, 36×112, 32×126 | 0.25~3.94 |
| 4200 | 60×70, 56×75, 50×84, 42×100, 40×105, 35×120 | 0.29~3.43 |

`compile_blocks()` 设置 `_native_flatten=True`，forward 在 block loop 前把 5D
`(B, T, H, W, D)` 展平为 fake-5D `(B, 1, seq_len, 1, D)`。Block 只看到 seq_len
这一个维度——同家族的所有分辨率共享同一个 compiled graph。

两个家族 = compile 只需追踪 **2 个 graph**（而非传统 ARB 的 24+ 个）。
无 padding，flash self-attention 不会看到填充 token——与 eager 路径 bit-exact。

### Per-block 编译

```python
for block in model.blocks:
    block.forward = torch.compile(block.forward, backend="inductor", dynamic=False)
model.final_layer.forward = torch.compile(model.final_layer.forward, ...)
```

每个 DiT block 独立编译，编译粒度小、graph 简单、Inductor 优化效果好。
`dynamic=False` 保证 Dynamo 按精确 int shape 做 guard（不做符号化推导）。

### 单代码路径设计

本实现**不使用条件分支**——所有 rearrange 已被等价的原生 PyTorch ops 替换
（reshape / permute / unsqueeze），无论 compile 是否启用都走同一条路径。
`_native_flatten` 仅控制 block loop 前后的展平/还原，不影响 block 内部逻辑。

## 使用方法

在训练配置中设置：

```yaml
torch_compile: true
```

开启后自动触发：
1. BucketManager 切换到 constant-token bucket 表（24 种分辨率）
2. 模型加载后（LoRA 注入后）调用 `compile_blocks()`
3. KV trim 自动禁用（会打 warning）
4. 首步训练有 ~30s 编译预热，之后每步稳定加速

## 兼容性

| 特性 | 兼容性 |
|------|--------|
| gradient_checkpointing | 兼容（`use_reentrant=False`）|
| LoRA / LyCORIS | 兼容（compile 发生在 LoRA 注入之后）|
| mixed_precision bf16 | 兼容（推荐）|
| mixed_precision fp16 | 兼容 |
| xformers | 部分（block 内走 compile SDPA，block 外 xformers 仍生效）|
| flash_attn | 同上 |
| kv_trim | 互斥，自动禁用并打 warning |
| cache_latents | 兼容 |

## 硬件要求

- **GPU**: NVIDIA Ampere 架构及以上（RTX 3060+, A100 等）
- **CUDA Toolkit**: 需要完整安装（非仅 runtime）
- **PyTorch**: >= 2.0（推荐 2.2+，Inductor 更稳定）
- **显存**: 与非 compile 模式持平或略低（Inductor fusion 优化）

## 失败回退

如果 compile 失败（CUDA toolkit 不完整、不支持的 GPU 等），训练会：
1. 打印 warning 日志
2. 重置 `_native_flatten = False`
3. 回退到正常训练模式继续

不会中断训练流程。

## 实现细节

### 文件职责

| 文件 | 职责 |
|------|------|
| `models/cosmos_predict2_modeling.py` | 原生 ops 前向 + `compile_blocks()` + `_native_flatten` |
| `runtime/training/dataset.py` | `CONSTANT_TOKEN_BUCKETS`（24 entries）+ `BucketManager(constant_token_mode=True)` |
| `runtime/training/phases/models.py` | compile 调用时机（LoRA 后）+ fallback |
| `runtime/training/phases/dataset.py` | constant_token_mode 联动 |
| `runtime/training/loop.py` | kv_trim 互斥逻辑 + warning |

### `_native_flatten` 机制

```
prepare_embedded_sequence → x: (B, T, H, W, D)
                                    ↓ flatten(1,3).unsqueeze(1).unsqueeze(3)
                              x: (B, 1, seq_len, 1, D)   ← block loop 看到的
                                    ↓ _unflatten_native_shape (torch.compiler.disable)
                              x: (B, T, H, W, D)         ← final_layer / unpatchify 看到的
```

`_unflatten_native_shape` 被 `@torch.compiler.disable(recursive=True)` 标记——
这保证 Python-int 的 (T, H, W, seq_len) 元组不进入 compile zone，避免
per-bucket 值 guard 导致 recompile。

### Dynamo cache_size_limit

`compile_blocks()` 自动计算所需的 cache budget：`2 * n + 8`
- `n` = 不同 token count 家族数（当前为 2）
- `2 *` 覆盖 fwd+bwd 共享同一 bytecode
- `+ 8` 覆盖 requires_grad / stride specializations

用 `max()` 确保不覆盖更高的外部预设。
