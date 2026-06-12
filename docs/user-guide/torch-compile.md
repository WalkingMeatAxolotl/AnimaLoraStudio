# torch.compile 训练加速

## 概述

`torch_compile: true` 启用 PyTorch Inductor 编译加速。通过静态 shape 保证 + per-block 编译，
实测可获得 30%~2× 的训练速度提升（取决于 GPU 架构和 batch size）。

这是一个**实验性功能**——默认关闭，不影响现有训练行为。

## 工作原理

### 为什么普通训练无法直接 compile

torch.compile 依赖 Dynamo 追踪（trace）计算图。当输入 tensor 的 shape 变化时，
Dynamo 必须重新追踪（recompile），导致：
- 每种新 shape 触发一次 ~30s 的编译
- ARB 分桶产生数十种不同的 (H, W)，每种都触发 recompile
- 结果是 compile 反而比不 compile 更慢

### 解决方案：Constant-token Bucketing

核心思想：**保证所有 bucket 的 patch token 数恒定**。

Anima DiT 的 patch_spatial=16，所以一张 (W, H) 的图产生 `(W/16) × (H/16)` 个 patch token。
如果所有 bucket 都满足 `(W/16) × (H/16) == 固定值`，则 DiT block 看到的序列长度不变——
compiled graph 可以被复用。

本实现使用两个 token-count 家族：

| 家族 | Token 数 | 分辨率 | 宽高比覆盖 |
|------|---------|--------|-----------|
| 4032 | (W/16)×(H/16)=4032 | 1024×1008, 1008×1024, 1152×896, 896×1152, 1344×768, 768×1344 | 0.57~1.75 |
| 4200 | (W/16)×(H/16)=4200 | 1120×960, 960×1120, 1200×896, 896×1200, 1344×800, 800×1344 | 0.60~1.68 |

两个家族 = compile 只需追踪 2 个 graph（而非传统 ARB 的 37+ 个）。

### Per-block 编译

不编译整个模型的 forward（太大、编译慢），而是：

```python
for block in model.blocks:
    block.forward = torch.compile(block.forward, backend="inductor")
model.final_layer.forward = torch.compile(model.final_layer.forward, backend="inductor")
```

每个 DiT block 独立编译，编译粒度小、graph 简单、Inductor 优化效果好。

### Compile-friendly 前向路径

torch.compile + Dynamo 对某些操作的追踪存在问题：

| 原操作 | 问题 | 替换方案 |
|--------|------|----------|
| `einops.rearrange` | Dynamo 无法完整追踪 einops 的字符串解析 | `reshape` / `permute` / `unsqueeze` |
| `@torch.autocast(...)` 装饰器 | context manager 在 compile 下有 guard 问题 | 手动 `.to(dtype)` cast |
| 动态 KV trim（裁剪 seq dim）| 引入动态 shape → recompile | compile 模式下自动禁用 |

这些替换通过 `_compile_friendly` flag 条件分支实现——**默认路径完全不变**。
只有调用 `model.compile_blocks()` 后才走 compile-friendly 路径。

## 使用方法

在训练配置中设置：

```yaml
torch_compile: true
```

开启后自动触发：
1. BucketManager 切换到 constant-token bucket 表
2. 模型加载后（LoRA 注入后）对每个 block 调用 `torch.compile`
3. KV trim 自动禁用（前端显示"已被接管"）
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
| kv_trim | 互斥，自动禁用 |
| cache_latents | 兼容 |

## 硬件要求

- **GPU**: NVIDIA Ampere 架构及以上（RTX 3060+, A100 等）
- **CUDA Toolkit**: 需要完整安装（非仅 runtime）
- **PyTorch**: >= 2.0（推荐 2.2+，Inductor 更稳定）
- **显存**: 与非 compile 模式持平或略低（Inductor fusion 优化）

## 失败回退

如果 compile 失败（CUDA toolkit 不完整、不支持的 GPU 等），训练会：
1. 打印 warning 日志
2. 重置所有 `_compile_friendly` flag 为 False
3. 回退到正常训练模式继续

不会中断训练流程。

## 实现细节

### 文件改动

| 文件 | 职责 |
|------|------|
| `models/anima_modeling_core.py` | `_compile_friendly` 条件分支 + `compile_blocks()` |
| `runtime/training/dataset.py` | `CONSTANT_TOKEN_BUCKETS` 表 + `BucketManager(constant_token_mode=True)` |
| `runtime/training/phases/models.py` | compile 调用时机（LoRA 后）+ fallback |
| `runtime/training/phases/dataset.py` | constant_token_mode 联动 |
| `runtime/training/loop.py` | kv_trim 互斥逻辑 |
| `studio/domain/training.py` | `torch_compile` 配置字段 + 接管字段 |

### Constant-token bucket 计算方法

寻找所有满足以下条件的 (W, H) 对：
- `W % 16 == 0` 且 `H % 16 == 0`（patch_spatial=16 整除）
- `(W/16) × (H/16) == target_token_count`
- 宽高比覆盖常用范围（竖图 9:16 到横图 16:9）

target_token_count 选取原则：
- 不选 4096（精确等于 1024² 时只有 1024×1024 一个解，宽高比太少）
- 选 4032 = 63×64 = 56×72 = 48×84 ...（因子对丰富）
- 选 4200 = 60×70 = 56×75 = 50×84 ...（补充 4032 不覆盖的中间宽高比）

### Unpatchify reshape 等价性证明

原 einops：`"B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)"`

等价 reshape/permute：
```python
# (B, T, H, W, p1*p2*t*C) -> (B, T, H, W, p1, p2, t, C)
x = x.reshape(B, T, H, W, p1, p2, t, C)
# -> (B, C, T, t, H, p1, W, p2)
x = x.permute(0, 7, 1, 6, 2, 4, 3, 5)
# -> (B, C, T*t, H*p1, W*p2)
x = x.reshape(B, C, T*t, H*p1, W*p2)
```

einops 的 `(p1 p2 t C)` factored dim 中，左侧变化最慢——
与 C-contiguous reshape `(..., p1, p2, t, C)` 的内存布局一致。
permute 目标维度顺序 `(B, C, T, t, H, p1, W, p2)` 确保合并后等价于
einops 的 `(T t)`, `(H p1)`, `(W p2)` 语义。

### AdaLN reshape 等价性

原：`rearrange(x, "b t d -> b t 1 1 d")` = 在 dim=2,3 插入 size-1 维度用于广播。
替换：`x.unsqueeze(2).unsqueeze(3)` — bit-exact 等价。

