# 0016 — 分阶段升级 LyCORIS v4 并隔离实验性 kernel backend

**状态**：Proposed
**日期**：2026-09-04
**决策者**：仓库 maintainer（待确认）

## 背景

LyCORIS 4.0.0 于 2026-09-01 发布。`requirements.txt` 当时只声明
`lycoris-lora>=1.9.0`，新环境因此越过 major 版本边界，导致 CI 连续失败。
v0.26.1 hotfix 已先将生产依赖收紧为：

```text
lycoris-lora>=3.4.0,<4.0
```

v4 的主要新增价值不是项目第一次获得 bypass mode：本项目的普通 LoRA 已显式使用
`bypass_mode=True`，Krea 2 FP8 LoKr 也因量化底模强制走 bypass。v4 新增的是在这些路径
及重建路径上自动调度 Triton、TileLang、`torch.compile` 或 eager Torch kernel。

本 ADR 记录正式升级前需要修改的兼容面、风险隔离方式和验收门槛。它不批准把 4.0.0
直接交付给用户。

## 已验证事实

### 保持兼容的部分

- `create_lycoris(...)`、`LycorisNetwork.apply_preset(...)`、`apply_to()`、
  `restore()`、`state_dict()` 等项目使用的 wrapper API 在 v4.0.0 保持可用。
- 当前 LoRA、LoKr、LoHa 的 CPU 创建、保存、加载和大部分聚焦测试保持通过。
- 使用未修改的项目代码分别对 v4.0.0 和 `4.0.1.dev20260902072855` 运行相同
  聚焦套件，结果均为 **131 passed / 3 failed**；3 项失败全部属于
  `tests/test_lycoris_patch.py` 的旧私有 import，dev wheel 没有改变这个兼容面。
- v3.4 与 v4.0.0 生成的 LoRA / LoKr safetensors 已做双向 bit-exact 加载验证。
- v3.4 LoKr 训练 checkpoint（含 AdamW optimizer state）可由 v4.0.0 恢复。
- 强制 `LYCORIS_KERNEL_BACKEND=torch` 后，RTX 5090 上 LoRA、LoKr、LoHa 的
  bf16 CUDA forward/backward 冒烟测试通过。
- Krea 2 FP8 推理的 `runtime/training/families/krea2/lora_fp8_merge.py` 按权重键
  自行 merge，不依赖 v4 kernel backend；现有权重格式未变。

### v4.0.0 的阻塞问题

1. **现有 LoKr device patch 无法导入。**
   `utils/lycoris_patch.py` 从 `lycoris.modules.lokr` 导入私有符号
   `make_kron` / `rebuild_tucker`；v4 已将实现迁到 `lycoris.functional.*`。
   聚焦测试因此在 `tests/test_lycoris_patch.py` 集中失败，线上 CI 症状与此一致。
2. **LoKr / LoHa rebuild rank-dropout 仍会创建 CPU mask。**
   v4.0.0 的对应 `get_weight()` 仍未给 `torch.rand(...)` 传 `device`；不能只删除旧
   patch。应改为按 major version 适配，或先推动上游修复后再移除本地补丁。
3. **无 Triton 的 CUDA 环境会被错误选到 compile backend。**
   在 RTX 5090、Torch 2.11 + CUDA 12.8、未安装 Triton 的环境中，v4 自动解析为
   `compile`，首次普通 LoRA forward 抛 `torch._inductor.exc.TritonMissing`。上游声称
   backend 会逐级回退，但该异常不在捕获范围内。
4. **v4.0.0 的 LoKr fused kernel 不支持部分实际 factor shape。**
   upstream PR #286 已于 2026-09-02 合并修复，但只进入
   `4.0.1.dev20260902072855` prerelease，尚无包含该修复的稳定版。
5. **与外层 `torch.compile` 仍非完全兼容。**
   upstream issue #287 仍开放。项目当前不编译整个 LyCORIS wrapper，但自动选择的
   per-op compile 仍会带来额外编译、冷启动和失败面。
6. **fused kernels 是可选且实验性的重量依赖。**
   Windows 上 `lycoris-lora[kernels]` 会额外解析 `triton-windows`、`tilelang`、
   `apache-tvm-ffi`、`z3-solver` 等；不应作为所有用户的默认依赖。

### T-LoRA 命名冲突

本项目现有 `lora_type=tlora` 不是 v4 新增的原生 `TLoraModule`：

- 项目实现以 `algo="locon"` 创建普通 `lora_up` / `lora_down`，再注入 timestep
  rank mask，保留既有 safetensors 和 checkpoint 键格式。
- v4 原生 T-LoRA 使用 `q_layer` / `p_layer` / `lambda_layer` 以及 base buffers，
  数学定义和序列化格式都不同。

本次 major 升级必须继续把项目 `tlora` 映射到兼容的 LoCon 实现。切换到 v4 原生
T-LoRA 是独立功能与迁移设计，不能借依赖升级静默替换。现有线性 Anima 路径可继续
工作；`_install_tlora_masks()` 中卷积 Tucker 分支的私有 import 应改到 v4 functional
API，并增加真正 inject + forward 的回归测试。

## 候选方案

### A. v4 stable + 默认 eager，kernel 显式 opt-in（建议）

先完成 wrapper、patch、迁移测试；程序在首次导入 LyCORIS 前将未显式配置的 backend
固定为 `torch`。高级用户可在进程启动前明确指定 `triton`，但 app 不承诺未验证的
TileLang / compile 路径。

**优点**：先拿到 v4 API 与后续维护基线；普通安装稳定；kernel 风险与 major 迁移解耦。
**缺点**：第一阶段不自动获得 fused kernel 加速，需要第二阶段安装与基准工作。

### B. v4 stable + 默认安装 `[kernels]` + auto

**优点**：用户无需配置即可尝试最快 backend。
**缺点**：扩大 Windows 安装体积和冲突面；4.0.0 已出现错误 fallback 与 shape 崩溃；
不适合作为 hotfix 后紧接的默认升级策略。

### C. 继续停留 v3.4

**优点**：当前已知稳定。
**缺点**：无法使用 v4 kernel 和后续 upstream 修复；本地私有 patch 长期背负。

## 暂定决策

采用 **A 的分阶段方案**，但只有下列 Phase 1 验收完成且存在包含 PR #286 的正式稳定版
后，才将本 ADR 改为 Accepted 并合并依赖升级。

## 需要修改的内容

### Phase 1 — v4 兼容基线（不承诺 kernel 加速）

1. **依赖与守卫**
   - 稳定版本目标暂定 `lycoris-lora>=4.0.1,<5.0`；不得依赖 dev/nightly。
   - 更新 `tests/test_dependency_constraints.py`，继续禁止无上界 major 漂移。
2. **backend 生命周期**
   - 在任何 `lycoris` module import 前解析 backend。
   - 未显式设置时默认 `torch`；保留外部 `LYCORIS_KERNEL_BACKEND` 覆盖能力。
   - 记录一次最终 backend、可用 backend 和降级原因；不得每层刷日志。
   - `auto` / `compile` 暂不作为 Studio 默认值。
3. **版本化 patch**
   - v3.4 patch 保留给 downgrade / release 分支。
   - 为 v4 使用 `lycoris.functional` API 重写 LoKr rank-dropout device 修复，或等价地
     依赖已验证的 upstream stable fix。
   - 加 v4 CUDA 回归：mask 与 weight 同 device，且 patch 幂等、只命中已知受影响版本。
   - 一并审计 LoHa 同类路径，避免只修 LoKr 后留下相同 GPU 错误。
4. **T-LoRA 兼容**
   - 保持项目 `tlora -> locon + custom mask`，不改现有权重格式。
   - 将 Tucker helper import 迁到 functional API。
   - 新增线性模块 inject、mask 生效、forward/backward、保存/恢复测试。
5. **迁移回归**
   - 固化 v3→v4 LoRA / LoKr 权重加载测试。
   - 固化 v3 checkpoint + optimizer state → v4 resume 测试。
   - 覆盖 LoRA、LoKr、LoHa、项目 T-LoRA、DoRA，以及 Krea 2 FP8 LoRA/LoKr。
6. **诊断与回滚**
   - 启动日志包含 LyCORIS 版本和实际 backend。
   - release note 说明 requirements hash 会让旧环境自动升级；保留 `<5.0` 回滚路径。

### Phase 2 — fused kernel 可选能力

1. 不修改核心 `requirements.txt` 为 `[kernels]`；提供显式可选安装入口。
2. 首批只评估 NVIDIA CUDA 的 Triton；TileLang、compile 分别设独立准入门槛。
3. 启动前做 import + 最小 CUDA forward/backward capability probe；失败必须回退到
   `torch`，不能等到真实训练首步才崩。
4. 在 Windows / Linux、项目支持的 Python / Torch / CUDA 组合上跑安装矩阵。
5. 用真实训练而非单算子数字比较：
   - Anima LoRA bypass；
   - Anima LoKr rebuild 与实验性 bypass；
   - Krea 2 FP8 LoRA / LoKr bypass；
   - warm 与 cold `it/s`、首轮编译耗时、峰值 VRAM、loss / 梯度有限性。
6. 只有稳定性和端到端收益都成立时，才考虑把 Triton 从 opt-in 提升为推荐项；
   `auto` 是否成为默认需另行确认。

## 验收门槛

- upstream 发布包含 PR #286 的正式稳定版。
- CPU CI 全绿；CUDA 至少覆盖普通安装（无 kernels extra）和 Triton opt-in 两组。
- 无 Triton 环境不得触发 `TritonMissing`。
- 所有支持算法 forward/backward 梯度有限，rank/module/dropout 语义有测试。
- v3 权重与训练 checkpoint 能继续加载；升级失败可通过依赖约束回滚到 v3.4。
- 真实训练基准单独报告 kernel device time 与全训练 `it/s`，不以官方算子微基准替代。

## 后果

- v4 API 迁移与性能实验不再绑成一次高风险发布。
- 第一阶段用户可能看不到额外速度，但不会被实验性编译器或大依赖阻断训练。
- 项目暂时继续维护少量 version-gated compatibility code；优先把通用 bug提交 upstream，
  上游稳定版覆盖后删除本地 patch。
- v4 原生 T-LoRA 留作独立算法名、格式和迁移方案，避免破坏老用户续训。

## 参考

- [LyCORIS v4.0.0 release](https://github.com/KohakuBlueleaf/LyCORIS/releases/tag/v4.0.0)
- [Kernel backend selection](https://github.com/KohakuBlueleaf/LyCORIS/blob/v4.0.0/docs/kernels/backends.md)
- [Kernel benchmarks](https://github.com/KohakuBlueleaf/LyCORIS/blob/v4.0.0/docs/kernels/benchmarks.md)
- [PR #286 — Fix LoKr factorization issue with kernels](https://github.com/KohakuBlueleaf/LyCORIS/pull/286)
- [Issue #287 — torch.compile compatibility](https://github.com/KohakuBlueleaf/LyCORIS/issues/287)
- ADR 0001：LoKr 适配器走 lycoris-lora，不切到 sd-scripts
