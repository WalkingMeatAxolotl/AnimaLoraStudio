# 0019 — 在 stable 升级前开放受限的 Triton 实验通道

**状态**：Proposed（用户已确认范围，R4a 待合入）
**日期**：2026-09-12
**决策者**：@WalkingMeatAxolotl

## 背景

ADR 0016 将 v4 基础兼容与 kernel 加速分开，并要求完整加速支持等待包含
LyCORIS #286 的 stable artifact。R3 制品审计确认 4.0.0 仍未包含该修复。
用户决定不改 LyCORIS pin，而是先开放不涉及 LoKr fused bypass 的受限实验范围。

该决策仅为 ADR 0016 的完整 Phase 2 增加一个独立实验切片，不解除 stable 升级门槛，
不承诺性能提升，也不将 Triton 设为推荐项或默认项。

## 候选方案

1. 等待所有上游修复进入 stable 后再提供任何 kernel 入口：风险最低，但阻塞独立的 LoRA/LoHa 验证。
2. 安装 `[kernels]` 并使用 auto：会带入 TileLang 等额外依赖，以及未验证的 compile/LoKr 路径，不采用。
3. 精确固定可选 Triton，仅允许已限制参数的 LoRA/LoHa Linear bypass：采用；默认 eager 不变。

## 决策

- `TrainingConfig.lycoris_backend` 是 Studio/CLI 选择的单一权威源，枚举为 `torch | triton`，默认 `torch`。
- 非 DoRA 的 LoRA/LoHa 可选择 Triton。三个 adapter dropout 都必须为 0；后端拒绝无效配置，底层 preflight 防御性回退时保留原参数。
- LoKr、DoRA、T-LoRA、Ortho 维持原数学路径；尤其不能为规避 kernel 问题而禁用 LoKr 的权重增量。
- 通过 adapter plugin 的 preflight 在大模型加载及首次 LyCORIS import 前配置 backend，沿用 R2 隔离 CUDA probe 与失败回退。
- 安装服务位于现有 runtime 服务层，提供状态、安装/重装和卸载端点；Settings 复用环境依赖行，不新增下载中心或常驻实验说明块。
- Windows x86_64 固定 `triton-windows==3.8.0.post28`，Linux x86_64 固定 `triton==3.8.0`；安装选择限定 Python 3.10–3.14、CUDA Torch 2.11。
- 安装仅用预编译 wheel、精确版本及 `--no-deps`，不让 pip 解析并替换 Torch；环境变化提示重启 Studio。
- 不修改 `lycoris-lora==4.0.0`，不安装 TileLang，不开放 UI 的 auto/compile 入口。

环境安装与训练选择作为同一 PR：单独提供安装不能让普通用户显式选择，单独提供选择又缺少可管理的依赖入口。

## 验证与限制

本地 Windows / RTX 5090 / Torch 2.11.0+cu128 / bf16：

- LoRA/LoHa 生产 probe 通过。
- Anima 2048/5120 四类代表层、两种算法共 16 组 eager/Triton 对照通过原定 bf16 容差。
- LoRA/LoHa × Torch/Triton 四组隔离合成短训完成，每组 2 次预热与 5 次有效更新；Triton dispatcher 未观察到降级。
- LoRA checkpoint 续训逐位一致；LoHa 恢复的状态逐位一致，但下一步梯度/optimizer 可能有微小差异，纯内存对照也能复现，不保证逐位确定性。
- 单次稳态 updates/s 为 LoRA Torch 1.832 / Triton 1.610，LoHa Torch 1.227 / Triton 1.252。固定顺序、单次短测且存在其他 GPU 显存占用，不能据此推导稳定加速。
- Triton 首步代价明显更高。Linux 与其他安装组合只做受控单元测试，未宣称真实硬件训练覆盖；Krea 2 FP8 完整训练不在本次实测范围。

## 后果

默认训练、旧权重格式和 R1 eager 基准保持不变。用户可以显式实验，但承担额外依赖、冷启动和非确定性成本。64×64 probe 不是所有形状的保证；真实运行错误仍正常传播。

完整安装矩阵、重复性能基准及 stable 升级继续由 roadmap 跟踪。支持矩阵在扩大前需要独立证据，不能把“可安装”当作“已在所有环境验收”。

## 参考

- [ADR 0016](0016-adopt-lycoris-v4-with-safe-kernel-rollout.md)
- [Roadmap #567](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/issues/567)
- [R3 #570](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/issues/570)
- [R4a #571](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/issues/571)
- [用户说明](../user-guide/triton-backend.md)
