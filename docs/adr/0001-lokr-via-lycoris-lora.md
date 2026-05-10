# 0001 — LoKr 适配器走 lycoris-lora，不切到 sd-scripts

**状态**：Accepted
**日期**：2025
**决策者**：仓库 maintainer

## 背景

当时仓库内有自实现的 `LoRALayer` / `LoKrLayer` / `LoRALinear` / `LoRAInjector`（`anima_train.py:875–1100`），是 LyCORIS 官方 LoKr 的简化重写，只覆盖了官方约 30% 特性面。要继续做 LoRA 工具集，必须扩能力（DoRA / dropout / LoHa / 多 GPU 等），三个方向：

1. **继续在仓库内手写**——加一个特性补一个类
2. **接 [`lycoris-lora`](https://github.com/KohakuBlueleaf/LyCORIS) 官方包**——保留 anima_train 训练循环、监控、Studio 后端
3. **整体切到 [`kohya-ss/sd-scripts`](https://github.com/kohya-ss/sd-scripts)**——废弃 anima_train，复用 sd-scripts 的训练核心

约束：
- Studio 是产品形态的核心（项目 / 版本 / 流水线 / SSE 监控）；切训练后端不能让 Studio 被边缘化为「sd-scripts 的 Web GUI」
- 老 ckpt 兼容**不是**强约束（用户可以重训）
- 维护者人力有限；想要"装 pip 即用"，避免长期维护 fork

## 候选方案

### 方案 A — 继续手写

在现有自实现上补 DoRA / rs-LoRA / dropout / Conv2d / Tucker。

**优点**：完全自己掌控；无新依赖。
**缺点**：每个新特性都是一笔工作；与社区生态（ComfyUI / sd-scripts / kohya-ss GUI）字段不对齐；自实现已经有隐性 bug（`_find_factor` 的 silent fallback to 1，让 LoKr 退化为 LoRA + 参数量爆炸）。

### 方案 B — 接 lycoris-lora 官方包

把 `LoRAInjector` 替换为 `create_lycoris(...) + apply_preset(ANIMA_PRESET)`，保留训练循环、optimizer、监控、断点续训等所有非适配器逻辑。

**已验证可行**：
- `LycorisNetwork(module, ...)` 接受任意单一 `nn.Module`，不依赖 SD/SDXL pipeline，接受 DiT
- `target_name` 接受当前层名列表（fnmatch + regex）
- 通过 `apply_preset({...})` 自定义层选择，绕过 PRESET 字典里的 SD 预设
- 与 PyTorch 标准 `state_dict` 互操作；ComfyUI LyCORIS 加载器直接读
- 依赖体积：纯 Python ~200KB，仅依赖 `einops` / `safetensors` / `torch`（已装）

**预估工时**：4–5 个工作日（不含老 ckpt 迁移）

### 方案 C — 切到 sd-scripts

把 `anima_train.py` 替换为 sd-scripts 的 `anima_train_network.py`，借力 kohya 团队的持续更新。

**Studio ↔ 训练 现有耦合面**（5 个文件接口，无函数级调用）：
| 接口 | 文件 | 当前协议 |
|---|---|---|
| 启动命令 | `studio/supervisor.py` | `subprocess: python anima_train.py --config X.yaml --monitor-state-file Y.json` |
| 训练配置 | `versions/{label}/config.yaml` | `studio/schema.py:TrainingConfig` 字段直接 dump |
| 进度状态 | `versions/{label}/monitor_state.json` | anima_train 写，Studio 轮询 mtime |
| 训练日志 | `studio_data/logs/task_*.log` | stdout 重定向，Studio `LogTailer` 行级追加 |
| 采样图 | `versions/{label}/output/samples/*.png` | anima_train 写，Studio HTTP 代理 |

切 sd-scripts = 这 5 个接口的协议全部要重定义。

**改动量**：
- 🟡 中：`schema.py:TrainingConfig` 整张重写（80+ 字段映射）；`argparse_bridge` 改 schema → TOML 双输出；`supervisor` 命令构造 + accelerate 配置生成；测试套件全部重写
- 🔴 大：**进度监控必须重做**——sd-scripts 不写 monitor_state.json，只 stdout 打印 tqdm。要么写脆弱的 stdout 解析器（方案 A），要么 fork sd-scripts 加 patch（方案 B），要么向上游提 PR（方案 C）。这是最大且最容易出问题的工作量
- 🔴 中：断点续训机制重做（state.pt 概念废弃，改 accelerate 的 save_state 目录）
- 🔴 中：sample 图监控改为目录扫描（不再有事件推送）

**预估工时**：3–5 周

**能拿到的额外能力**：
- 多 GPU（accelerate）
- `--blocks_to_swap` VRAM offload
- Adafactor + fused backward
- `--unsloth_offload_checkpointing`
- 多种 timestep 采样（sigma / sigmoid / shift / flux_shift）
- 多种 loss（l1 / l2 / huber / smooth_l1）
- per-module rank/lr (`network_reg_dims`)
- kohya 团队持续更新

**会失去的**：
- `state.pt` 简洁的断点续训语义
- `update_monitor()` 主动推送的丰富监控字段
- 自己写训练循环的灵活性（如未来加自定义 loss / sampler）
- Studio 作为「端到端流水线」的产品定位

## 决策

**采用方案 B**：接 lycoris-lora 官方包，保留 anima_train 训练循环。

## 理由

- **成本/收益最优**：4–5 天 vs 3–5 周。lycoris-lora 给到 LoHa / DoRA / dropout / rs-LoRA 等绝大多数缺失特性；多 GPU 等 sd-scripts 独有能力当前不是阻塞需求
- **保留产品形态**：Studio 仍然是端到端流水线；监控 SSE 协议、断点续训语义、训练循环全部不动。切 sd-scripts 会把 Studio 从「产品」降级为「sd-scripts 的 Web GUI」
- **可逆**：方案 B 是工作分支策略，未合并前可随时弃；合并后发现问题可 `git revert` PR 回退到 master 实现。**方案 C 不可逆**——schema 完全重写后没法回头
- **生态对齐**：lycoris-lora 输出的权重 ComfyUI / sd-scripts / kohya-ss GUI 都直接读，自动获得字段对齐
- **依赖轻**：lycoris-lora 是纯 Python ~200KB，仅依赖 einops / safetensors / torch（已装），无新增系统依赖
- **`_find_factor` 隐性 bug 顺带修掉**：自实现的 factor 搜索集合 `[target, 4, 2, 1]` 过窄，遇到不能整除的维度会 silent fallback 到 1，LoKr 退化为满矩阵 LoRA + 参数量爆炸。lycoris 的 `factorization()` 算法更稳健

## 后果

### 已落地

- `lycoris-lora>=3.0` 加入依赖
- `utils/lycoris_adapter.py`（`AnimaLycorisAdapter`）封装 lycoris 调用，替换自实现
- `utils/lycoris_patch.py` patch lycoris-lora 3.4.0 `LokrModule.get_weight` rank_dropout device bug（v0.5.0 修复，详见 CHANGELOG）
- Schema 暴露 `lora_algo` / `lora_dora` / `lora_rs` / `lora_dropout` / `lora_rank_dropout` / `lora_module_dropout` 等字段
- ComfyUI 加载验证通过；保存权重前缀 `lora_unet_*` 与下游对齐

### 新增的约束

- `lycoris-lora` 升级 API 变化时需要跟进；requirements 锁 `lycoris-lora>=3.0,<4.0` 防止 major 升级
- 老 ckpt 不兼容；旧分支 ckpt 加载会报清晰错误，提示重训

### 暂未做（仍可独立追加）

- Conv2d / Tucker 支持（Anima 是纯 DiT，主干 + TE + LLM Adapter 全是 `nn.Linear`，VAE 冻结，当前不需要）
- IA³ / GLoRA / BOFT 等冷门算法
- 多 GPU；如果未来确实需要，再单独开 ADR 评估是否切 sd-scripts

## 参考

- 官方源：[KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)
- 论文：*Navigating Text-To-Image Customization* (ICLR 2024)，LoKr 节在 §3.3
- 实现：`utils/lycoris_adapter.py`、`utils/lycoris_patch.py`
- v0.5.0 的 `attention_backend` 整合（PR #21）建立在本 ADR 落地基础上
