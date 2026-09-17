# Triton 实验性训练后端

Triton 是 LyCORIS 的可选计算后端，不是 Attention 后端。安装它不会自动改变训练配置；默认仍使用 eager Torch。本功能是受限实验通道，不保证更快，也不代表支持所有 LyCORIS 算法。

## 安装与选择

1. 打开 **设置 → 系统 → 环境 → Triton**。
2. 点击 **安装（自动匹配）**；已有版本可使用 **重装（自动匹配）**。
3. 安装完成后重启 Studio。
4. 先把训练的 `attention_backend` 设为 `none`（SDPA）或 `xformers`，再在高级 LoRA 配置中选择 `lycoris_backend: triton`。当前 Triton 与 `flash_attn` 互斥；恢复默认只需改回 `torch`，不必卸载 Triton。

安装目标是运行 Studio 的 Python 环境，不是模型目录或项目数据目录。仓库 venv 启动时，包安装到该 venv 的 `site-packages`。安装使用精确版本、预编译 wheel 和 `--no-deps`，不自动替换 Torch。

当前安装选择矩阵：

| 平台 | Python | Torch | 精确包版本 |
| --- | --- | --- | --- |
| Windows x86_64 | 3.10–3.14 | 2.11，CUDA 构建 | `triton-windows==3.8.0.post28` |
| Linux x86_64 | 3.10–3.14 | 2.11，CUDA 构建 | `triton==3.8.0` |

这是安装准入矩阵，不是上述所有组合都经过硬件验证的声明。安装包存在也不等于 GPU、驱动和实际训练算子可用；训练前仍要进行 CUDA 探测。不兼容的环境会禁用安装。安装失败时保留错误提示，不要通过升级 Torch 或装 `lycoris-lora[kernels]` 来绕过限制。

## 回滚与卸载

首选回滚方式是将训练配置改回 `lycoris_backend: torch`。保留已安装的 Triton
不会自动启用它，尤其在 Linux 上，不建议为了关闭本功能就卸载 Torch 自身可能依赖的包。

当前环境行与其他依赖项保持一致，只提供安装/重装和刷新。需要彻底卸载时，可在停止
训练后调用 `DELETE /api/triton/install`，然后重启 Studio。例如默认本地地址：

```powershell
Invoke-RestMethod -Method Delete -Uri 'http://127.0.0.1:8765/api/triton/install'
```

该端点卸载已检测到的 `triton` / `triton-windows` 发行包，不卸载 Torch。
但其他使用 Triton 的编译功能可能因此不可用；Linux 实际卸载后的其他 Torch 功能
未在本机验证。需要恢复时通过环境行重装，不能把本功能回滚等同于所有编译功能不受影响。

## 训练范围

| 配置 | Triton 实验通道 |
| --- | --- |
| 普通 LoRA | bypass |
| 普通 LoHa | bypass；Torch 下仍为原有 rebuild |
| LoKr、DoRA、T-LoRA、Ortho | 不开放，继续使用 Torch 和既有算法路径 |
| 三种 adapter dropout | 必须全部为 0 |
| Attention backend | `none`（SDPA）或 `xformers`；当前不兼容 `flash_attn` |

允许的配置示例：

```yaml
lora_type: lora  # 或 loha
lycoris_backend: triton
lora_dora: false
lora_dropout: 0.0
lora_rank_dropout: 0.0
lora_module_dropout: 0.0
attention_backend: none  # 或 xformers；不能使用 flash_attn
```

表单与后端 schema 共同限制不支持的组合；不能靠手改 YAML 让非零 dropout、DoRA，或 `attention_backend: flash_attn` 进入实验路径。环境卡片只管理依赖，训练约束属于训练配置，不在环境列表重复展示。

## 失败与诊断

训练在加载大模型、父进程首次导入 LyCORIS 前执行隔离 CUDA forward/backward probe。缺依赖、编译/执行异常、非有限输出或梯度、超时等失败会记录原因并回退 eager Torch；不会为了启用 Triton 偷偷改变算法或 dropout 参数。

探测只覆盖代表性调用，不能保证真实模型的每一种形状都受支持。真实训练中的 OOM、形状或运行时错误仍正常报错。默认 Torch 不启动这个探测子进程。

Studio 的 `lycoris_backend` 配置优先于 `LYCORIS_KERNEL_BACKEND` 环境变量。环境变量仍供不传显式配置的底层工具使用；用户界面不提供 `auto`、`compile` 或 `tilelang`。

## 已知的重复性限制

RTX 5090 / Torch 2.11.0+cu128 / Triton Windows 3.8.0.post28 的 bf16
LoHa bypass 测试中，checkpoint 恢复的权重和 optimizer 状态本身逐位一致；
但继续计算后，少数梯度及 optimizer 元素出现细小浮点差异。不经过磁盘的
相同状态内存对照也会出现此现象。本次下一步输出与 adapter 权重仍逐位一致，
但不能据此保证长期训练轨迹或 optimizer 状态的逐位确定性。
需要严格重复性的任务应保留 Torch 对照，不将 Triton 实验通道当作确定性保证。

## 验收边界

启用界面和安装成功不是性能结论。发布前还需记录实际硬件上的 LoRA/LoHa 数值与梯度对照、保存/加载/resume、真实形状及短训练结果，区分冷启动编译与稳态吞吐。现有 [R1 eager 基准](lycoris-benchmark.md) 仍严格固定 Torch，不能直接用它声称 Triton 加速。

上游 stable artifact 升级仍由 [R3 #570](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/issues/570) 独立跟踪；该实验通道不解除 LoKr fused bypass 的稳定版门槛。
