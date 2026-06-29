# 0003 — anima_train.py 模块化重构（plugin 边界 + adapter hook protocol）

**状态**：Accepted（已落地：`runtime/training/` 子包 + plugin registry + adapter hook）
**日期**：2026-05-14
**决策者**：@WalkingMeatAxolotl

## 背景

`runtime/anima_train.py` 已经长到 **2901 行**，其中 `main()` 占 **793 行**（L2105-L2897）。53 个 def/class 平铺在单一模块里，职责跨 10 个不同维度：CLI / 模型加载 / 文本编码 / 数据集 / 采样调度 / 训练状态 IO / wandb / 训练循环 / 周期 IO / 信号处理。

### 这次推迫重构的两个直接因素

1. **PR #49** (saltysalrua + Claude Sonnet 4.6 co-author，1508 行) 想往里塞 T-LoRA、Ortho-Hydra、OrthoGrad 三个 LoRA 变体。三方 review（正方/反方/PM）后决定**只把稳定性补丁合 dev**（PR #55，已合），三个变体放在 `experimental/pr49-adapters` parking lot 分支（见 [memory/pr49-handling]）。这次操作里被迫做了 9 commits 的 cherry-pick 拆分 + 手术分裂 c5e81c2、清理跨 commit 的死 dispatch 残留，主要痛点都来自 main() 的 dispatch 逻辑跟 variant-specific 实现混在一处。

2. PPSF 接入（PR #46）已经识别了两层重构需求，当时压在 [memory/anima_train_refactor_pending]：
   - **P0**：把 main() L2336-2374 的 optimizer kwargs dispatch 上提到 `utils/optimizer_utils.py`
   - **P1**：把 main() 拆 phase 到 `runtime/training/` 子包

PR #55 合 dev 后两个事情都可以做了，base 已经稳定。

### 现状盘点（基于 dev `ada49a3`）

`runtime/` 目录：

| 文件 | 行数 | 职责 |
|---|---:|---|
| `anima_train.py` | 2901 | 巨型 mega-script |
| `anima_daemon.py` | 651 | Studio 长驻进程（生成 / xy-matrix），`import anima_train as _T` 借加载逻辑 |
| `anima_generate.py` | 320 | 一次性推理 CLI，同上 |
| `anima_reg_ai.py` | 252 | 正则集 AI 打 caption，同上 |
| `train_monitor.py` | 168 | monitor_state.json 写入器（已独立，无需动） |

`anima_train.py` 53 个 def/class 按职责自然成簇：

| 簇 | 行号 | 函数/类 | 行数 |
|---|---|---|---:|
| bootstrap | 60-156 | `ensure_dependencies` / `load_yaml_config` / `apply_yaml_config` / `_lazy_imports` / `init_progress` | 96 |
| 进度+监控展示 | 183-369 | `render_loss_curve` / `render_curve_panel` / `WandBMonitor` / `init_wandb_monitor` | 186 |
| 模型加载基础设施 | 370-612 | `forward_with_optional_checkpoint` / `enable_xformers` / `find_diffusion_pipe_root` / `load_module_from_path` / `_strip_prefixes` / `_pick_best_prefix_remap` / `_load_safetensors_state_dict` / `resolve_path_best_effort` / `_load_weights_best_effort` | 242 |
| 模型加载主入口 | 614-775 | `ensure_models_namespace` / `load_anima_model` / `load_vae` / `load_text_encoders` | 161 |
| 文本编码 | 777-1071 | `encode_qwen` / `_parse_weighted_tag` / `_build_qwen_text_from_prompt` / `tokenize_t5_weighted` | 294 |
| flow/sigma/采样调度 | 822-961, 1677-1937 | `_time_snr_shift` / `_flow_sigmas_simple` / `_default_noise_sampler` / `_sample_er_sde_const_x0` / `sample_image` / `sample_t` / `make_noise` / `compute_loss_weight` | 400 |
| 训练状态 IO | 1073-1142 | `save_training_state` / `load_training_state` | 69 |
| dataset | 1144-1675 | `BucketManager` / `ImageDataset` / `RepeatDataset` / `MergedDataset` / `BucketBatchSampler` / `CachedLatentDataset` | 531 |
| collate | 1939-1962 | `collate_fn` / `collate_fn_cached` | 23 |
| CLI | 1963-2103 | `parse_args` / `_try_rich` / `_ask_*` / `_guess_default_paths` / `prompt_for_args` | 140 |
| main() | 2105-2897 | — | 793 |

### 外部对 `anima_train` 模块的依赖（不能破的契约）

三个 sister script 通过 `import anima_train as _T` 调用 `anima_train` 顶层名字：

- `runtime/anima_daemon.py:44, 160-218, 530, 650` — `_T.find_diffusion_pipe_root` / `_T.resolve_path_best_effort` / `_T.load_anima_model` / `_T.enable_xformers` / `_T.load_vae` / `_T.load_text_encoders` / `_T.sample_image`
- `runtime/anima_generate.py:43, 122-191, 344` — 同上 7 个
- `runtime/anima_reg_ai.py:43, 232-283` — 同上 7 个

测试直接 import 这些名字：

- `tests/test_anima_train_migration.py` — `parse_args` / `apply_yaml_config`（CLI 别名 + YAML 合并语义）
- `tests/test_lycoris_resume.py` — `save_training_state` / `load_training_state`（断点续训）

**约束 1**：重构后 `anima_train.py` 必须顶层 re-export 上述 9 个名字，否则 sister script 和测试全断。

### 其他被波及的引用

- `docs/adr/0002-webui-self-update.md` L17, L388 写死了 `runtime/anima_train.py#L2374-L2401`（Ctrl+C handler）—— 行号在重构后会变，要更新链接（或者改用 grep 关键词锚点）
- `models/cosmos_predict2_modeling.py:34` 注释 "cli.py / runtime/anima_train.py 启动期会调一次 enable" —— 注释级，不影响
- `README.md` / `docs/architecture/studio-pipeline.md` 等说明性提到 anima_train —— 不影响行为

## 候选方案

### 方案 A — 现状维持

继续在 `main()` 里加变体 dispatch。

- 优点：0 工时
- 缺点：parking lot rebase 越来越疼；下次再接论文级 LoRA 变体（T-LoRA 类）还得在 main() 中间塞代码；review diff 永远跨 10 个职责域
- 拒绝

### 方案 B — 只拆文件（不引入 plugin 抽象）

把 53 个 def/class 按职责搬到 `runtime/training/` 子包，main() 保留单函数形态，dispatch 仍写在里面。

- 优点：纯机械搬运，0 行为变化；单 PR review 友好
- 缺点：解决了"代码定位难"，但**没有解决"加变体要改 main()"**；下次加 LoRA 变体仍然要改训练编排逻辑
- 部分采纳（作为 PR-A）

### 方案 C — 拆文件 + 全套 plugin（含 adapter hook protocol）

在 B 的基础上引入 4 个 plugin 子包（adapter / optimizer / lr_scheduler / inference_sampler）+ adapter 用 Protocol 留扩展 hook（per-step、loss 加项）。

- 优点：加 / 删变体变成本地操作；parking lot 不再阻塞 dev 演进；论文级 adapter 变体（per-step 行为类）有明确扩展位
- 缺点：抽象成本 ~3 周；引入新的「约定」（registry 字典、Protocol hook）后续 maintainer 要学习
- **采纳**

## 决定

**采纳方案 C**，分三个 PR 落地，三次合到 dev：

| PR | 范围 | 行为变化 | 预估行数（净） |
|---|---|---|---:|
| **PR-A** 文件搬运 | 53 个 def/class 按职责移到 `runtime/training/` 子包；`anima_train.py` 变 thin shim re-export 9 个 sister/test 用名字 | 0 | -2600 / +2600（搬运） |
| **PR-B** main() phase 拆分 | 引入 `TrainingContext` dataclass；把 main() 793 行拆成 6 个 phase function；train_loop 抽出来 | 0 | -700 / +700（重组） |
| **PR-C** plugin 化 + Protocol | 4 个 plugin 子包 + `AdapterProtocol` 含 hook；main() dispatch 完全消失 | 0 | -200 / +500（净 +300） |

三个 PR 都**保持训练行为字节级一致**（同一 yaml 跑出来 loss 曲线相同），靠 `tests/test_anima_train_migration.py` 锁 CLI/YAML 行为，靠端到端冒烟训练验证 loop 不变。

### 设计原则

- **User-facing 0 变化**：`studio/schema.py:TrainingConfig` 保持单一大类，所有字段平铺；前端 UI / CLI 参数 / `config.yaml` 字段名 0 变动。Plugin 只在 `runtime/training/` 内部存在。
- **Sister-script / 测试 API 0 变化**：`anima_train.py` 顶层 re-export 全部 9 个被外部用到的名字。
- **Plugin = 显式字典，不要装饰器**：`BUILDERS = {"lokr": lycoris.build, ...}`，删除变体 = 删一行字典 + 删一个文件。装饰器 + side-effect import 看起来优雅，但 load-order 敏感、"全部注册项在哪里"找不到。
- **"新变体引入新 schema 字段 → subfolder"**：是 plugin 化的硬规则。adapter / optimizer / lr_scheduler / inference_sampler 每变体都带自己的配置字段 → 一变体一文件。timestep_sampling / loss_weighting / noise 是参数化的纯数学函数，不引入 schema 字段 → 单文件多函数。

## 目标布局

```
runtime/
  anima_train.py                 ← ~150 行 thin shim：main() 编排 + re-export
  anima_daemon.py                ← 不动（继续 import anima_train as _T）
  anima_generate.py              ← 不动
  anima_reg_ai.py                ← 不动
  train_monitor.py               ← 不动
  training/
    __init__.py                  ← package init；暴露 sister script 用的名字
    context.py                   ← TrainingContext dataclass / StepContext dataclass
    bootstrap.py                 ← deps / yaml / args 预处理
    cli.py                       ← parse_args / interactive / prompt_for_args
    observability.py             ← WandBMonitor + 曲线渲染 + emit
    model_loading.py             ← prefix 推断 / safetensors / 路径解析（内部 utils）
    models.py                    ← load_anima_model / load_vae / load_text_encoders（公开）
    text_encoding.py             ← qwen / t5 + tokenize_weighted
    state.py                     ← save/load_training_state（公开）
    dataset.py                   ← BucketManager / ImageDataset / Merged / Sampler / Cached / collate
    sampling.py                  ← sigma utils + sample_image（公开）
    timestep_sampling.py         ← sample_t 各 mode（一文件多 fn）
    loss_weighting.py            ← compute_loss_weight 各 scheme
    noise.py                     ← make_noise（offset + pyramid）
    loop.py                      ← train_loop.run(ctx)
    phases/
      __init__.py
      models.py                  ← model + vae + text_encoder 加载
      dataset.py                 ← build_datasets + dataloader
      optimizer.py               ← build_optimizer + scheduler + grad_clip
      resume.py                  ← state recovery + signal handler
      finalize.py                ← 最终保存 + 进度清理
    
    # 4 个 plugin 子包
    adapters/
      __init__.py                ← BUILDERS dict + build_adapter(args) + AdapterProtocol
      protocol.py                ← AdapterProtocol + StepContext
      lycoris.py                 ← 注册 lokr/loha/lora；内部调 utils.lycoris_adapter
    optimizers/
      __init__.py                ← BUILDERS + VALIDATORS dict
      adamw.py
      prodigy.py
      prodigy_plus_schedulefree.py
    schedulers/
      __init__.py                ← BUILDERS dict + build_scheduler(args, optimizer, total_steps)
      cosine.py
      cosine_with_restart.py
    inference_samplers/
      __init__.py                ← BUILDERS dict（暂只 er_sde）
      er_sde.py
```

`anima_train.py` 重构后大致是这样：

```python
# runtime/anima_train.py
"""Thin shim：本模块保留 main() 入口和向后兼容的 re-export。
真正实现在 runtime/training/ 子包内。"""

from training.cli import parse_args, prompt_for_args
from training.bootstrap import apply_yaml_config, ensure_dependencies, load_yaml_config
from training.model_loading import find_diffusion_pipe_root, resolve_path_best_effort
from training.models import enable_xformers, load_anima_model, load_text_encoders, load_vae
from training.sampling import sample_image
from training.state import load_training_state, save_training_state
from training.context import TrainingContext
from training import loop, phases

def main():
    args = parse_args()
    if args.config:
        args = apply_yaml_config(args, load_yaml_config(args.config))
    if args.interactive or _any_required_missing(args):
        args = prompt_for_args(args)
    ensure_dependencies(auto_install=args.auto_install)

    ctx = TrainingContext.bootstrap(args)
    ctx = phases.models.run(ctx)
    ctx = phases.dataset.run(ctx)
    ctx = phases.optimizer.run(ctx)
    ctx = phases.resume.run(ctx)
    loop.run(ctx)
    phases.finalize.run(ctx)


if __name__ == "__main__":
    main()
```

re-export 一段保证 `anima_daemon.py:_T.load_anima_model` 等调用 0 改动；main() 走 phase 编排。

## Plugin 模式细节

### Registry：显式字典

```python
# training/adapters/__init__.py
from typing import Callable
from .protocol import AdapterProtocol, StepContext
from . import lycoris

BUILDERS: dict[str, Callable[..., AdapterProtocol]] = {
    "lokr": lycoris.build,
    "loha": lycoris.build,
    "lora": lycoris.build,
}

def build_adapter(args) -> AdapterProtocol:
    if args.lora_type not in BUILDERS:
        raise ValueError(
            f"未知 lora_type={args.lora_type!r}；已注册: {sorted(BUILDERS)}"
        )
    return BUILDERS[args.lora_type](args)
```

```python
# training/adapters/lycoris.py
from utils.lycoris_adapter import AnimaLycorisAdapter
from .protocol import AdapterProtocol  # 实际是 Protocol 仅用于类型提示

def build(args) -> AdapterProtocol:
    return AnimaLycorisAdapter(
        algo=args.lora_type,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        factor=args.lokr_factor,
        dropout=args.lora_dropout,
        rank_dropout=args.lora_rank_dropout,
        module_dropout=args.lora_module_dropout,
        weight_decompose=args.lora_dora,
        rs_lora=args.lora_rs,
    )
```

### AdapterProtocol：必需方法 + 默认 no-op hook

```python
# training/adapters/protocol.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
import torch
from torch import nn, Tensor

@dataclass(frozen=True)
class StepContext:
    """传给 adapter hook 的最小上下文。"""
    global_step: int
    total_steps: Optional[int]
    epoch: int
    sigma_t: Tensor          # shape [B]，本 micro-batch 的 sigma
    args: object             # parse_args 出来的 Namespace；按需读字段

@runtime_checkable
class AdapterProtocol(Protocol):
    # ─── 必需 ───
    def inject(self, model: nn.Module) -> None: ...
    def get_param_groups(self, weight_decay: float) -> list[dict]: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
    
    # ─── 可选 hook：默认 no-op；变体按需重写 ───
    def on_step_begin(self, ctx: StepContext) -> None:
        """每 micro-batch 前向之前调用。
        
        T-LoRA / AdaLoRA / B-LoRA 在此按 sigma_t / step 调整 rank mask、
        激活子集、列丢弃等"运行时结构调整"。默认 no-op。"""
        return None
    
    def regularization_loss(self, ctx: StepContext) -> Optional[Tensor]:
        """返回要加到主 loss 上的正则项；None=无。
        
        OFT 返回 orthogonality penalty；Ortho-Hydra 返回 expert balance loss；
        默认 None。train_loop 收到 None 时不做任何额外操作。"""
        return None
    
    def excludes_weight_decay(self, param_name: str) -> bool:
        """该 param 是否应排除 weight_decay。
        
        替代当前 `injector.use_lokr` 硬编码检查。LoKr 实现里：
        return "w1" in param_name。默认 False。"""
        return False
```

`AnimaLycorisAdapter` 已经实现前 4 个必需方法；hook 全部 default no-op，所以**当前 lokr/loha/lora 行为零变**——唯一改的是 `optimizer_setup.py` 里把硬编码的 `injector.use_lokr` 判断换成 `adapter.excludes_weight_decay(name)`。

### train_loop 里的 hook 调用

```python
# training/loop.py（节选）
for batch_idx, batch in enumerate(dataloader):
    captions, latents = ...
    t = sample_t(bs, device, mode=ts_mode, shift=ts_shift)
    
    step_ctx = StepContext(
        global_step=global_step,
        total_steps=total_steps,
        epoch=epoch,
        sigma_t=t,
        args=args,
    )
    
    # ★ adapter hook 1：可调整运行时结构
    ctx.adapter.on_step_begin(step_ctx)
    
    noise = make_noise(latents, ...)
    noisy = (1 - t) * latents + t * noise
    target = noise - latents
    
    with torch.autocast("cuda", dtype=dtype):
        pred = forward_with_optional_checkpoint(model, noisy, t, cross, pad_mask, ...)
        loss_per_sample = F.mse_loss(pred.float(), target.float(), reduction="none")
        # ... loss_weight / loss_weighting / 正则集权重 ...
        loss = loss_per_sample.mean()
        
        # ★ adapter hook 2：变体可加正则项
        reg = ctx.adapter.regularization_loss(step_ctx)
        if reg is not None:
            loss = loss + reg
    
    if not torch.isfinite(loss):
        ...
    
    loss = loss / args.grad_accum
    loss.backward()
    ...
```

两个 hook 调用 + StepContext 构造 ≈ 8 行额外代码，对 LyCORIS-only 路径 0 性能影响（Python 函数调用 + None 返回）。

## 不同变体的真实落地案例

### Case 1：DoRA（NVIDIA 2024）

**情况**：已支持。当前 schema 有 `lora_dora: bool`，传给 `AnimaLycorisAdapter(weight_decompose=...)`，LyCORIS 内部走 DoRA 路径。

**Plugin 视角**：DoRA 不增加 schema 枚举值（不是新 `lora_type`），是 lokr/lora 的开关。Plugin 化后**不需要任何改动**。

### Case 2：新加一个纯权重类变体——VeRA（共享 A/B，只训 d/b 向量）

**情况**：未实现，作为示例。VeRA 引入新的层结构。

**步骤**：

1. 新文件 `utils/vera_adapter.py` 写实际算法（继承 `nn.Module`，实现 `inject` / `get_param_groups` / `save` / `load`）
2. 新文件 `training/adapters/vera.py`：
   ```python
   from utils.vera_adapter import AnimaVeRAAdapter
   def build(args):
       return AnimaVeRAAdapter(
           rank=args.lora_rank,
           shared_seed=args.vera_shared_seed,
           d_init=args.vera_d_init,
       )
   ```
3. `training/adapters/__init__.py` 字典加一行：`"vera": vera.build`
4. `studio/schema.py`：
   - `lora_type: Literal[..., "vera"]` 多一个值
   - 加 `vera_shared_seed: int = Field(default=42, _meta(group="LoRA", show_when="lora_type=='vera'"))` 等字段

**main() / loop / 其他 phase 0 改动**。Hook protocol 0 触及（VeRA 不需要 per-step 调整）。

### Case 3：新加一个 per-step 变体——T-LoRA（按 sigma_t 调 rank）

**情况**：parking lot 上有完整实现 (`utils/tlora_adapter.py`)。dev 这次重构后将来要回收时怎么做。

**T-LoRA 算法核心**：训练初期允许全 rank（学到的 LoRA 内容多），逐步按 noise level 屏蔽部分 columns，避免对低 noise 区间过拟合。需要 train_loop 每 step 把当前 sigma 喂给 adapter。

**步骤**：

1. `utils/tlora_adapter.py` 已存在于 parking lot，搬到 dev
2. 新文件 `training/adapters/tlora.py`：
   ```python
   from pathlib import Path
   from utils.tlora_adapter import AnimaTLoRAAdapter as _Inner
   from .protocol import StepContext
   
   class TLoRABundle:
       """包装 utils.AnimaTLoRAAdapter 以满足 AdapterProtocol，
       特别是 on_step_begin 这个 hook。"""
       
       def __init__(self, args):
           self._inner = _Inner(
               rank=args.lora_rank,
               alpha=args.lora_alpha,
               min_rank=args.tlora_min_rank,
               alpha_rank_scale=args.tlora_alpha_rank_scale,
               sig_type=args.tlora_sig_type,
           )
       
       def inject(self, model): self._inner.inject(model)
       def get_param_groups(self, wd): return self._inner.get_param_groups(wd)
       def save(self, path: Path): self._inner.save(path)
       def load(self, path: Path): self._inner.load(path)
       
       def on_step_begin(self, ctx: StepContext) -> None:
           # T-LoRA 的核心 hook：按当前 sigma 调 mask
           self._inner.set_sigma(ctx.sigma_t)
           # 也可按 step 调 mask（如 warmup 期 full rank）
           if ctx.total_steps:
               progress = ctx.global_step / ctx.total_steps
               self._inner.set_mask_progress(progress)
   
   def build(args):
       return TLoRABundle(args)
   ```
3. `training/adapters/__init__.py` 加 `"tlora": tlora.build`
4. `studio/schema.py` 加 `Literal[..., "tlora"]` + `tlora_*` 字段

train_loop 不动（已有 `on_step_begin` 调用点）。

**为什么不能 cherry-pick 原 PR #49 而要重写**：原 PR 把 `injector.set_sigma()` 写在 train_loop 内联代码，跟 main() 编排耦合。重构后 train_loop 通过 `adapter.on_step_begin(ctx)` 调，T-LoRA 实现 hook，main() 永远不见到 set_sigma 这种 variant-specific 调用。

### Case 4：变体要在主 loss 上加正则项——OFT（Orthogonal Fine-Tuning）

**情况**：未实现，示例。OFT 用正交矩阵参数化 LoRA delta；训练时需要在 loss 上加 orthogonality penalty 防止数值漂移。

**步骤**：

1. `utils/oft_adapter.py` 实现 `AnimaOFTAdapter`，含 `compute_orth_penalty()` 方法
2. `training/adapters/oft.py`：
   ```python
   class OFTBundle:
       def __init__(self, args):
           self._inner = AnimaOFTAdapter(rank=args.lora_rank, ...)
           self._penalty_weight = args.oft_orth_penalty
       
       # ... 必需 4 个方法委托给 _inner ...
       
       def regularization_loss(self, ctx: StepContext) -> Optional[Tensor]:
           penalty = self._inner.compute_orth_penalty()
           return penalty * self._penalty_weight
   
   def build(args): return OFTBundle(args)
   ```
3. 字典 + schema 各一行

train_loop 已经在 `if reg is not None: loss = loss + reg` 这里接住了。

### Case 5：混合需求——Ortho-Hydra（router lr + balance loss + per-step warmup）

**情况**：parking lot 上有完整实现。需求最杂：(1) router 参数要单独 lr scale（10x default）；(2) balance loss 要按 warmup ratio 加权；(3) 推理生态加载有兼容性问题（非标 LoRA key）——这个是 ADR 0001 不收入主仓的真正原因，跟 plugin 化无关。

**Plugin 化后能 cover 的部分**：
- router lr：当前 dev 走 `get_param_groups` 返回多组、每组带 `_is_router_group` flag、main() 里 pop 出来再乘 scale。重构后 adapter 在 `get_param_groups(wd)` 里直接返回多组 + 正确的 `lr` 字段，main()/optimizer phase 不再有 variant-specific 逻辑。
- balance loss：用 `regularization_loss(ctx)` 钩子，内部按 `ctx.global_step / total_steps` 算 warmup 系数。

**Plugin 化无法 cover 的部分**：
- 输出的 .safetensors 不是标准 LoRA key（含 `S_q / S_p / Q_basis / P_bases / lambda_layer / router`），a1111/ComfyUI 加载不了——这是结构性问题，不在重构范畴。如果将来生态出现 OrthoHydra loader，重新评估 [memory/pr49-handling] 的 GC 决定。

### Case 6：新优化器——Lion / CAME / Schedule-Free AdamW

**情况**：optimizer 类变体的接口比 adapter 简单（无 per-step hook），用纯 build 函数。

**步骤**（以 Lion 为例）：

1. `pip install lion-pytorch` —— `requirements.txt` 加一行
2. 新文件 `training/optimizers/lion.py`：
   ```python
   from lion_pytorch import Lion
   
   def build(args, params, lr, weight_decay):
       return Lion(
           params, lr=lr, weight_decay=weight_decay,
           betas=(args.lion_beta1, args.lion_beta2),
       )
   ```
3. `training/optimizers/__init__.py`：`BUILDERS["lion"] = lion.build`
4. `studio/schema.py`：`optimizer_type` Literal 加 `"lion"` + `lion_beta1` / `lion_beta2` 字段

main() / loop 0 改动。

**特殊情况**：如果该 optimizer 需要类似 PPSF 的"训练时切 averaged weights"行为（如 Schedule-Free AdamW），靠 `utils.optimizer_utils.optimizer_eval_mode` ctx manager 探测 `hasattr(optimizer, "train")` 自动 cover。

**校验逻辑**（如 PPSF 要求 `lr_scheduler=none`）：

```python
# training/optimizers/__init__.py
VALIDATORS: dict[str, Callable[[object], None]] = {}

# training/optimizers/prodigy_plus_schedulefree.py
def validate(args):
    if args.lr_scheduler != "none":
        raise SystemExit(
            f"ProdigyPlusScheduleFree requires lr_scheduler=none; "
            f"got lr_scheduler={args.lr_scheduler!r}"
        )

def build(args, params, lr, weight_decay):
    ...

# __init__.py
BUILDERS["prodigy_plus_schedulefree"] = ppsf.build
VALIDATORS["prodigy_plus_schedulefree"] = ppsf.validate
```

```python
# phases/optimizer.py 里：
def run(ctx):
    validator = VALIDATORS.get(ctx.args.optimizer_type)
    if validator:
        validator(ctx.args)
    ctx.optimizer = build_optimizer(
        ctx.args, ctx.adapter.get_param_groups(ctx.weight_decay),
        ctx.args.learning_rate, ctx.weight_decay,
    )
    return ctx
```

### Case 7：新 LR Scheduler——warmup_cosine

**情况**：现在只有 `cosine` / `cosine_with_restart`。要加带 warmup 的版本。

**步骤**：

1. 新文件 `training/schedulers/warmup_cosine.py`：
   ```python
   from torch.optim.lr_scheduler import LambdaLR
   import math
   
   def build(args, optimizer, total_steps):
       warmup = args.lr_scheduler_warmup_steps
       def lr_lambda(step):
           if step < warmup:
               return step / max(1, warmup)
           p = (step - warmup) / max(1, total_steps - warmup)
           return 0.5 * (1 + math.cos(math.pi * p))
       return LambdaLR(optimizer, lr_lambda=lr_lambda)
   ```
2. `training/schedulers/__init__.py`：`BUILDERS["warmup_cosine"] = warmup_cosine.build`
3. `studio/schema.py`：`lr_scheduler` Literal 加 `"warmup_cosine"` + `lr_scheduler_warmup_steps: int = Field(default=500, ...)`

### Case 8：新 Inference Sampler——Euler / DPM++2M

**情况**：当前 `sample_image` 内部硬编码走 `_sample_er_sde_const_x0`。

**步骤**：

1. 新文件 `training/inference_samplers/euler.py`：
   ```python
   import torch
   def sample(model, noise, cond, sigmas, *, device, dtype, cfg_scale, ...):
       """标准 Euler；输入 noise、目标 sigma 调度、cond，输出 latent。"""
       x = noise.clone()
       for i in range(len(sigmas) - 1):
           ...
       return x
   ```
2. `training/inference_samplers/__init__.py`：`BUILDERS["euler"] = euler.sample`
3. `training/sampling.py` 里 `sample_image` 改成：
   ```python
   def sample_image(model, vae, ..., sampler_name="er_sde", scheduler="simple", ...):
       sigmas = build_sigma_schedule(scheduler, steps)
       sampler = inference_samplers.BUILDERS[sampler_name]
       noise = torch.randn(shape, device=device, dtype=dtype)
       latent = sampler(model, noise, cond, sigmas, ...)
       return vae.decode(latent)
   ```
4. `studio/schema.py`：`sample_sampler_name` Literal 加 `"euler"`

Anima_train / loop / phases 0 改动。

### Case 9：删除一个变体

最简单的例子：哪天发现 `cosine_with_restart` 没人用，要清理。

**步骤**（3 行变化）：

1. `git rm training/schedulers/cosine_with_restart.py`
2. `training/schedulers/__init__.py` 删一行：`"cosine_with_restart": cosine_with_restart.build`
3. `studio/schema.py`：`lr_scheduler: Literal[...]` 去掉 `"cosine_with_restart"`

没人引用，没有 dispatch 逻辑要删。这就是"好去除"。

## 不变的部分（明确划界）

- `studio/schema.py` 保持单一 `TrainingConfig` 大类，所有字段平铺。**Plugin 不分裂 schema**。
- `studio/argparse_bridge.py` 0 改动（schema 字段还是从 Pydantic 派生 CLI 参数）
- 前端 UI / `config.yaml` / `studio.sh` / `studio/cli.py` / `studio/supervisor.py` 0 改动
- `runtime/anima_daemon.py` / `anima_generate.py` / `anima_reg_ai.py` / `train_monitor.py` 0 改动
- `utils/` 下的 adapter / optimizer 实现（`lycoris_adapter.py` / `optimizer_utils.py` / `caption_utils.py` 等）0 改动；plugin 子包只是"调度层"，复用现有实现
- 训练行为 0 变化：同一 `config.yaml` 跑出的 loss 曲线、最终 LoRA 权重应该 bit-for-bit 相同

## 风险与开放问题

### R1 — 三个 PR 期间 experimental/pr49-adapters 要 rebase 3 次

每次 dev 重构 PR 合入后，parking lot 分支都要 `git rebase dev`。变体 adapter 文件位置变（从 `utils/tlora_adapter.py` 不动，但 train_loop dispatch 从 main() 移走），rebase 时主要冲突在 main()/anima_train.py。

**缓解**：PR-C 合完之后，把 parking lot 上的 `tlora_adapter.py` / `orthohydra_adapter.py` 改写成符合新 Protocol 的 bundle 文件，存到 `training/adapters/` 里。之后 parking lot 不再有 anima_train 内联代码，rebase 痛苦消失。

[memory/pr49-handling] 已经有「重构后 experimental 要 rebase」提示，[memory/anima_train_refactor_pending] 状态也已更新。

### R2 — 三个 PR 期间 dev 不能接其他训练侧改动

PR-A 搬运 ~5000 行变化，跟任何 anima_train 改动都会冲突。建议 freeze 窗口（≈1 周完成 A，再花 ~3 天 B，再 ~3 天 C）。期间训练相关的 hotfix 只能等。

### R3 — sister-script re-export 漏掉的名字

`anima_train` 顶层有 53 个 def/class。我们只确认了 sister script + 测试用 9 个。其他名字（如 `_strip_prefixes` / `forward_with_optional_checkpoint` 等内部 util）理论不该被外部依赖，但万一有 hidden caller，PR-A 合入会立刻 ImportError。

**缓解**：PR-A 上线前跑 `grep -r "anima_train\." --include="*.py"` 在 `tests/` / `studio/` / `runtime/` 三个目录全搜一遍。

### R4 — Schema 与 registry 同步

加变体要碰 2 处（plugin 字典 + schema Literal）。容易 schema 加了枚举值但 plugin 没注册，或反之。

**缓解**：在 `training/adapters/__init__.py` 加 `_validate_schema_consistency()` 函数，启动期跑一次检查：

```python
def _validate_schema_consistency():
    from studio.schema import TrainingConfig
    schema_options = set(TrainingConfig.model_fields["lora_type"].annotation.__args__)
    registered = set(BUILDERS)
    if schema_options != registered:
        raise RuntimeError(
            f"adapter 注册与 schema 不同步：\n"
            f"  schema 有但未注册: {schema_options - registered}\n"
            f"  注册但 schema 没列: {registered - schema_options}"
        )
```

对 optimizer / scheduler / inference_sampler 同样做。

### R5 — `cosmos_predict2_modeling.py` 等其他训练相关模块改动

PR-A 触不到这些；如果 PR-B 或 PR-C 需要调整模型加载流程（如改 `load_anima_model` 签名），可能波及 `models/` 下的 .py。需要在 PR-B 设计时确认。

### R6 — 行号引用的 ADR 0002

PR-A 合入后 [docs/adr/0002-webui-self-update.md L17, L388] 的 `runtime/anima_train.py#L2374-L2401` 失效。

**缓解**：PR-A 同 PR 内更新 ADR 0002 的链接，从行号改为指向 `runtime/training/phases/resume.py:signal_handler` 或类似关键词锚点。

## 验收标准

PR-A：
- `tests/test_anima_train_migration.py` 全绿
- `tests/test_lycoris_resume.py` 全绿
- `tests/test_anima_generate_xy.py` 全绿（验 sister script API 未破）
- `python runtime/anima_train.py --help` 输出与重构前 diff 为空
- 端到端用 1 个小数据集训 100 step，最终 `safetensors` 与重构前 bit-for-bit 相同（或 loss 曲线在 fp 误差内一致）

PR-B：同上 + `python runtime/anima_train.py --config sample.yaml --max-steps 100` 跑通

PR-C：同上 + 新增 `tests/test_plugin_registry.py`：
- 注册 / 查找 / 删除 mock plugin 正确
- schema-registry 一致性检查触发
- `AdapterProtocol` runtime_checkable 对 `AnimaLycorisAdapter` 返回 True
- T-LoRA / OFT 等 mock adapter（仅 test 内定义）能挂上 hook 并被 train_loop 调用一次

## 相关

- [docs/adr/0001-lokr-via-lycoris-lora.md](0001-lokr-via-lycoris-lora.md) — 解释为什么 adapter 实现层是 LyCORIS（PR-C 不动它）
- [docs/adr/0002-webui-self-update.md](0002-webui-self-update.md) — Ctrl+C handler 行号引用要更新
- [memory/pr49-handling](../../C:/Users/Mei/.claude/projects/G--AnimaLoraStudio/memory/pr49-handling.md) — parking lot 政策；PR-C 后路径改善
- [memory/anima_train_refactor_pending](../../C:/Users/Mei/.claude/projects/G--AnimaLoraStudio/memory/anima_train_refactor_pending.md) — 本 ADR 是它的执行计划
- [memory/feedback_authoring_time_normalize](../../C:/Users/Mei/.claude/projects/G--AnimaLoraStudio/memory/feedback_authoring_time_normalize.md) — schema 单一大类（不分裂）符合"作者写时规范化"原则

## 已决议事项（2026-05-14 起手前 align）

- **schema↔registry 一致性自动校验**：PR-C **要加** `_validate_schema_consistency()`，启动期跑一次，针对 4 个 plugin 子包（adapter / optimizer / scheduler / inference_sampler）各做一遍。理由：dev 漏改 schema 或 registry 是隐 bug，跑训练才发现代价高于 5 行启动期检查。
- **inference_samplers/ 这次做**：虽然当前只有 er_sde 一个，仍建 subfolder + `BUILDERS` dict + `er_sde.py`，预留位。未来加 euler/dpmpp 直接塞文件不改 sample_image。
- **共享数学 util 就近放**：`noise.py` / `timestep_sampling.py` / `loss_weighting.py` 内部需要的小工具就放在各自文件，不建 `training/math_utils.py`。三者目前没共享代码，YAGNI。

## 起手前 freeze 状态（2026-05-14）

- dev 上无任何 in-flight 训练侧改动（用户已叫停）；PR-A 可以立刻起手
- 工作区 `_pr49_train.py` / `_tmp_pr49.diff` / `pr49.diff` / `pr18_review.md` 已清理（上轮 cherry-pick 残留）

## 验收策略（确认）

- **每个 PR 合 dev 的门槛**：单测全绿 + `python runtime/anima_train.py --help` diff 为空
- **PR-C 完成后**：用户本地跑完整 LoRA 训练 + 评估效果（不光看 loss，看出图质量是否漂移）

---

## 后续扩展记录

ADR 0003 三 PR（PR #56/#57/#58）合并后，原计划之外的 plugin 子包后续按需新增。本节记录
这些扩展，以验证 ADR 的"3-4 步本地加变体"承诺在真实新增场景下站得住脚。

### 扩展 1：`timestep_samplers/` — PR #63 / PR #66（2026-05-15）

**起因**：PR #63 引入 InfoNoise（I-MMSE 自适应 timestep 采样器），原始实现 hard-wire
`ctx.info_noise` 字段在 `TrainingContext` 上，loop.py 用三处 `if ctx.info_noise is not None`
守卫。这违反 ADR 0003 的 plugin 原则：

- "下一个想加 Min-SNR-aware sampler 的人会复制 hard-wire 模式" — 这正是 ADR 0003 "Case 1
  DoRA" 章节警告过的反模式
- 第二种 adaptive sampler 要么再加 `ctx.min_snr_sampler`，要么改成 dispatch chain，**两种都
  烂**

**PR #66 落地的结构**：

```
training/timestep_samplers/
├── protocol.py        ← TimestepSamplerProtocol（1 必需 + 3 可选 hook，runtime_checkable）
├── baseline.py        ← BaselineTimestepSampler 包装现有 sample_t 4 mode
├── infonoise.py       ← InfoNoiseScheduler + build
└── __init__.py        ← BUILDERS + build_timestep_sampler
```

接口设计（参考 `AdapterProtocol`）：

```python
class TimestepSamplerProtocol(Protocol):
    def sample(self, bs: int, device) -> Tensor: ...               # 必需

    # 可选 hook，非自适应采样器（baseline）默认 no-op
    def record(self, t: Tensor, raw_mse: Tensor) -> None: ...      # 给 adaptive 收集统计
    def maybe_refresh(self, global_step: int) -> None: ...         # 给 adaptive 周期性更新分布
    def status(self) -> dict: ...                                  # 给 wandb 监控
```

**调用方简化**（loop.py 三处 `if` 守卫消失）：

```python
t = ctx.timestep_sampler.sample(bs, ctx.device)           # 统一接口，baseline 也走这个
...
ctx.timestep_sampler.record(t.detach(), _raw_mse)         # baseline 是 no-op
...
ctx.timestep_sampler.maybe_refresh(ctx.global_step)       # baseline 是 no-op
```

**与原 ADR 的偏差**：这是第 5 个 plugin 子包，原 ADR 目标布局只列了 4 个（adapters /
optimizers / schedulers / inference_samplers）。属于"按需追加新维度"的正常扩展，不算
违反 ADR。

**派发策略差异**：`build_timestep_sampler` 用 **bool 开关派发**（`args.infonoise_enabled`），
而非其他 4 个 plugin 用的 `Literal` 派发。这是有意的：

| 派发方式 | 何时用 | 例子 |
|---|---|---|
| `Literal[...]` + 完全互斥 | 用户从一组中选一个（mutex），且每种语义清晰 | `lora_type` / `optimizer_type` / `lr_scheduler` |
| bool 开关 + 优先级链 | 启用条件正交，可能将来想"baseline 兜底 + 自适应叠加" | `infonoise_enabled` |

第 2 个 adaptive sampler（如 Min-SNR-aware）加入时再 review：如果跟 InfoNoise 互斥 → 切
`Literal`；如果可叠加（比如 Min-SNR-aware 作为 baseline，InfoNoise 在其上做 importance
sampling）→ 保留 bool 开关。

**`validate_schema_consistency()` 暂未加**：bool 派发不依赖 Literal 集合，schema 字段单点
真相，不需要双向校验。一旦切 Literal 派发就加。

**何时切 Literal 派发**：原则上 ≥3 种 adaptive sampler 时再切。当前 baseline + infonoise
两种，提前抽象违反 YAGNI。

### 评估：新结构对未来 plugin 维护的支持

对照 ADR 0003 "3-4 步本地加变体" 承诺，PR #66 引入 `timestep_samplers/` 后**新增同类**
（如 Min-SNR-aware sampler）：

| 步骤 | 改动 | 文件 |
|---|---|---|
| 1. 实现 | 新 `MinSnrAwareSampler` 类实现 protocol + `build()` 工厂 | `timestep_samplers/min_snr_aware.py`（新文件）|
| 2. 注册 | `BUILDERS["min_snr_aware"] = min_snr_aware.build` | `timestep_samplers/__init__.py`（1 行）|
| 3. 派发 | `if args.min_snr_aware_enabled: return BUILDERS["min_snr_aware"](args, total_steps)` | `timestep_samplers/__init__.py`（2 行）|
| 4. schema | `min_snr_aware_enabled: bool` + 该 sampler 专属字段 | `studio/schema.py` |

`loop.py` / `phases/optimizer.py` / `context.py` / `loop.py` 调用代码 **零改动**。

承诺达成。

### 待办

- 新增 plugin 子包加入 `phases/bootstrap.run()` 的 `validate_schema_consistency()` 调用
  时机：仅在该子包切到 Literal 派发后（当前 `timestep_samplers/` 不调用）
- 未来若再加 plugin 子包（loss_weighters / noise_schedulers 等），按本节模式记录
- 单 PR 不强制做"真模型 100 step loss bit-for-bit 一致"——代价过高且 fp 非确定性可能假阳。靠端到端的最终训练做 ground truth 验证
