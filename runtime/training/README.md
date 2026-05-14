# `runtime/training/` — 训练流水线包

`anima_train.py` 调起的训练全流程实现。ADR 0003 把原 2901 行单文件拆成本子包，**main()** 现在只是 11 行编排：

```python
def main():
    args = parse_args()
    ctx = TrainingContext(args=args)
    phases.bootstrap.run(ctx)
    phases.models.run(ctx)
    phases.dataset.run(ctx)
    phases.optimizer.run(ctx)
    phases.resume.run(ctx)
    loop.run(ctx)
    phases.finalize.run(ctx)
```

详细设计见 [`docs/adr/0003-anima-train-refactor.md`](../../docs/adr/0003-anima-train-refactor.md)。

## 目录结构

```
runtime/training/
├── context.py              ← TrainingContext dataclass（43 字段 + emit / get_next_sample_prompt / handle_interrupt 方法）
├── loop.py                 ← 主训练循环：for epoch / for batch / 累积 / forward / loss / 周期 IO
├── sample_runner.py        ← run_sample(ctx, prompt, path, ...) helper，消掉原 3 处重复的 sample 块
│
├── bootstrap.py            ← deps 检测 / yaml 加载 / 进度条 init（被 phases.bootstrap 调）
├── cli.py                  ← parse_args / interactive helpers
├── observability.py        ← WandBMonitor + loss 曲线 ASCII / Rich 渲染
├── model_loading.py        ← prefix 推断 / safetensors / 路径解析 / xformers / forward checkpoint
├── models.py               ← load_anima_model / load_vae / load_text_encoders（sister script 也用）
├── text_encoding.py        ← Qwen / T5 加权 tokenize
├── state.py                ← save / load_training_state
├── dataset.py              ← BucketManager + ImageDataset + 5 衍生类 + collate
├── sampling.py             ← 推理用 sample_image + sigma 调度（被 sister script 也用）
├── timestep_sampling.py    ← 训练 step 用 sample_t（logit_normal / uniform / mode）
├── noise.py                ← make_noise（offset + pyramid）
├── loss_weighting.py       ← compute_loss_weight（min_snr / cosmap / detail_inv_t）
│
├── phases/                 ← main() 的 6 个 phase；每个 run(ctx) in-place mutate
│   ├── bootstrap.py        ← yaml + 交互 + seed + device + wandb + monitor_state writer
│   ├── models.py           ← path resolve + 加载 transformer/vae/text encoders + LoRA inject
│   ├── dataset.py          ← build 主集 + 正则集 + dataloader + VAE roundtrip 自检
│   ├── optimizer.py        ← build_optimizer + validate + scheduler + total_steps
│   ├── resume.py           ← init_progress + state recovery + SIGINT + sample prompts + baseline
│   └── finalize.py         ← 最终 LoRA save + 清理 progress + 最终 loss curve + wandb finish
│
└── ── 4 个 plugin 子包 ──（加变体本地化的关键）
    ├── adapters/           ← LoRA 变体
    │   ├── protocol.py     ← AdapterProtocol + StepContext
    │   ├── lycoris.py      ← build_adapter for lokr/loha/lora
    │   └── __init__.py     ← BUILDERS dict + build_adapter + validate_schema_consistency
    │
    ├── optimizers/         ← AdamW / Prodigy / PPSF
    │   ├── adamw.py / prodigy.py / prodigy_plus_schedulefree.py
    │   └── __init__.py     ← BUILDERS + VALIDATORS + build_optimizer + validate_optimizer
    │
    ├── schedulers/         ← cosine / cosine_with_restart（"none" 是 schema-only 不开文件）
    │   ├── cosine.py / cosine_with_restart.py
    │   └── __init__.py     ← BUILDERS + build_scheduler
    │
    └── inference_samplers/ ← er_sde（未来加 euler / dpmpp 直接塞文件）
        ├── er_sde.py
        └── __init__.py     ← BUILDERS + build_inference_sampler
```

## 数据流

```
parse_args()                          ┐
        ↓                              │
TrainingContext(args=args)             │
        ↓                              │
phases.bootstrap.run(ctx)              │  填 device / dtype / output_dir / wandb / monitor
        ↓                              │
phases.models.run(ctx)                 │  填 repo_root / model / vae / qwen* / t5_tok / injector
        ↓                              ├─ 一次性 setup
phases.dataset.run(ctx)                │  填 bucket_mgr / dataset / reg_dataset / dataloader
        ↓                              │
phases.optimizer.run(ctx)              │  填 optimizer / scheduler / total_steps / trainable_params
        ↓                              │
phases.resume.run(ctx)                 │  填 progress / live / global_step / sample_prompts；
        ↓                              ┘  跑 baseline 采样；注册 SIGINT
loop.run(ctx)                          ──  for epoch / for batch（read+write 几乎所有 ctx.*）
        ↓
phases.finalize.run(ctx)               ──  final save + cleanup
```

**ctx 是单一可变状态包**，phase 函数签名都是 `run(ctx: TrainingContext) -> None`，in-place 改 ctx 上的字段。不返回值，不要做 `ctx = phase.run(ctx)` 模式。

## 加变体：3-4 步本地操作

### 加一个新 LoRA 变体（如 T-LoRA / OFT / VeRA）

1. **算法实现**：写 `utils/{variant}_adapter.py` 实现底层算法类
   *（注：算法层放 utils/ 还是 training/adapters/_impl/ 待 2026-05-15 决定，
   见 memory `utils_algo_placement_pending`）*
2. **registry 壳**：写 `training/adapters/{variant}.py` 含 `build(args) -> AdapterProtocol`
3. **注册**：`training/adapters/__init__.py` 的 `BUILDERS` dict 加一行
4. **schema**：`studio/schema.py` 的 `lora_type: Literal[...]` 多加一个值 + 加该变体专属字段（用 `_meta(group, show_when=f"lora_type=='{variant}'")`）

`main()` / `phases/models.py` / `loop.py` **零改动**。

如果新变体需要 per-step 调整内部结构（T-LoRA 按 sigma_t 调 mask），实现 `on_step_begin(ctx)` hook；如果需要加正则项到 loss（OFT 的 orthogonality penalty），实现 `regularization_loss(ctx) -> Tensor` hook。LyCORIS 走默认 no-op。

### 加一个新 optimizer（如 Lion / CAME）

1. **build wrapper**：写 `training/optimizers/{name}.py` 含 `build(args, params, lr, weight_decay) -> Optimizer`
2. **可选**：如果有启动期约束（PPSF 要 `lr_scheduler=none`），加 `validate(args)`
3. **注册**：`training/optimizers/__init__.py` 的 `BUILDERS` 字典加一行（有 validate 则同时加 `VALIDATORS`）
4. **schema**：`optimizer_type: Literal[...]` 加值 + 该变体专属字段
5. **依赖**：`requirements.txt` 加包（如有）

### 加一个新 lr scheduler（如 warmup_cosine / one_cycle）

1. `training/schedulers/{name}.py` 含 `build(args, optimizer, total_steps) -> LRScheduler`
2. `training/schedulers/__init__.py` `BUILDERS` 字典加一行
3. schema 的 `lr_scheduler: Literal[...]` 加值 + 该变体专属字段

### 加一个新 inference sampler（如 euler / dpmpp2m）

1. `training/inference_samplers/{name}.py` 含 `sample(denoise_fn, x, sigmas, **kw) -> Tensor`
2. `__init__.py` `BUILDERS` 字典加一行
3. 用户在 schema/yaml 写 `sample_sampler_name: {name}` 即生效（注：现 schema 是 `str` 不是 `Literal`，未注册的 name 会回退 sample_image 内 inline Euler）

### 删一个变体

逆操作：删文件 + 字典一行 + schema Literal 一项。`validate_schema_consistency()` 会在启动期保证不漏。

## AdapterProtocol hook：何时用哪个

```python
class AdapterProtocol(Protocol):
    # 必需 4 个
    def inject(self, model) -> None
    def get_param_groups(self, weight_decay) -> list[dict]
    def save(self, path)
    def load(self, path)

    # 可选 3 个 hook（默认 no-op）
    def on_step_begin(self, ctx: StepContext) -> None
    def regularization_loss(self, ctx) -> Optional[Tensor]
    def excludes_weight_decay(self, name) -> bool
```

| 变体类型 | 用哪个 hook | 示例 |
|---|---|---|
| 纯权重（结构 setup 后不变） | 都不用 | DoRA / rsLoRA / PiSSA / VeRA / LoRA-FA |
| LoRA+ 不同子模块不同 lr | `get_param_groups` 多返回组 | LoRA+ B 矩阵 16× lr |
| 按 sigma_t / step 调内部结构 | `on_step_begin(ctx)` | T-LoRA / AdaLoRA / B-LoRA |
| 训练 loss 加正则项 | `regularization_loss(ctx)` | OFT / Ortho-Hydra balance loss |
| weight_decay 按 param 名排除 | `excludes_weight_decay(name)` | LoKr 的 w1 |

`StepContext` 是 5 字段冻结 dataclass：`global_step / total_steps / epoch / sigma_t / args`。

## 跟 `utils/` 的关系

依赖方向 **单向**：`training/` → `utils/`，反过来从不发生。

```
training/adapters/lycoris.py            ← 5 行 build 壳子
        ↓ import
utils/lycoris_adapter.py                ← 算法实现层
        ↓ import
utils/lycoris_patch.py                  ← lycoris-lora 上游 bug 补丁
utils/lokr_preset.py                    ← DiT 层选择规则
```

- `training/` 知道「args / TrainingContext / phase / registry」
- `utils/` 知道「算法 / 库 API / 框架补丁」，**不知道**训练流水线存在
- 推理路径（`studio/services/inference_core`）也能复用 `utils/lycoris_adapter`，正因为它不绑训练上下文

未决项：**未来新 LoRA 变体的算法实现**继续放 utils/ 还是搬 `training/adapters/_impl/`，2026-05-15 决定。见 `memory/utils_algo_placement_pending`。

## Schema↔registry 一致性

`phases/bootstrap.run()` 在最早期就调 3 个 `validate_schema_consistency()`：

```python
from training.adapters import validate_schema_consistency as _va
from training.optimizers import validate_schema_consistency as _vo
from training.schedulers import validate_schema_consistency as _vs
_va(); _vo(); _vs()
```

逻辑：取 `TrainingConfig.{lora_type, optimizer_type, lr_scheduler}` 的 `Literal[...]` 集合，跟对应 `BUILDERS` keys 集合对比。失配 raise，启动期早 fail，避免训练跑半天才发现配错。

`schedulers/` 特殊：`"none"` 是 schema-only 不在 BUILDERS（`build_scheduler` 显式返回 None）；`SCHEMA_ONLY_OPTIONS = {"none"}` 跳过校验。

`sample_sampler_name` 是 `str` 不是 `Literal`，所以 `inference_samplers/` 没有 schema 校验，未注册名字走 sample_image 内 inline Euler 兜底。

## 测试

```bash
# 跟 training/ 直接相关的单测
pytest tests/test_anima_train_migration.py        # CLI / YAML / parse_args 契约
pytest tests/test_anima_generate_xy.py            # sister script `_T.X` 访问模式
pytest tests/test_plugin_registry.py              # registry 三件套 + Protocol hook
```

`test_plugin_registry.py` 防回归断言：`phases/optimizer.py` 不该再含 `if optimizer_type == "prodigy"` 字面量、`phases/models.py` 不该再 `AnimaLycorisAdapter(`、`sampling.py` 不该 `if sampler_name == "er_sde"`。

端到端验证靠用户**跑完整 LoRA 训练 + 评估出图**（ADR 0003 验收策略 R2）；单 PR 不强制 bit-for-bit。

## Sister script 契约

`runtime/anima_daemon.py` / `anima_generate.py` / `anima_reg_ai.py` 用 `import anima_train as _T` 然后 `_T.find_diffusion_pipe_root` / `_T.load_anima_model` / `_T.load_vae` / `_T.load_text_encoders` / `_T.sample_image` / `_T.enable_xformers` / `_T.resolve_path_best_effort`。

这 7 个名字 + 测试用的 `parse_args` / `apply_yaml_config` / `save_training_state` / `load_training_state` 都在 `runtime/anima_train.py` 顶层 re-export。修改 `training/` 内部时不要破坏这层契约——`tests/test_anima_generate_xy.py` 会捕获。

## 历史 + 延伸

- [ADR 0003](../../docs/adr/0003-anima-train-refactor.md) — 完整设计文档 + 9 个变体落地案例
- [ADR 0001](../../docs/adr/0001-lokr-via-lycoris-lora.md) — 为什么 adapter 走 lycoris-lora pip 包
- [ADR 0002](../../docs/adr/0002-webui-self-update.md) — Ctrl+C handler 现位置 `phases/resume.py:run()` 内的 `ctx.handle_interrupt`
- [`studio/schema.py`](../../studio/schema.py) — `TrainingConfig` 的 Literal 枚举 + 字段 `_meta(group, show_when, ...)` 给前端 UI

PR #56 / #57 / #58 是 ADR 0003 的三刀执行记录，commit history 干净，回滚精确。
