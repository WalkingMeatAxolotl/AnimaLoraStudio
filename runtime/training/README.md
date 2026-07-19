# `runtime/training/` — 训练流水线包

`anima_train.py` 调起的训练全流程实现。ADR 0003 把原 2901 行单文件拆成本子包，**main()** 现在只保留 phase 编排：

```python
def main():
    args = parse_args()
    ctx = TrainingContext(args=args)
    phases.bootstrap.run(ctx)
    phases.models.run(ctx)
    phases.dataset.run(ctx)
    phases.text_cache.run(ctx)
    phases.models.finish(ctx)
    phases.optimizer.run(ctx)
    phases.resume.run(ctx)
    loop.run(ctx)
    phases.finalize.run(ctx)
```

详细设计见 [`docs/adr/0003-anima-train-refactor.md`](../../docs/adr/0003-anima-train-refactor.md)。

## 目录结构

```
runtime/training/
├── context.py              ← TrainingContext dataclass（emit / get_next_sample_prompt / handle_interrupt 方法；
│                              文本栈经 ctx.text_stack 对族外 opaque）
├── loop.py                 ← 主训练循环：for epoch / for batch / 累积 / forward / loss / 周期 IO（族无关，
│                              前向经 ctx.family 派发，零 `if family == ...` 分支）
├── sample_runner.py        ← run_sample(ctx, prompt, path, ...) helper；采样经 ctx.family.sample_image 派发
│
├── bootstrap.py            ← deps 检测 / yaml 加载 / 进度条 init（被 phases.bootstrap 调）
├── cli.py                  ← parse_args / interactive helpers
├── observability.py        ← WandBMonitor + loss 曲线 ASCII / Rich 渲染
├── model_loading.py        ← prefix 推断 / safetensors / 路径解析 / xformers
├── vae.py                  ← VAEWrapper + load_vae（跨族共享；曾名 models.py）
├── sysmem.py               ← 显存 / 内存编排：working set trim + RAM/GPU 加载护栏（NVML 全卡视角）
├── state.py                ← save / load_training_state
├── snapshot.py             ← pause / resume 用的 state snapshot helpers（ADR 0006）
├── dataset.py              ← BucketManager + ImageDataset + 衍生类 + collate + navit sampler/collate
├── text_cache.py           ← varlen 文本 sidecar / task 档案 .text-cache prompt bundle 协议与原子 safetensors I/O
├── timestep_sampling.py    ← 训练 step 用 sample_t（logit_normal / uniform / mode）；
│                            被 timestep_samplers/baseline.py 复用
├── noise.py                ← make_noise（offset + pyramid）
├── loss_weighting.py       ← compute_loss_weight（min_snr / cosmap / detail_inv_t）；
│                            注意：是 *loss 权重*（Flow Matching 步级缩放系数），
│                            跟 *loss 类型*（mse / huber，见 losses/）正交
│
├── families/               ← 模型族 registry（架构级；加第 3 个族 = 纯新增文件 + 各 registry 一行注册）
│   ├── protocol.py         ← ModelFamily 九方法契约 + get_family / resolve_family（未知族 fail-fast）
│   ├── spec.py             ← ModelSpec 冻结面：LatentSpec / TextSpec / SamplingDefaults / 能力集 / 拒绝位
│   ├── latent_spaces.py    ← WAN21_F8C16 等跨族共享 latent 空间事实（单一实例）
│   ├── anima/              ← Anima 族：family / loader / forward / preset / sampling /
│   │                          text_encoding（Qwen 0.6B + T5）/ comfy_qwen / navit / leap / sra_align
│   ├── krea2/              ← Krea 2 族：loader（bf16/fp8 严格加载）/ preset（krea2_full 全 Linear）/
│   │                          sampling（FlowMatchEuler + simple 调度）/ text_encoding（Qwen3-VL 12 层 varlen）/
│   │                          quant_fp8（fp8 dequant 前向）/ lora_fp8_merge（LoRA merge 回写）
│   └── README.md           ← 三居所导览 + 加族步骤
│
├── phases/                 ← main() 的 7 个 phase；每个 run(ctx) in-place mutate
│   ├── bootstrap.py        ← yaml + 交互 + seed + device + wandb + monitor_state writer +
│   │                          调各 plugin 子包的 validate_schema_consistency
│   ├── models.py           ← path resolve + 按族加载（cached_varlen 族两段式：run 装 VAE/TE，
│   │                          finish 在 TE 释放后补装 DiT + LoRA inject）+ fp8_base 防呆 + RAM 护栏
│   ├── dataset.py          ← build 主集 + 正则集 + dataloader + VAE roundtrip 自检
│   ├── text_cache.py       ← cached_varlen 族预扫最终 caption + 调 family 编码缓存
│   ├── optimizer.py        ← build_optimizer + validate + scheduler + total_steps +
│   │                          build_timestep_sampler + build_loss
│   ├── resume.py           ← init_progress + state recovery + SIGINT + sample prompts + baseline
│   └── finalize.py         ← 最终 LoRA save + 清理 progress + 最终 loss curve + wandb finish
│
└── ── 6 个 plugin 子包 ──（加变体本地化的关键）
    ├── adapters/           ← LoRA 变体
    │   ├── protocol.py     ← AdapterProtocol + StepContext
    │   ├── lycoris.py      ← build_adapter for lokr/loha/lora（族 preset 显式注入）
    │   ├── ortho.py / tlora.py ← OrthoLoRA / T-LoRA builder
    │   └── __init__.py     ← BUILDERS dict + build_adapter + validate_schema_consistency
    │
    ├── optimizers/         ← adamw / automagic / came / lion / prodigy / prodigy_plus_schedulefree / soap / soap_sf
    │   └── __init__.py     ← BUILDERS + VALIDATORS + build_optimizer + validate_optimizer
    │
    ├── schedulers/         ← cosine / cosine_with_restart / cosine_with_warmup（"none" 是 schema-only 不开文件）
    │   └── __init__.py     ← BUILDERS + build_scheduler
    │
    ├── inference_samplers/ ← er_sde / dpmpp_3m_sde（Anima 用；Krea 2 的 Euler 归族内 sampling.py）
    │   └── __init__.py     ← BUILDERS + build_inference_sampler
    │
    ├── timestep_samplers/  ← 训练 timestep 采样器（PR #66 引入）
    │   ├── protocol.py     ← TimestepSamplerProtocol（可选 token_counts batch context）
    │   ├── baseline.py     ← sample_t 4 mode 的 thin wrapper（非自适应）
    │   ├── infonoise.py    ← InfoNoise I-MMSE 自适应采样器（arxiv 2602.18647）
    │   ├── krea2_shift.py  ← Krea2 动态 resolution shift（按每图 token 数修正）
    │   └── __init__.py     ← BUILDERS + build_timestep_sampler
    │
    └── losses/             ← 训练 loss 类型（mse / huber / ...）
        ├── protocol.py     ← LossProtocol（compute(pred, target, t) → Tensor）
        ├── mse.py          ← F.mse_loss 包装（默认）
        ├── huber.py        ← Huber loss with constant/snr/sigma delta schedule
        └── __init__.py     ← BUILDERS + build_loss + validate_schema_consistency
```

Anima 专属的训练步（navit / leap / sra_align）、文本编码（text_encoding / comfy_qwen）与推理采样
（sampling）已随多模型改造迁入 `families/anima/`；顶层不再有这些文件。

## 数据流

```
parse_args()                          ┐
        ↓                              │
TrainingContext(args=args)             │
        ↓                              │
phases.bootstrap.run(ctx)              │  填 device / dtype / output_dir / wandb / monitor
        ↓                              │
phases.models.run(ctx)                 │  常规族填完整模型栈；cached_varlen 族先填 VAE / text_stack
        ↓                              ├─ 一次性 setup
phases.dataset.run(ctx)                │  填 bucket_mgr / dataset / reg_dataset / dataloader
        ↓                              │
phases.text_cache.run(ctx)             │  cached_varlen 族写随图 sidecar；online 族 no-op
        ↓                              │
phases.models.finish(ctx)              │  TE 释放后补载 deferred DiT + injector（常规族 no-op）
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
3. schema 的 `sample_sampler_name: Literal[...]` 加值，并把它加进 `studio/domain/common.py`
   的 `FAMILY_SAMPLING` 对应族白名单（选项按族过滤 + 越族值校验都从这张表推导）

### 加一个新 timestep 采样器（如 Min-SNR-aware / P-Loss-aware）

跟其他 plugin 模式略有差异：当前 registry 用 **bool 开关派发**而非 `Literal` 枚举派发，因为
每个自适应 sampler 可能有不同的 args / 启用条件。

1. **实现**：写 `training/timestep_samplers/{name}.py` 含：
   - `class {Name}Sampler` 实现 `TimestepSamplerProtocol`（`sample` 必需；`record` /
     `maybe_refresh` / `status` 按需 override）
   - `build(args, total_steps) -> {Name}Sampler` 工厂
2. **注册**：`training/timestep_samplers/__init__.py` 的 `BUILDERS` 加一行
3. **派发**：同文件 `build_timestep_sampler` 加 if 分支（按优先级 `args.{name}_enabled == True`）
4. **schema**：`studio/schema.py` 加 `{name}_enabled: bool` + 该采样器专属字段

普通采样器对 `loop.py` / `phases/optimizer.py` / `context.py` **零改动**。如果算法需要
逐样本 latent token 数，令 sampler 声明 `requires_token_counts = True`；共享循环会通过
`sample(..., token_counts=...)` 注入通用 batch context，不得按 family 名称分支。

如果将来有 ≥3 个 adaptive sampler，可考虑重构成 `timestep_sampler_kind: Literal["baseline",
"infonoise", "min_snr_aware", ...]` 的 Literal 派发 + `validate_schema_consistency()`，跟
adapters / optimizers 一致；目前 2 个（baseline + infonoise）不值得这层抽象。

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
training/adapters/lycoris.py            ← build 壳子（族 preset 由 phases.models 经 ctx.family.lora_preset() 显式注入）
        ↓ import
utils/lycoris_adapter.py                ← 算法实现层
        ↓ import
utils/lycoris_patch.py                  ← lycoris-lora 上游 bug 补丁
```

- `training/` 知道「args / TrainingContext / phase / registry」
- `utils/` 知道「算法 / 库 API / 框架补丁」，**不知道**训练流水线存在
- 推理路径（`studio/services/inference_core`）也能复用 `utils/lycoris_adapter`，正因为它不绑训练上下文
- DiT 层选择规则（LoRA preset）是**族知识**，在 `families/{anima,krea2}/preset.py`，不在 utils/

## Schema↔registry 一致性

`phases/bootstrap.run()` 在最早期就调 4 个 `validate_schema_consistency()`（模型族另有一层：
`families/` 注册期自洽校验 + `resolve_family` 对未知族 fail-fast；schema 的 `model_family`
Literal 与 studio 侧 `FAMILY_ASSETS` / 能力矩阵的一致性由 `tests/test_model_family_gating.py`
同一性断言锁死）：

```python
from training.adapters import validate_schema_consistency as _va
from training.optimizers import validate_schema_consistency as _vo
from training.schedulers import validate_schema_consistency as _vs
from training.losses import validate_schema_consistency as _vl
_va(); _vo(); _vs(); _vl()
```

逻辑：取 `TrainingConfig.{lora_type, optimizer_type, lr_scheduler, loss_type}` 的 `Literal[...]` 集合，跟对应 `BUILDERS` keys 集合对比。失配 raise，启动期早 fail，避免训练跑半天才发现配错。

`schedulers/` 特殊：`"none"` 是 schema-only 不在 BUILDERS（`build_scheduler` 显式返回 None）；`SCHEMA_ONLY_OPTIONS = {"none"}` 跳过校验。

`sample_sampler_name` / `sample_scheduler` 是按族约束的 `Literal`（白名单单源在
`studio/domain/common.py:FAMILY_SAMPLING`）：anima 的 Literal-外历史值静默归并族默认（#256
迁移契约），krea2 的越族值直接报错。Krea 2 的 Euler 不走 `inference_samplers/` registry，
归族内 `families/krea2/sampling.py`。

`timestep_samplers/` 用 bool 开关派发（`infonoise_enabled`）而非 `Literal`，所以也没有
schema↔registry 一致性校验。当 adaptive sampler 数量 ≥3 时考虑切到 `Literal` 派发。

## 测试

```bash
# 跟 training/ 直接相关的单测
pytest tests/test_anima_train_migration.py        # CLI / YAML / parse_args 契约
pytest tests/test_anima_generate_xy.py            # sister script `_T.X` 访问模式
pytest tests/test_plugin_registry.py              # registry 三件套 + Protocol hook
pytest tests/test_infonoise.py                    # InfoNoise EMA 公式 + 状态机 + factory（含
                                                  # 论文 Algorithm 1 公式 codify，防 P0-2 类回归）
```

`test_plugin_registry.py` 防回归断言：`phases/optimizer.py` 不该再含 `if optimizer_type == "prodigy"` 字面量、`phases/models.py` 不该再 `AnimaLycorisAdapter(`、`sampling.py` 不该 `if sampler_name == "er_sde"`。

端到端验证靠用户**跑完整 LoRA 训练 + 评估出图**（ADR 0003 验收策略 R2）；单 PR 不强制 bit-for-bit。

## Sister script 契约

`runtime/anima_daemon.py` / `anima_generate.py` / `anima_reg_ai.py` 用 `import anima_train as _T` 然后 `_T.find_diffusion_pipe_root` / `_T.load_anima_model` / `_T.load_vae` / `_T.load_text_encoders` / `_T.sample_image` / `_T.enable_xformers` / `_T.resolve_path_best_effort`，多模型后另加 `_T.get_family` / `_T.resolve_family`（按族派发的咽喉；采样调用面实际走 `family.sample_image`）。

这 7 个名字 + 测试用的 `parse_args` / `apply_yaml_config` / `save_training_state` / `load_training_state` 都在 `runtime/anima_train.py` 顶层 re-export。修改 `training/` 内部时不要破坏这层契约——`tests/test_anima_generate_xy.py` 会捕获。

## 历史 + 延伸

- [ADR 0003](../../docs/adr/0003-anima-train-refactor.md) — 完整设计文档 + 9 个变体落地案例
- [ADR 0001](../../docs/adr/0001-lokr-via-lycoris-lora.md) — 为什么 adapter 走 lycoris-lora pip 包
- [ADR 0002](../../docs/adr/0002-webui-self-update.md) — Ctrl+C handler 现位置 `phases/resume.py:run()` 内的 `ctx.handle_interrupt`
- [`studio/domain/training.py`](../../studio/domain/training.py) — `TrainingConfig` 的 Literal 枚举 + 字段 `_meta(group, show_when, ...)` 给前端 UI（`studio/schema.py` 是兼容 shim）

PR #56 / #57 / #58 是 ADR 0003 的三刀执行记录，commit history 干净，回滚精确。
