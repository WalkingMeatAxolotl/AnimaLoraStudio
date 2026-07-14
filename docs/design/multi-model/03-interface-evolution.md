# 03 — ModelFamily 接口契约与长期演化（压力测试视角）

- 状态：视角文档，供 `04-synthesis.md` 收敛。不推翻 `00-decisions.md` D1–D9；本文把其 §6 接缝草案细化到方法签名级别，并给出修正建议（§4.5）。
- 日期：2026-07-13
- 前置阅读：`00-decisions.md`（背景、K2 档案、已锁定决策）。

## TL;DR

- **v1 冻结面（现在定死）**：`ModelSpec` 纯数据字段集；`ModelFamily` 八个方法的名字与签名（`load_dit / load_vae / load_text / encode_text_for_batch / forward_train / sample_image / lora_preset / lora_metadata`）；共享循环七条不变量 —— 其中最硬的三条是 **latent 恒 5D `(B,C,T,H,W)` 且 v1 内 `T==1`**、**t ∈ (0,1) 连续、x_t=(1−t)·x₀+t·ε、target=ε−x₀（rectified flow，唯一支持的 objective）**、**text cond 对循环完全 opaque**。
- **最大裂点（实地核对发现）**：D8 说「派发点 = `phases/models.py`」，但 loader 三件套 + `sample_image` 有 **5 个循环外调用方**（`anima_generate` / `anima_daemon` / `anima_reg_ai` / `studio/services/eval_samples` / `sample_runner`），K2 的 Generate/评估/RegAI 会全部绕过派发直接崩 —— registry 必须暴露在 `anima_train` 公共命名空间这一个咽喉上（§1.3、§4.5-①）。
- **压力测试结论**：Qwen-Image 几乎零裂（证明接口够用）；Wan 2.2 视频裂在 dataset/预览而非 forward（→ `temporal` 拒绝位字段，不进 v1 功能面）；SDXL 裂在 t 语义与 target 代数（→ `objective` 拒绝位字段，共享循环只做 rectified flow）；自回归 token 模型全线崩（证明「接口拒绝它」正确 —— 那是另一个产品，不是 family #N）。
- **反过度抽象**：采样器数值代码、文本编码内部、checkpoint 形状推断、显存策略、guidance 约定 —— 全部明确不抽象，宁可将来 per-family 重复（§4.3、§4.4）。

---

## 1. 实地核对：接口切点 vs 真实调用面

先盘点今天代码里「换一个模型族就必须变」的每一处，作为接口切分的证据基础。

### 1.1 训练主循环（`runtime/training/loop.py`）

| 触点 | 位置 | 事实 |
|---|---|---|
| batch 键 | `loop.py:201-219` | `captions`（恒有）、`latents`（缓存路径，5D）或 `pixel_values`（非缓存）、`navit_latents`（list，Anima-only）、`masks` / `loss_weight` / `is_reg`（可选） |
| 非缓存 VAE encode | `loop.py:215-219` | `pixels.unsqueeze(2)` → `[B,C,1,H,W]` 后 `vae.model.encode` —— 单帧假设写死在 unsqueeze |
| 文本编码（每步在线） | `loop.py:222-254` | Qwen 编码 + T5 tokenize + `ctx.model.preprocess_text_embeds(qwen_emb, t5_ids, t5xxl_weights)`（:242，融合层在 DiT 权重内）+ **pad 到 512**（:243-244）+ **kv_trim 桶截断**（:247-254）。全部 Anima 私货，但目前裸露在循环里 |
| t 采样 | `loop.py:258` | `ctx.timestep_sampler.sample(bs, device)` —— 已是 plugin（`timestep_samplers/baseline.py:30-38`），t ∈ (0,1)（`timestep_sampling.py:23`） |
| 分辨率 shift 修正 | `loop.py:263-270` | opt-in，SD3 sqrt 规则（`timestep_sampling.py:105-123`），`patch_spatial=2` 写死在 token 计数（`timestep_sampling.py:87-102`） |
| 加噪 + 目标 | `loop.py:288, 399-400` | `t_exp = t.view(-1,1,1,1,1)`（5D）；`noisy = (1-t)*latents + t*noise`；`target = noise - latents` —— rectified flow 代数在循环里 |
| pad_mask 构造 | `loop.py:310` | `zeros(bs,1,h,w)` —— Anima `concat_padding_mask` 的私有输入，循环替模型造 |
| 前向 | `loop.py:401-404` | `forward_with_optional_checkpoint(ctx.model, noisy, t.view(-1,1), cross, pad_mask, use_checkpoint)`；整个调用在 `torch.autocast`（:315）内 |
| 梯度检查点 | `model_loading.py:33-59` | 手工展开 Anima 内部 API：`prepare_embedded_sequence` / `t_embedder` / `t_embedding_norm` / `blocks` / `final_layer` / `unpatchify` —— 换族必换 |
| masked loss | `loop.py:408-411, 447-450` | `masks` (B,h,w) latent 分辨率 → `view(bs,1,1,h,w)` 广播到 (B,C,T,H,W)；加权 reduction `_masked_mean`（:111-120）只吃形状，family 无关 |
| loss / InfoNoise record / 加权 | `loop.py:413-446` | 全部只消费 `(pred, target, t, mask, loss_weight)` —— 符合 §6 草案「共享循环边界」判据 |
| NaViT 路径 | `loop.py:316-345`，`navit.py:113-129` | 调 `model.patchify_latents_to_tokens` / `model.forward_packed_navit` —— Anima 专属内部 API，D5 已定为 Anima-only 能力 |
| leap / SRA | `loop.py:346-396, 453-467` | 直接拿 `ctx.model` 多次前向 / 挂 hook —— 同为 Anima-only 能力（D5） |

**结论**：循环里真正需要搬进 family 的只有四块 —— 文本编码块（:222-254）、pad_mask 构造（:310）、前向派发（:401-404 + checkpoint 展开）、非缓存 encode 的 5D 组装（:215-219）。噪声/目标/loss/mask/权重代数全部留在循环（不变量见 §2.7）。

### 1.2 phase 协议（`runtime/training/phases/`）

编排顺序 `bootstrap → models → dataset → optimizer → resume → loop → finalize`（`phases/__init__.py:7`，`anima_train.py:130-136`）。每个 phase 是 `run(ctx: TrainingContext) -> None` 原地改 ctx。family 接口在各阶段的被调面：

| Phase | family 触点 | 证据 |
|---|---|---|
| bootstrap | **解析 family**（新增）：读 `args.model_family` 查 registry、能力校验；pause snapshot 覆盖 args（含未来的 `model_family`，snapshot freeze 天然保住跨 pause 的族一致性） | `phases/bootstrap.py:149-151`（snapshot 覆盖）、`178-188`（dtype 决策，family 无关） |
| models | **loader 三件套 + LoRA preset**：`load_anima_model`（:68-70）→ attention backend 开关（:63-75）→ `load_vae`（:78-79）→ `load_text_encoders`（:82-84）→ `build_adapter().inject()`（:89-91）→ `resume_lora`（:94-96）→ SRA（:99-117，Anima-only） | `phases/models.py` |
| dataset | 只消费 **latent 规格**：桶对齐 16px（`dataset.py:67`）、mask 下采样 //8（`dataset.py:719`）、token 计数 `patch_spatial=2`（`dataset.py:1077-1094`）、缓存判据（§1.4）；masked×navit 互斥闸（`phases/dataset.py:100-107`） | `phases/dataset.py:109-192` |
| optimizer | 零触点（`injector.get_param_groups` 走 adapter，`build_timestep_sampler` 走 plugin） | `phases/optimizer.py:31-91` |
| resume | `load_training_state` 灌 LoRA/optimizer/RNG（§1.5）；step-0 基线采样调 `run_sample`（:118-130） | `phases/resume.py:62-68` |
| finalize | `injector.save(final_path)`（LoRA 产物 metadata，§1.4）+ `eval_training_finished` 事件 | `phases/finalize.py:29-38` |

### 1.3 循环外调用面 —— D8 的修正证据（最重要的实地发现）

loader 三件套与 `sample_image` 的**全部**调用方：

| 调用方 | 位置 | 用途 |
|---|---|---|
| `phases/models.py` | `:68-84` | 训练 |
| `runtime/anima_generate.py` | `:136-160`（loader）、`:214, 380`（sample_image） | 测试页出图 CLI |
| `runtime/anima_daemon.py` | `:246-320`（loader）、`:744, 879`（sample_image） | Generate 常驻 daemon |
| `runtime/anima_reg_ai.py` | `:426-445`（loader）、`:491`（sample_image） | reg_ai 生成 |
| `studio/services/eval_samples.py` | `:546-628`（`import anima_train as _T` 后 `_T.load_anima_model` / `_T.load_vae` / `_T.load_text_encoders` / `_T.sample_image`） | 训练后评估出图 |
| `runtime/training/sample_runner.py` | `:78-86` | 训练中预览 |

`00-decisions.md` §6 草案与 D8 只写了「`phases/models.py` 查 registry」。**若只改这一点，K2 版本的 Generate / RegAI / 训练后评估会继续走 `load_anima_model` 直接崩**（`models.py:406-411` 只认 2048/5120 两档 `model_channels`，K2 width 6144 直接 `RuntimeError`）。好消息是所有旁路调用方都统一经过 `anima_train` 公共命名空间（`anima_train.py:88-94` re-export），派发只需要收口在这一个咽喉：registry 查询函数进 `anima_train` 公共 API（§2.3）。这不推翻 D8（supervisor/cmd_builder 不动、入口脚本不变依旧成立），是对派发面的补全。

### 1.4 缓存与产物

- **latent npz**：键 = `latent` / `latent_flipped` / `mask` / `mask_flipped` / `bucket_w` / `bucket_h`（`dataset.py:1160-1198`）；latent 存 `(16,1,h,w)`，collate stack 成 5D（`dataset.py:1320-1331`）。判据 = 存在性 + mtime + bucket 尺寸 + 键存在性（`dataset.py:985-1039`），**无任何模型/VAE 指纹**（`00-decisions.md` §3 已列为隐患，Phase-1 PR-1 修）。不兼容缓存有现成的「删除重 encode」路径（`dataset.py:1008-1011`）可挂新判据。
- **LoRA 产物 metadata**：`ss_network_dim/alpha/module` + `ss_network_args`（`utils/lycoris_adapter.py:363-380`），其中 `preset: "anima_full"` 已经是事实上的 per-family 标记但没有正式字段；target 列表与 `lora_prefix="lora_unet"` 写死在 `utils/lokr_preset.py:14-26`。
- **文本缓存**：现状无（每步在线，`loop.py:222-254`）；D3 给 K2 引入 varlen 预缓存。

### 1.5 恢复 / 暂停

`save_training_state`（`state.py:35-68`）存：`lora_state_dict` / `optimizer` / `scheduler` / `rng` / `monitor` / `timestep_sampler` / `sra_aligner` / `scaler` —— **不含任何基模型权重，也不含 family 标识**。pause snapshot 冻结整份 args（`phases/bootstrap.py:25-62`），未来 `model_family` 字段随 args 冻结，跨 pause 族一致性由 snapshot 机制免费获得。但**跨族误 resume 无防护**：`injector.load_state_dict(strict=False)` 对键完全不匹配的跨族 LoRA 只发 warning（`state.py:95-102`），会静默变成「全 missing 冷启动」继续训 —— 比崩溃更糟（§4.5-⑦）。

### 1.6 schema 与门控机制现状

- `TrainingConfig`（`studio/domain/training.py:24-30`，`extra="ignore"`）每字段带 `json_schema_extra={"group","control","show_when"?}`（:10-11）；4 个权重路径字段（:38-57）默认指向 Anima。
- `show_when` 求值器是**纯字段值比较**文法：`==` / `!=` / `||` / `&&`（`studio/domain/config_prune.py:49-66`），前端 `schema.ts` 逐字镜像；求值为假的字段落盘时被裁剪（`config_prune.py:69-90`）。**能力位不能直接进表达式** —— 门控必须编译成 `model_family==xxx` 形式的字段比较（§2.4）。
- 采样默认值（shift=3.0、er_sde/simple 白名单）写死在 `sampling.py:32-33, 344`。

---

## 2. 接口 v1 精确定义

### 2.1 `ModelSpec` —— 纯数据（frozen dataclass，声明式常量）

```python
# runtime/training/families/spec.py（示意；全部 frozen dataclass，无行为）

@dataclass(frozen=True)
class LatentSpec:
    fingerprint: str        # 缓存指纹，如 "wan21-f8c16"（D6：Anima 与 K2 同值 → 缓存共享）
    channels: int           # 16（今天散落在 models.py:456 / dataset.py:1284 / sampling.py:239 等 ≥8 处）
    spatial_stride: int     # 8（VAE f8；dataset.py:719 的 //8、sampling.py:332 的 //8 的单一来源）
    patch_spatial: int      # 2（models.py:416 / dataset.py:1080 / timestep_sampling.py:87 的单一来源）
    patch_temporal: int     # 1
    temporal: bool          # v1 恒 False —— 视频拒绝位（§3.2 结论），不是功能开关
    # 派生属性：align_px = spatial_stride * patch_spatial（=16，桶/采样尺寸对齐用，
    # 对应 dataset.py:67 与 sample_runner.py:54-55 两处写死的 16）
    # 注意：VAE 归一化 mean/std 向量不进 spec —— 那是权重侧事实，属 load_vae 内部（models.py:468-475）

@dataclass(frozen=True)
class TextSpec:
    strategy: Literal["online", "cached_varlen"]   # D3：Anima=online，K2=cached_varlen
    max_seq_len: int                               # 两族都是 512
    fingerprint: str                               # TE 指纹（cached_varlen 的缓存键成分；online 时仅记录用）
    # embed 内部形状（Anima (B,512,1024) 融合后 / K2 (B,≤512,12,2560) 堆叠）不进 spec：
    # cond 对循环 opaque（§2.7-4），形状是 family 内部事务

@dataclass(frozen=True)
class SamplingDefaults:
    samplers: tuple[str, ...]        # 白名单（Anima: ("er_sde","dpmpp_3m_sde")，对应 sampling.py:32）
    schedulers: tuple[str, ...]      # ("simple","sgm_uniform")
    default_sampler: str
    default_scheduler: str
    default_steps: int               # 25
    default_cfg: float               # 4.0
    shift_policy: ConstantShift | ResolutionAwareShift
    # ConstantShift(3.0)＝Anima（sampling.py:344）；
    # ResolutionAwareShift(base_shift=0.5, max_shift=1.15, ref_seq_len=...)＝K2。
    # typed union 而非裸 float —— 这就是版本化的能力位：新增策略＝加一个 dataclass 成员，
    # 不改方法签名。仅影响「采样端 sigma 表 + schema 默认值 overlay」；
    # 训练端 t 采样仍走 timestep_sampler plugin（family 只供默认参数，见 §4.5-⑨）。

@dataclass(frozen=True)
class ModelSpec:
    family_id: str                   # "anima" / "krea2" —— registry 键 & schema enum 值，永不改名
    display_name: str
    objective: Literal["rectified_flow"]   # v1 唯一合法值 —— eps/DDPM 拒绝位（§3.3 结论）
    latent: LatentSpec
    text: TextSpec
    sampling: SamplingDefaults
    capabilities: frozenset[str]     # 词表见 §2.4
    lora: LoraOutputSpec             # prefix（"lora_unet"）、preset_name（"anima_full"/"krea2_full"）
    config_defaults: Mapping[str, Any]
    # 创建 version 时叠进初始 yaml 的 per-family 默认值 overlay（如 K2 timestep_shift=2.5）。
    # 作者写时落盘，不做 runtime 魔法 —— 用户在 Train 页看到的就是生效值。
```

**错误约定**：`ModelSpec` 构造后由 registry 注册时做一次自洽校验（能力词表合法、`cached_varlen` ⇒ 不含 `caption_tag_ops` 能力等），违反 → `ValueError`，进程启动即死（对齐 `phases/bootstrap.py:121-128` 的 plugin schema 启动期校验风格）。

### 2.2 `ModelFamily` —— 行为（逐方法契约）

```python
class TextStack(Protocol):
    """family 私有的文本编码器持有物（Anima: Qwen+qwen_tok+t5_tok；K2: Qwen3-VL）。
    对外唯一承诺：encode 永远成功 —— cached_varlen 实现在 cache miss 时懒加载 TE 权重、
    编码、写缓存（预期只在用户临时改采样 prompt 时发生）。"""

@dataclass
class LoadedModels:
    dit: nn.Module
    vae: "VAELike"       # 协议同 VAEWrapper：encode (B,3,T,H,W)->(B,C,T,h,w)、
                         # decode (B,C,T,h,w)->(B,3,T,H*8,W*8)、.dtype、tiling 内部自治
                         # （models.py:130-203 现有语义原样冻结；dataset 缓存 dataset.py:1115
                         #  与 roundtrip 自检 phases/dataset.py:260-261 直接依赖）
    text: TextStack


class ModelFamily(Protocol):
    spec: ModelSpec

    # ───────────────────────────── 加载（models_phase + 全部旁路调用方，§1.3）
    def load_dit(self, path: str, device, dtype, *,
                 attention_backend: str,       # "flash_attn" | "xformers" | "none"
                 repo_root: Path) -> nn.Module:
        """替代 load_anima_model（models.py:339-447）。
        副作用自治：sys.path 注入、modeling 模块全局 attention 开关、
        checkpoint 形状推断（models.py:397-427）全是 family 内部事务。
        错误：权重覆盖率低 / 关键层缺失 → RuntimeError 带可操作文案
        （沿用 _load_weights_best_effort 风格，model_loading.py:302-308）。"""

    def load_vae(self, path: str, device, dtype, *, tiling: str) -> "VAELike": ...
        # Anima 与 K2 同款 Qwen-Image VAE（D6/D7）——两个 family 可以共同调用同一个
        # 共享 loader 工具函数，但方法本身在 family 上（第 3 族 VAE 不同款时零迁移）。

    def load_text(self, paths: TextPaths, device, dtype, *,
                  purpose: Literal["train", "generate"]) -> TextStack: ...
        # 替代 load_text_encoders（models.py:481-524）。purpose 保留现有
        # comfy_qwen 分叉语义（sampling.py:263-265：Generate runtime 用 comfy 编码器）。

    # ───────────────────────────── 文本条件（loop.py:222-254 整块下沉）
    def prepare_text_cache(self, captions: Iterable[str],
                           extra_prompts: Iterable[str]) -> None:
        """cached_varlen 族的预缓存钩子（Phase 2 接入 dataset_phase 之后、loop 之前）；
        extra_prompts = sample_prompts + negative（bootstrap 后已知，resume.py:110-113）。
        online 族实现为 no-op。缓存键 = sha256(caption) + spec.text.fingerprint（D3）。
        预缓存完成后 family 可自行释放 TE 权重（K2 省 ~8GB 常驻；TextStack.encode
        的懒加载承诺保证之后仍可用）。"""

    def encode_text_for_batch(self, text: TextStack, dit: nn.Module,
                              captions: list[str], device, dtype) -> "TextCond":
        """每步调用（loop 内、no_grad 下）。返回 opaque cond，循环原样透传给
        forward_train —— pad-to-512（loop.py:243-244）、kv_trim（loop.py:247-254）、
        preprocess_text_embeds（loop.py:242，融合层在 DiT 权重内所以需要 dit 参数）
        全部收进 AnimaFamily 实现。K2 实现＝查缓存（命中率应为 100%，
        tag ops 已被能力门控关闭 → caption 逐步确定，见 §2.4 交叉不变量）。
        约定：cond 首维语义上对应 batch，但循环不得索引/切片/pad 它。"""

    # ───────────────────────────── 训练前向（loop.py:401-404 + model_loading.py:33-59 下沉）
    def forward_train(self, dit: nn.Module,
                      noisy: Tensor,      # (B,C,T,H,W)，autocast dtype，T==1（v1 不变量）
                      t: Tensor,          # (B,) float32，t∈(0,1)
                      cond: "TextCond",
                      *, use_checkpoint: bool) -> Tensor:
        """返回 v_pred，形状 == noisy。
        - 调用点：标准路径每 micro-batch 一次（loop.py:401），autocast 上下文内
          （loop.py:315）——family 不得自行开 autocast。
        - t 的 reshape 自治：Anima 内部 view(-1,1)（今天 loop.py:402 替它做的）；
          采样端 1D expand（sampling.py:371-377）与训练端 (B,1) 的差异证明
          t 形状按摩本来就是 family 私事。
        - padding_mask（loop.py:310）由 family 内部构造 —— 它是 Anima
          concat_padding_mask 的私有输入，不是共享循环概念。
        - 梯度检查点策略（逐 block 重算，model_loading.py:39-58）family 内部自治。
        - 错误：不做 NaN 检查（循环已有，loop.py:476-479）；形状违约 → ValueError。"""

    # ───────────────────────────── 采样（sampling.py:248-472 整函数级别成为 family 方法）
    def sample_image(self, models: LoadedModels, prompt: str, *,
                     height: int = 1024, width: int = 1024,
                     steps: int = 25, cfg_scale: float = 4.0,
                     negative_prompt: str | None = None,
                     sampler_name: str | None = None,     # None → spec.sampling.default_sampler
                     scheduler: str | None = None,
                     device="cuda", dtype=torch.bfloat16,
                     step_callback=None, phase_callback=None,
                     seed: int | None = None) -> "PIL.Image.Image":
        """现有 sample_image 签名（sampling.py:248-259）就是 6 个调用方的事实标准，
        原样冻结，只把「5 个模型位置参数」收拢成 LoadedModels。
        sigma 表 / CFG 合批 / CONST x0 换算（sampling.py:402）/ 初始噪声 / VAE decode
        offload 决策全部 family 内部。不支持的 sampler/scheduler → ValueError
        （沿用 _resolve_parity_sampler_scheduler 风格，sampling.py:36-41）。
        不做 build_sampler() 细粒度拆分 —— 见 §4.5-④。"""

    # ───────────────────────────── LoRA（models_phase :89-91 / finalize :31-32）
    def lora_preset(self) -> dict[str, Any]:
        """lycoris apply_preset 字典（今天的 ANIMA_PRESET，utils/lokr_preset.py:14-26）。
        target_name / exclude_name / lora_prefix 全在这里。"""

    def lora_metadata(self) -> dict[str, str]:
        """附加进 safetensors metadata 的 per-family 键值，至少含
        {"ss_model_family": spec.family_id}；merge 进现有 ss_network_args
        （lycoris_adapter.py:363-380）。Comfy 侧 K2 键名约定（00-decisions §7
        open question）关闭后可能追加键，追加不算破坏契约。"""
```

**不进接口的方法（明确否决过的候选）**：`build_noise`（噪声与 offset 族无关，`noise.py` 共享）、`compute_target`（rectified flow 代数是循环不变量，§3.3）、`bucket_strategy`（dataset 只消费 `LatentSpec`）、`memory_plan`（D2 搁置，显存策略不是 family 轴）。

### 2.3 registry 与派发

```python
# runtime/training/families/__init__.py（与现有 7 套 plugin registry 同款，D1/§6-4）
_REGISTRY: dict[str, ModelFamily] = {}

def get_family(family_id: str) -> ModelFamily:
    """未知 id → ValueError，文案列出已注册 id（对齐 optimizers/losses registry 风格）。"""

def resolve_family(args) -> ModelFamily:
    return get_family(str(getattr(args, "model_family", "anima") or "anima"))
```

- 训练侧：`phases/bootstrap.py` 里 `ctx.family = resolve_family(args)`（能力校验也在此，fail-fast 于任何权重下载/加载之前）；`phases/models.py` 全部 `load_*` 与 `lora_preset` 改经 `ctx.family`。
- **旁路侧（§1.3 修正）**：`anima_train.py` 公共命名空间新增 re-export `get_family / resolve_family`；`anima_generate` / `anima_daemon` / `anima_reg_ai` / `eval_samples` 把 `_T.load_anima_model(...)` 三件套 + `_T.sample_image(...)` 替换为 `family = _T.resolve_family(cfg)` 后走 family 方法。cmd_builder / supervisor / 入口脚本零改动（D8 保持）。
- `TrainingContext` 增 `family: ModelFamily` 与 `models: LoadedModels`；`ctx.qwen_model / qwen_tok / t5_tok`（`context.py:58-60`）退役为 AnimaFamily 的 TextStack 内部字段（Phase-1 PR-2 迁移点）。

### 2.4 能力集（capabilities）与 schema / show_when / validator 接线

**词表（v1）**：`navit`、`sra`、`leap`、`compile_blocks`、`caption_tag_ops`（shuffle/dropout/keep_tokens 一组）、`online_text`、`text_cache`、`masked_loss`。Anima = 全量减 `text_cache`；K2 = `{masked_loss, text_cache}`（D5）。加词零成本（frozenset），删词/改语义 = 破坏性变更须过 04-synthesis。

**接线三层（利用现状机制，零新文法）**：

1. **schema 层（作者写时展开）**：`show_when` 求值器只认字段值比较（`config_prune.py:49-66`），能力位不能直接进表达式。在 `studio/domain/common.py` 加 authoring-time helper：

   ```python
   def cap_gate(capability: str) -> str:
       """从静态能力矩阵渲染 show_when 表达式。
       cap_gate("navit") → "model_family==anima"
       第 3 族支持 navit 时 → "model_family==anima||model_family==foo"（改矩阵一处）。"""
   ```

   `navit_packing` 等字段的 `_meta(...)` 追加 `show_when=cap_gate("navit")`（与现有 show_when 用 `&&` 复合）。前端 `schema.ts` / 落盘裁剪 `config_prune.py` / YAML 预览三个求值器**全部零改动** —— 它们看到的仍是普通字段比较表达式。这符合「作者写时规范化」而非 runtime 解析新文法。
2. **validator 层（双保险）**：`TrainingConfig` 加 `model_validator`：字段开启但 family 无对应能力 → 校验错误（拦手写 yaml / 裸 CLI，schema 层只管 UI 可见性与落盘裁剪）。
3. **runtime 层（最后防线）**：`phases/bootstrap.py` 里 `ctx.family` 解析后跑同一份校验（config 可能绕过 studio 直达 CLI）。

**交叉不变量**：`spec.text.strategy == "cached_varlen"` ⇒ `caption_tag_ops ∉ capabilities`。缓存键是 caption 内容 hash（D3），tag shuffle/dropout 会让每步 caption 漂移、缓存永 miss；registry 注册时校验（§2.1 错误约定），不留给各族自觉。

**D7 落地**：`model_family: Literal["anima","krea2"] = "anima"`（老配置零迁移）；`t5_tokenizer_path` 加 `show_when="model_family==anima"`；`transformer_path/vae_path/text_encoder_path` 三字段跨族复用不动。

### 2.5 缓存指纹协议

**latent npz（Phase-1 PR-1，独立价值）**：

- npz 追加两个标量键：`latent_fingerprint`（bytes，如 `b"wan21-f8c16"`）与 `layout_version`（int，v1=1）。键集其余六项（§1.4）冻结。
- `_is_cache_valid`（`dataset.py:985-1039`）追加两条判据：`latent_fingerprint != family.spec.latent.fingerprint` 或 `layout_version` 不识别 → 走既有「删除重 encode」路径（`dataset.py:1008-1011`）。
- **grandfather 规则**：无指纹键的存量 npz 视为 `wan21-f8c16`（历史上只有这一个 VAE 产出过缓存）——避免升级即全量重 encode。写入侧从 PR-1 起恒写指纹。
- D6 兑现方式：指纹是**latent 空间身份**而非 family 身份 —— Anima 与 K2 同为 `wan21-f8c16`，同一数据集两族互训零重算，无需任何显式「共享」逻辑。

**文本缓存（Phase 2，K2 起用）**：

- 键 = `sha256(caption 最终文本)` + `spec.text.fingerprint`（TE 权重指纹）+ 格式版本。caption 最终文本 = trigger 前置后的确定串（§2.4 交叉不变量保证确定性）。
- 存储 varlen（只存非 padding token，D3）；位置/文件格式是 `00-decisions.md` §7 open question，不在本文冻结 —— 本文只冻结**键协议**与「family 内部自治、dataset 层零感知」的边界：batch 恒带 `captions`，K2 family 在 `encode_text_for_batch` 里自己查缓存。dataset.py 因此对文本缓存零改动。

### 2.6 LoRA 产物与恢复态的 per-family 约定

- **safetensors metadata**：`ss_network_args` 增 `"model_family"` 键（经 `lora_metadata()` 注入）；`preset` 字段值改为 `spec.lora.preset_name`。Anima 存量产物无此键 → 读取侧视为 `"anima"`（grandfather）。
- **resume state**：`save_training_state`（`state.py:35-47`）的 dict 增 `"model_family"` 顶层键；`load_training_state` 校验与当前 family 不符 → `RuntimeError`（修 §1.5 的静默冷启动隐患）。老 state 无键 → 视为 `"anima"`。
- **resume_lora**（`phases/models.py:94-96`）：加载前读 metadata 校验 family，同上 fail-fast。
- Comfy 侧 K2 键名（`lora_prefix` 取值）等 musubi 产物核对结论出来前，`KREA2_PRESET` 的 `lora_prefix` 不冻结 —— 这是 spec 数据不是接口形状，晚定不破坏契约。

### 2.7 共享循环对 family 的不变量要求（v1 冻结）

凡实现 `ModelFamily` 者必须满足；循环端以这些为公理，不做防御分支：

1. **latent 恒 5D `(B,C,T,H,W)`，且 v1 内 `T == 1`**（`spec.latent.temporal=False` 的运行时体现）。C/stride/patch 以 spec 为准。桶保证 batch 内同尺寸（`phases/dataset.py:222-250`）。
2. **t 语义**：`(B,)` float32，t ∈ (0,1)（`timestep_sampling.py:23`），t→1 为纯噪声端。加噪 `x_t = (1−t)·x₀ + t·ε`、目标 `target = ε − x₀`（`loop.py:399-400`）是**循环的**代数，family 不得重定义 —— `spec.objective` 只有 `"rectified_flow"` 一个合法值。
3. **v_pred 同形**：`forward_train` 返回值形状 == `noisy`；loss plugin（`loop.py:413`）、InfoNoise record（`loop.py:418-438`）、masked mean（`loop.py:447-450`）都靠这一条。
4. **cond opaque**：循环对 `encode_text_for_batch` 的返回值零操作（不 pad、不 trim、不 cat）——CFG 合批之类的 cond 拼接只发生在 family 自己的 `sample_image` 内（`sampling.py:379-389` 是先例）。
5. **mask 口径**：`(B,h,w)` float ∈ [0,1]，latent 分辨率（`dataset.py:719` 下采样、`loop.py:408-411` 广播）。mask 语义（0=零梯度区）族无关。
6. **autocast 归循环**：`forward_train` 在循环的 autocast 内被调（`loop.py:315`），family 不自行嵌套。
7. **RNG 纪律**：family 方法内不得消耗全局 torch RNG 做「可选路径」决策（先例教训：`loop.py:296-307` leap 掷骰子刻意用 Python random 保对照实验可比）；采样 seed 由调用方管理（`sample_runner.py:76-77`）。

凡只消费 `(latents, noise, t, pred, target, loss, mask, loss_weight, is_reg)` 的功能（InfoNoise、masked loss、losses、noise offsets、loss weighting、optimizer/scheduler、LoRA/LoKr 算法、梯度累积、NaN 防护、state save）—— 留在共享循环，对新 family **零修改**（§6-3 原样确认，实地核对 §1.1 证明成立）。

---

## 3. 压力测试：4 个假想的「第 3 个模型族」

方法：逐一把假想族按 §2 接口走一遍 bootstrap→models→dataset→loop→sampling→产物全链路，记录裂点。「裂」定义为：接口方法签名或循环不变量必须改变才能容纳。

### 3.1 Qwen-Image（20B MMDiT，同 VAE 同 latent 空间 —— 最近邻）

理论上最顺的一族：rectified flow、f8c16 同款 VAE（与 Anima/K2 同一 latent 空间）、自然语言 prompt（Qwen2.5-VL 编码器）、Linear-only LoRA target。

| 裂点 | 位置 | 裂法 | 处置 |
|---|---|---|---|
| 无 | forward/loop/dataset/mask | t∈(0,1)、v-pred、5D、f8c16 全部命中不变量 | — |
| 文本编码器不同（Qwen2.5-VL vs Qwen3-VL） | `encode_text_for_batch` | 不裂：cond opaque + cached_varlen 机制直接复用 | 走 K2 同款策略 |
| latent 指纹 | §2.5 | 不裂反而增益：同为 `wan21-f8c16` → 三族共享 latent 缓存 | 零代码 |
| 采样 guidance / 动态 shift（Flux 风格） | `sample_image` | 不裂：整函数在 family 内，`shift_policy` union 已容纳 | 加 dataclass 成员即可 |
| 20B 权重 → bf16 ~40GB，超 D2 的 32GB 下限 | 资源面 | **不是接口裂点** —— fp8/block swap 是 D2 显式搁置的正交轴 | 拒绝把 `memory_plan` 塞进 family（§2.2 否决清单） |

**判决：接口零修改容纳。** 这一族的意义是正向证明：`ModelSpec` 声明 + opaque cond + family 级 `sample_image` 的切分对「同世代 MMDiT」是充分的。它训不训得动是显存问题，不是抽象问题。

### 3.2 Wan 2.2 类视频模型（T > 1 —— 时间维全线冲击）

| 裂点 | 位置 | 裂法 | 处置 |
|---|---|---|---|
| 数据加载 | `dataset.py` 全文件 | ImageDataset 读单图；视频要抽帧/片段采样/帧数桶（token 数 ∝ T×h×w，打包与桶策略全变） | **out-of-scope**，拒绝位挡住 |
| npz 布局 | `dataset.py:1160-1198` | latent `(C,1,h,w)` 的 T=1 是隐式的；视频 latent 体积 ×T，npz 单文件策略、mtime 判据、`bucket_w/h` 二维键全部不够 | `layout_version` 预留了升级通道，但 v1 不实现 |
| token 计数 / res-shift | `timestep_sampling.py:87-102`、`dataset.py:1077-1094` | `h//ps × w//ps` 漏乘 T 维 patch 数 | 同上 |
| masked loss | `dataset.py:719`、`loop.py:408-411` | mask 是 (h,w) 单帧；视频需要 per-frame 或广播语义决策 | 同上 |
| 非缓存 encode | `loop.py:215-219` | `unsqueeze(2)` 写死单帧 | 该行随 §2.2 收进 family，裂点已内化 |
| **forward / loss / 不变量代数** | `loop.py:288, 399-450` | **不裂**：`t.view(-1,1,1,1,1)`、噪声、target、masked mean 全是 5D 广播代数，T>1 天然成立 —— 这是继承 Wan 血统 5D 布局的意外红利 | 保持 5D 布局正是为此 |
| 预览/评估 | `sample_image` 返回单张 PIL；eval 管线、Generate UI、monitor 缩略图全部图片假设 | 返回类型都得变 | out-of-scope |

**判决：视频不进 v1 接口，显式 out-of-scope。** 但拒绝方式是**字段而非散落断言**：`LatentSpec.temporal` v1 恒 `False`，dataset_phase 与 registry 注册校验各查一次。关键洞察：裂区集中在 dataset/预览两端，**循环中段（forward→loss）因 5D 布局天然免疫** —— 将来若做视频，是「新增 VideoDataset + sample_video 方法」的加法演化，不是 v2 接口断裂。因此 v1 唯一要付的成本就是保住 5D 不变量不许任何人「优化」成 4D（squeeze T 的诱惑在 S3 会出现，必须顶住）。

### 3.3 SDXL 级 UNet（eps-pred + DDPM 离散 timestep + 4ch VAE + CLIP 双 encoder）

| 裂点 | 位置 | 裂法 | 处置 |
|---|---|---|---|
| t 语义 | 不变量 #2、`timestep_samplers/*` | 离散 0..999 int vs 连续 (0,1)；全部 6 种采样 mode、InfoNoise 的 I-MMSE 统计、`timestep_schedule_shift` Möbius 代数失效 | **拒绝**：`spec.objective` 拒绝位 |
| target 代数 | `loop.py:399-400` | `target = ε − x₀` ≠ ε；若把 target 计算下放 family，InfoNoise record（:418-438）、loss_weighting 的 SNR 公式（`loss_weighting.py`）、leap 轨迹构造全要跟着参数化 —— 「共享循环」名存实亡 | 同上：target 留循环，非 RF 不支持 |
| min_snr 等加权 | `loop.py:444-446` | SNR 由 alphas_cumprod 定义，与 RF 的 t 参数化完全不同源 | 同上 |
| 4ch VAE | `LatentSpec.channels=4` | **不裂**：channels 本来就是 spec 字段；指纹 `sdxl-f8c4` 自动隔离缓存 | spec 已容纳 |
| UNet 前向签名 | `forward_train` | **不裂**：UNet 想要 4D 就在 family 内 `squeeze(2)/unsqueeze(2)` —— 5D 循环布局保住，转换是 family 私事 | 已容纳（也验证了 §3.2 的「顶住 4D 诱惑」） |
| CLIP 双 encoder + pooled embeds | cond | **不裂**：cond opaque，family 返回自定义 struct（text_embeds + pooled + time_ids） | opaque 边界的正向验证 |
| conv LoRA target | `lora_preset` | **不裂**：preset 字典本来就有 `enable_conv`（`lokr_preset.py:15`） | 已容纳 |
| 采样器 | `sample_image` | DDPM/DPM++ 离散 scheduler 栈与 CONST flow 栈无共享价值 | family 内部，本就不共享 |

**判决：坚持「共享循环只做 rectified flow」，SDXL 明确不支持。** 裂的三处（t 语义、target 代数、SNR 加权）恰好都是循环的**心脏**而非边缘 —— 参数化它们等于把循环拆成 per-objective 双份，这正是 D1 否决过的「按模型分训练脚本」换了个马甲。而**没裂**的四处（4ch、UNet 4D、双 encoder cond、conv target）证明 spec 字段 + opaque cond + family 内 sample_image 的边界画对了位置。SDXL 用户有成熟的 kohya 生态，本产品瞄准的 Anima/K2/Qwen-Image/Wan 一代全是 flow matching —— `objective` 字段留着，真要支持 eps 系那天，付「循环 v2 或 per-objective loop」的全价，不预付。

### 3.4 极端案例（自选）：自回归 token 图像模型（LlamaGen / Janus 风格）

离散 VQ codebook + causal transformer + cross-entropy loss，「LoRA 微调」在这种模型上完全成立 —— 所以它是对「接口边界在哪」最诚实的探针。

| 接口件 | 遭遇 |
|---|---|
| `load_vae` | 要装的是 VQ tokenizer，`encode` 返回 int token 而非连续 latent —— VAELike 协议（5D float）语义崩塌 |
| `LatentSpec` | channels/stride/patch 无意义；「latent 缓存」变成 token 序列缓存，指纹协议勉强能借壳但判据全错 |
| t / noise / target | 不存在。`forward_train(noisy, t, cond)` 三个参数两个没有意义 |
| loss | cross-entropy over codebook —— losses plugin（mse/huber）、masked loss 的空间广播、loss_weighting 全部无对象 |
| `sample_image` | 自回归解码循环，与 sigma 表/CFG 合批完全异构（这条倒是能实现——唯一能实现的方法） |
| timestep_sampler / InfoNoise / noise offsets | 整个 plugin 生态失去存在前提 |

**判决：接口拒绝它，且拒绝是对的。** 8 个方法里 6 个要么无法实现要么实现为谎言（空转 stub）。这不是「能力集关掉几个开关」能表达的残缺 —— 是**身份**不同：本训练器的身份 = 「连续 latent 空间 + 连续时间 rectified flow 的回归式 LoRA 训练器」。压力测试的价值在于分清两类假设：

- **身份假设**（拒绝位承载）：存在 VAE 连续 latent、存在连续 t、回归式逐元素 loss → `objective` + `temporal` + VAELike 协议；
- **偶然假设**（spec/方法承载）：通道数、stride、文本 stack、shift 公式、LoRA target、采样器 → 全部已参数化。

AR 模型想进来，正确姿势是新产品线（新 loop + 新 dataset 契约），不是 family #N。接口不为它留任何钩子。

---

## 4. 结论

### 4.1 v1 冻结面（现在定死，破坏须过 synthesis 评审）

1. `ModelSpec` 字段集（§2.1）：`family_id / display_name / objective / latent / text / sampling / capabilities / lora / config_defaults`；`family_id` 字符串永不改名（它进 yaml、进 LoRA metadata、进 resume state）。
2. `ModelFamily` 八方法的名字与位置参数（§2.2）。演化只走 keyword-only 带默认值的追加（§4.2-②）。
3. 共享循环七不变量（§2.7），特别是：5D `T==1`、t∈(0,1) 连续、`x_t=(1−t)x₀+tε`、`target=ε−x₀`、v_pred 同形、cond opaque、autocast 归循环。
4. npz 缓存键集（6 旧键 + `latent_fingerprint` + `layout_version`）与 grandfather 规则（§2.5）。
5. 文本缓存**键协议**（caption hash + TE 指纹 + 版本）与「dataset 零感知」边界（§2.5）；存储格式不冻结。
6. 产物/恢复态 family 标记与 fail-fast 校验（§2.6）。
7. registry 派发协议：字符串 id、未知 id 报错列已注册项、经 `anima_train` 公共命名空间覆盖全部 6 个调用面（§2.3）。
8. 能力门控三层接线（cap_gate 展开 → pydantic validator → bootstrap 校验）与 `cached_varlen ⇒ ¬caption_tag_ops` 交叉不变量（§2.4）。

### 4.2 预留扩展点（版本化 / 能力位，不实现只留缝）

1. `capabilities` frozenset —— 加词零成本，是最主要的软演化轴。
2. 方法追加参数一律 **keyword-only 且带默认值**；**禁止 `**kwargs` 黑洞**（吞 typo、掩盖契约漂移；需要从 dataset 向 forward 穿新数据时走 batch dict 或 cond struct，不走签名）。
3. `shift_policy` typed union（§2.1）—— 新采样时刻表策略 = 加成员，不改签名。
4. `spec.objective`：v1 仅 `"rectified_flow"`；将来若扩 = 明知要付 per-objective loop 的全价（§3.3）。
5. `LatentSpec.temporal`：v1 恒 False；视频解锁点已定位在 dataset/预览两端（§3.2），循环免改。
6. npz `layout_version`：视频/新布局的升级通道。
7. `lora_metadata()` 返回 dict 可追加键（Comfy K2 键名核对结论的落点）。
8. `TextStack` / `TextCond` 是 per-family opaque 类型 —— 内部表示（融合张量 vs 12 层堆叠 vs 双 encoder struct）演化自由。

### 4.3 明确不抽象清单（宁可将来 per-family 重复）

| 不抽象项 | 理由 |
|---|---|
| 采样器/调度器数值代码（er_sde / dpmpp_3m_sde / K2 的 FlowMatchEuler 动态 shift） | Comfy parity 栈（`sampling.py` 全文件，含刻意复刻的「二次 shift」怪癖 :149-169）与 musubi 参考实现根本不同源；共享 Sampler 基类只会制造伪共性。inference_samplers plugin registry 保留为 **Anima 内部**组织方式，不上升为跨族契约 |
| 文本编码内部（Qwen+T5+LLMAdapter 融合 vs Qwen3-VL 中间层堆叠） | 连 helper 都不共享 —— `text_encoding.py` 400 行 T5 权重语法解析对 K2 是纯噪音 |
| checkpoint 形状推断 / 前缀 remap 启发式 | `_load_weights_best_effort`（`model_loading.py:272-314`）作为**工具函数**可被两族调用，但「从形状猜配置」（`models.py:397-427`）的推断逻辑 per-family 各写各的 |
| 梯度检查点展开策略（`model_loading.py:39-58`） | 逐 block 结构是模型私货 |
| attention backend 全局开关注入（`models.py:361-394`、`model_loading.py:66-118`） | 模块级全局变量开关是 Anima modeling 的历史私货，K2 modeling 按自己的方式来 |
| guidance 约定（K2 的 `cond + g·(cond−uncond)` vs Anima CFG） | `sample_image` 内部事务 |
| 显存策略（fp8 / block swap / offload 阈值） | D2 显式搁置；资源轴 ⊥ family 轴（§3.1） |
| eps/DDPM target、离散 timestep | §3.3 判决 |
| dataset / 桶 / 打包 / mask 管线 | 只消费 `LatentSpec` 四个整数，family 零注入点 |

### 4.4 反过度抽象论证

- **n=2，且第 2 个还没跑通**。抽象的最低成本时点是「第 2 个实例落地时」，最低风险形态是「刚好覆盖两个实例都被证实需要的面」。本文接口的每个方法都对应 §1 里一处**已存在的** Anima 调用点 + 一处 K2 档案里**已确认的**差异（`00-decisions.md` §2）—— 没有一个方法是为假想族预置的。压力测试产出的是两个**拒绝位字段**（`objective` / `temporal`）和若干「将来加法演化」的定位结论，不是预置钩子。
- **diffusers「刻意复制」哲学的适用面**：diffusers 在 pipeline/模型层逐模型复制而不建基类森林，代价是改一个 bug 要改 N 份 —— 它们有几百个 pipeline，我们只有 2-4 个族，复制成本更低、而错误抽象的耦合成本一样高。§4.3 清单里的每一项都是「两族实现看似相似但不同源」的地方 —— 共享它们意味着 Phase-1 的「对 Anima 零行为变化」承诺（`00-decisions.md` §5）要在每次 K2 改动时重新验证一遍。
- **既有 7 套 plugin registry 的经验边界**：它们成功是因为 plugin 是**叶子**（optimizer/loss 单点行为，接口 1-3 个方法，实例可长到几十个）；ModelFamily 是**枝干**（横跨 6 个 phase + 5 个旁路调用面），实例只会有个位数。所以 family 接口宁可方法粗（`sample_image` 整函数级）不要细（否决 `build_sampler()`），细粒度留给族内部的 plugin registry 去做。
- **克制的具体体现**：不抽象 target 计算（S3 会舒服些但代价是循环心脏参数化）；不抽象 dataset（S2 会舒服些但代价是给图片管线塞视频钩子）；不做 `**kwargs`；不做能力位新文法（编译成现有 show_when）；`load_vae` 明知两族同款也放 family 上而不是共享单例（第 3 族 VAE 不同款时零迁移，成本只是一行委托）。

### 4.5 对 `00-decisions.md` §6 草案的修正建议清单

| # | 草案原文 | 修正 | 依据 |
|---|---|---|---|
| ① | D8/§6-4「派发点 = `phases/models.py` 查 registry」 | 派发收口在 `anima_train` 公共命名空间（`get_family/resolve_family` re-export），覆盖 `anima_generate`:136 / `anima_daemon`:246,299 / `anima_reg_ai`:426 / `eval_samples.py`:582 / `sample_runner`:78 共 5 个旁路调用面；否则 K2 的 Generate/评估/RegAI 直接 `RuntimeError`（`models.py:411`） | §1.3 |
| ② | `encode_text` 一笔带过 | 明确 cond 对循环 **opaque**：pad-to-512（`loop.py:243-244`）与 kv_trim（`loop.py:247-254`）必须随编码块下沉进 AnimaFamily —— 草案不写清这两行会留在循环里成为暗桩 | §1.1、§2.2 |
| ③ | `forward_train(model, noisy, t, cond)` | 追加：pad_mask 构造（`loop.py:310`）收进 family（Anima 私有输入）；t 形状按摩 family 自治；autocast 归循环 | §2.2、§2.7-6 |
| ④ | `build_sampler()` 独立方法 | 撤销：冻结面直接是 `sample_image(...)` 整函数（现有 6 caller 的签名已是事实标准，`sampling.py:248-259`）；sigma/CFG/decode-offload 是一体的，细拆无收益 | §2.2 |
| ⑤ | ModelSpec 无拒绝位 | 增 `objective`（v1 仅 `"rectified_flow"`）与 `LatentSpec.temporal`（v1 恒 False）—— 把 S2/S3 判决固化为字段 + registry 注册校验，不散落断言 | §3.2、§3.3 |
| ⑥ | 「latent 指纹」只提概念 | 定协议细节：指纹 + `layout_version` 写进 npz 键（非文件名）；`_is_cache_valid` 加两判据走既有删除路径（`dataset.py:1008-1011`）；**无键存量缓存 grandfather 为 `wan21-f8c16`**，避免升级全量重 encode | §2.5 |
| ⑦ | 未提恢复面 | resume state 与 LoRA metadata 各加 `model_family` 标记 + 加载时 fail-fast：现状跨族 resume 会因 `strict=False`（`state.py:96`）静默全 missing 冷启动 | §1.5、§2.6 |
| ⑧ | D5「复用 show_when 裁剪机制」未给接线 | 求值器只认字段比较（`config_prune.py:49-66`）→ 能力位经 authoring-time `cap_gate()` 展开成 `model_family==...` 表达式，三个求值器零改动；另加 pydantic validator + bootstrap 双保险；补交叉不变量 `cached_varlen ⇒ ¬caption_tag_ops` | §2.4 |
| ⑨ | K2 分辨率感知 shift 的归属未定 | 定为 `spec.sampling.shift_policy`（typed union），只影响采样端 sigma 表与 schema 默认值 overlay；训练端 t 采样仍走 timestep_sampler plugin —— 且**不要**与现有 opt-in `timestep_shift_resolution_aware`（`loop.py:174-182`，SD3 sqrt 规则）合并，两者公式不同、语义不同（前者是族默认时刻表，后者是多分辨率校准修正） | §2.1 |
| ⑩ | 文本缓存只定了失效键（D3） | 补边界决策：缓存 family 内部自治、dataset 零感知（batch 恒带 `captions`）；`TextStack.encode` 承诺「永远成功」（miss 懒加载 TE）；预缓存范围 = 数据集 caption + sample_prompts + negative（使 K2 训练期可释放 ~8GB TE 常驻） | §2.2、§2.5 |
