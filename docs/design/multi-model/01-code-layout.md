# 多模型支持 · 01 目录结构与代码归属（视角一）

- 状态：设计稿，供 `04-synthesis.md` 收敛。
- 日期：2026-07-13
- 前置阅读：[`00-decisions.md`](00-decisions.md)（K2 档案、已锁定决策 D1-D9、分阶段计划）。本文不推翻任何已锁定决策，只在其框架内回答「代码放哪里」。
- 姊妹视角：`02-ecosystem-survey.md`（生态对标）、`03-interface-evolution.md`（接口冻结面）。接口的**方法签名**归 03，本文只定**文件归属**；两者边界在 §4.3 说明。

## TL;DR

- **推荐按层切（方案 A）**：`modeling/<family>/` 放结构定义、`runtime/training/families/<family>/` 放行为适配（ModelFamily 实现，即 D1 的适配层）、`studio/services/models/families/<family>.py` 放资产清单。族名字符串（`anima` / `krea2`）是贯穿三层的唯一 join key，由 `validate_schema_consistency` 模式（与既有 7 套 plugin registry 同款）在启动期锁一致。否决按族切自包含包：一个族横跨「有 torch 的训练子进程」与「无 torch 的 studio server」两种运行时，塞进一个包要么全 lazy import 要么拆两半，等于退化回按层切但丢了依赖方向卫生（`modeling → utils → runtime → studio`，`docs/AGENTS.md` §3.1）。
- **加第 3 个模型族** = 新建约 8 个文件（三层各一个目录/文件）+ 4 处一行式注册（runtime registry、studio registry、schema Literal、能力集默认值），共享循环（masked loss / InfoNoise / losses / optimizer / eval）**零修改**。
- **入口脚本不改名**（D8 已锁定）。`anima_` 前缀解释为产品名（Anima LoRA Studio）而非模型族限定，文档口径统一；未来若要改名的完整成本清单收录在 §5.2 备查。
- **exec-load 退役，改正常 import**。该机制是外部 diffusion-pipe checkout 时代的遗产（初始 commit `c6bd6b64` 即有；PR #303 移入 `modeling/` 后仅把新位置加到候选最前）。今天它的唯一现实产出是「双模块身份」病：attention backend 全局开关要跨至多 6 个模块别名传播（`runtime/training/model_loading.py:81-88`、`runtime/training/models.py:365-368`），多一族别名矩阵翻倍。`find_diffusion_pipe_root` 函数名保留（sister script 契约「可加不可减」），实现塌缩为常量 shim。
- **utils/**：算法（族无关）与 preset（族相关）分离——`ANIMA_PRESET` 迁入 `families/anima/preset.py`，`AnimaLycorisAdapter` 改为 preset 注入参数并更名 `LycorisAdapter`。与此前延后的「utils 完整重构方案」（lora/ 顶层包 + 4 helper + optimizer 拍扁）是**衔接关系、非前置非取代**，仅修订其中一项：preset 不再进 lora/ 包私有文件，归 families/。
- **磁盘模型目录不按族分**。维持 `models_root/{diffusion_models,vae,text_encoders,...}` 按资产类型组织：D6/D7 决定了 VAE 与 latent 缓存本来就是跨族共享资产，按族分目录会制造重复下载；唯一冲突点是 `text_encoders/` 扁平布局，新增族的 TE 一律落 `text_encoders/<safe_dir_name(repo)>/` 子目录，Anima 存量扁平布局保留识别、不迁移。
- **最大争议点**：`modeling/` 是否随本次改造把 Anima 文件下沉到 `modeling/anima/`（本文推荐做，随 exec-load 退役同刀，PR-2a）。不做的话 K2 文件与 Anima 文件在 `modeling/` 根目录混排（kohya 式扁平），机制上无害但每加一族根目录多 2-3 个文件、`find_diffusion_pipe_root` 锚点文件名也无法退役。退让方案在 §4.4 备查。

---

## 1. 范围与前提

本文回答 `00-decisions.md` §7 第一条 open question：modeling/ 多族布局、runtime 入口与姊妹脚本、utils/ 归属（含延后的 utils 重构衔接）、exec-load 与 `find_diffusion_pipe_root` 去留，外加 studio 侧 `services/models/` 的 per-family 化与磁盘目录、测试布局、迁移步骤。

约束（来自已锁定决策）：

| 决策 | 对本文的约束 |
|---|---|
| D1 ModelFamily 适配层 + registry | 必须存在一个「族适配器」的物理居所；registry 形态对齐既有 7 套 plugin registry |
| D5 K2 v1 能力集裁剪 | NaViT / SRA / LeapAlign 是 Anima-only 代码，归属必须体现这一点 |
| D6 latent 缓存跨族共享 | VAE 结构代码与磁盘 VAE 权重不能被划成 Anima 专属 |
| D7 三路径字段跨族复用 | `TrainingConfig` 的路径字段不拆 per-family 命名 |
| D8 supervisor/cmd_builder 不动、入口脚本不变 | §5 只在「不改名」前提下讨论语义表达 |
| §6 接缝草案 | ModelSpec（声明）与 ModelFamily（行为）分开，本文给它们各自的文件位置 |

---

## 2. 现状代码归属地图（2026-07-13 勘察）

按五个区域列出每个文件的多族判定。行数为当前实测。

### 2.1 `modeling/` — 结构定义层（无上游依赖，只 torch/einops）

| 文件 | 行数 | 判定 | 依据 |
|---|---|---|---|
| `modeling/anima_modeling.py` | 257 | Anima 专属 | Anima DiT / LLMAdapter 包装层 |
| `modeling/cosmos_predict2_modeling.py` | 2068 | Anima 专属 | Anima 的 Cosmos 主干；且是 attention backend 状态机的 owner（`modeling/cosmos_predict2_modeling.py:38` 自述「本模块是 anima_modeling.py 的 owner」），K2 的 MMDiT 与它无共享 |
| `modeling/wan/vae2_1.py` | 658 | **跨族共享** | D6：Anima 与 K2 同用 Qwen-Image VAE（= Wan2.1 latent 空间） |

历史：PR #303（commit `3e9160e5`，2026-06-21）把这三个文件从 `models/` git mv 到 `modeling/`，确立依赖方向 `modeling → utils → runtime → studio → tools`。`modeling/` 无 `__init__.py`，靠 namespace package + repo root 在 sys.path 上被正常 import（测试即如此：`tests/test_navit_packed_objective.py:17`）。

### 2.2 `runtime/` 入口脚本

| 文件 | 行数 | 判定 |
|---|---|---|
| `runtime/anima_train.py` | 140 | 共享编排壳。main() 只调 6 个 phase（`runtime/anima_train.py:118-136`）；顶部 re-export 段（`runtime/anima_train.py:58-111`）是 sister script 契约层 |
| `runtime/anima_daemon.py` | 1036 | 共享壳 + Anima 加载调用。经 `import anima_train as _T` 取 `find_diffusion_pipe_root` / `load_anima_model` 等（`runtime/anima_daemon.py:246,299`） |
| `runtime/anima_generate.py` | 411 | 同上（`runtime/anima_generate.py:136`） |
| `runtime/anima_reg_ai.py` | 519 | 同上（`runtime/anima_reg_ai.py:426`） |
| `runtime/train_monitor.py` | 214 | 共享（无族耦合，名字也无前缀） |

Sister script 契约（`docs/AGENTS.md` §3.2）：7 个名字 `find_diffusion_pipe_root / load_anima_model / load_vae / load_text_encoders / sample_image / enable_xformers / resolve_path_best_effort` 在 `anima_train.py` 顶层 re-export，「可加不可减不可改签名」，`tests/test_anima_generate_xy.py` 捕获破坏。

### 2.3 `runtime/training/` 子包

| 文件 | 行数 | 判定 | 关键耦合点 |
|---|---|---|---|
| `loop.py` | 784 | **共享循环** | 但模块级 import Anima 专属件：`loop.py:21-31`（leap、navit、`forward_with_optional_checkpoint`）；每步在线文本编码 `loop.py:229-236`（encode_qwen + T5）；masked loss reduction `loop.py:111-127` 与 timestep_sampler 派发 `loop.py:256-258` 是族无关的 |
| `context.py` / `phases/`（6 个） | 206 + ~920 | 共享 | 派发点 `phases/models.py:33-84`：find root → `load_anima_model` → `load_vae` → `load_text_encoders`；SRA 构造 `phases/models.py:99-117` 是 Anima 专属段 |
| `models.py` | 524 | **混居，待拆** | `load_anima_model`（形状推断只认 2048/5120，`models.py:397-427`）+ `load_text_encoders`（Qwen3-0.6B + T5，`models.py:481-524`）Anima 专属；`VAEWrapper` + `load_vae`（`models.py:34-478`，含 z_dim=16 硬编码 `models.py:284`、归一化统计 `models.py:468-475`）跨族共享（D6） |
| `model_loading.py` | 314 | **混居，待拆** | checkpoint 前缀推断 / 容错加载（`model_loading.py:170-314`）族无关；`forward_with_optional_checkpoint`（`model_loading.py:33-59`，直调 `prepare_embedded_sequence`/`blocks`/`final_layer`/`unpatchify`）Anima 专属；`find_diffusion_pipe_root`（`model_loading.py:125-154`）与 exec-load（`model_loading.py:157-163`）见 §6 |
| `text_encoding.py` | 382 | Anima 专属 | encode_qwen（Qwen3-0.6B 末层）+ T5 加权 tokenize + Comfy conditioning；K2 文本路径完全不同（Qwen3-VL-4B 12 中间层堆叠 + varlen 预缓存，D3） |
| `comfy_qwen.py` | 334 | Anima 专属 | ComfyUI parity 的 Qwen3-0.6B standalone encoder |
| `sampling.py` | 472 | Anima 专属 | Comfy KSampler parity 的 `sample_image`（er_sde/dpmpp_3m_sde、CONST flow、`sampling.py:33` 白名单）；K2 走 FlowMatchEuler 动态 shift |
| `navit.py` / `leap.py` / `sra_align.py` | 160/416/259 | Anima 专属 | D5 明文列为 Anima-only 门控 |
| `dataset.py` | 1591 | 共享 | z_dim/stride/patch 散点待 PR-1 收敛进 ModelSpec |
| `noise.py` / `loss_weighting.py` / `timestep_sampling.py` / `state.py` / `snapshot.py` / `observability.py` / `sample_runner.py` / `bootstrap.py` / `cli.py` | — | 共享 | 只消费 `(latents, noise, t, loss, mask)` 或与模型无关（§6.3 边界） |
| 7 套 plugin registry（`adapters/ optimizers/ schedulers/ inference_samplers/ timestep_samplers/ losses/` + studio 侧 eval） | — | 共享 | families/ 将成为第 8 套，且是首个架构级 registry |

### 2.4 `utils/`

| 文件 | 行数 | 判定 |
|---|---|---|
| `lycoris_adapter.py` | 542 | 算法族无关，但两处 Anima 耦合：模块级 `from utils.lokr_preset import apply as apply_anima_preset`（`utils/lycoris_adapter.py:27`）+ `inject()` 内 `apply_anima_preset(LycorisNetwork)`（`utils/lycoris_adapter.py:95`）；类名 `AnimaLycorisAdapter` 本身即族名 |
| `lokr_preset.py` | 34 | **Anima 专属**（`ANIMA_PRESET`，`utils/lokr_preset.py:14-26`：target_name 通配、排除 `llm_adapter*`、`lora_prefix="lora_unet"`） |
| `lycoris_patch.py` / `optimizer_utils.py` / `soap_optimizer.py` / `ortho_adapter.py` / `caption_utils.py` | 110/1796/708/391/301 | 族无关，本次不动 |

### 2.5 `studio/`

| 位置 | 判定 |
|---|---|
| `studio/services/models/paths.py`（443 行） | 混居：`ANIMA_REPO/ANIMA_VARIANTS/ANIMA_VAE_PATH/QWEN_*/T5_*`（`paths.py:22-52`）+ 4 个 target 函数（`paths.py:246-263`）+ `selected_anima` 解析三件套（`paths.py:355-402`）+ `default_paths_for_new_version`（`paths.py:427-443`）是 Anima 族资产；`models_root/safe_dir_name`（`paths.py:218-243`）与工具模型（WD14/CLTagger/upscaler/eval/taeflux，`paths.py:54-212,266-337`）族无关——**工具模型不是模型族，不参与 per-family 化** |
| `studio/services/models/catalog.py`（336）/ `downloader.py`（564）/ `sources.py`（370） | 消费 paths 常量，随 registry 化改遍历 |
| `studio/infrastructure/secrets.py:490-514` | `ModelsConfig.selected_anima: str = "1.0"` 单模型假设，PR-4 泛化 |
| `studio/domain/training.py:38-57` | 4 个权重路径字段（默认值指向 Anima）；D7 已定跨族复用 + `t5_tokenizer_path` 转 Anima-only show_when |
| `studio/supervisor/cmd_builder.py:49-53` | 按 task_type 选脚本，D8 锁定不动 |
| `studio/services/inference/daemon.py:48` / `studio/services/eval_samples.py:546,590` | 引用 `anima_daemon.py` 路径 / `import anima_train as _T`，随 §5 不改名而不动 |

---

## 3. 核心抉择：per-family 代码怎么切

### 3.1 候选方案

**方案 A — 按层切**：保持现有四层（modeling / runtime/training / utils / studio），每层内为族开子目录或子文件；族的「行为聚合点」是 `runtime/training/families/<fam>/`，即 D1 的 ModelFamily 适配器所在地。结构定义与资产清单按层留在原地。

**方案 B — 按族切**：顶层 `families/anima/`、`families/krea2/` 自包含包，一个族的 modeling + 加载 + 采样 + preset + 下载清单全在一个包里；共享循环 / registry 单独成 `core/`（或留 `runtime/training/`）。

**方案 C — 混合**：方案 A 的骨架 + 一条强纪律：「一个族 = 三层各一个同名单元」，族名字符串是唯一 join key，启动期校验三处对齐。——实践中 C 就是 A 的严格化，下表把 C 并入 A 评估，差异只在纪律是否落进校验代码（本文推荐落）。

### 3.2 对比

| 维度 | A/C 按层切（推荐） | B 按族切 |
|---|---|---|
| **加第 3 个模型族** | 新建 ~8 文件：`modeling/<fam>/`（1-2）、`runtime/training/families/<fam>/`（4-6）、`studio/services/models/families/<fam>.py`（1）；修改 4 处一行注册（两个 registry `__init__`、schema `model_family` Literal、能力集默认值表）。共享循环 0 修改 | 新建 1 个自包含包（文件数相近）+ 1-2 处注册。表面更整洁 |
| **共享循环功能演进**（masked loss / InfoNoise / eval / 新 loss） | 单点：`loop.py` / `losses/` / `timestep_samplers/` / studio eval 各改一处，所有族自动获得（只要功能停留在 §6.3 边界内）。族目录零触碰 | 取决于纪律：族包若只含适配器则同左；但自包含包天然诱导把 step 逻辑吸进族包，一旦发生就是 N 份漂移——这正是 D1 否决「按模型分训练脚本」时点名的维护 ×2 风险的回潮通道 |
| **import 与打包机制** | 维持现状：无 pip 打包、入口注入 sys.path（`runtime/anima_train.py:34-38`、`tests/conftest.py:19-20`）、`modeling → utils → runtime → studio` 单向依赖不变。studio 侧族文件保持无 torch（`paths.py` 现在就是纯路径计算，被 server 启动路径同步调用） | 族包同时被「torch 训练子进程」与「无 torch 的 FastAPI server」import。要么包内全部 lazy import（脆弱、易被一行顶层 import 打破），要么包内再分 `runtime 半区 / studio 半区`——那就是按层切套了一层壳，还把单向依赖变成了双向穿包 |
| **与 exec-load 机制的兼容 / 演化** | `modeling/` 仍是唯一模型结构代码根；exec-load 退役后 `from modeling.<fam> import ...` 一步到位，`find_diffusion_pipe_root` 收缩成 shim（§6） | 结构代码进族包后「模型代码根目录」概念消失，find root / DIFFUSION_PIPE_ROOT 外部 checkout 兼容必须同刀连根拔，无渐进路径 |
| **测试布局** | flat `tests/` 不动（CI 全量门禁按 `tests/` 路径收集，见 §9），新文件按命名约定归族 | 要么测试进族包（破坏 CI 收集约定与 conftest sys.path 注入）要么仍 flat（「自包含」名不副实） |
| **与 D1-D9 / 既有工程文化契合** | D8 派发点 `phases/models.py` 查 registry 直接落地；registry 形态与 7 套 plugin registry 同款（`runtime/training/README.md:59-88` 的「加变体 = 加文件 + 字典一行」文化直接复用） | 也能实现 D1-D8，但 PR-4 的 studio 泛化要跨进族包，依赖方向需要重新立法 + AGENTS.md §3.1 改写 |
| **风险** | 一个族散在三处，靠命名 + 校验维系整体感；新人找「K2 的全部代码」要看三个目录 | 族内混层导致的 import 事故是运行时才炸（server 启动拉起 torch、显存被 server 进程占用等） |

### 3.3 推荐与理由

**推荐方案 A（含 C 的纪律强化）**。决定性理由按权重排序：

1. **运行时形态**：本项目一个「族」横跨两种进程——训练子进程（torch、独占 GPU）与 studio server（常驻、无 torch、Windows 下要秒启动）。按族切的自包含包在这个前提下不可能真自包含，B 的核心卖点不成立。
2. **共享循环是本项目的主要资产**（masked loss、InfoNoise、7 套 registry、eval 体系、暂停恢复、观测），D1 的选型逻辑就是保它单点演进。按层切让「共享的留在层里、族有的进族目录」有一条物理防线：任何人往 `families/<fam>/` 提交 step 逻辑在 review 里一眼可见。
3. **迁移是渐进的**：方案 A 的每一步都是 `git mv` + import 修正，Phase 1「对 Anima 零行为变化」可逐 PR 验证；方案 B 需要一次性重立依赖方向。

「新人找 K2 全部代码要看三处」的缓解：三处目录名严格同名（`anima` / `krea2`），并在 `runtime/training/families/README.md` 放一张「一个族的三个居所」导览表（迁移 PR-2b 附带）。

---

## 4. 目标目录树（推荐方案）

标注：**[不动]** 原地保留；**[git mv]** 整文件移动保历史；**[抽出]** 函数级搬移（源文件保留并瘦身）；**[新建]**；**[Phase 3]** K2 阶段才出现。

### 4.1 全树

```
modeling/                                    # 层 1：结构定义（只依赖 torch/einops）
├── __init__.py                              # [新建] 空占位（namespace → regular package）
├── anima/
│   ├── __init__.py                          # [新建] re-export Anima / set_attention_backend
│   ├── anima_modeling.py                    # [git mv] ← modeling/anima_modeling.py
│   └── cosmos_predict2_modeling.py          # [git mv] ← modeling/cosmos_predict2_modeling.py
├── krea2/                                   # [Phase 3]
│   ├── __init__.py
│   └── krea2_modeling.py                    # diffusers Krea2Transformer2DModel + musubi 双参考移植
└── wan/
    ├── __init__.py                          # [新建]
    └── vae2_1.py                            # [不动] 跨族共享 VAE（D6），不进任何族目录

runtime/
├── anima_train.py                           # [不动] D8；顶层 re-export 兼容层继续（§5）
├── anima_generate.py / anima_daemon.py
│   / anima_reg_ai.py / train_monitor.py     # [不动]
└── training/
    ├── loop.py / context.py / dataset.py / noise.py / loss_weighting.py
    │   / timestep_sampling.py / state.py / snapshot.py / observability.py
    │   / sample_runner.py / bootstrap.py / cli.py / phases/                 # [不动] 共享循环
    │                                        #（loop.py / phases/models.py 内部改为经 ctx.family 派发，见 §4.3）
    ├── models.py                            # [瘦身] 只留 VAEWrapper + load_vae（跨族共享）
    ├── model_loading.py                     # [瘦身] 只留前缀推断/容错加载/resolve_path/enable_xformers
    │                                        #  + find_diffusion_pipe_root shim（§6）
    ├── text_cache.py                        # [Phase 2 新建] varlen 文本缓存共享基建（格式族无关，编码器由 family 提供）
    ├── adapters/ optimizers/ schedulers/ inference_samplers/
    │   timestep_samplers/ losses/           # [不动] 7 套 plugin registry
    │                                        #（inference_samplers/ Phase 3 加 flow_euler.py：solver 实现共享，
    │                                        #  族只在 ModelSpec overlay 里选默认值与白名单）
    └── families/                            # [新建] 第 8 套 registry（首个架构级）
        ├── __init__.py                      # FAMILIES dict + get_family() + validate_schema_consistency()
        ├── protocol.py                      # ModelFamily Protocol（方法集由 03 冻结）
        ├── spec.py                          # ModelSpec frozen dataclass（§6.1 声明式常量）
        ├── README.md                        # 「一个族的三个居所」导览 + 加族步骤
        ├── anima/
        │   ├── __init__.py                  # AnimaFamily 组装 + ANIMA_SPEC
        │   ├── loader.py                    # [抽出] ← models.py 的 load_anima_model / load_text_encoders
        │   ├── forward.py                   # [抽出] ← model_loading.py 的 forward_with_optional_checkpoint
        │   ├── text_encoding.py             # [git mv] ← training/text_encoding.py（encode_qwen + T5 加权，见 §4.2）
        │   ├── comfy_qwen.py                # [git mv] ← training/comfy_qwen.py
        │   ├── sampling.py                  # [git mv] ← training/sampling.py（Comfy parity sample_image）
        │   ├── navit.py                     # [git mv] ← training/navit.py（D5 Anima-only）
        │   ├── leap.py                      # [git mv] ← training/leap.py（D5）
        │   ├── sra_align.py                 # [git mv] ← training/sra_align.py（D5）
        │   └── preset.py                    # [抽出] ← utils/lokr_preset.py 的 ANIMA_PRESET（§7）
        └── krea2/                           # [Phase 3]
            ├── __init__.py                  # Krea2Family + KREA2_SPEC（分辨率感知动态 shift 参数在 spec overlay）
            ├── loader.py                    # Krea-2-Raw 加载 + fp8/block-swap 兜底挂点（D2 搁置，仅留位置）
            ├── text_encoding.py             # Qwen3-VL-4B 12 层堆叠提取（D3 varlen 缓存的编码端）
            ├── sampling.py                  # FlowMatchEuler 动态 shift 采样组装
            └── preset.py                    # KREA2_PRESET（musubi 参考：264 个 Linear 全 target）

utils/
├── lycoris_adapter.py                       # [改] preset 改注入参数；AnimaLycorisAdapter → LycorisAdapter + 兼容别名（§7）
├── lokr_preset.py                           # [删] 内容迁 families/anima/preset.py；保留 1 个 release 的 re-export shim（§7.3）
└── lycoris_patch.py / optimizer_utils.py
    / soap_optimizer.py / ortho_adapter.py / caption_utils.py               # [不动]

studio/
├── domain/training.py                       # [改·PR-3] + model_family 字段 + 能力集门控；位置不动
├── infrastructure/secrets.py                # [改·PR-4] ModelsConfig.selected_anima → per-family map + 迁移
├── supervisor/cmd_builder.py                # [不动] D8
└── services/models/
    ├── paths.py                             # [瘦身] 留 models_root / safe_dir_name / 工具模型（wd14、cltagger、
    │                                        #  upscaler、eval、taeflux——它们不是模型族）
    ├── families/                            # [新建]
    │   ├── __init__.py                      # FAMILY_ASSETS registry（key 与 runtime 侧一致）
    │   ├── anima.py                         # [抽出] ← paths.py 的 ANIMA_*/QWEN_*/T5_* 常量 + 4 个 target 函数
    │   │                                    #  + selected 解析 + default_paths_for_new_version
    │   └── krea2.py                         # [Phase 3] Krea-2-Raw / Turbo / Qwen3-VL-4B 清单
    ├── catalog.py / downloader.py / sources.py   # [改] 遍历 FAMILY_ASSETS 出各族 section

tests/                                       # [不动] flat 布局维持（§9）
```

### 4.2 任务点名的 Anima 专属代码归属

| 代码 | 现位置 | 归属 | 说明 |
|---|---|---|---|
| `navit.py` | `runtime/training/navit.py` | `families/anima/navit.py` | D5 Anima-only；`loop.py:31` 的模块级 import 改为能力门控内 lazy import |
| `leap.py` | `runtime/training/leap.py` | `families/anima/leap.py` | 同上（`loop.py:21-24`） |
| `sra_align.py` | `runtime/training/sra_align.py` | `families/anima/sra_align.py` | 构造点 `phases/models.py:99-117` 一并挪进 AnimaFamily 的加载后 hook |
| `comfy_qwen.py` | `runtime/training/comfy_qwen.py` | `families/anima/comfy_qwen.py` | 只被 `load_text_encoders`（`models.py:496-498`）引用，跟着 loader 走 |
| `text_encoding.py` 的 T5 部分 | `runtime/training/text_encoding.py` | **整文件**进 `families/anima/text_encoding.py` | T5 加权 tokenize（喂 LLMAdapter 的 T5 IDs）是 Anima 独有；encode_qwen（Qwen3-0.6B 末层）同样不被 K2 复用（K2 取 Qwen3-VL 12 中间层，提取方式不同）。其中 `_parse_weighted_tag` 的 `(tag:1.2)` 权重语法解析理论上通用，但 D3 已定 K2 关闭 tag 生态门控，不预先上提（YAGNI）；未来第三个 tag 系模型出现时再抽到共享层 |
| `lokr_preset.py` 的 `ANIMA_PRESET` | `utils/lokr_preset.py:14-26` | `families/anima/preset.py` | 见 §7 |
| `forward_with_optional_checkpoint` | `model_loading.py:33-59` | `families/anima/forward.py` | 直调 Anima 内部 API（00-decisions §3 点名）；梯度检查点是 family 内部职责（§6.2） |
| `load_anima_model` / `load_text_encoders` | `models.py:339-447,481-524` | `families/anima/loader.py` | 形状推断 2048/5120 两档、llm_adapter 缺失兜底都是族知识 |
| `sample_image` 及 Comfy parity 管线 | `sampling.py` | `families/anima/sampling.py` | er_sde/dpmpp_3m_sde 的 **solver 实现**留在共享 `inference_samplers/`；`sampling.py` 里的编排（sigma 调度 shift=3.0、CONST flow、conditioning 组装）是族行为 |

**留在共享层的边界样本**（防止「Anima 用的都算 Anima 的」扩大化）：`VAEWrapper` 与 `load_vae`（D6 跨族）、`make_noise`、`compute_loss_weight`、`sample_t`、masked loss reduction（`loop.py:111-127`）、7 套 plugin registry 全部、`dataset.py`（z_dim 等经 ModelSpec 参数化后完全族无关）。

### 4.3 与 03（接口视角）的边界

本文只声明：ModelFamily 的**物理形状**是「`families/<fam>/__init__.py` 组装 + 若干实现模块」，registry 是「`FAMILIES: dict[str, ModelFamily]` + `get_family()` + schema 一致性校验」。`forward_train / encode_text / build_sampler / lora_preset` 等方法的签名、能力集的表达形式（字符串集合 vs 字段）、`loop.py` 里 navit/leap 分支改「family 能力门控 + lazy import」还是「family 提供 step 策略对象」——归 `03-interface-evolution.md` 冻结。无论 03 选哪种，文件归属不变。

### 4.4 退让方案（若不动 `modeling/`）

保持 `modeling/` 根目录扁平，K2 以 `modeling/krea2_modeling.py`（或 `modeling/krea2/`）加入，Anima 文件不动。省去 PR-2a 的 3 个文件移动与 ~6 处 import 修正；代价：目录不对称、exec-load 的锚点文件名 `anima_modeling.py`（`model_loading.py:150`）永久保留、attention backend 别名传播列表无法收缩。机制上无害，纯观感与长期维护取舍。**本文推荐移动**，理由：反正 exec-load 退役（§6）要动同一批 import 与传播列表，边际成本接近零，不动等于把一次性成本换成永久不对称。

---

## 5. 入口脚本与命名

### 5.1 结论：不改名（D8 已锁定），语义靠口径不靠文件名

D8 明文「supervisor/cmd_builder 不动，入口脚本不变」。本节只回答不改名之下泛化语义怎么表达：

1. **口径统一**：`anima_` 前缀解释为**产品名**（Anima LoRA Studio）的历史入口名，非模型族限定。落点：`runtime/anima_train.py` 模块 docstring 第一行加一句「入口名中的 anima 指产品线，模型族由 `--model-family` / yaml `model_family` 决定」；`docs/architecture/project-structure.md` 与 `runtime/training/README.md` 同步一句话。
2. **派发不变**：`studio/supervisor/cmd_builder.py:49-53` 继续按 task_type 选脚本；族派发发生在脚本内部 `phases/models.py` 查 registry（D8），对 supervisor 完全透明——K2 训练任务的命令行和 Anima 一模一样，只是 yaml 里 `model_family: krea2`。
3. **`train_monitor.py` 先例**：runtime 下本就有无前缀脚本，说明前缀从来不是硬约定。

### 5.2 改名成本清单（备查，不建议 Phase 1-4 执行）

若未来（K2 稳定后）仍想改名 `anima_train.py → train.py` 等，触碰面实测如下：

- `studio/supervisor/cmd_builder.py:49-53`（3 处路径）
- `studio/services/inference/daemon.py:48`（`_DAEMON_SCRIPT`）
- `studio/services/eval_samples.py:546`（`import anima_train as _T`）
- sister script 三处 `import anima_train as _T`（`anima_daemon.py` / `anima_generate.py` / `anima_reg_ai.py`）+ `tools/spike/vae_stress.py:109`
- tests：`test_anima_train_migration.py` / `test_anima_generate_xy.py` / `test_anima_reg_caption.py` / `test_anima_daemon_comfy_parity_runtime.py` 等 ≥4 文件
- 文档：`docs/AGENTS.md` §3.2、`runtime/training/README.md:255-257`、ADR 0003、user-guide 若干
- 用户侧：裸 CLI 用户的命令行习惯 + 各处教程截图——这是唯一不可 grep 的成本，需要 release note + 旧名 shim（`anima_train.py` 只剩 `from train import *; main()`）保一个大版本

结论维持：改名收益纯观感，成本里含用户习惯这种长尾项，D8 的锁定是对的。

---

## 6. exec-load 机制去留

### 6.1 考证：它为什么存在

- **初始 commit（`c6bd6b64`）即有**。项目脱胎于「训练脚本 + 外部 diffusion-pipe checkout」工作流：模型结构代码不在本仓库，靠 `DIFFUSION_PIPE_ROOT` 环境变量或约定目录（`runtime/models/`、`repo/models/` 等）定位，再用 `importlib.util.spec_from_file_location` 按文件路径 exec 加载（`model_loading.py:157-163`）。`find_diffusion_pipe_root` 的候选链与函数名本身就是这段历史的化石（`model_loading.py:128-133` 注释列举「CLI 直接 cd 进 scripts/ 跑」「外部 diffusion-pipe checkout 把代码放 models/」两个场景）。
- **PR #303（`3e9160e5`）** 把模型代码收进仓库 `modeling/` 后，**只是把新位置加到候选最前**，全部旧候选与 exec-load 原样保留，commit message 明言「保留 models/ + DIFFUSION_PIPE_ROOT 作为 fallback（兼容外部 diffusion-pipe checkout）」。

### 6.2 今天的真实状态与代价

- 模型代码 100% 随仓库发布；tests 已在正常 import（`tests/test_models_flash_attn.py:180`、`tests/test_navit_packed_objective.py:17`、`tests/test_packed_navit_forward.py:22` 均 `from modeling.anima_modeling import ...`）；所有入口都已把 repo root 注入 sys.path（`runtime/anima_train.py:34-38`、`tests/conftest.py:19-20`）。exec-load 与正常 import **并存**。
- 并存的直接病灶：**双模块身份**。exec-load 产出裸名 `anima_modeling` 模块对象，与 import 系统里的 `modeling.anima_modeling` 是两个独立对象，模块级全局状态（attention backend 开关）互不可见。于是 `enable_xformers` 要向 6 个候选模块名广播（`model_loading.py:81-88`），`load_anima_model` 再广播 4 个（`models.py:365-368`）。每加一个模型族，这个别名矩阵翻倍，漏一条就是静默性能退化（fast path 没开、无报错）。
- `load_vae` 同样 exec-load `wan/vae2_1.py`（`models.py:452`），同病。

### 6.3 结论：改正常 import，exec-load 与外部 checkout 兼容退役

多族下**不保留 per-family exec**——K2 从第一天起 `from modeling.krea2 import ...`，Anima 在 PR-2a 切换：

1. `families/anima/loader.py` 用 `from modeling.anima.anima_modeling import Anima`（正常 import，单一模块身份）；`load_vae` 用 `from modeling.wan.vae2_1 import WanVAE_`。attention backend 传播列表塌缩为真名一条。
2. **`find_diffusion_pipe_root` 名字保留**（sister 契约 7 名之一，「可加不可减不可改签名」，`docs/AGENTS.md` §3.2），实现塌缩为返回 `modeling/anima/` 目录的常量 shim + docstring 标注 deprecated。`load_anima_model(transformer_path, device, dtype, repo_root, ...)` 等契约签名不动，`repo_root` 参数改为忽略（shim 层注释说明）。
3. **`DIFFUSION_PIPE_ROOT` 与外部 checkout 候选链删除**，环境变量保留一个 release：检测到设置时打 warning「外部模型代码目录已不支持，忽略该变量」，下个 release 删检测。（是否需要这一个 release 的缓冲，见 §11 待拍板。）
4. `load_module_from_path` 函数删除；`tests/test_find_diffusion_pipe_root.py` 收缩为 shim 行为断言（或整文件删除，其 ast 抽取式测试手法随机制退役失去意义）。

---

## 7. utils/ 归属：算法与 preset 分离

### 7.1 切法

原则：**算法族无关，target 选择（preset）族相关**。LyCORIS 的 lokr/loha/lora 数学对任何 Linear 堆都成立；「打哪些层、排除什么、保存键名前缀」才是族知识。

1. `ANIMA_PRESET` → `families/anima/preset.py`。K2 的 `KREA2_PRESET`（Phase 3）对照 musubi 默认（全部 264 个 Linear、dim32/alpha32 起点）放 `families/krea2/preset.py`。**注意 `lora_prefix` / 保存键名是 Comfy 生态契约的一部分**，K2 的键名约定是 00-decisions §7 的 open question（等 musubi 产物 vs ComfyUI 加载核对），preset 文件是该结论未来的落点。
2. `utils/lycoris_adapter.py`：删掉 `utils/lycoris_adapter.py:27` 的模块级 preset import；`__init__` 增加 `preset: dict` 必填参数，`inject()`（`utils/lycoris_adapter.py:95`）改 `LycorisNetwork.apply_preset(self._preset)`。类名 `AnimaLycorisAdapter → LycorisAdapter`，旧名保留别名（`studio/services/inference` 侧引用与 tests 逐步切换）。
3. 接线：`runtime/training/adapters/lycoris.py` 的 `build(args)` 从 family registry 取 `get_family(args.model_family).lora_preset()` 传入——adapter plugin registry（管「什么算法」）与 family registry（管「打哪些层」）正交，互不吞并。

### 7.2 与延后的「utils 完整重构方案」的关系：衔接，一处修订

此前（2026-05-14）三方 review 形成、用户决定延后的方案要点：顶层 `lora/` 包（protocol + 4 helper + lycoris 算法 + 私有 preset/patch）、optimizer_utils 拍扁进 `runtime/training/optimizers/`、caption_utils 进 studio。与本方案的关系：

- **不是前置**：多模型 Phase 1 只需要 §7.1 的三步外科手术，不触发该方案的执行条件；不要为多模型顺手做 lora/ 包（scope creep）。
- **不冲突，方向一致**：该方案的核心（把混杂 concern 从 lycoris_adapter 拆出）与本次「把族知识从算法层拿出」同向。
- **一处修订**：该方案原目标树把 preset 放 `lora/_lokr_preset.py`（包内私有）。多族之后 preset 是 per-family 资产，**归 `families/<fam>/preset.py`，不进 lora/ 包**——将来执行 utils 重构时以本文档为准；lora/ 包只收算法 + protocol + helpers。其余部分（helpers 四件套、optimizer 拍扁、caption_utils 迁移）与多模型改造正交，何时执行仍按原触发条件。

### 7.3 兼容期

`utils/lokr_preset.py` 保留 1 个 release 的 re-export shim（`from runtime 侧新位置 import ANIMA_PRESET` 不可行——utils 不能反向依赖 runtime，见 AGENTS.md §3.1；shim 直接内联同字面 dict + DeprecationWarning，或干脆一步删除并 grep 确认仓库内零残留引用后不留 shim）。倾向**直接删**：`lokr_preset` 只有 `lycoris_adapter.py:27` 一个仓库内 caller，无外部 pin 面。

---

## 8. studio/services/models 的 per-family 化

### 8.1 代码：常量按族收编，工具模型不动

- `studio/services/models/families/anima.py` [抽出]：`ANIMA_REPO / ANIMA_VARIANTS / LATEST_ANIMA / ANIMA_VAE_PATH`（`paths.py:22-33`）、`QWEN_REPO / QWEN_FILES`（`paths.py:35-45`）、`T5_REPO / T5_FILES`（`paths.py:47-52`）、4 个 target 函数（`paths.py:246-263`）、`find_anima_main / selected_anima_variant / selected_anima_transformer_path / anima_transformer_path_for`（`paths.py:340-402`）、`default_paths_for_new_version`（`paths.py:427-443`，签名加 `family: str = "anima"`）。
- `families/__init__.py` 定义 `FAMILY_ASSETS: dict[str, FamilyAssets]`：每族声明「repo 清单（含 size 预估）、target 路径函数、selected 解析、新建 version 默认路径」。`catalog.py` / `downloader.py` 从点名 import 常量（`catalog.py:14-44` 现状）改为遍历 registry 出 per-family section；`sources.py`（HF/MS 镜像路由）族无关不动。
- `paths.py` 瘦身后保留：`models_root / safe_dir_name` + WD14 / CLTagger / upscaler / eval / taeflux / CCIP 全部工具模型段落。**判定标准**：出现在 `TrainingConfig` 权重路径字段里的是「模型族资产」，其余（打标 / 放大 / 评估 / 预览解码）是「工具模型」，永远不进 families/。
- `secrets.ModelsConfig`（`studio/infrastructure/secrets.py:514`）：`selected_anima: str` → `selected: dict[str, str]`（family → variant key 或 custom 路径），pydantic before-validator 读老键迁移（老配置 `selected_anima: "1.0"` → `selected: {"anima": "1.0"}`），与既有字段重命名迁移模式一致。API 层 `studio/api/routers/models.py:106-107` 等引用点同步。

### 8.2 磁盘：不按族分目录，按资产类型 + repo 子目录

现状 `models_root/` 固定子目录：`diffusion_models/`（扁平文件）、`vae/`、`text_encoders/`（Qwen3-0.6B 文件**直接平铺**）、`t5_tokenizer/`、加各工具模型目录。逐一裁决：

| 目录 | 决定 | 理由 |
|---|---|---|
| `diffusion_models/` | 保持扁平，K2 权重直接落 `krea2-raw-*.safetensors` / turbo 同 | 上游文件名本身携带族名且全局唯一（ANIMA_VARIANTS 的文件名模式即先例）；按族分目录得不到任何消歧收益 |
| `vae/` | 保持，K2 的 `vae_path` 指向**同一个** `qwen_image_vae.safetensors` | D7 明文：无需新下载。按族分目录 = 强制复制或 symlink（Windows 上 symlink 有权限坑），纯负收益 |
| `text_encoders/` | **唯一需要改的点**：新增族的 TE 落 `text_encoders/<safe_dir_name(repo)>/`（如 `text_encoders/Qwen_Qwen3-VL-4B-Instruct/`）；Anima 的 Qwen3-0.6B 存量扁平布局保留识别（探测顺序：族清单声明的子目录 → 扁平 legacy），不做迁移 | 现状 `qwen_dir()` 返回 `root/"text_encoders"` 且文件平铺（`paths.py:258-259` + `downloader.py:107-109`），第二个 TE 直接落进来会文件名冲突（都有 `config.json` / `model.safetensors`）。零存量迁移遵循「老 schema 读兼容」偏好 |
| `t5_tokenizer/` | 保持；D7 已把 `t5_tokenizer_path` 定为 Anima-only show_when 字段 | K2 无此概念 |
| 工具模型目录 | 全部不动 | 与族无关 |

不按族分的总依据：D6（latent 缓存跨族共享）与 D7（VAE 文件共享）已经决定了磁盘资产的自然键是「内容/repo」而不是「族」；族与资产是多对多（VAE 一对二），目录树表达不了多对多，registry 里的清单才是权威映射。

---

## 9. 测试布局

- **flat `tests/` 维持**。CI 全量门禁按 `tests/` 路径收集（本机跑全量也必须带 `tests/` 路径避开 `tmp/`），`tests/conftest.py` 的 sys.path 注入是所有测试的共同地基，不随族拆包。
- 命名约定（新文件生效，存量不重命名）：`test_families_<fam>_*.py`（族内单元：preset 内容、spec 常量、loader 的纯逻辑部分）+ `test_model_families.py`（registry 三件套：FAMILIES ↔ schema Literal 一致性、get_family 派发、能力集门控与 show_when 裁剪联动）。
- CI 无 GPU 的现实约束下，per-family 可测面 = spec / preset / registry / 派发 / schema 一致性；真加载与前向靠用户真卡验证（ADR 0003 验收策略 R2 先例）。`test_plugin_registry.py` 的防回归断言模式（「`phases/models.py` 不该再出现 `load_anima_model(` 直调字面量」）复制一条给 families registry。
- 迁移期专项：`tests/test_anima_generate_xy.py`（sister `_T.X` 契约）与 `tests/test_anima_train_migration.py` 在每个迁移 PR 都必须绿——它们就是「对 Anima 零行为变化」的机器裁判。`tests/test_find_diffusion_pipe_root.py` 随 §6.3 收缩或删除。

---

## 10. 迁移步骤（与 00-decisions Phase 1 PR-1..PR-4 对齐）

所有整文件移动用 `git mv`（PR #303 先例）；函数级拆分遵循「大块留原文件保历史，小块移出」。PR-2 因同时含机械移动与行为适配，拆 2a/2b 两刀（细化不改变 00-decisions 的四刀总量语义）。

### PR-1 — ModelSpec + latent 指纹（无移动，纯新建 + 收敛）

| 动作 | 文件 |
|---|---|
| 新建 | `runtime/training/families/__init__.py`（暂只暴露 SPECS）、`families/spec.py`（ModelSpec frozen dataclass：z_dim/stride/patch/归一化统计/latent 指纹 `wan21-f8c16`/文本规格/默认值 overlay/能力集）、`families/anima/__init__.py`（ANIMA_SPEC 实例） |
| 修改 | `runtime/training/dataset.py`（z_dim/stride/patch ≥8 处散点 → spec；`_is_cache_valid` 加模型指纹 + layout 版本）、`runtime/training/models.py:284`（z_dim 硬编码）、`runtime/training/models.py:468-475`（归一化统计移入 spec、load_vae 读 spec） |
| 测试 | 新建 `tests/test_model_spec.py` |

### PR-2a — modeling 归位 + exec-load 退役（机械刀）

| 动作 | 文件 |
|---|---|
| git mv | `modeling/anima_modeling.py → modeling/anima/anima_modeling.py`、`modeling/cosmos_predict2_modeling.py → modeling/anima/cosmos_predict2_modeling.py` |
| 新建 | `modeling/__init__.py`、`modeling/anima/__init__.py`、`modeling/wan/__init__.py` |
| 修改 | `runtime/training/models.py`（exec-load → 正常 import；传播列表塌缩）、`runtime/training/model_loading.py`（find_diffusion_pipe_root → shim、删 load_module_from_path、enable_xformers 别名表塌缩、DIFFUSION_PIPE_ROOT 弃用警告）、`modeling/anima/anima_modeling.py` 内部相对引用、3 个直接 import 的测试（`test_models_flash_attn.py:180` 等）、`THIRD_PARTY_NOTICES.md:35`、`docs/architecture/project-structure.md`（中英） |
| 验证 | 用户真卡跑一次训练 + 测试出图（PR #303 同款要求：真实加载无自动化覆盖） |

### PR-2b — AnimaFamily 行为适配器（对 Anima 零行为变化）

| 动作 | 文件 |
|---|---|
| 新建 | `families/protocol.py`（接 03 冻结面）、`families/anima/` 组装代码、`families/README.md` |
| git mv | `runtime/training/{text_encoding,comfy_qwen,sampling,navit,leap,sra_align}.py → runtime/training/families/anima/` |
| 抽出 | `models.py` 的 `load_anima_model`/`load_text_encoders` → `families/anima/loader.py`（models.py 留 VAEWrapper + load_vae）；`model_loading.py` 的 `forward_with_optional_checkpoint` → `families/anima/forward.py`；`utils/lokr_preset.py` 的 ANIMA_PRESET → `families/anima/preset.py`（utils 侧删文件，§7.3） |
| 修改 | `phases/models.py`（查 registry 派发，D8）、`loop.py`（`loop.py:21-31` 模块级 import 改经 family / 能力门控 lazy import；`loop.py:229-236` 文本编码经 family）、`sample_runner.py`、`utils/lycoris_adapter.py`（preset 注入 + 更名）、`adapters/lycoris.py`（build 接 preset）、`anima_train.py` re-export 段改 import 来源（**对外 7 名不变**）、`phases/bootstrap.py` 挂 families 的 validate_schema_consistency |
| 测试 | `test_anima_generate_xy.py` / `test_anima_train_migration.py` 全绿为硬门禁；新增 `test_model_families.py` |

### PR-3 — schema `model_family` + 能力集门控（无移动）

`studio/domain/training.py` 加 `model_family: Literal["anima"] = "anima"`（Phase 3 扩 `"krea2"`；默认值保证老配置零迁移，D7）+ Anima-only 字段批量加 `show_when="model_family=='anima'"`（复用落盘裁剪机制）；`argparse_bridge` 自动获得 CLI 字段；families registry 的 schema 一致性校验开始生效。

### PR-4 — studio 侧 per-family 化

| 动作 | 文件 |
|---|---|
| 新建 | `studio/services/models/families/{__init__,anima}.py` |
| 抽出 | `paths.py` 的 Anima 段落（§8.1 清单）→ `families/anima.py`，paths.py 瘦身 |
| 修改 | `catalog.py` / `downloader.py`（遍历 FAMILY_ASSETS）、`secrets.py` ModelsConfig 迁移、`studio/api/routers/models.py` 引用点、前端 Settings 模型卡（读 catalog 新结构） |

### Phase 2 / Phase 3 增量（本文范围内的文件动作）

- Phase 2：新建 `runtime/training/text_cache.py`（varlen 格式族无关）+ `phases/` 加缓存 phase；存放位置（sidecar vs 集中目录）是 00-decisions §7 open question，不在本文裁决。
- Phase 3：新建 `modeling/krea2/`、`families/krea2/`（5 文件）、`studio/services/models/families/krea2.py`、`inference_samplers/flow_euler.py`；修改 4 处一行注册。**这份清单就是 §3.2「加第 3 个模型族」维度的实测口径**——K2 自己先走一遍。

---

## 11. 待拍板（超出本文决定权）

1. `DIFFUSION_PIPE_ROOT` 退役节奏：保留一个 release 的弃用警告（本文倾向）vs PR-2a 直接删。
2. `runtime/training/models.py` 瘦身后是否顺手 `git mv` 为 `vae.py`（名实相符 vs 又多一处 import 修正；本文倾向改名，成本仅仓库内 grep 可尽的引用）。
3. `modeling/` 下沉 `modeling/anima/`（§4.4 退让方案 vs 推荐方案）——本文推荐做，若用户选退让方案，PR-2a 缩为纯 exec-load 退役、目标树相应保持扁平。
4. K2 LoRA 保存键名（`families/krea2/preset.py` 的 `lora_prefix`）等 Comfy 侧核对结论——00-decisions §7 已列，Phase 3 前必须闭环。
