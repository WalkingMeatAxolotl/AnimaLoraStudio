# 多模型支持：决策记录（Krea 2 为首个第二模型族）

- 状态：已确认决策的记录。架构细化（目录结构 / 代码归属 / 接口冻结面）由同目录三份视角文档深化，最终在 `04-synthesis.md` 收敛。
- 日期：2026-07-13
- 相关：`01-code-layout.md`（代码归属视角）、`02-ecosystem-survey.md`（训练器生态对标）、`03-interface-evolution.md`（接口契约与演化压力测试）

## 1. 背景与目标

训练器目前只支持 Anima 一个模型族，全链路对其硬编码。本次引入 Krea 2（下称 K2）作为第二个模型族，并借此机会建立能支撑未来更多模型族的架构。本文只记录已确认决策；未确认事项全部收在 §7 Open Questions。

## 2. Krea 2 档案（训练视角）

| 组件 | 内容 |
|---|---|
| 主干 | 12.9B 单流 MMDiT：28 blocks、width 6144、GQA（48Q/12KV）+ gated sigmoid attention、QKNorm、zero-center RMSNorm、SwiGLU 4x、3D axial RoPE |
| 训练目标 | rectified flow / v-parameterization（与 Anima 同族） |
| Timestep shift | 分辨率感知动态 shift（base_shift 0.5 → max_shift 1.15，按 image_seq_len 插值）；1024px 约等效 discrete shift 2.5（musubi 推荐值） |
| 文本条件 | Qwen3-VL-4B-Instruct，取 12 个中间层堆叠 `(B, 512, 12, 2560)`，融合层在 DiT 权重内；max_seq_len 512；自然语言 prompt，非 tag 生态 |
| VAE | Qwen-Image VAE（f8、16ch），**与 Anima 完全同款、归一化统计一致**；patch=2 → align 16 也与 Anima 一致 |
| 权重 | `krea/Krea-2-Raw`（训练用）+ Turbo（TDM 蒸馏 8 步，推理/验证用）；guidance 约定 `cond + g·(cond−uncond)` |
| 许可 | 社区许可：年营收 <$1M 且 <50 席位免费商用，明确允许 LoRA 训练；部署方有内容安全义务 |
| 参考实现 | diffusers v0.39 `Krea2Pipeline`/`Krea2Transformer2DModel`；kohya musubi-tuner 实验性支持（fp8_scaled + blocks_to_swap≤26、默认 target 全部 264 个 Linear、dim32/alpha32、`krea2_shift` 分辨率感知 timestep 采样） |

## 3. 现状关键事实（2026-07-13 勘察）

- 无任何模型族抽象层；`TrainingConfig` 无 arch 字段，仅 4 个权重路径字段（默认指向 Anima）。
- 模型结构靠 `load_anima_model` 从 checkpoint 张量形状推断（`runtime/training/models.py:397-427`，只认 2048/5120 两档）。
- `forward_with_optional_checkpoint`（`runtime/training/model_loading.py:33-59`）与 `navit.py` 直接调用 DiT 内部 API（`prepare_embedded_sequence`/`blocks`/`final_layer`/`unpatchify`）。
- LoRA target 写死在 `utils/lokr_preset.py:ANIMA_PRESET`。
- `sample_image`（`runtime/training/sampling.py`）为 Comfy KSampler parity：er_sde/dpmpp_3m_sde、shift=3.0、CONST flow。
- 文本条件每步在线编码（Qwen3-0.6B 末层 + T5 IDs 进 LLMAdapter），**无文本缓存**。
- 隐患：latent npz 缓存无模型/VAE 指纹（`dataset.py:_is_cache_valid`，换 VAE 同桶尺寸会静默复用错 latent）；`z_dim=16 / stride 8 / patch 2` 散落 ≥8 处无单一来源；前端 `anima_main`/`selected_anima` 固化单模型假设。
- 已有 7 套 plugin registry（optimizer/scheduler/loss/timestep_sampler/inference_sampler/adapter/eval），均非架构级——但证明 registry 文化已就位。

## 4. 已确认决策

| # | 决策 | 备注 |
|---|---|---|
| D1 | 采用 **ModelFamily 适配层 + registry**（路线 A） | 否决「按模型分训练脚本」（loop 功能维护 ×2）与「外挂 musubi 当后端」（产品一致性）；musubi 仅作 Phase 0 验证工具 |
| D2 | **显存基建（fp8 / block swap）搁置**；K2 支持下限 = 32GB（bf16） | 24GB 数学上不可行（bf16 权重 ~25.8GB）；未来要下探 24GB 的唯一解锁是 fp8_scaled |
| D3 | K2 文本条件走 **varlen 预缓存**（只存非 padding token；失效键 = caption 内容 hash + TE 指纹） | K2 是自然语言 prompt 模型，tag shuffle/dropout/keep_tokens 对它语义不适用 → 门控关闭，非功能损失。Anima 维持每步在线编码不变 |
| D4 | `model_family` 落在**训练版本级** | 同一项目可并存 Anima/K2 版本，数据集与打标共享 |
| D5 | K2 v1 能力集 = 核心循环 + masked loss + latent/text 缓存 + 训练中采样 | NaViT / SRA / LeapAlign / torch.compile 分块编译 = Anima-only 门控（family 声明能力集 + 复用 `show_when` 裁剪机制）；后续按需逐个开放 |
| D6 | latent 指纹：Anima 与 K2 同为 `wan21-f8c16` → **latent 缓存跨族共享** | 同一数据集两边训不重算 |
| D7 | schema：新增 `model_family`（默认 `anima`，老配置零迁移）；`transformer_path/vae_path/text_encoder_path` 三字段跨族复用；`t5_tokenizer_path` 转 Anima-only `show_when` | K2 的 vae_path 指向同一个 Qwen-Image VAE 文件，无需新下载 |
| D8 | 派发点：supervisor/cmd_builder 不动，入口脚本不变，`phases/models.py` 按 `model_family` 查 registry | |
| D9 | K2 数据集打标推荐链路 = LLM tagger 自然语言 caption | WD14 tag 链路机制上仍可用但非推荐；trigger word 前置继续有效 |
| D10 | K2 的 TE 缓存是训练配置且默认开启；关闭时完全不读写文本 sidecar，Qwen3-VL 常驻并逐 batch 在线编码 | 默认路径用磁盘换显存，预缓存后释放约 9GB TE；80/100GB 云端但本地盘紧张时可关闭，用显存换磁盘。Anima 行为不变；schema/UI 字段在 K2 family 接线阶段落地 |

## 5. 分阶段计划

- **Phase 0 — 验证（不写代码）**：musubi-tuner 手动训一个 K2 LoRA。产出：32GB Windows（WDDM）真实显存峰值、速度、质量、Comfy 推理链路核对。若撞显存崖，最小兜底为 K2 专属小规模 block swap，而非整套显存基建。
- **Phase 1 — 纯重构（对 Anima 零行为变化）**：
  - PR-1：散落的 z_dim/stride/patch 收敛进 ModelSpec + latent npz 缓存加指纹与 layout 版本（现存隐患，独立价值）。
  - PR-2：ModelFamily registry + AnimaFamily 包住现有 loader/forward/sampler；`forward_with_optional_checkpoint`/`navit.py` 移入 AnimaFamily。
  - PR-3：schema 加 `model_family` + 能力集门控管道。
  - PR-4：studio 侧 `ModelsConfig`/catalog 把 `selected_anima` 泛化为 per-family。
- **Phase 2 — 文本缓存基建**（varlen 格式、缓存阶段接入 phases）。
- **Phase 3 — Krea2Family**：modeling 移植（diffusers + musubi 双参考）、加载器、FlowMatchEuler 动态 shift 采样、KREA2_PRESET、下载 catalog（Krea-2-Raw + Turbo + Qwen3-VL-4B）。
- **Phase 4 — UI**：版本级模型族选择、Settings 下载卡、Generate 页接入。

## 6. ModelFamily 接缝草案（待三份视角文档压力测试后冻结）

1. **ModelSpec（声明式常量）**：latent 规格（z_dim/stride/patch/归一化/指纹）、文本规格（seq_len/缓存策略/embed 形状）、默认值 overlay（shift/sampler 白名单/采样默认）、能力集。
2. **ModelFamily 适配器（行为）**：`load_dit / load_text / load_vae`、`encode_text`、`forward_train(model, noisy, t, cond) → v_pred`（梯度检查点为 family 内部职责）、`lora_preset()`、`build_sampler()`。
3. **共享循环边界**：凡只消费 `(latents, noise, t, loss, mask)` 的功能（InfoNoise、masked loss、losses、noise offsets、optimizer/scheduler、LoRA/LoKr 等算法）留在共享循环，零修改。
4. **派发**：`phases/models.py` 查 registry；registry 模式与现有 7 套 plugin registry 同款。

## 7. Open Questions

- 目录结构与代码归属（→ `01-code-layout.md`）：modeling/ 多族布局、runtime 入口与姊妹脚本、utils/ 归属（含此前延后的 utils 重构方案如何衔接）、exec-load 与 `find_diffusion_pipe_root` 机制去留。
- 生态对标启示（→ `02-ecosystem-survey.md`）：kohya/musubi、diffusers、ai-toolkit、OneTrainer、SimpleTuner、diffusion-pipe 的多模型切分方式与反模式。
- 接口冻结面与演化（→ `03-interface-evolution.md`）：接口 v1 冻结哪些方法、预留哪些扩展点、明确不抽象什么；用未来候选模型压力测试。
- 入口脚本 `anima_train.py` 是否改名/泛化（涉及 supervisor cmd_builder 与文档）。
- text cache 的存放位置与文件格式细节（sidecar vs 集中缓存目录）。
- Comfy 侧 K2 LoRA 键名约定核对（musubi 产物 vs ComfyUI 加载，决定我们的保存格式）。
- 32GB Windows WDDM 实测余量（Phase 0 产出）。

## 8. 参考

- Krea 2 Technical Report: https://www.krea.ai/blog/krea-2-technical-report
- diffusers Krea2Pipeline: https://huggingface.co/docs/diffusers/api/pipelines/krea2
- musubi-tuner K2 文档: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md
- Krea 2 商业许可: https://www.krea.ai/krea-2-commercial-license
