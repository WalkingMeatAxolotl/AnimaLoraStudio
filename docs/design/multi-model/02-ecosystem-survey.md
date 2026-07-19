# 多模型支持：训练器生态对标调研（02-ecosystem-survey）

- 状态：调研完成。结论供 `04-synthesis.md` 收敛时引用。
- 日期：2026-07-13
- 相关：`00-decisions.md`（已确认决策）、`01-code-layout.md`（代码归属视角）、`03-interface-evolution.md`（接口契约视角）
- 调研对象：kohya sd-scripts、kohya musubi-tuner、diffusers、ostris ai-toolkit、OneTrainer、tdrussell diffusion-pipe，补充 SimpleTuner 与 ComfyUI（model detection 机制）。全部基于 2026-07 时点的真实仓库目录树、源码与 PR/commit/issue 证据；推断处单独标注。

## TL;DR

1. **全行业收敛到同一形态**：共享训练循环/基建 + 每族一个「适配单元」（薄子类 / 自治目录 / adapter 类），差别只在单元的胖瘦与登记方式。没有一家靠「每族复制完整训练脚本」活得好——diffusers examples（同一 bug 修 10 个脚本）与 sd-scripts full-finetune 线是现成反面教材。我们 D1 选 ModelFamily 适配层与全行业一致。
2. **声明式常量与行为方法分离**是共同结构：ComfyUI 的 `supported_models` 声明类 + `LatentFormat` 类、SimpleTuner 的类属性、musubi 的架构常量，对应我们 §6 的 ModelSpec；load/forward/save 行为对应 ModelFamily 适配器。§6 的两分法得到全生态印证。
3. **加新族的黄金标准 diff**：以新增文件为主、共享代码改动 < 100 行、登记点少而可数（musubi 3 处、ai-toolkit 2 行、SimpleTuner 3 处）。我们应用现有 `validate_schema_consistency` 模式把登记点做成启动期校验——这一点可以做得比所有对标项目更好。
4. **缓存指纹的正确做法是 ai-toolkit 式**：hash 掺 `latent_space_version`（空间标识而非族名），天然支持我们 D6 的 Anima/K2 跨族共享缓存；musubi 按族短名隔离反而做不到共享。OneTrainer 缓存无模型标识、靠训前清缓存兜底，是明确反模式（我们现状同病，PR-1 要修的正是它）。
5. **最大的两类债**：巨型共享文件沉积（train_util.py 曾 6888 行、SimpleTuner common.py 6205 行）与「每族各写一份近乎相同代码」的复制漂移（diffusion-pipe 每个 adapter 复制 timestep 采样）。对我们的直接推论：**timestep 采样必须留在共享循环**，K2 的分辨率感知 shift 应做成 timestep_sampler registry 的新策略 + ModelSpec 默认值指向它，而不是塞进 family 的 forward。
6. 旁注：Anima 已被 sd-scripts、diffusion-pipe、OneTrainer、ComfyUI、diffusers（LoRA 键名转换）全部收编，各家的 Anima 实现（shift 默认值、LoRA 键名、target 选择）可作为我们接口决策的交叉参照。

---

## 1. kohya sd-scripts —— 模板方法胖基类 + 薄入口脚本 + strategy 体系

SD 生态最老牌 LoRA 训练器。main 分支现含 SD1/2、SDXL、SD3、FLUX(+Chroma)、Lumina、HunyuanImage、Anima 共 7+ 族。

**边界与接口**：每族一个薄入口 `{family}_train_network.py`，继承 [train_network.py](https://github.com/kohya-ss/sd-scripts/blob/main/train_network.py) 的 `NetworkTrainer`（2047 行；`train()` 主循环约 950 行不被覆写）。基类默认实现即 SD1.5 行为——SD1.5 没有自己的子类。可覆写 hook 约 27 个，关键的有：`load_target_model`、`get_tokenize_strategy`、`get_latents_caching_strategy`、`get_text_encoding_strategy`、`cache_text_encoder_outputs_if_needed`、`call_unet`、`get_noise_scheduler`、`get_noise_pred_and_target`、`post_process_loss`、`sample_images`、`on_step_start`。覆写量与「离 SD1.5 的距离」成正比：SDXL 子类 183 行/13 个方法（同为 eps-pred，噪声/损失全继承）；SD3 500 行、Flux 555 行、各约 20 个方法——flow matching 族整体替换 `get_noise_pred_and_target` 绕开 `call_unet`。

library/ 下每族固定「三件套 + strategy」：`{family}_models.py`（网络结构自带实现）、`{family}_utils.py`（权重加载/键名转换）、`{family}_train_utils.py`（采样/argparse/timestep）；[strategy_base.py](https://github.com/kohya-ss/sd-scripts/blob/main/library/strategy_base.py)（647 行）定义 `TokenizeStrategy` / `TextEncodingStrategy` / `TextEncoderOutputsCachingStrategy` / `LatentsCachingStrategy` 四个抽象类，每族一个 `strategy_{family}.py` 实现——这是 TE 数量（1/2/3 个）与缓存格式差异的收口点。kohya 自评这套体系 "better than the chaos it was in before"（[issue #1924](https://github.com/kohya-ss/sd-scripts/issues/1924)）。

**加新族 diff**：[Anima PR #2260](https://github.com/kohya-ss/sd-scripts/pull/2260)（2026-02）是最干净实证——新建 9 个文件（入口 ×2、`anima_models/`_utils`/_train_utils`/_vae`、`strategy_anima.py`、`networks/lora_anima.py`）+ tokenizer 资产/docs/tests，只改 2 个共享文件（strategy_base.py、train_util.py）。历史上 SD3/FLUX 直接开在 `sd3` 分支上由 kohya 本人 push（无 PR），该分支承担主开发约 1.5 年后才随 v0.10.0 合回 main。

**LoRA target/键名**：每族一个整文件复制的 [networks/lora_{family}.py](https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora_flux.py)（lora.py 1410 行 vs lora_flux.py 1451 行，大量重复），差异集中在类常量：`UNET_TARGET_REPLACE_MODULE=["Transformer2DModel"]`（SD）vs `["DoubleStreamBlock"]/["SingleStreamBlock"]`（Flux）vs `["SingleDiTBlock"]`（SD3）。键名前缀 DiT 族**故意沿用 `lora_unet`**，lora_flux.py 内注释原文 `# make ComfyUI compatible`。近期新增 `networks/network_base.py` 试图收敛，但各复制文件仍在。

**缓存防串用**：per-model npz 文件名后缀——`{stem}_{W:04d}x{H:04d}_sd.npz` / `_sdxl.npz` / `_sd3.npz` / `_flux.npz`（基类 `cache_suffix` property），TE 输出同理 `_flux_te.npz`。纯文件名约定，无哈希校验。

**timestep 异参**：三层表达——per-family argparse 默认值（[flux_train_utils.py](https://github.com/kohya-ss/sd-scripts/blob/main/library/flux_train_utils.py) `--discrete_flow_shift` 默认 3.0、`--timestep_sampling` choices 含 `flux_shift`；SD3 侧 `--training_shift` 默认 1.0 + `--weighting_scheme`）、代码 if/elif 分支、运行时权重探测族内变体（dev/schnell）。坑的实证：shift 只对部分 sampling 模式生效曾造成用户困惑（[issue #2383](https://github.com/kohya-ss/sd-scripts/issues/2383)），后来专门加日志解释。

**痛点**：`train_util.py` 拆分前 **6888 行**（v0.9.1 时 5729 行），2026-06 [PR #2372](https://github.com/kohya-ss/sd-scripts/pull/2372) 才拆成单一职责模块，PR 标题即 "Refactor for ai agents"。full-finetune 线（`sd3_train.py`/`flux_train.py`/`anima_train.py`）不走 NetworkTrainer 基类、互为复制。kohya 在 #1924 自述："*Musubi tuner is much simpler to implement, so maybe sd-scripts can be simplified by dropping less used features*"——此后新架构一律只进 musubi（[musubi issue #411](https://github.com/kohya-ss/musubi-tuner/issues/411)）。

## 2. kohya musubi-tuner —— 无历史包袱的重写：胖基类 + 每族薄包 + 缓存文件名内嵌架构短名

kohya 的新一代 DiT/视频训练器（2024-12 起），2026-07 已支持 14 族（HunyuanVideo、Wan2.1/2.2、FramePack、FLUX Kontext/FLUX.2、Qwen-Image 三变体、Z-Image、Kandinsky 5、HiDream-O1、Ideogram 4、**Krea 2**）。是本次调研与我们处境最接近的参照（较新代码库 + 已支持 K2）。

**边界与接口**：基类 [training/trainer_base.py](https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/training/trainer_base.py)（约 2500 行，2026-05 [PR #950](https://github.com/kohya-ss/musubi-tuner/pull/950) 从 hv_train_network.py 抽出）。**abstract hooks**（源码标注 `# region model specific`）：`architecture`/`architecture_full_name`（property）、`handle_model_specific_args`（子类必须设 `_i2v_training`、`default_guidance_scale`，可设 `default_discrete_flow_shift`）、`load_vae`/`load_transformer`/`compile_transformer`、`scale_shift_latents`、`call_dit(...) -> DiTOutput`、`process_sample_prompts`/`do_inference`、`convert_weight_keys`（默认恒等）。另有 **extension seams**（`process_batch`、`compute_loss`、`on_transformer_loaded`、`on_train_start`、`on_post_optimizer_step`、`on_before/after_sample_images`、`extra_trainable_params` 等），注释明言 "*no API stability guarantees... if you fork, expect breakage*"。模板方法 `train()` 拆成私有步骤；`get_noisy_model_input_and_timesteps`、optimizer/scheduler、`sample_images` 全共享。

子类覆写实例：`WanNetworkTrainer` 额外覆写 timestep 采样（Wan2.2 high/low 双 DiT 按 `timestep_boundary` 切换）；`QwenImageNetworkTrainer` 的 `architecture` property 是动态的（按 is_edit/is_layered 返回 `qi`/`qie`/`qil` 三个架构名 = 三个缓存命名空间）；`Krea2NetworkTrainer` 覆写 `on_before/after_sample_images` 做 RAW 训练 / Turbo 采样的基座权重热切换（`--turbo_dit`）。

**缓存脚本共享**：公共模块 [cache_latents.py](https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/cache_latents.py) 提供 `encode_datasets` + `setup_parser_common`；每族脚本极薄——[krea2_cache_latents.py](https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/krea2_cache_latents.py) 全文 86 行，只做「建数据集（带 `ARCHITECTURE_KREA2`）+ 定义本族 `encode_and_save_batch` + 回调公共 `encode_datasets`」。Krea 2 直接复用 `qwen_image_utils.load_vae`（同款 VAE，与我们 D6/D7 同一判断）。

**LoRA target/键名**：与 sd-scripts 相反，[networks/lora.py](https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/networks/lora.py) 是唯一通用实现；每族的 `lora_wan.py`/`lora_qwen_image.py`/`lora_krea2.py` 只有 65-72 行：target 常量（`["WanAttentionBlock"]`、`["QwenImageTransformerBlock"]`、**Krea2 = `None` 即全模型所有 Linear**，对应官方推荐练全部 264 个 Linear）+ 默认 exclude_patterns（qwen 必须排 `.*(_mod_).*` modulation Linear；K2 的 modulation 是裸 Parameter 天然不会被包）+ `create_arch_network` 委托。**键名前缀全族统一 `"lora_unet"`**。ComfyUI 键名差异用独立转换脚本解决（`convert_z_image_lora_to_comfy.py` 等）。

**缓存防串用（双层）**：[architectures.py](https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/dataset/architectures.py) 给每族短名+全名（`"wan"`、`"qi"`、`"kr2"`）；主防线是**文件名**——latent 缓存 `{basename}_{W:04d}x{H:04d}_{短名}.safetensors`、TE 缓存 `{basename}_{短名}_te.safetensors`，数据集扫描按 `glob(f"*_{arch}.safetensors")` 天然隔离；次防线是 safetensors metadata `{"architecture": 全名, "format_version": "1.0.1"}`（cache_io.py，加载侧未见强校验——推断：文件名即隔离机制，metadata 供人查验）。cache_io.py 文件头注释自嘲 "*We use simple if-else approach to support multiple architectures*"。

**timestep 异参**：`--timestep_sampling` 的 **choices 本身就是 per-model 菜单**——`sigma/uniform/sigmoid/shift/flux_shift/flux2_shift/ideogram4_shift/qwen_shift/krea2_shift/logsnr/...`（[parser_common.py](https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/training/parser_common.py)）。`krea2_shift` 在共享基类内实现为 `get_lin_function(x1=256, y1=0.5, x2=6400, y2=1.15)` 的分辨率动态 mu（与我们 00-decisions §2 的 base_shift 0.5→max_shift 1.15 互证）。**per-model 推荐值不进代码默认值而进 docs**：[docs/krea2.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md) 推荐 discrete 2.5@1024²，变分辨率直接用 `krea2_shift`；[docs/qwen_image.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/qwen_image.md) 推荐 2.2。基类残留 `default_discrete_flow_shift = 14.5`（HunyuanVideo 遗产，带 `TODO may be None is better`）。

**加新族真实 diff**：[Krea 2 PR #980](https://github.com/kohya-ss/musubi-tuner/pull/980)（2026-06，+2978/−8，24 文件）：新建 21 个（`krea2/` 包 4 文件 + 4 入口 ×src/root 双份 + `lora_krea2.py` + docs + tests）；修改共享文件仅 7 个且极小（architectures.py +2、parser_common.py +1、trainer_base.py +5、cache_io.py +36、attention.py +38）。[Qwen-Image PR #408](https://github.com/kohya-ss/musubi-tuner/pull/408) 同模式。**加一族 ≈ 纯新增 + 共享改动 <100 行，无中央 registry，登记点 3 处靠人记**。

**痛点**：14 族 × 4 脚本 ≈ 60 个入口 + root 层 4 行 shim 镜像（无统一 CLI）；但 gh 搜索 "too many scripts" 零结果——社区答案是第三方 GUI（[issue #55](https://github.com/kohya-ss/musubi-tuner/issues/55)）。kohya 在写给 AI agent 的 [.ai/context/overview.md](https://github.com/kohya-ss/musubi-tuner/blob/main/.ai/context/overview.md) 自述模式："*Each architecture follows the same pattern: an architecture-specific subdirectory for model code, plus top-level scripts... that share the common training/ and dataset/ infrastructure*"，并承认 "*No formal test suite*"。新族一律标 experimental，社区贡献由 kohya 以 follow-up PR 收口。

## 3. diffusers —— 刻意不抽象的 single-file policy：库的哲学，不是应用的哲学

**政策边界**（[官方哲学文档](https://huggingface.co/docs/diffusers/conceptual/philosophy)）：pipeline 主体、scheduler、模型 forward **必须复制**（"prefer copy-pasted code over hasty abstractions"；新模型的做法是 "copy the existing file as a starting point and adapt it"）；共享白名单极窄——`embeddings.py`、`normalization.py`、attention processor，外加整个基础设施层（`from_pretrained`/loaders/mixins）。即：**「论文数学」层复制，「基础设施」层抽象**。

**训练脚本现实**：`examples/dreambooth/` 下 **20 个 `train_*.py` 变体**，每个约 2000 行；实测 flux 版与 sd3 版 **约 85% 的行完全相同**。`# Copied from` + `make fix-copies` 机制只覆盖 `src/diffusers`（check_copies.py 硬编码路径），**examples/ 同步纯靠人肉**。直接后果：[PR #13899](https://github.com/huggingface/diffusers/pull/13899) "Fix fp16 LoRA unscale crash... in **remaining** DreamBooth LoRA scripts" 一次改 10 个脚本（标题的 remaining 说明此前已有另一批）。官方立场（examples README）：examples "are just that – examples"，预期用户自己改，不追求生产级。

**LoRA 键名的账单**：[lora_conversion_utils.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/lora_conversion_utils.py) 约 3000+ 行、**26 个手写转换函数**，按「模型 × 训练器生态」逐个实现（`_convert_kohya_flux_lora_to_diffusers` 独占约 570 行；**含 `_convert_non_diffusers_anima_lora_to_diffusers`**）。单文件 checkpoint 识别同理：[single_file_utils.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/single_file_utils.py) 的 `CHECKPOINT_KEY_NAMES` 是「模型 → 特征张量键名」指纹表 + 平铺 if 链。

**适用条件（对我们最重要的一课）**：这套哲学的前提是「产物被读和 fork 多于被调用、模型发布后静态」（[设计哲学博文](https://huggingface.co/blog/transformers-design-philosophy)）；官方自己也说构建 feature-complete 应用要用 Modular Diffusers 而非 pipeline。训练器应用不满足前提：功能（暂停/恢复、masked loss、eval）跨模型持续演进、训练脚本恰恰不是静态的——#13899 式 10 连修就是 "model stasis" 假设在训练侧失效的证明（推断）。**diffusers 能容忍 examples 失同步是因为定位就是教学参考；产品级训练器没有这个退路。**

## 4. ostris ai-toolkit —— arch 注册 + 窄 hook + 缓存版本号：单人维护 20+ 族的低摩擦配方

config 驱动的流行训练器（11.3k stars），几乎所有新模型第一时间收编。

**边界与接口**：[toolkit/models/base_model.py](https://github.com/ostris/ai-toolkit/blob/main/toolkit/models/base_model.py) 的 `BaseModel`（约 1600 行），子类以类属性 `arch` 为唯一标识。**必须实现**：`load_model`、`get_generation_pipeline`/`generate_single_image`（采样只收 embeds 不收文本）、`get_noise_prediction`（训练前向；timestep 契约统一 0..1000，模型内部自行换算）、`get_prompt_embeds`、`get_model_has_grad`/`get_te_has_grad`。**可选窄 hook**：`encode_images`/`decode_latents`（VAE 差异）、`condition_noisy_latents`（control/edit 注入点）、`get_loss_target`、`convert_lora_weights_before_save/load`、`get_transformer_block_names` 等。基类还有一排**能力 flag**：`is_flow_matching`、`is_multistage`、`do_masked_loss`、`latent_space_version` 等。训练循环在共享的 `BaseSDTrainProcess`（2806 行）+ `SDTrainer`（2186 行），按「缓存/编码 latent → 采 timestep 加噪 → `condition_noisy_latents` → `predict_noise` → `get_loss_target` → MSE → `scale_loss`」的固定序列调 hook。

**注册机制**：`get_all_models()` 用 pkgutil 扫 `extensions_built_in/` 收集模块级 `AI_TOOLKIT_MODELS` 列表；config yaml 里 `model.arch: "qwen_image"` 选型；匹配不到 fallback 到 legacy `StableDiffusion` god-class。**双世代并存是现实痛点**：老族（sd1/sdxl/flux 等）走 legacy 类，训练代码里仍有 `if self.sd.is_flux or 'flex' in self.sd.arch` 式特判泄漏（BaseSDTrainProcess.py:1196）。

**加新族 diff**：[Krea2 commit 99be3d96](https://github.com/ostris/ai-toolkit/commit/99be3d96)（PR #906，+1307/−1）：新增 `krea2/` 目录（注册 3 行 + 主类 474 行 + vendor 的 `src/{mmdit,pipeline,text_encoder}.py`）；改动仅 `__init__.py` +2 行注册、UI options.ts +21、README/version 各 +1。**共享训练代码零改动**。官方还有逐行注释的 [example_model 模板](https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/diffusion_models/example_model/README.md)（明说给人和 AI agent 用）。

**LoRA**：模型类设 `self.target_lora_modules = ["QwenImageTransformer2DModel"]`（顶层模块类名），共享 `LoRASpecialNetwork` 递归包裹；保存键名转换是 per-model 方法 `convert_lora_weights_before_save`（qwen: `"transformer." → "diffusion_model."`，即 ComfyUI 约定）。

**缓存防串用（本次调研最优解）**：latent 缓存文件名 = `{stem}_{md5(info_dict)}.safetensors`，[info_dict 掺 `latent_space_version`](https://github.com/ostris/ai-toolkit/blob/main/toolkit/dataloader_mixins.py) 连同 crop/scale/flip 一起入 hash；文本嵌入同理掺 `text_embedding_space_version`（默认返回 arch）。版本值优先取模型声明——**共享 latent 空间的模型声明同值即可复用缓存**（与我们 D6「Anima/K2 同为 wan21-f8c16 跨族共享」完全同构）；模型实现破坏兼容时 bump 版本作废用户缓存。

**timestep 异参**：模型类可提供 `get_train_scheduler()` 静态方法 + 文件顶部自己的 `scheduler_config` dict（qwen: shift 1.0/dynamic shifting）；采样分布是全局旋钮 `timestep_type`（sigmoid/linear/lognorm_blend/...）。即「**分布归 config、shift 参数归模型 scheduler**」。多专家模型用 `is_multistage + multistage_boundaries` 裁 timestep 采样区间。

**痛点**：贡献极度集中（作者 1241 commits，第二名 5）；缓存相关 bug 反复（#374/#405/#425/#689/#779）；[issue #693](https://github.com/ostris/ai-toolkit/issues/693)（Z-Image 训练质量差被社区点名）反映「模型接得快、per-model 调优跟不上」；模型类本身还在世代迁移（z_image 后来才移到新基类）。

## 5. OneTrainer —— 族 × 训练方法矩阵式工厂：维护者亲证的 boilerplate 之痛与瘦身路线

带 GUI 的多模型训练器，与我们「上面有一层 UI」的处境最像。ModelType 枚举 29 个成员（含 **ANIMA**、KREA2）× TrainingMethod 4 种（FINE_TUNE/LORA/EMBEDDING/FINE_TUNE_VAE）。

**目录职责**（[官方 ProjectStructure.md](https://github.com/Nerogar/OneTrainer/blob/master/docs/ProjectStructure.md)）：`modules/model/`（数据类，族内自带 `encode_text`/`pack_latents`/`calculate_timestep_shift` 等模型知识）、`modelSetup/`（训练准备，基类抽象 `create_parameters`/`setup_model`/`predict`/`calculate_loss`/`after_optimizer_step`）、`modelLoader/`、`modelSaver/`、`modelSampler/`、`dataLoader/`（基于自研 MGDS）；`GenericTrainer` 拼装一切——训练循环模型无关。

**矩阵与工厂**：2026-01 前是散落 if/elif；[PR #1211](https://github.com/Nerogar/OneTrainer/pull/1211)「Factory pattern for model components」（动机原文："*make it easier to implement new models by having the model-specific code less spread around the entire codebase*"）+ [PR #1498](https://github.com/Nerogar/OneTrainer/pull/1498) 改成装饰器注册表：`@factory.register(BaseModelSetup, ModelType.FLUX_DEV_1, TrainingMethod.LORA)`，factory.py 仅 37 行（重复注册 raise），`create_*` 一行分派，sampler/dataLoader 支持按 `(type)` 降级查找。**实测规模**：modelSetup/ 70 个文件（45 具体类 + 18 族中间 Base + 7 mixin）、modelLoader/ 83、modelSaver/ 96——modules/ 全部 557 个 py。跨族算法复用全靠 mixin：`ModelSetupDiffusionLossMixin`（masked loss/MinSNR/debiased/log-cosh）、`ModelSetupNoiseMixin`（offset noise/timestep 采样）、`ModelSetupFlowMatchingMixin` 等。

**加新族 diff**：[Anima PR #1487](https://github.com/Nerogar/OneTrainer/pull/1487)（2026-07 合并）：23 文件 +1436 行——**新增 15 个文件**（11 py：Model/DataLoader/Loader/Sampler/saver ×4/BaseAnimaSetup+FineTune+LoRA Setup）+ **8 处横切修改**（ModelType 枚举、3 个 UI View、muon_util 层规则等）。[Z-Image PR #1195](https://github.com/Nerogar/OneTrainer/pull/1195) 结构几乎相同（26 文件 +1611）。注意 Anima 的 loader 已因 `Generic*ModelLoader` 类工厂从老族的 6 文件缩到 1 文件。

**LoRA**：共享 `LoRAModuleWrapper(root_module, prefix, config, module_filter, ...)`，族差异在 (a) Setup 传的 prefix；(b) 族 Base 类的 `LAYER_PRESETS` 字典（attn-mlp/attn-only/blocks/full）；(c) `fusion_groups()` 处理 qkv 融合。对外格式在 `convert_lora_util.py`（[PR #1563](https://github.com/Nerogar/OneTrainer/pull/1563) 支持 Diffusers/Kohya/Comfy/Legacy 四种输出）；键名坑真实存在（PR #1294 Z-Image prefix workaround、#1589 Ideogram qkv 修复）。

**缓存（反模式）**：MGDS `DiskCache` 的 group key 是 concept 路径/seed 等子配置的 sha256，**不含模型标识**；防串台靠 `clear_cache_before_training` 默认 True + workspace 隔离——用户责任而非键控（推断：有意换取实现简单）。

**timestep 异参**：mixin 分层（通用 `_get_timestep_discrete/_continuous` + shift 参数；flow 族混入 `_add_noise_discrete`）+ 族 `Base<Family>Setup.predict()` 内的特有逻辑（Flux 的 `calculate_timestep_shift(h, w)` 动态 shift，`config.dynamic_timestep_shifting` 开关）。

**痛点（维护者自述，非推断）**：现任维护者 dxqb 开着 [issue #1203](https://github.com/Nerogar/OneTrainer/issues/1203)「[Feat]: Simpler new model support」，直言 "*With the amount of new models released, it could be easier to implement new models*"，点名：dataLoader/Loader/Saver 各处 boilerplate、**sampler 与 `predict()` 重复前向代码（训练/采样条件不一致风险）**、tokenization 每族重复、「改这些会破坏既有模型、需要大量测试」。OneTrainer 的演化方向不是消灭矩阵，而是**把格子越做越薄**——公共体沉入 mixin/类工厂，格子只剩注册装饰器 + 族差异（推断）。

## 6. tdrussell diffusion-pipe —— models/ 每族一个 adapter 类：接口最接近我们 §6 草案的一家

DeepSpeed pipeline-parallel 训练器，26 族。分派是 [train.py](https://github.com/tdrussell/diffusion-pipe/blob/main/train.py) L310-379 的 `if model_type == 'flux': ... elif ...` 硬编码链。

**基类结构**：[models/base.py](https://github.com/tdrussell/diffusion-pipe/blob/main/models/base.py) 实为三类——`CommonPipeline`（共享：`configure_adapter`、采样、媒体预处理、类属性默认 `spatial_compression=8`/`channels=16`）、`BasePipeline`（老路线，手工加载权重）、`ComfyPipeline`（2025-11 改 GPL-3 后新路线：直接复用 ComfyUI 的模型/VAE/CLIP 加载代码，支持从 Comfy fp8_scaled 权重直训）。

**接口清单**：类属性 `name`（缓存目录键）/`checkpointable_layers`/`adapter_target_modules`；方法 `load_diffusion_model`（缓存完成后才加载 DiT）、`get_vae`/`get_text_encoders`、`get_call_vae_fn`/`get_call_text_encoder_fn`（返回闭包，输出待缓存 dict）、**`prepare_inputs`（核心：缓存 batch → 采 timestep、加噪、算 flow target、mask 下采样，返回 tensor tuple）**、`to_layers`（切 pipeline stage；一切跨层数据必须是 flat tensor tuple——接口最大的形状约束）、`save_adapter`/`save_model`（per-model 格式）、`get_param_groups`（**Anima 覆写做 `llm_adapter_lr`**）、`get_loss_fn`（默认 MSE+mask 加权 fp32）、`enable_block_swap`（默认 NotImplemented）。

**加新族 diff**：Ideogram4（[commit d6a8f562](https://github.com/tdrussell/diffusion-pipe/commit/d6a8f562)）：新建 models/ideogram4.py(+352) + train.py(+8/-2 一个 elif) + dataset.py(+3/-2)。**Anima（[commit 3abbfff5](https://github.com/tdrussell/diffusion-pipe/commit/3abbfff5)）不建新文件**——作为 Cosmos-Predict2 变体改 `models/cosmos_predict2.py`(+227/-59) + 新 `llm_adapter.py`，train.py 仅 1 行（两个 model_type 共用一个 Pipeline 类，类内 `self.name = 'anima'`）。README 自称 "Easily add new models by implementing a single subclass"，从 diff 看基本属实（300-600 行/族）。

**LoRA**：`adapter_target_modules = ['QwenImageTransformerBlock']`（模块类名），共享 `configure_adapter` 收集匹配模块下所有 Linear 全名 → `peft.LoraConfig`（也支持 LoKr）。保存格式 per-model，[docs/supported_models.md](https://github.com/tdrussell/diffusion-pipe/blob/main/docs/supported_models.md) 逐一标注：SDXL=Kohya 格式、Flux/SD3=Diffusers 格式、**Wan/Qwen-Image/Anima 等=ComfyUI 格式**。

**缓存**：写在每个数据集目录内 `{dataset}/cache/{model.name}/`，按模型族名分子目录（Anima 与 cosmos 共用类但 `name` 不同 → 缓存自然隔离），配 fingerprint（utils/cache.py）+ `--regenerate_cache`/`--trust_cache` flags。换族免清缓存；但**同族实现变更时要手动清**（README：Flux2 改 attention masking 后 "Delete cache folder... or else you might get Tensor shape errors"）。代价：text embeds 全缓存 → 不支持训 TE（SDXL 靠 `model.name == 'sdxl'` 特判例外——共享代码里的族名 if，坏味道实证）。

**timestep 异参（反面教材）**：**没有共享实现**——每个 adapter 在 `prepare_inputs` 里各写一份近乎相同的采样代码（qwen_image.py 与 cosmos_predict2.py 结构一致），共享的只有 `time_shift`/`get_lin_function` helper。旋钮在 config `[model]` 节（`timestep_sample_method`/`sigmoid_scale`/`shift`/`flux_shift`）。**Anima 示例 config 不设 shift（即 ≈1.0）**，只建议更低 LR + `llm_adapter_lr = 0`——可作我们 timestep_shift A/B（baseline 3.0 vs 更低）的旁证。

**痛点**：README 明说原生 Windows "difficult or impossible"（DeepSpeed 硬依赖，须 WSL2；[issue #268](https://github.com/tdrussell/diffusion-pipe/issues/268) 用户自述装环境数小时）；feature × model 矩阵不齐（Full Fine Tune 8 族 ❌、fp8 5 族 ❌、block swap 非全族）；TOML config 无 schema 校验；缓存类 issue 多（#444/#492/#509/#313）。

## 7. 补充一：SimpleTuner —— 声明式类属性 + 惰性注册表 + 按族扇出的文档债

37 族注册（含 anima、krea2），每族一个自治目录 `simpletuner/helpers/models/{family}/`（`model.py` + vendor 的 `pipeline.py`/`transformer.py`——不从 diffusers import，隔离好但上游修 bug 不自动同步）。

**接口**：[common.py](https://github.com/bghira/SimpleTuner/blob/main/simpletuner/helpers/models/common.py)（**6205 行**）分层基类 `ModelFoundation → ImageModelFoundation / VideoModelFoundation / AudioModelFoundation`；抽象方法仅 `model_predict`、`_encode_prompts` 等少数几个，**大头是类属性声明**：`NAME`、`PREDICTION_TYPE`、`AUTOENCODER_CLASS`、`LATENT_CHANNEL_COUNT`、`PIPELINE_CLASSES`、`HUGGINGFACE_PATHS`（flavour→repo 映射，承担同族多版本）、`TEXT_ENCODER_CONFIGURATION`、`DEFAULT_LORA_TARGET`。通用 load 逻辑读属性完成加载。

**注册**：每族 `ModelRegistry.register("qwen_image", QwenImage)` + `model_metadata.json` + `LazyModelClass` 惰性导入（CLI 列 flavour 不触发重依赖 import，为启动速度）。加新族（[krea2 首 commit 08b05a94](https://github.com/bghira/SimpleTuner/commit/08b05a942654cd2ac006cfc741072896e8b35258)，2312 行/11 文件）= 新目录 4 文件 + metadata/common/WebUI 三处一行级登记 + 单测。

**缓存**：text embed 文件名 = `md5(key) + f"-{model_family}"`（族名直接编进文件名）；VAE 缓存靠目录 `cache/vae/{model_family}/{dataset_id}`。v2.0 release 明言升级后缓存需重建。

**痛点**：横切特性（TREAD/CREPA/量化/block-swap）全沉积在 6205 行基类；横切改动按族扇出（维护者自开 issue 集群 #2601–#2607，multi-stage validation 要给每族各开一张票）；文档扇出更重——quickstart 234 个文件（每族 × 6 语言），AGENTS.md 强制改文档同步所有翻译。open issue 仅 20，但与激进关票风格及单维护者高强度投入有关，不能直接推出架构无维护成本（推断）。

## 8. 补充二：ComfyUI —— 键名指纹检测 + 声明类 + LatentFormat：推理侧的 ModelSpec 样板

非训练器，但其「从 checkpoint 自动识别模型族」与声明式模型档案值得单列。

**两段式识别**（[model_detection.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_detection.py)）：先统计权重前缀，再用**指纹键存在性判族 + 张量形状反推超参**（`'{}txt_norm.weight' → "qwen_image"`、`'{}txtfusion.projector.weight' → "krea2"`；层数用 `count_blocks` 数出来）。然后对 [supported_models.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/supported_models.py)（约 100 个类）线性扫描、首个 `matches()` 命中即胜——列表顺序是语义的一部分（特化子类必须排前，推断）。

**声明类**：每族一个 class 只声明 `unet_config` 匹配模板、`sampling_settings`（**Anima shift=3.0、Qwen-Image shift=1.15**）、`latent_format = latent_formats.Wan21`（类引用）、`supported_inference_dtypes`、`memory_usage_factor`、`get_model()`、`clip_target()`。[latent_formats.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py) 约 35 个子类，字段：`scale_factor`/`latent_channels`/`latent_rgb_factors`（零成本预览）/`taesd_decoder_name`；`Wan21`（**Anima/Qwen-Image 同用**）用逐通道 mean/std 归一化——与我们「Anima=Qwen-Image VAE=Wan2.1 latent 空间」的既有结论互证。LoRA 键名转换集中在 [comfy/lora.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/lora.py)，把 diffusers/PEFT/kohya 键统一映射到内部 `diffusion_model.*`。

**加新族 diff 极小**：Qwen-Image（[commit c0124002](https://github.com/comfyanonymous/ComfyUI/commit/c012400240d4867cd63a45220eb791b91ad47617)）8 文件 +561 行；Anima（[commit abe2ec26](https://github.com/comfyanonymous/ComfyUI/commit/abe2ec26a61ff670b9c0e71e4821c873368c8728)，PR #12012）6 文件 +326 行，**model_detection.py 仅 +2 行**（复用 cosmos_predict2 检测分支改标签）。用户零配置、单 checkpoint 自动定族。

## 9. 横向对比总表

| 项目 | 适配单元形态 | 接口风格 | 加一族真实 diff | LoRA target/键名 | 缓存防串用 | timestep 异参表达 | 最痛的债 |
|---|---|---|---|---|---|---|---|
| **sd-scripts** | 薄入口子类 + library 三件套 + strategy_* | 模板方法胖基类，~27 hook | Anima #2260：新建 9 文件、改 2 共享 | 每族整文件复制 lora_*.py；前缀统一 `lora_unet`（Comfy 兼容） | npz 文件名后缀 `_sd`/`_flux`… | per-family argparse 默认 + if/elif + 权重探测变体 | train_util 6888 行巨石（已拆）；FFT 线复制；长寿 sd3 分支 |
| **musubi-tuner** | 每族包 + 4 个薄入口脚本 | 胖基类 abstract hooks + extension seams（不保稳定） | Krea2 #980：24 文件、新建 21、共享 <100 行 | lora.py 通用 + 每族 ~70 行常量；前缀统一 `lora_unet` | **文件名内嵌架构短名** + metadata（无强校验） | `--timestep_sampling` choices 菜单；推荐值进 docs 不进 default | 14 族 × 4 ≈ 60 入口；登记点靠人记；无测试套件 |
| **diffusers** | 每 pipeline/每脚本整份复制 | 刻意不抽象（single-file policy） | 复制现有文件再改 | peft 统一 + 3000 行外来格式转换账单 | n/a | 每脚本自带 | 同一 bug 修 10 个脚本；examples 无 fix-copies 覆盖 |
| **ai-toolkit** | 每族 extension 目录 + BaseModel 子类 | 基类 + 窄 hook + 能力 flag；`arch` 属性扫描注册 | Krea2：新目录 + 2 行注册，**共享零改动** | `target_lora_modules` 属性 + per-model 保存键名转换 | **hash 掺 latent_space_version（可跨族共享）** | per-model scheduler_config + 全局 timestep_type | 单人瓶颈；legacy 双轨并存、arch 特判泄漏 |
| **OneTrainer** | 族 × 训练方法矩阵格子 + 装饰器工厂 | 注册表 + 族 Base + 7 组 mixin | Anima #1487：15 新文件 + **8 处横切修改**（含 3 个 UI View） | 共享 Wrapper + prefix + LAYER_PRESETS + 4 格式导出 | **无模型标识**；默认训前清缓存（用户责任） | mixin 分层 + 族 Setup.predict 内特有逻辑 | 矩阵 boilerplate（维护者自开 #1203）；sampler/predict 双份前向 |
| **diffusion-pipe** | models/ 每族 adapter 类 | 类属性 + 方法；train.py if/elif 链分派 | Ideogram4：新 py+352 + elif 一行；Anima 寄生 cosmos 文件 | `adapter_target_modules` 类名匹配 → peft；save per-model（Anima=Comfy 格式） | 缓存目录按 `model.name` 分 + fingerprint | **各 adapter 的 prepare_inputs 内复制**（反面教材） | Windows 不可用；feature × model 矩阵不齐 |
| **SimpleTuner** | 每族自治目录（vendor 实现） | 声明式类属性 + 6205 行分层基类 + 惰性注册表 | Krea2：11 文件、新目录 4 + 3 处一行登记 | `DEFAULT_LORA_TARGET` 类属性 | text 文件名带族名；VAE 按族目录 | `PREDICTION_TYPE` 属性 + per-族 model.py | 横切特性沉积基类；改动/文档按族扇出 |
| **ComfyUI**（推理） | supported_models 声明类 + LatentFormat 类 | 键名指纹自动检测 + 声明式档案 | Anima：6 文件 +326、检测 +2 行 | comfy/lora.py 集中多生态键映射 | n/a | `sampling_settings` per-族声明 | 首命中线性扫描的顺序语义；集中文件持续膨胀 |

## 10. 对我们的启示

### 10.1 值得采纳的模式

- **P1 适配单元 = 「声明常量 + 行为方法」两件套**。ComfyUI（unet_config/sampling_settings/latent_format 声明类）、SimpleTuner（类属性）、ai-toolkit（能力 flag + 窄 hook）三家从不同方向收敛到同一结构。我们 §6 的 ModelSpec / ModelFamily 两分法与之同构，方向正确；ComfyUI 的 `LatentFormat` 类（scale/channels/latent_rgb_factors 每族一份）就是 ModelSpec latent 部分的现成样板——我们的 latent2rgb 预览系数也应进 ModelSpec。
- **P2 加新族 = 纯新增文件 + 少而可数且有校验的登记点**。musubi（3 处登记）、ai-toolkit（2 行）、SimpleTuner（3 处）都做到共享代码近零改动，但登记一致性全靠人记。我们已有 `validate_schema_consistency()` 模式（losses registry ↔ schema Literal 启动期双向校验），把它扩展到 `model_family`（registry keys ↔ schema Literal ↔ catalog 条目），可以做到全生态没人做到的「登记漏一处启动即 fail」。
- **P3 缓存指纹 = 空间标识 + layout 版本，而非族名**。ai-toolkit 的 `latent_space_version`/`text_embedding_space_version` 掺 hash 是最优解：既防串用，又允许同 latent 空间的族声明同值共享缓存——这正是 D6（Anima/K2 同 `wan21-f8c16`）需要的机制；musubi 按族短名隔离反而使同 VAE 的 Krea2/Qwen-Image 各存一份。PR-1 给 npz 缓存加指纹时应直接采用「指纹 = latent 空间标识 + layout 版本」，族名不入键。同族实现变更 bump 版本作废旧缓存（ai-toolkit / diffusion-pipe README 都有「不清缓存报 shape error」的教训）。
- **P4 timestep：算法进共享 registry 菜单，数值默认进 per-family overlay**。musubi 把 `krea2_shift` 做成共享基类内的 `--timestep_sampling` choices 项、推荐数值放 docs；ai-toolkit「分布归 config、shift 归模型 scheduler_config」；ComfyUI `sampling_settings` 声明。共同点是**分辨率感知 shift 这类算法只实现一次**。
- **P5 键名约定向 ComfyUI 看齐是行业事实标准**。musubi 全族统一 `lora_unet` 前缀、sd-scripts 注释明写 "make ComfyUI compatible"、diffusion-pipe/ai-toolkit 用 `diffusion_model.` 前缀保存。且**保存键名转换是 per-model 的窄钩子**（musubi `convert_weight_keys`、ai-toolkit `convert_lora_weights_before_save`），不要试图写通用映射——diffusers 那 3000 行 `lora_conversion_utils.py` 是被动承接全生态格式的账单，训练器只需管好自己的输出格式。
- **P6 studio 层是我们的差异化优势，不是负担**。各家加新族最横切的改动恰是 UI/文档：OneTrainer 要手改 3 个 UI View、ai-toolkit 要改 options.ts、SimpleTuner 文档按族 × 语言扇出 234 份。我们 pydantic schema 驱动表单 + `show_when`/能力集裁剪意味着 family 声明能力集后 UI 自动收敛——没有任何对标项目有这个机制（他们的 GUI 都是手写的）。D5 的能力集门控要坚持走 schema 管道而非前端 if。
- **P7 用「检测」做防呆而非选型**。ComfyUI 的键名指纹检测支撑「用户丢任意 checkpoint 自动定族」；我们 v1 用显式 `model_family` 字段（D7）更简单可控，但可以把现有 `load_anima_model` 的形状推断泛化为 family 声明的「指纹校验」：加载时验证 checkpoint 与所选 family 匹配，防止用户给 K2 版本配了 Anima 权重路径——消费级用户场景里这类错误必然出现。

### 10.2 反模式警告

- **A1 巨型共享文件沉积**。train_util.py 6888 行（kohya 最后靠 "Refactor for ai agents" 拆掉）、SimpleTuner common.py 6205 行、ai-toolkit BaseSDTrainProcess 2806 行。横切功能（我们的 InfoNoise/masked loss/eval/暂停恢复）必须继续走 plugin registry/独立模块，禁止向 ModelFamily 基类或共享 loop 文件堆积。
- **A2 复制漂移**。diffusers examples 同 bug 修 10 遍；diffusion-pipe 每个 adapter 复制一份 timestep 采样；sd-scripts FFT 线整脚本复制。判据：**凡「每族各写一份近乎相同代码」的面，迟早还债**。接口设计时如果发现两个 family 的某方法实现将逐字相同，说明该逻辑应上移到共享循环或 spec 参数。
- **A3 矩阵爆炸**。OneTrainer 族 × 方法 → modelSaver/ 96 个文件，维护者自开 issue 求简化。我们只有 LoRA/LoKr 一种训练方法，天然免疫；但若未来加全参微调，**必须作为 family 内的一个模式参数而非第二维度的类矩阵**。
- **A4 双世代并存 / 迁移不彻底**。ai-toolkit legacy `StableDiffusion` god-class 与 `BaseModel` 两套并存多年，`is_flux` 特判泄漏进训练循环至今。对我们：Phase 1 PR-2 必须把 Anima **完整**迁入 AnimaFamily（含 `forward_with_optional_checkpoint`/navit.py），不留「Anima 走老路径、K2 走新路径」的过渡态——过渡态会永久化。
- **A5 族名 if 泄漏进共享代码**。diffusion-pipe 的 `model.name == 'sdxl'` 特判、ai-toolkit 的 `'flex' in self.sd.arch`。执行纪律：共享循环里出现 `if family == "anima"` 即为坏味道，一律改为能力 flag / spec 字段查询（D5 的能力集就是为此准备的）。
- **A6 长寿开发分支**。sd-scripts 的 sd3 分支承担主开发 1.5 年不合 main。我们 Phase 1-4 的小步 PR 序列是对的，坚持每个 PR 独立可合。
- **A7 缓存无标识 + 「用户自己清」**。OneTrainer 默认训前清缓存、diffusion-pipe/musubi 换实现要手动清且报错难懂。消费级 Windows 用户不会读 README——指纹失配必须自动重算并在 UI 提示，不能抛 shape error。
- **A8 入口脚本线性膨胀**（musubi 60 个入口）。对我们影响有限（supervisor 拉起单入口，D8 已定派发点在 `phases/models.py`），但印证了「入口不动、内部派发」的选择：不要走 per-family 姊妹脚本路线。

### 10.3 对 00-decisions §6 接缝草案的对标修正意见

逐条对照 §6 草案与各家真实接口后的具体建议（供 03/04 文档冻结接口时采纳）：

1. **§6.1 ModelSpec：整体印证，建议补 3 项**。(a) `latent_rgb_factors`（预览投影系数，学 ComfyUI LatentFormat——我们已知 Anima 必须用 Wan21 系数的教训）；(b) 能力集之外再留**布尔能力 flag 面**（学 ai-toolkit `is_flow_matching`/`do_masked_loss`），供共享循环做细粒度查询，能力集列表管 UI 裁剪、flag 管代码路径，两者职责不同；(c) 缓存指纹字段明确为「空间标识 + layout version」两段（见 P3），族名不参与。
2. **§6.2 适配器方法：草案 7 个方法覆盖了 musubi abstract hooks 的最小集，建议增 1 改 1**。**增**：`convert_weight_keys` / 保存键名钩子（musubi、ai-toolkit 都单列；我们的 `lora_preset()` 目前只管 target，键名约定与 Comfy 兼容转换也需 per-family 落点，且 §7 的「Comfy 键名核对」open question 正落在这里——v1 就要冻结）。**改**：`forward_train` 的签名建议对齐 musubi `call_dit -> DiTOutput`（返回 pred+target 数据类而非裸 `v_pred`）——K2 与 Anima 虽同为 RF，但保留 target 构造的族内自由度可以吸收未来非 v-pred 族，而 loss 仍留共享循环。**明确不加**：per-family 的 timestep 采样钩子（见下条）与 `handle_model_specific_args` 型配置校验钩子（pydantic schema + show_when 已覆盖，这是 studio 层替我们省掉的一类接口）。
3. **§6.3 共享循环边界：印证，外加一条硬规则**。「凡只消费 (latents, noise, t, loss, mask) 的功能留共享循环」得到全部对标印证（OneTrainer 的 masked loss 就在 DiffusionLossMixin 一次实现全族生效）。硬规则：**timestep 采样属于共享循环**——diffusion-pipe 把它放进 adapter 导致每族复制（A2 实证），musubi 把它放在共享基类 + choices 菜单（P4 正解）。K2 的分辨率感知 shift 应实现为我们现有 timestep_sampler registry 的新策略（如 `krea2_dynamic_shift`），ModelSpec 的默认值 overlay 指向它；不要做成 Krea2Family 的方法。
4. **§6.4 派发：印证 + 升级**。registry 派发与 OneTrainer factory（37 行、重复注册 raise）/ai-toolkit 扫描注册同款；我们的升级空间是 P2 的启动期一致性校验（schema Literal ↔ registry ↔ 下载 catalog 三方对齐），把 musubi「3 处登记靠人记」的弱点变成我们的强项。
5. **两条 §7 open question 的生态答案**。(a) 入口脚本改名：对标支持「不改」——ComfyUI/ai-toolkit/SimpleTuner 都是稳定入口 + 内部派发，musubi 的 per-family 入口是它没有 config schema 层的补偿（A8）；`anima_train.py` 文件名的历史兼容成本远低于动 supervisor。(b) text cache 位置：diffusion-pipe（数据集目录内 per-model 子目录）与我们 mask sidecar 的同目录哲学最接近，但 text cache 按 D3 带 caption hash + TE 指纹，集中或 sidecar 均可，关键是失效键设计（P3）而非位置。
6. **Phase 0/3 的直接输入**：musubi K2 参考面——target = 全部 264 Linear（`KREA2_TARGET_REPLACE_MODULES = None`）、dim32/alpha32、`krea2_shift` 或 discrete 2.5@1024²、训练 RAW / 采样可热切 Turbo（`--turbo_dit`，实现在 `on_before/after_sample_images`）——我们 `build_sampler()` 设计时需决定验证采样用 Raw 还是 Turbo，musubi 已给出可抄的答案。另：diffusion-pipe Anima 示例 shift≈1.0 + ComfyUI Anima 声明 shift=3.0，两者并存恰说明「训练 shift 与推理 shift 是两个旋钮」，也旁证我们 timestep_shift A/B（3.0 vs 更低）值得做。

## Sources

**kohya sd-scripts**
- https://github.com/kohya-ss/sd-scripts/blob/main/train_network.py （NetworkTrainer 基类与 hooks）
- https://github.com/kohya-ss/sd-scripts/blob/main/sdxl_train_network.py 、sd3_train_network.py 、flux_train_network.py 、anima_train_network.py （子类覆写量）
- https://github.com/kohya-ss/sd-scripts/blob/main/library/strategy_base.py 、strategy_sd.py 、strategy_flux.py （strategy 体系与缓存后缀）
- https://github.com/kohya-ss/sd-scripts/blob/main/library/flux_train_utils.py 、sd3_train_utils.py （per-family argparse 默认值、flux_shift）
- https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py 、lora_flux.py 、lora_sd3.py （target 常量、lora_unet 前缀、Comfy 兼容注释）
- https://github.com/kohya-ss/sd-scripts/pull/2260 （Anima 支持 PR，加新族文件清单）
- https://github.com/kohya-ss/sd-scripts/pull/2372 （train_util 拆分 "Refactor for ai agents"）
- https://github.com/kohya-ss/sd-scripts/issues/1924 （kohya 自述 "Musubi tuner is much simpler"）
- https://github.com/kohya-ss/sd-scripts/issues/2383 （shift 只对部分采样模式生效的用户困惑）

**kohya musubi-tuner**
- https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/training/trainer_base.py （abstract hooks + extension seams）
- https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/training/parser_common.py （timestep_sampling choices 菜单）
- https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/cache_latents.py 、krea2_cache_latents.py （缓存脚本共享模式）
- https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/dataset/architectures.py 、cache_io.py 、image_video_dataset.py （架构短名与缓存文件名）
- https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/networks/lora.py 、lora_wan.py 、lora_qwen_image.py 、lora_krea2.py （通用 LoRA + 每族常量）
- https://github.com/kohya-ss/musubi-tuner/pull/980 （Krea 2 支持 PR）、pull/408 （Qwen-Image）、pull/950 （基类抽出重构）
- https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md 、docs/qwen_image.md （per-model 推荐值进 docs）
- https://github.com/kohya-ss/musubi-tuner/blob/main/.ai/context/overview.md （kohya 架构自述）
- https://github.com/kohya-ss/musubi-tuner/issues/411 、issues/55

**diffusers**
- https://huggingface.co/docs/diffusers/conceptual/philosophy （single-file policy 定义与共享白名单）
- https://huggingface.co/blog/transformers-design-philosophy （反 DRY 论证、Copied-from 起源）
- https://github.com/huggingface/diffusers/tree/main/examples/dreambooth （20 个 train 变体；flux/sd3 版 85% 相同为实测）
- https://github.com/huggingface/diffusers/blob/main/examples/README.md （examples 定位四原则）
- https://github.com/huggingface/diffusers/blob/main/utils/check_copies.py （fix-copies 只覆盖 src/）
- https://github.com/huggingface/diffusers/pull/13899 （同一 bug 修 10 个脚本）
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/lora_conversion_utils.py （26 个键名转换函数，含 Anima）
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/single_file_utils.py （CHECKPOINT_KEY_NAMES 指纹）

**ostris ai-toolkit**
- https://github.com/ostris/ai-toolkit/blob/main/toolkit/models/base_model.py （BaseModel 接口与能力 flag）
- https://github.com/ostris/ai-toolkit/blob/main/toolkit/util/get_model.py 、toolkit/extension.py 、extensions_built_in/diffusion_models/__init__.py （arch 扫描注册）
- https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/diffusion_models/example_model/README.md （官方加模型模板）
- https://github.com/ostris/ai-toolkit/blob/main/toolkit/dataloader_mixins.py 、toolkit/data_loader.py （latent_space_version 掺 hash）
- https://github.com/ostris/ai-toolkit/blob/main/jobs/process/BaseSDTrainProcess.py 、extensions_built_in/sd_trainer/SDTrainer.py （共享循环调 hook 序列）
- https://github.com/ostris/ai-toolkit/commit/99be3d96 （Krea2 添加 diff）
- https://github.com/ostris/ai-toolkit/issues/693 （Z-Image 训练质量社区批评）

**OneTrainer**
- https://github.com/Nerogar/OneTrainer/blob/master/docs/ProjectStructure.md （目录职责官方说明）
- https://github.com/Nerogar/OneTrainer/pull/1211 、pull/1498 （if/elif → 装饰器工厂重构）
- https://github.com/Nerogar/OneTrainer/pull/1487 （Anima 支持 PR：15 新文件 + 8 处横切）、pull/1195 （Z-Image）
- https://github.com/Nerogar/OneTrainer/pull/1563 、pull/1294 （LoRA 多格式导出与键名坑）
- https://github.com/Nerogar/OneTrainer/issues/1203 （维护者自述矩阵 boilerplate 痛点）
- 源码：modules/util/factory.py、modules/modelSetup/BaseModelSetup.py、BaseFluxSetup.py、modules/module/LoRAModule.py、Nerogar/mgds DiskCache.py

**tdrussell diffusion-pipe**
- https://github.com/tdrussell/diffusion-pipe （README：Windows 不可用、ComfyPipeline 转向）
- https://github.com/tdrussell/diffusion-pipe/blob/main/models/base.py （CommonPipeline/BasePipeline/ComfyPipeline 接口）
- https://github.com/tdrussell/diffusion-pipe/blob/main/train.py （if/elif 分派链、共享循环）
- https://github.com/tdrussell/diffusion-pipe/blob/main/utils/dataset.py （cache/{model.name} 目录、sdxl 特判）
- https://github.com/tdrussell/diffusion-pipe/blob/main/docs/supported_models.md （per-model 保存格式与 feature 矩阵）
- https://github.com/tdrussell/diffusion-pipe/commit/3abbfff5 （Anima 寄生 cosmos_predict2）、commit/d6a8f562 （Ideogram4）
- issues：#268（WSL2 之痛）、#313、#420、#434、#444、#488、#492、#509

**SimpleTuner**
- https://github.com/bghira/SimpleTuner/blob/main/simpletuner/helpers/models/common.py （6205 行分层基类与类属性声明面）
- https://github.com/bghira/SimpleTuner/blob/main/simpletuner/helpers/models/registry.py 及 model_metadata.json（惰性注册表）
- https://github.com/bghira/SimpleTuner/blob/main/simpletuner/helpers/caching/text_embeds.py 、caching/vae.py 、data_backend/factory.py （缓存带族名）
- https://github.com/bghira/SimpleTuner/commit/08b05a942654cd2ac006cfc741072896e8b35258 （krea2 初始 commit）
- https://github.com/bghira/SimpleTuner/releases/tag/v2.0 、AGENTS.md、issues #2601–#2607（横切扇出）

**ComfyUI**
- https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_detection.py （键名指纹 + 形状反推）
- https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/supported_models.py 、supported_models_base.py （声明类与 matches 机制）
- https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py （LatentFormat 体系、Wan21 逐通道归一）
- https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/lora.py （多生态键名集中映射）
- https://github.com/comfyanonymous/ComfyUI/commit/c012400240d4867cd63a45220eb791b91ad47617 （Qwen-Image）、commit/abe2ec26a61ff670b9c0e71e4821c873368c8728 （Anima，检测仅 +2 行）
