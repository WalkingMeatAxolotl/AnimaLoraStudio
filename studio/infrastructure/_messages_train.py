"""训练主链（runtime/training + anima_train）的用户可见 INFO 文案。

msg_id 前缀：``train.*``。终稿来源 tmp/log-text-audit/rewrite-a-training.md
的「msg_id 字典汇总」节；新增条目双语齐全，占位符用 ``{name}`` 形式。

分两组：
- **标签类**（``train.label_*`` / ``train.vram_stage_*`` / ``train.vram_ram_suffix``）
  ——被其它条目以占位符引用的可译片段，调用点先 ``msg()`` 渲染再当参数传；
- **叙事行**——一行日志一条 msg_id。

数值格式化由调用点负责（``f"{x:.2f}"`` 后作为字符串传入），字典模板只做
占位。这样译者不必理解 ``:.2f`` 这类格式串，也避免 ``str.format`` 在类型
不符时抛错。
"""
from __future__ import annotations

MESSAGES: dict[str, dict[str, str]] = {
    # ─── 标签类（被其它条目以占位符引用）───
    "train.label_training_set": {
        "zh": "训练集",
        "en": "training set",
    },
    "train.label_regularization_set": {
        "zh": "正则集",
        "en": "regularization set",
    },
    "train.vram_stage_transformer_loaded": {
        "zh": "Transformer 加载后",
        "en": "after Transformer load",
    },
    "train.vram_stage_block_swap_ready": {
        "zh": "Block 交换挂载后",
        "en": "after block swap attach",
    },
    "train.vram_stage_train_start": {
        "zh": "训练开始前",
        "en": "before training starts",
    },
    # train.vram_snapshot 的可选末段（无内存读数时整段省略）
    "train.vram_ram_suffix": {
        "zh": "，可用内存 {ram} GB",
        "en": ", RAM free {ram} GB",
    },

    # ─── 叙事行 ───
    "train.deps_installing": {
        "zh": "安装缺失依赖: {missing}",
        "en": "Installing missing dependencies: {missing}",
    },
    "train.force_exit": {
        "zh": "收到第二次中断信号: 强制退出",
        "en": "Second interrupt received: forcing exit",
    },
    "train.pause_signal": {
        "zh": "收到暂停信号: 正在退出，保留最近一次 epoch 的恢复点",
        "en": "Pause signal received: exiting, keeping the latest epoch resume state",
    },
    "train.paused_with_state": {
        "zh": "已暂停，恢复点: {path}",
        "en": "Paused, resume state: {path}",
    },
    "train.caption_json_mode": {
        "zh": "打标文本来源: JSON（按分类打乱标签顺序）",
        "en": "Caption source: JSON (tags shuffled by category)",
    },
    "train.dataset_summary": {
        "zh": "训练集: {images} 张图 → {samples} 样本（含 repeat；打标文本 JSON {json_count} / TXT {txt_count}）",
        "en": "Training set: {images} images -> {samples} samples (with repeat; captions JSON {json_count} / TXT {txt_count})",
    },
    "train.dataset_folder": {
        "zh": "文件夹 {name}: {images} 张 × repeat {repeat} × 分辨率 {resolutions} = {samples} 样本",
        "en": "Folder {name}: {images} images x repeat {repeat} x resolutions {resolutions} = {samples} samples",
    },
    "train.vae_cache_check": {
        "zh": "VAE 缓存: 检查 {dataset}",
        "en": "VAE cache: checking {dataset}",
    },
    "train.vae_cache_encode_todo": {
        "zh": "VAE 缓存: {dataset} 需编码 {todo}/{total} 张图",
        "en": "VAE cache: {dataset}, {todo}/{total} images to encode",
    },
    "train.vae_cache_all_hit": {
        "zh": "VAE 缓存: {dataset} 全部命中（{total} 张图），跳过编码",
        "en": "VAE cache: {dataset} fully cached ({total} images), encoding skipped",
    },
    "train.vae_cache_tiled_summary": {
        "zh": "VAE 缓存: {n} 张图超出单次像素预算，已改用分块编码",
        "en": "VAE cache: {n} image(s) over the per-pass pixel budget, encoded in tiles",
    },
    "train.vae_cache_progress": {
        "zh": "VAE 缓存: 编码进度 {done}/{total}",
        "en": "VAE cache: encoded {done}/{total}",
    },
    "train.res_shift_enabled": {
        "zh": "[res-shift] 分辨率修正已启用: 以 {base}px 为基准，按每张图的实际尺寸缩放 timestep shift",
        "en": "[res-shift] Resolution-aware timestep shift enabled: scaled per image against the {base}px baseline",
    },
    "train.progress": {
        "zh": "epoch={epoch}/{epochs} step={step} loss={loss} {sra}lr={lr} speed={speed} it/s",
        "en": "epoch={epoch}/{epochs} step={step} loss={loss} {sra}lr={lr} speed={speed} it/s",
    },
    "train.sampling_step": {
        "zh": "采样中 (step {step}): {prompt}",
        "en": "Sampling (step {step}): {prompt}",
    },
    "train.sampling_epoch": {
        "zh": "采样中 (epoch {epoch}): {prompt}",
        "en": "Sampling (epoch {epoch}): {prompt}",
    },
    "train.lora_saved_step": {
        "zh": "已保存 LoRA (step {step}): {path}",
        "en": "Saved LoRA (step {step}): {path}",
    },
    "train.lora_saved_epoch": {
        "zh": "已保存 LoRA (epoch {epoch}): {path}",
        "en": "Saved LoRA (epoch {epoch}): {path}",
    },
    # 注：rewrite-a §1.6 遗留观察采纳「删 loop.py 的 emit、留 state.py:82」，
    # 故 train.resume_state_saved_{step,epoch} 无调用点，未入本字典。
    "train.resume_state_saved": {
        "zh": "已保存恢复点: {path}（epoch={epoch}, step={step}）",
        "en": "Saved resume state: {path} (epoch={epoch}, step={step})",
    },
    "train.resume_state_loading": {
        "zh": "读取恢复点: {path}",
        "en": "Loading resume state: {path}",
    },
    "train.resume_done": {
        "zh": "已从恢复点继续训练: epoch={epoch}, step={step}",
        "en": "Resumed training: epoch={epoch}, step={step}",
    },
    "train.wandb_artifact_uploaded": {
        "zh": "W&B 已上传: {name}（{file}，{size} MB，{elapsed} s）",
        "en": "W&B upload done: {name} ({file}, {size} MB, {elapsed} s)",
    },
    "train.wandb_enabled": {
        "zh": "W&B 监控已启用: project={project}, run={run}, mode={mode}",
        "en": "W&B monitoring enabled: project={project}, run={run}, mode={mode}",
    },
    "train.xformers_enabled": {
        "zh": "xformers 已启用",
        "en": "xformers enabled",
    },
    "train.weights_loaded": {
        "zh": "{label} 权重加载完成: 匹配 {matched}/{total}（{coverage}）",
        "en": "{label} weights loaded: {matched}/{total} matched ({coverage})",
    },
    "train.vae_loaded": {
        "zh": "VAE 加载完成",
        "en": "VAE loaded",
    },
    "train.vram_snapshot": {
        "zh": "[vram] {stage}: torch 已分配 {alloc} GB / 保留 {reserved} GB，全卡已用 {used} GB / {total} GB{ram}",
        "en": "[vram] {stage}: torch allocated {alloc} GB / reserved {reserved} GB, device used {used} GB / {total} GB{ram}",
    },
    "train.pause_snapshot_applied": {
        "zh": "已套用暂停时保存的训练参数: {path}",
        "en": "Applied the training settings saved at pause time: {path}",
    },
    "train.sample_seed_random": {
        "zh": "采样种子设为 0，本次改用随机种子: {seed}",
        "en": "Sample seed is 0, using a random seed for this run: {seed}",
    },
    "train.config_loaded": {
        "zh": "加载配置文件: {path}",
        "en": "Loaded config file: {path}",
    },
    "train.navit_native_enabled": {
        "zh": "[navit] 原生尺寸已启用: 每张图按自身尺寸训练（不再缩放到分辨率档），token 预算 {budget}，超预算时 {policy}",
        "en": "[navit] Native resolution enabled: each image trains at its own size instead of a fixed resolution bucket, token budget {budget}, over-budget policy {policy}",
    },
    "train.masked_loss_enabled": {
        "zh": "[masked-loss] 已启用: {n}/{total} 张图带 mask（其余按整图学习）",
        "en": "[masked-loss] Enabled: {n}/{total} images have a mask (the rest train on the full image)",
    },
    "train.reg_set_summary": {
        "zh": "正则集: {path}（{samples} 样本，按文件夹 repeat{weight}{caption}）",
        "en": "Regularization set: {path} ({samples} samples, per-folder repeat{weight}{caption})",
    },
    "train.vae_selftest_saved": {
        "zh": "VAE 自检图已保存: {path}（图片经 VAE 编码再解码的还原效果）",
        "en": "VAE self-test image saved: {path} (an image encoded and then decoded again by the VAE)",
    },
    "train.attention_sdpa": {
        "zh": "attention_backend=none: 不启用 flash_attn / xformers，使用 PyTorch SDPA",
        "en": "attention_backend=none: neither flash_attn nor xformers is enabled, using PyTorch SDPA",
    },
    "train.loading_transformer": {
        "zh": "加载 Transformer",
        "en": "Loading Transformer",
    },
    "train.block_swap_active": {
        "zh": "Block 交换生效: 换出末尾 {n}/{total} 层，常驻内存 {pinned} GB",
        "en": "Block swap active: {n}/{total} tail layers offloaded, {pinned} GB pinned memory",
    },
    "train.loading_vae": {
        "zh": "加载 VAE",
        "en": "Loading VAE",
    },
    "train.loading_text_encoder": {
        "zh": "加载文本编码器",
        "en": "Loading text encoder",
    },
    "train.loading_text_encoder_file": {
        "zh": "加载文本编码器: Qwen3-VL（{path}）",
        "en": "Loading text encoder: Qwen3-VL ({path})",
    },
    "train.injecting_lora": {
        "zh": "注入 LoRA 适配器（{lora_type}）",
        "en": "Injecting LoRA adapter ({lora_type})",
    },
    "train.resume_from_lora": {
        "zh": "从已有 LoRA 继续训练: {path}",
        "en": "Continuing from an existing LoRA: {path}",
    },
    "train.fp8_base_detected": {
        "zh": "检测到 fp8 底模: 按 fp8 训练（权重以 fp8 常驻显存，前向时逐层还原成 bf16 计算）",
        "en": "fp8 base model detected: training in fp8 (weights stay in fp8 in VRAM, each layer is restored to bf16 for the forward pass)",
    },
    "train.text_cache_order": {
        "zh": "文本缓存已开启: 先加载 VAE 与文本编码器，缓存完成并释放后再加载 Transformer",
        "en": "Text cache enabled: loading the VAE and the text encoder first, then the Transformer once captions are cached and the encoder is released",
    },
    "train.text_cache_done_loading": {
        "zh": "文本缓存完成，文本编码器已释放，继续加载 Transformer",
        "en": "Text cache done, text encoder released, loading the Transformer",
    },
    "train.weight_decay": {
        "zh": "权重衰减已启用: {optimizer} weight_decay={wd}",
        "en": "Weight decay enabled: {optimizer} weight_decay={wd}",
    },
    "train.weight_decay_lokr": {
        "zh": "权重衰减已启用: {optimizer} weight_decay={wd}（LoKr 的 w1 不参与）",
        "en": "Weight decay enabled: {optimizer} weight_decay={wd} (LoKr w1 excluded)",
    },
    "train.grad_clip": {
        "zh": "梯度裁剪已启用: max_norm={value}",
        "en": "Gradient clipping enabled: max_norm={value}",
    },
    "train.step_plan": {
        "zh": "训练规模: {samples} 样本，每 epoch {steps_per_epoch} 步，共 {total_steps} 步",
        "en": "Training size: {samples} samples, {steps_per_epoch} steps per epoch, {total_steps} steps total",
    },
    "train.text_cache_off": {
        "zh": "[text-cache] 缓存已关闭: 文本编码器常驻显存，每个 batch 现场编码打标文本",
        "en": "[text-cache] Cache off: the text encoder stays in VRAM and encodes captions batch by batch",
    },
    "train.text_cache_plan": {
        "zh": "[text-cache] 预缓存 {images} 张图的打标文本 + {prompts} 条采样提示词",
        "en": "[text-cache] Caching captions for {images} images plus {prompts} sample prompts",
    },
    "train.text_cache_progress": {
        "zh": "[text-cache] 编码进度 {done}/{total}",
        "en": "[text-cache] Encoded {done}/{total}",
    },
    "train.text_cache_hits": {
        "zh": "[text-cache] 已缓存 {hit}/{total} 条打标文本，需编码 {todo} 条",
        "en": "[text-cache] {hit}/{total} captions already cached, {todo} to encode",
    },
    "train.text_cache_all_hit": {
        "zh": "[text-cache] 打标文本全部已缓存（{n} 条），跳过编码",
        "en": "[text-cache] All {n} captions already cached, encoding skipped",
    },
    "train.loss_curve": {
        "zh": "Loss 曲线（前 {n} 步）:",
        "en": "Loss curve (first {n} steps):",
    },
    "train.finished": {
        "zh": "训练完成，最终 LoRA: {path}",
        "en": "Training finished, final LoRA: {path}",
    },
    "train.monitor_history_restored": {
        "zh": "监控面板历史已恢复: {n} 个 loss 点",
        "en": "Dashboard history restored: {n} loss points",
    },
    "train.baseline_sampling": {
        "zh": "采样中 (step 0，基线)",
        "en": "Sampling (step 0, baseline)",
    },
    "train.baseline_sampling_skipped": {
        "zh": "跳过基线采样: 从 step {step} 恢复，不是从 step 0 开始",
        "en": "Baseline sampling skipped: resumed at step {step}, not from step 0",
    },
    "train.lr_schedule": {
        "zh": "学习率调度: {detail}",
        "en": "LR schedule: {detail}",
    },
    "train.infonoise_warmup_auto": {
        "zh": "InfoNoise 预热步数自动设为 {steps} 步（总步数 {total} 的 20%）",
        "en": "InfoNoise warmup set to {steps} steps (20% of {total} total steps)",
    },
    "train.infonoise_warmup_floor": {
        "zh": "InfoNoise 预热步数自动设为下限 {steps} 步（总步数 {total} 的 20% 不足 {steps}）",
        "en": "InfoNoise warmup set to the {steps}-step minimum (20% of {total} total steps is below it)",
    },
    "train.infonoise_warmup_unknown_total": {
        "zh": "InfoNoise 预热步数自动设为 {steps} 步（总步数未知，按 5000 步估算）",
        "en": "InfoNoise warmup set to {steps} steps (total step count unknown, estimated from 5000)",
    },
    "train.infonoise_enabled": {
        "zh": "InfoNoise 已启用: 训练中自适应调整 timestep 采样分布",
        "en": "InfoNoise enabled: the timestep sampling distribution adapts during training",
    },
    "train.flash_attn_on": {
        "zh": "flash_attn 已启用（训练与采样都使用）",
        "en": "flash_attn enabled (used for both training and sampling)",
    },
    "train.flash_attn_off_by_setting": {
        "zh": "flash_attn 未启用: attention_backend={backend}",
        "en": "flash_attn not enabled: attention_backend={backend}",
    },
    "train.flash_attn_off_missing": {
        "zh": "flash_attn 未启用: 未安装 flash_attn，使用 PyTorch SDPA",
        "en": "flash_attn not enabled: the flash_attn package is not installed, using PyTorch SDPA",
    },
    "train.anima_model_loaded": {
        "zh": "Anima 模型加载完成: {channels}ch，{blocks} 层",
        "en": "Anima model loaded: {channels}ch, {blocks} blocks",
    },
    "train.text_encoder_loaded": {
        "zh": "文本编码器加载完成",
        "en": "Text encoder loaded",
    },
    "train.krea2_sdpa_only": {
        "zh": "Krea2 固定使用 PyTorch SDPA: 已忽略 attention_backend={backend}",
        "en": "Krea2 always uses PyTorch SDPA: attention_backend={backend} ignored",
    },
    "train.fp8_merge_done": {
        "zh": "LoRA 已合并进 fp8 底模: {loras} 份 LoRA → {layers} 个层",
        "en": "LoRA merged into the fp8 base model: {loras} LoRA(s) into {layers} layers",
    },
}
