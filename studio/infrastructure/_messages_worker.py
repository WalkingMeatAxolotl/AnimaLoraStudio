"""子进程域（daemon / generate / reg_ai / utils / workers）的用户可见 INFO 文案。

msg_id 前缀：``daemon.* generate.* regai.* worker.<name>.* optim.* model.*``。
终稿来源 tmp/log-text-audit/rewrite-b-subprocess.md 的「msg_id 字典汇总」节。

占位符两语言同名同数（约定 C4）；``key=value`` 的 key 与主题 tag 一律英文小写
不翻译（C1/C2），保证一条 grep 同时命中中英两种日志。
"""
from __future__ import annotations

MESSAGES: dict[str, dict[str, str]] = {
    # --- model.*：三个 runtime 入口共用的模型加载/编排 ---------------------
    "model.load_text_encoder": {
        "zh": "加载文本编码器: {path}",
        "en": "Loading text encoder: {path}",
    },
    "model.load_text_encoder_ahead": {
        "zh": "加载文本编码器（先于 Transformer，用于预编码）: {path}",
        "en": "Loading text encoder ahead of the transformer for pre-encoding: {path}",
    },
    "model.load_transformer": {
        "zh": "加载 Transformer（{family}）: {path}",
        "en": "Loading transformer ({family}): {path}",
    },
    "model.load_vae": {
        "zh": "加载 VAE",
        "en": "Loading VAE",
    },
    "model.block_swap_enabled": {
        "zh": "Block 交换: 后 {n} 层留在内存，用到时再搬进显存",
        "en": "Block swap: last {n} blocks stay in RAM, moved to VRAM on demand",
    },
    "model.unloaded": {
        "zh": "已卸载模型: 可用内存 {ram} GB",
        "en": "Model unloaded: {ram} GB RAM available",
    },

    # --- daemon.* ---------------------------------------------------------
    "daemon.ready": {
        "zh": "出图服务已就绪，等待任务",
        "en": "Generation daemon ready, waiting for tasks",
    },
    "daemon.dit_yields_for_text_encoder": {
        "zh": "预编码 prompt: 先把 Transformer 移到内存腾显存（显存策略 {policy}）",
        "en": (
            "Pre-encoding prompts: moved the transformer to RAM first to free VRAM "
            "(VRAM policy {policy})"
        ),
    },

    # --- generate.*：daemon 与 anima_generate 共用 -------------------------
    "generate.prompts_precached_released": {
        "zh": "已预编码 {n} 条 prompt: 文本编码器已释放",
        "en": "Pre-encoded {n} prompts: text encoder released",
    },
    "generate.prompts_precached_resident": {
        "zh": "已预编码 {n} 条 prompt: 文本编码器保持驻留（显存策略 performance）",
        "en": "Pre-encoded {n} prompts: text encoder stays resident (VRAM policy performance)",
    },
    "generate.start": {
        "zh": "开始出图: {prompts} 个 prompt × {count} 次 = {total} 张",
        "en": "Generating {total} images: {prompts} prompts × {count} each",
    },
    "generate.image_start": {
        "zh": "出图 {idx}/{total}: seed={seed} prompt={prompt}",
        "en": "Image {idx}/{total}: seed={seed} prompt={prompt}",
    },
    "generate.image_saved": {
        "zh": "已保存: {path}",
        "en": "Saved: {path}",
    },
    "generate.done": {
        "zh": "出图完成: {ok}/{total} 张",
        "en": "Generation finished: {ok}/{total} images",
    },
    "generate.canceled": {
        "zh": "出图已取消: task={task_id}",
        "en": "Generation canceled: task={task_id}",
    },
    "generate.xy_shared_seed": {
        "zh": "XY 共享种子: {seed}（seed 设 0 时随机取一个）",
        "en": "XY grid shared seed: {seed} (randomized because seed was 0)",
    },
    "generate.xy_start": {
        "zh": "开始 XY 出图: {nx}×{ny} = {total} 张",
        "en": "Generating XY grid: {nx}×{ny} = {total} images",
    },
    "generate.xy_cell": {
        "zh": (
            "XY 第 {xi},{yi} 格: x={xv} y={yv} steps={steps} cfg={cfg} "
            "seed={seed} sampler={sampler}"
        ),
        "en": (
            "XY cell {xi},{yi}: x={xv} y={yv} steps={steps} cfg={cfg} "
            "seed={seed} sampler={sampler}"
        ),
    },
    "generate.xy_done": {
        "zh": "XY 出图完成: {ok}/{total} 张",
        "en": "XY grid finished: {ok}/{total} images",
    },

    # --- regai.*：正则集 AI 生成 -------------------------------------------
    "regai.train_scanned": {
        "zh": "训练集共 {n} 张图",
        "en": "Train set: {n} images",
    },
    "regai.incremental_plan": {
        "zh": "增量模式: 需生成 {todo}/{total} 张",
        "en": "Incremental mode: {todo}/{total} images to generate",
    },
    "regai.nothing_to_do": {
        "zh": "每张训练图都已有正则图，无需生成",
        "en": "Every train image already has a regularization image; nothing to generate",
    },
    "regai.full_mode_clear": {
        "zh": "全量模式: 已清空正则集目录的旧内容",
        "en": "Full mode: cleared the existing regularization images",
    },
    "regai.precache_strategy": {
        "zh": "按批预编码 caption: 每批 {batch} 条，编码后释放文本编码器",
        "en": (
            "Pre-encoding captions in batches of {batch}; the text encoder is "
            "released after each batch"
        ),
    },
    "regai.progress": {
        "zh": "已生成 {done}/{total} 张",
        "en": "Generated {done}/{total} images",
    },
    "regai.done": {
        "zh": "正则集生成完成: {ok}/{total} 张",
        "en": "Regularization images finished: {ok}/{total}",
    },

    # --- lora.*：注入 / 保存 / 加载 ----------------------------------------
    "lora.injected": {
        "zh": "已注入 {algo} 到 {n} 层（{detail}）",
        "en": "Injected {algo} into {n} layers ({detail})",
    },
    "lora.tlora_mask_enabled": {
        "zh": "T-LoRA 时间步 rank 掩码已启用: {n}/{total} 层（min_rank={min_rank} alpha_rank_scale={scale}）",
        "en": (
            "T-LoRA timestep rank mask enabled: {n}/{total} layers "
            "(min_rank={min_rank} alpha_rank_scale={scale})"
        ),
    },
    "lora.saved": {
        "zh": "LoRA 已保存: {path}",
        "en": "LoRA saved: {path}",
    },
    "lora.saved_baked": {
        "zh": "LoRA 已保存（OrthoLoRA 已烘焙成普通 LoRA）: {path}",
        "en": "LoRA saved (OrthoLoRA baked into a plain LoRA): {path}",
    },
    "lora.loaded": {
        "zh": "已加载 LoRA 权重: {path}",
        "en": "Loaded LoRA weights: {path}",
    },
    "lora.reg_dims_applied": {
        "zh": "已按 lora_reg_dims 覆盖 {n} 个模块的 rank",
        "en": "Applied lora_reg_dims rank overrides to {n} modules",
    },

    # --- optim.* -----------------------------------------------------------
    "optim.created": {
        "zh": "优化器 {name} 已创建: {params}",
        "en": "Optimizer {name} created: {params}",
    },
    "optim.trainable_params": {
        "zh": "可训练参数: {tensors} 个张量 / {elements} 个元素",
        "en": "Trainable parameters: {tensors} tensors / {elements} elements",
    },

    # --- caption.* ---------------------------------------------------------
    "caption.convert_done": {
        "zh": "已转换 {n} 个 caption 文件",
        "en": "Converted {n} caption files",
    },

    # --- worker.download.* -------------------------------------------------
    "worker.download.start": {
        "zh": "开始下载: tag={tag} count={count} source={source} exclude={exclude}",
        "en": "Download started: tag={tag} count={count} source={source} exclude={exclude}",
    },
    "worker.download.done": {
        "zh": "下载完成: 已保存 {saved} 张",
        "en": "Download finished: {saved} images saved",
    },

    # --- worker.tag.* ------------------------------------------------------
    "worker.tag.start": {
        "zh": "开始打标: tagger={tagger} version={version} images={total} on_existing={mode}",
        "en": "Tagging started: tagger={tagger} version={version} images={total} on_existing={mode}",
    },
    "worker.tag.no_images": {
        "zh": "没有需要打标的图（范围 {scope}）",
        "en": "No images to tag in scope {scope}",
    },
    "worker.tag.trigger_word": {
        "zh": "触发词 {word} 会写在每张 caption 的第一位",
        "en": "Trigger word {word} is written first in every caption",
    },
    "worker.tag.overrides": {
        "zh": "本次打标参数覆盖: {overrides}",
        "en": "Tagger options overridden for this run: {overrides}",
    },
    "worker.tag.ready": {
        "zh": "{tagger} 模型已就绪",
        "en": "{tagger} model ready",
    },
    "worker.tag.progress": {
        "zh": "已打标 {done}/{total}",
        "en": "Tagged {done}/{total}",
    },
    "worker.tag.done": {
        "zh": "打标完成: {done}/{total} 张（跳过 {skipped}，失败 {errors}）",
        "en": "Tagging finished: {done}/{total} images (skipped {skipped}, failed {errors})",
    },

    # --- worker.regbuild.* -------------------------------------------------
    "worker.regbuild.start": {
        "zh": (
            "开始构建正则集: version={version} source={source} max_tags={max_tags} "
            "auto_tag={auto_tag} incremental={incremental} auto_dedup={auto_dedup} "
            "postprocess={pp_method}/{pp_max_crop}"
        ),
        "en": (
            "Regularization build started: version={version} source={source} "
            "max_tags={max_tags} auto_tag={auto_tag} incremental={incremental} "
            "auto_dedup={auto_dedup} postprocess={pp_method}/{pp_max_crop}"
        ),
    },
    "worker.regbuild.built": {
        "zh": "正则集已构建: {actual}/{target} 张",
        "en": "Regularization set built: {actual}/{target} images",
    },
    "worker.regbuild.autotag_no_images": {
        "zh": "[auto-tag] 正则集没有图，跳过打标",
        "en": "[auto-tag] no regularization images to tag, skipping",
    },
    "worker.regbuild.autotag_start": {
        "zh": "[auto-tag] 开始给正则集打标: tagger={tagger} images={total}",
        "en": "[auto-tag] tagging the regularization set: tagger={tagger} images={total}",
    },
    "worker.regbuild.autotag_ready": {
        "zh": "[auto-tag] {tagger} 模型已就绪",
        "en": "[auto-tag] {tagger} model ready",
    },
    "worker.regbuild.autotag_progress": {
        "zh": "[auto-tag] 已打标 {done}/{total}",
        "en": "[auto-tag] tagged {done}/{total}",
    },
    "worker.regbuild.autotag_done": {
        "zh": "[auto-tag] 打标完成: {done}/{total} 张（失败 {errors}）",
        "en": "[auto-tag] tagging finished: {done}/{total} (failed {errors})",
    },
    "worker.regbuild.dedup_scan": {
        "zh": "[dedup] 第 {round}/{rounds} 轮: 扫描重复图",
        "en": "[dedup] round {round}/{rounds}: scanning for duplicates",
    },
    "worker.regbuild.dedup_none": {
        "zh": "[dedup] 第 {round} 轮: 没有可删的重复图，结束去重",
        "en": "[dedup] round {round}: no duplicates left, deduplication done",
    },
    "worker.regbuild.dedup_purged": {
        "zh": "[dedup] 第 {round} 轮: 已删 {count} 张重复图",
        "en": "[dedup] round {round}: removed {count} duplicates",
    },
    "worker.regbuild.dedup_target_met": {
        "zh": "[dedup] 第 {round} 轮: 已达目标张数，结束去重",
        "en": "[dedup] round {round}: target count reached, deduplication done",
    },
    "worker.regbuild.dedup_refill": {
        "zh": "[dedup] 第 {round} 轮: 还缺 {shortfall} 张，按增量模式补足",
        "en": "[dedup] round {round}: {shortfall} images short, refilling incrementally",
    },
    "worker.regbuild.dedup_refilled": {
        "zh": "[dedup] 第 {round} 轮: 补足后 {actual}/{target} 张",
        "en": "[dedup] round {round}: {actual}/{target} images after refill",
    },

    # --- worker.preprocess.* -----------------------------------------------
    "worker.preprocess.no_images": {
        "zh": "没有需要处理的图",
        "en": "No images to process",
    },
    "worker.preprocess.upscale_start": {
        "zh": (
            "开始放大: mode={mode} model={model} tile={tile}+{pad} device={device} "
            "target={target} images={total}"
        ),
        "en": (
            "Upscaling started: mode={mode} model={model} tile={tile}+{pad} "
            "device={device} target={target} images={total}"
        ),
    },
    "worker.preprocess.model_ready": {
        "zh": "放大器 {model} 已加载到 {device}",
        "en": "Upscaler {model} loaded on {device}",
    },
    "worker.preprocess.progress": {
        "zh": "已处理 {done}/{total}",
        "en": "Processed {done}/{total}",
    },
    "worker.preprocess.upscale_done": {
        "zh": "放大完成: 成功 {succeeded}，失败 {failed}，跳过 {skipped}",
        "en": "Upscaling finished: succeeded={succeeded} failed={failed} skipped={skipped}",
    },
    "worker.preprocess.no_crops": {
        "zh": "没有裁剪框，无需处理",
        "en": "No crop boxes to apply",
    },
    "worker.preprocess.crop_start": {
        "zh": "开始裁剪: images={total}",
        "en": "Cropping started: images={total}",
    },
    "worker.preprocess.crop_done": {
        "zh": "裁剪完成: 成功 {succeeded}，失败 {failed}，跳过 {skipped}",
        "en": "Cropping finished: succeeded={succeeded} failed={failed} skipped={skipped}",
    },

    # --- worker.eval.* -----------------------------------------------------
    "worker.eval.start": {
        "zh": "开始评估: session={session} candidates={n}（含基线） stages={stages}",
        "en": "Evaluation started: session={session} candidates={n} (baseline included) stages={stages}",
    },
    "worker.eval.generate_all_done": {
        "zh": "[generate] 全部候选都已出图，跳过出图阶段",
        "en": "[generate] every candidate already has samples, skipping the generation stage",
    },
    "worker.eval.candidate_skipped": {
        "zh": "[generate] {label} 已出图，跳过",
        "en": "[generate] {label} already generated, skipping",
    },
    "worker.eval.candidate_start": {
        "zh": "[generate] {label} 开始出图: run={run_id}",
        "en": "[generate] {label} generating: run={run_id}",
    },
    "worker.eval.candidate_done": {
        "zh": "[generate] {label} 出图完成: {done}/{total}",
        "en": "[generate] {label} generated {done}/{total}",
    },
    "worker.eval.metric_start": {
        "zh": "[{stage}] {label} 开始算指标: run={run_id}",
        "en": "[{stage}] {label} scoring: run={run_id}",
    },
    "worker.eval.done": {
        "zh": "评估完成: session={session} candidates={n} metrics_done={m}",
        "en": "Evaluation finished: session={session} candidates={n} metrics_done={m}",
    },
}
