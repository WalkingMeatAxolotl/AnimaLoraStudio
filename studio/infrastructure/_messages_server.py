"""server 用户面（cli / 下载卡 / updater / services 进度回调）的用户可见 INFO 文案。

msg_id 前缀：``cli.* download.* modelsrc.* upscale.* update.* eval.* booru.* reg.*``。
终稿来源 tmp/log-text-audit/rewrite-c-server.md 的「msg_id 字典汇总」节。

``upscale.*`` 本轮 0 条：``inference/upscaler.py`` 的两条 ``on_log`` 都判 DEBUG，
按 Q1 走英文排障行，不进字典。
"""
from __future__ import annotations

MESSAGES: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------------ cli.*
    "cli.npm_install_stale": {
        "zh": "package.json/package-lock.json 比 node_modules 新，运行 npm install（可能需要几分钟）",
        "en": "package.json/package-lock.json are newer than node_modules; "
              "running npm install (this can take a few minutes)",
    },
    "cli.npm_install_missing": {
        "zh": "{rel} 不存在或不完整，运行 npm install（超时 3 分钟）",
        "en": "{rel} is missing or incomplete; running npm install (3 min timeout)",
    },
    "cli.reinstall_python_deps": {
        "zh": "未检测到 fastapi，重新安装 Python 依赖（requirements.txt，可能需要几分钟）",
        "en": "fastapi is missing; reinstalling Python dependencies from "
              "requirements.txt (this can take a few minutes)",
    },
    "cli.build_frontend": {
        "zh": "构建前端（npm run build）",
        "en": "Building the frontend (npm run build)",
    },
    "cli.process_spawned": {
        "zh": "{label} 已启动：pid={pid} cmd={cmd}",
        "en": "{label} started: pid={pid} cmd={cmd}",
    },
    "cli.process_exited": {
        "zh": "{label} 已退出：rc=0",
        "en": "{label} exited: rc=0",
    },
    "cli.stopping_process": {
        "zh": "正在停止 {label}",
        "en": "Stopping {label}",
    },
    "cli.flash_attn_enabled": {
        "zh": "flash_attn 已启用",
        "en": "flash_attn enabled",
    },
    "cli.torch_gpu": {
        "zh": "torch {version}（GPU：{name}）",
        "en": "torch {version} (GPU: {name})",
    },
    "cli.torch_cpu_only": {
        "zh": "torch {version}（CPU-only 构建，未检测到 NVIDIA GPU）",
        "en": "torch {version} (CPU-only build, no NVIDIA GPU detected)",
    },
    "cli.onnxruntime_gpu": {
        "zh": "onnxruntime：{installed}=={ver}（CUDA EP 可用）",
        "en": "onnxruntime: {installed}=={ver} (CUDA EP available)",
    },
    "cli.onnxruntime_cpu": {
        "zh": "onnxruntime：{installed}=={ver}（仅 CPU，未检测到 NVIDIA GPU）",
        "en": "onnxruntime: {installed}=={ver} (CPU only, no NVIDIA GPU detected)",
    },
    "cli.torch_already_tag": {
        "zh": "torch 已是 {tag}，跳过重装",
        "en": "torch is already {tag}; reinstall skipped",
    },
    "cli.torch_reinstall_start": {
        "zh": "--torch {tag} 已指定（当前 {current_build}），开始重装\n  按 Ctrl+C 可跳过",
        "en": "--torch {tag} requested (current: {current_build}); reinstalling\n"
              "  press Ctrl+C to skip",
    },
    "cli.torch_reinstall_done": {
        "zh": "torch 重装完成：{version}（{tag}）",
        "en": "torch reinstall done: {version} ({tag})",
    },
    "cli.frontend_dist_missing": {
        "zh": "studio/web/dist 不存在，先构建前端",
        "en": "studio/web/dist is missing; building the frontend first",
    },
    "cli.frontend_dist_stale": {
        "zh": "studio/web/dist 比 src 旧（git pull 后未重建？），重新构建前端",
        "en": "studio/web/dist is older than src (not rebuilt after a git pull?); "
              "rebuilding the frontend",
    },
    "cli.backend_started": {
        "zh": "后端已启动 → {url}",
        "en": "Backend started → {url}",
    },
    "cli.stopped_ctrl_c": {
        "zh": "已停止（Ctrl+C）",
        "en": "Stopped (Ctrl+C)",
    },
    "cli.launcher_reload": {
        "zh": "launcher 文件有更新（cli.py / studio.sh / studio.bat），以退出码 42 让 wrapper 重新加载",
        "en": "launcher files changed (cli.py / studio.sh / studio.bat); "
              "exiting with code 42 so the wrapper reloads",
    },
    "cli.restart_requested": {
        "zh": "收到重启请求，正在重启",
        "en": "Restart requested; restarting",
    },
    "cli.dev_urls": {
        "zh": "前端 → {frontend_url}　后端 → {backend_url}",
        "en": "frontend → {frontend_url}　backend → {backend_url}",
    },
    "cli.pytest_start": {
        "zh": "运行 pytest",
        "en": "Running pytest",
    },
    "cli.vitest_skipped_no_npm": {
        "zh": "跳过 vitest（未安装 npm）",
        "en": "vitest skipped (npm is not installed)",
    },
    "cli.vitest_skipped_no_modules": {
        "zh": "跳过 vitest（node_modules 缺失，先运行 npm install）",
        "en": "vitest skipped (node_modules is missing; run npm install first)",
    },
    "cli.vitest_start": {
        "zh": "运行 vitest",
        "en": "Running vitest",
    },

    # -------------------------------------------------------------- install.*
    "install.torch_reinstall_registered": {
        "zh": "已注册 torch 重装: target={target}（下次启动时执行）",
        "en": "torch reinstall registered: target={target} (runs on the next start)",
    },
    "install.torch_reinstall_start": {
        "zh": "检测到待执行的 torch 重装: target={target}，开始安装\n"
              "  按 Ctrl+C 可跳过本次安装（marker 保留，下次启动重试）\n"
              "  若希望永久跳过，删除 marker 文件: {marker}",
        "en": "pending torch reinstall detected: target={target}; installing now\n"
              "  press Ctrl+C to skip this run (the marker is kept and retried "
              "on the next start)\n"
              "  to skip permanently, delete the marker file: {marker}",
    },
    "install.torch_reinstall_done": {
        "zh": "torch 重装完成: {version}（{tag}）",
        "en": "torch reinstall done: version={version} tag={tag}",
    },

    # ---------------------------------------------------------------- booru.*
    "booru.canceled": {
        "zh": "下载已停止：用户请求",
        "en": "Download stopped: user request",
    },
    "booru.no_more_posts": {
        "zh": "已到最后一页：服务器返回空页",
        "en": "Reached the last page: the server returned an empty page",
    },
    "booru.saved": {
        "zh": "[{n}/{total}] 已保存 {name}",
        "en": "[{n}/{total}] saved {name}",
    },
    "booru.page_short_end": {
        "zh": "已到最后一页：本页 {n} 条 < 每页上限 {limit}",
        "en": "Reached the last page: {n} posts < page limit {limit}",
    },
    "booru.no_valid_posts": {
        "zh": "本页没有可用图片，停止",
        "en": "No usable posts on this page; stopping",
    },
    "booru.summary": {
        "zh": "下载完成：已保存 {saved}，已跳过 {skipped}",
        "en": "Download finished: saved={saved} skipped={skipped}",
    },

    # ------------------------------------------------------------------ reg.*
    "reg.build_start": {
        "zh": "[reg] 正则集构建开始：来源 {api}，训练集 {train_dir}",
        "en": "[reg] Regularization set build started: source {api}, "
              "training set {train_dir}",
    },
    "reg.source_ids": {
        "zh": "[reg] 训练集图片 ID 共 {n} 个，用于去重",
        "en": "[reg] Collected {n} training set image IDs for de-duplication",
    },
    "reg.mode_flat": {
        "zh": "[reg] flat 模式：目标 {total} 张，全部放在 1_data/",
        "en": "[reg] Flat mode: target {total} images, all in 1_data/",
    },
    "reg.mode_mirror": {
        "zh": "[reg] mirror 模式：目标 {total} 张（训练集共 {train_total} 张），镜像 {n} 个子文件夹",
        "en": "[reg] Mirror mode: target {total} images (training set has "
              "{train_total}), mirroring {n} subfolders",
    },
    "reg.mode_incremental": {
        "zh": "[reg] incremental 模式：已有 {existing} 张、{n} 个子文件夹",
        "en": "[reg] Incremental mode: {existing} existing images across {n} subfolders",
    },
    "reg.deleted_ids_excluded_total": {
        "zh": "[reg] 已排除 booru 上已删除的 ID 共 {n} 个",
        "en": "[reg] Excluded {n} post IDs that were deleted on booru in total",
    },
    "reg.subfolder_start": {
        "zh": "子文件夹开始：{label}",
        "en": "Subfolder started: {label}",
    },
    "reg.subfolder_plan": {
        "zh": "目标 {target} 张，批次 {batch}，最多 {max_tags} 个 tag",
        "en": "Target {target} images, batch {batch}, up to {max_tags} tags",
    },
    "reg.deleted_ids_excluded_sub": {
        "zh": "已排除 booru 上已删除的 {n} 个 ID",
        "en": "Excluded {n} post IDs that were deleted on booru",
    },
    "reg.incremental_reuse": {
        "zh": "增量模式：沿用已有 {existing} 张（起点 {downloaded}/{target}）",
        "en": "Incremental mode: reusing {existing} existing images "
              "(starting at {downloaded}/{target})",
    },
    "reg.incremental_done": {
        "zh": "增量模式：已有图片已达目标，无需补足",
        "en": "Incremental mode: the existing images already meet the target",
    },
    "reg.canceled": {
        "zh": "构建已停止：用户请求",
        "en": "Build stopped: user request",
    },
    "reg.weights_reached": {
        "zh": "所有标签已达目标权重",
        "en": "All tags reached their target weight",
    },
    "reg.image_saved": {
        "zh": "[{n}/{target}] 已保存 {post_id}",
        "en": "[{n}/{target}] saved {post_id}",
    },
    "reg.total_limit_reached": {
        "zh": "已达总数量限制 {total}",
        "en": "Total image limit {total} reached",
    },
    "reg.subfolder_done": {
        "zh": "子文件夹 {label} 完成：{n}/{target}（已跳过 {skipped}）",
        "en": "Subfolder {label} finished: {n}/{target} (skipped {skipped})",
    },
    "reg.total_target_reached": {
        "zh": "[reg] 已达总目标 {total}，跳过剩余子文件夹",
        "en": "[reg] Total target {total} reached; the remaining subfolders are skipped",
    },
    "reg.build_done": {
        "zh": "[reg] 正则集构建完成：{n}/{total} 张（{ok}/{subs} 个子文件夹达 80%）",
        "en": "[reg] Regularization set build finished: {n}/{total} images "
              "({ok}/{subs} subfolders reached 80%)",
    },
    "reg.pp_collect": {
        "zh": "[postprocess] 收集图片（方式 {method}，最大裁剪 {max_crop}）",
        "en": "[postprocess] Collecting images (method {method}, max crop {max_crop})",
    },
    "reg.pp_no_images": {
        "zh": "[postprocess] 没有图片，跳过",
        "en": "[postprocess] No images; skipped",
    },
    "reg.pp_image_count": {
        "zh": "[postprocess] 共 {n} 张图片",
        "en": "[postprocess] {n} images",
    },
    "reg.pp_clusters": {
        "zh": "[postprocess] 聚为 {n} 个分辨率组",
        "en": "[postprocess] Grouped into {n} resolution clusters",
    },
    "reg.pp_cluster_start": {
        "zh": "[postprocess] 处理分辨率组 {cid}（{n} 张）→ {target}",
        "en": "[postprocess] Processing resolution cluster {cid} ({n} images) → {target}",
    },
    "reg.pp_canceled": {
        "zh": "[postprocess] 后处理已停止：用户请求",
        "en": "[postprocess] Postprocess stopped: user request",
    },
    "reg.pp_done": {
        "zh": "[postprocess] 后处理完成：已处理 {processed} 张，分辨率组 {clusters} 个",
        "en": "[postprocess] Postprocess finished: processed={processed} "
              "clusters={clusters}",
    },
    "reg.analysis_subfolder": {
        "zh": "[{key}] {n} 张图片，{tags} 种 tag",
        "en": "[{key}] {n} images, {tags} distinct tags",
    },

    # ----------------------------------------------------------------- eval.*
    "eval.ccip_start": {
        "zh": "[eval-ccip] 开始评分：run={run_id} 模型={model}",
        "en": "[eval-ccip] Scoring: run={run_id} model={model}",
    },
    "eval.ccip_loading": {
        "zh": "[eval-ccip] 加载 CCIP 模型（阈值 {threshold}）",
        "en": "[eval-ccip] Loading the CCIP model (threshold {threshold})",
    },
    "eval.ccip_extract": {
        "zh": "[eval-ccip] 提取 {n} 对图片的特征",
        "en": "[eval-ccip] Extracting features for {n} image pairs",
    },
    "eval.ccip_done": {
        "zh": "[eval-ccip] 角色一致性完成：{status}",
        "en": "[eval-ccip] Character consistency done: {status}",
    },
    "eval.clip_start": {
        "zh": "[eval-clip] 开始评分：run={run_id} 模型={model}",
        "en": "[eval-clip] Scoring: run={run_id} model={model}",
    },
    "eval.clip_loading": {
        "zh": "[eval-clip] 在 {device} 上加载 CLIP",
        "en": "[eval-clip] Loading CLIP on {device}",
    },
    "eval.clip_encoding": {
        "zh": "[eval-clip] 编码{label}图片 {start}-{end}",
        "en": "[eval-clip] Encoding {label} images {start}-{end}",
    },
    "eval.clip_done": {
        "zh": "[eval-clip] 完成：图文匹配 {clip_t}，图像相似度 {clip_i}",
        "en": "[eval-clip] Done: prompt match {clip_t}, image similarity {clip_i}",
    },
    "eval.dino_start": {
        "zh": "[eval-dino] 开始评分：run={run_id} 模型={model}",
        "en": "[eval-dino] Scoring: run={run_id} model={model}",
    },
    "eval.dino_loading": {
        "zh": "[eval-dino] 在 {device} 上加载 DINO",
        "en": "[eval-dino] Loading DINO on {device}",
    },
    "eval.dino_encoding": {
        "zh": "[eval-dino] 编码{label}图片 {start}-{end}",
        "en": "[eval-dino] Encoding {label} images {start}-{end}",
    },
    "eval.dino_done": {
        "zh": "[eval-dino] 图像相似度完成：{status}",
        "en": "[eval-dino] Image similarity done: {status}",
    },
    "eval.tag_start": {
        "zh": "[eval-tag] 开始评分：run={run_id}（WD14 tag 召回）",
        "en": "[eval-tag] Scoring: run={run_id} (WD14 tag recall)",
    },
    "eval.tag_loading": {
        "zh": "[eval-tag] 加载 WD14",
        "en": "[eval-tag] Loading WD14",
    },
    "eval.tag_tagging": {
        "zh": "[eval-tag] 用 WD14 给 {n} 张生成图打标",
        "en": "[eval-tag] Tagging {n} generated images with WD14",
    },
    "eval.tag_progress": {
        "zh": "[eval-tag] {done}/{total}",
        "en": "[eval-tag] {done}/{total}",
    },
    "eval.tag_done": {
        "zh": "[eval-tag] tag 召回完成：{status}",
        "en": "[eval-tag] Tag recall done: {status}",
    },
    "eval.samples_daemon_ready": {
        "zh": "[eval-samples] 出图服务就绪，底模只加载一次",
        "en": "[eval-samples] Generation service ready; the base model is loaded once",
    },
    "eval.samples_progress": {
        "zh": "[eval-samples] {n}/{total} prompt={prompt}",
        "en": "[eval-samples] {n}/{total} prompt={prompt}",
    },
    "eval.samples_daemon_stopped": {
        "zh": "[eval-samples] 出图服务已退出，显存已释放",
        "en": "[eval-samples] Generation service stopped; VRAM released",
    },

    # ------------------------------------------------------------- download.*
    "download.anima_base": {
        "zh": "下载 Anima 主模型 [{variant}]（约 {size} GB）→ {target}",
        "en": "Downloading Anima base model [{variant}] (~{size} GB) → {target}",
    },
    "download.anima_vae": {
        "zh": "下载 Anima VAE（约 {size} MB）→ {target}",
        "en": "Downloading Anima VAE (~{size} MB) → {target}",
    },
    "download.anima_te_ms": {
        "zh": "下载 Anima 文本编码器（ModelScope 权重 + HuggingFace tokenizer）→ {target}",
        "en": "Downloading the Anima text encoder (ModelScope weights + "
              "HuggingFace tokenizer) → {target}",
    },
    "download.anima_te_hf": {
        "zh": "下载 Anima 文本编码器 Qwen3-0.6B-Base（约 {size} GB）→ {target}",
        "en": "Downloading the Anima text encoder Qwen3-0.6B-Base (~{size} GB) → {target}",
    },
    "download.krea2_base": {
        "zh": "下载 Krea 2 [{variant}]（约 {size} GB）→ {target}",
        "en": "Downloading Krea 2 [{variant}] (~{size} GB) → {target}",
    },
    "download.krea2_te": {
        "zh": "下载 Krea 2 文本编码器 Qwen3-VL-4B-Instruct（约 {size} GB，来源 {source}）→ {target}",
        "en": "Downloading Krea 2 text encoder Qwen3-VL-4B-Instruct "
              "(~{size} GB, source {source}) → {target}",
    },
    "download.krea2_te_fp8": {
        "zh": "下载 Krea 2 文本编码器 Qwen3-VL fp8（约 {size} GB，来源 {source}）→ {target}",
        "en": "Downloading Krea 2 text encoder Qwen3-VL fp8 "
              "(~{size} GB, source {source}) → {target}",
    },
    "download.t5_tokenizer": {
        "zh": "下载 T5 tokenizer（{n} 个文件）→ {target}",
        "en": "Downloading the T5 tokenizer ({n} files) → {target}",
    },
    "download.cltagger": {
        "zh": "下载 CLTagger → {target}",
        "en": "Downloading CLTagger → {target}",
    },
    "download.upscaler": {
        "zh": "下载放大模型 {label}（约 {size} MB）→ {target}",
        "en": "Downloading upscaler {label} (~{size} MB) → {target}",
    },
    "download.custom_upscaler": {
        "zh": "下载自定义放大模型 {repo_id}/{subpath}（来源 {source}）→ {target}",
        "en": "Downloading custom upscaler {repo_id}/{subpath} (source {source}) → {target}",
    },
    "download.custom_base": {
        "zh": "下载自定义主模型 {repo_id}/{filename} → {target}",
        "en": "Downloading custom base model {repo_id}/{filename} → {target}",
    },
    "download.wd14": {
        "zh": "下载 WD14 {model_id} → {target}",
        "en": "Downloading WD14 {model_id} → {target}",
    },
    "download.wd14_via_ms": {
        "zh": "下载 WD14 {model_id}（经 ModelScope：{ms_repo}）→ {target}",
        "en": "Downloading WD14 {model_id} (via ModelScope: {ms_repo}) → {target}",
    },
    "download.eval_model": {
        "zh": "下载评估模型 {kind} {model_id} → {target}",
        "en": "Downloading evaluation model {kind} {model_id} → {target}",
    },
    "download.eval_model_via_ms": {
        "zh": "下载评估模型 {kind} {model_id}（经 ModelScope：{ms_repo}）→ {target}",
        "en": "Downloading evaluation model {kind} {model_id} "
              "(via ModelScope: {ms_repo}) → {target}",
    },
    "download.ccip": {
        "zh": "下载 CCIP {variant} → {target}",
        "en": "Downloading CCIP {variant} → {target}",
    },

    # ------------------------------------------------------------- modelsrc.*
    "modelsrc.file_done": {
        "zh": "{name} 已下载（来源 {source}）",
        "en": "{name} downloaded (source {source})",
    },
    "modelsrc.dir_done": {
        "zh": "{name} 已下载（来源 {source}）",
        "en": "{name} downloaded (source {source})",
    },
    "modelsrc.dir_present": {
        "zh": "{name} 已存在，跳过（来源 {source}）",
        "en": "{name} is already present; skipped (source {source})",
    },
    "modelsrc.present_summary": {
        "zh": "已存在 {n}/{total} 个文件，跳过",
        "en": "{n}/{total} files already present; skipped",
    },

    # --------------------------------------------------------------- update.*
    "update.models_preserved": {
        "zh": "[updater] 已保留 {n} 个已下载的模型文件，更新无需重新下载",
        "en": "[updater] Kept {n} downloaded model files; the update does not "
              "re-download them",
    },
    "update.applying": {
        "zh": "[updater] 正在应用更新 → {target}{force}",
        "en": "[updater] Applying update → {target}{force}",
    },
    "update.git_updated": {
        "zh": "[updater] 代码已更新 → {commit}",
        "en": "[updater] Code updated → {commit}",
    },
    "update.pip_install": {
        "zh": "[updater] requirements.txt 有变更，正在 pip install（可能需要几分钟）",
        "en": "[updater] requirements.txt changed; running pip install "
              "(this can take a few minutes)",
    },
    "update.npm_install": {
        "zh": "[updater] studio/web/package.json 有变更，正在 npm install（可能需要几分钟）",
        "en": "[updater] studio/web/package.json changed; running npm install "
              "(this can take a few minutes)",
    },
}
