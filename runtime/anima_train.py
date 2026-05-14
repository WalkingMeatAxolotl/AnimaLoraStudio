#!/usr/bin/env python
"""
Anima LoRA Trainer v2 - 支持 LyCORIS + 训练时推理
基于 trainerV1.01 重构，轻量单文件

特性：
- 标准 LoRA 和 LyCORIS LoKr 双模式
- 训练时推理出图
- Flow Matching 训练
- ARB 分桶
- 依赖自动检测与安装
- Rich 进度条 + ASCII Loss 曲线
- 梯度检查点支持
- Caption 预处理 (shuffle/keep_tokens)
"""

import argparse
import logging
import math
import os
import random
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Optional

# 脚本在 runtime/ 下按裸脚本启动（`python runtime/anima_train.py`）。
# 把仓库根 + runtime/ 注入 sys.path，让 `import utils.*` / `import train_monitor` 等
# 不需要改成包导入。
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "runtime"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# Windows 控制台默认 cp936，logging / print 写中文会 UnicodeEncodeError，
# 默认 handler 的 errors='backslashreplace' 会把中文转成 \uXXXX 形式 ——
# 这就是 task log 里看到的「检查 VAE」之类乱码的来源。
# 强制 stdout/stderr UTF-8 + replace 让中文 / emoji 永远直出。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ADR 0003 PR-A：模块化拆分。下面这一段从 training.* re-export，给 sister script /
# test 维持 `anima_train.X` 的访问路径。新代码请直接 import 子模块。
from training.bootstrap import (  # noqa: E402
    apply_yaml_config,
    ensure_dependencies,
    init_progress,
    load_yaml_config,
)
from training.observability import (  # noqa: E402
    WandBMonitor,
    init_wandb_monitor,
    render_curve_panel,
    render_loss_curve,
)
from training.model_loading import (  # noqa: E402
    _load_safetensors_state_dict,
    _load_weights_best_effort,
    _pick_best_prefix_remap,
    _strip_prefixes,
    enable_xformers,
    find_diffusion_pipe_root,
    forward_with_optional_checkpoint,
    load_module_from_path,
    resolve_path_best_effort,
)
from training.text_encoding import (  # noqa: E402
    _build_qwen_text_from_prompt,
    _parse_weighted_tag,
    encode_qwen,
    tokenize_t5_weighted,
)
from training.state import load_training_state, save_training_state  # noqa: E402
from training.models import (  # noqa: E402
    ensure_models_namespace,
    load_anima_model,
    load_text_encoders,
    load_vae,
)
from training.sampling import sample_image  # noqa: E402
from training.dataset import (  # noqa: E402
    BucketBatchSampler,
    BucketManager,
    CachedLatentDataset,
    ImageDataset,
    MergedDataset,
    RepeatDataset,
    collate_fn,
    collate_fn_cached,
)
from training.cli import (  # noqa: E402
    parse_args,
    prompt_for_args,
)
from training.timestep_sampling import sample_t  # noqa: E402
from training.noise import make_noise  # noqa: E402
from training.loss_weighting import compute_loss_weight  # noqa: E402


# 模型加载（load_anima_model / load_vae / load_text_encoders / ensure_models_namespace）
# 已搬到 training.models（公开，sister script 用）。
# 采样调度 sigma 工具 + sample_image 已搬到 training.sampling（ADR 0003 PR-A）。

# LoRA / LoKr 实现：见 utils.lycoris_adapter.AnimaLycorisAdapter
# （历史自实现版本于 Stage 3c 删除，使用 lycoris-lora 包替代）

# 训练状态保存/恢复（save_training_state / load_training_state）已搬到
# training.state；caption / tokenize 已搬到 training.text_encoding。




# 训练时推理 sample_image 已搬到 training.sampling（ADR 0003 PR-A）。


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()

    # 加载 YAML 配置文件
    config_path = None
    config_dir = None
    if args.config:
        logger.info(f"加载配置文件: {args.config}")
        config_path = Path(args.config).resolve()
        config_dir = config_path.parent
        config = load_yaml_config(args.config)
        args = apply_yaml_config(args, config)

    # bridge 已为 prefer_json bool 自动产生 --prefer-json / --no-prefer-json，
    # 此处无需再做兼容处理。

    # 交互模式检查
    required = [args.data_dir, args.transformer_path, args.vae_path, args.text_encoder_path]
    if args.interactive or any(not x for x in required):
        args = prompt_for_args(args)

    # 依赖检测
    ensure_dependencies(auto_install=args.auto_install)

    # 延迟导入
    import numpy as np
    from PIL import Image

    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    wandb_monitor = init_wandb_monitor(args, output_dir, config_path)

    # 训练监控状态写入（PP6.1）：永远开启，文件路径优先来自 --monitor-state-file，
    # 否则落到 output_dir/monitor_state.json。Studio 前端通过 /api/state?task_id=
    # 读这个文件，不再启动训练侧 HTTP server（Studio 自己是 monitor）。
    monitor_server = True  # 兼容下方分支判断；实际代表「写状态文件」
    try:
        from train_monitor import set_state_file, update_monitor
        state_path = (
            Path(args.monitor_state_file)
            if getattr(args, "monitor_state_file", None)
            else output_dir / "monitor_state.json"
        )
        set_state_file(state_path)
        update_monitor(
            total_epochs=int(args.epochs or 0),
            config={
                "model": {"lokr": "Anima LoKr"}.get(args.lora_type, "Anima LoRA"),
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.learning_rate,
                "resolution": args.resolution,
                "data_dir": str(args.data_dir),
            },
        )
        logger.info(f"📊 训练监控状态文件: {state_path}")
    except Exception as e:
        logger.warning(f"监控状态写入初始化失败: {e}")
        monitor_server = None

    # 查找模型代码
    repo_root = find_diffusion_pipe_root()
    logger.info(f"模型代码路径: {repo_root}")

    # 解析路径：相对路径优先按 config 位置 / AnimaLoraToolkit 目录解析
    script_dir = Path(__file__).resolve().parent
    bases = [
        Path.cwd(),
        config_dir,
        config_dir.parent if config_dir else None,
        script_dir,
        script_dir.parent,
        repo_root,
        repo_root.parent,
    ]
    args.transformer_path = resolve_path_best_effort(args.transformer_path, bases)
    args.vae_path = resolve_path_best_effort(args.vae_path, bases)
    args.text_encoder_path = resolve_path_best_effort(args.text_encoder_path, bases)
    args.t5_tokenizer_path = resolve_path_best_effort(args.t5_tokenizer_path, bases)
    args.data_dir = resolve_path_best_effort(args.data_dir, bases)
    reg_data_dir = getattr(args, "reg_data_dir", "") or ""
    if reg_data_dir:
        args.reg_data_dir = resolve_path_best_effort(reg_data_dir, bases)

    # 按 attention_backend 决策：xformers / flash_attn / none。
    # load_anima_model 内部按 flash_attn 参数设 flash_attn 全局开关；
    # xformers 是 model 层面的额外注入（与 flash_attn 互斥）。
    backend = getattr(args, "attention_backend", "flash_attn")
    use_flash = (backend == "flash_attn")

    # 加载模型
    logger.info("加载 Transformer...")
    model = load_anima_model(
        args.transformer_path, device, dtype, repo_root, flash_attn=use_flash,
    )

    if backend == "xformers":
        enable_xformers(model)
    elif backend == "none":
        logger.info("attention_backend=none，flash_attn / xformers 都不启用，走 PyTorch SDPA")

    logger.info("加载 VAE...")
    vae = load_vae(args.vae_path, device, dtype, repo_root)

    logger.info("加载文本编码器...")
    qwen_model, qwen_tok, t5_tok = load_text_encoders(
        args.text_encoder_path, args.t5_tokenizer_path, device, dtype
    )

    # 注入 LoRA
    logger.info(f"注入 {args.lora_type.upper()}...")
    from utils.lycoris_adapter import AnimaLycorisAdapter
    injector = AnimaLycorisAdapter(
        algo=args.lora_type,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        factor=args.lokr_factor,
        dropout=float(getattr(args, "lora_dropout", 0.0) or 0.0),
        rank_dropout=float(getattr(args, "lora_rank_dropout", 0.0) or 0.0),
        module_dropout=float(getattr(args, "lora_module_dropout", 0.0) or 0.0),
        weight_decompose=bool(getattr(args, "lora_dora", False)),
        rs_lora=bool(getattr(args, "lora_rs", False)),
    )
    injector.inject(model)
    
    # 从已有 LoRA 继续训练
    if getattr(args, "resume_lora", "") and Path(args.resume_lora).exists():
        injector.load(args.resume_lora)
        logger.info(f"将从已有 LoRA 继续训练: {args.resume_lora}")

    # 数据集
    bucket_mgr = BucketManager(args.resolution)
    base_dataset = ImageDataset(
        args.data_dir, args.resolution, bucket_mgr,
        shuffle_caption=args.shuffle_caption,
        keep_tokens=args.keep_tokens,
        flip_augment=args.flip_augment,
        tag_dropout=args.tag_dropout,
        prefer_json=args.prefer_json,
    )
    dataset = base_dataset

    # 正则数据集（Kohya 风格，防过拟合）
    reg_data_dir = getattr(args, "reg_data_dir", "") or ""
    reg_dataset = None
    if reg_data_dir:
        if not Path(reg_data_dir).exists():
            logger.warning(f"正则数据集路径不存在，已跳过: {reg_data_dir}")
        elif len(base_dataset) == 0:
            logger.warning("主数据集为空，正则集已跳过")
        else:
            reg_caption = (getattr(args, "reg_caption", "") or "").strip()
            reg_base = ImageDataset(
                reg_data_dir, args.resolution, bucket_mgr,
                shuffle_caption=args.shuffle_caption,
                keep_tokens=args.keep_tokens,
                flip_augment=args.flip_augment,
                tag_dropout=0.0,  # 正则集通常不用 dropout
                prefer_json=args.prefer_json,
                caption_override=reg_caption if reg_caption else None,
            )
            reg_dataset = reg_base
            reg_weight = float(getattr(args, "reg_weight", 1.0) or 1.0)
            cap_preview = f", caption=\"{reg_caption[:50]}{'...' if len(reg_caption) > 50 else ''}\"" if reg_caption else ""
            weight_info = f", weight={reg_weight}" if reg_weight != 1.0 else ""
            logger.info(f"正则数据集: {reg_data_dir} ({len(reg_base)} 样本, per-folder repeat{weight_info}){cap_preview}")

    # 缓存 VAE latents（在 repeat 之前）
    use_cached = getattr(args, "cache_latents", False)
    if use_cached:
        dataset = CachedLatentDataset(dataset, vae, device, dtype)
    if reg_dataset is not None and use_cached:
        reg_dataset = CachedLatentDataset(reg_dataset, vae, device, dtype)

    # repeat: 主数据集和正则数据集均通过文件夹名 Kohya 风格 repeat（如 5_concept），无需全局 repeat
    if reg_dataset is not None:
        reg_weight = float(getattr(args, "reg_weight", 1.0) or 1.0)
        dataset = MergedDataset(dataset, reg_dataset, reg_weight=reg_weight)

    if args.num_workers > 0 and os.name == "nt":
        logger.warning("num_workers > 0 在 Windows 上容易崩溃：已强制设为 0（避免多进程 spawn 问题）")
        args.num_workers = 0

    if use_cached:
        batch_sampler = BucketBatchSampler(
            dataset, batch_size=args.batch_size,
            drop_last=True, shuffle=True,
            seed=getattr(args, "seed", 42),
        )
        dataloader = DataLoader(
            dataset, batch_sampler=batch_sampler,
            collate_fn=collate_fn_cached,
            num_workers=args.num_workers,
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )

    # 训练前自检：VAE encode->decode 循环（快速排除 VAE/scale/shape 问题）
    try:
        if len(base_dataset) > 0:
            item0 = base_dataset[0]
            pixels0 = item0["pixel_values"].unsqueeze(0).to(device, dtype=dtype)  # [1,3,H,W]
            with torch.no_grad():
                z0 = vae.model.encode(pixels0.unsqueeze(2), vae.scale)   # [1,16,1,h,w]
                recon0 = vae.model.decode(z0, vae.scale).squeeze(2)      # [1,3,H,W]
                recon0 = (recon0.clamp(-1, 1) + 1) / 2
            arr0 = (recon0[0].permute(1, 2, 0).detach().cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
            Image.fromarray(arr0).save(sample_dir / "vae_roundtrip.png")
            logger.info("VAE roundtrip 自检已保存: samples/vae_roundtrip.png")
    except Exception as e:
        logger.warning(f"VAE roundtrip 自检失败（若 sample 仍是噪点，请优先修这个）: {e}")

    # 优化器
    weight_decay = float(getattr(args, "weight_decay", 0.01) or 0.0)
    param_groups = injector.get_param_groups(weight_decay)
    optimizer_type = (getattr(args, "optimizer_type", "adamw") or "adamw").lower()
    from utils.optimizer_utils import create_optimizer, optimizer_eval_mode
    optimizer_extra: dict = {}
    optimizer_overrides: dict = {}
    if optimizer_type == "prodigy":
        optimizer_extra["d_coef"] = float(getattr(args, "prodigy_d_coef", 1.0))
        optimizer_extra["safeguard_warmup"] = bool(getattr(args, "prodigy_safeguard_warmup", True))
    elif optimizer_type == "prodigy_plus_schedulefree":
        # Schedule-Free 不需要 scheduler，启动期强校验
        lr_sched_cfg = (getattr(args, "lr_scheduler", "none") or "none").lower()
        if lr_sched_cfg != "none":
            raise SystemExit(
                f"ProdigyPlusScheduleFree requires lr_scheduler=none "
                f"(Schedule-Free is scheduler-free by construction); got "
                f"lr_scheduler={lr_sched_cfg!r}. Set lr_scheduler=none or pick a "
                f"different optimizer."
            )
        optimizer_extra["d_coef"] = float(getattr(args, "ppsf_d_coef", 1.0))
        optimizer_extra["prodigy_steps"] = int(getattr(args, "ppsf_prodigy_steps", 0))
        optimizer_extra["split_groups"] = bool(getattr(args, "ppsf_split_groups", True))
        optimizer_extra["split_groups_mean"] = bool(getattr(args, "ppsf_split_groups_mean", False))
        optimizer_extra["use_speed"] = bool(getattr(args, "ppsf_use_speed", False))
        optimizer_extra["fused_back_pass"] = bool(getattr(args, "ppsf_fused_back_pass", False))
        optimizer_extra["use_stableadamw"] = bool(getattr(args, "ppsf_use_stableadamw", True))
        optimizer_overrides["betas"] = (
            float(getattr(args, "ppsf_beta1", 0.9)),
            float(getattr(args, "ppsf_beta2", 0.99)),
        )
    optimizer = create_optimizer(
        optimizer_type=optimizer_type,
        params=param_groups,
        learning_rate=args.learning_rate,
        weight_decay=weight_decay,
        **optimizer_overrides,
        **optimizer_extra,
    )
    if weight_decay > 0:
        wd_info = f"{optimizer_type} weight_decay={weight_decay}"
        if injector.use_lokr:
            wd_info += "（w1 排除 weight_decay）"
        logger.info(wd_info)
    grad_clip = float(getattr(args, "grad_clip_max_norm", 0) or 0)
    if grad_clip > 0:
        logger.info(f"梯度裁剪 max_norm={grad_clip}")
    trainable_params = [p for group in optimizer.param_groups for p in group["params"]]

    # 计算总步数
    try:
        steps_per_epoch = len(dataloader) // args.grad_accum
    except Exception:
        steps_per_epoch = None

    # total_steps：训练实际会跑到的步数。终止条件是「epoch 上限和 max_steps
    # 哪个先到就停」(见下方 max_steps break + for epoch 自然退出)，所以
    # 取两个候选的 min，进度条才不会出现「100 epoch 跑完了但只显示 86%」。
    by_epochs = (
        steps_per_epoch * args.epochs
        if steps_per_epoch is not None and args.epochs and args.epochs > 0
        else None
    )
    by_max_steps = (
        args.max_steps if (args.max_steps and args.max_steps > 0) else None
    )
    candidates = [c for c in (by_epochs, by_max_steps) if c is not None and c > 0]
    total_steps = min(candidates) if candidates else None

    logger.info(
        f"数据集大小: {len(dataset)}, 每 epoch 步数: {steps_per_epoch}, "
        f"总步数: {total_steps} (by_epochs={by_epochs}, by_max_steps={by_max_steps})"
    )

    # 学习率调度器
    scheduler = None
    lr_sched = getattr(args, "lr_scheduler", "none") or "none"
    if lr_sched == "cosine":
        eta_min = float(getattr(args, "lr_scheduler_eta_min", 0.0) or 0.0)
        if total_steps is None:
            logger.warning("cosine 调度器需要已知 total_steps，回退到 none")
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=eta_min
            )
            logger.info(f"学习率调度: cosine (T_max={total_steps}, eta_min={eta_min})")
    elif lr_sched == "cosine_with_restart":
        t0 = int(getattr(args, "lr_scheduler_t0", 500) or 500)
        t_mult = int(getattr(args, "lr_scheduler_t_mult", 2) or 2)
        eta_min = float(getattr(args, "lr_scheduler_eta_min", 0.0) or 0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min
        )
        logger.info(f"学习率调度: cosine_with_restart (T_0={t0}, T_mult={t_mult}, eta_min={eta_min})")

    # 初始化进度显示
    progress, task_id, progress_kind = init_progress(not args.no_progress, total_steps)
    use_rich = progress_kind == "rich"
    use_plain = progress == "plain"
    live = None
    loss_history = []
    speed_ema = None

    if use_rich:
        try:
            from rich.console import Group
            from rich.live import Live
            curve_panel = None
            if args.loss_curve_steps > 0 and not args.no_live_curve:
                curve_panel = render_curve_panel([], width=min(60, args.loss_curve_steps), height=10)
            group = Group(progress, curve_panel) if curve_panel is not None else Group(progress)
            live = Live(group, refresh_per_second=10)
            live.start()
        except Exception:
            live = None
            progress.start()

    def emit(msg):
        if use_plain:
            print()
        if live:
            live.console.print(msg)
        elif use_rich:
            progress.console.print(msg)
        else:
            print(msg)

    # 训练循环
    global_step = 0
    start_epoch = 0
    
    # 从训练状态恢复（断点续训）
    if getattr(args, "resume_state", "") and Path(args.resume_state).exists():
        start_epoch, global_step, loss_history, saved_monitor_state = load_training_state(
            args.resume_state, injector, optimizer, scheduler
        )
        emit(f"从断点恢复训练: epoch={start_epoch}, step={global_step}")
        
        # 恢复监控面板的历史数据（loss 曲线等）
        if monitor_server and saved_monitor_state:
            try:
                from train_monitor import restore_monitor_state
                restore_monitor_state(
                    losses=saved_monitor_state.get("losses"),
                    lr_history=saved_monitor_state.get("lr_history"),
                    epoch=start_epoch,
                    step=global_step,
                    total_steps=total_steps,
                )
                emit(f"监控面板历史数据已恢复: {len(saved_monitor_state.get('losses', []))} 个 loss 点")
            except Exception as e:
                emit(f"监控数据恢复失败: {e}")
    
    # Ctrl+C 信号处理：保存状态后退出
    interrupted = False
    def signal_handler(sig, frame):
        nonlocal interrupted
        if interrupted:
            emit("强制退出...")
            sys.exit(1)
        interrupted = True
        emit("\n检测到 Ctrl+C，正在保存训练状态...")
        state_path = output_dir / f"training_state_step{global_step}.pt"
        # 获取监控面板数据用于恢复 loss 曲线
        monitor_data = None
        if monitor_server:
            try:
                from train_monitor import get_state
                monitor_data = get_state()
            except Exception:
                pass
        # Schedule-Free 系优化器（PPSF）保存前切到 averaged weights — 否则存的是
        # 训练用的 y 而不是真正应该被使用的 x。非 SF 优化器此 ctx 静默 no-op。
        with optimizer_eval_mode(optimizer):
            save_training_state(state_path, injector, optimizer, current_epoch, global_step, loss_history, monitor_state=monitor_data, scheduler=scheduler)
            # 同时保存 LoRA 权重
            lora_path = output_dir / f"{args.output_name}_interrupted_step{global_step}.safetensors"
            injector.save(lora_path)
        wandb_monitor.finish()
        emit(f"已保存！下次使用 --resume-state \"{state_path}\" 继续训练")
        sys.exit(0)
    
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    current_epoch = start_epoch
    model.train()
    if optimizer_type == "prodigy_plus_schedulefree" and hasattr(optimizer, "train"):
        optimizer.train()
    step_start_time = time.perf_counter()

    # 设置采样提示词列表（支持多角色轮换）
    sample_prompts = getattr(args, "sample_prompts", []) or []
    if not sample_prompts and args.sample_prompt:
        sample_prompts = [args.sample_prompt]
    sample_prompt_idx = 0

    def get_next_sample_prompt():
        """获取下一个采样提示词（轮换）"""
        nonlocal sample_prompt_idx
        if not sample_prompts:
            return "1girl, masterpiece"
        prompt = sample_prompts[sample_prompt_idx % len(sample_prompts)]
        sample_prompt_idx += 1
        return prompt

    # Step 0 初始采样（基线效果，测试所有提示词）
    # 只在新训练时执行（global_step == 0），resume 时跳过
    sampling_enabled = args.sample_steps > 0 or args.sample_every > 0
    if global_step == 0 and sampling_enabled:
        emit("采样中 (step 0, 基线)...")
        # PPSF：sample 时切到 averaged weights，事后切回训练权重
        with optimizer_eval_mode(optimizer):
            model.eval()
            s_w = int(getattr(args, "sample_width", 0) or 0) or int(args.resolution)
            s_h = int(getattr(args, "sample_height", 0) or 0) or int(args.resolution)
            s_cfg = float(getattr(args, "sample_cfg_scale", 4.0) or 4.0)
            s_neg = str(getattr(args, "sample_negative_prompt", "") or "")
            s_seed = int(getattr(args, "sample_seed", 0) or 0)
            s_steps = int(getattr(args, "sample_infer_steps", 25) or 25)
            s_sampler = str(getattr(args, "sample_sampler_name", "er_sde") or "er_sde")
            s_sched = str(getattr(args, "sample_scheduler", "simple") or "simple")
            for i, prompt in enumerate(sample_prompts[:3]):  # 最多测试 3 个
                if s_seed:
                    torch.manual_seed(s_seed + i)
                img = sample_image(
                    model, vae, qwen_model, qwen_tok, t5_tok,
                    prompt, height=s_h, width=s_w, steps=s_steps, cfg_scale=s_cfg,
                    negative_prompt=(s_neg or None),
                    sampler_name=s_sampler,
                    scheduler=s_sched,
                    device=device, dtype=dtype
                )
                sample_path = sample_dir / f"step_0_baseline_{i}.png"
                img.save(sample_path)
                emit(f"基线采样保存: step_0_baseline_{i}.png")
                if wandb_monitor.log_samples:
                    wandb_monitor.log_image(
                        "samples/baseline",
                        sample_path,
                        caption=f"step 0 baseline {i}: {prompt}",
                        step=0,
                    )
                if monitor_server:
                    try:
                        update_monitor(sample_path=sample_path)
                    except Exception:
                        pass
            model.train()
    elif global_step > 0 and sampling_enabled:
        emit(f"跳过启动基线采样（从 step {global_step} 恢复，非 step 0）")

    for epoch in range(start_epoch, args.epochs):
        current_epoch = epoch
        if use_cached and hasattr(dataloader, "batch_sampler") and hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(dataloader):
            # 在累积周期开始时记录时间
            if batch_idx % args.grad_accum == 0:
                step_start_time = time.perf_counter()

            captions = batch["captions"]

            # 获取 latents（缓存模式或实时编码）
            if use_cached:
                latents = batch["latents"].to(device, dtype=dtype)
            else:
                pixels = batch["pixel_values"].to(device, dtype=dtype)
                with torch.no_grad():
                    pixels_5d = pixels.unsqueeze(2)  # [B,C,1,H,W]
                    latents = vae.model.encode(pixels_5d, vae.scale)

            bs = latents.shape[0]

            # 文本编码
            with torch.no_grad():
                # 参考指南/ComfyUI：Qwen 通道不传权重；T5 通道提供 token 权重
                qwen_texts = [_build_qwen_text_from_prompt(c) for c in captions]
                qwen_emb, qwen_attn = encode_qwen(qwen_model, qwen_tok, qwen_texts, device)
                t5_ids, t5_attn, t5_w = tokenize_t5_weighted(t5_tok, captions, max_length=512)
                t5_ids = t5_ids.to(device)
                t5_attn = t5_attn.to(device)
                t5_w = t5_w.to(device, dtype=torch.float32)
                cross = model.preprocess_text_embeds(qwen_emb, t5_ids)
                if cross.shape[1] < 512:
                    cross = F.pad(cross, (0, 0, 0, 512 - cross.shape[1]))
                # KV trim：把 padding 截到最近有效 token bucket（64/128/256/512）
                # t5_attn=1 表示有效 token；取批次内最大实际长度再 round up
                if getattr(args, "kv_trim", False):
                    _actual = int(t5_attn.sum(dim=-1).max().item())
                    _bucket = 512  # _actual > 512 时兜底（不裁，保持原行为）
                    for _b in (64, 128, 256, 512):
                        if _b >= _actual:
                            _bucket = _b
                            break
                    cross = cross[:, :_bucket, :].contiguous()

            # Flow Matching
            t = sample_t(
                bs, device,
                mode=str(getattr(args, "timestep_sampling", "logit_normal") or "logit_normal"),
                shift=float(getattr(args, "timestep_shift", 3.0) or 3.0),
            )
            t_exp = t.view(-1, 1, 1, 1, 1)
            noise = make_noise(
                latents,
                noise_offset=float(getattr(args, "noise_offset", 0.0) or 0.0),
                pyramid_iters=int(getattr(args, "pyramid_noise_iters", 0) or 0),
                pyramid_discount=float(getattr(args, "pyramid_noise_discount", 0.35) or 0.35),
            )
            noisy = (1 - t_exp) * latents + t_exp * noise
            target = noise - latents

            # 前向
            pad_mask = torch.zeros(bs, 1, latents.shape[-2], latents.shape[-1], device=device, dtype=dtype)
            with torch.autocast("cuda", dtype=dtype):
                pred = forward_with_optional_checkpoint(
                    model, noisy, t.view(-1, 1), cross, pad_mask,
                    use_checkpoint=args.grad_checkpoint
                )
                loss_per_sample = F.mse_loss(pred.float(), target.float(), reduction="none")
                # 按样本加权（正则集可降低权重）
                if "loss_weight" in batch:
                    w = batch["loss_weight"].to(device).view(-1, *([1] * (loss_per_sample.dim() - 1)))
                    loss_per_sample = loss_per_sample * w
                # timestep-dependent loss 权重
                lw_scheme = str(getattr(args, "loss_weighting", "none") or "none")
                if lw_scheme != "none":
                    lw = compute_loss_weight(
                        t,
                        scheme=lw_scheme,
                        min_snr_gamma=float(getattr(args, "min_snr_gamma", 5.0) or 5.0),
                        weight_cap_ratio=float(getattr(args, "weight_cap_ratio", 0.0) or 0.0),
                    ).to(device=device, dtype=torch.float32)
                    loss_per_sample = loss_per_sample * lw.view(-1, *([1] * (loss_per_sample.dim() - 1)))
                loss = loss_per_sample.mean()

            # NaN 检测：forward 出 NaN 时跳过本 micro-batch
            if not torch.isfinite(loss):
                logger.warning(f"step {global_step} micro-batch {batch_idx}: loss={loss.item():.4g}，跳过")
                optimizer.zero_grad()
                continue

            # 反向传播
            loss = loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                # NaN 梯度检测：跳过本次 update，清零继续
                has_nan_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in trainable_params
                )
                if has_nan_grad:
                    logger.warning(f"step {global_step}: 梯度含 NaN/Inf，跳过 optimizer.step()")
                    optimizer.zero_grad()
                    continue

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
                optimizer.step()
                if scheduler is not None and optimizer_type != "prodigy_plus_schedulefree":
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # 记录 loss 历史
                loss_val = float(loss.item() * args.grad_accum)
                if args.loss_curve_steps and len(loss_history) < args.loss_curve_steps:
                    loss_history.append(loss_val)

                # 更新进度显示
                now = time.perf_counter()
                lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
                
                # 更新训练监控面板
                if monitor_server:
                    try:
                        update_monitor(
                            loss=loss_val, lr=lr, epoch=epoch+1,
                            total_epochs=int(args.epochs or 0),
                            step=global_step,
                            total_steps=total_steps, speed=speed_ema or 0
                        )
                    except Exception:
                        pass
                dt_step = now - step_start_time
                steps_per_sec = (1.0 / dt_step) if dt_step > 0 else 0.0
                speed_ema = steps_per_sec if speed_ema is None else (0.9 * speed_ema + 0.1 * steps_per_sec)
                wandb_monitor.log(
                    {
                        "train/loss": loss_val,
                        "train/lr": float(lr),
                        "train/epoch": epoch + 1,
                        "train/speed_it_s": float(speed_ema or 0),
                    },
                    step=global_step,
                )

                if use_rich:
                    desc = f"epoch {epoch+1}/{args.epochs} step {global_step}/{total_steps or '?'}"
                    progress.update(task_id, advance=1, description=desc,
                                    loss=loss_val, lr=float(lr), speed=float(speed_ema or 0))
                    if live and args.loss_curve_steps > 0 and not args.no_live_curve:
                        panel = render_curve_panel(loss_history, width=min(60, args.loss_curve_steps), height=10)
                        if panel is not None:
                            from rich.console import Group
                            live.update(Group(progress, panel))
                elif use_plain:
                    print(f"epoch {epoch+1}/{args.epochs} step {global_step} loss={loss_val:.6f} lr={lr:.2e} speed={speed_ema:.2f} it/s", end="\r", flush=True)
                elif args.log_every and global_step % args.log_every == 0:
                    print(f"epoch={epoch} step={global_step} loss={loss_val:.6f} lr={lr:.2e} speed={steps_per_sec:.2f} it/s")

                # 按 step 采样（轮换提示词）
                if args.sample_steps > 0 and global_step % args.sample_steps == 0:
                    prompt = get_next_sample_prompt()
                    prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
                    emit(f"采样中 (step {global_step}): {prompt_short}")
                    # PPSF：sample 走 averaged weights
                    with optimizer_eval_mode(optimizer):
                        model.eval()
                        s_w = int(getattr(args, "sample_width", 0) or 0) or int(args.resolution)
                        s_h = int(getattr(args, "sample_height", 0) or 0) or int(args.resolution)
                        s_cfg = float(getattr(args, "sample_cfg_scale", 4.0) or 4.0)
                        s_neg = str(getattr(args, "sample_negative_prompt", "") or "")
                        s_steps = int(getattr(args, "sample_infer_steps", 25) or 25)
                        s_sampler = str(getattr(args, "sample_sampler_name", "er_sde") or "er_sde")
                        s_sched = str(getattr(args, "sample_scheduler", "simple") or "simple")
                        img = sample_image(
                            model, vae, qwen_model, qwen_tok, t5_tok,
                            prompt, height=s_h, width=s_w, steps=s_steps, cfg_scale=s_cfg,
                            negative_prompt=(s_neg or None),
                            sampler_name=s_sampler,
                            scheduler=s_sched,
                            device=device, dtype=dtype
                        )
                        sample_path = sample_dir / f"step_{global_step}.png"
                        img.save(sample_path)
                        emit(f"采样保存: step_{global_step}.png")
                        if wandb_monitor.log_samples:
                            wandb_monitor.log_image(
                                "samples/step",
                                sample_path,
                                caption=f"step {global_step}: {prompt}",
                                step=global_step,
                            )
                        if monitor_server:
                            try:
                                update_monitor(sample_path=sample_path)
                            except Exception:
                                pass
                        model.train()

                # 定期保存 LoRA 权重（按 step）
                save_every_steps = getattr(args, "save_every_steps", 0)
                if save_every_steps > 0 and global_step % save_every_steps == 0:
                    lora_path = output_dir / f"{args.output_name}_step{global_step}.safetensors"
                    # PPSF：保存 averaged weights 的 LoRA
                    with optimizer_eval_mode(optimizer):
                        injector.save(lora_path)
                    emit(f"Saved LoRA: {lora_path}")

                # 定期保存训练状态（断点续训）
                save_state_every = getattr(args, "save_state_every", 0)
                if save_state_every > 0 and global_step % save_state_every == 0:
                    state_path = output_dir / f"training_state_step{global_step}.pt"
                    # 获取监控面板数据用于恢复 loss 曲线
                    monitor_data = None
                    if monitor_server:
                        try:
                            from train_monitor import get_state
                            monitor_data = get_state()
                        except Exception:
                            pass
                    # PPSF：state + LoRA 都走 averaged weights
                    with optimizer_eval_mode(optimizer):
                        save_training_state(state_path, injector, optimizer, epoch, global_step, loss_history, monitor_state=monitor_data, scheduler=scheduler)
                        # 同时保存 LoRA 权重
                        lora_path = output_dir / f"{args.output_name}_step{global_step}.safetensors"
                        injector.save(lora_path)

                # 检查 max_steps
                if args.max_steps and global_step >= args.max_steps:
                    break

        # epoch 结束后的操作
        current_epoch = epoch + 1
        if not args.max_steps or global_step < args.max_steps:
            # 保存 checkpoint
            if args.save_every > 0 and current_epoch % args.save_every == 0:
                save_path = output_dir / f"{args.output_name}_epoch{current_epoch}.safetensors"
                # PPSF：保存 averaged weights 的 LoRA
                with optimizer_eval_mode(optimizer):
                    injector.save(save_path)
                emit(f"Saved LoRA: {save_path}")

            # 采样（轮换提示词）
            if args.sample_every > 0 and current_epoch % args.sample_every == 0:
                prompt = get_next_sample_prompt()
                prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
                emit(f"采样中 (epoch {current_epoch}): {prompt_short}")
                # PPSF：sample 走 averaged weights
                with optimizer_eval_mode(optimizer):
                    model.eval()
                    s_w = int(getattr(args, "sample_width", 0) or 0) or int(args.resolution)
                    s_h = int(getattr(args, "sample_height", 0) or 0) or int(args.resolution)
                    s_cfg = float(getattr(args, "sample_cfg_scale", 4.0) or 4.0)
                    s_neg = str(getattr(args, "sample_negative_prompt", "") or "")
                    s_steps = int(getattr(args, "sample_infer_steps", 25) or 25)
                    s_sampler = str(getattr(args, "sample_sampler_name", "er_sde") or "er_sde")
                    s_sched = str(getattr(args, "sample_scheduler", "simple") or "simple")
                    img = sample_image(
                        model, vae, qwen_model, qwen_tok, t5_tok,
                        prompt, height=s_h, width=s_w, steps=s_steps, cfg_scale=s_cfg,
                        negative_prompt=(s_neg or None),
                        sampler_name=s_sampler,
                        scheduler=s_sched,
                        device=device, dtype=dtype
                    )
                    sample_path = sample_dir / f"epoch_{current_epoch}.png"
                    img.save(sample_path)
                    emit(f"采样保存: epoch_{current_epoch}.png")
                    if wandb_monitor.log_samples:
                        wandb_monitor.log_image(
                            "samples/epoch",
                            sample_path,
                            caption=f"epoch {current_epoch}: {prompt}",
                            step=global_step,
                        )
                    model.train()

                    # 更新监控面板
                    if monitor_server:
                        try:
                            update_monitor(sample_path=sample_path)
                        except Exception:
                            pass

        # 检查 max_steps
        if args.max_steps and global_step >= args.max_steps:
            break

    # 最终保存
    final_path = output_dir / f"{args.output_name}.safetensors"
    # PPSF：最终输出走 averaged weights
    with optimizer_eval_mode(optimizer):
        injector.save(final_path)

    # 清理进度显示
    if live:
        live.stop()
    elif use_rich:
        progress.stop()

    # 显示最终 loss 曲线
    if args.loss_curve_steps and loss_history:
        chart = render_loss_curve(loss_history, width=min(80, len(loss_history)), height=10)
        emit(f"Loss curve (first {len(loss_history)} steps):\n{chart}")

    emit(f"Saved final LoRA: {final_path}")
    wandb_monitor.finish()
    logger.info("训练完成!")


if __name__ == "__main__":
    main()
