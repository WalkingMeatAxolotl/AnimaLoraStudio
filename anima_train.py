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
import os
import random
import subprocess
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# 依赖检测
# ============================================================================

def ensure_dependencies(auto_install=False):
    """检测并可选自动安装缺失依赖"""
    required = {
        "numpy": "numpy",
        "PIL": "Pillow",
        "safetensors": "safetensors",
        "transformers": "transformers",
        "einops": "einops",
        "torchvision": "torchvision",
        "yaml": "pyyaml",
    }
    missing = []
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
        except Exception:
            missing.append(pip_name)
    if not missing:
        return
    missing_list = ", ".join(sorted(set(missing)))
    print(f"Missing dependencies: {missing_list}")
    if not auto_install:
        print(f"Install them with:\n  {sys.executable} -m pip install {missing_list}")
        raise SystemExit(1)
    cmd = [sys.executable, "-m", "pip", "install", *sorted(set(missing))]
    print("Installing missing dependencies...")
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:
        print(f"Auto-install failed: {exc}")
        raise SystemExit(1)
    # Re-check after install
    still_missing = []
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
        except Exception:
            still_missing.append(pip_name)
    if still_missing:
        still_list = ", ".join(sorted(set(still_missing)))
        print(f"Still missing: {still_list}")
        raise SystemExit(1)


# ============================================================================
# YAML 配置加载
# ============================================================================

def load_yaml_config(config_path):
    """加载 YAML 配置文件"""
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed. Install with: pip install pyyaml")
        raise SystemExit(1)

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def apply_yaml_config(args, config):
    """将 YAML 配置应用到 args，命令行参数优先"""
    # 字段映射: YAML key -> args attribute
    mapping = {
        # 模型路径
        "transformer_path": "transformer",
        "vae_path": "vae",
        "text_encoder_path": "qwen",
        "t5_tokenizer_path": "t5_tokenizer",
        # 数据集
        "data_dir": "data_dir",
        "reg_data_dir": "reg_data_dir",
        "reg_caption": "reg_caption",
        "reg_weight": "reg_weight",
        "resolution": "resolution",
        "repeats": "repeats",
        "shuffle_caption": "shuffle_caption",
        "keep_tokens": "keep_tokens",
        "flip_augment": "flip_augment",
        "tag_dropout": "tag_dropout",
        "prefer_json": "prefer_json",
        "cache_latents": "cache_latents",
        # LoRA 配置
        "lora_type": "lora_type",
        "lora_rank": "lora_rank",
        "lora_alpha": "lora_alpha",
        "lokr_factor": "lokr_factor",
        "resume_lora": "resume_lora",
        # 训练参数
        "epochs": "epochs",
        "max_steps": "max_steps",
        "batch_size": "batch_size",
        "grad_accum": "grad_accum",
        "learning_rate": "lr",
        "lr_scheduler": "lr_scheduler",
        "lr_scheduler_t0": "lr_scheduler_t0",
        "lr_scheduler_t_mult": "lr_scheduler_t_mult",
        "lr_scheduler_eta_min": "lr_scheduler_eta_min",
        "weight_decay": "weight_decay",
        "grad_clip_max_norm": "grad_clip_max_norm",
        "mixed_precision": "mixed_precision",
        "grad_checkpoint": "grad_checkpoint",
        "xformers": "xformers",
        "num_workers": "num_workers",
        # 输出与保存
        "output_dir": "output_dir",
        "output_name": "output_name",
        "save_every": "save_every",
        "save_every_steps": "save_every_steps",
        "save_state_every": "save_state_every",
        "resume_state": "resume_state",
        "seed": "seed",
        # 采样
        "sample_every": "sample_every",
        "sample_steps": "sample_steps",
        "sample_prompt": "sample_prompt",
        "sample_prompts": "sample_prompts",
        "sample_cfg_scale": "sample_cfg_scale",
        "sample_negative_prompt": "sample_negative_prompt",
        "sample_width": "sample_width",
        "sample_height": "sample_height",
        "sample_seed": "sample_seed",
        "sample_infer_steps": "sample_infer_steps",
        "sample_sampler_name": "sample_sampler_name",
        "sample_scheduler": "sample_scheduler",
        # 进度显示与监控
        "loss_curve_steps": "loss_curve_steps",
        "no_progress": "no_progress",
        "log_every": "log_every",
        "no_monitor": "no_monitor",
        "monitor_host": "monitor_host",
        "monitor_port": "monitor_port",
        "no_browser": "no_browser",
    }

    # 需要特殊处理的默认值（用于判断命令行是否显式设置）
    defaults = {
        "transformer": "",
        "vae": "",
        "qwen": "",
        "t5_tokenizer": "",
        "data_dir": "",
        "reg_data_dir": "",
        "reg_caption": "",
        "resolution": 1024,
        "repeats": 1,
        "shuffle_caption": False,
        "keep_tokens": 0,
        "flip_augment": False,
        "tag_dropout": 0.0,
        "prefer_json": True,
        "cache_latents": False,
        "lora_type": "lokr",
        "lora_rank": 32,
        "lora_alpha": 32.0,
        "lokr_factor": 8,
        "resume_lora": "",
        "epochs": 10,
        "max_steps": 0,
        "batch_size": 1,
        "grad_accum": 1,
        "lr": 1e-4,
        "lr_scheduler": "none",
        "lr_scheduler_t0": 500,
        "lr_scheduler_t_mult": 2.0,
        "lr_scheduler_eta_min": 0.0,
        "weight_decay": 0.01,
        "grad_clip_max_norm": 1.0,
        "mixed_precision": "bf16",
        "grad_checkpoint": False,
        "xformers": False,
        "num_workers": 0,
        "output_dir": "./output",
        "output_name": "anima_lora",
        "save_every": 0,
        "save_every_steps": 0,
        "save_state_every": 0,
        "resume_state": "",
        "seed": 42,
        "sample_every": 0,
        "sample_steps": 0,
        "sample_prompt": "1girl, masterpiece",
        "sample_prompts": [],
        "sample_cfg_scale": 4.0,
        "sample_negative_prompt": "",
        "sample_width": 0,
        "sample_height": 0,
        "sample_seed": 0,
        "sample_infer_steps": 25,
        "sample_sampler_name": "er_sde",
        "sample_scheduler": "simple",
        "loss_curve_steps": 100,
        "no_progress": False,
        "log_every": 10,
        "no_monitor": False,
        "monitor_host": "127.0.0.1",
        "monitor_port": 8765,
        "no_browser": False,
    }

    for yaml_key, arg_attr in mapping.items():
        if yaml_key not in config:
            continue
        yaml_value = config[yaml_key]
        if yaml_value is None:
            continue

        # 检查命令行是否显式设置了该参数（与默认值不同）
        current_value = getattr(args, arg_attr, None)
        default_value = defaults.get(arg_attr)

        # 如果当前值等于默认值，或者属性不存在（current_value 为 None），则使用 YAML 配置
        # 特殊处理：列表类型的默认值用 [] 表示，但 argparse 未定义时返回 None
        if current_value == default_value or current_value is None:
            setattr(args, arg_attr, yaml_value)

    return args


# Lazy imports after dependency check
def _lazy_imports():
    global np, Image
    import numpy as np
    from PIL import Image


# ============================================================================
# 进度和 Loss 曲线可视化
# ============================================================================

def init_progress(show_progress, total_steps):
    """初始化 Rich 进度条"""
    if not show_progress:
        return None, None, None
    try:
        from rich.progress import (
            BarColumn, MofNCompleteColumn, Progress, TextColumn,
            TimeElapsedColumn, TimeRemainingColumn,
        )
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss={task.fields[loss]:.4f}"),
            TextColumn("lr={task.fields[lr]:.2e}"),
            TextColumn("speed={task.fields[speed]:.2f} it/s"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=10,
        )
        task = progress.add_task("train", total=total_steps, loss=0.0, lr=0.0, speed=0.0)
        return progress, task, "rich"
    except Exception:
        return "plain", None, None


def render_loss_curve(losses, width=60, height=10):
    """渲染 ASCII Loss 曲线"""
    if not losses:
        return ""
    if width < 5:
        width = 5
    values = losses
    if len(values) > width:
        step = len(values) / width
        buckets = []
        for i in range(width):
            start = int(i * step)
            end = int((i + 1) * step)
            end = max(end, start + 1)
            chunk = values[start:end]
            buckets.append(sum(chunk) / len(chunk))
        values = buckets
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        max_v = min_v + 1e-8
    grid = [[" " for _ in range(len(values))] for _ in range(height)]
    for i, v in enumerate(values):
        y = int((v - min_v) / (max_v - min_v) * (height - 1))
        y = height - 1 - y
        grid[y][i] = "*"
    lines = ["".join(row) for row in grid]
    lines.append(f"min={min_v:.4f} max={max_v:.4f}")
    return "\n".join(lines)


def render_curve_panel(losses, width=60, height=10):
    """渲染 Rich Panel 包装的 Loss 曲线"""
    try:
        from rich.panel import Panel
        from rich.text import Text
    except Exception:
        return None
    chart = render_loss_curve(losses, width=width, height=height)
    return Panel(Text(chart), title="Loss curve (recent)", expand=False)


# ============================================================================
# 梯度检查点
# ============================================================================

def forward_with_optional_checkpoint(model, latents, timesteps, cross, padding_mask, use_checkpoint=False):
    """带可选梯度检查点的前向传播"""
    if not use_checkpoint:
        return model(latents, timesteps, cross, padding_mask=padding_mask)
    from torch.utils.checkpoint import checkpoint

    x_B_T_H_W_D, rope_emb, extra_pos_emb = model.prepare_embedded_sequence(
        latents, fps=None, padding_mask=padding_mask,
    )
    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(1)
    t_embedding, adaln_lora = model.t_embedder(timesteps)
    t_embedding = model.t_embedding_norm(t_embedding)

    block_kwargs = {
        "rope_emb_L_1_1_D": rope_emb,
        "adaln_lora_B_T_3D": adaln_lora,
        "extra_per_block_pos_emb": extra_pos_emb,
    }

    for block in model.blocks:
        def custom_forward(x, blk=block):
            return blk(x, t_embedding, cross, **block_kwargs)
        x_B_T_H_W_D = checkpoint(custom_forward, x_B_T_H_W_D, use_reentrant=False)

    x_B_T_H_W_O = model.final_layer(x_B_T_H_W_D, t_embedding, adaln_lora_B_T_3D=adaln_lora)
    return model.unpatchify(x_B_T_H_W_O)


# ============================================================================
# xformers 支持
# ============================================================================

def enable_xformers(model):
    """为模型启用 xformers memory efficient attention"""
    try:
        from xformers.ops import memory_efficient_attention
    except ImportError:
        logger.warning("xformers 未安装，跳过启用")
        return False

    enabled_count = 0
    for name, module in model.named_modules():
        # 查找 attention 模块并替换
        if hasattr(module, "set_use_memory_efficient_attention_xformers"):
            module.set_use_memory_efficient_attention_xformers(True)
            enabled_count += 1
        elif hasattr(module, "enable_xformers_memory_efficient_attention"):
            module.enable_xformers_memory_efficient_attention()
            enabled_count += 1

    if enabled_count > 0:
        logger.info(f"xformers 已启用: {enabled_count} 个模块")
        return True

    # 如果模型没有内置支持，尝试 monkey patch
    logger.info("xformers 已加载，将在 attention 计算中使用")
    return True


# ============================================================================
# 模型加载工具
# ============================================================================

def find_diffusion_pipe_root():
    """查找 diffusion-pipe 模型代码路径"""
    candidates = [
        Path(__file__).parent / "diffusion_models",
        Path(__file__).parent / "models",
        Path(os.environ.get("DIFFUSION_PIPE_ROOT", "")) if os.environ.get("DIFFUSION_PIPE_ROOT") else None,
    ]
    for candidate in candidates:
        if candidate and (candidate / "anima_modeling.py").exists():
            return candidate
        if candidate and (candidate / "models" / "anima_modeling.py").exists():
            return candidate / "models"
    raise RuntimeError("找不到 anima_modeling.py，请设置 DIFFUSION_PIPE_ROOT 或放置模型代码")


def load_module_from_path(module_name, file_path):
    """动态加载 Python 模块"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strip_prefixes(key: str, prefixes: list[str]) -> str:
    """反复剥离前缀（支持 module.model. 这种复合前缀）"""
    if not prefixes:
        return key
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p) :]
                changed = True
    return key


def _pick_best_prefix_remap(sd_keys: list[str], model_keys: set[str]) -> tuple[list[str], int]:
    """
    从常见前缀组合里选择“命中最多 model_keys”的 remap 方案。
    返回 (prefixes, matched_count)。
    """
    candidates: list[tuple[str, list[str]]] = [
        ("none", []),
        ("net.", ["net."]),
        ("model.", ["model."]),
        ("module.", ["module."]),
        ("module.+model.", ["module.", "model."]),
        ("module.model.", ["module.model."]),
        ("diffusion_model.", ["diffusion_model."]),
        ("model.diffusion_model.", ["model.diffusion_model."]),
        ("transformer.", ["transformer."]),
        ("vae.", ["vae."]),
        ("first_stage_model.", ["first_stage_model."]),
        ("net.+model.", ["net.", "model."]),
        ("net.model.", ["net.model."]),
    ]

    best_prefixes: list[str] = []
    best_matched = -1
    for _name, prefixes in candidates:
        matched = 0
        for k in sd_keys:
            kk = _strip_prefixes(k, prefixes)
            if kk in model_keys:
                matched += 1
        if matched > best_matched:
            best_matched = matched
            best_prefixes = prefixes
    return best_prefixes, best_matched


def _load_safetensors_state_dict(path: Path) -> dict:
    from safetensors import safe_open

    sd = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    return sd


def resolve_path_best_effort(path_str: str, bases: list[Path]) -> str:
    """
    将相对路径按多个 base 尝试解析到一个真实存在的路径。
    主要用于：无论从 repo 根目录还是 AnimaLoraToolkit 目录启动，都能找到 models/* 文件。
    """
    if not path_str:
        return path_str

    p = Path(path_str)
    if p.is_absolute():
        return str(p)

    # 先按原样（相对 cwd）试一下
    if p.exists():
        return str(p)

    # 逐 base 拼接尝试
    for b in bases:
        if not b:
            continue
        try:
            cand = (Path(b) / p).resolve()
        except Exception:
            cand = Path(b) / p
        if cand.exists():
            return str(cand)

    # 常见：配置写了 AnimaLoraToolkit/xxx，但启动目录已经在 AnimaLoraToolkit 下
    parts = p.parts
    if parts and parts[0].lower() in ("animaloratoolkit", "anima_trainer", "anima-trainer"):
        p2 = Path(*parts[1:])
        if p2.exists():
            return str(p2)
        for b in bases:
            if not b:
                continue
            cand = Path(b) / p2
            if cand.exists():
                return str(cand)

    return path_str


def _load_weights_best_effort(model: torch.nn.Module, sd: dict, label: str) -> dict:
    """
    更健壮的权重加载：
    - 自动尝试剥离常见前缀（model./module./...）
    - 打印匹配率、missing/unexpected
    - 关键模块未加载时直接报错（避免“采样全噪点”还继续训练）
    """
    model_keys = set(model.state_dict().keys())
    sd_keys = list(sd.keys())
    prefixes, matched = _pick_best_prefix_remap(sd_keys, model_keys)
    remapped = {_strip_prefixes(k, prefixes): v for k, v in sd.items()}

    incompatible = model.load_state_dict(remapped, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])

    matched_after = len(set(remapped.keys()) & model_keys)
    coverage = matched_after / max(1, len(model_keys))
    remap_name = "+".join(prefixes) if prefixes else "none"

    logger.info(
        f"{label} 权重加载: remap={remap_name}, 匹配 {matched_after}/{len(model_keys)} ({coverage:.1%}), "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )

    # 关键层缺失会直接导致输出接近 0，采样就是纯噪点
    critical_prefixes = ("x_embedder.", "blocks.", "final_layer.")
    critical_missing = [k for k in missing if k.startswith(critical_prefixes)]
    if coverage < 0.60 or len(critical_missing) > 0:
        preview_missing = ", ".join(critical_missing[:8])
        raise RuntimeError(
            f"{label} 权重看起来没有正确加载（remap={remap_name}, coverage={coverage:.1%}）。"
            f"关键参数缺失: {preview_missing or 'N/A'}。\n"
            f"这通常表示你选错了 .safetensors（不是完整 transformer/vae 权重），或 checkpoint key 前缀不匹配。"
        )
    return {
        "remap": remap_name,
        "coverage": coverage,
        "missing": missing,
        "unexpected": unexpected,
    }


def ensure_models_namespace(repo_root):
    """确保 models 命名空间可用"""
    repo_root = Path(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(repo_root.parent) not in sys.path:
        sys.path.insert(0, str(repo_root.parent))


def load_anima_model(transformer_path, device, dtype, repo_root):
    """加载 Anima transformer 模型"""
    from safetensors import safe_open

    ensure_models_namespace(repo_root)

    # 加载模型类
    cosmos_modeling = load_module_from_path(
        "cosmos_predict2_modeling",
        repo_root / "cosmos_predict2_modeling.py",
    )
    anima_modeling = load_module_from_path(
        "anima_modeling",
        repo_root / "anima_modeling.py",
    )
    Anima = anima_modeling.Anima

    # 从 checkpoint 推断配置
    with safe_open(transformer_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith("x_embedder.proj.1.weight"):
                w = f.get_tensor(k)
                break

    in_channels = (w.shape[1] // 4) - 1  # concat_padding_mask=True
    model_channels = w.shape[0]

    if model_channels == 2048:
        num_blocks, num_heads = 28, 16
    elif model_channels == 5120:
        num_blocks, num_heads = 36, 40
    else:
        raise RuntimeError(f"未知的 model_channels={model_channels}")

    config = dict(
        max_img_h=240, max_img_w=240, max_frames=128,
        in_channels=in_channels, out_channels=16,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        model_channels=model_channels,
        num_blocks=num_blocks, num_heads=num_heads,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d", pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=True, adaln_lora_dim=256,
        rope_h_extrapolation_ratio=4.0 if in_channels == 16 else 3.0,
        rope_w_extrapolation_ratio=4.0 if in_channels == 16 else 3.0,
        rope_t_extrapolation_ratio=1.0,
    )

    model = Anima(**config)

    # 加载权重
    sd = _load_safetensors_state_dict(Path(transformer_path))
    info = _load_weights_best_effort(model, sd, label="Transformer")

    # 如果 checkpoint 中完全没有 llm_adapter 权重，随机初始化会把 cross-attn 条件搞乱，直接禁用更安全
    has_llm_adapter = any("llm_adapter" in k for k in sd.keys())
    if not has_llm_adapter and hasattr(model, "llm_adapter"):
        try:
            model.llm_adapter = None
            logger.warning("检测到 checkpoint 不包含 llm_adapter 权重：已禁用 llm_adapter（回退为直接使用 Qwen embeddings）")
        except Exception:
            pass
    model = model.to(device=device, dtype=dtype)
    model.requires_grad_(False)

    logger.info(f"Anima 模型加载完成: {model_channels}ch, {num_blocks} blocks")
    return model


def load_vae(vae_path, device, dtype, repo_root):
    """加载 VAE"""
    wan_vae = load_module_from_path("wan_vae", repo_root / "wan" / "vae2_1.py")
    WanVAE = wan_vae.WanVAE_

    cfg = dict(
        dim=96, z_dim=16, dim_mult=[1, 2, 4, 4],
        num_res_blocks=2, attn_scales=[],
        temperal_downsample=[False, True, True], dropout=0.0,
    )

    model = WanVAE(**cfg).eval().requires_grad_(False)

    sd = _load_safetensors_state_dict(Path(vae_path))
    _load_weights_best_effort(model, sd, label="VAE")
    model = model.to(device=device, dtype=dtype)

    # VAE 归一化参数
    mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ], dtype=dtype, device=device)
    std = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ], dtype=dtype, device=device)

    class VAEWrapper:
        pass

    wrapper = VAEWrapper()
    wrapper.model = model
    wrapper.mean = mean
    wrapper.std = std
    wrapper.scale = [mean, 1.0 / std]

    logger.info("VAE 加载完成")
    return wrapper


def load_text_encoders(qwen_path, t5_tokenizer_path, device, dtype):
    """加载文本编码器"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    # Qwen
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        qwen_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval().requires_grad_(False)

    # T5 tokenizer
    if t5_tokenizer_path and Path(t5_tokenizer_path).exists():
        t5_tokenizer = T5Tokenizer.from_pretrained(t5_tokenizer_path)
    else:
        t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")

    logger.info("文本编码器加载完成")
    return qwen_model, qwen_tokenizer, t5_tokenizer


# ============================================================================
# 文本编码
# ============================================================================

def encode_qwen(model, tokenizer, texts, device, max_length=512):
    """Qwen 文本编码"""
    # Qwen3 tokenizer 对空字符串可能返回 0 tokens（会导致模型内部 reshape 失败）
    # ComfyUI 的 AnimaTokenizer 设置了 min_length=1，这里做同等兜底。
    if isinstance(texts, str):
        texts = [texts]
    texts = [(" " if (t is None or str(t).strip() == "") else str(t)) for t in texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    # 仍可能出现空序列（极端 tokenizer 行为），强制塞 1 个 token
    if inputs["input_ids"].ndim == 2 and inputs["input_ids"].shape[1] == 0:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        bs = len(texts)
        inputs["input_ids"] = torch.full((bs, 1), int(pad_id), dtype=torch.long)
        inputs["attention_mask"] = torch.ones((bs, 1), dtype=torch.long)
    inputs = inputs.to(device)

    with torch.inference_mode():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden = outputs.hidden_states[-1]
    # 清零 padding 位置
    mask = inputs["attention_mask"].unsqueeze(-1)
    hidden = hidden * mask

    return hidden, inputs["attention_mask"]


def _time_snr_shift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    """ComfyUI ModelSamplingDiscreteFlow.time_snr_shift"""
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def _flow_sigmas_simple(steps: int, *, shift: float = 3.0, timesteps: int = 1000, device: str = "cpu") -> torch.Tensor:
    """
    复刻 ComfyUI:
    - supported_models.Anima 的 sampling_settings: shift=3.0, multiplier=1.0
    - ModelSamplingDiscreteFlow + simple_scheduler(model_sampling, steps)

    返回：sigmas (steps+1,) float32，从高到低，末尾带 0.0
    """
    ts = torch.arange(1, timesteps + 1, device=device, dtype=torch.float32) / float(timesteps)  # (0, 1]
    sigmas_full = _time_snr_shift(float(shift), ts)  # (0, 1]

    ss = len(sigmas_full) / float(steps)
    sigmas = [float(sigmas_full[-(1 + int(i * ss))]) for i in range(steps)]
    sigmas.append(0.0)
    sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)

    # ComfyUI offset_first_sigma_for_snr: CONST 下避免 sigma=1 导致 logit inf
    if sigmas.numel() > 0 and sigmas[0] >= 1.0:
        sigmas[0] = float(_time_snr_shift(float(shift), torch.tensor(1.0 - 1e-4, device=device, dtype=torch.float32)))
    return sigmas


def _default_noise_sampler(x: torch.Tensor, seed: int | None):
    """参考 ComfyUI k_diffusion_sampling.default_noise_sampler"""
    if seed is not None:
        if x.device.type == "cpu":
            seed = int(seed) + 1
        g = torch.Generator(device=x.device)
        g.manual_seed(int(seed))
    else:
        g = None

    def _sample(_sigma, _sigma_next):
        return torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=g)

    return _sample


@torch.no_grad()
def _sample_er_sde_const_x0(
    denoise_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    seed: int | None = None,
    s_noise: float = 1.0,
    max_stage: int = 3,
):
    """
    Extended Reverse-Time SDE solver（ER-SDE-Solver-3）在 CONST(flow) 噪声日程下的实现。
    参考 ComfyUI 的 k_diffusion_sampling.sample_er_sde（删去 model_patcher 依赖）。
    """
    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    if sigmas.numel() <= 1:
        return x

    noise_sampler = _default_noise_sampler(x, seed=seed)

    # CONST: half_log_snr = log((1 - t) / t) = -logit(t)
    eps = 1e-12
    t = sigmas.clamp(min=eps, max=1.0 - eps)
    half_log_snrs = torch.log((1 - t) / t)
    er_lambdas = half_log_snrs.neg().exp()  # er_lambda = t / (1 - t)

    old_denoised = None
    old_denoised_d = None

    def noise_scaler(lam: torch.Tensor) -> torch.Tensor:
        # default_er_sde_noise_scaler
        lam = lam.to(x.device, dtype=torch.float32)
        return lam * ((lam ** 0.3).exp() + 10.0)

    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        denoised = denoise_fn(x, sigma)

        stage_used = min(int(max_stage), i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = 1.0 - sigmas[i]
            alpha_t = 1.0 - sigmas[i + 1]
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

            # Stage 1 (Euler)
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised

            if stage_used >= 2 and old_denoised is not None:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = noise_scaler(lambda_pos)

                # Stage 2
                s = torch.sum(1 / scaled_pos) * lambda_step_size
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d

                if stage_used >= 3 and old_denoised_d is not None:
                    # Stage 3
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)) * denoised_u

                old_denoised_d = denoised_d

            # Stochastic term
            if s_noise and float(s_noise) > 0:
                noise = noise_sampler(float(sigmas[i]), float(sigmas[i + 1]))
                sde_scale = (er_lambda_t ** 2 - (er_lambda_s ** 2) * (r ** 2)).clamp(min=0).sqrt().nan_to_num(nan=0.0)
                x = x + alpha_t * noise * float(s_noise) * sde_scale

        old_denoised = denoised

    return x


def _parse_weighted_tag(tag: str) -> tuple[str, float]:
    """
    解析单个 tag 的权重（参考指南“权重控制”）。
    支持：
    - (tag:1.5)
    - (tag) / ((tag))  => 1.1^n
    - [tag]            => 1/1.1
    """
    import re

    s = tag.strip()
    if not s:
        return "", 1.0

    # 显式 (xxx:1.23)
    m = re.fullmatch(r"\(\s*(.+?)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*\)", s)
    if m:
        return m.group(1).strip(), float(m.group(2))

    # 统计外层 () / [] 深度
    w = 1.0
    while True:
        s2 = s.strip()
        if len(s2) >= 2 and s2[0] == "(" and s2[-1] == ")":
            s = s2[1:-1].strip()
            w *= 1.1
            continue
        if len(s2) >= 2 and s2[0] == "[" and s2[-1] == "]":
            s = s2[1:-1].strip()
            w /= 1.1
            continue
        break
    return s.strip(), float(w)


def _build_qwen_text_from_prompt(prompt: str) -> str:
    # Qwen 通道不传权重，只传“干净标签文本”（参考 ComfyUI anima-kai 的做法）
    parts = [p.strip() for p in prompt.split(",") if p.strip()]
    clean = []
    for p in parts:
        t, _w = _parse_weighted_tag(p)
        if t:
            clean.append(t)
    return ", ".join(clean)


def tokenize_t5_weighted(tokenizer, texts, max_length=512):
    """
    参考 ComfyUI 的 anima-kai：按逗号切分 tag，逐 tag 分词，并为每个 token 附带权重。
    返回：input_ids, attention_mask(1=有效), token_weights
    """
    import torch

    if isinstance(texts, str):
        texts = [texts]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    all_ids = []
    all_w = []
    for text in texts:
        tags = [t.strip() for t in str(text).split(",") if t.strip()]
        ids = []
        ws = []
        for tag in tags:
            clean_tag, weight = _parse_weighted_tag(tag)
            if not clean_tag:
                continue
            tok = tokenizer(clean_tag, add_special_tokens=False)
            for tid in tok["input_ids"]:
                ids.append(int(tid))
                ws.append(float(weight))

        # 末尾补一个 eos（ComfyUI 也是最后加一个终止 token）
        ids.append(int(eos_id))
        ws.append(1.0)

        # 截断到 max_length（保留最后一个 eos）
        if max_length and len(ids) > max_length:
            ids = ids[: max_length - 1] + [int(eos_id)]
            ws = ws[: max_length - 1] + [1.0]

        all_ids.append(torch.tensor(ids, dtype=torch.long))
        all_w.append(torch.tensor(ws, dtype=torch.float32))

    # pad 到 batch 内最长
    max_len = max(x.numel() for x in all_ids) if all_ids else 1
    input_ids = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long)
    token_w = torch.zeros((len(all_w), max_len), dtype=torch.float32)
    attention_mask = torch.zeros((len(all_ids), max_len), dtype=torch.long)

    for i, (ids, ws) in enumerate(zip(all_ids, all_w)):
        L = ids.numel()
        input_ids[i, :L] = ids
        token_w[i, :L] = ws
        attention_mask[i, :L] = 1

    return input_ids, attention_mask, token_w


# ============================================================================
# LoRA 实现
# ============================================================================

class LoRALayer(torch.nn.Module):
    """标准 LoRA 层"""
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_down = torch.nn.Linear(in_features, rank, bias=False)
        self.lora_up = torch.nn.Linear(rank, out_features, bias=False)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        torch.nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.lora_up(self.dropout(self.lora_down(x))) * self.scaling


class LoKrLayer(torch.nn.Module):
    """LyCORIS LoKr 层 (ComfyUI 兼容) — w2 低秩分解版"""
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, factor=8, dropout=0.0):
        super().__init__()
        self.alpha = alpha

        # 自动调整 factor 确保能整除
        factor = self._find_factor(in_features, out_features, factor)
        self.factor = factor

        self.in_dim = in_features // factor
        self.out_dim = out_features // factor

        # cap rank 防止小层溢出
        self.rank = min(rank, self.out_dim, self.in_dim)
        self.scaling = alpha / self.rank

        # LoKr 分解: W = kron(w1, w2_a @ w2_b) * scaling
        # w1: [factor, factor]
        # w2_a: [out_dim, rank], w2_b: [rank, in_dim]
        self.lokr_w1 = torch.nn.Parameter(torch.empty(factor, factor))
        self.lokr_w2_a = torch.nn.Parameter(torch.empty(self.out_dim, self.rank))
        self.lokr_w2_b = torch.nn.Parameter(torch.empty(self.rank, self.in_dim))
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

        # LyCORIS 标准初始化: w1=kaiming, w2_a=kaiming, w2_b=zeros
        torch.nn.init.kaiming_uniform_(self.lokr_w1, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=5**0.5)
        torch.nn.init.zeros_(self.lokr_w2_b)

    def _find_factor(self, in_f, out_f, target_factor):
        """找到能同时整除 in_features 和 out_features 的 factor"""
        for f in [target_factor, 4, 2, 1]:
            if in_f % f == 0 and out_f % f == 0:
                return f
        return 1

    def forward(self, x):
        w2 = self.lokr_w2_a @ self.lokr_w2_b
        weight = torch.kron(self.lokr_w1, w2)
        return F.linear(self.dropout(x), weight) * self.scaling


class LoRALinear(torch.nn.Module):
    """LoRA 包装的 Linear 层"""
    def __init__(self, original, rank=4, alpha=1.0, dropout=0.0, use_lokr=False, factor=8):
        super().__init__()
        self.original = original
        self.use_lokr = use_lokr

        if use_lokr:
            self.adapter = LoKrLayer(
                original.in_features, original.out_features,
                rank=rank, alpha=alpha, factor=factor, dropout=dropout
            )
        else:
            self.adapter = LoRALayer(
                original.in_features, original.out_features,
                rank=rank, alpha=alpha, dropout=dropout
            )

        self.adapter.to(device=original.weight.device, dtype=original.weight.dtype)
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.adapter(x)

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias


class LoRAInjector:
    """LoRA 注入器"""
    DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "output_proj", "mlp.layer1", "mlp.layer2"]

    def __init__(self, rank=32, alpha=16.0, dropout=0.0, use_lokr=False, factor=8, targets=None):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.use_lokr = use_lokr
        self.factor = factor
        self.targets = targets or self.DEFAULT_TARGETS
        self.injected = {}

    def inject(self, model):
        """注入 LoRA 到模型"""
        for name, module in list(model.named_modules()):
            if not isinstance(module, torch.nn.Linear):
                continue
            if not any(t in name for t in self.targets):
                continue

            lora_linear = LoRALinear(
                module, rank=self.rank, alpha=self.alpha,
                dropout=self.dropout, use_lokr=self.use_lokr, factor=self.factor
            )

            # 替换模块
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], lora_linear)
            self.injected[name] = lora_linear

        logger.info(f"注入 {'LoKr' if self.use_lokr else 'LoRA'} 到 {len(self.injected)} 层")
        return self.injected

    def get_params(self):
        """获取可训练参数"""
        params = []
        for lora in self.injected.values():
            params.extend(lora.adapter.parameters())
        return params

    def get_param_groups(self, weight_decay):
        """获取参数组，LoKr 模式下 w1 排除 weight_decay"""
        if not self.use_lokr or weight_decay == 0:
            return [{"params": self.get_params(), "weight_decay": weight_decay}]

        no_decay_params = []  # w1
        decay_params = []     # w2_a, w2_b
        for lora in self.injected.values():
            no_decay_params.append(lora.adapter.lokr_w1)
            decay_params.append(lora.adapter.lokr_w2_a)
            decay_params.append(lora.adapter.lokr_w2_b)

        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def state_dict(self):
        """导出 LoRA 权重 (ComfyUI 兼容格式)"""
        sd = {}
        for name, lora in self.injected.items():
            # ComfyUI 格式: lora_unet_{key} 其中 key 是模型路径的 . 替换成 _
            base = "lora_unet_" + name.replace(".", "_")
            sd[f"{base}.alpha"] = torch.tensor(self.alpha)

            if self.use_lokr:
                sd[f"{base}.lokr_w1"] = lora.adapter.lokr_w1.data.clone()
                sd[f"{base}.lokr_w2_a"] = lora.adapter.lokr_w2_a.data.clone()
                sd[f"{base}.lokr_w2_b"] = lora.adapter.lokr_w2_b.data.clone()
            else:
                sd[f"{base}.lora_down.weight"] = lora.adapter.lora_down.weight.data.clone()
                sd[f"{base}.lora_up.weight"] = lora.adapter.lora_up.weight.data.clone()
        return sd

    def save(self, path):
        """保存为 safetensors (ComfyUI 兼容)"""
        from safetensors.torch import save_file
        sd = self.state_dict()
        meta = {
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(self.alpha),
            "ss_network_module": "lycoris.kohya" if self.use_lokr else "networks.lora",
            "ss_network_args": f'{{"algo": "lokr", "factor": {self.factor}}}' if self.use_lokr else "{}",
        }
        save_file(sd, path, metadata=meta)
        logger.info(f"LoRA 保存到: {path}")

    def load(self, path):
        """从 safetensors 加载已有 LoRA 权重（用于继续训练）"""
        from safetensors import safe_open
        
        logger.info(f"加载已有 LoRA 权重: {path}")
        
        # 读取权重
        sd = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        
        loaded_count = 0
        for name, lora in self.injected.items():
            base = "lora_unet_" + name.replace(".", "_")
            
            if self.use_lokr:
                w1_key = f"{base}.lokr_w1"
                w2a_key = f"{base}.lokr_w2_a"
                w2b_key = f"{base}.lokr_w2_b"
                w2_old_key = f"{base}.lokr_w2"
                if w1_key in sd and w2a_key in sd and w2b_key in sd:
                    lora.adapter.lokr_w1.data.copy_(sd[w1_key])
                    lora.adapter.lokr_w2_a.data.copy_(sd[w2a_key])
                    lora.adapter.lokr_w2_b.data.copy_(sd[w2b_key])
                    loaded_count += 1
                elif w1_key in sd and w2_old_key in sd:
                    logger.warning(f"跳过旧格式 lokr_w2 全矩阵层: {name}（需重新训练）")
            else:
                down_key = f"{base}.lora_down.weight"
                up_key = f"{base}.lora_up.weight"
                if down_key in sd and up_key in sd:
                    lora.adapter.lora_down.weight.data.copy_(sd[down_key])
                    lora.adapter.lora_up.weight.data.copy_(sd[up_key])
                    loaded_count += 1

        logger.info(f"从 checkpoint 加载了 {loaded_count}/{len(self.injected)} 层 LoRA 权重")


# ============================================================================
# 训练状态保存/恢复（断点续训）
# ============================================================================

def save_training_state(path, injector, optimizer, epoch, global_step, loss_history=None, rng_state=None, monitor_state=None, scheduler=None):
    """保存完整训练状态，支持断点续训"""
    state = {
        "lora_state_dict": injector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "loss_history": loss_history or [],
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "random": random.getstate(),
        },
        "monitor_state": monitor_state,  # 保存监控面板数据（用于恢复 loss 曲线）
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)
    logger.info(f"训练状态已保存: {path} (epoch={epoch}, step={global_step})")


def load_training_state(path, injector, optimizer, scheduler=None):
    """加载训练状态，返回 (epoch, global_step, loss_history, monitor_state)"""
    logger.info(f"加载训练状态: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    
    # 加载 LoRA 权重
    lora_sd = state["lora_state_dict"]
    for name, lora in injector.injected.items():
        base = "lora_unet_" + name.replace(".", "_")
        if injector.use_lokr:
            w1_key = f"{base}.lokr_w1"
            w2a_key = f"{base}.lokr_w2_a"
            w2b_key = f"{base}.lokr_w2_b"
            w2_old_key = f"{base}.lokr_w2"
            if w1_key in lora_sd and w2a_key in lora_sd and w2b_key in lora_sd:
                lora.adapter.lokr_w1.data.copy_(lora_sd[w1_key])
                lora.adapter.lokr_w2_a.data.copy_(lora_sd[w2a_key])
                lora.adapter.lokr_w2_b.data.copy_(lora_sd[w2b_key])
            elif w1_key in lora_sd and w2_old_key in lora_sd:
                logger.warning(f"跳过旧格式 lokr_w2 全矩阵层: {name}（需重新训练）")
        else:
            down_key = f"{base}.lora_down.weight"
            up_key = f"{base}.lora_up.weight"
            if down_key in lora_sd and up_key in lora_sd:
                lora.adapter.lora_down.weight.data.copy_(lora_sd[down_key])
                lora.adapter.lora_up.weight.data.copy_(lora_sd[up_key])
    
    # 加载优化器状态
    optimizer.load_state_dict(state["optimizer_state_dict"])

    # 加载调度器状态
    if scheduler is not None and "scheduler_state_dict" in state:
        try:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        except Exception as e:
            logger.warning(f"调度器状态恢复失败（将从头开始）: {e}")

    # 恢复随机数状态
    if "rng_state" in state:
        rng = state["rng_state"]
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"])
        if rng.get("random") is not None:
            random.setstate(rng["random"])
    
    epoch = state.get("epoch", 0)
    global_step = state.get("global_step", 0)
    loss_history = state.get("loss_history", [])
    monitor_state = state.get("monitor_state", None)  # 恢复监控数据
    
    logger.info(f"训练状态已恢复: epoch={epoch}, step={global_step}")
    return epoch, global_step, loss_history, monitor_state


# ============================================================================
# 数据集
# ============================================================================

class BucketManager:
    """ARB 分桶管理"""
    def __init__(self, base_reso=1024, min_reso=512, max_reso=2048, step=64):
        self.base_reso = base_reso
        self.buckets = self._generate(min_reso, max_reso, step, base_reso)

    def _generate(self, min_r, max_r, step, base):
        buckets = []
        base_area = base * base
        for w in range(min_r, max_r + 1, step):
            for h in range(min_r, max_r + 1, step):
                if abs(w * h - base_area) / base_area > 0.1:
                    continue
                if max(w/h, h/w) > 2.0:
                    continue
                buckets.append((w, h))
        return buckets

    def get_bucket(self, w, h):
        aspect = w / h
        best = (self.base_reso, self.base_reso)
        best_diff = float("inf")
        for bw, bh in self.buckets:
            diff = abs(aspect - bw/bh)
            if diff < best_diff:
                best_diff = diff
                best = (bw, bh)
        return best


class ImageDataset(Dataset):
    """
    图像数据集
    
    支持两种 caption 格式：
    1. JSON 文件（优先）- 支持分类 shuffle
    2. TXT 文件（回退）- 传统 shuffle
    """
    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, data_dir, resolution=1024, bucket_mgr=None,
                 shuffle_caption=False, keep_tokens=0, flip_augment=False,
                 tag_dropout=0.0, prefer_json=True, caption_override=None):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.bucket_mgr = bucket_mgr
        self.shuffle_caption = shuffle_caption
        self.keep_tokens = keep_tokens
        self.flip_augment = flip_augment
        self.tag_dropout = tag_dropout
        self.prefer_json = prefer_json
        self.caption_override = caption_override  # 正则集：统一 caption，如 "1girl, solo"
        
        # 尝试导入 caption_utils（直接导入避开 __init__.py）
        self.caption_utils = None
        if prefer_json:
            try:
                import importlib.util
                import sys
                
                # 直接加载 caption_utils.py
                utils_path = Path(__file__).parent / "utils" / "caption_utils.py"
                if utils_path.exists():
                    spec = importlib.util.spec_from_file_location("caption_utils", utils_path)
                    caption_module = importlib.util.module_from_spec(spec)
                    sys.modules["caption_utils"] = caption_module
                    spec.loader.exec_module(caption_module)
                    
                    self.caption_utils = {
                        "load_and_build": caption_module.load_and_build_caption,
                        "load_json": caption_module.load_caption_json,
                        "normalize": caption_module.normalize_caption_json,
                        "build": caption_module.build_caption_from_json,
                    }
                    logger.info("JSON caption 模式已启用（分类 shuffle）")
                else:
                    logger.warning(f"caption_utils.py 未找到: {utils_path}")
            except Exception as e:
                logger.warning(f"caption_utils 加载失败: {e}，回退到 TXT 模式")
        
        self.samples = self._scan()
        json_count = sum(1 for s in self.samples if s.get("json_path"))
        txt_count = len(self.samples) - json_count
        unique_count = len(set(id(s) for s in self.samples))
        logger.info(f"数据集: {unique_count} 张图 → {len(self.samples)} 样本（含 repeat）(JSON: {json_count}, TXT: {txt_count})")

    @staticmethod
    def _parse_repeats_from_dir(name: str) -> int:
        """从文件夹名解析 Kohya 风格重复次数，如 '5_concept' → 5"""
        prefix = name.split("_", 1)[0]
        if prefix.isdigit():
            return max(int(prefix), 1)
        return 1

    def _make_sample(self, img_path):
        """为单张图构建 sample dict，找不到 caption 返回 None"""
        sample = {"image": img_path}
        json_path = img_path.with_suffix(".json")
        if self.prefer_json and json_path.exists():
            sample["json_path"] = json_path
            sample["txt_path"] = None
        else:
            txt_path = img_path.with_suffix(".txt")
            if not txt_path.exists():
                txt_path = img_path.with_suffix(".caption")
            if not txt_path.exists():
                return None
            sample["json_path"] = None
            sample["txt_path"] = txt_path
        return sample

    def _scan(self):
        """扫描数据集目录，支持 Kohya 风格文件夹重复。

        目录结构示例::

            dataset/
            ├── 1_old/       ← repeat 1
            │   ├── img.jpg
            │   └── img.txt
            └── 5_new/       ← repeat 5
                ├── img.jpg
                └── img.txt

        没有数字前缀的文件夹或根目录下的图片按 repeat=1 处理。
        """
        unique_samples = []
        folder_info = []  # (folder_name, repeat, count) for logging

        # 收集根目录下的图片（repeat=1）
        root_count = 0
        for p in sorted(self.data_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in self.EXTS:
                s = self._make_sample(p)
                if s:
                    s["_repeat"] = 1
                    unique_samples.append(s)
                    root_count += 1
        if root_count:
            folder_info.append(("(root)", 1, root_count))

        # 收集子文件夹中的图片（解析 repeat）
        for subdir in sorted(self.data_dir.iterdir()):
            if not subdir.is_dir():
                continue
            repeats = self._parse_repeats_from_dir(subdir.name)
            count = 0
            for img_path in sorted(subdir.rglob("*")):
                if img_path.suffix.lower() not in self.EXTS:
                    continue
                s = self._make_sample(img_path)
                if s:
                    s["_repeat"] = repeats
                    unique_samples.append(s)
                    count += 1
            if count:
                folder_info.append((subdir.name, repeats, count))

        # 展开 repeat：将每个样本按其 repeat 次数复制
        samples = []
        for s in unique_samples:
            r = s.pop("_repeat", 1)
            for _ in range(r):
                samples.append(s)

        # 日志：每个文件夹的 repeat 信息
        if folder_info:
            for name, rep, cnt in folder_info:
                logger.info(f"  文件夹 {name}: {cnt} 张 × repeat {rep} = {cnt * rep} 样本")

        return samples

    def _process_caption_txt(self, caption):
        """处理 TXT caption: 传统 tag 打乱 + keep_tokens"""
        if not caption:
            return ""
        if "," in caption:
            tags = [t.strip() for t in caption.split(",")]
        else:
            tags = caption.split()

        if self.keep_tokens > 0:
            kept = tags[:self.keep_tokens]
            rest = tags[self.keep_tokens:]
            if self.shuffle_caption:
                random.shuffle(rest)
            tags = kept + rest
        elif self.shuffle_caption:
            random.shuffle(tags)

        return ", ".join(tags)

    def _process_caption_json(self, json_path):
        """处理 JSON caption: 分类 shuffle"""
        if self.caption_utils is None:
            return None
        
        try:
            raw_json = self.caption_utils["load_json"](json_path)
            if raw_json is None:
                return None
            
            # 检查是否已经是标准格式
            if "tags" in raw_json and "meta" in raw_json:
                normalized = raw_json
            else:
                normalized = self.caption_utils["normalize"](raw_json)
            
            # 构建 caption（分类 shuffle）
            return self.caption_utils["build"](
                normalized,
                shuffle_appearance=self.shuffle_caption,
                shuffle_tags=self.shuffle_caption,
                shuffle_environment=self.shuffle_caption,
                tag_dropout=self.tag_dropout,
            )
        except Exception as e:
            logger.warning(f"JSON 处理失败 {json_path}: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import numpy as np
        from PIL import Image
        sample = self.samples[idx]
        img = Image.open(sample["image"]).convert("RGB")
        
        # 获取 caption（正则集可用 caption_override 统一覆盖）
        caption = None
        if self.caption_override is not None:
            caption = self.caption_override
        elif sample.get("json_path"):
            caption = self._process_caption_json(sample["json_path"])
        
        if caption is None and sample.get("txt_path"):
            caption = sample["txt_path"].read_text(encoding="utf-8").strip()
            caption = self._process_caption_txt(caption)
        
        if caption is None:
            caption = ""

        # ARB 分桶
        if self.bucket_mgr:
            tw, th = self.bucket_mgr.get_bucket(img.width, img.height)
        else:
            tw = th = self.resolution

        # 缩放裁剪
        scale = max(tw / img.width, th / img.height)
        nw, nh = int(img.width * scale), int(img.height * scale)
        img = img.resize((nw, nh), Image.LANCZOS)

        left = (nw - tw) // 2
        top = (nh - th) // 2
        img = img.crop((left, top, left + tw, top + th))

        # 水平翻转增强
        if self.flip_augment and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 转 tensor [-1, 1]
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        return {"pixel_values": tensor, "caption": caption}


class RepeatDataset(Dataset):
    """Kohya 风格数据集重复"""
    def __init__(self, dataset, repeats=1):
        self.dataset = dataset
        self.repeats = max(1, int(repeats))

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


class MergedDataset(Dataset):
    """合并主数据集与正则数据集（Kohya 风格 reg）"""
    def __init__(self, main_dataset, reg_dataset, reg_weight: float = 1.0):
        self.main_dataset = main_dataset
        self.reg_dataset = reg_dataset
        self.reg_weight = float(reg_weight)
        self._main_len = len(main_dataset)
        self._reg_len = len(reg_dataset)

        # 为 BucketBatchSampler 构建 bucket_for_index
        self.bucket_for_index = self._build_bucket_for_index()

    def _get_cached_dataset(self, d):
        if hasattr(d, "bucket_for_index"):
            return d
        if hasattr(d, "dataset"):
            return self._get_cached_dataset(d.dataset)
        return None

    def _build_bucket_for_index(self):
        main_cached = self._get_cached_dataset(self.main_dataset)
        reg_cached = self._get_cached_dataset(self.reg_dataset)
        buckets = []
        if main_cached and main_cached.bucket_for_index:
            main_base_len = len(main_cached.bucket_for_index)
            for idx in range(self._main_len):
                b = main_cached.bucket_for_index[idx % main_base_len]
                buckets.append(b if b is not None else (0, 0))
        else:
            buckets.extend([(0, 0)] * self._main_len)
        if reg_cached and reg_cached.bucket_for_index:
            reg_base_len = len(reg_cached.bucket_for_index)
            for idx in range(self._reg_len):
                b = reg_cached.bucket_for_index[idx % reg_base_len]
                buckets.append(b if b is not None else (0, 0))
        else:
            buckets.extend([(0, 0)] * self._reg_len)
        return buckets

    def __len__(self):
        return self._main_len + self._reg_len

    def __getitem__(self, idx):
        if idx < self._main_len:
            item = self.main_dataset[idx]
            item["loss_weight"] = 1.0
            return item
        item = self.reg_dataset[idx - self._main_len]
        item["loss_weight"] = self.reg_weight
        return item


class BucketBatchSampler:
    """Batch sampler that groups samples by bucket so latents in each batch have the same size."""
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self._cached_dataset = self._get_cached_dataset(dataset)
        self._base_len = len(self._cached_dataset) if self._cached_dataset else 0

    def _get_cached_dataset(self, d):
        if hasattr(d, "bucket_for_index"):
            return d
        if hasattr(d, "dataset"):
            return self._get_cached_dataset(d.dataset)
        return None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        if self._cached_dataset is None:
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch
            return

        bucket_to_indices = {}
        for idx in range(len(self.dataset)):
            base_idx = idx % self._base_len
            bucket = self._cached_dataset.bucket_for_index[base_idx]
            if bucket is None:
                bucket = (0, 0)
            bucket_to_indices.setdefault(bucket, []).append(idx)

        buckets = list(bucket_to_indices.keys())
        if self.shuffle:
            rng.shuffle(buckets)
        for bucket in buckets:
            indices = bucket_to_indices[bucket]
            if self.shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch


class CachedLatentDataset(Dataset):
    """Kohya 风格 npz 文件缓存的数据集"""
    def __init__(self, base_dataset, vae, device, dtype, cache_dir=None):
        import numpy as np
        self.base_dataset = base_dataset
        self.np = np
        # 获取原始数据集的 samples 列表
        self.samples = self._get_base_samples(base_dataset)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.bucket_for_index = []
        self._build_cache(vae, device, dtype)

    def _get_base_samples(self, dataset):
        """获取原始 ImageDataset 的 samples"""
        if hasattr(dataset, "samples"):
            return dataset.samples
        elif hasattr(dataset, "dataset"):
            return self._get_base_samples(dataset.dataset)
        return []

    def _get_npz_path(self, img_path):
        """获取图像对应的 npz 缓存路径"""
        img_path = Path(img_path)
        return img_path.with_suffix(".npz")

    def _is_cache_valid(self, img_path, npz_path):
        """检查缓存是否有效（图像未修改，且格式含 latent 键）。
        若为其他模型的不兼容缓存，则删除并返回 False。"""
        if not npz_path.exists():
            return False
        if npz_path.stat().st_mtime < img_path.stat().st_mtime:
            return False
        try:
            data = self.np.load(npz_path)
            if "latent" not in data.files:
                npz_path.unlink()
                logger.debug(f"已删除不兼容缓存: {npz_path.name}")
                return False
        except Exception:
            try:
                npz_path.unlink()
            except Exception:
                pass
            return False
        return True

    def _build_cache(self, vae, device, dtype):
        """构建/加载 npz 缓存"""
        logger.info("检查 VAE latent 缓存...")
        to_encode = []
        for i, sample in enumerate(self.samples):
            img_path = sample["image"]
            npz_path = self._get_npz_path(img_path)
            if not self._is_cache_valid(img_path, npz_path):
                to_encode.append(i)

        if to_encode:
            logger.info(f"需要编码 {len(to_encode)}/{len(self.samples)} 张图像...")
            self._encode_and_save(to_encode, vae, device, dtype)
        else:
            logger.info(f"所有 {len(self.samples)} 张图像已缓存")

        self._fill_bucket_for_index()

    def _fill_bucket_for_index(self):
        """Fill bucket_for_index for all samples (needed for BucketBatchSampler).
        Uses latent spatial shape (h, w) as grouping key so batches have consistent tensor sizes."""
        self.bucket_for_index = [None] * len(self.samples)
        for i in range(len(self.samples)):
            npz_path = self._get_npz_path(self.samples[i]["image"])
            if not npz_path.exists():
                continue
            data = self.np.load(npz_path)
            latent = data["latent"]
            s = latent.shape
            if len(s) == 5:
                _, _, _, h, w = s
            else:
                _, _, h, w = s
            self.bucket_for_index[i] = (int(h), int(w))

    def _encode_and_save(self, indices, vae, device, dtype):
        """编码图像并保存为 npz"""
        for count, i in enumerate(indices):
            item = self.base_dataset[i]
            pixels = item["pixel_values"].unsqueeze(0).to(device, dtype=dtype)
            _, _, ph, pw = pixels.shape
            bucket_w, bucket_h = pw, ph
            with torch.no_grad():
                pixels_5d = pixels.unsqueeze(2)
                latent = vae.model.encode(pixels_5d, vae.scale)
            latent_np = latent.squeeze(0).cpu().float().numpy()
            npz_path = self._get_npz_path(self.samples[i]["image"])
            self.np.savez(npz_path, latent=latent_np, bucket_w=bucket_w, bucket_h=bucket_h)
            if (count + 1) % 10 == 0 or count == len(indices) - 1:
                logger.info(f"  编码进度: {count + 1}/{len(indices)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        npz_path = self._get_npz_path(sample["image"])
        data = self.np.load(npz_path)
        latent = torch.from_numpy(data["latent"])
        
        # 获取 base_dataset 的引用（处理可能的嵌套）
        base = self.base_dataset
        while hasattr(base, "dataset"):
            base = base.dataset
        
        # 处理 caption（正则集 caption_override 优先）
        caption = None
        if getattr(base, "caption_override", None) is not None:
            caption = base.caption_override
        elif sample.get("json_path") and hasattr(base, "_process_caption_json"):
            caption = base._process_caption_json(sample["json_path"])
        
        if caption is None and sample.get("txt_path"):
            caption = sample["txt_path"].read_text(encoding="utf-8").strip()
            if hasattr(base, "_process_caption_txt"):
                caption = base._process_caption_txt(caption)
        
        if caption is None:
            caption = ""
        
        return {"latent": latent, "caption": caption}


# ============================================================================
# 训练时推理
# ============================================================================

@torch.no_grad()
def sample_image(
    model, vae, qwen_model, qwen_tokenizer, t5_tokenizer,
    prompt, height=1024, width=1024, steps=25, cfg_scale=4.0, 
    negative_prompt=None,
    sampler_name: str = "er_sde",
    scheduler: str = "simple",
    device="cuda",
    dtype=torch.bfloat16,
):
    """训练时采样预览（尽量对齐 ComfyUI KSampler）
    
    Args:
        negative_prompt: 负面提示词，默认使用标准负面提示词
        sampler_name: 采样器（推荐：er_sde）
        scheduler: 调度器（推荐：simple）
    """
    import numpy as np
    from PIL import Image
    model.eval()
    
    logger.info(f"[Debug] Sampling start. Prompt: {prompt[:50]}...")
    
    # Check VAE scale
    if isinstance(vae.scale, list) and len(vae.scale) == 2:
        m, s = vae.scale
        logger.info(f"[Debug] VAE scale: mean_shape={m.shape}, std_inv_shape={s.shape}")
        logger.info(f"[Debug] VAE scale values: mean={m.mean().item():.4f}, std_inv={s.mean().item():.4f}")

    # 默认负面提示词 (参考 Anima Prompt Guide)
    if negative_prompt is None:
        negative_prompt = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, bad anatomy, bad hands, bad feet, missing fingers, extra fingers, text, watermark, logo, signature, username, artist name, copyright name"

    # 文本编码
    try:
        # 有条件 (positive prompt)
        qwen_text = _build_qwen_text_from_prompt(prompt)
        qwen_embeds, qwen_attn = encode_qwen(qwen_model, qwen_tokenizer, [qwen_text], device)
        logger.info(f"[Debug] Qwen embeds: {qwen_embeds.shape}, mean={qwen_embeds.mean().item():.4f}")
        
        t5_ids, t5_attn, t5_w = tokenize_t5_weighted(t5_tokenizer, [prompt], max_length=512)
        t5_ids = t5_ids.to(device)
        t5_attn = t5_attn.to(device)
        t5_w = t5_w.to(device, dtype=torch.float32)
        cross_cond = model.preprocess_text_embeds(qwen_embeds, t5_ids)
        if cross_cond.shape[1] < 512:
            cross_cond = F.pad(cross_cond, (0, 0, 0, 512 - cross_cond.shape[1]))

        # 无条件/负面提示词 (negative prompt)
        qwen_text_uncond = _build_qwen_text_from_prompt(negative_prompt)
        qwen_embeds_uncond, qwen_attn_uncond = encode_qwen(qwen_model, qwen_tokenizer, [qwen_text_uncond], device)
        t5_ids_uncond, t5_attn_uncond, t5_w_uncond = tokenize_t5_weighted(t5_tokenizer, [negative_prompt], max_length=512)
        t5_ids_uncond = t5_ids_uncond.to(device)
        t5_attn_uncond = t5_attn_uncond.to(device)
        t5_w_uncond = t5_w_uncond.to(device, dtype=torch.float32)
        cross_uncond = model.preprocess_text_embeds(qwen_embeds_uncond, t5_ids_uncond)
        if cross_uncond.shape[1] < 512:
            cross_uncond = F.pad(cross_uncond, (0, 0, 0, 512 - cross_uncond.shape[1]))
            
    except Exception as e:
        logger.error(f"[Debug] Encoding failed: {e}")
        raise e

    # sigmas（对齐 ComfyUI supported_models.Anima: shift=3.0, multiplier=1.0）
    lat_h, lat_w = height // 8, width // 8
    if str(scheduler).lower() != "simple":
        logger.warning(f"采样 scheduler={scheduler} 未实现，回退 simple")
    sigmas = _flow_sigmas_simple(steps, shift=3.0, device=device)

    # 初始化噪声（ComfyUI CONST.noise_scaling: x = sigma*noise + (1-sigma)*latent_image；txt2img latent_image=0）
    x = torch.randn(1, 16, 1, lat_h, lat_w, device=device, dtype=torch.float32) * float(sigmas[0])
    logger.info(f"[Debug] Latents init: {x.shape}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")

    pad_mask = torch.zeros(1, 1, lat_h, lat_w, device=device, dtype=dtype)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    def denoise_fn(x_in: torch.Tensor, sigma_in: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(sigma_in):
            sigma_in = torch.tensor(float(sigma_in), device=x_in.device, dtype=torch.float32)
        sigma_b = sigma_in.view(1, 1).to(device=x_in.device, dtype=dtype)
        sigma_5d = sigma_in.view(1, 1, 1, 1, 1).to(device=x_in.device, dtype=torch.float32)

        with torch.autocast(device_type=device_type, dtype=dtype):
            v_cond = model(x_in.to(device=x_in.device, dtype=dtype), sigma_b, cross_cond, padding_mask=pad_mask)
            v_uncond = model(x_in.to(device=x_in.device, dtype=dtype), sigma_b, cross_uncond, padding_mask=pad_mask)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)

        if torch.isnan(v).any():
            raise RuntimeError("v contains NaN during sampling")

        # CONST(flow): denoised x0 = x - sigma * v
        return x_in - sigma_5d * v.float()

    sampler_name_l = str(sampler_name).lower().strip()
    logger.info(f"[Debug] Sampler={sampler_name_l}, Scheduler=simple, steps={steps}, cfg={cfg_scale}")

    if sampler_name_l == "er_sde":
        x = _sample_er_sde_const_x0(denoise_fn, x, sigmas, seed=None, s_noise=1.0, max_stage=3)
    else:
        # fallback: 简化 Euler ODE（deterministic），与 flow 兼容
        for i in range(len(sigmas) - 1):
            sigma = float(sigmas[i])
            sigma_next = float(sigmas[i + 1])
            denoised = denoise_fn(x, sigmas[i])
            d = (x - denoised) / max(sigma, 1e-6)
            x = x + d * (sigma_next - sigma)

    # VAE 解码
    latents = x.to(device=device, dtype=dtype)
    logger.info(f"[Debug] Final latents: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
    try:
        images = vae.model.decode(latents, vae.scale)
        images = images.squeeze(2)  # [B,C,H,W]
        images = (images.clamp(-1, 1) + 1) / 2

        # 转 PIL
        img = images[0].permute(1, 2, 0).cpu().float().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)

        model.train()
        return Image.fromarray(img)
    except Exception as e:
        logger.error(f"[Debug] VAE decode failed: {e}")
        raise e


# ============================================================================
# 训练辅助
# ============================================================================

def sample_t(bs, device):
    """采样时间步 (logit-normal)"""
    t = torch.sigmoid(torch.randn(bs, device=device))
    shift = 3.0
    t = (t * shift) / (1 + (shift - 1) * t)
    return t


def collate_fn(batch):
    """DataLoader collate"""
    pixels = torch.stack([b["pixel_values"] for b in batch])
    captions = [b["caption"] for b in batch]
    result = {"pixel_values": pixels, "captions": captions}
    if "loss_weight" in batch[0]:
        result["loss_weight"] = torch.tensor([b["loss_weight"] for b in batch], dtype=torch.float32)
    return result


def collate_fn_cached(batch):
    """DataLoader collate for cached latents"""
    latents = torch.stack([b["latent"] for b in batch])
    captions = [b["caption"] for b in batch]
    result = {"latents": latents, "captions": captions}
    if "loss_weight" in batch[0]:
        result["loss_weight"] = torch.tensor([b["loss_weight"] for b in batch], dtype=torch.float32)
    return result


# ============================================================================
# 参数解析
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Anima LoRA Trainer v2")
    # 配置文件
    p.add_argument("--config", default="", help="YAML 配置文件路径")
    # 路径
    p.add_argument("--data-dir", default="", help="数据集目录")
    p.add_argument("--transformer", default="", help="transformer safetensors")
    p.add_argument("--vae", default="", help="VAE safetensors")
    p.add_argument("--qwen", default="", help="Qwen 模型目录")
    p.add_argument("--t5-tokenizer", default="", help="T5 tokenizer 目录")
    p.add_argument("--output-dir", default="./output", help="输出目录")
    p.add_argument("--output-name", default="anima_lora", help="输出名称")

    # 训练参数
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-scheduler", default="none", choices=["none", "cosine", "cosine_with_restart"], help="学习率调度器")
    p.add_argument("--lr-scheduler-t0", type=int, default=500, help="cosine_with_restart: 首次 restart 周期 (step)")
    p.add_argument("--lr-scheduler-t-mult", type=float, default=2.0, help="cosine_with_restart: 每次 restart 周期倍数")
    p.add_argument("--lr-scheduler-eta-min", type=float, default=0.0, help="cosine/cosine_with_restart: 最小学习率")
    p.add_argument("--weight-decay", type=float, default=0.01, help="AdamW 权重衰减 (L2 正则, 0=禁用)")
    p.add_argument("--grad-clip-max-norm", type=float, default=1.0, help="梯度裁剪最大范数 (0=禁用)")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--mixed-precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--grad-checkpoint", action="store_true", help="启用梯度检查点减少显存")
    p.add_argument("--xformers", action="store_true", help="启用 xformers memory efficient attention")
    p.add_argument("--max-steps", type=int, default=0, help="最大训练步数 (0=无限制)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")

    # 数据集参数
    p.add_argument("--repeats", type=int, default=1, help="(已弃用，改用文件夹名定义 repeat，如 5_concept)")
    p.add_argument("--reg-data-dir", default="", help="正则数据集目录（防过拟合，Kohya 风格）")
    p.add_argument("--reg-repeats", type=int, default=1, help="正则集每张图重复次数")
    p.add_argument("--reg-caption", default="", help="正则集统一 caption，如 1girl, solo（空则用各图自带）")
    p.add_argument("--shuffle-caption", action="store_true", help="打乱 caption tags（分类 shuffle）")
    p.add_argument("--keep-tokens", type=int, default=0, help="保留前 N 个 tokens 不打乱")
    p.add_argument("--flip-augment", action="store_true", help="随机水平翻转增强")
    p.add_argument("--tag-dropout", type=float, default=0.0, help="Tag dropout 概率 (0-1)")
    p.add_argument("--no-prefer-json", action="store_true", help="禁用 JSON 优先模式")
    p.add_argument("--cache-latents", action="store_true", help="缓存 VAE latent 加速训练")

    # LoRA 参数
    p.add_argument("--lora-type", choices=["lora", "lokr"], default="lokr")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lokr-factor", type=int, default=8)
    p.add_argument("--resume-lora", default="", help="从已有 LoRA 继续训练（safetensors 路径）")

    # 采样参数
    p.add_argument("--sample-every", type=int, default=0, help="每 N 个 epoch 采样一次 (0=禁用)")
    p.add_argument("--sample-steps", type=int, default=0, help="每 N 个 step 采样一次 (0=禁用)")
    p.add_argument("--sample-prompt", default="1girl, masterpiece", help="采样提示词")
    p.add_argument("--sample-cfg-scale", type=float, default=4.0, help="采样 CFG（设为 1 表示不做 CFG，仅用正面条件）")
    p.add_argument("--sample-negative-prompt", default="", help="采样负面提示词（留空使用默认负面）")
    p.add_argument("--sample-width", type=int, default=0, help="采样宽度（0=跟随 resolution）")
    p.add_argument("--sample-height", type=int, default=0, help="采样高度（0=跟随 resolution）")
    p.add_argument("--sample-seed", type=int, default=0, help="采样随机种子（0=不固定）")
    p.add_argument("--sample-infer-steps", type=int, default=25, help="采样推理步数（对齐 ComfyUI 默认 25）")
    p.add_argument("--sample-sampler-name", default="er_sde", help="采样器名称（对齐 ComfyUI: er_sde）")
    p.add_argument("--sample-scheduler", default="simple", help="采样 scheduler（对齐 ComfyUI: simple）")

    # 保存参数
    p.add_argument("--save-every", type=int, default=0, help="每 N 个 epoch 保存 (0=仅结束时)")
    p.add_argument("--save-state-every", type=int, default=0, help="每 N 步保存完整训练状态（可断点续训）")
    p.add_argument("--resume-state", default="", help="从训练状态恢复（.pt 文件路径）")
    p.add_argument("--seed", type=int, default=42)

    # 进度显示
    p.add_argument("--no-progress", action="store_true", help="禁用动态进度显示")
    p.add_argument("--loss-curve-steps", type=int, default=100, help="Loss 曲线显示步数 (0=禁用)")
    p.add_argument("--no-live-curve", action="store_true", help="禁用实时 Loss 曲线刷新")
    p.add_argument("--no-monitor", action="store_true", help="禁用 Web 监控面板")
    p.add_argument("--monitor-host", default="127.0.0.1", help="监控面板绑定地址（默认仅本机；局域网/云端访问用 0.0.0.0）")
    p.add_argument("--monitor-port", type=int, default=8765, help="监控面板端口")
    p.add_argument("--no-browser", action="store_true", help="不自动打开监控面板浏览器")
    p.add_argument("--log-every", type=int, default=10, help="日志输出间隔")

    # 依赖和交互
    p.add_argument("--auto-install", action="store_true", help="自动安装缺失依赖")
    p.add_argument("--interactive", action="store_true", help="交互模式，提示输入缺失参数")

    return p.parse_args()


# ============================================================================
# 交互模式辅助函数
# ============================================================================

def _try_rich():
    try:
        from rich.prompt import Prompt, Confirm
        return Prompt, Confirm
    except Exception:
        return None, None


def _ask_str(label, default=""):
    Prompt, _ = _try_rich()
    if Prompt:
        return Prompt.ask(label, default=default) if default else Prompt.ask(label)
    raw = input(f"{label}{f' [{default}]' if default else ''}: ").strip()
    return raw or default


def _ask_bool(label, default=False):
    _, Confirm = _try_rich()
    if Confirm:
        return Confirm.ask(label, default=default)
    raw = input(f"{label} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true", "t")


def _ask_int(label, default):
    while True:
        raw = _ask_str(label, str(default))
        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer.")


def _ask_float(label, default):
    while True:
        raw = _ask_str(label, str(default))
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def _guess_default_paths():
    base = Path(__file__).resolve().parent
    transformer = base / "anima" / "diffusion_models" / "anima-preview.safetensors"
    vae = base / "anima" / "vae" / "qwen_image_vae.safetensors"
    qwen = base / "anima" / "text_encoders"
    return {
        "transformer": str(transformer) if transformer.exists() else "",
        "vae": str(vae) if vae.exists() else "",
        "qwen": str(qwen) if qwen.exists() else "",
    }


def prompt_for_args(args):
    """交互式提示输入缺失参数"""
    defaults = _guess_default_paths()
    args.data_dir = args.data_dir or _ask_str("数据集目录 (images + .txt)", "")
    args.transformer = args.transformer or _ask_str("Transformer 路径 (.safetensors)", defaults["transformer"])
    args.vae = args.vae or _ask_str("VAE 路径 (.safetensors)", defaults["vae"])
    args.qwen = args.qwen or _ask_str("Qwen 模型目录", defaults["qwen"])
    args.output_dir = _ask_str("输出目录", args.output_dir)
    args.output_name = _ask_str("输出名称", args.output_name)
    args.resolution = _ask_int("分辨率", args.resolution)
    args.batch_size = _ask_int("Batch size", args.batch_size)
    args.grad_accum = _ask_int("梯度累积", args.grad_accum)
    args.lr = _ask_float("学习率", args.lr)
    args.grad_checkpoint = _ask_bool("启用梯度检查点?", args.grad_checkpoint)
    args.epochs = _ask_int("Epochs", args.epochs)
    args.max_steps = _ask_int("最大步数 (0=无限制)", args.max_steps)
    args.lora_rank = _ask_int("LoRA rank", args.lora_rank)
    args.lora_alpha = _ask_float("LoRA alpha", args.lora_alpha)
    args.loss_curve_steps = _ask_int("Loss 曲线步数 (0=禁用)", args.loss_curve_steps)
    args.auto_install = _ask_bool("自动安装缺失依赖?", args.auto_install)
    args.save_every_epoch = _ask_bool("每个 epoch 保存?", args.save_every_epoch)
    args.mixed_precision = _ask_str("混合精度 (bf16/fp32)", args.mixed_precision)
    return args


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

    # 处理 --no-prefer-json 参数
    if getattr(args, "no_prefer_json", False):
        args.prefer_json = False
    elif not hasattr(args, "prefer_json"):
        args.prefer_json = True  # 默认启用

    # 交互模式检查
    required = [args.data_dir, args.transformer, args.vae, args.qwen]
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

    # 启动训练监控面板
    monitor_server = None
    if not getattr(args, "no_monitor", False):
        try:
            from train_monitor import start_monitor_server, update_monitor
            monitor_server = start_monitor_server(
                host=getattr(args, "monitor_host", "127.0.0.1"),
                port=int(getattr(args, "monitor_port", 8765) or 8765),
                output_dir=output_dir,
                open_browser=(not getattr(args, "no_browser", False)),
            )
            update_monitor(config={
                "model": "Anima LoKr" if args.lora_type == "lokr" else "Anima LoRA",
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "resolution": args.resolution,
                "data_dir": str(args.data_dir),
            })
        except Exception as e:
            logger.warning(f"监控面板启动失败: {e}")

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
    args.transformer = resolve_path_best_effort(args.transformer, bases)
    args.vae = resolve_path_best_effort(args.vae, bases)
    args.qwen = resolve_path_best_effort(args.qwen, bases)
    args.t5_tokenizer = resolve_path_best_effort(getattr(args, "t5_tokenizer", ""), bases)
    args.data_dir = resolve_path_best_effort(args.data_dir, bases)
    reg_data_dir = getattr(args, "reg_data_dir", "") or ""
    if reg_data_dir:
        args.reg_data_dir = resolve_path_best_effort(reg_data_dir, bases)

    # 加载模型
    logger.info("加载 Transformer...")
    model = load_anima_model(args.transformer, device, dtype, repo_root)

    # 启用 xformers
    if args.xformers:
        enable_xformers(model)

    logger.info("加载 VAE...")
    vae = load_vae(args.vae, device, dtype, repo_root)

    logger.info("加载文本编码器...")
    qwen_model, qwen_tok, t5_tok = load_text_encoders(
        args.qwen, args.t5_tokenizer, device, dtype
    )

    # 注入 LoRA
    logger.info(f"注入 {args.lora_type.upper()}...")
    injector = LoRAInjector(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        use_lokr=(args.lora_type == "lokr"),
        factor=args.lokr_factor,
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
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    if weight_decay > 0:
        wd_info = f"AdamW weight_decay={weight_decay}"
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

    if args.max_steps and args.max_steps > 0:
        total_steps = args.max_steps
    elif steps_per_epoch is not None:
        total_steps = steps_per_epoch * args.epochs
    else:
        total_steps = None

    logger.info(f"数据集大小: {len(dataset)}, 每 epoch 步数: {steps_per_epoch}, 总步数: {total_steps}")

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
        save_training_state(state_path, injector, optimizer, current_epoch, global_step, loss_history, monitor_state=monitor_data, scheduler=scheduler)
        # 同时保存 LoRA 权重
        lora_path = output_dir / f"{args.output_name}_interrupted_step{global_step}.safetensors"
        injector.save(lora_path)
        emit(f"已保存！下次使用 --resume-state \"{state_path}\" 继续训练")
        sys.exit(0)
    
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    current_epoch = start_epoch
    model.train()
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

            # Flow Matching
            t = sample_t(bs, device)
            t_exp = t.view(-1, 1, 1, 1, 1)
            noise = torch.randn_like(latents)
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
                loss = loss_per_sample.mean()

            # 反向传播
            loss = loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
                optimizer.step()
                if scheduler is not None:
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
                            loss=loss_val, lr=lr, epoch=epoch+1, step=global_step,
                            total_steps=total_steps, speed=speed_ema or 0
                        )
                    except Exception:
                        pass
                dt_step = now - step_start_time
                steps_per_sec = (1.0 / dt_step) if dt_step > 0 else 0.0
                speed_ema = steps_per_sec if speed_ema is None else (0.9 * speed_ema + 0.1 * steps_per_sec)

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
                injector.save(save_path)
                emit(f"Saved LoRA: {save_path}")

            # 采样（轮换提示词）
            if args.sample_every > 0 and current_epoch % args.sample_every == 0:
                prompt = get_next_sample_prompt()
                prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
                emit(f"采样中 (epoch {current_epoch}): {prompt_short}")
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
    logger.info("训练完成!")


if __name__ == "__main__":
    main()
