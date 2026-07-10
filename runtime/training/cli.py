"""命令行 / 交互模式入口：parse_args + prompt_for_args。

抽自原 runtime/anima_train.py L1963-2103（ADR 0003 PR-A）。被 test_anima_train_migration.py
直接调 parse_args / apply_yaml_config。

公开：
- parse_args — 走 studio.argparse_bridge.build_parser，从 TrainingConfig 自动生成
- prompt_for_args — 交互模式补缺失字段
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def parse_args():
    """从 studio.schema.TrainingConfig 自动生成 parser；额外补 schema 之外的
    CLI-only 开关（auto-install / interactive / no-live-curve / 已弃用的
    --repeats 和 --reg-repeats）。
    """
    from studio.infrastructure.argparse_bridge import build_parser
    from studio.schema import TrainingConfig

    p = build_parser(TrainingConfig, prog="anima_train", description="Anima LoRA Trainer v2")
    # schema 之外的 CLI-only 开关
    p.add_argument("--auto-install", action="store_true", help="自动安装缺失依赖")
    p.add_argument("--interactive", action="store_true", help="交互模式，提示输入缺失参数")
    p.add_argument("--no-live-curve", action="store_true", help="禁用实时 Loss 曲线刷新")
    # PP6.1 — 监控状态文件路径；不传则默认写到 output_dir/monitor_state.json
    # 注：旧 monitor server 的 --no-monitor / --monitor-host / --monitor-port /
    # --no-browser 已随 TrainingConfig 字段一并删除（yaml 老键静默丢弃）。
    p.add_argument(
        "--monitor-state-file",
        type=str,
        default=None,
        help="训练监控 state.json 输出路径（默认 output_dir/monitor_state.json）",
    )
    # 已弃用：每图重复改用文件夹名前缀（如 5_concept）
    p.add_argument("--repeats", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--reg-repeats", type=int, default=1, help=argparse.SUPPRESS)
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
    """猜默认模型路径（仅在用户没在 yaml/CLI 显式指定时用）。

    根目录：优先 `secrets.models.root`（Studio 设置页配置），否则 `REPO_ROOT/models/`
    （与 schema.py 默认 + WD14 已用的 `models/wd14/` 对齐）。

    Transformer：用户可能装多个 Anima 版本（preview / preview2 / preview3-base / 1.0），
    按 ANIMA_VARIANTS 顺序找第一个存在的（latest 优先）。
    """
    # 注：原 runtime/anima_train.py 用 Path(__file__).resolve().parent 拿 runtime/；
    # 这里 cli.py 在 runtime/training/ 下，多一层往上才能保持等价语义。
    repo_root = Path(__file__).resolve().parent.parent
    # secrets 不一定可 import（直接 CLI 跑训练时 studio package 可用；其他场景兜底）
    base: Optional[Path] = None
    transformer_path: str = ""
    try:
        from studio.services.models import find_anima_main, models_root
        base = models_root()
        existing = find_anima_main(base)
        if existing:
            transformer_path = str(existing)
    except Exception:
        base = repo_root / "models"
    if not base:
        base = repo_root / "models"
    if not transformer_path:
        # services 不可用 / 都没下载 → 给最新版默认名作为提示，方便用户填路径
        candidate = base / "diffusion_models" / "anima-base-v1.0.safetensors"
        transformer_path = str(candidate) if candidate.exists() else ""

    vae = base / "vae" / "qwen_image_vae.safetensors"
    qwen = base / "text_encoders"
    return {
        "transformer": transformer_path,
        "vae": str(vae) if vae.exists() else "",
        "qwen": str(qwen) if qwen.exists() else "",
    }


def prompt_for_args(args):
    """交互式提示输入缺失参数"""
    defaults = _guess_default_paths()
    args.data_dir = args.data_dir or _ask_str("数据集目录 (images + .txt)", "")
    args.transformer_path = args.transformer_path or _ask_str("Transformer 路径 (.safetensors)", defaults["transformer"])
    args.vae_path = args.vae_path or _ask_str("VAE 路径 (.safetensors)", defaults["vae"])
    args.text_encoder_path = args.text_encoder_path or _ask_str("Qwen 模型目录", defaults["qwen"])
    args.output_dir = _ask_str("输出目录", args.output_dir)
    args.output_name = _ask_str("输出名称", args.output_name)
    # resolution 现在是 list[int]。单值（含默认 [1024]）照常交互问一档；已设多分辨率
    # （GUI / config 驱动）则不打扰。
    _res = args.resolution
    if not isinstance(_res, (list, tuple)):
        args.resolution = [_ask_int("分辨率", int(_res))]
    elif len(_res) <= 1:
        args.resolution = [_ask_int("分辨率", int(_res[0]) if _res else 1024)]
    args.batch_size = _ask_int("Batch size", args.batch_size)
    args.grad_accum = _ask_int("梯度累积", args.grad_accum)
    args.learning_rate = _ask_float("学习率", args.learning_rate)
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
