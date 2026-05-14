"""bootstrap_phase：args + yaml + 交互 + seed + device/dtype + 输出目录 + wandb + monitor_state。

抽自 main() L113-185（ADR 0003 PR-B）。
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch

from training.bootstrap import apply_yaml_config, ensure_dependencies, load_yaml_config
from training.cli import prompt_for_args
from training.context import TrainingContext
from training.observability import init_wandb_monitor


logger = logging.getLogger(__name__)


def run(ctx: TrainingContext) -> None:
    """完成训练前一切非模型/数据的准备：

    - 加载 yaml config（如有）+ 交互模式补缺字段
    - ensure_dependencies
    - 设种子 / 选 device / dtype
    - 建 output_dir + sample_dir
    - 初始化 wandb_monitor + monitor_state.json 写入器
    """
    args = ctx.args

    # PR-C：启动期校验所有 plugin 子包 schema 一致性，避免运行半天才发现配错
    from training.adapters import validate_schema_consistency as _validate_adapters
    from training.optimizers import validate_schema_consistency as _validate_optimizers
    from training.schedulers import validate_schema_consistency as _validate_schedulers
    _validate_adapters()
    _validate_optimizers()
    _validate_schedulers()

    # 加载 YAML 配置文件
    if args.config:
        logger.info(f"加载配置文件: {args.config}")
        ctx.config_path = Path(args.config).resolve()
        ctx.config_dir = ctx.config_path.parent
        config = load_yaml_config(args.config)
        ctx.args = apply_yaml_config(args, config)
        args = ctx.args

    # bridge 已为 prefer_json bool 自动产生 --prefer-json / --no-prefer-json，
    # 此处无需再做兼容处理。

    # 交互模式检查
    required = [args.data_dir, args.transformer_path, args.vae_path, args.text_encoder_path]
    if args.interactive or any(not x for x in required):
        ctx.args = prompt_for_args(args)
        args = ctx.args

    # 依赖检测
    ensure_dependencies(auto_install=args.auto_install)

    # 延迟导入：保留原 main() 顺序 —— ensure_dependencies 之后才能 import numpy/PIL
    import numpy as np

    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ctx.device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx.dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32

    # 创建输出目录
    ctx.output_dir = Path(args.output_dir)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    ctx.sample_dir = ctx.output_dir / "samples"
    ctx.sample_dir.mkdir(exist_ok=True)
    ctx.wandb_monitor = init_wandb_monitor(args, ctx.output_dir, ctx.config_path)

    # 训练监控状态写入（PP6.1）：永远开启，文件路径优先来自 --monitor-state-file，
    # 否则落到 output_dir/monitor_state.json。Studio 前端通过 /api/state?task_id=
    # 读这个文件，不再启动训练侧 HTTP server（Studio 自己是 monitor）。
    ctx.monitor_server = True  # 兼容下方分支判断；实际代表「写状态文件」
    try:
        from train_monitor import set_state_file, update_monitor
        state_path = (
            Path(args.monitor_state_file)
            if getattr(args, "monitor_state_file", None)
            else ctx.output_dir / "monitor_state.json"
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
        ctx.monitor_server = None
