"""bootstrap_phase：args + yaml + 交互 + seed + device/dtype + 输出目录 + wandb + monitor_state。

抽自 main() L113-185（ADR 0003 PR-B）。
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import torch

from training.bootstrap import apply_yaml_config, ensure_dependencies, load_yaml_config
from training.cli import prompt_for_args
from training.context import TrainingContext
from training.observability import init_wandb_monitor


logger = logging.getLogger(__name__)


def _maybe_apply_pause_snapshot(args, resume_state_path: Path) -> None:
    """读 pause snapshot 覆盖 args（ADR 0006 PR-3 / §5.7）。

    args.resume_state = `…/pause_step_<N>.pt` → snapshot = `…/pause_step_<N>.config.json`。
    snapshot 不存在 → 静默跳过（用户走 ResumeFieldPicker 选周期 save 文件
    起新 task 的旧路径）。

    覆盖规则：
    - snapshot["args"] 内所有字段写到 args namespace，**例外**：
      - `resume_state` 不覆盖（snapshot 记录的是 pause 前的 args，那时 resume_state
        是空；现在我们才用它续训）
      - `config` 不覆盖（snapshot 记录的是用户当时的 yaml 路径，用户可能已删/改名）
    - snapshot["sample_prompts"] → args.sample_prompts（resume_phase 会读这个）
    """
    snapshot_path = resume_state_path.with_suffix(".config.json")
    if not snapshot_path.exists():
        return  # 不是 pause state，沿用现有 args
    try:
        raw = snapshot_path.read_text(encoding="utf-8")
        snapshot = json.loads(raw)
    except Exception as exc:
        logger.warning(
            f"读取 pause snapshot 失败，沿用现有 args: {snapshot_path} ({exc})"
        )
        return
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("args"), dict):
        logger.warning(f"pause snapshot schema 不识别，沿用现有 args: {snapshot_path}")
        return
    logger.info(f"加载 pause snapshot 覆盖训练参数: {snapshot_path}")
    snap_args: dict = snapshot["args"]
    skipped = {"resume_state", "config"}
    for k, v in snap_args.items():
        if k in skipped:
            continue
        setattr(args, k, v)
    sp = snapshot.get("sample_prompts")
    if isinstance(sp, list):
        args.sample_prompts = sp


def _resolve_sample_seed(args) -> None:
    """sample_seed=0 → 训练开始时随机抽一次写回 args，并 log。

    Why：sample_seed=0 走 sample_runner 时不调 torch.manual_seed，整批
    采样跟着 global RNG 漂移，跨 epoch 同 prompt 出图不同 → 看不出是
    模型收敛还是噪声变了。抽一次固定下来，整轮训练同 prompt 同 seed。

    与 pause snapshot 协作：snapshot 写整份 args.dict()，resolved 值会
    被 freeze；resume 经 _maybe_apply_pause_snapshot 灌回，跨 pause 仍
    用同一 seed。用户重新起 task 时若 yaml 还是 0，启动重抽一次新随机。
    """
    if int(getattr(args, "sample_seed", 0) or 0):
        return
    args.sample_seed = random.randint(1, 2**31 - 1)
    logger.info(f"sample_seed=0 → 训练用随机种子: {args.sample_seed}")


def _prepend_trigger_to_sample_prompts(args) -> None:
    """trigger_word 非空 → prepend 到 sample_prompt / sample_prompts 每条。

    与 caption 端行为一致（tag_worker 也把 trigger 写为第一个 tag）：训练
    采样图必带 trigger，能直观验证 LoRA 是否激活。判定"已含 trigger"用 token
    级匹配（按逗号 split 后等值比较，不区分大小写），避免被 substring 误判。
    空 prompt 不注入，防止生成残缺的 ``"trigger, "`` 字符串。
    """
    trigger = (getattr(args, "trigger_word", "") or "").strip()
    if not trigger:
        return
    lower = trigger.lower()

    def _contains(prompt: str) -> bool:
        return any(t.strip().lower() == lower for t in prompt.split(",") if t.strip())

    sp = getattr(args, "sample_prompt", "") or ""
    if sp and not _contains(sp):
        args.sample_prompt = f"{trigger}, {sp}"

    sps = getattr(args, "sample_prompts", None) or []
    if sps:
        args.sample_prompts = [
            (f"{trigger}, {p}" if (p and not _contains(p)) else p) for p in sps
        ]


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
    from training.losses import validate_schema_consistency as _validate_losses
    from training.optimizers import validate_schema_consistency as _validate_optimizers
    from training.schedulers import validate_schema_consistency as _validate_schedulers
    _validate_adapters()
    _validate_optimizers()
    _validate_schedulers()
    _validate_losses()

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

    # ADR 0006 PR-3：pause 文件旁边的 .config.json snapshot 覆盖 args。
    # 触发条件：args.resume_state 指向的 .pt 旁边有同前缀的 .config.json。
    # 仅 pause 触发的 state 会带 snapshot（PR-2 handle_interrupt 写）；周期
    # save 没有 snapshot，ResumeFieldPicker 起新 task 走原路径（用户当前
    # yaml config）。Snapshot freeze 是 ADR §5.7 的核心 — resume 时 task 的
    # 训练参数严格用暂停那一刻的值，跟用户后续改 version / preset / yaml
    # 完全解耦。
    if getattr(args, "resume_state", None):
        _maybe_apply_pause_snapshot(args, Path(args.resume_state))
        ctx.args = args

    # 交互模式检查
    required = [args.data_dir, args.transformer_path, args.vae_path, args.text_encoder_path]
    if args.interactive or any(not x for x in required):
        ctx.args = prompt_for_args(args)
        args = ctx.args

    # 触发词注入：caption 端 tag_worker 把 trigger 写为第一个 tag，这里同步
    # 注入 sample_prompt(s)，让采样图天然带 trigger。pause snapshot 已 freeze
    # trigger_word（写在 args 里），resume 也照此 normalize 一次幂等。
    _prepend_trigger_to_sample_prompts(args)
    ctx.args = args

    # 依赖检测
    ensure_dependencies(auto_install=args.auto_install)

    # 延迟导入：保留原 main() 顺序 —— ensure_dependencies 之后才能 import numpy/PIL
    import numpy as np

    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    _resolve_sample_seed(args)

    ctx.device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx.dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32

    if ctx.device == "cuda":
        if getattr(args, "tf32_matmul", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            logger.info("TF32 matmul 已启用")
        if getattr(args, "cudnn_benchmark", True) and not getattr(args, "torch_compile", False):
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark 已启用")

    # 创建输出目录
    ctx.output_dir = Path(args.output_dir)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    # 采样图落到 task 档案根的 samples/。supervisor 按 task 注入
    # `--monitor-state-file <studio_data>/tasks/<id>/monitor/state.json`，
    # sample_dir 取其上跳一层的 `samples/` —— `tasks/<id>/samples/`，跟 monitor/
    # 同级，整组（snapshot/ monitor/ samples/ run.log）就是 task 完整档案。
    # 没传 --monitor-state-file（纯 CLI 训练 / 兼容老版本注入路径）退回
    # output_dir/samples，samples.py 仍可在 monitor_dir 周围多候选搜回。
    _msf = getattr(args, "monitor_state_file", None)
    ctx.sample_dir = (Path(_msf).parent.parent / "samples") if _msf else (ctx.output_dir / "samples")
    ctx.sample_dir.mkdir(parents=True, exist_ok=True)
    # ADR 0006 Addendum 2：auto_epoch_state.pt 同样归 task 档案 —— tasks/<id>/state/，
    # 跟 samples/ 同根。没传 --monitor-state-file（纯 CLI）→ None，
    # ctx.auto_state_dir() fallback 到 output_dir/state/task_<id>/（行为不变）。
    ctx.task_archive_state_dir = (Path(_msf).parent.parent / "state") if _msf else None
    # supervisor 启动训练时通过 env LORA_TASK_ID 注入 queue task id（ADR 0006）。
    # 用于 ctx.state_dir() 计算 per-task state 子目录；env 不存在时 fallback unknown。
    _env_tid = os.environ.get("LORA_TASK_ID")
    if _env_tid:
        try:
            ctx.lora_task_id = int(_env_tid)
        except ValueError:
            logger.warning(f"LORA_TASK_ID={_env_tid!r} 不是 int，按 unknown 处理")
    ctx.wandb_monitor = init_wandb_monitor(args, ctx.output_dir, ctx.config_path)

    # Loss 函数（mse / huber；通过 losses/ plugin registry 派发）
    # 不依赖 total_steps，跟 timestep_sampler/scheduler 不同；放 bootstrap 而非
    # optimizer phase 避免架构错位。
    from training.losses import build_loss
    ctx.loss_fn = build_loss(args)

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
