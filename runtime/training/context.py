"""TrainingContext：所有 phase 共享的状态包（ADR 0003 PR-B）。

把原 runtime/anima_train.py 793 行 main() 里的所有 local 变量收到一个 dataclass，
让 phase 函数能 take ctx → mutate → return None 这种风格走流水线。

设计原则：
- 字段类型清晰；late-populated 的用 `Optional[X] = None` 显示
- 进度展示、信号处理等带闭包的逻辑收到本类的方法上（emit / handle_interrupt /
  get_next_sample_prompt），避免 main() 里的 nonlocal 闭包
- 不持有 args 之外的"输入"——任何 yaml / cli 行为都先 merge 进 args，再开始 phase

每个 phase 函数签名：`def run(ctx: TrainingContext) -> None`（in-place mutate）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass
class TrainingContext:
    # ─── bootstrap_phase 填充 ───
    args: Any  # argparse.Namespace
    config_path: Optional[Path] = None
    config_dir: Optional[Path] = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    output_dir: Optional[Path] = None
    sample_dir: Optional[Path] = None
    wandb_monitor: Any = None         # observability.WandBMonitor
    monitor_server: Optional[bool] = None  # 旧名兼容：True=monitor_state.json 写入活跃

    # ─── models_phase 填充 ───
    repo_root: Optional[Path] = None
    model: Any = None
    vae: Any = None
    qwen_model: Any = None
    qwen_tok: Any = None
    t5_tok: Any = None
    injector: Any = None

    # ─── dataset_phase 填充 ───
    bucket_mgr: Any = None
    base_dataset: Any = None
    dataset: Any = None
    reg_dataset: Any = None
    use_cached: bool = False
    dataloader: Any = None

    # ─── optimizer_phase 填充 ───
    weight_decay: float = 0.0
    optimizer: Any = None
    optimizer_type: str = "adamw"
    grad_clip: float = 0.0
    trainable_params: list = field(default_factory=list)
    steps_per_epoch: Optional[int] = None
    total_steps: Optional[int] = None
    scheduler: Any = None
    timestep_sampler: Any = None    # training.timestep_samplers.TimestepSamplerProtocol
    loss_fn: Any = None             # training.losses.LossProtocol

    # ─── resume_phase 填充 ───
    global_step: int = 0
    start_epoch: int = 0
    current_epoch: int = 0
    loss_history: list = field(default_factory=list)
    speed_ema: Optional[float] = None
    progress: Any = None
    task_id: Any = None
    use_rich: bool = False
    use_plain: bool = False
    live: Any = None
    sample_prompts: list = field(default_factory=list)
    sample_prompt_idx: int = 0
    interrupted: bool = False

    # ─── 共用方法 ───

    def emit(self, msg: str) -> None:
        """打印一条 user-facing 消息，按当前进度显示模式分流。

        移植自原 main() 内 emit 闭包；行为完全一致。
        """
        if self.use_plain:
            print()
        if self.live:
            self.live.console.print(msg)
        elif self.use_rich:
            self.progress.console.print(msg)
        else:
            print(msg)

    def get_next_sample_prompt(self) -> str:
        """取下一个采样提示词（轮换；sample_prompts 为空则返回默认）。"""
        if not self.sample_prompts:
            return "1girl, masterpiece"
        prompt = self.sample_prompts[self.sample_prompt_idx % len(self.sample_prompts)]
        self.sample_prompt_idx += 1
        return prompt

    def handle_interrupt(self, sig, frame) -> None:
        """Ctrl+C 信号处理：保存 state + LoRA + finish wandb，然后退出。

        重复触发（已 interrupted 状态再来一次 Ctrl+C）= 强退。
        """
        # 延迟 import 避免循环依赖（state.py / observability.py 间接 import 本模块）
        from training.state import save_training_state
        from train_monitor import get_state
        from utils.optimizer_utils import optimizer_eval_mode

        if self.interrupted:
            self.emit("强制退出...")
            sys.exit(1)
        self.interrupted = True
        self.emit("\n检测到 Ctrl+C，正在保存训练状态...")
        state_path = self.output_dir / f"training_state_step{self.global_step}.pt"
        monitor_data = None
        if self.monitor_server:
            try:
                monitor_data = get_state()
            except Exception:
                pass
        # Schedule-Free 系优化器（PPSF）保存前切到 averaged weights — 否则存的是
        # 训练用的 y 而不是真正应该被使用的 x。非 SF 优化器此 ctx 静默 no-op。
        with optimizer_eval_mode(self.optimizer):
            save_training_state(
                state_path, self.injector, self.optimizer,
                self.current_epoch, self.global_step, self.loss_history,
                monitor_state=monitor_data, scheduler=self.scheduler,
            )
            lora_path = self.output_dir / f"{self.args.output_name}_interrupted_step{self.global_step}.safetensors"
            self.injector.save(lora_path)
        self.wandb_monitor.finish()
        self.emit(f"已保存！下次使用 --resume-state \"{state_path}\" 继续训练")
        sys.exit(0)
