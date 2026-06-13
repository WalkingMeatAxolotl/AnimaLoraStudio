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
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from training.losses.protocol import LossProtocol


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
    # supervisor 启动训练时通过 env LORA_TASK_ID 注入 queue task id；CLI 直接跑
    # 时 env 不存在 → None → state_dir() fallback 到 task_unknown 子目录。
    # 注意：跟 progress bar 的 task_id 字段（line ~80）是两回事，故意起不同名字。
    lora_task_id: Optional[int] = None
    # ADR 0006 Addendum 2：auto_epoch_state.pt 落 task 档案（studio_data/tasks/
    # <id>/state/）。bootstrap 从 --monitor-state-file 推出档案根后填充（跟
    # sample_dir 同一约定）；纯 CLI 没传 → None → auto_state_dir() fallback
    # 到 state_dir()。
    task_archive_state_dir: Optional[Path] = None

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
    loss_fn: Optional["LossProtocol"] = None

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

    # ─── loop.py epoch backup（ADR 0006 Addendum 1 方案 Δ）───
    # 每 epoch 末尾覆盖式写 auto_epoch_state.pt 后填充这两个字段。
    # handle_interrupt 读它们 emit pause_state event；None 表示首 epoch 还没结束
    # → supervisor 标 canceled 而非 paused（无可恢复进度）。
    last_auto_epoch_state_path: Optional[Path] = None
    last_auto_epoch_config_path: Optional[Path] = None

    # ─── 共用方法 ───

    def state_dir(self) -> Path:
        """用户周期 save（save_state_every*）写 state 的目录，per-task 隔离。

        ADR 0006 §5.3：同一 version 下多 task 跑 state 文件互相覆盖是 latent
        bug，加 task_id 子目录隔离。env LORA_TASK_ID 没设（CLI 直接跑）时
        fallback 到 task_unknown/。
        """
        assert self.output_dir is not None, "state_dir() called before bootstrap_phase"
        tid = self.lora_task_id if self.lora_task_id is not None else "unknown"
        d = self.output_dir / "state" / f"task_{tid}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def auto_state_dir(self) -> Path:
        """auto_epoch_state.pt（系统级恢复点）的落盘目录（ADR 0006 Addendum 2）。

        跟用户周期 save 分家：auto backup 是 task 档案的一部分（同 run.log /
        monitor/ / samples/），落 `studio_data/tasks/<id>/state/` —— 生命周期
        跟 task 行绑定（删 task 一并清），且服务端可从 task id 直接推算路径。
        用户周期 save 是用户产物，留在 state_dir()（version output 树，
        ResumeFieldPicker 按 version 扫得到）。

        纯 CLI（没传 --monitor-state-file）fallback 到 state_dir()，行为不变。
        """
        if self.task_archive_state_dir is not None:
            self.task_archive_state_dir.mkdir(parents=True, exist_ok=True)
            return self.task_archive_state_dir
        return self.state_dir()

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
        """Pause / Ctrl+C 信号处理（ADR 0006 Addendum 1 方案 Δ）：Pause = Cancel + 立即释放 GPU。

        信号来源：
          - CLI Ctrl+C：POSIX SIGINT / Windows SIGBREAK（由 resume phase 注册）
          - Supervisor pause：Windows CTRL_BREAK_EVENT / POSIX SIGINT

        新流程（不再 mid-epoch save）：
          1. wandb finish（让 supervisor 读到事件时一切 IO 已完成）
          2. emit __EVENT__:pause_state，state_path 指向**最近一次 epoch 末** auto_epoch_state.pt
             （由 loop.py 每 epoch 末尾覆盖式写盘，ctx.last_auto_epoch_state_path 字段维护）
          3. 首 epoch 内（last_auto_epoch_state_path is None）→ emit state_path=None，
             supervisor 据此走 cancel 分支（ADR 0006 Addendum 1 决策第 3 条）
          4. sys.exit(0)

        放弃 mid-epoch save 的理由（详见 ADR Addendum 1 三方 audit）：
          - grad_accum 周期未守 → partial backward grad 悬挂
          - dataloader 进度不存 → resume 5% double-train（Prodigy d 估计偏）
          - current_epoch 语义二义性（mid-epoch 路径保 epoch / epoch-end 路径保 epoch+1）
          - InfoNoise / cosine restart T_cur 漂移
          - 真正符合"暂停 = 立即释放 GPU"产品语义

        重复触发（已 interrupted 状态再来一次）= 强退。
        """
        # 延迟 import 避免循环依赖
        from training.snapshot import emit_event

        if self.interrupted:
            self.emit("强制退出...")
            sys.exit(1)
        self.interrupted = True
        self.emit("\n检测到暂停信号，正在退出（保留最近一次 epoch 备份用于 resume）...")
        try:
            self.wandb_monitor.finish()
        except Exception:
            pass
        # emit 在 wandb finish 后 — 让 supervisor 读到事件时一切 IO 已完成。
        emit_event("pause_state", {
            "state_path": str(self.last_auto_epoch_state_path) if self.last_auto_epoch_state_path else None,
            "config_path": str(self.last_auto_epoch_config_path) if self.last_auto_epoch_config_path else None,
            "step": self.global_step,
        })
        if self.last_auto_epoch_state_path:
            self.emit(f"已暂停！恢复点: {self.last_auto_epoch_state_path}")
        else:
            self.emit("首个 epoch 未完成，无 auto 备份可恢复 → 任务将标 canceled。")
        sys.exit(0)
