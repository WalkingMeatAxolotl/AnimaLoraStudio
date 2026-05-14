"""训练 main() 的 phase 拆分（ADR 0003 PR-B）。

每个 phase 是个 `run(ctx: TrainingContext) -> None` 函数，in-place mutate ctx。
main() 编排成：bootstrap → models → dataset → optimizer → resume → train_loop → finalize。
"""

from training.phases import bootstrap, dataset, finalize, models, optimizer, resume

__all__ = ["bootstrap", "models", "dataset", "optimizer", "resume", "finalize"]
