"""训练 main() 的 phase 拆分（ADR 0003 PR-B）。

每个 phase 是个 `run(ctx: TrainingContext) -> None` 函数，in-place mutate ctx。
main() 编排成：bootstrap → models.run → dataset → text_cache → models.finish →
optimizer → resume → train_loop → finalize。``models.finish`` 只为缓存后延迟加载
DiT 的族补齐模型栈，其余族是 no-op。
"""

from training.phases import bootstrap, dataset, finalize, models, optimizer, resume, text_cache

__all__ = [
    "bootstrap", "models", "dataset", "text_cache", "optimizer", "resume", "finalize",
]
