"""Opt-in, synchronous training boundary notifications for embedded callers.

Observers must not mutate context, tensors, RNG or training control flow. Exceptions
propagate: an interrupted observation is not a completed run. The default None
installs nothing; timing, synchronization and persistence belong to the caller.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from training.context import TrainingContext

PhaseName = Literal[
    "bootstrap", "models", "dataset", "text_cache", "models_finish",
    "optimizer", "resume", "loop", "finalize",
]
UpdateReason = Literal["updated", "no_gradients", "nonfinite_gradients"]


class TrainingObserver(Protocol):
    def phase_started(self, name: PhaseName, ctx: TrainingContext) -> None: ...
    def phase_finished(self, name: PhaseName, ctx: TrainingContext) -> None: ...
    def loop_started(self, ctx: TrainingContext) -> None: ...
    def loop_finished(self, ctx: TrainingContext) -> None: ...
    def forward_started(self, ctx: TrainingContext, *, batch_size: int) -> None: ...
    def backward_finished(self, ctx: TrainingContext, *, loss_is_finite: bool) -> None: ...
    def optimizer_step_finished(self, ctx: TrainingContext, *, reason: UpdateReason) -> None: ...
