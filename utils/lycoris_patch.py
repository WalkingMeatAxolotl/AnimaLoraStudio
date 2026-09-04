"""Patch LyCORIS LoKr rank-dropout masks onto the weight device.

Affected releases create the rank-dropout mask on CPU and multiply it into a
CUDA weight.  The compatibility wrapper delegates weight construction to the
installed LyCORIS implementation (preserving the v3 implementation or the v4
kernel dispatcher), then applies the same dropout semantics on the output
weight's device.

The patch is exact-version guarded.  Add a release only after reproducing the
bug against that artifact, and remove releases once their upstream
implementation is fixed.
"""
from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

logger = logging.getLogger(__name__)

# Confirmed by source inspection and CUDA reproduction.  The dev build is used
# only by the migration branch while waiting for the first post-4.0.0 stable.
KNOWN_AFFECTED_VERSIONS: frozenset[str] = frozenset(
    {
        "3.4.0",
        "4.0.0",
        "4.0.1.dev20260902072855",
    }
)

PatchStatus = Literal[
    "applied",
    "skipped_not_installed",
    "skipped_version_unknown",
    "skipped_already_patched",
]

_PATCHED_FLAG = "_anima_lokr_device_patched"
_ORIGINAL_GET_WEIGHT_ATTR = "_anima_lokr_original_get_weight"


def apply_lokr_device_patch() -> PatchStatus:
    """Patch known-affected ``LokrModule.get_weight`` implementations.

    The wrapper intentionally avoids importing v3 private helpers removed by
    v4.  Rebuild work remains owned by the installed LyCORIS implementation;
    only rank dropout is temporarily disabled there and reapplied on-device.
    Repeated calls are idempotent.
    """
    try:
        installed = version("lycoris-lora")
    except PackageNotFoundError:
        return "skipped_not_installed"

    if installed not in KNOWN_AFFECTED_VERSIONS:
        logger.debug(
            "lycoris-lora %s is not in the known-affected set %s; rank_dropout "
            "device patch skipped",
            installed,
            sorted(KNOWN_AFFECTED_VERSIONS),
        )
        return "skipped_version_unknown"

    try:
        from lycoris.modules.lokr import LokrModule
    except Exception as exc:  # pragma: no cover - corrupt/incompatible install
        logger.warning(
            "lycoris-lora %s is installed but LokrModule could not be imported "
            "(%s); the rank_dropout device patch was skipped",
            installed,
            exc,
        )
        return "skipped_not_installed"

    if getattr(LokrModule, _PATCHED_FLAG, False):
        return "skipped_already_patched"

    import torch  # noqa: PLC0415 - avoid a torch import when LyCORIS is absent

    original_get_weight = LokrModule.get_weight

    def _get_weight_fixed(self, shape):  # type: ignore[no-untyped-def]
        rank_dropout = self.rank_dropout
        if not (self.training and rank_dropout):
            return original_get_weight(self, shape)

        # Delegate reconstruction to the installed release.  In v4 this keeps
        # kron_weight and its selected backend intact instead of copying a
        # private implementation into Studio.
        self.rank_dropout = 0.0
        try:
            weight = original_get_weight(self, shape)
        finally:
            self.rank_dropout = rank_dropout

        drop = (
            torch.rand(weight.size(0), device=weight.device) > rank_dropout
        ).to(weight.dtype)
        drop = drop.view(-1, *[1] * len(weight.shape[1:]))
        if self.rank_dropout_scale:
            drop /= drop.mean()
        weight *= drop
        return weight

    setattr(LokrModule, _ORIGINAL_GET_WEIGHT_ATTR, original_get_weight)
    LokrModule.get_weight = _get_weight_fixed
    setattr(LokrModule, _PATCHED_FLAG, True)
    logger.debug(
        "lycoris-lora %s: patched LokrModule.get_weight for the rank_dropout "
        "device bug",
        installed,
    )
    return "applied"
