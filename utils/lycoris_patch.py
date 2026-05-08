"""lycoris-lora 3.4.0 的 LokrModule.get_weight rank_dropout device bug patch。

上游 bug：`torch.rand(weight.size(0))` 没传 `device=`，生成 CPU mask，
与 CUDA weight 相乘时报 device mismatch。仅在 `rank_dropout > 0` 且
模块处于 training 模式时触发。

为什么不只靠 lycoris_adapter.py 的 model.train() hijack：
- hijack 只保证 sample/eval 时 network 进 eval 模式（不触发 rank_dropout 分支）
- 但用户若配置 `rank_dropout > 0`，正常 training step 仍走 rank_dropout 分支 ——
  hijack 不覆盖这条路径，仍会撞 bug
- 因此从根上把 LokrModule.get_weight 替换成带 `device=` 的版本

版本守卫：
- 只对 KNOWN_AFFECTED_VERSIONS 内的版本 patch
- 其他版本（包括上游已修的版本）log warn 并跳过；避免覆盖上游已 fix 的实现
- 上游 fix 后请把对应 KNOWN_AFFECTED_VERSIONS 项删掉

上游 issue：https://github.com/KohakuBlueleaf/LyCORIS/issues —— 待提
"""
from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

logger = logging.getLogger(__name__)

# 已知确认受 rank_dropout device bug 影响的 lycoris-lora 版本。
# 经实测：3.4.0 的 `lycoris/modules/lokr.py:get_weight` 走
# `torch.rand(weight.size(0))`（CPU mask），与 CUDA weight 相乘失败。
KNOWN_AFFECTED_VERSIONS: frozenset[str] = frozenset({"3.4.0"})

PatchStatus = Literal[
    "applied",  # 命中受影响版本，已 patch
    "skipped_not_installed",  # 没装 lycoris
    "skipped_version_unknown",  # 装了但版本不在已知受影响集合（warn）
    "skipped_already_patched",  # 同进程内已 patch，幂等返回
]

_PATCHED_FLAG = "_anima_lokr_device_patched"


def apply_lokr_device_patch() -> PatchStatus:
    """检查 lycoris-lora 版本并按需 patch LokrModule.get_weight。

    幂等：同进程内多次调用只 patch 一次。
    """
    try:
        installed = version("lycoris-lora")
    except PackageNotFoundError:
        return "skipped_not_installed"

    try:
        from lycoris.modules.lokr import LokrModule, make_kron, rebuild_tucker
    except Exception as exc:  # pragma: no cover - 装了 lycoris-lora 但 import 异常的边界
        logger.warning(
            "lycoris-lora %s 已安装但 lycoris.modules.lokr 导入失败: %s；跳过 device patch",
            installed,
            exc,
        )
        return "skipped_not_installed"

    if getattr(LokrModule, _PATCHED_FLAG, False):
        return "skipped_already_patched"

    if installed not in KNOWN_AFFECTED_VERSIONS:
        logger.warning(
            "lycoris-lora %s 不在已知受 rank_dropout device bug 影响的版本集合 %s；"
            "跳过 patch（假定上游已修。若你训练时报 device mismatch，请在 issue 上报版本）",
            installed,
            sorted(KNOWN_AFFECTED_VERSIONS),
        )
        return "skipped_version_unknown"

    import torch  # noqa: PLC0415  延迟到此处避免顶层 import 副作用

    def _get_weight_fixed(self, shape):  # type: ignore[no-untyped-def]
        weight = make_kron(
            self.lokr_w1 if self.use_w1 else self.lokr_w1_a @ self.lokr_w1_b,
            (
                self.lokr_w2
                if self.use_w2
                else (
                    rebuild_tucker(self.lokr_t2, self.lokr_w2_a, self.lokr_w2_b)
                    if self.tucker
                    else self.lokr_w2_a @ self.lokr_w2_b
                )
            ),
            self.scale,
        )
        dtype = weight.dtype
        if shape is not None:
            weight = weight.view(shape)
        if self.training and self.rank_dropout:
            drop = (
                torch.rand(weight.size(0), device=weight.device) > self.rank_dropout
            ).to(dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            if self.rank_dropout_scale:
                drop /= drop.mean()
            weight *= drop
        return weight

    LokrModule.get_weight = _get_weight_fixed
    setattr(LokrModule, _PATCHED_FLAG, True)
    logger.info(
        "lycoris-lora %s: 已 patch LokrModule.get_weight（rank_dropout device 修复）",
        installed,
    )
    return "applied"
