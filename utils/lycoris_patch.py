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
    import torch.nn.functional as TF  # for bypass_diff patch

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

    # ── 显存优化 patch：full matrix 模式走 bypass，避免 Kron 展开完整矩阵 ──
    # 原 forward 在 full matrix 模式（use_w1=True, use_w2=True）下每次
    # 都通过 Kronecker 积展开完整权重，MLP 8192×2048 层展开 ~32 MiB，在
    # 8GB 显卡上叠加 280 层 + 优化器状态直接 OOM。
    # bypass_forward 使用因式分解方式直接计算 ΔW·x，等效且无需展开。
    _orig_forward = LokrModule.forward

    def _forward_memopt(self, x, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Full matrix 模式 + 无 dropout/DoRA：走 bypass 省显存
        # （dropout/DoRA 在 bypass 路径不生效，所以仅无这些特性时启用）
        if (
            getattr(self, "use_w1", False)
            and getattr(self, "use_w2", False)
            and not getattr(self, "bypass_mode", None)
            and not getattr(self, "wd", False)
            and not (getattr(self, "module_dropout", 0) and self.training)
            and not (getattr(self, "rank_dropout", 0) and self.training)
        ):
            return self.bypass_forward(x, self.multiplier)
        return _orig_forward(self, x, *args, **kwargs)

    # ── bypass_forward_diff 内存优化：显式释放中间激活值 ──
    # 原始 bypass_forward_diff 在 Linear 路径下同时持有 hb（F.linear 输出）
    # 和 hc（第二次 F.linear 输出），每个都是 [batch, 1024, 8] × bf16 ≈ 256 MiB
    # （MLP layer1: 16384×8192）。两个中间量共存 + h*scale 临时量直接 OOM。
    # 优化：运算间隙显式 del + empty_cache 确保中间量不在峰值时共存。
    _orig_bypass_diff = LokrModule.bypass_forward_diff

    def _bypass_diff_memopt(self, h, scale=1):
        is_conv = self.module_type.startswith("conv")
        if self.use_w2:
            ba = self.lokr_w2
        else:
            a = self.lokr_w2_b
            b = self.lokr_w2_a
            if self.tucker:
                t = self.lokr_t2
                a = a.view(*a.shape, *[1] * (len(t.shape) - 2))
                b = b.view(*b.shape, *[1] * (len(t.shape) - 2))
            elif is_conv:
                a = a.view(*a.shape, *self.shape[2:])
                b = b.view(*b.shape, *[1] * (len(self.shape) - 2))

        if self.use_w1:
            c = self.lokr_w1
        else:
            c = self.lokr_w1_a @ self.lokr_w1_b
        uq = c.size(1)

        if is_conv:
            bsz, _, *rest = h.shape
            h_in_group = h.reshape(bsz * uq, -1, *rest)
        else:
            h_in_group = h.reshape(*h.shape[:-1], uq, -1)

        if self.use_w2:
            hb = self.op(h_in_group, ba, **self.kw_dict)
        else:
            if is_conv:
                if self.tucker:
                    ha = self.op(h_in_group, a)
                    ht = self.op(ha, t, **self.kw_dict)
                    hb = self.op(ht, b)
                else:
                    ha = self.op(h_in_group, a, **self.kw_dict)
                    hb = self.op(ha, b)
            else:
                ha = self.op(h_in_group, a, **self.kw_dict)
                hb = self.op(ha, b)

        # 释放不再需要的中间量，为下一次 F.linear 腾空间
        del h_in_group

        if is_conv:
            hb = hb.view(bsz, -1, *hb.shape[1:])
            h_cross_group = hb.transpose(1, -1)
        else:
            h_cross_group = hb.transpose(-1, -2)

        hc = TF.linear(h_cross_group, c)
        # 释放 hb/h_cross_group 的存储（~256 MiB），仅保留 hc
        del hb, h_cross_group

        if is_conv:
            hc = hc.transpose(1, -1)
            h = hc.reshape(bsz, -1, *hc.shape[3:])
        else:
            hc = hc.transpose(-1, -2)
            h = hc.reshape(*hc.shape[:-2], -1)

        # 不在此处乘 scale*scalar，交给 bypass_forward 用 torch.add(..., alpha=) 零拷贝
        return h

    LokrModule.bypass_forward_diff = _bypass_diff_memopt

    # ── bypass_forward 零拷贝缩放：torch.add(a, b, alpha=α) 不创建 α*b 临时量 ──
    _orig_bypass_fwd = LokrModule.bypass_forward

    def _bypass_fwd_memopt(self, x, scale=1):
        org = self.org_forward(x)
        diff = self.bypass_forward_diff(x, scale=1.0)
        alpha = float(scale * self.scalar)
        diff = self.drop(diff)  # dropout 仅作用于 LoRA 贡献
        if alpha == 1.0:
            return org + diff
        return torch.add(org, diff, alpha=alpha)

    LokrModule.bypass_forward = _bypass_fwd_memopt

    LokrModule.forward = _forward_memopt
    setattr(LokrModule, "_anima_lokr_memopt_patched", True)

    # ── 原始 device patch ──
    LokrModule.get_weight = _get_weight_fixed
    setattr(LokrModule, _PATCHED_FLAG, True)
    logger.info(
        "lycoris-lora %s: 已 patch LokrModule.get_weight（rank_dropout device 修复 + full matrix bypass 省显存）",
        installed,
    )
    return "applied"
