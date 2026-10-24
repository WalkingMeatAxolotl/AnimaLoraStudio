"""MOSS 的 NaViT / Patch-n-Pack 训练步：逐图加噪 → 打包前向 → 逐图 loss。

与 Anima 的 ``families/anima/navit.py`` 同语义：G 张尺寸各异的图拼进一条序列，
每图带自己的 timestep 与文本，loss 先在图内对全部元素取均值、再在图间平均（每图
等权，不按 token 数加权）。打包前向见 ``MossImageTransformer.forward_packed``。
"""

from __future__ import annotations

from typing import Sequence

import torch

from training.noise import make_noise


def navit_packed_forward_and_loss(
    model,
    latents_list: Sequence[torch.Tensor],
    t_per_image: torch.Tensor,
    contexts: Sequence[torch.Tensor],
    loss_fn,
    *,
    noise_offset: float = 0.0,
    pyramid_iters: int = 0,
    pyramid_discount: float = 0.35,
    use_checkpoint: bool = False,
    per_image_weights: torch.Tensor | None = None,
):
    """一个 MOSS NaViT 训练步。

    Args:
        model: ``MossImageTransformer``。
        latents_list: G 个 clean latent，每个 ``[1, 32, 1, h_i, w_i]`` 或 ``[32, 1, h_i, w_i]``
            （FLUX2_F8C32_BN 排布；后者是 ``collate_fn_navit_pack`` 的产物）。
        t_per_image: ``[G]`` 逐图 flow-matching timestep。
        contexts: G 个 ``[L_i, 4096]`` 有效文本 token（已去 padding）。
        loss_fn: ``LossProtocol``（``compute(pred, target, t)`` 返回逐元素 loss）。
        noise_offset / pyramid_iters / pyramid_discount: 透传 ``make_noise``。
        use_checkpoint: 逐 block 梯度检查点。
        per_image_weights: ``[G]`` 逐图权重（正则集 loss_weight × loss_weighting），None = 等权。

    Returns:
        (loss, pred, info) —— ``pred`` 为逐图预测列表；``info`` 带 ``visual_seqlens`` 与
        detached ``per_image_loss``，与 Anima 版同形。
    """
    count = len(latents_list)
    if count == 0:
        raise ValueError("navit pack is empty")
    if len(contexts) != count:
        raise ValueError(f"contexts has {len(contexts)} entries, expected G={count}")
    t_per_image = t_per_image.reshape(-1)
    if t_per_image.shape[0] != count:
        raise ValueError(f"t_per_image has {t_per_image.shape[0]} entries, expected G={count}")

    noisy_list, target_list = [], []
    for i, latent in enumerate(latents_list):
        if latent.dim() == 4:  # collate_fn_navit_pack 给的是 [C, T, h, w]
            latent = latent.unsqueeze(0)
        noise = make_noise(
            latent,
            noise_offset=noise_offset,
            pyramid_iters=pyramid_iters,
            pyramid_discount=pyramid_discount,
        )
        t = t_per_image[i].to(dtype=latent.dtype)
        noisy_list.append((1 - t) * latent + t * noise)
        target_list.append(noise - latent)

    preds = model.forward_packed(noisy_list, t_per_image, contexts, use_checkpoint=use_checkpoint)

    per_image = torch.stack([
        loss_fn.compute(pred.float(), target.float(), t_per_image[i:i + 1]).mean()
        for i, (pred, target) in enumerate(zip(preds, target_list))
    ])
    if per_image_weights is not None:
        per_image = per_image * per_image_weights.to(device=per_image.device, dtype=per_image.dtype)

    loss = per_image.mean()
    info = {
        "visual_seqlens": [int(p.shape[-2] * p.shape[-1]) // 4 for p in preds],
        "per_image_loss": per_image.detach(),
    }
    return loss, preds, info
