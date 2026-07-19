"""NaViT / Patch-n-Pack 训练步骤核心：逐图加噪 → 块对角打包前向 → 逐图 loss。

G 张异构图拼进一条序列，每图只注意自己的 token（block-diagonal self-attn）和
自己的 caption（block-diagonal cross-attn），每图带自己的 timestep（per-token AdaLN）。
零 padding、走 xformers varlen 快内核。

公开：
- pack_cross_embeddings — 把 per-image text embedding 拼成一条序列
- navit_packed_forward_and_loss — 一个 NaViT 训练步（前向 + 逐图 loss）
"""

from __future__ import annotations

from typing import Sequence

import torch

from training.noise import make_noise


def pack_cross_embeddings(
    cross: torch.Tensor,
    t5_attn: torch.Tensor | None,
    navit_text_trim_padding: bool = False,
) -> tuple[torch.Tensor, list[int]]:
    """把 per-image text embedding 拼成一条序列，供 block-diagonal cross-attn。

    Args:
        cross: ``[G, L, D]`` — per-image text embedding（L=512 padded）。
        t5_attn: ``[G, L]`` — per-image attention mask（1=有效 token）。
            仅 ``navit_text_trim_padding=True`` 时使用。
        navit_text_trim_padding: True 时按每图有效 token 数截断 padding（cross-attn
            提速、不注意 padding 位）；False 时每图带完整 512-pad（与标准路径行为一致）。

    Returns:
        (cross_packed ``[1, ΣL, D]``, text_seqlens ``[G]``)
    """
    G, L, D = cross.shape
    if navit_text_trim_padding and t5_attn is not None:
        tlens = t5_attn.sum(dim=1).clamp(min=1).tolist()
        tlens = [min(int(n), L) for n in tlens]
        cross_packed = torch.cat(
            [cross[i : i + 1, : tlens[i], :] for i in range(G)], dim=1
        )
        return cross_packed, tlens
    # Legacy 512-pad path: every image carries the full padded caption.
    cross_packed = cross.reshape(1, G * L, D)
    return cross_packed, [L] * G


def navit_packed_forward_and_loss(
    model,
    latents_list: Sequence[torch.Tensor],
    t_per_image: torch.Tensor,
    cross_packed: torch.Tensor,
    text_seqlens: Sequence[int],
    loss_fn,
    *,
    noise_offset: float = 0.0,
    pyramid_iters: int = 0,
    pyramid_discount: float = 0.35,
    use_checkpoint: bool = False,
    per_image_weights: torch.Tensor | None = None,
):
    """一个 NaViT/Patch-n-Pack 训练步：逐图加噪 → 打包前向 → 逐图 loss。

    G 张异构图各自加噪（各自的 t / 形状）→ patchify → 拼成一条序列 →
    ``forward_packed_navit`` 块对角前向 → 逐图 token loss 的 segment 均值。

    Args:
        model: ``MiniTrainDIT``（需有 ``patchify_latents_to_tokens`` /
            ``forward_packed_navit``）。
        latents_list: G 个 clean latent，每个 ``[1,C,T,h_i,w_i]`` 或 ``[C,T,h_i,w_i]``。
        t_per_image: ``[G]`` — per-image flow-matching timestep。
        cross_packed: ``[1, ΣL, D]`` — text embedding 拼接序列。
        text_seqlens: ``[G]`` — per-image caption token 数（sum == ΣL）。
        loss_fn: ``LossProtocol`` — ``compute(pred, target, t) -> per-element loss``。
        noise_offset / pyramid_iters / pyramid_discount: 噪声参数（透传 ``make_noise``）。
        use_checkpoint: 逐块梯度检查点（峰值激活 ≈ 1 block）。
        per_image_weights: ``[G]`` per-image 权重（正则集 ``loss_weight`` × ``loss_weighting``
            的 t-dependent 权重，由训练循环按 per-image t 组合传入；None=等权，行为中立）。

    Returns:
        (loss, pred, info) — ``loss`` 为 per-image 均值（带梯度）；
        ``info`` 携带 ``visual_seqlens`` 与 detached ``per_image_loss`` 供 telemetry。
    """
    G = len(latents_list)
    if G == 0:
        raise ValueError("navit pack is empty")
    if len(text_seqlens) != G:
        raise ValueError(f"text_seqlens has {len(text_seqlens)} entries, expected G={G}")

    t_per_image = t_per_image.reshape(-1)
    if t_per_image.shape[0] != G:
        raise ValueError(f"t_per_image has {t_per_image.shape[0]} entries, expected G={G}")

    noisy_tok_list, target_tok_list, grid_list, vseq = [], [], [], []
    for i, lat in enumerate(latents_list):
        if lat.dim() == 4:
            lat = lat.unsqueeze(0)
        ti = t_per_image[i].to(dtype=lat.dtype)
        noise_i = make_noise(
            lat,
            noise_offset=noise_offset,
            pyramid_iters=pyramid_iters,
            pyramid_discount=pyramid_discount,
        )
        t_exp = ti.view(1, 1, 1, 1, 1)
        noisy_i = (1 - t_exp) * lat + t_exp * noise_i
        target_i = noise_i - lat
        # noisy 与 target 同形，在 batch 维拼成 [2,C,T,h,w] 一次 patchify 后切片：
        # rearrange 逐 batch 行独立 → 与分别调用逐 bit 一致，patchify 调用减半。
        btok, bgrid, _m, bsize = model.patchify_latents_to_tokens(
            torch.cat([noisy_i, target_i], dim=0)
        )
        noisy_tok_list.append(btok[:1])
        target_tok_list.append(btok[1:])
        grid_list.append(bgrid[:1])
        vseq.append(int(btok[:1].shape[1]))

    tokens = torch.cat(noisy_tok_list, dim=1)         # [1, ΣN, M]
    target_tokens = torch.cat(target_tok_list, dim=1)
    grid = torch.cat(grid_list, dim=2)                # [1, 2, ΣN]

    pred = model.forward_packed_navit(
        tokens, t_per_image, cross_packed, grid, vseq,
        [int(s) for s in text_seqlens],
        use_checkpoint=use_checkpoint,
    )

    # Per-image loss: elementwise loss → patch 维均值 → 图内 token 均值。
    # loss_fn.compute 返回 per-element loss（reduction='none'），对任意 shape 都成立。
    # Studio 的 mse/huber 都不使用 t 参数（constant huber），故传 t_per_image [G] 无害。
    loss_map = loss_fn.compute(pred.float(), target_tokens.float(), t_per_image)  # [1, ΣN, M]
    token_loss = loss_map.mean(dim=-1)[0]              # [ΣN] fp32

    counts = torch.tensor(vseq, device=token_loss.device)
    seg_id = torch.repeat_interleave(
        torch.arange(G, device=token_loss.device), counts
    )
    onehot = (
        seg_id.unsqueeze(0) == torch.arange(G, device=token_loss.device).unsqueeze(1)
    ).to(token_loss.dtype)                             # [G, ΣN]
    per_image = (onehot @ token_loss) / counts.to(token_loss.dtype)  # [G], grad-bearing

    # per-image 权重：正则集 loss_weight × timestep-dependent loss_weighting
    # （min_snr / cosmap / detail_inv_t）。由训练循环按 per-image t 组合后传入，与标准
    # 路径的 per-sample 加权对称——navit 的逐图 t 正好对应 per-sample SNR 权重语义。
    # None 时行为中立（等价全 1.0）。
    if per_image_weights is not None:
        per_image = per_image * per_image_weights.to(
            device=per_image.device, dtype=per_image.dtype
        )

    loss = per_image.mean()
    info = {
        "visual_seqlens": vseq,
        "per_image_loss": per_image.detach(),
    }
    return loss, pred, info
