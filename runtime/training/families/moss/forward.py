"""MOSS 训练前向接缝（MOSS 设计稿 §5.3；ModelFamily.forward_train 在 PR-4 接到这里）。

共享循环的约定：``noisy = (1 − t)·x₀ + t·ε``、``target = ε − x₀``，t∈(0,1) 即 σ；
latent 是 ``FLUX2_F8C32_BN`` 排布（32ch、f8，``[B, 32, 1, h, w]``）。DiT 直接吃这两样，
patchify / ×1000 都在模型内部完成。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class MossTextCondition:
    """右 padding 的 Qwen3.5 last_hidden_state 与有效 token mask（文本缓存产物，PR-3）。"""

    context: Tensor  #: [B, S, 4096]
    attention_mask: Tensor  #: [B, S]，1 = 有效 token


def forward_train(dit, noisy: Tensor, t: Tensor, cond: MossTextCondition, *,
                  use_checkpoint: bool = False) -> Tensor:
    batch = noisy.shape[0]
    if batch > 1:
        # 逐样本 forward 对每个样本把每层调一遍，开 block swap 时换出层每个样本都搬一次；
        # 打包前向每层只调一次（块对角注意力，数学同逐样本）
        mask = cond.attention_mask.bool()
        contexts = [cond.context[i, mask[i]] for i in range(batch)]
        timesteps = torch.as_tensor(t).reshape(-1).expand(batch)
        preds = dit.forward_packed(
            [noisy[i:i + 1] for i in range(batch)], timesteps, contexts,
            use_checkpoint=use_checkpoint,
        )
        return torch.cat(preds)
    return dit(
        noisy,
        t,
        cond.context,
        attention_mask=cond.attention_mask,
        use_checkpoint=use_checkpoint,
    )
