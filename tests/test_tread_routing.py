"""TREAD token routing 测试（arXiv 2501.04765）。

不依赖真实 Anima 权重：用 stub 模型（暴露 prepare_embedded_sequence / t_embedder /
blocks / final_layer / unpatchify）验证路由 plumbing：
- tread_route_indices：形状 / 升序 / 数量 / 取值范围 / 逐样本
- 路由段只更新保留 token、丢弃 token 段内恒等旁路（gather/scatter/伪网格 reshape 正确）
- identity block 下 tread 路径与全 token 路径逐元素相等（plumbing 无损）
- model.eval() / extra_per_block_pos_emb / T>1 等边界行为

真实权重的逐 bit 等价 + 单步耗时下降由 tools/tread_smoke.py（real-weight smoke test）覆盖。
测试卫生：纯 CPU、固定随机种子。
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from training.model_loading import forward_with_optional_checkpoint, tread_route_indices


# ---------------------------------------------------------------------------
# tread_route_indices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ratio,n,expected_keep", [(0.5, 16, 8), (0.25, 16, 12), (0.9, 10, 1)])
def test_route_indices_count_and_sorted(ratio, n, expected_keep) -> None:
    torch.manual_seed(0)
    idx = tread_route_indices(4, n, ratio, device="cpu")
    assert idx.shape == (4, expected_keep)
    # 升序 + 取值范围 + 逐行唯一
    for row in idx:
        assert torch.equal(row, row.sort().values)
        assert int(row.min()) >= 0 and int(row.max()) < n
        assert len(set(row.tolist())) == len(row)


def test_route_indices_keep_at_least_one() -> None:
    idx = tread_route_indices(2, 8, 0.999, device="cpu")
    assert idx.shape == (2, 1)


def test_route_indices_per_sample_differs() -> None:
    torch.manual_seed(0)
    idx = tread_route_indices(8, 64, 0.5, device="cpu")
    # 逐样本独立抽样：极不可能 8 行全相同
    assert not all(torch.equal(idx[0], idx[j]) for j in range(1, 8))


# ---------------------------------------------------------------------------
# Stub 模型
# ---------------------------------------------------------------------------

class _AddBlock(nn.Module):
    """每次 forward 给所有 token +delta；忽略 rope/adaln（只测路由 plumbing）。"""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, x_B_T_H_W_D, emb, cross, rope_emb_L_1_1_D=None,
                adaln_lora_B_T_3D=None, extra_per_block_pos_emb=None):
        return x_B_T_H_W_D + self.delta


class _StubModel(nn.Module):
    def __init__(self, n_blocks=6, h=4, w=4, d=8, t=1, extra_pos=False):
        super().__init__()
        self.blocks = nn.ModuleList([_AddBlock(1.0) for _ in range(n_blocks)])
        self._h, self._w, self._d, self._t = h, w, d, t
        self._extra_pos = extra_pos

    def prepare_embedded_sequence(self, latents, fps=None, padding_mask=None):
        b = latents.shape[0]
        x = torch.zeros(b, self._t, self._h, self._w, self._d)
        rope = torch.zeros(self._t * self._h * self._w, 1, 1, 4)
        extra = torch.zeros_like(x) if self._extra_pos else None
        return x, rope, extra

    def t_embedder(self, timesteps):
        b = timesteps.shape[0]
        return torch.zeros(b, self._t, self._d), torch.zeros(b, self._t, 3 * self._d)

    def t_embedding_norm(self, t):
        return t

    def final_layer(self, x, t_embedding, adaln_lora_B_T_3D=None):
        return x

    def unpatchify(self, x):
        return x

    def forward(self, latents, timesteps, cross, padding_mask=None):
        # 非路由参考前向（与 forward_with_optional_checkpoint 的 model() 兜底一致）
        x, rope, extra = self.prepare_embedded_sequence(latents)
        t, adaln = self.t_embedder(timesteps)
        for blk in self.blocks:
            x = blk(x, t, cross, rope_emb_L_1_1_D=rope, adaln_lora_B_T_3D=adaln,
                    extra_per_block_pos_emb=extra)
        return self.unpatchify(self.final_layer(x, t, adaln_lora_B_T_3D=adaln))


def _run(model, **tread_kw):
    b = 2
    latents = torch.zeros(b, model._d, model._h, model._w)
    timesteps = torch.zeros(b)
    cross = torch.zeros(b, 4, model._d)
    pad = torch.zeros(b, 1, model._h, model._w)
    return forward_with_optional_checkpoint(model, latents, timesteps, cross, pad, **tread_kw)


# ---------------------------------------------------------------------------
# 路由语义
# ---------------------------------------------------------------------------

def test_routed_segment_only_updates_kept_tokens() -> None:
    """+1/block：保留 token 走满 n_blocks 次，丢弃 token 在段内旁路少 seg_len 次。"""
    torch.manual_seed(0)
    n_blocks, h, w = 6, 4, 4
    model = _StubModel(n_blocks=n_blocks, h=h, w=w).train()
    seg_s, seg_e = 1, 5  # blocks[1:5) 共 4 个进路由段
    seg_len = seg_e - seg_s
    ratio = 0.5
    out = _run(model, tread_ratio=ratio, tread_start_layer=seg_s, tread_end_layer=seg_e)

    n_tok = h * w
    n_keep = round(n_tok * (1 - ratio))
    vals = out.reshape(out.shape[0], n_tok, -1)[..., 0]  # (B, N) 每 token 的累计值
    # 保留 token == n_blocks；丢弃 token == n_blocks - seg_len
    kept = (vals == float(n_blocks)).sum(dim=1)
    dropped = (vals == float(n_blocks - seg_len)).sum(dim=1)
    assert torch.all(kept == n_keep)
    assert torch.all(dropped == n_tok - n_keep)


def test_identity_blocks_routing_is_lossless() -> None:
    """identity block 下，tread 路径输出与全 token 路径逐元素相等（plumbing 无损）。"""
    n_blocks, h, w, d = 6, 4, 4, 8
    model = _StubModel(n_blocks=n_blocks, h=h, w=w, d=d).train()
    for blk in model.blocks:
        blk.delta = 0.0
    base = torch.full((2, n_blocks * 0 + 1 * 1, h, w, d), 3.14)  # 任意非零参考

    # 用一个会改值的 block 也行，但 identity 最直接：注入非零初始值
    def _prep(latents, fps=None, padding_mask=None):
        return base.clone(), torch.zeros(h * w, 1, 1, 4), None
    model.prepare_embedded_sequence = _prep

    out_tread = _run(model, tread_ratio=0.5, tread_start_layer=1, tread_end_layer=5)
    out_plain = _run(model, tread_ratio=0.0)
    assert torch.allclose(out_tread, out_plain)


def test_tread_off_in_eval_mode() -> None:
    """model.eval() 时即使传 tread_ratio>0 也不路由（采样/eval 双保险）。"""
    n_blocks, h, w = 6, 4, 4
    model = _StubModel(n_blocks=n_blocks, h=h, w=w).eval()
    out = _run(model, tread_ratio=0.5, tread_start_layer=1, tread_end_layer=5)
    n_tok = h * w
    vals = out.reshape(out.shape[0], n_tok, -1)[..., 0]
    assert torch.all(vals == float(n_blocks))  # 全 token 都走满，无丢弃


def test_tread_rejects_extra_pos_emb() -> None:
    model = _StubModel(extra_pos=True).train()
    with pytest.raises(RuntimeError, match="extra_per_block_pos_emb"):
        _run(model, tread_ratio=0.5, tread_start_layer=1, tread_end_layer=5)


def test_tread_rejects_temporal_gt_one() -> None:
    model = _StubModel(t=2).train()
    with pytest.raises(RuntimeError, match="T=1"):
        _run(model, tread_ratio=0.5, tread_start_layer=1, tread_end_layer=5)


def test_tread_rejects_bad_segment() -> None:
    model = _StubModel(n_blocks=6).train()
    with pytest.raises(ValueError, match="TREAD"):
        _run(model, tread_ratio=0.5, tread_start_layer=4, tread_end_layer=2)


def test_negative_layer_indices_resolve() -> None:
    """负索引按 python 切片语义解析（end=-1 → 倒数第 1 之前）。"""
    torch.manual_seed(0)
    n_blocks, h, w = 6, 4, 4
    model = _StubModel(n_blocks=n_blocks, h=h, w=w).train()
    # blocks[1:-1) = blocks[1:5)，seg_len=4
    out = _run(model, tread_ratio=0.5, tread_start_layer=1, tread_end_layer=-1)
    n_tok = h * w
    vals = out.reshape(out.shape[0], n_tok, -1)[..., 0]
    assert (vals == float(n_blocks - 4)).any()  # 有 token 少走 4 个 block
