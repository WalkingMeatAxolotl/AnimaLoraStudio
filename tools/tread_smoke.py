"""TREAD 真模型冒烟测试（arXiv 2501.04765）。

加载 Anima 底模（CPU），验证：
1) 手动 per-block 路径与 model() 直通路径数值一致（tread 关）— 证明改写的前向无回归
2) TREAD 路由开启时前向可跑、输出有限、形状正确、扰动量级温和（非爆炸）
3) 打印 CPU 参考耗时（direct / manual / tread）便于粗看路由是否真的少算

CI 不跑（需真实权重）；维护者本地验证 + 复核单步提速时用。

用法:
  python tools/tread_smoke.py --base <models/transformers/anima-base.safetensors> \
      [--dtype fp32|bf16] [--hw 32] [--ratio 0.5] [--start 2] [--end -2]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "runtime"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from training.models import load_anima_model  # noqa: E402
from training.model_loading import forward_with_optional_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Anima transformer safetensors 路径")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--hw", type=int, default=32, help="latent 边长（32 → 16x16=256 token）")
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--start", type=int, default=2)
    ap.add_argument("--end", type=int, default=-2)
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    t0 = time.time()
    model = load_anima_model(args.base, "cpu", dtype, REPO_ROOT / "models", flash_attn=False)
    print(f"model loaded in {time.time() - t0:.1f}s; blocks={len(model.blocks)}")
    model.train()  # TREAD 仅训练模式生效；权重已 requires_grad_(False)

    torch.manual_seed(0)
    lat = torch.randn(1, 16, 1, args.hw, args.hw, dtype=dtype)
    cross = torch.randn(1, 512, 1024, dtype=dtype)
    ts = torch.tensor([[0.7]], dtype=dtype)
    pm = torch.zeros(1, 1, args.hw, args.hw, dtype=dtype)

    with torch.no_grad():
        t0 = time.time()
        out_direct = model(lat, ts, cross, padding_mask=pm)
        t_direct = time.time() - t0
        t0 = time.time()
        out_manual = forward_with_optional_checkpoint(model, lat, ts, cross, pm, use_checkpoint=True)
        t_manual = time.time() - t0
        t0 = time.time()
        out_tread = forward_with_optional_checkpoint(
            model, lat, ts, cross, pm, use_checkpoint=True,
            tread_ratio=args.ratio, tread_start_layer=args.start, tread_end_layer=args.end)
        t_tread = time.time() - t0

    d_manual = (out_manual.float() - out_direct.float()).abs().max().item()
    ref = out_direct.float()
    rel_tread = ((out_tread.float() - ref).norm() / ref.norm().clamp(min=1e-12)).item()
    print(f"shapes: direct={tuple(out_direct.shape)} tread={tuple(out_tread.shape)}")
    print(f"[1] manual-vs-direct max|diff| = {d_manual:.3e}  (期望 ~0：per-block 路径数值等价)")
    print(f"[2] tread finite={bool(torch.isfinite(out_tread).all())}  "
          f"rel|out_tread-out_direct|={rel_tread:.4f}  (期望 0.05~0.5 量级：扰动温和非爆炸)")
    print(f"timing(cpu参考): direct={t_direct:.1f}s manual={t_manual:.1f}s tread={t_tread:.1f}s")
    ok = (d_manual < 1e-3 if args.dtype == "fp32" else d_manual < 0.1) \
        and bool(torch.isfinite(out_tread).all()) and out_tread.shape == out_direct.shape \
        and 0.0 < rel_tread < 1.0
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
