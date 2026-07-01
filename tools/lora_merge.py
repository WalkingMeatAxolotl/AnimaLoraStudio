"""多个 LoRA 按权重精确 merge 成单个 plain LoRA（调色 POC）。

背景：Generate 页多 LoRA 叠加是线性的（studio/services/inference/core.py::apply_loras，
每份 LoRA 独立 inject，forward 累加 delta），即：

    ΔW_total = Σ_i  weight_i · (alpha_i / rank_i) · ΔW_i

因此把若干 LoRA（含 slider 调色 LoRA 的负权重）merge 成一个文件在数学上是
可精确完成的 —— 不需要 SVD 近似：

  - plain LoRA 层：ΔW = up @ down，直接把 weight 烘进因子后按秩维拼接。
  - LoKr 层（w2 分解形态）：利用 Kronecker 混合积恒等式
        kron(w1, w2a @ w2b) = kron(w1, w2a) @ kron(I, w2b)
    无损展开成秩 ≤ in_l·r 的两矩阵乘积，再与其他源拼接。

输出为 algo="lora" 的 kohya 格式文件（lora_unet_* 前缀 + ss_* metadata），
alpha=rank（缩放因子 1），挂载权重 1.0 即等价于原多 LoRA 组合。

用法示例（style ×1.0 + 色温 ×-5 + 饱和 ×-5）：

    python tools/lora_merge.py \
        --lora path/to/style.safetensors 1.0 \
        --lora path/to/temperature_v6.safetensors -5 \
        --lora path/to/saturation_v6.safetensors -5 \
        --out path/to/merged.safetensors --verify

支持范围（POC）：plain LoRA（Linear 层，无 lora_mid）、LoKr（w1 全矩阵或
w1_a/w1_b 分解 + w2_a/w2_b 分解）。DoRA（weight_decompose）路径非线性、
LoKr 双全矩阵形态展开秩过大，均直接报错拒绝。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from studio.services.inference.core import LoRAMeta, read_lora_meta  # noqa: E402


def _load_layers(path: Path) -> dict[str, dict[str, torch.Tensor]]:
    """按层名（第一个 '.' 之前）分组读取全部张量。"""
    from safetensors import safe_open

    layers: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            layer, _, suffix = key.partition(".")
            layers.setdefault(layer, {})[suffix] = f.get_tensor(key)
    return layers


def _layer_factors(
    layer: str, tensors: dict[str, torch.Tensor], meta: LoRAMeta, path: Path
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """单层张量 → (up (out×r), down (r×in), scale)，fp32，未含用户权重。

    scale = alpha / rank，与 LyCORIS LoConModule / LokrModule 的 self.scale 一致
    （rs_lora 已在入口拒绝，无 √rank 分支）。
    """
    if "dora_scale" in tensors:
        raise SystemExit(f"{path.name} 层 {layer} 含 dora_scale（DoRA 非线性），无法线性 merge")

    if "lora_down.weight" in tensors:  # plain LoRA (LoCon)
        down = tensors["lora_down.weight"].float()
        up = tensors["lora_up.weight"].float()
        if "lora_mid.weight" in tensors or down.dim() != 2:
            raise SystemExit(f"{path.name} 层 {layer} 非 Linear plain LoRA（POC 未支持 conv/tucker）")
        rank = down.shape[0]
        alpha = float(tensors["alpha"]) if "alpha" in tensors else float(rank)
        return up, down, alpha / rank

    if "lokr_w2_a" in tensors:  # LoKr，w2 分解形态
        if "lokr_t2" in tensors:
            raise SystemExit(f"{path.name} 层 {layer} 为 LoKr tucker 形态，POC 未支持")
        if "lokr_w1" in tensors:
            w1 = tensors["lokr_w1"].float()
        else:
            w1 = (tensors["lokr_w1_a"] @ tensors["lokr_w1_b"]).float()
        w2a = tensors["lokr_w2_a"].float()
        w2b = tensors["lokr_w2_b"].float()
        # kron(w1, w2a@w2b) = kron(w1, w2a) @ kron(I_{w1列数}, w2b)
        up = torch.kron(w1, w2a)                                # (out, in_l·r)
        down = torch.kron(torch.eye(w1.shape[1]), w2b)          # (in_l·r, in)
        rank = w2a.shape[1]  # LyCORIS lora_dim；scale = alpha / lora_dim
        alpha = float(tensors["alpha"]) if "alpha" in tensors else float(rank)
        return up, down, alpha / rank

    if "lokr_w2" in tensors:
        raise SystemExit(
            f"{path.name} 层 {layer} 为 LoKr 双全矩阵形态，精确展开秩过大，POC 未支持（需 SVD 路径）"
        )
    raise SystemExit(f"{path.name} 层 {layer} 张量形态无法识别: {sorted(tensors)}")


def _trim_factors(
    up: torch.Tensor, down: torch.Tensor, energy: Optional[float], cap: Optional[int]
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """QR + 核 SVD 把 up@down 截到保留 energy 比例奇异值能量的最小秩，再按 cap 封顶。"""
    q1, r1 = torch.linalg.qr(up)          # up = q1 @ r1
    q2, r2 = torch.linalg.qr(down.T)      # down = r2.T @ q2.T
    u, s, vh = torch.linalg.svd(r1 @ r2.T)
    keep = s.shape[0]
    if energy is not None:
        cum = torch.cumsum(s * s, dim=0)
        keep = min(keep, int(torch.searchsorted(cum, energy * cum[-1]).item()) + 1)
    if cap is not None:
        keep = min(keep, cap)
    new_up = q1 @ (u[:, :keep] * s[:keep])
    new_down = (vh[:keep] @ q2.T)
    return new_up, new_down, keep


def merge(
    sources: list[tuple[Path, float]],
    out_path: Path,
    save_dtype: torch.dtype,
    trim_energy: Optional[float],
    rank_cap: Optional[int],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """执行 merge 并写盘。返回各层最终 (up, down)（fp32，供 verify 复用）。"""
    metas = [read_lora_meta(str(p)) for p, _ in sources]
    for (p, _), m in zip(sources, metas):
        if m.weight_decompose:
            raise SystemExit(f"{p.name} 训练时开了 weight_decompose（DoRA），非线性，无法 merge")
        if m.rs_lora:
            raise SystemExit(f"{p.name} 训练时开了 rs_lora，缩放语义未在 POC 覆盖")

    all_layers: list[dict[str, dict[str, torch.Tensor]]] = [_load_layers(p) for p, _ in sources]
    layer_names = sorted(set().union(*[set(d) for d in all_layers]))

    merged: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    max_rank = 0
    for layer in layer_names:
        ups, downs = [], []
        for (path, weight), layers, meta in zip(sources, all_layers, metas):
            if layer not in layers:
                continue
            up, down, scale = _layer_factors(layer, layers[layer], meta, path)
            coeff = weight * scale
            # 平衡烘焙：|coeff| 开方分摊到两侧，符号归 up，压低半精度存储误差
            s = abs(coeff) ** 0.5
            ups.append(up * (s if coeff >= 0 else -s))
            downs.append(down * s)
        up = torch.cat(ups, dim=1)
        down = torch.cat(downs, dim=0)
        if trim_energy is not None or rank_cap is not None:
            up, down, _ = _trim_factors(up, down, trim_energy, rank_cap)
        merged[layer] = (up, down)
        max_rank = max(max_rank, up.shape[1])

    # 每层保留自己的秩（不零填充）。秩 ≠ max_rank 的层写进 lora_reg_dims
    # （fullmatch 精确 pattern），studio 加载侧由 _apply_reg_dims_ 按层重建形状；
    # ComfyUI 等外部 loader 直接从张量形状 + per-layer alpha 读，天然兼容。
    state: dict[str, torch.Tensor] = {}
    reg_dims: dict[str, int] = {}
    for layer, (up, down) in merged.items():
        r = up.shape[1]
        if r != max_rank:
            reg_dims[layer] = r
        state[f"{layer}.lora_up.weight"] = up.to(save_dtype).contiguous()
        state[f"{layer}.lora_down.weight"] = down.to(save_dtype).contiguous()
        # studio 构建侧 scale 恒为 alpha_global/rank_global=1（reg_dims 不重算 scale）；
        # per-layer alpha=r 让「alpha/rank」式 loader 也得到 scale=1
        state[f"{layer}.alpha"] = torch.tensor(float(r))

    # metadata 对齐 AnimaLycorisAdapter.save()（utils/lycoris_adapter.py）的约定，
    # 保证 read_lora_meta / Generate 页按 algo=lora、scale=1 重建
    network_args = {
        "algo": "lora",
        "preset": "anima_full",
        "dropout": 0.0,
        "rank_dropout": 0.0,
        "module_dropout": 0.0,
        "weight_decompose": False,
        "rs_lora": False,
    }
    if reg_dims:
        network_args["lora_reg_dims"] = reg_dims
    provenance = [{"file": p.name, "weight": w} for p, w in sources]
    metadata = {
        "ss_network_dim": str(max_rank),
        "ss_network_alpha": str(float(max_rank)),
        "ss_network_module": "lycoris.kohya",
        "ss_network_args": json.dumps(network_args),
        "anima_merge_sources": json.dumps(provenance, ensure_ascii=False),
    }
    from safetensors.torch import save_file

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out_path), metadata=metadata)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"已写出 {out_path}  (rank={max_rank}, {len(merged)} 层, {size_mb:.0f} MB)")
    return merged


def verify(
    sources: list[tuple[Path, float]],
    out_path: Path,
    spectrum: bool,
) -> None:
    """逐层对比：写盘文件重建的 ΔW vs 各源 ΔW 加权和（fp32 参考）。"""
    metas = [read_lora_meta(str(p)) for p, _ in sources]
    all_layers = [_load_layers(p) for p, _ in sources]
    out_meta = read_lora_meta(str(out_path))
    out_layers = _load_layers(out_path)

    worst = (0.0, "")
    errs: list[float] = []
    energy_ranks: list[int] = []
    for layer, tensors in out_layers.items():
        up = tensors["lora_up.weight"].float()
        down = tensors["lora_down.weight"].float()
        alpha = float(tensors["alpha"])
        got = (alpha / down.shape[0]) * (up @ down)

        ref = torch.zeros_like(got)
        for (path, weight), layers, meta in zip(sources, all_layers, metas):
            if layer not in layers:
                continue
            u, d, scale = _layer_factors(layer, layers[layer], meta, path)
            ref += (weight * scale) * (u @ d)

        denom = ref.norm().item() or 1.0
        rel = (got - ref).norm().item() / denom
        errs.append(rel)
        if rel > worst[0]:
            worst = (rel, layer)

        if spectrum:
            s = torch.linalg.svdvals(ref)
            cum = torch.cumsum(s * s, dim=0)
            energy_ranks.append(int(torch.searchsorted(cum, 0.999 * cum[-1]).item()) + 1)

    print(
        f"verify: {len(errs)} 层  相对 Frobenius 误差 "
        f"max={max(errs):.3e} (层 {worst[1]})  mean={sum(errs) / len(errs):.3e}  "
        f"(仅存储 dtype 量化噪声；fp32 存储时应 <1e-6)"
    )
    if spectrum and energy_ranks:
        energy_ranks.sort()
        n = len(energy_ranks)
        print(
            f"谱分析: 覆盖 99.9% 能量所需秩  p50={energy_ranks[n // 2]}  "
            f"p90={energy_ranks[int(n * 0.9)]}  max={energy_ranks[-1]} / 实际 rank {out_meta.rank}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--lora",
        nargs=2,
        action="append",
        required=True,
        metavar=("PATH", "WEIGHT"),
        help="源 LoRA 与权重，可重复。权重语义与 Generate 页强度滑条一致",
    )
    parser.add_argument("--out", required=True, help="输出 safetensors 路径")
    parser.add_argument(
        "--dtype", choices=["fp32", "fp16", "bf16"], default="fp16",
        help="存储 dtype（默认 fp16；fp32 无量化损失但体积翻倍）",
    )
    parser.add_argument(
        "--trim-energy", type=float, default=None, metavar="E",
        help="可选 SVD 截秩，保留奇异值能量比例 E（如 0.999）。默认不截，完全精确",
    )
    parser.add_argument(
        "--rank-cap", type=int, default=None, metavar="N",
        help="可选每层秩上限（SVD 截断，牺牲精度换体积）。默认不封顶",
    )
    parser.add_argument("--verify", action="store_true", help="写盘后逐层数值校验")
    parser.add_argument(
        "--spectrum", action="store_true",
        help="verify 时附带奇异值谱分析（评估未来 resize 空间，较慢）",
    )
    args = parser.parse_args(argv)

    sources: list[tuple[Path, float]] = []
    for path_str, weight_str in args.lora:
        path = Path(path_str)
        if not path.exists():
            raise SystemExit(f"文件不存在: {path}")
        sources.append((path, float(weight_str)))

    save_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    for path, weight in sources:
        meta = read_lora_meta(str(path))
        print(f"  {path.name}  weight={weight}  algo={meta.algo} rank={meta.rank} alpha={meta.alpha}")

    merge(sources, Path(args.out), save_dtype, args.trim_energy, args.rank_cap)
    if args.verify or args.spectrum:
        verify(sources, Path(args.out), args.spectrum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
