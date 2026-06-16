#!/usr/bin/env python3
"""VAE-only 显存/提交压力测试 —— 隔离 VAE，复现「偶发卡死 + commit 暴涨」。

只加载 VAE（不加载 transformer/qwen，排除混淆），在递增分辨率/batch 下跑
encode + decode（走 *raw* WanVAE_.model，不经 VAEWrapper 的 tile 回退，以便看到
真实峰值 workspace）。每步进程内同时报：

  - torch GPU：allocated / reserved / **本步峰值 reserved** / mem_get_info free
  - 系统提交 commit（GetPerformanceInfo，复用 tools/mem_probe.py）

熔断：commit 超 --max-commit-gb 立即停，保护机器。
torch 显存历史全程记录，结束/Ctrl+Break 转储 snapshot（喂 pytorch.org/memory_viz）。

跑（务必用 venv 解释器）：
  venv/Scripts/python.exe tools/spike/vae_stress.py \
      --vae models/vae/qwen_image_vae.safetensors \
      --res 512,768,1024,1536,2048 --batch 1 --dtype bf16 --max-commit-gb 130
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import sys
import time
from pathlib import Path

# ---- 复现 anima_generate 的 import 顺序：torch 先（expandable_segments 因此不自动生效）
import torch  # noqa: E402

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
_RUNTIME = _REPO / "runtime"
for _p in (_RUNTIME, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 复用 mem_probe 的系统提交读数
_mp_spec = importlib.util.spec_from_file_location("mem_probe", _REPO / "tools" / "mem_probe.py")
_mp = importlib.util.module_from_spec(_mp_spec)
_mp_spec.loader.exec_module(_mp)
GB = 1024.0 ** 3


def commit_gb() -> float:
    try:
        ct, _, _ = _mp.system_commit_bytes()
        return ct / GB
    except Exception:  # noqa: BLE001
        return -1.0


def gpu_line(tag: str) -> str:
    a = torch.cuda.memory_allocated() / GB
    r = torch.cuda.memory_reserved() / GB
    mr = torch.cuda.max_memory_reserved() / GB
    free, total = torch.cuda.mem_get_info()
    return (f"{tag:18s} commit={commit_gb():6.1f}G | "
            f"gpu_alloc={a:5.2f}G reserved={r:5.2f}G peakRsv={mr:5.2f}G "
            f"free={free/GB:5.1f}/{total/GB:.1f}G")


def main() -> None:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="models/vae/qwen_image_vae.safetensors")
    ap.add_argument("--res", default="512,768,1024,1536,2048", help="逗号分隔的边长")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dtype", choices=["bf16", "fp32", "fp16"], default="bf16")
    ap.add_argument("--mode", choices=["both", "encode", "decode"], default="both")
    ap.add_argument("--max-commit-gb", type=float, default=130.0, help="commit 超此值即熔断")
    ap.add_argument("--snapshot", default="G:/tmp/vae_mem.pickle")
    ap.add_argument("--cudnn-benchmark", action="store_true")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[args.dtype]
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    device = "cuda"

    print(gpu_line("[start]"))
    print(f"torch {torch.__version__} cudnn {torch.backends.cudnn.version()} "
          f"cudnn.benchmark={torch.backends.cudnn.benchmark} dtype={args.dtype} batch={args.batch}")

    # torch 显存历史 + Ctrl+Break 转储
    try:
        torch.cuda.memory._record_memory_history(max_entries=200_000)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] record_memory_history 不可用: {e}")

    def _dump(*_):
        try:
            Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._dump_snapshot(args.snapshot)
            print(f"\n[snapshot] dumped {args.snapshot}")
        except Exception as e:  # noqa: BLE001
            print(f"[snapshot] fail: {e}")
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _dump)

    # ---- 加载 VAE（用 app 同一条 load_vae 路径）
    import anima_train as _T  # noqa: E402
    repo_root = _T.find_diffusion_pipe_root()
    vae_path = _T.resolve_path_best_effort(args.vae, [Path.cwd(), _REPO, repo_root])
    print(f"[load] VAE {vae_path} (dtype={args.dtype}) ...")
    t = time.perf_counter()
    vae = _T.load_vae(vae_path, device, dtype, repo_root)
    torch.cuda.synchronize()
    print(gpu_line("[vae loaded]") + f"  load={time.perf_counter()-t:.1f}s")

    resolutions = [int(x) for x in args.res.split(",") if x.strip()]
    B = args.batch

    for res in resolutions:
        c0 = commit_gb()
        if c0 >= args.max_commit_gb:
            print(f"\n[ABORT] commit {c0:.1f}G ≥ 熔断阈值 {args.max_commit_gb}G，停止加压。")
            break
        print(f"\n--- res={res}x{res} batch={B} ---")
        torch.cuda.reset_peak_memory_stats()
        try:
            if args.mode in ("encode", "both"):
                px = torch.randn(B, 3, res, res, device=device, dtype=dtype).clamp(-1, 1)
                torch.cuda.synchronize(); t = time.perf_counter(); c_pre = commit_gb()
                with torch.no_grad():
                    z = vae.model.encode(px.unsqueeze(2), vae.scale)  # [B,16,1,h,w]
                torch.cuda.synchronize()
                print(gpu_line(f"  encode {res}") +
                      f"  Δcommit={commit_gb()-c_pre:+.1f}G  t={time.perf_counter()-t:.2f}s  z={tuple(z.shape)}")
                del px
            else:
                h = res // 8
                z = torch.randn(B, 16, 1, h, h, device=device, dtype=dtype)

            if args.mode in ("decode", "both"):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(); t = time.perf_counter(); c_pre = commit_gb()
                with torch.no_grad():
                    img = vae.model.decode(z, vae.scale)  # raw, 无 tile
                torch.cuda.synchronize()
                print(gpu_line(f"  decode {res}") +
                      f"  Δcommit={commit_gb()-c_pre:+.1f}G  t={time.perf_counter()-t:.2f}s  img={tuple(img.shape)}")
                del img
            del z
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [OOM] res={res}: {str(e)[:140]}")
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  [ERR] res={res}: {type(e).__name__}: {str(e)[:180]}")
            torch.cuda.empty_cache()

    _dump()
    print("\n" + gpu_line("[end]"))
    print(f"提示：把 {args.snapshot} 拖到 https://pytorch.org/memory_viz 看最大分配调用栈。")


if __name__ == "__main__":
    main()
