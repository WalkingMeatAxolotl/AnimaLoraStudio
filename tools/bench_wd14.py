"""WD14 打标性能诊断：分阶段计时 + EP / preload / 模型 / 线程数自检。

跑法（仓库根目录）：
    venv/bin/python tools/bench_wd14.py [<图目录>] [--n 10] [--model <hf_id>]
    # Windows: venv\\Scripts\\python.exe tools\\bench_wd14.py ...

不传图目录时默认扫 `studio_data/projects/*/raw_*` 找最近一批图，取前 N 张。
所有日志同时打 stdout 与 `bench_wd14.log`。

它能回答的问题：
1. 当前 onnxruntime 是 GPU 包还是 CPU 包，CUDA EP 真能用吗？
2. preload 命中了几个 torch CUDA so？
3. CPU EP 实际跑在几个 thread？
4. 单图分阶段：preprocess / session.run / postprocess 各占多少？
5. CPU vs GPU 的实测吞吐差多少倍（如果两个 provider 都能创 session）？
"""
from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LOG_PATH = REPO_ROOT / "bench_wd14.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("bench_wd14")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_default_images(n: int) -> list[Path]:
    """扫 studio_data/projects/*/raw_*/ 找最近一批图。"""
    base = REPO_ROOT / "studio_data" / "projects"
    if not base.exists():
        return []
    candidates: list[Path] = []
    for proj in base.iterdir():
        if not proj.is_dir():
            continue
        for sub in proj.iterdir():
            if sub.is_dir() and sub.name.startswith("raw"):
                for f in sub.iterdir():
                    if f.suffix.lower() in IMG_EXTS:
                        candidates.append(f)
                        if len(candidates) >= n:
                            return candidates[:n]
    return candidates[:n]


def report_environment() -> None:
    """打 onnxruntime 包名/版本/EP/CPU/preload 状态。"""
    log.info("=" * 60)
    log.info("ENVIRONMENT")
    log.info("=" * 60)
    log.info("python: %s", sys.version.split()[0])
    log.info("platform: %s", sys.platform)
    log.info("cpu count: %s", os.cpu_count())

    # PP9.5 — preload 在本模块 import 时已跑过；直接读结果
    from studio.services import onnxruntime_setup as ors

    rt = ors.current_runtime()
    log.info(
        "onnxruntime: installed=%s version=%s",
        rt["installed"], rt["version"],
    )
    log.info("providers (advertised): %s", rt["providers"])
    log.info("cuda_available (advertised): %s", rt["cuda_available"])
    log.info("cuda_load_error (last): %s", rt.get("cuda_load_error"))

    pre = rt.get("preload") or {}
    log.info(
        "preload: applied=%s skip=%s candidates=%s preloaded=%d errors=%d",
        pre.get("applied"),
        pre.get("platform_skip"),
        pre.get("candidates"),
        len(pre.get("preloaded") or []),
        len(pre.get("errors") or []),
    )
    for path in pre.get("preloaded") or []:
        log.info("  preload OK: %s", os.path.basename(path))
    for path, reason in pre.get("errors") or []:
        log.info("  preload FAIL: %s — %s", os.path.basename(path), reason)

    cuda = ors.detect_cuda()
    log.info(
        "nvidia-smi: available=%s driver=%s gpu=%s",
        cuda["available"], cuda.get("driver_version"), cuda.get("gpu_name"),
    )


def probe_provider(model_path: str, provider: str) -> tuple[bool, str, object]:
    """真去创 InferenceSession；返回 (ok, msg, session_or_err_str)。

    主要分辨 provider 在 advertised list 里但 dlopen 真挂的情况。
    """
    import onnxruntime as ort

    try:
        sess = ort.InferenceSession(model_path, providers=[provider])
    except Exception as exc:  # noqa: BLE001
        return False, str(exc), None
    actual = sess.get_providers()
    return True, f"providers={actual}", sess


def time_stage(fn, *args, **kwargs):
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    return res, (time.perf_counter() - t0) * 1000


def bench_once(tagger, sess, paths: list[Path], label: str) -> None:
    """对给定 session 跑 paths，分阶段计时打表。"""
    import numpy as np

    log.info("-" * 60)
    log.info("RUN [%s] on %d images", label, len(paths))
    log.info("intra_op_num_threads: %s", sess.get_session_options().intra_op_num_threads or "(default)")
    log.info("inter_op_num_threads: %s", sess.get_session_options().inter_op_num_threads or "(default)")

    # 注入 session 到 tagger（绕过 prepare 自己创的）
    tagger._session = sess
    if not tagger._tags:
        # selected_tags.csv 必须读出来才能 postprocess；走一次 prepare 拿 tags
        # （它会再创一个 session，丢掉就行）
        old = tagger._session
        tagger._session = None
        tagger.prepare()
        tagger._session = old
    name_in = sess.get_inputs()[0].name

    # warmup（首次 run 含 graph 优化 / kernel 编译，必须排除）
    from PIL import Image
    with Image.open(paths[0]) as im:
        warm = tagger._preprocess(im)
    sess.run(None, {name_in: np.stack([warm], axis=0).copy()})

    pre_ms: list[float] = []
    inf_ms: list[float] = []
    post_ms: list[float] = []
    for p in paths:
        with Image.open(p) as im:
            arr, t_pre = time_stage(tagger._preprocess, im)
        batch = np.stack([arr], axis=0).copy()
        logits, t_inf = time_stage(sess.run, None, {name_in: batch})
        _, t_post = time_stage(tagger._postprocess_one, logits[0][0])
        pre_ms.append(t_pre)
        inf_ms.append(t_inf)
        post_ms.append(t_post)

    def stats(xs: list[float]) -> str:
        return (
            f"mean={statistics.mean(xs):7.1f}ms "
            f"median={statistics.median(xs):7.1f}ms "
            f"min={min(xs):7.1f}ms max={max(xs):7.1f}ms"
        )

    log.info("preprocess  : %s", stats(pre_ms))
    log.info("session.run : %s", stats(inf_ms))
    log.info("postprocess : %s", stats(post_ms))
    total = sum(pre_ms) + sum(inf_ms) + sum(post_ms)
    log.info(
        "TOTAL %.1fs over %d images = %.2f img/s (%.1fms/img)",
        total / 1000, len(paths), len(paths) * 1000 / total, total / len(paths),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="图目录；不传时自动找 raw_*")
    parser.add_argument("--n", type=int, default=10, help="跑几张（warmup 不计）")
    parser.add_argument(
        "--model",
        default=None,
        help="覆盖 secrets.wd14.model_id（如 SmilingWolf/wd-vit-tagger-v3）",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "cpu", "gpu", "both"],
        default="both",
        help="跑哪个 EP；both = 都跑做对比",
    )
    args = parser.parse_args()

    report_environment()

    if args.path:
        d = Path(args.path)
        paths = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS][: args.n]
    else:
        paths = find_default_images(args.n)
    if len(paths) < 2:
        log.error("找不到足够的图（需要 ≥2，找到 %d）。指定路径或先建 project。", len(paths))
        return 1
    log.info("images: %d 张，第一张 %s", len(paths), paths[0].name)

    from studio.services.wd14_tagger import WD14Tagger

    overrides = {"model_id": args.model} if args.model else {}
    tagger = WD14Tagger(overrides=overrides or None)
    model_dir = tagger._resolve_model_dir()
    onnx_path = str(model_dir / "model.onnx")
    size_mb = (model_dir / "model.onnx").stat().st_size / (1024 * 1024)
    log.info("model: %s (%.1f MB)", model_dir.name, size_mb)

    log.info("=" * 60)
    log.info("PROBE PROVIDERS")
    log.info("=" * 60)
    cpu_ok, cpu_msg, cpu_sess = probe_provider(onnx_path, "CPUExecutionProvider")
    log.info("CPU EP: ok=%s %s", cpu_ok, cpu_msg if cpu_ok else cpu_msg[:200])
    gpu_ok, gpu_msg, gpu_sess = (False, "skipped (no advertised CUDA EP)", None)
    import onnxruntime as ort

    if "CUDAExecutionProvider" in ort.get_available_providers():
        gpu_ok, gpu_msg, gpu_sess = probe_provider(onnx_path, "CUDAExecutionProvider")
        log.info("GPU EP: ok=%s %s", gpu_ok, gpu_msg if gpu_ok else gpu_msg[:300])
    else:
        log.info("GPU EP: %s", gpu_msg)

    if args.provider in ("cpu", "both") and cpu_ok:
        bench_once(tagger, cpu_sess, paths, "CPU")
    if args.provider in ("gpu", "both") and gpu_ok:
        bench_once(tagger, gpu_sess, paths, "GPU")
    if args.provider == "auto":
        sess = gpu_sess if gpu_ok else cpu_sess
        bench_once(tagger, sess, paths, "GPU" if gpu_ok else "CPU")

    log.info("done. log saved to %s", LOG_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
