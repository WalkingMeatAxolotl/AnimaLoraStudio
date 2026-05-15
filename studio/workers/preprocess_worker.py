"""预处理 worker 子进程入口（放大第一阶段）。

由 supervisor 启动：`python -m studio.workers.preprocess_worker --job-id N`。

读 project_jobs 行 → 解析 params → 串行调
`studio.services.upscaler.upscale_file()` → 写日志 → 退出码反映成败。

日志规范：只走 stdout（supervisor 重定向到 log 文件），不要再 open 同一个
log 文件，避免 LogTailer 读两次。

取消：worker 主体在每张图前检测 SIGTERM/CTRL_BREAK 信号（Python 解释器
默认对 SIGTERM 抛 KeyboardInterrupt 在 main thread 里）；当前轮的图处理完
后干净退出，已写盘的产物保留（增量）。
"""
from __future__ import annotations

import argparse
import signal
import sys
import traceback
from pathlib import Path

from studio import db, preprocess, project_jobs, projects
from studio.services import model_downloader, upscaler


_stop_requested = False


def _on_signal(_signum, _frame) -> None:  # pragma: no cover - signal path
    global _stop_requested
    _stop_requested = True


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _on_signal)
    if hasattr(signal, "SIGBREAK"):  # Windows
        signal.signal(signal.SIGBREAK, _on_signal)  # type: ignore[attr-defined]


def run(job_id: int) -> int:  # noqa: PLR0912, PLR0915 - 主流程线性可读
    _install_signal_handlers()

    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    if not job:
        print(f"[error] job {job_id} not found", flush=True)
        return 1
    if job["kind"] != preprocess.PREPROCESS_KIND:
        print(f"[error] wrong kind: {job['kind']}", flush=True)
        return 1

    params = job.get("params_decoded") or {}

    def log(line: str) -> None:
        print(line, flush=True)

    try:
        with db.connection_for() as conn:
            project = projects.get_project(conn, job["project_id"])
        if not project:
            log(f"[error] project {job['project_id']} missing")
            return 1

        mode = params.get("mode", "all")
        names = params.get("names") or None
        model_label = params.get("model", preprocess.DEFAULT_MODEL)
        tile_size = int(params.get("tile_size", preprocess.DEFAULT_TILE_SIZE))
        tile_pad = int(params.get("tile_pad", preprocess.DEFAULT_TILE_PAD))
        device = params.get("device", preprocess.DEFAULT_DEVICE)

        download_dir, preprocess_dir = preprocess.project_paths(project)
        preprocess_dir.mkdir(parents=True, exist_ok=True)

        # 模型权重必须先下载（UI 在开始按钮前会引导用户下载）
        model_path = model_downloader.upscaler_target(model_label)
        if not model_path.exists():
            log(
                f"[error] 模型权重不存在：{model_path}（请先在设置页下载 {model_label}）"
            )
            return 1

        try:
            sources = preprocess.resolve_targets(project, mode=mode, names=names)
        except preprocess.PreprocessError as exc:
            log(f"[error] 解析目标失败: {exc}")
            return 1

        total = len(sources)
        if total == 0:
            log("[done] 没有需要处理的图（已全部预处理）")
            return 0

        log(
            f"[start] mode={mode} model={model_label} tile={tile_size}+{tile_pad} "
            f"device={device} total={total}"
        )

        # 提示用户：若 cuda 不可用而 device='auto'，会自动降级 cpu（在 upscaler 里 log）
        # 这里只打输入参数，实际 device 在每张 upscale 调用时由 upscaler 决定。

        succeeded = 0
        failed = 0
        skipped = 0

        for idx, src_name in enumerate(sources, start=1):
            if _stop_requested:
                log(f"[cancel] 收到取消信号，已处理 {idx - 1}/{total}")
                break
            src_path = download_dir / src_name
            if not src_path.exists():
                log(f"[skip] ({idx}/{total}) {src_name}: 源已不存在")
                skipped += 1
                continue
            dst_path = preprocess.product_path_for(preprocess_dir, src_name)
            # 'all' 是增量（已 resolve 过），'all_force' / 'selected' 可能已存在
            if dst_path.exists() and mode == "all":
                skipped += 1
                continue
            log(f"[upscale] ({idx}/{total}) {src_name} → {dst_path.name}")
            try:
                upscaler.upscale_file(
                    src_path,
                    dst_path,
                    model_path=model_path,
                    label=model_label,
                    tile_size=tile_size,
                    tile_pad=tile_pad,
                    device=device,
                    on_log=log,
                    # 256 给 grid，768 给 curate alt-hover 大图。worker 阶段付一次
                    # decode 代价，前端首次浏览就秒开。
                    prewarm_thumb_sizes=[256, 768],
                )
                succeeded += 1
            except Exception as exc:  # noqa: BLE001 — 单张失败不影响其他
                log(f"[fail] {src_name}: {exc}")
                failed += 1

        log(
            f"[done] succeeded={succeeded} failed={failed} skipped={skipped}"
        )
        # 即使部分失败也返 0（成功完成 job 流程）；失败信息在日志里。
        # 失败率高时用户重跑选中即可。
        return 0
    except Exception as exc:  # noqa: BLE001
        log(f"[error] {exc}")
        print(traceback.format_exc(), flush=True)
        return 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", type=int, required=True)
    args = p.parse_args()
    sys.exit(run(args.job_id))


if __name__ == "__main__":
    main()
