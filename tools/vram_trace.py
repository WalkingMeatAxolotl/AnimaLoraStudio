#!/usr/bin/env python3
r"""High-frequency NVIDIA VRAM tracer for Studio generation diagnostics.

This is intentionally device-first: Windows Task Manager and NVML report the
whole adapter, while ``torch.cuda.max_memory_*`` only sees PyTorch's allocator
inside one process.  Recording both the absolute adapter usage and its delta
from the start makes those numbers comparable.

Examples (run from the repository root)::

    venv\Scripts\python tools\vram_trace.py
    venv\Scripts\python tools\vram_trace.py --pid 12345 --interval-ms 20
    venv\Scripts\python tools\vram_trace.py --duration 120 --out tmp/vram.csv

By default the tracer also follows the newest ``studio_data/tasks/*/run.log``
and puts best-effort Krea2 load-stage markers in the CSV.  Per-process VRAM is
often unavailable with NVIDIA's Windows WDDM driver; device-wide readings are
still valid in that case.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

MIB = 1024.0 ** 2

try:
    import psutil  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    psutil = None


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _valid_used_memory(value: Any, total: int) -> int | None:
    """Normalize NVML's WDDM 'not available' sentinel to ``None``."""
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    if value < 0 or value > max(total * 2, 1 << 50):
        return None
    return value


def merge_process_rows(rows: Iterable[Iterable[Any]], total: int) -> dict[int, int | None]:
    """Merge compute/graphics NVML rows without double-counting a PID."""
    merged: dict[int, int | None] = {}
    for group in rows:
        for item in group:
            try:
                pid = int(item.pid)
            except (AttributeError, TypeError, ValueError):
                continue
            used = _valid_used_memory(getattr(item, "usedGpuMemory", None), total)
            previous = merged.get(pid)
            if previous is None:
                merged[pid] = used
            elif used is not None:
                # The same CUDA process may appear in both lists.  Each value is
                # its complete residency, so use max rather than sum.
                merged[pid] = max(previous, used)
    return merged


@dataclass(frozen=True)
class GpuSample:
    used: int
    total: int
    utilization: int | None
    processes: dict[int, int | None]


class GpuReader:
    def __init__(self, index: int):
        self.index = index
        self.nvml = None
        self.handle = None
        self.smi = shutil.which("nvidia-smi")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The pynvml package is deprecated.*",
                    category=FutureWarning,
                )
                import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:  # noqa: BLE001
            self.nvml = None
            self.handle = None
        if self.nvml is None and self.smi is None:
            raise RuntimeError("pynvml and nvidia-smi are both unavailable")

    @property
    def is_nvml(self) -> bool:
        return self.nvml is not None

    def _process_rows(self) -> list[list[Any]]:
        groups: list[list[Any]] = []
        assert self.nvml is not None
        for name in (
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ):
            fn = getattr(self.nvml, name, None)
            if not callable(fn):
                continue
            try:
                groups.append(list(fn(self.handle)))
            except Exception:  # noqa: BLE001 - unsupported on some WDDM drivers
                pass
        return groups

    def read(self) -> GpuSample:
        if self.nvml is not None:
            mem = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            utilization = None
            try:
                utilization = int(self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
            except Exception:  # noqa: BLE001
                pass
            processes = merge_process_rows(self._process_rows(), int(mem.total))
            return GpuSample(int(mem.used), int(mem.total), utilization, processes)

        assert self.smi is not None
        out = subprocess.check_output(
            [
                self.smi,
                f"--id={self.index}",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip().splitlines()[0]
        used_mib, total_mib, util = (part.strip() for part in out.split(","))
        return GpuSample(
            int(float(used_mib) * MIB),
            int(float(total_mib) * MIB),
            int(float(util)),
            {},
        )

    def close(self) -> None:
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass


def target_pids(root_pid: int | None, include_children: bool) -> set[int]:
    if root_pid is None:
        return set()
    result = {root_pid}
    if not include_children or psutil is None:
        return result
    try:
        result.update(child.pid for child in psutil.Process(root_pid).children(recursive=True))
    except Exception:  # noqa: BLE001 - process may exit between samples
        pass
    return result


def classify_stage(line: str) -> str | None:
    """Extract coarse model-lifecycle markers from a Studio task log line."""
    lowered = line.lower()
    checks = (
        (("prompt 预编码失败",), "precache_failed"),
        (("预编码", "te 已释放"), "te_released"),
        (("loading text encoders",), "load_te"),
        (("加载 krea2 qwen",), "load_te"),
        (("loading transformer",), "load_dit"),
        (("loading vae",), "load_vae"),
        (("采样完成", "加载 vae"), "load_vae_after_sample"),
        (("vae 加载完成",), "vae_ready"),
        (("fp8 merge",), "lora_merge"),
        (("显存编排", "dit 让位"), "dit_yield_cpu"),
        (("unloading model",), "unload_all"),
        (("generate failed",), "failed"),
    )
    for needles, stage in checks:
        if all(needle in lowered for needle in needles):
            return stage
    return None


class TaskLogFollower:
    def __init__(self, setting: str, repo_root: Path):
        self.setting = setting
        self.repo_root = repo_root
        self.path: Path | None = None
        self.offset = 0
        self.stage = "waiting"
        self.last_scan = 0.0

    def _candidate(self) -> Path | None:
        if self.setting == "off":
            return None
        if self.setting != "auto":
            path = Path(self.setting).expanduser()
            return path if path.exists() else None
        paths = list((self.repo_root / "studio_data" / "tasks").glob("*/run.log"))
        return max(paths, key=lambda path: path.stat().st_mtime_ns) if paths else None

    def poll(self, now: float) -> list[tuple[str, str]]:
        if self.setting == "off":
            return []
        events: list[tuple[str, str]] = []
        if now - self.last_scan >= 0.5:
            self.last_scan = now
            candidate = self._candidate()
            if candidate is not None and candidate != self.path:
                self.path = candidate
                # Existing old task: follow future appends.  A newly created or
                # still-empty task is read from byte zero on the next poll.
                age = time.time() - candidate.stat().st_mtime
                self.offset = candidate.stat().st_size if age > 2.0 else 0
                self.stage = "waiting"
                events.append(("task_log", str(candidate)))
        if self.path is None:
            return events
        try:
            size = self.path.stat().st_size
            if size < self.offset:  # truncated/replaced
                self.offset = 0
            if size == self.offset:
                return events
            with self.path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self.offset)
                for line in handle:
                    stage = classify_stage(line)
                    if stage is not None:
                        self.stage = stage
                        events.append((stage, line.strip()))
                self.offset = handle.tell()
        except OSError:
            pass
        return events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="High-frequency Studio VRAM tracer")
    parser.add_argument("--gpu-index", type=int, default=0, help="NVML GPU index (default: 0)")
    parser.add_argument("--pid", type=int, help="optional Studio/daemon PID to aggregate")
    parser.add_argument(
        "--no-children", action="store_true", help="do not include descendants of --pid",
    )
    parser.add_argument(
        "--interval-ms", type=_positive_float, default=50.0,
        help="sampling interval in milliseconds (default: 50)",
    )
    parser.add_argument(
        "--duration", type=_positive_float, help="stop automatically after N seconds",
    )
    parser.add_argument("--out", default="tmp/vram_trace.csv", help="output CSV path")
    parser.add_argument(
        "--task-log", default="auto", metavar="PATH|auto|off",
        help="follow task log for lifecycle markers (default: auto)",
    )
    parser.add_argument(
        "--peak-step-mib", type=_positive_float, default=128.0,
        help="print a new peak after it grows by this much (default: 128)",
    )
    return parser


def _fmt_mib(value: int | None) -> str:
    return "" if value is None else f"{value / MIB:.1f}"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gpu = GpuReader(args.gpu_index)
    except Exception as exc:  # noqa: BLE001
        print(f"[vram_trace] 无法读取 GPU：{exc}", file=sys.stderr)
        return 2
    interval = args.interval_ms / 1000.0
    if not gpu.is_nvml and interval < 0.5:
        interval = 0.5
        print("[vram_trace] pynvml 不可用，nvidia-smi 回退采样间隔限制为 500ms")

    follower = TaskLogFollower(args.task_log, repo_root)
    fields = (
        "timestamp", "t_rel_s", "stage", "gpu_used_mib", "gpu_free_mib",
        "gpu_total_mib", "gpu_delta_mib", "gpu_util_pct", "target_gpu_mib",
        "target_pids", "gpu_processes", "event",
    )
    started_wall = datetime.now().astimezone()
    started = time.perf_counter()
    baseline: int | None = None
    peak_used = -1
    peak_time = 0.0
    peak_stage = "waiting"
    last_printed_peak = -1
    samples = 0

    print(
        f"[vram_trace] GPU#{args.gpu_index} interval={interval * 1000:.0f}ms "
        f"pid={args.pid or 'device-wide'} -> {out_path}"
    )
    print("[vram_trace] 先启动本脚本，再在 Studio 点开始生成；Ctrl+C 结束。")

    try:
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            next_tick = started
            while True:
                now = time.perf_counter()
                elapsed = now - started
                if args.duration is not None and elapsed >= args.duration:
                    break
                if now < next_tick:
                    time.sleep(next_tick - now)
                    now = time.perf_counter()
                    elapsed = now - started
                next_tick = max(next_tick + interval, now)

                events = follower.poll(now)
                for stage, detail in events:
                    print(f"[stage] {elapsed:8.3f}s {stage}: {detail}")
                event_text = " | ".join(f"{stage}: {detail}" for stage, detail in events)

                try:
                    sample = gpu.read()
                except Exception as exc:  # noqa: BLE001
                    print(f"[vram_trace] GPU 读取失败：{exc}", file=sys.stderr)
                    time.sleep(max(interval, 0.5))
                    continue
                if baseline is None:
                    baseline = sample.used
                pids = target_pids(args.pid, not args.no_children)
                target_values = [sample.processes.get(pid) for pid in pids]
                target_known = [value for value in target_values if value is not None]
                target_used = sum(target_known) if target_known else None
                # WDDM commonly returns dozens of graphics PIDs with an unknown
                # byte count.  Omitting unrelated NA rows keeps a 50ms trace
                # compact without losing any measurable residency.
                process_text = ";".join(
                    f"{pid}:{_fmt_mib(used) or 'NA'}"
                    for pid, used in sorted(sample.processes.items())
                    if used is not None or pid in pids
                )

                if sample.used > peak_used:
                    peak_used = sample.used
                    peak_time = elapsed
                    peak_stage = follower.stage
                    if (
                        last_printed_peak < 0
                        or sample.used - last_printed_peak >= args.peak_step_mib * MIB
                    ):
                        print(
                            f"[peak]  {elapsed:8.3f}s {sample.used / MIB:9.1f} MiB "
                            f"(delta {(sample.used - baseline) / MIB:+.1f}) "
                            f"stage={follower.stage}"
                        )
                        last_printed_peak = sample.used

                writer.writerow({
                    "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                    "t_rel_s": f"{elapsed:.6f}",
                    "stage": follower.stage,
                    "gpu_used_mib": _fmt_mib(sample.used),
                    "gpu_free_mib": _fmt_mib(sample.total - sample.used),
                    "gpu_total_mib": _fmt_mib(sample.total),
                    "gpu_delta_mib": _fmt_mib(sample.used - baseline),
                    "gpu_util_pct": "" if sample.utilization is None else sample.utilization,
                    "target_gpu_mib": _fmt_mib(target_used),
                    "target_pids": ";".join(str(pid) for pid in sorted(pids)),
                    "gpu_processes": process_text,
                    "event": event_text,
                })
                handle.flush()
                samples += 1
    except KeyboardInterrupt:
        print("\n[vram_trace] 手动停止。")
    finally:
        gpu.close()

    if baseline is None or peak_used < 0:
        print("[vram_trace] 没有采到有效数据。")
        return 1
    print(
        "[vram_trace] SUMMARY\n"
        f"  开始时间 : {started_wall.isoformat(timespec='seconds')}\n"
        f"  采样数   : {samples}\n"
        f"  基线     : {baseline / MIB:.1f} MiB\n"
        f"  峰值     : {peak_used / MIB:.1f} MiB\n"
        f"  净增峰值 : {(peak_used - baseline) / MIB:+.1f} MiB\n"
        f"  峰值时刻 : {peak_time:.3f}s ({peak_stage})\n"
        f"  CSV      : {out_path.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
