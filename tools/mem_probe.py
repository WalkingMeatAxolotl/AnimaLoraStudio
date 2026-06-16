#!/usr/bin/env python3
"""mem_probe —— 内存/显存峰值定位探针（Windows 优先）。

目的：定位「训练 sample / 测试 generate 偶发卡死 + 系统提交虚拟内存暴涨 20-30G」。
关键判据是：那一坨内存到底在 **GPU 侧**（被 WDDM 镜像成 system commit）还是
**CPU/host 侧**。本探针每隔 --interval 秒同时采样：

  - 系统提交（Commit Charge）：GetPerformanceInfo().CommitTotal —— 即任务管理器
    「性能 > 内存 > 已提交」那条，也就是用户看到暴涨的那个数。
  - 目标进程私有提交（PrivateUsage ≈ 任务管理器进程「提交大小」）。
  - GPU 显存：NVML（无则回退 nvidia-smi），device 级 used/total + 尽量给 per-pid。

每帧算 Δ（与上一帧之差）。当 |Δcommit| 或 |Δgpu_used| 超过 --spike-gb 时，打印
醒目告警 + 一句**判据**：

  - Δgpu ≈ Δcommit  → GPU 分配被 WDDM 镜像到 commit（显存侧爆，多半是某次卷积
    workspace / 大 bucket 分辨率；8G 卡上会直接 OOM 走分块所以没事，大显存卡反而
    吃满 commit 卡死）。
  - Δcommit ≫ Δgpu  → CPU/host 侧分配（pinned 内存 / 巨大 CPU 张量 / 泄漏）。

可选 --pyspy：抓到 spike 时自动 `py-spy dump --pid <pid>`，零侵入拿到当时正在
执行的 Python 调用栈（需 `pip install py-spy`）。

全程写 CSV（--out），事后可画图/比对。

用法
----
1) 先把训练/出图跑起来，拿到那个 python 进程的 PID（任务管理器 / `tasklist`）。
2) 另开一个终端：
     python tools/mem_probe.py --pid <PID> --interval 0.25 --spike-gb 4 --pyspy
   或不知道 PID 时按名字挑（选私有提交最大的 python）：
     python tools/mem_probe.py --name python --interval 0.25 --spike-gb 4
3) 复现卡死。看终端的 [SPIKE] 行 + 末尾 SUMMARY，或事后读 mem_probe.csv。

依赖：psutil（强烈建议）。NVML(pynvml) 可选；没有就用 nvidia-smi。系统提交用
ctypes 直接读，无需第三方。
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import shutil
import subprocess
import sys
import time
from ctypes import wintypes

GB = 1024.0 ** 3

try:
    import psutil  # type: ignore
except Exception:  # noqa: BLE001
    psutil = None


# ---------------------------------------------------------------- 系统提交（Commit Charge）
class _PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", wintypes.DWORD),
        ("ProcessCount", wintypes.DWORD),
        ("ThreadCount", wintypes.DWORD),
    ]


def system_commit_bytes() -> tuple[float, float, float]:
    """返回 (CommitTotal, CommitLimit, CommitPeak) 字节。仅 Windows。"""
    pi = _PERFORMANCE_INFORMATION()
    pi.cb = ctypes.sizeof(pi)
    ok = ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    if not ok:
        raise ctypes.WinError()
    ps = pi.PageSize
    return pi.CommitTotal * ps, pi.CommitLimit * ps, pi.CommitPeak * ps


# ---------------------------------------------------------------- 目标进程
def resolve_pid(pid: int | None, name: str | None) -> int:
    if pid:
        return pid
    if psutil is None:
        sys.exit("未装 psutil 且未给 --pid，无法按名字查找。请装 psutil 或传 --pid。")
    name_l = (name or "python").lower()
    best, best_priv = None, -1
    for p in psutil.process_iter(["name"]):
        try:
            if name_l in (p.info["name"] or "").lower():
                priv = p.memory_info().private
                if priv > best_priv:
                    best, best_priv = p.pid, priv
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if best is None:
        sys.exit(f"找不到名字含 '{name_l}' 的进程；用 --pid 指定。")
    print(f"[mem_probe] 按名字 '{name_l}' 选中 PID={best}（私有提交最大）")
    return best


def proc_mem_bytes(proc) -> tuple[float, float]:
    """(private/commit, working_set) 字节。"""
    mi = proc.memory_info()
    # psutil Windows: memory_info() 含 private(PrivateUsage) 与 rss(WorkingSet)
    private = getattr(mi, "private", getattr(mi, "pagefile", mi.vms))
    return float(private), float(mi.rss)


# ---------------------------------------------------------------- GPU
class _GpuReader:
    def __init__(self, index: int):
        self.index = index
        self.nvml = None
        self.handle = None
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:  # noqa: BLE001
            self.nvml = None
        self.smi = shutil.which("nvidia-smi")

    def read(self, target_pid: int) -> tuple[float, float, float]:
        """返回 (used, total, proc_used) 字节；proc_used 取不到时为 -1。"""
        if self.nvml is not None:
            try:
                m = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                proc_used = -1.0
                try:
                    for pr in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
                        if pr.pid == target_pid and pr.usedGpuMemory not in (None, 0):
                            proc_used = float(pr.usedGpuMemory)
                except Exception:  # noqa: BLE001
                    pass  # WDDM 下 per-pid 常不可用
                return float(m.used), float(m.total), proc_used
            except Exception:  # noqa: BLE001
                pass
        if self.smi:
            try:
                out = subprocess.check_output(
                    [self.smi, f"--id={self.index}",
                     "--query-gpu=memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
                ).strip().splitlines()[0]
                used_mb, total_mb = (float(x) for x in out.split(","))
                return used_mb * 1024 * 1024, total_mb * 1024 * 1024, -1.0
            except Exception:  # noqa: BLE001
                pass
        return -1.0, -1.0, -1.0


# ---------------------------------------------------------------- 主循环
def main() -> None:
    # 控制台可能是 cp932/gbk 等非 utf-8（本机即 cp932）；中文 print 会 UnicodeEncodeError
    # 把探针自己搞崩。统一改 utf-8 + errors=replace。
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")  # py3.7+
        except Exception:  # noqa: BLE001
            pass

    if os.name != "nt":
        print("[mem_probe] 注意：系统提交读数走 Windows API；非 Windows 上仅 GPU/进程可用。")

    ap = argparse.ArgumentParser(description="内存/显存峰值定位探针")
    ap.add_argument("--pid", type=int, default=None, help="目标进程 PID")
    ap.add_argument("--name", type=str, default="python", help="按名字找（--pid 优先）")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.25, help="采样间隔秒")
    ap.add_argument("--spike-gb", type=float, default=4.0, help="Δcommit/Δgpu 超过此值(GB)即告警")
    ap.add_argument("--out", type=str, default="mem_probe.csv")
    ap.add_argument("--pyspy", action="store_true", help="spike 时自动 py-spy dump 目标进程栈")
    args = ap.parse_args()

    pid = resolve_pid(args.pid, args.name)
    if psutil is None:
        sys.exit("本脚本需要 psutil 取进程内存：pip install psutil")
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        sys.exit(f"PID {pid} 不存在。")
    gpu = _GpuReader(args.gpu_index)
    pyspy = shutil.which("py-spy") if args.pyspy else None
    if args.pyspy and not pyspy:
        print("[mem_probe] 未找到 py-spy，--pyspy 忽略（pip install py-spy）")

    fields = ["t_rel_s", "sys_commit_GB", "sys_commit_limit_GB", "sys_commit_pct",
              "proc_private_GB", "proc_wset_GB", "gpu_used_GB", "gpu_total_GB",
              "gpu_proc_GB", "d_commit_GB", "d_gpu_GB", "note"]
    f = open(args.out, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(fields)

    print(f"[mem_probe] PID={pid} GPU#{args.gpu_index} interval={args.interval}s "
          f"spike>{args.spike_gb}G → {args.out}  (Ctrl+C 结束)")

    t0 = time.perf_counter()
    prev_commit = prev_gpu = None
    peak = {"commit": 0.0, "private": 0.0, "gpu": 0.0}
    spikes: list[str] = []

    try:
        while True:
            if not proc.is_running():
                print("[mem_probe] 目标进程已退出，停止。")
                break
            t = time.perf_counter() - t0
            try:
                commit, climit, cpeak = system_commit_bytes()
            except Exception as e:  # noqa: BLE001
                commit = climit = cpeak = -1.0
                _ = e
            try:
                priv, wset = proc_mem_bytes(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            gused, gtotal, gproc = gpu.read(pid)

            d_commit = (commit - prev_commit) / GB if prev_commit is not None and commit >= 0 else 0.0
            d_gpu = (gused - prev_gpu) / GB if prev_gpu is not None and gused >= 0 else 0.0
            prev_commit = commit if commit >= 0 else prev_commit
            prev_gpu = gused if gused >= 0 else prev_gpu

            peak["commit"] = max(peak["commit"], commit)
            peak["private"] = max(peak["private"], priv)
            peak["gpu"] = max(peak["gpu"], gused)

            note = ""
            if abs(d_commit) >= args.spike_gb or abs(d_gpu) >= args.spike_gb:
                # 判据
                if gused >= 0 and abs(d_gpu) >= args.spike_gb and abs(d_commit - d_gpu) <= max(2.0, 0.3 * abs(d_gpu)):
                    verdict = "GPU侧(WDDM镜像到commit)"
                elif abs(d_commit) >= args.spike_gb and (gused < 0 or abs(d_commit) - abs(d_gpu) >= args.spike_gb):
                    verdict = "CPU/host侧"
                else:
                    verdict = "混合/待查"
                note = f"SPIKE {verdict} dCommit={d_commit:+.1f}G dGPU={d_gpu:+.1f}G"
                line = (f"[SPIKE] t={t:7.2f}s  {verdict}  "
                        f"Δcommit={d_commit:+.1f}G  Δgpu={d_gpu:+.1f}G  "
                        f"commit={commit/GB:.1f}G gpu_used={gused/GB if gused>=0 else -1:.1f}G "
                        f"proc_priv={priv/GB:.1f}G")
                print("\n" + "!" * 78 + "\n" + line + "\n" + "!" * 78)
                spikes.append(line)
                if pyspy:
                    try:
                        dump = subprocess.check_output(
                            [pyspy, "dump", "--pid", str(pid), "--nonblocking"],
                            text=True, timeout=15, stderr=subprocess.STDOUT)
                        with open("mem_probe_pyspy.log", "a", encoding="utf-8") as pf:
                            pf.write(f"\n===== SPIKE t={t:.2f}s {verdict} =====\n{dump}\n")
                        print("[mem_probe] py-spy 栈已追加到 mem_probe_pyspy.log")
                    except Exception as e:  # noqa: BLE001
                        print(f"[mem_probe] py-spy dump 失败: {e}")

            w.writerow([f"{t:.3f}",
                        f"{commit/GB:.3f}" if commit >= 0 else "",
                        f"{climit/GB:.3f}" if climit >= 0 else "",
                        f"{100*commit/climit:.1f}" if climit > 0 else "",
                        f"{priv/GB:.3f}", f"{wset/GB:.3f}",
                        f"{gused/GB:.3f}" if gused >= 0 else "",
                        f"{gtotal/GB:.3f}" if gtotal >= 0 else "",
                        f"{gproc/GB:.3f}" if gproc >= 0 else "",
                        f"{d_commit:+.3f}", f"{d_gpu:+.3f}", note])
            f.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[mem_probe] 手动停止。")
    finally:
        f.close()
        print("\n" + "=" * 60 + "\nSUMMARY")
        print(f"  峰值 系统提交 commit : {peak['commit']/GB:.1f} G")
        print(f"  峰值 进程私有提交     : {peak['private']/GB:.1f} G")
        print(f"  峰值 GPU used        : {peak['gpu']/GB:.1f} G" if peak['gpu'] >= 0 else "  GPU 不可读")
        print(f"  CSV                  : {os.path.abspath(args.out)}")
        if spikes:
            print(f"  捕获 {len(spikes)} 次 spike：")
            for s in spikes:
                print("    " + s)
        else:
            print("  未捕获 spike（没复现，或 --spike-gb 设太高）。")
        print("=" * 60)


if __name__ == "__main__":
    main()
