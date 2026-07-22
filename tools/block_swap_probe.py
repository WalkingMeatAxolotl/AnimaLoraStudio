#!/usr/bin/env python3
"""block_swap_probe —— block swap 可行性 Gate-0 探针。

配套文档：``docs/design/block-swap.md``。**实现代码之前必须先跑本探针**，
数据不达门槛则方案直接否决。

block swap 的成败只看一个不等式（doc §2.2）：

    可完全遮蔽  ⟺  T_transfer(block) < T_compute(block)
    T_transfer = 单 block 字节数 / 有效 PCIe 带宽

本探针分六段回答它，外加用户明确关注的硬件影响（doc §3）：

  A 链路体检   PCIe 实际 gen/width vs 最大值、replay 基线、GPU/RAM 容量
  B 带宽矩阵   pinned/pageable × H2D/D2H × 多种 size；pinned 分配耗时
  C 计算基准   真实 SingleStreamBlock 在真实 shape 下的前向/反向耗时
  D 遮蔽判据   B/C 比值 → blocks_to_swap × (省显存, 加时间) 曲线
  E 端到端     真双 stream swap 循环 vs 全常驻循环的 wall clock 差
  F 稳定性     持续负载下温度 / 功耗 / PCIe replay 增量 / 可用 RAM

用法
----
    ./venv/Scripts/python.exe tools/block_swap_probe.py
    ./venv/Scripts/python.exe tools/block_swap_probe.py --stages ABCD  # 跳过耗时段
    ./venv/Scripts/python.exe tools/block_swap_probe.py --resolution 1536 --soak-seconds 120
    ./venv/Scripts/python.exe tools/block_swap_probe.py --out probe.csv

无 CUDA 时只跑 A 段并退出。依赖 torch；pynvml 可选（缺失则 A/F 段降级）。

范围：C/D/E/F 段只覆盖 **krea2**（28 层同构 SingleStreamBlock，是方案的目标
族）。Anima 的 Block forward 需要 rope/adaln_lora 等一串预备张量，构造成本高
而收益低（24GB 已够用），本探针只统计其权重规模，计算基准留待需要时再补。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

# 本机终端是 cp932，中文输出不重定向会崩（见 windows_console_cp932_utf8）
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "runtime"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

MIB = 1024 ** 2
GIB = 1024 ** 3

#: 采集到的所有观测点，末尾可选写 CSV
RECORDS: list[dict] = []


def record(stage: str, metric: str, value, unit: str = "", note: str = "") -> None:
    RECORDS.append(
        {"stage": stage, "metric": metric, "value": value, "unit": unit, "note": note}
    )


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ------------------------------------------------------------------ NVML 封装
class Nvml:
    """pynvml 薄封装：缺库 / 缺 API 时整体降级为 None，不让探针挂掉。"""

    def __init__(self) -> None:
        self.handle = None
        self._nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001
            self._nvml = None
            self.handle = None

    @property
    def ok(self) -> bool:
        return self.handle is not None

    def _call(self, name: str, *args):
        if not self.ok:
            return None
        fn = getattr(self._nvml, name, None)
        if fn is None:
            return None
        try:
            return fn(self.handle, *args)
        except Exception:  # noqa: BLE001
            return None

    def pcie_gen(self) -> tuple[int | None, int | None]:
        return self._call("nvmlDeviceGetCurrPcieLinkGeneration"), self._call(
            "nvmlDeviceGetMaxPcieLinkGeneration"
        )

    def pcie_width(self) -> tuple[int | None, int | None]:
        return self._call("nvmlDeviceGetCurrPcieLinkWidth"), self._call(
            "nvmlDeviceGetMaxPcieLinkWidth"
        )

    def replay_counter(self) -> int | None:
        return self._call("nvmlDeviceGetPcieReplayCounter")

    def temperature(self) -> int | None:
        if not self.ok:
            return None
        try:
            return self._nvml.nvmlDeviceGetTemperature(
                self.handle, self._nvml.NVML_TEMPERATURE_GPU
            )
        except Exception:  # noqa: BLE001
            return None

    def power_watts(self) -> float | None:
        milliwatts = self._call("nvmlDeviceGetPowerUsage")
        return None if milliwatts is None else milliwatts / 1000.0

    def name(self) -> str | None:
        raw = self._call("nvmlDeviceGetName")
        if isinstance(raw, bytes):
            return raw.decode("utf-8", "replace")
        return raw

    def shutdown(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass


# ------------------------------------------------------------------ A 链路体检
def stage_a(nvml: Nvml) -> dict:
    """PCIe 链路是否降级 + 容量基线 + replay 基线（doc §3.2 ②）。"""
    section("A · 链路体检")
    import torch

    info: dict = {}

    cuda_ok = torch.cuda.is_available()
    info["cuda"] = cuda_ok
    print(f"torch {torch.__version__}   CUDA 可用: {cuda_ok}")
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["vram_total"] = props.total_memory
        print(f"GPU        : {props.name}")
        print(f"显存       : {props.total_memory / GIB:.1f} GB")
        record("A", "vram_total", props.total_memory / GIB, "GB", props.name)

    from training.sysmem import available_ram_bytes

    avail = available_ram_bytes()
    if avail is not None:
        info["ram_available"] = avail
        print(f"可用内存   : {avail / GIB:.1f} GB")
        record("A", "ram_available", round(avail / GIB, 1), "GB")

    if nvml.ok:
        cur_gen, max_gen = nvml.pcie_gen()
        cur_width, max_width = nvml.pcie_width()
        info["pcie_gen"] = cur_gen
        info["pcie_width"] = cur_width
        degraded = []
        if cur_gen and max_gen and cur_gen < max_gen:
            degraded.append(f"gen {cur_gen} < 最大 {max_gen}")
        if cur_width and max_width and cur_width < max_width:
            degraded.append(f"width x{cur_width} < 最大 x{max_width}")
        print(f"PCIe 链路  : gen{cur_gen} x{cur_width}（最大 gen{max_gen} x{max_width}）")
        record("A", "pcie_gen", cur_gen, "", f"max={max_gen}")
        record("A", "pcie_width", cur_width, "", f"max={max_width}")
        if degraded:
            # 空闲时链路会自动降到 gen1 省电，负载下才拉满 —— 此处仅提示
            print(f"  ! 当前处于降级态（{'; '.join(degraded)}）")
            print("    空闲省电降速属正常，B 段有负载时会复测；若届时仍降级则是真降级")

        replay = nvml.replay_counter()
        info["replay_base"] = replay
        print(f"PCIe replay: {replay}（基线，F 段看增量）")
        record("A", "pcie_replay_base", replay)
    else:
        print("pynvml 不可用 —— PCIe 链路与 replay 计数无法采集（A/F 段降级）")

    return info


# ------------------------------------------------------------------ B 带宽矩阵
def _time_copy(src, dst_device: str, iters: int) -> float:
    """返回单次拷贝耗时中位数（秒）。用 cuda event 计时，排除 launch 抖动。"""
    import torch

    events = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        src.to(dst_device, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        events.append(start.elapsed_time(end) / 1000.0)
    return statistics.median(events)


def stage_b(args) -> dict:
    """pinned vs pageable × H2D vs D2H 的有效带宽（doc §2.1）。"""
    section("B · 传输带宽矩阵")
    import torch

    sizes_mib = [16, 64, 256, 1024]
    results: dict = {}

    print(f"{'size':>8} {'pinned H2D':>12} {'pageable H2D':>14} {'pinned D2H':>12}")
    print("-" * 50)
    for size_mib in sizes_mib:
        numel = size_mib * MIB // 2  # bf16

        pin_start = time.perf_counter()
        try:
            pinned = torch.empty(numel, dtype=torch.bfloat16, pin_memory=True)
        except RuntimeError as exc:
            # doc §3.2 ①：pinned 分配失败是硬错误，实现必须有降级路径
            print(f"{size_mib:>6}MB  pinned 分配失败：{exc}")
            record("B", f"pin_alloc_fail_{size_mib}MB", 1, "", str(exc))
            continue
        pin_alloc = time.perf_counter() - pin_start

        pageable = torch.empty(numel, dtype=torch.bfloat16)
        on_gpu = torch.empty(numel, dtype=torch.bfloat16, device="cuda")

        # warmup
        pinned.to("cuda", non_blocking=True)
        torch.cuda.synchronize()

        t_pin_h2d = _time_copy(pinned, "cuda", args.iters)
        t_page_h2d = _time_copy(pageable, "cuda", args.iters)
        t_pin_d2h = _time_copy(on_gpu, "cpu", args.iters)

        payload = size_mib * MIB
        bw_pin = payload / t_pin_h2d / GIB
        bw_page = payload / t_page_h2d / GIB
        bw_d2h = payload / t_pin_d2h / GIB
        results[size_mib] = {
            "pinned_h2d": bw_pin,
            "pageable_h2d": bw_page,
            "pinned_d2h": bw_d2h,
            "pin_alloc_s": pin_alloc,
        }
        print(
            f"{size_mib:>6}MB {bw_pin:>10.1f}GB/s {bw_page:>12.1f}GB/s {bw_d2h:>10.1f}GB/s"
        )
        record("B", f"pinned_h2d_{size_mib}MB", round(bw_pin, 2), "GB/s")
        record("B", f"pageable_h2d_{size_mib}MB", round(bw_page, 2), "GB/s")
        record("B", f"pin_alloc_{size_mib}MB", round(pin_alloc * 1000, 1), "ms")

        del pinned, pageable, on_gpu
        torch.cuda.empty_cache()

    if results:
        big = max(results)
        peak = results[big]["pinned_h2d"]
        print(f"\n有效带宽（取 {big}MB pinned H2D）: {peak:.1f} GB/s")
        print(
            f"pinned 相对 pageable 提速: "
            f"{results[big]['pinned_h2d'] / results[big]['pageable_h2d']:.2f}×"
        )
        print(
            f"pinned 分配耗时: {results[big]['pin_alloc_s'] * 1000:.0f} ms / {big}MB"
            "  ← 实现必须预分配复用，不能每步 alloc"
        )
        results["effective_bw"] = peak
    return results


# ------------------------------------------------------------------ C 计算基准
def _build_krea2_block(device, dtype):
    """实例化单个真实 SingleStreamBlock + 其前向所需的合成输入。

    输入构造逐行对齐 ``SingleStreamDiT.forward``（krea2_modeling.py:444-522）：
    combined = cat(text, image) 后过 block，vec 是 tproj 出来的 (B, 6F)。
    """
    import torch
    from modeling.krea2 import KREA2_CONFIG
    from modeling.krea2.krea2_modeling import PositionalEncoding, SingleStreamBlock

    cfg = KREA2_CONFIG
    block = SingleStreamBlock(
        cfg.features, cfg.heads, cfg.multiplier, cfg.bias, cfg.kvheads
    ).to(device=device, dtype=dtype)
    return block, cfg, PositionalEncoding


def _krea2_inputs(cfg, PositionalEncoding, resolution: int, batch: int, device, dtype):
    import torch

    latent = resolution // 8              # VAE f8（WAN21_F8C16）
    grid = latent // cfg.patch            # patch=2
    image_len = grid * grid
    text_len = 512                        # TextSpec.max_seq_len
    seq_len = text_len + image_len

    head_dim = cfg.features // cfg.heads
    axes = (
        head_dim - 12 * (head_dim // 16),
        6 * (head_dim // 16),
        6 * (head_dim // 16),
    )
    posemb = PositionalEncoding(axes, theta=cfg.theta)

    text_pos = torch.zeros(batch, text_len, 3, device=device, dtype=torch.float32)
    image_pos = torch.zeros(grid, grid, 3, device=device, dtype=torch.float32)
    image_pos[..., 1] = torch.arange(grid, device=device)[:, None]
    image_pos[..., 2] = torch.arange(grid, device=device)[None, :]
    image_pos = image_pos.reshape(1, image_len, 3).expand(batch, -1, -1)
    freqs = posemb(torch.cat((text_pos, image_pos), dim=1))

    x = torch.randn(batch, seq_len, cfg.features, device=device, dtype=dtype)
    vec = torch.randn(batch, 6 * cfg.features, device=device, dtype=dtype)
    return x, vec, freqs, seq_len


def stage_c(args) -> dict:
    """真实 block 的前向 / 前向+反向耗时（T_compute）。"""
    section("C · 单 block 计算基准（krea2）")
    import torch

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    block, cfg, PositionalEncoding = _build_krea2_block(device, dtype)
    x, vec, freqs, seq_len = _krea2_inputs(
        cfg, PositionalEncoding, args.resolution, args.batch, device, dtype
    )

    param_bytes = sum(p.numel() * p.element_size() for p in block.parameters())
    param_count = sum(p.numel() for p in block.parameters())
    total_bytes = param_bytes * cfg.layers

    print(f"分辨率     : {args.resolution}² → seq_len {seq_len}（text 512 + image {seq_len - 512}）")
    print(f"单 block   : {param_count / 1e6:.1f}M 参数, {param_bytes / MIB:.0f} MB @ {args.dtype}")
    print(f"{cfg.layers} 层合计  : {total_bytes / GIB:.2f} GB（不含 txtfusion / embed / last）")
    record("C", "block_params", round(param_count / 1e6, 1), "M")
    record("C", "block_bytes", round(param_bytes / MIB), "MB", args.dtype)
    record("C", "dit_blocks_bytes", round(total_bytes / GIB, 2), "GB")

    # 默认参数早绑定（krea2_modeling.py:517 同款）：末尾 del 这几个名字释放显存，
    # 闭包捕获会让它们变成悬空引用
    def once(backward: bool, blk=block, inp=x, mod=vec, rope=freqs):
        if backward:
            out = blk(inp, mod, rope)
            out.sum().backward()
            blk.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                blk(inp, mod, rope)

    for backward in (False, True):
        if backward:
            for p in block.parameters():
                p.requires_grad_(True)
            x.requires_grad_(True)
        for _ in range(3):  # warmup
            once(backward)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.iters):
            start = time.perf_counter()
            once(backward)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        median = statistics.median(samples)
        label = "前向+反向" if backward else "前向"
        print(f"{label:>9} : {median * 1000:.2f} ms")
        record("C", "fwd_bwd_ms" if backward else "fwd_ms", round(median * 1000, 2), "ms")
        if backward:
            t_bwd = median
        else:
            t_fwd = median

    del block, x, vec, freqs
    torch.cuda.empty_cache()
    return {
        "param_bytes": param_bytes,
        "total_bytes": total_bytes,
        "layers": cfg.layers,
        "t_fwd": t_fwd,
        "t_fwd_bwd": t_bwd,
        "seq_len": seq_len,
    }


# ------------------------------------------------------------------ D 遮蔽判据
def stage_d(bandwidth: dict, compute: dict) -> None:
    """把 B 和 C 的数字代进 doc §2.2 的不等式，给出 blocks_to_swap 曲线。"""
    section("D · 遮蔽判据")

    bw = bandwidth.get("effective_bw")
    if not bw:
        print("B 段无有效带宽，跳过")
        return

    param_bytes = compute["param_bytes"]
    t_transfer = param_bytes / (bw * GIB)
    t_fwd = compute["t_fwd"]
    t_fwd_bwd = compute["t_fwd_bwd"]

    print(f"T_transfer (单 block) : {t_transfer * 1000:.2f} ms  "
          f"（{param_bytes / MIB:.0f} MB ÷ {bw:.1f} GB/s）")
    print(f"T_compute  前向       : {t_fwd * 1000:.2f} ms")
    print(f"T_compute  前向+反向  : {t_fwd_bwd * 1000:.2f} ms")
    record("D", "t_transfer_ms", round(t_transfer * 1000, 2), "ms")

    ratio_fwd = t_transfer / t_fwd
    print(f"\n传输/计算比（前向，推理口径）  : {ratio_fwd:.2f}")
    print(f"传输/计算比（前反向，训练口径）: {t_transfer / t_fwd_bwd:.2f}")
    record("D", "ratio_fwd", round(ratio_fwd, 3))
    record("D", "ratio_train", round(t_transfer / t_fwd_bwd, 3))

    if ratio_fwd < 1:
        print("→ 传输快于计算：**理论可完全遮蔽**（前向与训练均是）")
    else:
        print(f"→ 传输慢于计算：每 block 暴露 {(t_transfer - t_fwd) * 1000:.1f} ms（前向口径）")

    # LoRA 训练：底模冻结，只有 H2D，前向 1 次 + 反向重算 1 次（doc §2.3/2.4）
    layers = compute["layers"]
    per_block = param_bytes / GIB
    # 训练一步里每个 block 有两个驻留窗口：前向一次，反向（含 checkpoint
    # 重算）一次。两段计算时间不同，必须分别与传输时间比（doc §2.4）。
    t_bwd_only = max(t_fwd_bwd - t_fwd, 0.0)
    exposed_fwd = max(0.0, t_transfer - t_fwd)
    exposed_bwd = max(0.0, t_transfer - t_bwd_only)
    print(f"\nblocks_to_swap 曲线（LoRA 训练口径，单步 2 遍 H2D）")
    print(f"  前向段暴露 {exposed_fwd * 1000:.1f} ms/block、"
          f"反向段暴露 {exposed_bwd * 1000:.1f} ms/block")
    print(f"{'N':>4} {'省显存':>10} {'暴露/步':>12} {'相对基线':>10}")
    print("-" * 40)
    base_step = t_fwd_bwd * layers
    for n in (0, 4, 8, 14, 20, 28):
        if n > layers:
            continue
        exposed = (exposed_fwd + exposed_bwd) * n
        saved = per_block * n
        overhead = exposed / base_step * 100 if base_step else 0
        print(f"{n:>4} {saved:>9.2f}GB {exposed * 1000:>10.1f}ms {overhead:>9.1f}%")
        record("D", f"swap_{n}_saved_gb", round(saved, 2), "GB")
        record("D", f"swap_{n}_overhead_pct", round(overhead, 1), "%")

    print("\n注：这是理论上界（假设预取零开销、stream 完美重叠）。E 段测真实值。")


# ------------------------------------------------------------------ E 端到端
def stage_e(args, compute: dict) -> dict:
    """真双 stream swap 循环 vs 全常驻循环的 wall clock 差。"""
    section("E · 端到端 swap 循环 vs 全常驻")
    import torch
    from modeling.krea2 import KREA2_CONFIG
    from modeling.krea2.krea2_modeling import PositionalEncoding, SingleStreamBlock

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    cfg = KREA2_CONFIG
    n = args.swap_blocks

    x0, vec, freqs, _ = _krea2_inputs(
        cfg, PositionalEncoding, args.resolution, args.batch, device, dtype
    )

    def make_block(dev):
        return SingleStreamBlock(
            cfg.features, cfg.heads, cfg.multiplier, cfg.bias, cfg.kvheads
        ).to(device=dev, dtype=dtype)

    need = compute["param_bytes"] * n / GIB
    free = torch.cuda.mem_get_info()[0] / GIB
    print(f"对照组需常驻 {n} block ≈ {need:.1f} GB，当前空闲 {free:.1f} GB")
    if need + 3 > free:
        print("显存不足以建立全常驻对照组，请减小 --swap-blocks")
        return {}

    # ---- 基线：N 个 block 全部常驻 GPU
    resident = [make_block(device) for _ in range(n)]
    for _ in range(2):
        h = x0
        for blk in resident:
            with torch.no_grad():
                h = blk(h, vec, freqs)
    torch.cuda.synchronize()

    samples = []
    for _ in range(args.iters):
        start = time.perf_counter()
        h = x0
        for blk in resident:
            with torch.no_grad():
                h = blk(h, vec, freqs)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    t_resident = statistics.median(samples)
    print(f"全常驻     : {t_resident * 1000:.1f} ms / {n} block")
    record("E", "resident_ms", round(t_resident * 1000, 1), "ms", f"{n} blocks")

    # ---- swap：权重在 CPU pinned，2 个 GPU buffer 轮转 + 独立 copy stream 预取
    #
    # 两种传输粒度都测，因为差别是本探针最关键的实现指导：
    #   per_tensor —— 逐 param `copy_`（朴素实现），单 block 十余次小传输
    #   flat       —— 权重重绑到一段连续 buffer，单 block 一次大传输（musubi 做法）
    cpu_weights = [
        {k: v.detach().to("cpu").pin_memory() for k, v in blk.state_dict().items()}
        for blk in resident
    ]
    del resident
    torch.cuda.empty_cache()

    def rebind_flat(block):
        """把 block 的所有 parameter 重绑到一段连续 GPU buffer 的 view 上。"""
        entries = list(block.named_parameters())
        total = sum(p.numel() for _, p in entries)
        flat = torch.empty(total, device=device, dtype=dtype)
        offset = 0
        for _, param in entries:
            count = param.numel()
            view = flat[offset:offset + count].view(param.shape)
            view.copy_(param.detach())
            param.data = view
            offset += count
        return flat, [name for name, _ in entries]

    def build_swap(flat_mode: bool):
        buffers = [make_block(device), make_block(device)]
        copy_stream = torch.cuda.Stream()
        ready = [torch.cuda.Event() for _ in range(2)]
        done = [torch.cuda.Event() for _ in range(2)]

        if flat_mode:
            flats, order = zip(*(rebind_flat(b) for b in buffers))
            order = order[0]
            cpu_flat = []
            for weights in cpu_weights:
                parts = [weights[name].reshape(-1) for name in order]
                cpu_flat.append(torch.cat(parts).pin_memory())

            def prefetch(idx: int, slot: int) -> None:
                with torch.cuda.stream(copy_stream):
                    flats[slot].copy_(cpu_flat[idx], non_blocking=True)
                    ready[slot].record(copy_stream)
        else:
            buf_params = [dict(b.state_dict()) for b in buffers]

            def prefetch(idx: int, slot: int) -> None:
                with torch.cuda.stream(copy_stream):
                    for key, dst in buf_params[slot].items():
                        dst.copy_(cpu_weights[idx][key], non_blocking=True)
                    ready[slot].record(copy_stream)

        def swap_pass():
            h = x0
            prefetch(0, 0)
            for i in range(n):
                slot = i % 2
                if i + 1 < n:
                    # 该 slot 上一轮的计算必须先完成，否则预取会覆盖正在被读的
                    # 权重（朴素实现最容易漏的数据竞争，且漏了时间还偏乐观）
                    nxt = (i + 1) % 2
                    if i >= 1:
                        copy_stream.wait_event(done[nxt])
                    prefetch(i + 1, nxt)
                torch.cuda.current_stream().wait_event(ready[slot])
                with torch.no_grad():
                    h = buffers[slot](h, vec, freqs)
                done[slot].record(torch.cuda.current_stream())
            return h

        return swap_pass, buffers

    results: dict = {}
    for flat_mode, label in ((False, "per_tensor"), (True, "flat")):
        swap_pass, buffers = build_swap(flat_mode)
        for _ in range(2):
            swap_pass()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.iters):
            start = time.perf_counter()
            swap_pass()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        t_swap = statistics.median(samples)
        overhead = (t_swap - t_resident) / t_resident * 100
        print(f"swap({label:>10}) : {t_swap * 1000:.1f} ms / {n} block"
              f"   开销 {overhead:+.1f}%")
        record("E", f"swap_{label}_ms", round(t_swap * 1000, 1), "ms", f"{n} blocks")
        record("E", f"overhead_{label}_pct", round(overhead, 1), "%")
        results[label] = {"t_swap": t_swap, "overhead": overhead, "swap_pass": swap_pass}
        if flat_mode:
            best = results
        else:
            del buffers
            torch.cuda.empty_cache()

    saved = compute["param_bytes"] * n / GIB
    overhead = min(r["overhead"] for r in results.values())
    print(f"\n最优实测开销 : {overhead:+.1f}%（前向口径）")
    print(f"换来省显存   : {saved:.2f} GB（{n} block 不常驻）")
    record("E", "saved_gb", round(saved, 2), "GB")

    if overhead < 15:
        print("→ 达到 doc §5 建议门槛（<15%）")
    elif overhead > 30:
        print("→ 超过 doc §5 否决线（>30%）")
    else:
        print("→ 落在 15%~30% 灰区，需用户裁定")
    print("注：前向口径是**最坏情形** —— 训练的反向段计算时间远长于传输，更易遮蔽")

    winner = min(results.values(), key=lambda r: r["overhead"])
    return {
        "swap_pass": winner["swap_pass"],
        "t_swap": winner["t_swap"],
        "t_resident": t_resident,
        "overhead": overhead,
        "modes": {k: v["overhead"] for k, v in results.items()},
    }


# ------------------------------------------------------------------ G 训练口径
def stage_g(args, compute: dict) -> dict:
    """训练口径端到端：gradient checkpointing + 反向逆序预取（B10）。

    E 段只测前向，训练口径此前是由 D 段公式外推的（doc §5.3-1）。本段实测
    完整一步的时序，模拟的是 **LoRA + gradient checkpointing** 这个本仓唯一
    的真实场景：

      前向  底模 frozen、不保存中间激活（checkpoint 语义），逐 block 换入
      反向  逆序 N-1→0，每个 block 换入后「重算前向 + 反向」在同一驻留窗口
            内完成（doc §2.4 说的合流），只求 grad_input 不求权重梯度

    数值正确性不在本段范围内（buffer 权重被轮转覆盖，梯度无意义）——只测时序。
    LoRA 参数本身常驻 GPU 不参与 swap，其计算量相对底模可忽略。
    """
    section("G · 训练口径端到端（checkpoint + 反向逆序预取）")
    import torch
    from modeling.krea2 import KREA2_CONFIG
    from modeling.krea2.krea2_modeling import PositionalEncoding, SingleStreamBlock

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    cfg = KREA2_CONFIG
    n = args.swap_blocks

    x0, vec, freqs, _ = _krea2_inputs(
        cfg, PositionalEncoding, args.resolution, args.batch, device, dtype
    )

    def make_block(dev):
        blk = SingleStreamBlock(
            cfg.features, cfg.heads, cfg.multiplier, cfg.bias, cfg.kvheads
        ).to(device=dev, dtype=dtype)
        blk.requires_grad_(False)  # 底模 frozen（LoRA 场景）
        return blk

    need = compute["param_bytes"] * n / GIB
    free = torch.cuda.mem_get_info()[0] / GIB
    print(f"对照组需常驻 {n} block ≈ {need:.1f} GB，当前空闲 {free:.1f} GB")
    if need + 4 > free:
        print("显存不足以建立全常驻对照组，请减小 --swap-blocks")
        return {}

    def step(get_block, after=None):
        """一步 checkpoint 训练的时序。get_block(i) 返回第 i 层就绪的 module，
        after(i) 在该层算完后调用（swap 路径用它记录「该 slot 可以被覆盖了」）。"""
        saved = []
        h = x0
        for i in range(n):
            saved.append(h)
            with torch.no_grad():
                h = get_block(i, forward=True)(h, vec, freqs)
            if after is not None:
                after(i)
        grad = torch.ones_like(h)
        for i in reversed(range(n)):
            inp = saved[i].detach().requires_grad_(True)
            with torch.enable_grad():
                out = get_block(i, forward=False)(inp, vec, freqs)
            grad = torch.autograd.grad(out, inp, grad)[0]
            if after is not None:
                after(i)
        return grad

    # 两条路径**交错**测量：顺序测会被 GPU 时钟状态差异污染（先跑的那条在冷态
    # 时钟未 boost）—— 首版顺序测量给出 -2.2% 的荒谬负开销就是这么来的
    def measure_ab(fn_a, fn_b) -> tuple[list[float], list[float]]:
        for _ in range(2):
            fn_a()
            fn_b()
        torch.cuda.synchronize()
        a_samples: list[float] = []
        b_samples: list[float] = []
        for _ in range(args.iters):
            for fn, bucket in ((fn_a, a_samples), (fn_b, b_samples)):
                start = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                bucket.append(time.perf_counter() - start)
        return a_samples, b_samples

    # ---- 基线：全常驻 + 同样的 checkpoint 重算语义（与 swap 同时驻留以便交错）
    resident = [make_block(device) for _ in range(n)]

    def resident_step():
        return step(lambda i, forward: resident[i])

    # ---- swap：前向顺序预取 + 反向逆序预取
    cpu_weights = [
        {k: v.detach().to("cpu").pin_memory() for k, v in blk.state_dict().items()}
        for blk in resident
    ]

    buffers = [make_block(device), make_block(device)]
    buf_params = [dict(b.state_dict()) for b in buffers]
    copy_stream = torch.cuda.Stream()
    ready = [torch.cuda.Event() for _ in range(2)]
    done = [torch.cuda.Event() for _ in range(2)]
    fetched = [-1, -1]  # 每个 slot 当前装着哪一层，避免重复搬运

    def prefetch(idx: int, slot: int) -> None:
        """把第 idx 层权重搬进 slot。未 record 过的 done event wait 是 no-op，
        所以首轮不需要特判。"""
        if idx < 0 or idx >= n or fetched[slot] == idx:
            return
        with torch.cuda.stream(copy_stream):
            # 该 slot 上一次的计算必须先完成，否则会覆盖正在被读的权重
            copy_stream.wait_event(done[slot])
            for key, dst in buf_params[slot].items():
                dst.copy_(cpu_weights[idx][key], non_blocking=True)
            ready[slot].record(copy_stream)
        fetched[slot] = idx

    def get_swapped(i: int, forward: bool):
        slot = i % 2
        prefetch(i, slot)                       # 本层（通常已被上一轮预取命中）
        prefetch(i + 1 if forward else i - 1, (i + 1) % 2 if forward else (i - 1) % 2)
        torch.cuda.current_stream().wait_event(ready[slot])
        return buffers[slot]

    def swap_step():
        # 每步重置：模拟权重不在 GPU 常驻，反向阶段也必须重新搬
        # （真实实现可复用前向末尾还在位的那两层，此处取保守估计）
        fetched[0] = fetched[1] = -1
        return step(get_swapped, after=lambda i: done[i % 2].record(
            torch.cuda.current_stream()
        ))

    res_samples, swap_samples = measure_ab(resident_step, swap_step)
    t_resident = statistics.median(res_samples)
    t_swap = statistics.median(swap_samples)
    spread = (
        statistics.stdev(res_samples) / t_resident * 100
        if len(res_samples) > 1 else 0.0
    )

    overhead = (t_swap - t_resident) / t_resident * 100
    saved_gb = compute["param_bytes"] * n / GIB
    print(f"全常驻+checkpoint : {t_resident * 1000:.1f} ms / {n} block"
          f"   (min {min(res_samples) * 1000:.1f}, 抖动 ±{spread:.1f}%)")
    print(f"block swap        : {t_swap * 1000:.1f} ms / {n} block"
          f"   (min {min(swap_samples) * 1000:.1f})")
    print(f"\n训练口径实测开销  : {overhead:+.1f}%")
    if abs(overhead) < spread:
        print(f"  ! 开销小于基线自身抖动（±{spread:.1f}%）—— 结论是「在噪声内」，"
              f"不是精确值")
    print(f"换来省显存        : {saved_gb:.2f} GB")
    record("G", "resident_ms", round(t_resident * 1000, 1), "ms", f"{n} blocks")
    record("G", "swap_ms", round(t_swap * 1000, 1), "ms", f"{n} blocks")
    record("G", "overhead_pct", round(overhead, 1), "%")
    record("G", "baseline_spread_pct", round(spread, 1), "%")

    per_block_ms = (t_swap - t_resident) / n * 1000
    print(f"每 block 额外     : {per_block_ms:+.2f} ms（前向+反向两个驻留窗口合计）")
    record("G", "per_block_extra_ms", round(per_block_ms, 2), "ms")

    # 不显式 del：resident / buffers 被上面两个闭包持有，函数返回后一并回收
    return {
        "overhead": overhead,
        "t_swap": t_swap,
        "t_resident": t_resident,
        "spread": spread,
    }


# ------------------------------------------------------------------ F 稳定性
def stage_f(args, nvml: Nvml, e_result: dict) -> None:
    """持续负载下的温度 / 功耗 / replay 增量 / 可用 RAM（doc §3.2）。"""
    section(f"F · 稳定性与硬件影响（{args.soak_seconds}s 持续 swap 负载）")
    import torch
    from training.sysmem import available_ram_bytes

    swap_pass = e_result.get("swap_pass")
    if swap_pass is None:
        print("E 段未产出 swap 循环，跳过")
        return

    replay_start = nvml.replay_counter()
    ram_start = available_ram_bytes()
    temps: list[int] = []
    powers: list[float] = []
    passes = 0

    print(f"{'t':>6} {'温度':>6} {'功耗':>8} {'可用RAM':>10} {'replay':>8}")
    print("-" * 44)
    start = time.perf_counter()
    next_sample = 0.0
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= args.soak_seconds:
            break
        swap_pass()
        torch.cuda.synchronize()
        passes += 1
        if elapsed >= next_sample:
            temp = nvml.temperature()
            power = nvml.power_watts()
            ram = available_ram_bytes()
            replay = nvml.replay_counter()
            if temp is not None:
                temps.append(temp)
            if power is not None:
                powers.append(power)
            print(
                f"{elapsed:>5.0f}s {str(temp) + '°C':>6} "
                f"{(f'{power:.0f}W' if power else '-'):>8} "
                f"{(f'{ram / GIB:.1f}GB' if ram else '-'):>10} "
                f"{str(replay):>8}"
            )
            next_sample = elapsed + 5.0

    replay_end = nvml.replay_counter()
    ram_end = available_ram_bytes()
    print(f"\n完成 {passes} 轮，{passes * args.swap_blocks} 次 block 换入")

    if replay_start is not None and replay_end is not None:
        delta = replay_end - replay_start
        print(f"PCIe replay 增量 : {delta}")
        record("F", "replay_delta", delta)
        if delta == 0:
            print("  → 链路零重传，PCIe 侧健康（doc §3.2 ② 通过）")
        else:
            print("  ! 出现链路重传：检查插槽 / riser / 是否走 chipset 通道")
    if temps:
        print(f"温度 max/mean    : {max(temps)}°C / {statistics.mean(temps):.0f}°C")
        record("F", "temp_max", max(temps), "°C")
    if powers:
        print(f"功耗 max/mean    : {max(powers):.0f}W / {statistics.mean(powers):.0f}W")
        record("F", "power_max", round(max(powers)), "W")
    if ram_start and ram_end:
        drift = (ram_start - ram_end) / MIB
        print(f"可用内存漂移     : {drift:+.0f} MB（pinned 常驻，不应持续增长）")
        record("F", "ram_drift_mb", round(drift))


# ------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(
        description="block swap 可行性 Gate-0 探针（见 docs/design/block-swap.md）"
    )
    parser.add_argument("--stages", default="ABCDEF", help="要跑的阶段，如 ABCD")
    parser.add_argument("--resolution", type=int, default=1024, help="训练/推理边长")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    parser.add_argument("--iters", type=int, default=10, help="每项测量的采样次数")
    parser.add_argument("--swap-blocks", type=int, default=8, help="E 段对照的 block 数")
    parser.add_argument("--soak-seconds", type=int, default=60, help="F 段持续时长")
    parser.add_argument("--out", type=Path, help="观测点写入的 CSV 路径")
    args = parser.parse_args()

    stages = args.stages.upper()
    nvml = Nvml()
    try:
        info = stage_a(nvml) if "A" in stages else {}

        import torch

        if not torch.cuda.is_available():
            print("\n无 CUDA 设备 —— B 段之后全部跳过。")
            return 0

        bandwidth = stage_b(args) if "B" in stages else {}
        compute = stage_c(args) if "C" in stages else {}
        if "D" in stages and bandwidth and compute:
            stage_d(bandwidth, compute)
        e_result = stage_e(args, compute) if "E" in stages and compute else {}
        g_result = stage_g(args, compute) if "G" in stages and compute else {}
        if "F" in stages and e_result:
            stage_f(args, nvml, e_result)

        section("结论摘要")
        if e_result:
            print(f"推理口径（前向）实测开销 {e_result['overhead']:+.1f}%")
        if g_result:
            print(f"训练口径（checkpoint+反向逆序）实测开销 {g_result['overhead']:+.1f}%")
        if compute:
            print(f"省显存 {compute['param_bytes'] * args.swap_blocks / GIB:.2f} GB "
                  f"（{args.swap_blocks} block, {args.resolution}²）；"
                  f"全 {compute['layers']} 层可省 {compute['total_bytes'] / GIB:.2f} GB")
        print("判读见 docs/design/block-swap.md §5 门槛与 §7 开放问题")

        if args.out:
            with args.out.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["stage", "metric", "value", "unit", "note"]
                )
                writer.writeheader()
                writer.writerows(RECORDS)
            print(f"\n观测点已写入 {args.out}（{len(RECORDS)} 条）")
        return 0
    finally:
        nvml.shutdown()


if __name__ == "__main__":
    sys.exit(main())
