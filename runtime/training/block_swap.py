"""Block swap —— DiT 的逐层权重换入换出（消费级卡下探 K2 显存门槛）。

设计与实测：``docs/design/block-swap.md``。一句话：DiT 的 N 个 transformer block
串行堆叠、任一时刻只算一个，所以把其中若干层的权重常驻 CPU pinned memory、算到
才搬进显存，可用「约 8ms/block 的固定时间」换回「每层约 0.8GB 显存」（krea2 实测），
把 K2 LoRA 训练的显存下限从 32GB 拉到 24GB 以下，且不付精度代价。

本模块是 **family 无关的机制核心**（doc §9.2 刀 1）：只认一个 ``nn.ModuleList``，
不认 krea2/anima。接线（哪个 family、哪个循环）在各 family 的行为适配层完成。

关键实现约束（doc §9.1，务必先读）：

- **原地换 ``param.data``，不是 buffer 轮转**。LyCORIS ``apply_to()`` 让 LoRA 模块
  持有原 Linear 引用并包住其 forward；若前向走另一个 module 实例，会**完全绕过
  LoRA**，训练静默学不到东西。所以 module 对象自始至终不变，只切换 ``.data`` 指向。
- 由此**自动正确**处理 fp8：``weight_scale`` 是绑在 module 上的非持久 buffer，module
  不变则 scale 恒与权重配对；换权重只换 ``.data``，fp8 张量原样搬（dtype 不变）。
- pinned 主副本**启动时一次性分配**，运行期不再 alloc（分配失败只可能发生在启动、
  可 fail-fast，doc §8.1）。

前向/反向的预取时序不在本模块——本模块只提供「把某层权重搬到某个 GPU 槽位」和
「某层算完、其 GPU 槽位可回收」两个原语，循环编排留给调用方（family 的 forward）。

**换出层的权重只在它自己的 forward 窗口内有效。** 一次 pass 结束后，某换出层的
``param.data`` 仍指向它当时用的 GPU 槽位，而那个槽位早已被后面的层覆盖（双缓冲
轮转：rel 0 与 rel 2 共用槽 0）。这不是 bug 而是设计的必然结果 —— 窗口外要读或改
权重（导出、检查、推理侧的 fp8 LoRA merge）**必须先 ``restore_masters()``**，
CPU pinned 主副本才是持久且完整的那份。tests 里有专门一条钉死这个语义。
"""

from __future__ import annotations

import logging
from typing import Iterator

import torch
from torch import nn


logger = logging.getLogger(__name__)

_GIB = 1024 ** 3

def _pow2_ceil(n: int) -> int:
    """>= n 的最小 2 的幂（n <= 1 → 1）。与 CachingHostAllocator 的取整同口径。"""
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _pow2_decomposition(total: int) -> list[int]:
    """把 total 拆成若干**互不相同**的 2 的幂（即二进制表示），降序。"""
    sizes: list[int] = []
    bit = 1
    while total:
        if total & 1:
            sizes.append(bit)
        total >>= 1
        bit <<= 1
    sizes.reverse()
    return sizes


def _alloc_pinned_bytes(nbytes: int) -> torch.Tensor:
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)


class PinnedAllocationError(RuntimeError):
    """pinned（页锁定）内存分配失败（doc §8.1 / B6：报错不静默降级）。

    ``cudaHostAlloc`` 失败在 torch 里也报 ``CUDA error: out of memory``，极易被
    误读成显存不够；这里把「是主机页锁定内存、要锁多少、Windows 上限在哪」说清楚。
    """

    def __init__(self, nbytes: int, detail: str) -> None:
        self.nbytes = nbytes
        self.detail = detail
        super().__init__(
            f"pinned（页锁定）内存分配失败：本次要锁定 {nbytes / _GIB:.2f} GB。"
            f"这不是显存不足 —— 换出层的权重要锁在系统内存里，Windows 对可锁定"
            f"内存有系统级上限（经验上约为物理内存的一半）。请调小 blocks_to_swap，"
            f"或关闭其他占内存的应用后重试。底层错误：{detail}"
        )


class PinnedPacker:
    """把多个张量打包进少数几块 **2 的幂大小** 的 pinned 大块里，再切 view 返回。

    为什么不能逐张量 ``pin_memory()``：PyTorch 的 host caching allocator 会把**每次**
    pinned 分配向上取整到 2 的幂（``CachingHostAllocator.h`` ``PowerOf2Ceil``），
    而 DiT 的权重尺寸恰恰都很吃亏 —— krea2 的 16384×6144 fp8 = 96MB 占 128MB、
    6144×6144 = 36MB 占 64MB、1536×6144 = 9MB 占 16MB，28 层 11.32GB 的权重实际
    锁定 **16.63GB**（1.47×）。Windows 对 ``cudaHostAlloc`` 的上限约为物理内存一半，
    32GB 内存的机器就是在这里撞死的（5080 真机案例），而护栏按 11.32GB 放行。

    做法：按**总字节数**的二进制分解一次性预分配若干块（8G+2G+1G+256M+…，每块
    恰为 2 的幂 → allocator 零取整），每个张量按 best-fit 装进某块并 256B 对齐，
    返回的是块上的 view（``is_pinned()`` 成立、可直接 ``param.data = view``、
    H2D 拷贝照常）。多族多配置模拟下实际锁定 = 权重字节 × 1.00–1.03。

    - ``total_bytes`` 是调用方算好的**将要 pin 的精确字节数**（不是估算：高估
      即白锁）；给 0 则不预分配，退化为按需开块。
    - 装不进任何已有块的张量（块边界碎片）走溢出路径：单独开一块
      ``pow2_ceil(nbytes)`` —— 与逐张量 pin 等价，永远不比旧行为差。
    - 所有分配在**构造时**完成（B6：失败只发生在启动那一刻，fail-fast），失败抛
      ``PinnedAllocationError``。
    - 释放跟随张量：所有 view 丢引用后大块回到 host 缓存池，再由
      ``release_pinned_host_cache`` 真正还给系统（§9.7 不变）。
    """

    #: 总量向上取整的粒度：太粗则小配置（anima 8 层 1GB）白锁一大截，太细则
    #: 块数变多、边界碎片变多。64MB 在全部真实配置模拟里都在 0.3%–3% 内。
    GRANULARITY = 64 * 1024 ** 2
    #: 块内偏移对齐（任何 dtype 的 view 都合法，且对 DMA 友好）
    ALIGN = 256

    def __init__(
        self,
        total_bytes: int = 0,
        *,
        granularity: int = GRANULARITY,
        align: int = ALIGN,
        allocate=None,
    ) -> None:
        self._align = int(align)
        self._allocate = allocate or _alloc_pinned_bytes
        # [buffer(uint8 pinned), 已用字节]
        self._chunks: list[list] = []
        self.packed_bytes = 0
        self.overflow_chunks = 0
        planned = 0
        if total_bytes > 0:
            planned = -(-int(total_bytes) // granularity) * granularity
        for size in _pow2_decomposition(planned):
            self._chunks.append([self._new_chunk(size), 0])

    def _new_chunk(self, nbytes: int) -> torch.Tensor:
        try:
            buf = self._allocate(nbytes)
        except RuntimeError as exc:  # cudaHostAlloc 失败（torch.AcceleratorError 亦是其子类）
            raise PinnedAllocationError(self.allocated_bytes + nbytes, str(exc)) from exc
        if buf.numel() != nbytes or buf.dtype != torch.uint8:
            raise RuntimeError("PinnedPacker 的 allocate 必须返回 nbytes 个 uint8")
        return buf

    @property
    def allocated_bytes(self) -> int:
        """实际分配（= 实际锁定）的字节数。"""
        return sum(int(buf.numel()) for buf, _used in self._chunks)

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def _reserve(self, nbytes: int) -> tuple[torch.Tensor, int]:
        """在某块里划出 ``nbytes``（best-fit + 对齐），返回 (块, 起始偏移)。"""
        best = None  # (剩余, chunk, 起始偏移)
        for chunk in self._chunks:
            buf, used = chunk
            offset = -(-used // self._align) * self._align
            left = int(buf.numel()) - offset - nbytes
            if left >= 0 and (best is None or left < best[0]):
                best = (left, chunk, offset)
        if best is None:
            chunk = [self._new_chunk(_pow2_ceil(nbytes)), 0]
            self._chunks.append(chunk)
            self.overflow_chunks += 1
            offset = 0
        else:
            _left, chunk, offset = best
        chunk[1] = offset + nbytes
        self.packed_bytes += nbytes
        return chunk[0], offset

    def pin(self, tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """把 ``tensor`` 的内容拷进 pinned 大块，返回同形状的 pinned view。

        ``dtype`` 给定时顺带 cast（拷贝即转换，省一次中间副本）。``tensor`` 可在
        任意设备（GPU 张量直接 D2H 进 pinned）。
        """
        target_dtype = dtype or tensor.dtype
        nbytes = tensor.numel() * torch.empty(0, dtype=target_dtype).element_size()
        buf, offset = self._reserve(nbytes)
        out = buf[offset: offset + nbytes].view(target_dtype).view(tuple(tensor.shape))
        out.copy_(tensor)
        return out


class PinnedBlockSwap:
    """管理一个 ``nn.ModuleList`` 中末尾 ``num_swap`` 个 block 的权重换入换出。

    "末尾" 而非任意子集：DiT 前向从 0 到 N-1，把靠后的层换出可让前面的层先跑完、
    为后面腾出的时间窗口最大（且与 musubi ``blocks_to_swap`` 语义一致——它也是从
    尾部数）。前 ``N - num_swap`` 个 block 权重常驻 GPU 不受影响。

    生命周期：
        swap = PinnedBlockSwap(blocks, num_swap, device)   # 分配 pinned + GPU 槽
        # 每步前向：
        for i, block in enumerate(blocks):
            swap.ensure_resident(i)      # 换出层 → 确保权重在 GPU（含预取等待）
            h = block(h, ...)
            swap.release(i)              # 换出层 → 标记其 GPU 槽可被下一层复用
        # 反向逆序同理（调用方按 reversed 顺序调 ensure_resident/release）

    非换出层（``i < first_swapped``）的 ensure_resident/release 是 no-op，调用方
    可以无条件调用，不必自己判断边界。
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        num_swap: int,
        device: torch.device | str,
        *,
        num_slots: int = 2,
    ) -> None:
        total = len(blocks)
        if num_swap <= 0:
            raise ValueError("PinnedBlockSwap 的 num_swap 必须为正（0 = 不该构造本对象）")
        if num_swap > total:
            raise ValueError(
                f"num_swap={num_swap} 超过 block 总数 {total}"
            )
        if num_slots < 2:
            raise ValueError("num_slots 至少为 2（双缓冲：算当前 + 预取下一）")

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                f"block swap 需要 CUDA 设备，收到 {self.device}"
            )
        self.blocks = blocks
        self.total = total
        self.num_swap = num_swap
        self.first_swapped = total - num_swap  # 第一个被换出的 block index

        # 每个被换出 block 的 CPU pinned 权重主副本：
        #   [block 相对序号] -> {param 名: pinned CPU tensor}
        # 用相对序号（0 .. num_swap-1）避免和绝对 index 混淆。
        self._cpu_weights: list[dict[str, torch.Tensor]] = []
        # 每个 param 的形状/dtype 元信息，用于在 GPU 槽里建对应 buffer
        self._param_specs: list[list[tuple[str, torch.Size, torch.dtype]]] = []

        # GPU 槽位：num_slots 份，每份能放下任一被换出 block 的全部 param。
        #   _slot_buffers[slot] = {param 名: GPU tensor}
        self._slot_buffers: list[dict[str, torch.Tensor]] = []
        # 每个槽当前装着哪个相对序号的 block（-1 = 空）
        self._slot_holds: list[int] = [-1] * num_slots
        self.num_slots = num_slots

        self._copy_stream = torch.cuda.Stream(device=self.device)
        # ready[slot]：该槽的权重搬运已完成（计算流 wait 它才能读）
        self._ready = [torch.cuda.Event() for _ in range(num_slots)]
        # done[slot]：该槽上一次计算已完成（拷贝流 wait 它才能覆盖，防数据竞争）
        self._done = [torch.cuda.Event() for _ in range(num_slots)]

        self._pinned_bytes = 0
        self._handles: list = []
        self._build(blocks)

    # ------------------------------------------------------------------ 构造
    def _build(self, blocks: nn.ModuleList) -> None:
        """把被换出的 block 权重搬到 CPU pinned，并在 GPU 上预留槽位。

        pinned 分配失败在此抛出（启动期，可 fail-fast，doc §8.1）。
        """
        try:
            # 尚未 pinned 的基权重全部经 PinnedPacker 打包（逐张量 pin_memory 会被
            # host allocator 按 2 的幂取整、白锁最多 1.47×，见 PinnedPacker）。
            # 先数总量再一次性分配：失败即 fail-fast，不会搬了一半才炸。
            to_pack = 0
            for absolute in range(self.first_swapped, self.total):
                for param in blocks[absolute].parameters():
                    if param.requires_grad or (
                        param.device.type == "cpu" and param.is_pinned()
                    ):
                        continue
                    to_pack += param.numel() * param.element_size()
            packer = PinnedPacker(to_pack) if to_pack > 0 else None

            for rel, absolute in enumerate(range(self.first_swapped, self.total)):
                block = blocks[absolute]
                cpu_w: dict[str, torch.Tensor] = {}
                specs: list[tuple[str, torch.Size, torch.dtype]] = []
                for name, param in block.named_parameters():
                    # **只管理冻结的基权重**。可训练参数（LoRA）必须原地不动、
                    # 常驻 GPU：它们是优化器的目标，被搬走会破坏训练；而且它们
                    # 相对底模极小，没有换出的价值。
                    if param.requires_grad:
                        continue
                    # 已在 CPU 且已 pinned 就地接管（loader 可直接把尾部层载到 CPU
                    # pinned，让 GPU 峰值从不经过完整模型——12/16GB 目标的前提）；
                    # 否则（CPU 可分页 / 还在 GPU）打包进 pinned 大块。
                    src = param.detach()
                    if src.device.type == "cpu" and src.is_pinned():
                        pinned = src
                    else:
                        pinned = packer.pin(src)
                    cpu_w[name] = pinned
                    specs.append((name, param.shape, param.dtype))
                    self._pinned_bytes += pinned.numel() * pinned.element_size()
                    # 权重主副本已在 CPU pinned——立即释放 GPU 上的原权重（显存
                    # 收益所在，否则要到首次 forward rebind 才兑现）。.data 指向
                    # 这份 pinned CPU 张量（而非 empty(0)）：保留 shape/dtype，
                    # 让**构造后**才注入的 LyCORIS 能正确读到基权重形状；后续
                    # ensure_resident 再把 .data 切到 GPU 槽。
                    param.data = pinned
                self._cpu_weights.append(cpu_w)
                self._param_specs.append(specs)
            torch.cuda.empty_cache()  # 归还刚释放的原权重段给分配器

            # 预留 GPU 槽：容量 = 被换出 block 里最大的那个（同构 DiT 里都一样大）
            for _slot in range(self.num_slots):
                buf: dict[str, torch.Tensor] = {}
                for name, shape, dtype in self._param_specs[0]:
                    buf[name] = torch.empty(shape, dtype=dtype, device=self.device)
                self._slot_buffers.append(buf)
        except RuntimeError as exc:  # pinned / GPU 分配失败
            self._pinned_bytes = 0
            raise BlockSwapAllocationError(
                self.num_swap, self.first_swapped, str(exc)
            ) from exc

        logger.debug(
            "block_swap: ready swapped=%d/%d pinned=%.2f GB gpu_slots=%d",
            self.num_swap, self.total, self._pinned_bytes / _GIB, self.num_slots,
        )

    @property
    def pinned_bytes(self) -> int:
        """CPU pinned 主副本总字节（护栏预算依据）。"""
        return self._pinned_bytes

    # ------------------------------------------------------------------ 原语
    def _slot_for(self, rel: int) -> int:
        """相对序号 → 使用的 GPU 槽（双缓冲下 rel 的奇偶）。"""
        return rel % self.num_slots

    def _fetch(self, rel: int) -> None:
        """在拷贝流上把第 rel 个换出 block 的权重搬进它的槽（若尚未在位）。"""
        if rel < 0 or rel >= self.num_swap:
            return
        slot = self._slot_for(rel)
        if self._slot_holds[slot] == rel:
            return  # 已在位（通常是上一步预取命中）
        with torch.cuda.stream(self._copy_stream):
            # 该槽上一次计算必须先完成，否则覆盖正在被读的权重（数据竞争）。
            # 未 record 过的 Event.wait 是 no-op，首轮天然安全。
            self._copy_stream.wait_event(self._done[slot])
            buf = self._slot_buffers[slot]
            src = self._cpu_weights[rel]
            for name, dst in buf.items():
                dst.copy_(src[name], non_blocking=True)
            self._ready[slot].record(self._copy_stream)
        self._slot_holds[slot] = rel
        self._rebind(rel, slot)

    def _rebind(self, rel: int, slot: int) -> None:
        """把该 block 的基权重 ``.data`` 指到槽 buffer（原地换，不换 module）。

        只重绑**构造时登记过的**参数名：构造之后新增的参数（LoRA 注入在
        block 内建子模块的情形）不归本组件管，遍历 named_parameters() 会
        撞上它们。
        """
        block = self.blocks[self.first_swapped + rel]
        buf = self._slot_buffers[slot]
        params = dict(block.named_parameters())
        for name, _shape, _dtype in self._param_specs[rel]:
            param = params.get(name)
            if param is not None:
                param.data = buf[name]

    def ensure_resident(self, absolute_index: int, *, prefetch_next: int | None = None) -> None:
        """确保第 ``absolute_index`` 个 block 的权重已在 GPU 且计算流可安全读取。

        对非换出层（常驻）是 no-op。``prefetch_next`` 若给出（下一个要用的 block
        绝对序号），顺带发起它的预取——这是遮蔽传输的关键，调用方应传前向的 i+1
        或反向的 i-1。
        """
        rel = absolute_index - self.first_swapped
        if rel < 0:
            return  # 常驻层
        self._fetch(rel)
        if prefetch_next is not None:
            nxt = prefetch_next - self.first_swapped
            if 0 <= nxt < self.num_swap:
                self._fetch(nxt)
        torch.cuda.current_stream().wait_event(self._ready[self._slot_for(rel)])

    def release(self, absolute_index: int) -> None:
        """标记该 block 计算已在计算流上发起完毕，其 GPU 槽可被后续 block 覆盖。

        对非换出层是 no-op。必须在该 block 的 forward 调用之后调用。
        """
        rel = absolute_index - self.first_swapped
        if rel < 0:
            return
        self._done[self._slot_for(rel)].record(torch.cuda.current_stream())

    def reset(self) -> None:
        """一步（前向或反向）开始前重置槽占用状态。

        双缓冲槽在上一步末尾装着最后两层；新的一步从头/尾开始，需要重新预取。
        不释放显存，只清 hold 标记。
        """
        self._slot_holds = [-1] * self.num_slots

    def restore_masters(self) -> None:
        """把所有被管理的参数 ``.data`` 指回 CPU pinned 主副本。

        推理侧必需（doc §9.6）。两个场景：

        1. **fp8 LoRA merge**：merge 会写 ``module.weight``。若此刻 ``.data`` 指向
           GPU 槽，写进去的 delta 会被下一层的换入**直接覆盖** —— merge 静默丢失。
           先 restore 再 merge，delta 落在主副本上，之后每次换入带的都是 merged 权重。
        2. **任何要读/改权重的外部操作**（导出、检查、重新量化）：主副本是唯一
           完整且稳定的那份，GPU 槽只是轮转窗口。
        """
        for rel in range(self.num_swap):
            block = self.blocks[self.first_swapped + rel]
            params = dict(block.named_parameters())
            master = self._cpu_weights[rel]
            for name, _shape, _dtype in self._param_specs[rel]:
                param = params.get(name)
                if param is not None:
                    param.data = master[name]
        # 槽内容已与 param 解绑，标记为空避免下次误判命中
        self._slot_holds = [-1] * self.num_slots

    def managed_data_ptrs(self) -> set[int]:
        """被本组件管理的张量**存储地址**集合（CPU 主副本 + GPU 槽）。

        给外部的「搬运整个模型」操作用：这些张量**不能**被 ``module.to(device)``
        之类的一刀切搬上 GPU，否则 block swap 白做（见 ``move_module_excluding``）。

        用 ``data_ptr()`` 而非 ``id()``：``param.data`` 每次访问都返回**新的**
        Python 包装对象，``id()`` 不稳定，拿它比对会全部漏判。
        """
        ptrs = set()
        for weights in self._cpu_weights:
            ptrs.update(t.data_ptr() for t in weights.values())
        for buf in self._slot_buffers:
            ptrs.update(t.data_ptr() for t in buf.values())
        return ptrs

    def attach(self) -> None:
        """给每个换出 block 注册前向 + 反向钩子，接管换入换出。

        这是**推荐的接线方式**：完全从外部生效，不需要改模型的 forward 循环
        （krea2 的循环在 parity 敏感的 ``modeling/`` 内，不宜改动，doc §7.1）。

        **四个钩子缺一不可**（前向 pre/post + 反向 pre/post）：

        - 前向 pre 取回权重、post 放开槽位；
        - **反向 pre 必须再取一次** —— 前向 post 已经放开了槽位（不放开的话前向
          期间没有任何 done 事件、双缓冲失去保护），到本 block 反向时槽里装的
          早已是别的层。

        曾经以为「开 gradient checkpointing 后反向重算会触发 forward hook，逆序
        换入自动成立」，**那是错的**：重算的 forward_hook 根本不触发（实测
        checkpoint 下 pre 触发 2N 次而 post 只 N 次），且重算之后本 block 的反向
        仍要读权重。少了反向钩子，梯度会**静默**算错 —— 不报错、不 NaN，只是数值
        不对，真机实测偏差达非确定性噪声底的 300 倍，PPSF 的 d 估计随即炸掉。
        回归见 ``tests/test_block_swap_grad_fidelity.py``（必须用真实尺寸 +
        噪声底校准，小张量测试对此完全不敏感）。

        幂等：重复调用不会重复注册。
        """
        if self._handles:
            return
        for rel in range(self.num_swap):
            absolute = self.first_swapped + rel
            block = self.blocks[absolute]

            def pre_hook(_module, _args, idx=absolute):
                self.ensure_resident(idx, prefetch_next=idx + 1)

            def post_hook(_module, _args, output, idx=absolute):
                self.release(idx)
                return output

            def backward_pre_hook(_module, _gout, idx=absolute):
                # **反向必须自己把权重取回来。** 前向结束就放开了槽位，到本 block
                # 反向时槽里装的早已是别的层 —— 少了这一步梯度会**静默**算错
                # （不报错、不 NaN，只是数值不对；真机实测偏差达噪声底 300 倍，
                # PPSF 的 d 估计直接炸掉）。开 checkpoint 时紧邻的重算刚把权重
                # 放好，这里命中直接返回、零额外传输。
                self.ensure_resident(idx, prefetch_next=idx - 1)

            def backward_hook(_module, _gin, _gout, idx=absolute):
                self.release(idx)

            self._handles.append(block.register_forward_pre_hook(pre_hook))
            self._handles.append(block.register_forward_hook(post_hook))
            self._handles.append(block.register_full_backward_pre_hook(backward_pre_hook))
            self._handles.append(block.register_full_backward_hook(backward_hook))
        logger.debug("block_swap: forward/backward hooks attached blocks=%d", self.num_swap)

    def detach(self) -> None:
        """移除 attach 注册的钩子（权重不还原，需要时自行 ensure_resident）。"""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def close(self) -> None:
        """彻底放手：摘钩子 + 把被管理参数指向空张量 + 丢弃主副本与 GPU 槽。

        （名字刻意不叫 ``release`` —— 那个已经是「某层算完、槽位可复用」的
        每层原语，语义完全不同。）

        **调用后模型不可再用**，只在确定用完时调（训练收尾 / 模型卸载）。

        为什么需要它，而不是「把 model 引用丢掉就行」：pinned 主副本被 param.data
        引用着，而持有 block 的不只有 `ctx.model` —— LyCORIS injector 持 org_module、
        optimizer 持参数、hook 闭包也可能持有。真机实测只丢 swap 对象归还 0 字节，
        丢 model 之后才归还。与其到处找持有者，不如本对象主动把参数指走。

        注意仍需再调 ``release_pinned_host_cache()`` 才真正还给操作系统（那是
        host caching allocator 的另一层，doc §9.7）。
        """
        self.detach()
        for rel in range(min(self.num_swap, len(self._param_specs))):  # 幂等
            block = self.blocks[self.first_swapped + rel]
            params = dict(block.named_parameters())
            for name, _shape, dtype in self._param_specs[rel]:
                param = params.get(name)
                if param is not None:
                    param.data = torch.empty(0, dtype=dtype, device=self.device)
        self._cpu_weights.clear()
        self._slot_buffers.clear()
        self._param_specs.clear()
        self._slot_holds = []
        self._pinned_bytes = 0

    def iter_forward(self) -> Iterator[tuple[int, nn.Module]]:
        """前向遍历便捷封装：yield (index, block)，自动 ensure_resident+预取+release。

        调用方：``for i, block in swap.iter_forward(): h = block(h, ...)``
        注意 release 在 yield 返回后调用，所以调用方必须在循环体内完成 forward。
        """
        self.reset()
        for i in range(self.total):
            self.ensure_resident(i, prefetch_next=i + 1)
            yield i, self.blocks[i]
            self.release(i)


def release_pinned_host_cache() -> None:
    """把 pinned（页锁定）内存还给操作系统。

    与 ``torch.cuda.empty_cache()`` 是**两件事**：后者只管设备侧。pinned 走
    PyTorch 独立的 host caching allocator，释放张量只是还给那个缓存池 —— 真机
    实测 pin 6GB 后 ``del`` + ``gc.collect()`` 归还 **0 字节**，调用本函数才
    归还 8GB（doc §9.7）。

    block swap 的主副本可达 11GB+，漏掉这步就是「卸载了但内存没还」，而且页
    锁定内存连换页都不行，其他程序完全用不到。与 ``_cuda_clearCublasWorkspaces``
    是同一类问题（C++/分配器层常驻，Python GC 看不见）的 host 侧版本。

    **调用时机**：只在确定不再需要那批权重时（模型卸载 / 训练收尾）。出图或训练
    过程中绝不能调 —— pinned 里装的就是模型权重本身。

    内部 API，缺失/失败静默跳过（下轮加载会复用缓存，只是内存不还系统）。
    """
    try:
        torch._C._host_emptyCache()
    except Exception:  # noqa: BLE001
        pass


def move_module_excluding(module: nn.Module, device, swap: "PinnedBlockSwap | None") -> None:
    """把 ``module`` 搬到 ``device``，但**跳过 block swap 管理的参数**。

    推理侧的 daemon 会在每个任务前把整个模型搬回 GPU（采样期 offload 之后要搬
    回来）。那是个一刀切的 ``module.to(device)`` —— 在 block swap 下会把换出层的
    CPU pinned 主副本一起搬上卡，swap 白做，而且瞬时占用等于完整模型，在 12GB
    卡上直接 OOM。

    ``swap`` 为 None 时退化为普通的 ``module.to(device)``（零行为变化）。
    """
    if module is None or not hasattr(module, "to"):
        return
    if swap is None:
        module.to(device)
        return
    managed = swap.managed_data_ptrs()
    target = torch.device(device)
    for _name, param in module.named_parameters(recurse=True):
        if param.data.data_ptr() in managed or param.data.device == target:
            continue
        param.data = param.data.to(target)
        if param.grad is not None:
            param.grad = param.grad.to(target)
    for _name, buf in module.named_buffers(recurse=True):
        if buf.data_ptr() in managed or buf.device == target:
            continue
        # buffer 要经所属 module 重新注册才能换实例；直接改 .data 对
        # 非 Parameter 的 Tensor 同样生效（buffer 存的就是 Tensor）
        buf.data = buf.data.to(target)


class BlockSwapAllocationError(RuntimeError):
    """pinned / GPU 槽分配失败（doc §8.1 / B6：报错不静默降级）。

    只可能在启动期 ``PinnedBlockSwap`` 构造时抛出。携带足够上下文让上层给出
    可操作的用户文案（关掉占内存的应用 / 调小 blocks_to_swap）。
    """

    def __init__(self, num_swap: int, first_swapped: int, detail: str) -> None:
        self.num_swap = num_swap
        self.first_swapped = first_swapped
        self.detail = detail
        super().__init__(
            f"block swap 预分配失败（换出末尾 {num_swap} 个 block）：{detail}"
        )
