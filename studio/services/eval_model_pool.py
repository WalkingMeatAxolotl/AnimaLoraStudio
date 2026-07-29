"""指标模型的阶段级复用池。

一次评估里，`_stage_metric` 的形状本来就是「一个指标跑完所有候选，再下一个指标」：

    for runner in runners:            # clip → dino → ccip → tag
        for cand in candidates:       # 候选 0..N
            run_<runner>_job(...)

但四个 runner 的 `_default_scorer` 是**旧模型**留下的：那时候「每个候选 × 每个指标
= 一个独立子进程」，加载写在函数体里是唯一可能也是正确的形状 —— 反正进程马上就没
了。合成一个 worker 进程之后，同一个 CLIP 就被反反复复加载 N 次（200 个 checkpoint
= 200 次），纯属白花时间。

这里给每个 runner 一个模型持有者：

- **惰性**：仍在原来那个位置加载。dino / ccip 都有「没有配对参考图就早退」的分支在
  加载之前，硬把加载提到阶段外会让本来不用加载的情形也加载。
- **阶段级生命周期**：`_stage_metric` 在 finally 里 release，所以跑 DINO 时 CLIP
  已经不在卡上了 —— 这也是不能用模块级 `lru_cache` 的原因（那会一直留到进程退出）。
- **换 model_name 自动换**：key 不同就先释放旧的再加载新的。
"""
from __future__ import annotations

import gc
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ModelPool:
    """一个 runner 的模型持有者。非线程安全 —— 指标阶段本来就是串行的。"""

    def __init__(self, label: str) -> None:
        self._label = label
        self._key: Optional[str] = None
        self._value: Any = None

    @property
    def loaded(self) -> bool:
        return self._value is not None

    def get(self, key: str, loader: Callable[[], Any]) -> Any:
        """按 key 取；命中直接复用，未命中先释放旧的再 `loader()`。"""
        if self._value is not None and self._key == key:
            return self._value
        self.release()
        self._value = loader()
        self._key = key
        return self._value

    def release(self, progress: Optional[Callable[[str], None]] = None) -> None:
        """丢掉模型引用并把显存还给 caching allocator。未加载时是 no-op。"""
        if self._value is None:
            return
        self._value = None
        self._key = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            # torch 没装 / CUDA 不可用都不该让评估失败 —— 引用已经丢了，
            # 剩下的交给 GC
            logger.debug("%s: empty_cache 跳过", self._label, exc_info=True)
        if progress is not None:
            progress(f"[eval-{self._label}] 模型已释放")
