"""ONNX 本地 tagger 共享基础设施。

WD14 和 CLTagger 都走本地 onnxruntime 的同一套 batch / CUDA fallback /
preprocess 线程池逻辑；差异在模型文件结构、tag 元数据格式、preprocess 流程、
postprocess 阈值规则。基类封住共性，子类填具体。
"""
from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image

from . import onnxruntime_setup
from .tagger import ProgressFn, TagResult

logger = logging.getLogger(__name__)


def safe_dir_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace("\\", "_")


@contextlib.contextmanager
def captured_fd_stderr():
    """临时把 fd 2 重定向到 tempfile，context 退出后把 bytes append 到 yield 的 list。

    onnxruntime 在 CUDA dlopen 失败（缺 libcublasLt.so.12 / cublasLt64_12.dll
    / libcurand 等）时，把关键报错直写到 fd 2（C 层 fprintf(stderr) /
    spdlog），不走 Python sys.stderr —— `try/except str(exc)` 看不到。
    **静默降级**路径（InferenceSession 不抛但 get_providers 没 CUDA）更要靠
    这条 fd 2 输出诊断，否则用户看到的只有「降级了」+ 没原因。

    旧版 silenced_fd_stderr 把它丢 /dev/null：日志干净但出问题完全瞎；
    现在改成 capture，让 _create_session 在错误信息里把它转发出来给 UI 显示。
    """
    try:
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    saved = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    captured: list[bytes] = []
    try:
        os.dup2(tmp.fileno(), 2)
        yield captured
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        try:
            tmp.seek(0)
            captured.append(tmp.read())
        finally:
            tmp.close()


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _decode_captured(buf: list[bytes]) -> str:
    """captured_fd_stderr 收到的 bytes → 干净文本（去 ANSI 色码 / NUL 字节）。"""
    raw = b"".join(buf)
    if not raw:
        return ""
    text = raw.decode("utf-8", errors="replace")
    text = _ANSI_RE.sub("", text)
    text = text.replace("\x00", "")
    return text.strip()


def _augment_with_capture(base: str, captured: list[bytes]) -> str:
    """如果 fd 2 捕到 ORT C 层报错，拼到 base 后面；空就原样返回。"""
    c_err = _decode_captured(captured)
    if not c_err:
        return base
    return f"{base}\nORT 原始报错:\n{c_err}"


class OnnxTaggerBase:
    """ONNX 本地 tagger 共享逻辑（CUDA fallback / batch 决议 / preprocess 线程池）。

    子类需要实现：
    - `prepare()`：解析模型文件 → 调 `_create_session(model_path)` → 装 metadata
    - `_preprocess(img) -> ndarray`：单张图预处理
    - `_postprocess_one(logits) -> (tags, raw_scores)`：单张后处理
    - `_get_batch_size_cfg() -> int`：从 secrets 读 batch_size

    子类必须设 `name`，用于日志 / 线程命名。
    """
    name: str = "onnx_base"
    requires_service = False

    def __init__(self) -> None:
        self._session = None
        self._input_name: Optional[str] = None
        self._output_names: Optional[list[str]] = None
        # 推理期 CUDA 失败 → _fallback_to_cpu_session() 用它重建 CPU session。
        # session 创建成功后由 _create_session 设上。
        self._model_path: Optional[Path] = None

    # -------------------- 子类实现 --------------------

    def prepare(self) -> None:
        raise NotImplementedError

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        raise NotImplementedError

    def _postprocess_one(
        self, logits: np.ndarray
    ) -> tuple[list[str], dict[str, float]]:
        raise NotImplementedError

    def _get_batch_size_cfg(self) -> int:
        raise NotImplementedError

    # -------------------- session 创建（含 CUDA fallback） --------------------

    def _create_session(self, model_path: Path) -> None:
        """创建 onnxruntime InferenceSession，CUDA 失败自动降 CPU。

        副作用：设 `_session` / `_input_name` / `_output_names` / `_model_path`，
        并 stash CUDA 错给 Settings UI（成功路径同时清掉旧错记录）。
        """
        self._model_path = model_path
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - install hint
            raise RuntimeError(
                "未安装 onnxruntime；请安装 onnxruntime 或 onnxruntime-gpu"
            ) from exc

        providers = ["CPUExecutionProvider"]
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # PP9.5 — get_available_providers() 报 CUDA 可用 ≠ 真能 dlopen。
        # 缺 CUDA runtime 时挂在 InferenceSession 创建（抛异常）或静默降 CPU
        # （不抛但 get_providers 没 CUDA）。两条路径 ORT 都把关键报错直写 fd 2，
        # 不走 Python sys.stderr —— 必须 capture 才能告诉用户根因。
        cuda_attempt = "CUDAExecutionProvider" in providers
        captured_stderr: list[bytes] = []
        try:
            if cuda_attempt:
                with captured_fd_stderr() as captured_stderr:
                    self._session = ort.InferenceSession(
                        str(model_path), providers=providers
                    )
            else:
                self._session = ort.InferenceSession(
                    str(model_path), providers=providers
                )
        except Exception as exc:  # noqa: BLE001
            if not cuda_attempt:
                raise
            err = _augment_with_capture(str(exc), captured_stderr)
            logger.warning(
                "%s CUDA session 创建失败，降级 CPU 重试: %s", self.name, err
            )
            onnxruntime_setup.record_cuda_load_error(err)
            self._session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        else:
            # onnxruntime 在 CUDA EP dlopen 失败时**不抛异常** —— 内部 silently
            # fallback 到下一个 EP（CPU）。光看 try/except 不够，必须比对实际
            # session.get_providers()；不一致 → 用户实际跑 CPU，但 UI 看不到。
            if cuda_attempt:
                actual = list(self._session.get_providers())
                if "CUDAExecutionProvider" not in actual:
                    base = (
                        f"CUDA EP 静默降级到 CPU（InferenceSession 未抛异常，"
                        f"但 get_providers={actual}）。"
                    )
                    captured_msg = _decode_captured(captured_stderr)
                    if captured_msg:
                        msg = f"{base}\nORT 原始报错:\n{captured_msg}"
                    else:
                        msg = (
                            base
                            + "常见原因：CUDA 驱动版本不够 / runtime so/DLL 缺失 / cuDNN ABI 错位。"
                        )
                    logger.warning("%s %s", self.name, msg)
                    onnxruntime_setup.record_cuda_load_error(msg)
                else:
                    onnxruntime_setup.record_cuda_load_error(None)
            else:
                onnxruntime_setup.record_cuda_load_error(None)

        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

    def _fallback_to_cpu_session(self) -> bool:
        """CUDA 推理失败后降 CPU 重建 session；成功返回 True。

        与 `_create_session` 的差别：那里是 session 创建期挂（dlopen 失败），
        这里是创建后推理期挂（典型 cuBLAS / cuDNN ABI 错位 → CUBLAS_STATUS_*）。
        重建后 `_session.get_providers()` 只剩 CPU，`_effective_batch_size`
        会自动把 batch_n 降到 1，后续批次不再走 CUDA。
        """
        if self._model_path is None:
            return False
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError:
            return False
        try:
            logger.warning(
                "%s CUDA 推理失败，降级 CPU InferenceSession（后续批次同样走 CPU）",
                self.name,
            )
            self._session = ort.InferenceSession(
                str(self._model_path), providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]
            onnxruntime_setup.record_cuda_load_error(
                "CUDA 推理时发生 cuBLAS / CUDA 错误，已自动降级 CPU 运行"
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("%s CPU session 降级失败: %s", self.name, exc)
            return False

    @staticmethod
    def _is_cuda_inference_error(exc: BaseException) -> bool:
        """判断推理异常是否是 CUDA 相关 → 触发 CPU fallback 重试一次。

        典型关键词：CUBLAS_STATUS_*、CUDNN_STATUS_*、CUDAExecutionProvider 报名。
        OOM 单独识别（"out of memory" / "OOM"），降 CPU 后多半也跑不动但至少给
        用户清晰错误而非黑盒崩。
        """
        msg = str(exc)
        keywords = ("CUBLAS", "CUDNN", "CUDAExecutionProvider", "out of memory", "OOM")
        return any(k in msg for k in keywords) or "CUDA" in msg

    # -------------------- batch + iteration --------------------

    def _effective_batch_size(self) -> int:
        n = max(1, int(self._get_batch_size_cfg() or 1))
        if self._session is not None:
            providers = list(self._session.get_providers())
            if "CUDAExecutionProvider" not in providers:
                return 1
        return n

    def _preprocess_one_safe(
        self, indexed: tuple[int, Path]
    ) -> tuple[int, Optional[np.ndarray], Optional[str]]:
        """`(j, path) → (j, arr_or_None, err_or_None)`，给 ThreadPool 用。

        PIL.Image.open / resize / paste 在 C 层释放 GIL —— 多线程能真正并行，
        在弱单核 + 强 GPU（云上 EPYC + RTX 5090）上把 preprocess 从瓶颈解开。
        """
        j, p = indexed
        try:
            with Image.open(p) as raw:
                return j, self._preprocess(raw), None
        except Exception as exc:  # noqa: BLE001
            return j, None, str(exc)

    def tag(
        self,
        image_paths: list[Path],
        on_progress: ProgressFn = lambda d, t: None,
    ) -> Iterator[TagResult]:
        if self._session is None:
            self.prepare()
        assert self._session is not None
        assert self._input_name is not None
        total = len(image_paths)
        batch_n = self._effective_batch_size()
        done = 0
        i = 0

        # batch > 1 才开池：CPU EP 路径 batch_n 已强制 1，pool=None 走单线程
        # 兼容路径，零回归。
        pool: Optional[ThreadPoolExecutor] = None
        if batch_n > 1:
            pool = ThreadPoolExecutor(
                max_workers=batch_n, thread_name_prefix=f"{self.name}-prep"
            )
        try:
            yield from self._tag_loop(
                image_paths, batch_n, pool, total, done, i, on_progress
            )
        finally:
            if pool is not None:
                pool.shutdown(wait=False)

    def _tag_loop(
        self,
        image_paths: list[Path],
        batch_n: int,
        pool: Optional[ThreadPoolExecutor],
        total: int,
        done: int,
        i: int,
        on_progress: ProgressFn,
    ) -> Iterator[TagResult]:
        assert self._session is not None
        while i < total:
            chunk = image_paths[i : i + batch_n]
            arrs: list[np.ndarray] = []
            ok_idx: list[int] = []
            errs: dict[int, str] = {}
            if pool is None:
                for j, p in enumerate(chunk):
                    j, arr, err = self._preprocess_one_safe((j, p))
                    if err is None and arr is not None:
                        arrs.append(arr)
                        ok_idx.append(j)
                    else:
                        errs[j] = err or "preprocess failed"
            else:
                for j, arr, err in pool.map(
                    self._preprocess_one_safe, list(enumerate(chunk))
                ):
                    if err is None and arr is not None:
                        arrs.append(arr)
                        ok_idx.append(j)
                    else:
                        errs[j] = err or "preprocess failed"
            logits_batch: Optional[np.ndarray] = None
            if arrs:
                batch = np.stack(arrs, axis=0).copy()
                try:
                    logits_batch = self._session.run(
                        self._output_names, {self._input_name: batch}
                    )[0]
                except Exception as exc:  # noqa: BLE001
                    # CUDA 推理失败（cuBLAS / cuDNN / OOM）→ 降 CPU session 重试一次
                    if self._is_cuda_inference_error(exc) and self._fallback_to_cpu_session():
                        try:
                            logits_batch = self._session.run(
                                self._output_names, {self._input_name: batch}
                            )[0]
                        except Exception as exc2:  # noqa: BLE001
                            for j in ok_idx:
                                errs[j] = f"inference failed: {exc2}"
                            logits_batch = None
                    else:
                        for j in ok_idx:
                            errs[j] = f"inference failed: {exc}"
                        logits_batch = None
            ok_pos = 0
            for j, p in enumerate(chunk):
                if j in errs:
                    yield {"image": p, "tags": [], "error": errs[j]}
                elif logits_batch is not None and ok_pos < logits_batch.shape[0]:
                    tags, raw_scores = self._postprocess_one(logits_batch[ok_pos])
                    ok_pos += 1
                    yield {"image": p, "tags": tags, "raw_scores": raw_scores}
                else:
                    yield {"image": p, "tags": [], "error": "no logits"}
                done += 1
                on_progress(done, total)
            i += batch_n
