"""WD14 ONNX 打标（PP4）。

模型解析顺序：
    1. secrets.wd14.local_dir 给了 → 必须含 model.onnx + selected_tags.csv
    2. models/wd14/{model_id}/ 存在 → 用本地
    3. 否则 huggingface_hub.snapshot_download 拉到 models/wd14/{model_id}/

依赖：onnxruntime（CPU 默认；GPU 请用户自行装 onnxruntime-gpu）+
huggingface_hub + Pillow + numpy。
"""
from __future__ import annotations

import csv
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image, ImageOps

from .. import secrets
from ..paths import REPO_ROOT
from . import onnxruntime_setup
from .tagger import ProgressFn, TagResult

logger = logging.getLogger(__name__)


def _safe_dir_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace("\\", "_")


class WD14Tagger:
    name = "wd14"
    requires_service = False

    def __init__(self, overrides: dict | None = None) -> None:
        """`overrides` 是本次打标的临时覆盖（仅内存生效）。

        合并自 `secrets.WD14Config` 的同名字段（`threshold_general` /
        `threshold_character` / `model_id` / `local_dir` / `blacklist_tags`）；
        值为 None 的项沿用全局 settings，不影响 secrets.json 文件。
        """
        self._overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
        self._session = None
        self._tags: list[str] = []
        self._tag_categories: list[int] = []  # 0=general, 4=character, 9=rating
        self._input_size: int = 448  # 默认；prepare 时覆盖
        self._input_name: str | None = None

    # -------------------- config --------------------

    def _cfg(self) -> "secrets.WD14Config":
        """全局 secrets + 本次 overrides 合并出本次生效的配置。"""
        base = secrets.load().wd14.model_dump()
        for k, v in self._overrides.items():
            if k in base:
                base[k] = v
        return secrets.WD14Config(**base)

    # -------------------- model resolution --------------------

    def _resolve_model_dir(self) -> Path:
        cfg = self._cfg()
        if cfg.local_dir:
            d = Path(cfg.local_dir)
            if not (d / "model.onnx").exists() or not (d / "selected_tags.csv").exists():
                raise FileNotFoundError(
                    f"local_dir 缺少 model.onnx 或 selected_tags.csv: {d}"
                )
            return d
        default = REPO_ROOT / "models" / "wd14" / _safe_dir_name(cfg.model_id)
        if (default / "model.onnx").exists() and (default / "selected_tags.csv").exists():
            return default
        return self._download_model(cfg.model_id, default)

    def _download_model(self, model_id: str, target: Path) -> Path:
        from huggingface_hub import snapshot_download
        token = secrets.load().huggingface.token or None
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
            allow_patterns=["model.onnx", "selected_tags.csv"],
            token=token,
        )
        return target

    # -------------------- protocol --------------------

    def is_available(self) -> tuple[bool, str]:
        try:
            d = self._resolve_model_dir()
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        return True, f"模型: {d.name}"

    def prepare(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - install hint
            raise RuntimeError(
                "未安装 onnxruntime；请 `pip install onnxruntime` "
                "或 `onnxruntime-gpu`"
            ) from exc

        model_dir = self._resolve_model_dir()
        # 优先 GPU，回退 CPU
        providers = ["CPUExecutionProvider"]
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # PP9.5 — get_available_providers() 报 CUDA 可用 ≠ 真能 dlopen。
        # onnxruntime-gpu 在创 session 时才真去 dlopen libcurand/libcublas/...，
        # 缺系统 CUDA runtime 会挂在这里。捕异常降 CPU 重试，把原因 stash 给
        # Settings UI 显示，避免「打标全挂」的硬错误。
        model_path = str(model_dir / "model.onnx")
        try:
            self._session = ort.InferenceSession(model_path, providers=providers)
        except Exception as exc:  # noqa: BLE001
            if "CUDAExecutionProvider" not in providers:
                raise
            err = str(exc)
            logger.warning(
                "WD14 CUDA session 创建失败，降级 CPU 重试: %s", err
            )
            onnxruntime_setup.record_cuda_load_error(err)
            self._session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
        else:
            # 成功（无论 CUDA 还是 CPU）→ 清掉旧错误状态
            onnxruntime_setup.record_cuda_load_error(None)
        # 输入：通常 [N, H, W, C]；H==W
        ish = self._session.get_inputs()[0].shape
        self._input_name = self._session.get_inputs()[0].name
        # ish 可能是 ['N', 448, 448, 3] 或动态符号；尝试拿到 H
        for dim in ish[1:]:
            if isinstance(dim, int) and dim > 0:
                self._input_size = dim
                break

        with open(model_dir / "selected_tags.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # SmilingWolf 模型用 underscore，UI 习惯空格
                self._tags.append(row["name"].replace("_", " "))
                self._tag_categories.append(int(row.get("category", 0)))

    # -------------------- inference --------------------

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        """单图 → [H, W, 3] BGR float32。batch 推理时调用方负责 stack 成 [N, H, W, 3]。"""
        size = self._input_size
        img = ImageOps.exif_transpose(img) or img
        if img.mode != "RGB":
            img = img.convert("RGB")
        # 等比缩到 size，长边 == size，再用白色 pad 成正方形
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
        arr = np.asarray(canvas, dtype=np.float32)
        # WD14 训练用 BGR
        arr = arr[..., ::-1]
        return arr

    def _postprocess_one(
        self, scores: np.ndarray
    ) -> tuple[list[str], dict[str, float]]:
        """单张图的概率向量 → (sorted_tags, raw_scores_dict)。"""
        cfg = self._cfg()
        out: list[tuple[str, float]] = []
        blacklist = set(cfg.blacklist_tags)
        for i, p in enumerate(scores):
            if i >= len(self._tags):
                break
            tag, cat = self._tags[i], self._tag_categories[i]
            if tag in blacklist:
                continue
            # category: 9=rating（不参与阈值，丢弃）；4=character；其余按 general
            if cat == 9:
                continue
            thr = cfg.threshold_character if cat == 4 else cfg.threshold_general
            p_f = float(p)
            if p_f >= thr:
                out.append((tag, p_f))
        out.sort(key=lambda x: -x[1])
        return [t for t, _ in out], dict(out)

    def _effective_batch_size(self) -> int:
        """batch_size 决议：用户配置 → CPU EP 时强制 1（CPU batch 增益负的）。"""
        cfg = self._cfg()
        n = max(1, int(cfg.batch_size or 1))
        if self._session is not None:
            providers = list(self._session.get_providers())
            if "CUDAExecutionProvider" not in providers:
                return 1
        return n

    def _preprocess_one_safe(
        self, indexed: tuple[int, Path]
    ) -> tuple[int, Optional[np.ndarray], Optional[str]]:
        """`(j, path) → (j, arr_or_None, err_or_None)`，给 ThreadPool 用。

        PIL.Image.open / thumbnail / paste 在 C 层释放 GIL —— 多线程能真正并行，
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
        total = len(image_paths)
        batch_n = self._effective_batch_size()
        done = 0
        i = 0

        # PP10 — batch > 1 时 preprocess 走线程池，把单核 PIL 瓶颈拆开。
        # CPU EP 路径 batch_n 强制 1，pool=None 走原单线程逻辑（零回归）。
        pool: Optional[ThreadPoolExecutor] = None
        if batch_n > 1:
            pool = ThreadPoolExecutor(
                max_workers=batch_n, thread_name_prefix="wd14-prep"
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
            ok_idx: list[int] = []  # chunk 内成功 preprocess 的索引
            errs: dict[int, str] = {}
            if pool is None:
                # 单线程路径（CPU EP / batch=1，与 PP8 行为一致）
                for j, p in enumerate(chunk):
                    j, arr, err = self._preprocess_one_safe((j, p))
                    if err is None and arr is not None:
                        arrs.append(arr)
                        ok_idx.append(j)
                    else:
                        errs[j] = err or "preprocess failed"
            else:
                # 并发 preprocess —— pool.map 保序，结果按 chunk index 顺序回来
                for j, arr, err in pool.map(
                    self._preprocess_one_safe, list(enumerate(chunk))
                ):
                    if err is None and arr is not None:
                        arrs.append(arr)
                        ok_idx.append(j)
                    else:
                        errs[j] = err or "preprocess failed"
            # batch 推理（剩 0 张就跳过）
            logits_batch: Optional[np.ndarray] = None
            if arrs:
                try:
                    batch = np.stack(arrs, axis=0).copy()
                    logits_batch = self._session.run(
                        None, {self._input_name: batch}
                    )[0]
                except Exception as exc:  # noqa: BLE001
                    # 整 batch 推理失败 → 当作每张图都报错
                    for j in ok_idx:
                        errs[j] = f"inference failed: {exc}"
                    logits_batch = None
            # 按原 chunk 顺序 yield 结果
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
