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
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ... import secrets
from ..models import downloader as model_downloader
from .onnx_base import OnnxTaggerBase


class WD14Tagger(OnnxTaggerBase):
    name = "wd14"

    def __init__(self, overrides: dict | None = None) -> None:
        """`overrides` 是本次打标的临时覆盖（仅内存生效）。

        合并自 `secrets.WD14Config` 的同名字段（`threshold_general` /
        `threshold_character` / `model_id` / `local_dir` / `blacklist_tags`）；
        值为 None 的项沿用全局 settings，不影响 secrets.json 文件。
        """
        super().__init__()
        self._overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
        self._tags: list[str] = []
        self._tag_categories: list[int] = []  # 0=general, 4=character, 9=rating
        self._input_size: int = 448  # 默认；prepare 时覆盖

    # -------------------- config --------------------

    def _cfg(self) -> "secrets.WD14Config":
        """全局 secrets + 本次 overrides 合并出本次生效的配置。"""
        base = secrets.load().wd14.model_dump()
        for k, v in self._overrides.items():
            if k in base:
                base[k] = v
        return secrets.WD14Config(**base)

    def _get_batch_size_cfg(self) -> int:
        return int(self._cfg().batch_size or 1)

    # -------------------- model resolution --------------------

    def _local_model_dir_status(self) -> tuple[Path, bool]:
        cfg = self._cfg()
        if cfg.local_dir:
            d = Path(cfg.local_dir)
            ok = (d / "model.onnx").exists() and (d / "selected_tags.csv").exists()
            return d, ok
        d = model_downloader.wd14_target_dir(model_downloader.models_root(), cfg.model_id)
        ok = (d / "model.onnx").exists() and (d / "selected_tags.csv").exists()
        return d, ok

    def _resolve_model_dir(self) -> Path:
        cfg = self._cfg()
        if cfg.local_dir:
            d = Path(cfg.local_dir)
            if not (d / "model.onnx").exists() or not (d / "selected_tags.csv").exists():
                raise FileNotFoundError(
                    f"local_dir 缺少 model.onnx 或 selected_tags.csv: {d}"
                )
            return d
        default = model_downloader.wd14_target_dir(model_downloader.models_root(), cfg.model_id)
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
            d, ok = self._local_model_dir_status()
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        if ok:
            return True, f"模型: {d.name}"
        if self._cfg().local_dir:
            return False, f"local_dir 缺少 model.onnx 或 selected_tags.csv: {d}"
        return False, f"需下载模型: {d.name}"

    def prepare(self) -> None:
        if self._session is not None:
            return
        model_dir = self._resolve_model_dir()
        self._create_session(model_dir / "model.onnx")
        # 输入：通常 [N, H, W, C]；H==W；动态符号 fallback 到默认 448
        assert self._session is not None
        ish = self._session.get_inputs()[0].shape
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
