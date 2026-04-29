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
from typing import Iterator

import numpy as np
from PIL import Image, ImageOps

from .. import secrets
from ..paths import REPO_ROOT
from .tagger import ProgressFn, TagResult


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
        self._session = ort.InferenceSession(
            str(model_dir / "model.onnx"), providers=providers
        )
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
        return np.expand_dims(arr, 0).copy()

    def _postprocess(
        self, logits: np.ndarray
    ) -> tuple[list[str], dict[str, float]]:
        cfg = self._cfg()
        scores = logits[0]
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

    def tag(
        self,
        image_paths: list[Path],
        on_progress: ProgressFn = lambda d, t: None,
    ) -> Iterator[TagResult]:
        if self._session is None:
            self.prepare()
        assert self._session is not None
        total = len(image_paths)
        for i, p in enumerate(image_paths):
            try:
                with Image.open(p) as raw:
                    arr = self._preprocess(raw)
                logits = self._session.run(None, {self._input_name: arr})[0]
                tags, raw_scores = self._postprocess(logits)
                yield {"image": p, "tags": tags, "raw_scores": raw_scores}
            except Exception as exc:  # noqa: BLE001
                yield {"image": p, "tags": [], "error": str(exc)}
            on_progress(i + 1, total)
