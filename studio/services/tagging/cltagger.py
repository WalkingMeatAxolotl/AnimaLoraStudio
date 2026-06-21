"""CLTagger ONNX 打标。

CLTagger 与 WD14 都走本地 onnxruntime，但模型资产和后处理不同：
CLTagger 使用 `tag_mapping.json`，输出 logits 需 sigmoid，角色标签有单独阈值。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ... import secrets
from .. import models as model_downloader
from .onnx_base import OnnxTaggerBase

logger = logging.getLogger(__name__)


def _blacklist_key(tag: str) -> str:
    """blacklist 比对归一键：下划线↔空格、大小写、首尾空格都不敏感。"""
    return tag.replace("_", " ").strip().lower()


_CATEGORY_ALIASES = {
    "0": "General",
    "general": "General",
    "tag": "General",
    "tags": "General",
    "4": "Character",
    "character": "Character",
    "characters": "Character",
    "9": "Rating",
    "rating": "Rating",
    "copyright": "Copyright",
    "meta": "Meta",
    "model": "Model",
    "quality": "Quality",
}


def _normalize_category(raw: object | None) -> str:
    text = str(raw or "").strip()
    if not text:
        return "General"
    return _CATEGORY_ALIASES.get(text.lower(), text)


@dataclass
class _LabelData:
    names: list[str | None]
    categories: list[str]


class CLTagger(OnnxTaggerBase):
    name = "cltagger"

    def __init__(self, overrides: dict | None = None) -> None:
        super().__init__()
        self._overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
        self._labels: _LabelData | None = None
        self._input_size: int = 448
        self._input_layout: str = "nchw"
        cfg = self._cfg()
        self._is_v2 = model_downloader.is_cltagger_v2_paths(
            cfg.model_path, cfg.tag_mapping_path
        )

    def _cfg(self) -> "secrets.CLTaggerConfig":
        base = secrets.load().cltagger.model_dump()
        for k, v in self._overrides.items():
            if k in base:
                base[k] = v
        model_path, tag_mapping_path = model_downloader.cltagger_canonical_file_paths(
            str(base.get("model_id", "")),
            str(base.get("model_path", "")),
            str(base.get("tag_mapping_path", "")),
        )
        base["model_path"] = model_path
        base["tag_mapping_path"] = tag_mapping_path
        return secrets.CLTaggerConfig(**base)

    def _get_batch_size_cfg(self) -> int:
        return int(self._cfg().batch_size or 1)

    def _compute_base(self) -> Path:
        cfg = self._cfg()
        return model_downloader.cltagger_target_root(model_downloader.models_root(), cfg.model_id)

    def _local_model_files_status(self) -> tuple[Path, Path, bool]:
        cfg = self._cfg()
        base = self._compute_base()
        model_path = base / cfg.model_path
        mapping_path = base / cfg.tag_mapping_path
        # v2 权重在外部 sidecar model.onnx.data（2GB+）里，必须连它一起校验：
        # 否则 onnx 图就绪但权重缺失，is_available 会误报"可用"，prepare 时
        # onnxruntime 才在加载 external data 处炸，错误对用户是黑盒。
        required = model_downloader.cltagger_required_files(
            cfg.model_path, cfg.tag_mapping_path
        )
        if all((base / f).exists() for f in required):
            return model_path, mapping_path, True
        return model_path, mapping_path, False

    def _resolve_model_files(self) -> tuple[Path, Path]:
        cfg = self._cfg()
        model_path, mapping_path, ok = self._local_model_files_status()
        if ok:
            return model_path, mapping_path
        model_downloader.download_cltagger(self._compute_base(), cfg, on_log=logger.info)
        model_path, mapping_path, ok = self._local_model_files_status()
        if not ok:
            base = self._compute_base()
            missing = [
                str(base / f)
                for f in model_downloader.cltagger_required_files(
                    cfg.model_path, cfg.tag_mapping_path
                )
                if not (base / f).exists()
            ]
            raise FileNotFoundError(
                f"CLTagger 下载后仍缺少 {', '.join(missing)}（请到设置→模型管理查看下载日志）"
            )
        return model_path, mapping_path

    def is_available(self) -> tuple[bool, str]:
        try:
            model_path, mapping_path, ok = self._local_model_files_status()
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        if not ok:
            return False, f"需下载模型: {model_path.parent.name}"
        return True, f"模型: {model_path.parent.name}"

    def prepare(self) -> None:
        if self._session is not None:
            return
        model_path, mapping_path = self._resolve_model_files()
        self._create_session(model_path)
        assert self._session is not None
        inputs = list(self._session.get_inputs())
        input_meta = self._select_input_meta(inputs)
        self._input_name = input_meta.name
        if self._is_v2:
            logits = [o.name for o in self._session.get_outputs() if o.name == "logits"]
            if logits:
                self._output_names = logits
        self._input_layout = self._infer_input_layout(input_meta.shape)
        self._input_size = self._infer_input_size(input_meta.shape, self._input_layout)
        self._labels = self._load_tag_mapping(mapping_path)

    def _select_input_meta(self, inputs: list[object]) -> object:
        if self._is_v2:
            for meta in inputs:
                if getattr(meta, "name", None) == "pixel_values":
                    return meta
        return inputs[0]

    @staticmethod
    def _infer_input_layout(shape: list[object]) -> str:
        if len(shape) >= 4:
            if shape[1] == 3:
                return "nchw"
            if shape[-1] == 3:
                return "nhwc"
        return "nchw"

    @staticmethod
    def _infer_input_size(shape: list[object], layout: str) -> int:
        dims = list(shape)
        if len(dims) >= 4:
            if layout == "nchw":
                for dim in (dims[2], dims[3]):
                    if isinstance(dim, int) and dim > 0:
                        return dim
            else:
                for dim in (dims[1], dims[2]):
                    if isinstance(dim, int) and dim > 0:
                        return dim
        for dim in reversed(dims):
            if isinstance(dim, int) and dim > 0 and dim != 3:
                return dim
        return 448

    @staticmethod
    def _load_tag_mapping(mapping_path: Path) -> _LabelData:
        raw = json.loads(mapping_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Unsupported CLTagger tag_mapping.json format")

        category_ids, category_tags = CLTagger._parse_vocabulary_categories(raw)
        if "idx_to_tag" in raw:
            idx_to_tag = CLTagger._parse_idx_to_tag(raw["idx_to_tag"])
            tag_to_category = CLTagger._parse_tag_to_category(raw, category_ids)
        elif "tag_to_idx" in raw:
            idx_to_tag = CLTagger._parse_tag_to_idx(raw["tag_to_idx"])
            tag_to_category = CLTagger._parse_tag_to_category(raw, category_ids)
        else:
            idx_to_tag = {}
            tag_to_category = {}
            for k, v in raw.items():
                if not isinstance(v, dict):
                    raise ValueError("Unsupported CLTagger tag_mapping.json format")
                tag = str(v["tag"])
                idx_to_tag[int(k)] = tag
                tag_to_category[tag] = _normalize_category(v.get("category", "General"))

        size = max(idx_to_tag.keys(), default=-1) + 1
        names: list[str | None] = [None] * size
        categories = ["General"] * size
        for idx, tag in idx_to_tag.items():
            names[idx] = tag
            categories[idx] = tag_to_category.get(tag, category_tags.get(tag, "General"))
        return _LabelData(names=names, categories=categories)

    @staticmethod
    def _parse_idx_to_tag(raw: object) -> dict[int, str]:
        if isinstance(raw, dict):
            return {int(k): str(v) for k, v in raw.items()}
        if isinstance(raw, list):
            return {i: str(v) for i, v in enumerate(raw) if str(v).strip()}
        raise ValueError("Unsupported CLTagger idx_to_tag format")

    @staticmethod
    def _parse_tag_to_idx(raw: object) -> dict[int, str]:
        if not isinstance(raw, dict):
            raise ValueError("Unsupported CLTagger tag_to_idx format")
        return {int(v): str(k) for k, v in raw.items()}

    @staticmethod
    def _parse_tag_to_category(
        raw: dict[str, object], category_ids: dict[str, str]
    ) -> dict[str, str]:
        tag_to_category_raw = raw.get("tag_to_category", {})
        if not isinstance(tag_to_category_raw, dict):
            return {}
        out: dict[str, str] = {}
        for tag, category in tag_to_category_raw.items():
            category_text = str(category).strip()
            out[str(tag)] = category_ids.get(
                category_text,
                _normalize_category(category_text),
            )
        return out

    @staticmethod
    def _parse_vocabulary_categories(raw: dict[str, object]) -> tuple[dict[str, str], dict[str, str]]:
        categories = raw.get("categories", {})
        if not isinstance(categories, dict):
            return {}, {}
        id_to_name: dict[str, str] = {}
        tag_to_name: dict[str, str] = {}
        for key, value in categories.items():
            key_text = str(key).strip()
            if isinstance(value, (str, int, float, bool)):
                id_to_name[key_text] = _normalize_category(value)
            elif isinstance(value, list):
                category_name = _normalize_category(key_text)
                for tag in value:
                    tag_text = str(tag).strip()
                    if tag_text:
                        tag_to_name[tag_text] = category_name
        return id_to_name, tag_to_name

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        size = self._input_size
        img = ImageOps.exif_transpose(img) or img
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA") if "transparency" in img.info else img.convert("RGB")
        if img.mode == "RGBA":
            canvas = Image.new("RGB", img.size, (255, 255, 255))
            canvas.paste(img, mask=img.split()[3])
            img = canvas
        if img.width != img.height:
            side = max(img.width, img.height)
            canvas = Image.new("RGB", (side, side), (255, 255, 255))
            canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
            img = canvas
        img = img.resize((size, size), Image.Resampling.BICUBIC)
        if self._is_v2:
            arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
            if self._input_layout == "nchw":
                arr = arr.transpose(2, 0, 1)
            return arr

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr[..., ::-1]  # RGB -> BGR
        if self._input_layout == "nchw":
            arr = arr.transpose(2, 0, 1)
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        else:
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        return (arr - mean) / std

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # clip ±30 同时吞 ±inf：sigmoid(±30) ≈ 1/0，超界本来也要饱和
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _postprocess_one(
        self, logits: np.ndarray
    ) -> tuple[list[str], dict[str, float]]:
        cfg = self._cfg()
        assert self._labels is not None
        scores = self._sigmoid(logits)
        out: list[tuple[str, float]] = []
        # blacklist 比对归一（与 wd14 一致）：下划线↔空格、大小写不敏感。
        # 注意此处 tag 还是原始下划线形式（输出时才转空格），归一后比对。
        blacklist = {_blacklist_key(b) for b in cfg.blacklist_tags}
        # CLTagger 7 category gate：General / Character 走阈值（不在此表里
        # 始终参与），其余按 cfg 开关；未知 category 当 General 处理。
        category_gates = {
            "Copyright": cfg.add_copyright_tag,
            "Meta": cfg.add_meta_tag,
            "Model": cfg.add_model_tag,
            "Rating": cfg.add_rating_tag,
            "Quality": cfg.add_quality_tag,
        }
        for i, p in enumerate(scores):
            if i >= len(self._labels.names):
                break
            tag = self._labels.names[i]
            if not tag or _blacklist_key(tag) in blacklist:
                continue
            cat = self._labels.categories[i]
            if cat in category_gates and not category_gates[cat]:
                continue
            thr = cfg.threshold_character if cat == "Character" else cfg.threshold_general
            p_f = float(p)
            if p_f >= thr:
                out.append((tag.replace("_", " "), p_f))
        out.sort(key=lambda x: -x[1])
        return [t for t, _ in out], dict(out)
