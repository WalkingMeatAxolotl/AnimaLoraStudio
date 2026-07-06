"""数据集与 collate：ARB 分桶 + ImageDataset + 正则集 merge + cached latent。

NaViT / Patch-n-Pack 块对角打包（Phase 2 数据层）：
- 分块 VAE encode：委托 VAEWrapper._tiled_encode（cache_encode_tiled）
- token 预算打包 NavitPackBatchSampler + pack_indices_by_budget / pack_indices_ffd_windowed
- CachedLatentDataset 扩展：分块 encode、token_count_for_index
- collate_fn_navit_pack —— 异构 latent 逐图列表 collate

抽自原 runtime/anima_train.py L1144-1675 + L1939-1962（ADR 0003 PR-A）。

公开：
- BucketManager / ImageDataset / RepeatDataset / MergedDataset
- BucketBatchSampler / CachedLatentDataset
- collate_fn / collate_fn_cached — DataLoader collate
- NavitPackBatchSampler / collate_fn_navit_pack — 块对角打包
"""

from __future__ import annotations

import logging
import math
import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class BucketManager:
    """ARB 分桶管理.

    SYNC WITH ``studio/web/src/lib/trainBuckets.ts``. The crop page on the web
    UI predicts trainer buckets to pre-align cluster crops so the trainer
    doesn't re-resize them — that prediction depends on a TS port of this
    class. Any change to the algorithm or to the default parameters
    (``base_reso``, ``step``, the 0.1 area tolerance, the
    ``aspect_ratio_limit`` R, the min/max derivation) MUST land in both files
    in the same commit, or the frontend's predicted bucket ≠ trainer's actual
    bucket and crops will silently degrade.

    The bucket set is a pure function of ``(base_reso, aspect_ratio_limit,
    step)``:

    - ``aspect_ratio_limit`` (R, default 2.0) symmetrically caps the widest
      bucket at R:1 and the tallest at 1:R.
    - ``min_reso`` / ``max_reso`` are the edge-length search bounds. When not
      given they are **derived** from ``(base_reso, R)`` — at constant area
      base² the most extreme bucket has edges ``base·√R × base/√R``, so the
      bounds round outward to ``≈ base/√R`` and ``≈ base·√R`` (one ``step`` of
      margin so quantization never clips; the area band + AR cap do the real
      cut). Passing them explicitly (tests / special cases) overrides the
      derivation. The old hard-wired 512/2048 degrade at small base — e.g.
      base=512 left only the 512×512 square, killing all AR variety — which is
      why the bounds now scale with base.

    See ``docs/design/preprocess-crop-design.md`` §7 for the crop UX policy and
    ``docs/design/multi-resolution-training-design.md`` §6 for the derivation.
    """
    def __init__(self, base_reso=1024, min_reso=None, max_reso=None, step=64,
                 aspect_ratio_limit=2.0):
        self.base_reso = base_reso
        self.aspect_ratio_limit = aspect_ratio_limit
        self.step = step
        if min_reso is None or max_reso is None:
            span = math.sqrt(aspect_ratio_limit)
            derived_min = max(step, int(math.floor(base_reso / span / step) * step) - step)
            derived_max = int(math.ceil(base_reso * span / step) * step) + step
            if min_reso is None:
                min_reso = derived_min
            if max_reso is None:
                max_reso = derived_max
        self.min_reso = min_reso
        self.max_reso = max_reso
        self.buckets = self._generate(min_reso, max_reso, step, base_reso, aspect_ratio_limit)

    def _generate(self, min_r, max_r, step, base, ar_limit):
        # Keep algorithm identical to trainBuckets.generateBuckets() in TS:
        #   - double loop over (w, h) in [min_r, max_r] step `step`
        #   - area within ±10% of base² (the 0.1 below)
        #   - max AR ratio ≤ ar_limit (R)
        # Default-param consumers (base=1024, R=2.0) should see exactly the same
        # 37 buckets on both sides — covered by
        # `studio/web/src/lib/trainBuckets.test.ts` asserting count == 37.
        buckets = []
        base_area = base * base
        for w in range(min_r, max_r + 1, step):
            for h in range(min_r, max_r + 1, step):
                if abs(w * h - base_area) / base_area > 0.1:
                    continue
                if max(w/h, h/w) > ar_limit:
                    continue
                buckets.append((w, h))
        return buckets

    def get_bucket(self, w, h):
        # Snap by ABSOLUTE AR distance — not relative. The TS port
        # `trainBuckets.snapToBucket()` mirrors this exactly. Multiple buckets
        # may share the same aspect ratio under the ±10% area band (e.g.
        # 1472²/1536²/1600² when base=1536); in that tie, prefer the bucket
        # whose area is closest to base² so exact-square inputs land on the
        # configured base square instead of the first smaller square.
        aspect = w / h
        base_area = self.base_reso * self.base_reso
        best = (self.base_reso, self.base_reso)
        best_score = (float("inf"), float("inf"))
        for bw, bh in self.buckets:
            score = (abs(aspect - bw/bh), abs(bw * bh - base_area))
            if score < best_score:
                best_score = score
                best = (bw, bh)
        return best


class ImageDataset(Dataset):
    """
    图像数据集
    
    支持两种 caption 格式：
    1. JSON 文件（优先）- 支持分类 shuffle
    2. TXT 文件（回退）- 传统 shuffle
    """
    # 保持与 studio/datasets.py:IMAGE_EXTS 同步（anima_train.py 是独立 CLI 脚本，
    # 不强制 import studio package；改一处时另一处也要跟着改）。
    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

    def __init__(self, data_dir, resolution=1024, bucket_mgr=None,
                 shuffle_caption=False, keep_tokens=0, flip_augment=False,
                 tag_dropout=0.0, prefer_json=True, caption_override=None,
                 resolutions=None, aspect_ratio_limit=2.0):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        # 多分辨率：bucket_mgr 是 base 分辨率的 manager（向后兼容，单一 ARB 路径仍走它，
        # 含 None→方桶语义）。非 base 分辨率（文件夹 px 覆盖 / config 列表的其它档）的
        # manager 由 _bucket_mgr_for 按需建在 bucket_mgrs 里。每个样本带 target_reso
        # 决定走哪套桶；不指定 target_reso（或 == base）时走 bucket_mgr。
        self.bucket_mgr = bucket_mgr
        self.aspect_ratio_limit = aspect_ratio_limit
        self.resolutions = [int(r) for r in resolutions] if resolutions else [resolution]
        self.bucket_mgrs = {}
        self.shuffle_caption = shuffle_caption
        self.keep_tokens = keep_tokens
        self.flip_augment = flip_augment
        self.tag_dropout = tag_dropout
        self.prefer_json = prefer_json
        self.caption_override = caption_override  # 正则集：统一 caption，如 "1girl, solo"
        
        # 尝试导入 caption_utils（直接导入避开 __init__.py）
        self.caption_utils = None
        if prefer_json:
            try:
                import importlib.util
                import sys
                
                # 直接加载 caption_utils.py（ADR 0003 PR-A 后 utils/ 在仓库根，
                # 不在 runtime/utils/；__file__ 是 runtime/training/dataset.py，
                # 因此要回溯三层 parent 到仓库根。）
                utils_path = Path(__file__).parent.parent.parent / "utils" / "caption_utils.py"
                if utils_path.exists():
                    spec = importlib.util.spec_from_file_location("caption_utils", utils_path)
                    caption_module = importlib.util.module_from_spec(spec)
                    sys.modules["caption_utils"] = caption_module
                    spec.loader.exec_module(caption_module)
                    
                    self.caption_utils = {
                        "load_and_build": caption_module.load_and_build_caption,
                        "load_json": caption_module.load_caption_json,
                        "normalize": caption_module.normalize_caption_json,
                        "build": caption_module.build_caption_from_json,
                    }
                    logger.info("JSON caption 模式已启用（分类 shuffle）")
                else:
                    logger.warning(f"caption_utils.py 未找到: {utils_path}")
            except Exception as e:
                logger.warning(f"caption_utils 加载失败: {e}，回退到 TXT 模式")
        
        self.samples = self._scan()
        json_count = sum(1 for s in self.samples if s.get("json_path"))
        txt_count = len(self.samples) - json_count
        unique_count = len(set(id(s) for s in self.samples))
        logger.info(f"数据集: {unique_count} 张图 → {len(self.samples)} 样本（含 repeat）(JSON: {json_count}, TXT: {txt_count})")
        self._preflight_json_captions()
        self.bucket_for_index = self._build_bucket_for_index()

    def _build_bucket_for_index(self):
        """预扫每张图尺寸，算出每个样本的桶 (tw, th)，供 BucketBatchSampler 按桶分批。

        非缓存路径必需：``collate_fn`` 用 ``torch.stack`` 拼一个 batch 的 pixel_values，
        若 batch 混入不同桶尺寸会崩；``BucketBatchSampler`` 靠 ``dataset.bucket_for_index``
        把同尺寸样本分进同一 batch。缓存路径不读这份（``CachedLatentDataset`` 从 npz
        latent shape 自建一份并作为外层 wrapper 暴露），但这份也很便宜（只读图片 header）。

        多分辨率：每个样本按其 ``target_reso`` 走对应 manager（``_bucket_mgr_for``），同一
        图在不同 reso 落不同桶，故按 ``(图, target_reso)`` 去重；图只 open 一次复用尺寸。
        某档无 manager（base 档且 ``bucket_mgr=None`` 即不分桶）→ None → sampler 退回普通切批。
        """
        from PIL import Image
        dims: dict = {}     # 图路径 → (w, h)
        by_key: dict = {}   # (图路径, target_reso) → 桶 (tw, th)
        out = []
        for s in self.samples:
            target_reso = s.get("target_reso")
            mgr = self._bucket_mgr_for(target_reso)
            if mgr is None:
                out.append(None)
                continue
            path = str(s["image"])
            key = (path, target_reso)
            if key not in by_key:
                if path not in dims:
                    try:
                        with Image.open(s["image"]) as im:
                            dims[path] = (im.width, im.height)
                    except Exception:
                        dims[path] = None
                wh = dims[path]
                by_key[key] = mgr.get_bucket(*wh) if wh else None
            out.append(by_key[key])
        return out

    @staticmethod
    def _parse_folder_meta(name: str) -> tuple[int | None, int, str]:
        """解析文件夹名 ``[Npx_][R_]label`` → ``(reso_override, repeat, label)``。

        token 顺序（均可选）：``\\d+px`` 分辨率前缀 → ``\\d+`` repeat 前缀 → 其余为 label。

        - ``1024px_2_data`` → ``(1024, 2, 'data')``
        - ``768px_concept`` → ``(768, 1, 'concept')``
        - ``1024px_data``   → ``(1024, 1, 'data')``
        - ``5_concept``（Kohya 风格，向后兼容）→ ``(None, 5, 'concept')``
        - ``concept``       → ``(None, 1, 'concept')``

        分辨率值 snap 到最近的 64 倍数（half-up）并 clamp 到 ``[256, 4096]``（与 schema
        validator 和前端 ``Math.round`` 一致，避免偏心桶 / 跨语言取整分歧）。
        SYNC WITH ``studio/web/src/lib/folderMeta.ts`` 的 ``parseFolderMeta``——两处解析必须一致。
        """
        reso: int | None = None
        repeat = 1
        rest = name
        m = re.match(r"^(\d+)px_(.*)$", rest)
        if m:
            raw = int(m.group(1))
            reso = max(256, min(4096, (raw + 32) // 64 * 64))  # round-half-up，对齐 JS Math.round
            rest = m.group(2)
        m = re.match(r"^(\d+)_(.*)$", rest)
        if m:
            repeat = max(int(m.group(1)), 1)
            rest = m.group(2)
        return reso, repeat, rest

    @staticmethod
    def _parse_repeats_from_dir(name: str) -> int:
        """从文件夹名解析 Kohya 风格重复次数，如 '5_concept' → 5（兼容旧调用）。"""
        return ImageDataset._parse_folder_meta(name)[1]

    def _bucket_mgr_for(self, reso):
        """取 reso 对应的 BucketManager。

        base 分辨率（reso 为 None 或 == self.resolution）走 self.bucket_mgr —— 保持
        旧路径不变（含 bucket_mgr=None → 方桶）。其它分辨率按需建 manager 缓存进
        bucket_mgrs。
        """
        if reso is None or reso == self.resolution:
            return self.bucket_mgr
        mgr = self.bucket_mgrs.get(reso)
        if mgr is None:
            mgr = BucketManager(reso, aspect_ratio_limit=self.aspect_ratio_limit)
            self.bucket_mgrs[reso] = mgr
        return mgr

    def _make_sample(self, img_path):
        """为单张图构建 sample dict，找不到 caption 返回 None"""
        sample = {"image": img_path}
        json_path = img_path.with_suffix(".json")
        if self.prefer_json and json_path.exists():
            sample["json_path"] = json_path
            sample["txt_path"] = None
        else:
            txt_path = img_path.with_suffix(".txt")
            if not txt_path.exists():
                txt_path = img_path.with_suffix(".caption")
            if not txt_path.exists():
                return None
            sample["json_path"] = None
            sample["txt_path"] = txt_path
        return sample

    def _scan(self):
        """扫描数据集目录，支持 Kohya 风格 repeat + 多分辨率。

        目录名 ``[Npx_][R_]label``::

            dataset/
            ├── 5_new/          ← repeat 5，用 config 的 resolutions
            ├── 1024px_2_hires/ ← repeat 2，固定 1024（覆盖列表，不 fan-out）
            └── old/            ← repeat 1，用 config 的 resolutions

        - 带 ``Npx_`` 前缀 → 该文件夹固定用 N 分辨率，覆盖 config 列表、不 fan-out。
        - 无 px 前缀 → 用 ``self.resolutions``；列表多于一档时每张图在每档各一份（fan-out）。

        每张唯一图展开成 ``repeat × 该文件夹分辨率数`` 个样本，每个带 ``target_reso``。
        """
        unique = []  # (sample_dict, repeat, resos)
        folder_info = []  # (name, repeat, resos, count) for logging

        # 根目录图片（repeat=1，无 px → 用 resolutions）
        root_count = 0
        for p in sorted(self.data_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in self.EXTS:
                s = self._make_sample(p)
                if s:
                    unique.append((s, 1, self.resolutions))
                    root_count += 1
        if root_count:
            folder_info.append(("(root)", 1, self.resolutions, root_count))

        # 子文件夹（解析 px 覆盖 + repeat）
        for subdir in sorted(self.data_dir.iterdir()):
            if not subdir.is_dir():
                continue
            reso_override, repeats, _label = self._parse_folder_meta(subdir.name)
            resos = [reso_override] if reso_override else self.resolutions
            count = 0
            for img_path in sorted(subdir.rglob("*")):
                if img_path.suffix.lower() not in self.EXTS:
                    continue
                s = self._make_sample(img_path)
                if s:
                    unique.append((s, repeats, resos))
                    count += 1
            if count:
                folder_info.append((subdir.name, repeats, resos, count))

        # 展开：repeat × 分辨率 fan-out；每个展开样本带 target_reso。
        # 同一 (图, reso) 的 repeat 份共享一个 dict；不同 reso 各自 copy 以带各自 target_reso。
        samples = []
        for s, repeat, resos in unique:
            for target_reso in resos:
                item = dict(s)
                item["target_reso"] = target_reso
                for _ in range(repeat):
                    samples.append(item)

        # 日志：每个文件夹的 repeat × 分辨率
        for name, rep, resos, cnt in folder_info:
            reso_str = "/".join(str(r) for r in resos)
            logger.info(
                f"  文件夹 {name}: {cnt} 张 × repeat {rep} × 分辨率[{reso_str}] "
                f"= {cnt * rep * len(resos)} 样本"
            )

        return samples

    def _preflight_json_captions(self):
        """开训前预检所有 JSON caption，构建失败直接拒绝开训（fail-fast）。

        JSON 样本没有 .txt 兜底（``_make_sample`` 里 prefer_json 命中时
        txt_path=None），caption 构建失败会在 ``__getitem__`` 静默退成空
        caption——连触发词都不剩，整炉 LoRA 白炼且训练照常跑完（#345）。
        与其训练中逐样本 warning 刷屏，不如开训前一次性报清楚并中止。

        shuffle=False + dropout=0 保证预检确定性且不消耗随机数状态。
        caption_override 全局覆盖时不读 caption 文件，跳过。
        """
        if self.caption_override is not None:
            return
        json_paths = []
        seen = set()
        for s in self.samples:
            jp = s.get("json_path")
            if jp and jp not in seen:
                seen.add(jp)
                json_paths.append(jp)
        if not json_paths:
            return
        if self.caption_utils is None:
            raise ValueError(
                f"数据集含 {len(json_paths)} 个 JSON caption，但 caption_utils 加载失败"
                f"（见上方 warning），这些图将以空 caption 训练，已拒绝开训。"
            )
        bad = []
        for jp in json_paths:
            try:
                caption = self.caption_utils["load_and_build"](
                    jp, shuffle=False, tag_dropout=0.0
                )
            except Exception:
                caption = None
            if caption is None:
                bad.append(jp)
        if bad:
            preview = "\n".join(f"  - {p}" for p in bad[:5])
            more = f"\n  ...等共 {len(bad)} 个" if len(bad) > 5 else ""
            raise ValueError(
                f"{len(bad)} 个 JSON caption 解析失败，对应图片将以空 caption"
                f"（连触发词都没有）参与训练，已拒绝开训。"
                f"请在打标页检查或重新打标这些文件：\n{preview}{more}"
            )

    def _process_caption_txt(self, caption):
        """处理 TXT caption：kohya 语义的 keep_tokens + shuffle + tag_dropout。

        keep_tokens 前缀既不参与打乱也不参与 dropout（kohya 同款语义——dropout
        可能丢掉触发词是生态已知行为，保护靠用户显式配 keep_tokens，不做隐式
        按值保护）；其余 tag 先 shuffle 再逐个独立 dropout，无保底。
        """
        if not caption:
            return ""
        if "," in caption:
            tags = [t.strip() for t in caption.split(",")]
        else:
            tags = caption.split()

        kept = tags[:self.keep_tokens]
        rest = tags[self.keep_tokens:]
        if self.shuffle_caption:
            random.shuffle(rest)
        if self.tag_dropout > 0:
            rest = [t for t in rest if random.random() > self.tag_dropout]

        return ", ".join(kept + rest)

    def _process_caption_json(self, json_path):
        """处理 JSON caption: 分类 shuffle"""
        if self.caption_utils is None:
            return None

        try:
            # 走 caption_utils 的权威编排（load → 判断标准格式 → normalize → build）。
            # 早期这里 copy 了一份判断，用 `"tags" in raw_json` 只查 key 是否存在，会把
            # Studio 打标写出的简化形式 {"tags": [list], "meta": {trigger}} 误判为标准
            # 格式直接喂给 build，导致 build 对 list 调 .get() 崩（#345）。load_and_build
            # 用 isinstance(tags, dict) 正确判断：list 形式走 normalize 搬到 tags.tags，
            # 复用单一源避免逻辑再次漂移。
            return self.caption_utils["load_and_build"](
                json_path,
                shuffle=self.shuffle_caption,
                tag_dropout=self.tag_dropout,
            )
        except Exception as e:
            logger.warning(f"JSON 处理失败 {json_path}: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 默认 path：DataLoader 不能传额外参数，所以由 flip_augment 决定是否随机翻转。
        # CachedLatentDataset 想显式控制 flip 时直接调 get_with_flip(idx, flip=...)，
        # 在 cache 阶段对每张图各 encode 一次 flip=False / flip=True，避免随机性 baked
        # 进 npz（kohya 风格双份 latent）。
        flip = self.flip_augment and random.random() > 0.5
        return self.get_with_flip(idx, flip=flip)

    def get_with_flip(self, idx, *, flip: bool):
        """带显式 flip 控制的 __getitem__。

        flip=True/False：强制翻 / 不翻，调用方负责决策；用于 cache 双份编码。
        flip 与 self.flip_augment 解耦，不读 self.flip_augment 也不掷随机数。
        """
        import numpy as np
        from PIL import Image
        sample = self.samples[idx]
        img = Image.open(sample["image"]).convert("RGB")

        # 获取 caption（正则集可用 caption_override 统一覆盖）
        caption = None
        if self.caption_override is not None:
            caption = self.caption_override
        elif sample.get("json_path"):
            caption = self._process_caption_json(sample["json_path"])

        if caption is None and sample.get("txt_path"):
            caption = sample["txt_path"].read_text(encoding="utf-8").strip()
            caption = self._process_caption_txt(caption)

        if caption is None:
            caption = ""

        # ARB 分桶（按样本 target_reso 选对应 manager；base 档走 self.bucket_mgr）
        target_reso = sample.get("target_reso")
        mgr = self._bucket_mgr_for(target_reso)
        if mgr:
            tw, th = mgr.get_bucket(img.width, img.height)
        else:
            tw = th = target_reso or self.resolution

        # 缩放裁剪
        scale = max(tw / img.width, th / img.height)
        nw, nh = int(img.width * scale), int(img.height * scale)
        img = img.resize((nw, nh), Image.LANCZOS)

        left = (nw - tw) // 2
        top = (nh - th) // 2
        img = img.crop((left, top, left + tw, top + th))

        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 转 tensor [-1, 1]
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        return {"pixel_values": tensor, "caption": caption}


class RepeatDataset(Dataset):
    """Kohya 风格数据集重复"""
    def __init__(self, dataset, repeats=1):
        self.dataset = dataset
        self.repeats = max(1, int(repeats))

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


class MergedDataset(Dataset):
    """合并主数据集与正则数据集（Kohya 风格 reg）"""
    def __init__(self, main_dataset, reg_dataset, reg_weight: float = 1.0):
        self.main_dataset = main_dataset
        self.reg_dataset = reg_dataset
        self.reg_weight = float(reg_weight)
        self._main_len = len(main_dataset)
        self._reg_len = len(reg_dataset)

        # 为 BucketBatchSampler 构建 bucket_for_index
        self.bucket_for_index = self._build_bucket_for_index()

    def _get_cached_dataset(self, d):
        if hasattr(d, "bucket_for_index"):
            return d
        if hasattr(d, "dataset"):
            return self._get_cached_dataset(d.dataset)
        return None

    def _build_bucket_for_index(self):
        main_cached = self._get_cached_dataset(self.main_dataset)
        reg_cached = self._get_cached_dataset(self.reg_dataset)
        buckets = []
        if main_cached and main_cached.bucket_for_index:
            main_base_len = len(main_cached.bucket_for_index)
            for idx in range(self._main_len):
                b = main_cached.bucket_for_index[idx % main_base_len]
                buckets.append(b if b is not None else (0, 0))
        else:
            buckets.extend([(0, 0)] * self._main_len)
        if reg_cached and reg_cached.bucket_for_index:
            reg_base_len = len(reg_cached.bucket_for_index)
            for idx in range(self._reg_len):
                b = reg_cached.bucket_for_index[idx % reg_base_len]
                buckets.append(b if b is not None else (0, 0))
        else:
            buckets.extend([(0, 0)] * self._reg_len)
        return buckets

    def __len__(self):
        return self._main_len + self._reg_len

    def __getitem__(self, idx):
        if idx < self._main_len:
            item = self.main_dataset[idx]
            item["loss_weight"] = 1.0
            item["is_reg"] = False
            return item
        item = self.reg_dataset[idx - self._main_len]
        item["loss_weight"] = self.reg_weight
        item["is_reg"] = True
        return item


class BucketBatchSampler:
    """Batch sampler that groups samples by bucket so latents in each batch have the same size."""
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self._cached_dataset = self._get_cached_dataset(dataset)
        self._base_len = len(self._cached_dataset) if self._cached_dataset else 0

    def _get_cached_dataset(self, d):
        if hasattr(d, "bucket_for_index"):
            return d
        if hasattr(d, "dataset"):
            return self._get_cached_dataset(d.dataset)
        return None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        # ARB 下实际 batch 数 = Σ_bucket f(n_b, bs)；用全局 n 会偏（每桶各自有零头）。
        # 没有桶信息时退回到全局公式（线性 DataLoader 行为）。
        if self._cached_dataset is None:
            n = len(self.dataset)
            if self.drop_last:
                return n // self.batch_size
            return (n + self.batch_size - 1) // self.batch_size
        counts = {}
        for idx in range(len(self.dataset)):
            base_idx = idx % self._base_len
            bucket = self._cached_dataset.bucket_for_index[base_idx]
            if bucket is None:
                bucket = (0, 0)
            counts[bucket] = counts.get(bucket, 0) + 1
        total = 0
        for n in counts.values():
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        if self._cached_dataset is None:
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch
            return

        bucket_to_indices = {}
        for idx in range(len(self.dataset)):
            base_idx = idx % self._base_len
            bucket = self._cached_dataset.bucket_for_index[base_idx]
            if bucket is None:
                bucket = (0, 0)
            bucket_to_indices.setdefault(bucket, []).append(idx)

        buckets = list(bucket_to_indices.keys())
        if self.shuffle:
            rng.shuffle(buckets)
        for bucket in buckets:
            indices = bucket_to_indices[bucket]
            if self.shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch


# ============================================================ 分块 VAE encode
# cache_encode_tiled：把超大图按像素块切、逐块 encode 后在 latent 网格羽化拼接。
# 峰值显存从 ∝ 整图像素降到 ∝ 单块像素。

_CACHE_ENCODE_MAX_PIXELS = 4 * 1024 * 1024


class CachedLatentDataset(Dataset):
    """Kohya 风格 npz 文件缓存的数据集。

    flip_augment + cache_latents 同开时按 kohya 双份 latent 模式：
      - cache 阶段对每张图 encode 两次（flip=False / flip=True），分别存到
        npz 的 `latent` / `latent_flipped` 键
      - 训练时 __getitem__ 50% 概率取 flipped 版本
    旧版本静默把"cache 阶段那次随机翻转"baked 进 npz，导致 flip 永久失效 +
    50% 数据被永久镜像污染；新版通过 _is_cache_valid 检测缺 latent_flipped
    键，自动重 encode 修复。
    """
    def __init__(self, base_dataset, vae, device, dtype, cache_dir=None, cache_batch_size=1,
                 encode_tiled=False, encode_tile_px=1024, encode_tile_overlap=128,
                 encode_max_pixels=0):
        import numpy as np
        self.base_dataset = base_dataset
        self.base_image_dataset = self._get_base_image_dataset(base_dataset)
        self.np = np
        # 获取原始数据集的 samples 列表
        self.samples = self._get_base_samples(base_dataset)
        # 同一张图 fan-out 到多个分辨率时 npz 必须分文件（否则不同分辨率 latent 互相
        # 覆盖）。只有真出现在 >1 个 target_reso 的图走 r{reso} 命名；单分辨率图保持
        # img.npz，不动现有缓存。
        _resos_per_img: dict[str, set] = {}
        for s in self.samples:
            _resos_per_img.setdefault(str(s["image"]), set()).add(s.get("target_reso"))
        self._multi_reso = {img for img, rs in _resos_per_img.items() if len(rs) > 1}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.bucket_for_index = []
        self.token_count_for_index = []  # NaViT 打包器读（默认空，由 _fill 填充）
        self.cache_batch_size = max(1, int(cache_batch_size or 1))
        # cache 是否需要双份 latent —— 取决于底层 ImageDataset.flip_augment
        self.flip_augment = bool(
            getattr(self.base_image_dataset, "flip_augment", False)
        )
        # cache_encode_tiled（opt-in）：超大图改走分块 encode + latent 羽化拼接，
        # 峰值显存 ∝ 单块像素。阈值内的图路径不变（逐字节等价）。
        self.encode_tiled = bool(encode_tiled)
        self.encode_tile_px = int(encode_tile_px or 1024)
        self.encode_tile_overlap = int(encode_tile_overlap or 128)
        self.encode_max_pixels = (
            int(encode_max_pixels) if int(encode_max_pixels or 0) > 0
            else _CACHE_ENCODE_MAX_PIXELS
        )
        self._build_cache(vae, device, dtype)

    def _get_base_samples(self, dataset):
        """获取原始 ImageDataset 的 samples"""
        if hasattr(dataset, "samples"):
            return dataset.samples
        elif hasattr(dataset, "dataset"):
            return self._get_base_samples(dataset.dataset)
        return []

    def _get_base_image_dataset(self, dataset):
        if hasattr(dataset, "samples") and hasattr(dataset, "bucket_mgr"):
            return dataset
        if hasattr(dataset, "dataset"):
            return self._get_base_image_dataset(dataset.dataset)
        return None

    def _expected_bucket_size(self, img_path, target_reso=None):
        base = self.base_image_dataset
        if base is None:
            return None
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                if hasattr(base, "_bucket_mgr_for"):
                    mgr = base._bucket_mgr_for(target_reso)
                else:
                    mgr = getattr(base, "bucket_mgr", None)
                if mgr:
                    return mgr.get_bucket(img.width, img.height)
                resolution = int(target_reso or getattr(base, "resolution"))
                return (resolution, resolution)
        except Exception:
            return None

    def _get_npz_path(self, img_path, target_reso=None):
        """图像对应的 npz 缓存路径。

        单分辨率图 → ``img.npz``（不动现有缓存）；同图 fan-out 到多分辨率 →
        ``img.r{reso}.npz``，避免不同分辨率 latent 互相覆盖。
        """
        img_path = Path(img_path)
        if target_reso is not None and str(img_path) in getattr(self, "_multi_reso", set()):
            return img_path.with_suffix(f".r{int(target_reso)}.npz")
        return img_path.with_suffix(".npz")

    def _is_cache_valid(self, img_path, npz_path, target_reso=None):
        """检查缓存是否有效（图像未修改，且格式兼容当前 flip_augment 设置）。

        - 缺 `latent` 键 / 其他模型的不兼容缓存 → 删除重 encode
        - flip_augment=True 且 npz 缺 `latent_flipped` 键 → 失效重 encode（旧
          单份 cache 即"flip 永久 baked"的污染状态，必须重 encode 修复）
        - flip_augment=False 且 npz 有 `latent_flipped` → 仍视为有效（双份
          cache 是 flip 模式的超集，关 flip 后只读 latent 不浪费）
        - bucket 尺寸不匹配 → 失效
        """
        if not npz_path.exists():
            return False
        if npz_path.stat().st_mtime < img_path.stat().st_mtime:
            return False
        try:
            with self.np.load(npz_path) as data:
                if "latent" not in data.files:
                    npz_path.unlink()
                    logger.debug(f"已删除不兼容缓存: {npz_path.name}")
                    return False
                if getattr(self, "flip_augment", False) and "latent_flipped" not in data.files:
                    return False
                expected_bucket = self._expected_bucket_size(img_path, target_reso)
                if expected_bucket is not None:
                    if "bucket_w" not in data.files or "bucket_h" not in data.files:
                        return False
                    if (int(data["bucket_w"]), int(data["bucket_h"])) != expected_bucket:
                        return False
        except Exception:
            try:
                npz_path.unlink()
            except Exception:
                pass
            return False
        return True

    def _build_cache(self, vae, device, dtype):
        """构建/加载 npz 缓存。

        per-folder repeat（5_concept 前缀）让 samples 里同一张图重复 N 次；多分辨率
        fan-out 还让同一张图带不同 target_reso 出现多次。npz 落点由
        `_get_npz_path(img, target_reso)` 决定 —— 单分辨率图用 `img.npz`，fan-out 到多
        分辨率的图用 `img.r{reso}.npz` 分文件。按 npz_path 去重，每个 (图, reso) 最多
        encode 一次；否则同 npz 会被反复覆盖写 N 次（flip_augment 模式下再乘 2）。
        """
        logger.info("检查 VAE latent 缓存...")
        to_encode = []
        seen_npz = set()
        unique_total = 0
        for i, sample in enumerate(self.samples):
            img_path = sample["image"]
            npz_path = self._get_npz_path(
                img_path, sample.get("target_reso"))
            if npz_path in seen_npz:
                continue
            seen_npz.add(npz_path)
            unique_total += 1
            if not self._is_cache_valid(img_path, npz_path, sample.get("target_reso")):
                to_encode.append(i)

        if to_encode:
            logger.info(f"需要编码 {len(to_encode)}/{unique_total} 张图像...")
            self._encode_and_save(to_encode, vae, device, dtype)
        else:
            logger.info(f"所有 {unique_total} 张图像已缓存")

        self._fill_bucket_for_index()

    def _fill_bucket_for_index(self):
        """Fill bucket_for_index for all samples (needed for BucketBatchSampler).
        Uses latent spatial shape (h, w) as grouping key so batches have consistent tensor sizes.

        Also fills token_count_for_index (NaViT packer reads it; patch_spatial=2)."""
        self.bucket_for_index = [None] * len(self.samples)
        self.token_count_for_index = [0] * len(self.samples)
        patch_spatial = 2
        for i in range(len(self.samples)):
            npz_path = self._get_npz_path(
                self.samples[i]["image"], self.samples[i].get("target_reso"))
            if not npz_path.exists():
                continue
            with self.np.load(npz_path) as data:
                latent = data["latent"]
                s = latent.shape
            if len(s) == 5:
                _, _, _, h, w = s
            else:
                _, _, h, w = s
            self.bucket_for_index[i] = (int(h), int(w))
            self.token_count_for_index[i] = (int(h) // patch_spatial) * (int(w) // patch_spatial)

    def _encode_and_save(self, indices, vae, device, dtype):
        """编码图像并保存为 npz。

        flip_augment=True 时对每张图编码两次（flip=False / flip=True）分别存到
        `latent` / `latent_flipped` 键；训练时 __getitem__ 随机选其一。
        flip_augment=False 时只编码一次，存 `latent`。

        按实际 bucket 尺寸分组并批量送入 VAE；不同尺寸不能 stack，分别攒批。
        cache_encode_tiled=True 时，超像素预算的图改走分块 encode + latent 羽化拼接。
        """
        base_img = self.base_image_dataset
        want_flip = self.flip_augment and base_img is not None
        pending = {}
        encoded_count = 0

        def _encode_pixels(pixel_tensors):
            pixels = torch.stack(pixel_tensors, dim=0).to(device, dtype=dtype)
            with torch.inference_mode():
                # 走 VAEWrapper.encode（含 auto/on 分块），大图/大 batch 不会撞 VRAM 崖
                latents = vae.encode(pixels.unsqueeze(2))
            return latents.detach().cpu().float()

        def _encode_tiled_single(pixel_tensor):
            """分块 encode 单张图（cache_encode_tiled 超像素预算时）。"""
            pixels = pixel_tensor.unsqueeze(0).to(device, dtype=dtype).unsqueeze(2)  # [1,C,1,H,W]
            with torch.inference_mode():
                # 直接用 VAEWrapper 的分块 encode（可配 tile 尺寸）：单层分块 + 统一
                # cosine 羽化，避免外层再套一层 vae.encode 导致的双重分块。
                lat = vae._tiled_encode(
                    pixels, self.encode_tile_px, self.encode_tile_overlap
                )
            return lat.detach().cpu().float()[0]

        def _flush(bucket_key):
            nonlocal encoded_count
            batch = pending.pop(bucket_key, [])
            if not batch:
                return

            h, w = int(batch[0]["bucket_h"]), int(batch[0]["bucket_w"])
            use_tiled = (
                getattr(self, "encode_tiled", False)
                and h > 0 and w > 0
                and h * w > self.encode_max_pixels
            )

            if use_tiled:
                logger.info(
                    "[cache-tiled] %dx%d 超像素预算，分块 encode（tile=%d overlap=%d）",
                    w, h, self.encode_tile_px, self.encode_tile_overlap,
                )
                for entry in batch:
                    lat = _encode_tiled_single(entry["pixels"])
                    lat_f = _encode_tiled_single(entry["pixels_flipped"]) if want_flip else None
                    npz_kwargs = {"latent": lat.numpy()}
                    if lat_f is not None:
                        npz_kwargs["latent_flipped"] = lat_f.numpy()
                    _entry_sample = self.samples[entry["index"]]
                    npz_path = self._get_npz_path(
                        _entry_sample["image"], _entry_sample.get("target_reso"))
                    self.np.savez(
                        npz_path,
                        bucket_w=entry["bucket_w"],
                        bucket_h=entry["bucket_h"],
                        **npz_kwargs,
                    )
                    encoded_count += 1
                    if encoded_count % 10 == 0 or encoded_count == len(indices):
                        logger.info(f"  编码进度: {encoded_count}/{len(indices)}")
                return

            latents = _encode_pixels([entry["pixels"] for entry in batch])
            if want_flip:
                latents_flipped = _encode_pixels([entry["pixels_flipped"] for entry in batch])
            else:
                latents_flipped = [None] * len(batch)

            for n, entry in enumerate(batch):
                npz_kwargs = {"latent": latents[n].numpy()}
                if want_flip:
                    npz_kwargs["latent_flipped"] = latents_flipped[n].numpy()

                _entry_sample = self.samples[entry["index"]]
                npz_path = self._get_npz_path(
                    _entry_sample["image"], _entry_sample.get("target_reso"))
                self.np.savez(
                    npz_path,
                    bucket_w=entry["bucket_w"],
                    bucket_h=entry["bucket_h"],
                    **npz_kwargs,
                )
                encoded_count += 1
                if encoded_count % 10 == 0 or encoded_count == len(indices):
                    logger.info(f"  编码进度: {encoded_count}/{len(indices)}")

        logger.info(f"VAE cache batch size: {self.cache_batch_size}")
        for i in indices:
            if base_img is not None:
                # 显式控制 flip，避免随机性 baked 进 npz
                item = base_img.get_with_flip(i, flip=False)
            else:
                item = self.base_dataset[i]
            pixels = item["pixel_values"]
            _, ph, pw = pixels.shape
            bucket_w, bucket_h = pw, ph

            pixels_flipped = None
            if want_flip:
                item_f = base_img.get_with_flip(i, flip=True)
                pixels_flipped = item_f["pixel_values"]

            bucket_key = (bucket_h, bucket_w)
            pending.setdefault(bucket_key, []).append({
                "index": i,
                "pixels": pixels,
                "pixels_flipped": pixels_flipped,
                "bucket_w": bucket_w,
                "bucket_h": bucket_h,
            })
            if len(pending[bucket_key]) >= self.cache_batch_size:
                _flush(bucket_key)

        for bucket_key in list(pending):
            _flush(bucket_key)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        npz_path = self._get_npz_path(
            sample["image"], sample.get("target_reso"))
        data = self.np.load(npz_path)
        # flip_augment=True 且 npz 有 latent_flipped 时 50% 概率取镜像版本，
        # 跟非 cache 路径 ImageDataset.__getitem__ 的 flip 概率一致。
        # 没有 latent_flipped 键（flip_augment=False 时的单份 cache）就只读 latent。
        use_flip = (
            self.flip_augment
            and "latent_flipped" in data.files
            and random.random() > 0.5
        )
        latent_key = "latent_flipped" if use_flip else "latent"
        latent = torch.from_numpy(data[latent_key])

        # 获取 base_dataset 的引用（处理可能的嵌套）
        base = self.base_dataset
        while hasattr(base, "dataset"):
            base = base.dataset
        
        # 处理 caption（正则集 caption_override 优先）
        caption = None
        if getattr(base, "caption_override", None) is not None:
            caption = base.caption_override
        elif sample.get("json_path") and hasattr(base, "_process_caption_json"):
            caption = base._process_caption_json(sample["json_path"])
        
        if caption is None and sample.get("txt_path"):
            caption = sample["txt_path"].read_text(encoding="utf-8").strip()
            if hasattr(base, "_process_caption_txt"):
                caption = base._process_caption_txt(caption)
        
        if caption is None:
            caption = ""
        
        return {
            "latent": latent,
            "caption": caption,
            # navit collate 需要逐图 image 路径
            "image": str(sample["image"]),
        }


def collate_fn(batch):
    """DataLoader collate"""
    pixels = torch.stack([b["pixel_values"] for b in batch])
    captions = [b["caption"] for b in batch]
    result = {"pixel_values": pixels, "captions": captions}
    if "loss_weight" in batch[0]:
        result["loss_weight"] = torch.tensor([b["loss_weight"] for b in batch], dtype=torch.float32)
        result["is_reg"] = torch.tensor([b["is_reg"] for b in batch], dtype=torch.bool)
    return result


def collate_fn_cached(batch):
    """DataLoader collate for cached latents"""
    latents = torch.stack([b["latent"] for b in batch])
    captions = [b["caption"] for b in batch]
    result = {"latents": latents, "captions": captions}
    if "loss_weight" in batch[0]:
        result["loss_weight"] = torch.tensor([b["loss_weight"] for b in batch], dtype=torch.float32)
        result["is_reg"] = torch.tensor([b["is_reg"] for b in batch], dtype=torch.bool)
    return result


# =================================================== NaViT / Patch-n-Pack 打包
# token 预算打包器 + 块对角 collate：把不同 token 数的图拼进一个训练序列（零 padding）。


def pack_indices_by_budget(token_counts, token_budget, order, max_images_per_pack=0):
    """贪心 next-fit 打包：把样本索引分进 token 总数 ≤ budget 的包。

    NaViT 块对角打包无 padding，一个包的代价 = 各图 token 数之和。``order`` 是
    已打乱的索引序列；自身 token 数超 budget 的图单独成包（调用方 warn）。
    结果覆盖 ``order`` 中每个索引恰好一次，保持顺序。
    """
    packs = []
    cur, cur_sum = [], 0
    cap = int(max_images_per_pack or 0)
    budget = int(token_budget)
    for idx in order:
        n = int(token_counts[idx])
        over_budget = bool(cur) and (cur_sum + n > budget)
        over_count = cap > 0 and len(cur) >= cap
        if over_budget or over_count:
            packs.append(cur)
            cur, cur_sum = [], 0
        cur.append(idx)
        cur_sum += n
    if cur:
        packs.append(cur)
    return packs


def pack_indices_ffd_windowed(token_counts, token_budget, order,
                              max_images_per_pack=0, window=0):
    """First-Fit-Decreasing 窗口化打包：在（已打乱的）``order`` 的窗口内做 FFD。

    经典 FFD（按尺寸降序，逐个放入第一个能放下的桶）比 next-fit 打包更紧——更少、
    更满的包 ⇒ 更少 optimizer step、更少浪费的 token 预算（见 NeMo sequence-packing /
    ICLR'23 "Efficient Sequence Packing"）。代价：全局降序排序会让每 epoch 把相同图
    分到一起（尺寸顺序固定），削弱小数据 SGD 的 batch 多样性。

    解决方案是 ``window``：``order`` 被分成 ``window`` 大小的连续窗口，FFD 在每个
    窗口内运行。因 ``order`` 每 epoch 重新打乱，窗口成员（及分组）跨 epoch 变化，
    而窗口内降序排序仍恢复大部分填充收益。``window<=0`` 表示一个全局窗口（最大填充，
    但每 epoch 包固定——仅适合单 pass 数据）。

    覆盖 ``order`` 中每个索引恰好一次。自身超 budget 的图单独成包（同 next-fit）。
    """
    budget = int(token_budget)
    cap = int(max_images_per_pack or 0)
    win = int(window or 0)
    order = list(order)
    if win <= 0:
        windows = [order]
    else:
        windows = [order[i:i + win] for i in range(0, len(order), win)]

    packs = []
    for w in windows:
        items = sorted(w, key=lambda i: int(token_counts[i]), reverse=True)
        bins = []  # each: [list_of_indices, summed_tokens]
        for idx in items:
            n = int(token_counts[idx])
            placed = False
            for b in bins:
                over_count = cap > 0 and len(b[0]) >= cap
                if (not over_count) and (b[1] + n <= budget):
                    b[0].append(idx)
                    b[1] += n
                    placed = True
                    break
            if not placed:
                bins.append([[idx], n])
        packs.extend(b[0] for b in bins)
    return packs


def _lookup_token_count_walk(d, idx):
    """通过遍历数据集包装器解析样本的 token 数（NaViT 打包器的自由函数版本）。"""
    main = getattr(d, "main_dataset", None)
    reg = getattr(d, "reg_dataset", None)
    if main is not None and reg is not None:
        ml = getattr(d, "_main_len", len(main))
        if idx < ml:
            return _lookup_token_count_walk(main, idx)
        return _lookup_token_count_walk(reg, idx - ml)
    inner = getattr(d, "dataset", None)
    if inner is not None and inner is not d and not isinstance(inner, list):
        return _lookup_token_count_walk(inner, idx % len(inner))
    counts = getattr(d, "token_count_for_index", None)
    if counts is not None and len(counts) > 0:
        return int(counts[idx % len(counts)])
    inner = getattr(d, "base_dataset", None)
    if inner is not None and inner is not d:
        return _lookup_token_count_walk(inner, idx % len(inner))
    return 0


def _walk_attr_list(dataset, attr):
    """在单链包装器（RepeatDataset/CachedLatentDataset）中查找叶数据集的
    per-index 列表属性 ``attr``，通过 ``% len`` 映射到 ``len(dataset)``。
    对 MergedDataset（两分支）或属性不存在时返回 None。"""
    cur = dataset
    for _ in range(12):
        if getattr(cur, "main_dataset", None) is not None and getattr(cur, "reg_dataset", None) is not None:
            return None  # MergedDataset: not a single chain
        v = getattr(cur, attr, None)
        if v is not None and len(v) > 0:
            n = len(dataset)
            return [v[i % len(v)] for i in range(n)]
        nxt = getattr(cur, "dataset", None)
        if nxt is None or nxt is cur or isinstance(nxt, list):
            nxt = getattr(cur, "base_dataset", None)
        if nxt is None or nxt is cur:
            return None
        cur = nxt
    return None


def dataset_token_counts(dataset, patch_spatial=2):
    """NaViT 打包的逐索引 token 数。

    优先用已填充的 ``token_count_for_index``（CachedLatentDataset 填充）。若该字段
    全 0 或不存在，则从缓存 latent 形状 ``bucket_for_index = (h, w)``（latent px）
    推导为 ``(h // patch_spatial) * (w // patch_spatial)``——即 patchify 后的 token 数。
    若两者都不可用，逐索引 walk 兜底（返回 0 → 打包器会 fail-fast）。
    """
    counts = _walk_attr_list(dataset, "token_count_for_index")
    if counts is not None and any(int(c) > 0 for c in counts):
        return [int(c) for c in counts]

    shapes = _walk_attr_list(dataset, "bucket_for_index")
    if shapes is not None:
        ps = max(1, int(patch_spatial))
        derived = []
        for s in shapes:
            if not s:
                derived.append(0)
                continue
            h, w = int(s[0]), int(s[1])
            derived.append((h // ps) * (w // ps))
        if any(c > 0 for c in derived):
            return derived

    return [int(_lookup_token_count_walk(dataset, i) or 0) for i in range(len(dataset))]


class NavitPackBatchSampler:
    """为 NaViT/Patch-n-Pack 块对角训练产出数据集索引包。

    每个产出的列表是一个打包训练序列：其各图 token 数之和 ≤ ``token_budget``，
    整包作为一个零 padding 的块对角 forward。把"每步图片数"与单图形状解耦——
    不同 token 数和长宽比的图可以共享一个包，小数据集也能填满大 effective batch。
    """

    def __init__(self, dataset, token_budget, max_images_per_pack=0,
                 shuffle=True, seed=42, drop_last=False,
                 strategy="next_fit", ffd_window=256):
        self.dataset = dataset
        self.token_budget = int(token_budget)
        self.max_images_per_pack = int(max_images_per_pack or 0)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.strategy = str(strategy or "next_fit").lower()
        if self.strategy not in ("next_fit", "ffd"):
            raise ValueError(
                f"navit pack strategy 必须是 'next_fit' 或 'ffd'，收到 {strategy!r}"
            )
        self.ffd_window = int(ffd_window or 0)
        self.epoch = 0
        self.token_counts = dataset_token_counts(dataset)
        self._cached_packs = None
        # Fail-fast：全 0 token 数意味着无法解析每图尺寸（token_count_for_index 与
        # bucket_for_index 都不可用/全 0）。不检查的话 `cur_sum + 0 > budget` 永远
        # 不触发 → 整个数据集打包成一个 ~500k-token 序列 → OOM。
        if not self.token_counts or not any(int(c) > 0 for c in self.token_counts):
            raise RuntimeError(
                "[NavitPack] 无法解析任一样本的 token 数（token_count_for_index 与 "
                "bucket_for_index 都不可用/全 0）。NaViT 打包需要缓存数据集 "
                "（cache_latents=true）以拿到每图 latent 形状。"
            )
        mx = max(self.token_counts) if self.token_counts else 0
        if self.token_counts and self.token_budget < mx:
            logger.warning(
                "[NavitPack] token_budget=%d < 最大单图 token=%d：该图将单独成包，"
                "可能超出预算并 OOM。建议 token_budget >= 最大单图 token。",
                self.token_budget, mx,
            )
        logger.info(
            "[NavitPack] dataset_len=%d token_budget=%d max_images_per_pack=%s "
            "strategy=%s ffd_window=%s (token 数范围 %d..%d)",
            len(self.token_counts), self.token_budget,
            self.max_images_per_pack or "∞", self.strategy,
            (self.ffd_window or "全局") if self.strategy == "ffd" else "-",
            min(self.token_counts) if self.token_counts else 0, mx,
        )
        if self.strategy == "ffd" and self.ffd_window <= 0:
            logger.warning(
                "[NavitPack] strategy=ffd 且 ffd_window<=0（全局 FFD）：每 epoch 的包将完全相同"
                "（按尺寸排序固定），削弱小数据 SGD 的 batch 多样性。多 epoch 训练建议设正窗口。"
            )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._cached_packs = None

    def _build_packs(self):
        order = list(range(len(self.token_counts)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        if self.strategy == "ffd":
            packs = pack_indices_ffd_windowed(
                self.token_counts, self.token_budget, order,
                self.max_images_per_pack, self.ffd_window,
            )
        else:
            packs = pack_indices_by_budget(
                self.token_counts, self.token_budget, order, self.max_images_per_pack
            )
        if self.drop_last and len(packs) > 1:
            last_sum = sum(self.token_counts[i] for i in packs[-1])
            if last_sum < self.token_budget:
                packs = packs[:-1]
        return packs

    def __iter__(self):
        packs = self._build_packs()
        self._cached_packs = packs
        for pack in packs:
            yield pack

    def __len__(self):
        if self._cached_packs is None:
            self._cached_packs = self._build_packs()
        return len(self._cached_packs)


def collate_fn_navit_pack(batch):
    """NaViT 打包 collate。

    一个包内的缓存 latent 有不同的空间形状，无法 stack；保留为列表。训练循环
    将每张图 patchify 为 token，拼接 token 和 per-image RoPE grid，编码 caption 并
    拼接对应的 ``text_seqlens``，然后调用 ``forward_packed_navit``。
    """
    latents = [b["latent"] for b in batch]        # each [C, T, h_i, w_i]
    captions = [b["caption"] for b in batch]
    images = [b.get("image", "") for b in batch]
    result = {
        "navit_latents": latents,
        "captions": captions,
        "images": images,
    }
    # 正则集降权：与 collate_fn_cached 对齐——透传 loss_weight / is_reg 供训练循环
    # 在 per-image loss 上应用（navit 路径同样尊重 batch 的 loss_weight）。
    if "loss_weight" in batch[0]:
        result["loss_weight"] = torch.tensor(
            [b["loss_weight"] for b in batch], dtype=torch.float32
        )
        result["is_reg"] = torch.tensor([b["is_reg"] for b in batch], dtype=torch.bool)
    return result
