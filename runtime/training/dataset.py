"""数据集与 collate：ARB 分桶 + ImageDataset + 正则集 merge + cached latent。

抽自原 runtime/anima_train.py L1144-1675 + L1939-1962（ADR 0003 PR-A）。

公开：
- BucketManager / ImageDataset / RepeatDataset / MergedDataset
- BucketBatchSampler / CachedLatentDataset
- collate_fn / collate_fn_cached — DataLoader collate
"""

from __future__ import annotations

import logging
import random
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
    (``base_reso``, ``min_reso``, ``max_reso``, ``step``, the 0.1 area
    tolerance, the 2.0 AR cap) MUST land in both files in the same commit,
    or the frontend's predicted bucket ≠ trainer's actual bucket and crops
    will silently degrade.

    See ``docs/design/preprocess-crop-design.md`` §7 for the UX policy and
    rationale.
    """
    def __init__(self, base_reso=1024, min_reso=512, max_reso=2048, step=64):
        self.base_reso = base_reso
        self.buckets = self._generate(min_reso, max_reso, step, base_reso)

    def _generate(self, min_r, max_r, step, base):
        # Keep algorithm identical to trainBuckets.generateBuckets() in TS:
        #   - double loop over (w, h) in [min_r, max_r] step `step`
        #   - area within ±10% of base² (the 0.1 below)
        #   - max AR ratio ≤ 2.0 (the 2.0 below)
        # Default-param consumers should see exactly the same 37 buckets on
        # both sides — covered by `studio/web/src/lib/trainBuckets.test.ts`
        # asserting count == 37.
        buckets = []
        base_area = base * base
        for w in range(min_r, max_r + 1, step):
            for h in range(min_r, max_r + 1, step):
                if abs(w * h - base_area) / base_area > 0.1:
                    continue
                if max(w/h, h/w) > 2.0:
                    continue
                buckets.append((w, h))
        return buckets

    def get_bucket(self, w, h):
        # Snap by ABSOLUTE AR distance — not relative. The TS port
        # `trainBuckets.snapToBucket()` mirrors this exactly.
        aspect = w / h
        best = (self.base_reso, self.base_reso)
        best_diff = float("inf")
        for bw, bh in self.buckets:
            diff = abs(aspect - bw/bh)
            if diff < best_diff:
                best_diff = diff
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
                 tag_dropout=0.0, prefer_json=True, caption_override=None):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.bucket_mgr = bucket_mgr
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

    @staticmethod
    def _parse_repeats_from_dir(name: str) -> int:
        """从文件夹名解析 Kohya 风格重复次数，如 '5_concept' → 5"""
        prefix = name.split("_", 1)[0]
        if prefix.isdigit():
            return max(int(prefix), 1)
        return 1

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
        """扫描数据集目录，支持 Kohya 风格文件夹重复。

        目录结构示例::

            dataset/
            ├── 1_old/       ← repeat 1
            │   ├── img.jpg
            │   └── img.txt
            └── 5_new/       ← repeat 5
                ├── img.jpg
                └── img.txt

        没有数字前缀的文件夹或根目录下的图片按 repeat=1 处理。
        """
        unique_samples = []
        folder_info = []  # (folder_name, repeat, count) for logging

        # 收集根目录下的图片（repeat=1）
        root_count = 0
        for p in sorted(self.data_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in self.EXTS:
                s = self._make_sample(p)
                if s:
                    s["_repeat"] = 1
                    unique_samples.append(s)
                    root_count += 1
        if root_count:
            folder_info.append(("(root)", 1, root_count))

        # 收集子文件夹中的图片（解析 repeat）
        for subdir in sorted(self.data_dir.iterdir()):
            if not subdir.is_dir():
                continue
            repeats = self._parse_repeats_from_dir(subdir.name)
            count = 0
            for img_path in sorted(subdir.rglob("*")):
                if img_path.suffix.lower() not in self.EXTS:
                    continue
                s = self._make_sample(img_path)
                if s:
                    s["_repeat"] = repeats
                    unique_samples.append(s)
                    count += 1
            if count:
                folder_info.append((subdir.name, repeats, count))

        # 展开 repeat：将每个样本按其 repeat 次数复制
        samples = []
        for s in unique_samples:
            r = s.pop("_repeat", 1)
            for _ in range(r):
                samples.append(s)

        # 日志：每个文件夹的 repeat 信息
        if folder_info:
            for name, rep, cnt in folder_info:
                logger.info(f"  文件夹 {name}: {cnt} 张 × repeat {rep} = {cnt * rep} 样本")

        return samples

    def _process_caption_txt(self, caption):
        """处理 TXT caption: 传统 tag 打乱 + keep_tokens"""
        if not caption:
            return ""
        if "," in caption:
            tags = [t.strip() for t in caption.split(",")]
        else:
            tags = caption.split()

        if self.keep_tokens > 0:
            kept = tags[:self.keep_tokens]
            rest = tags[self.keep_tokens:]
            if self.shuffle_caption:
                random.shuffle(rest)
            tags = kept + rest
        elif self.shuffle_caption:
            random.shuffle(tags)

        return ", ".join(tags)

    def _process_caption_json(self, json_path):
        """处理 JSON caption: 分类 shuffle"""
        if self.caption_utils is None:
            return None
        
        try:
            raw_json = self.caption_utils["load_json"](json_path)
            if raw_json is None:
                return None
            
            # 检查是否已经是标准格式
            if "tags" in raw_json and "meta" in raw_json:
                normalized = raw_json
            else:
                normalized = self.caption_utils["normalize"](raw_json)
            
            # 构建 caption（分类 shuffle）
            return self.caption_utils["build"](
                normalized,
                shuffle_appearance=self.shuffle_caption,
                shuffle_tags=self.shuffle_caption,
                shuffle_environment=self.shuffle_caption,
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

        # ARB 分桶
        if self.bucket_mgr:
            tw, th = self.bucket_mgr.get_bucket(img.width, img.height)
        else:
            tw = th = self.resolution

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
    def __init__(self, base_dataset, vae, device, dtype, cache_dir=None, cache_batch_size=4):
        import numpy as np
        self.base_dataset = base_dataset
        self.base_image_dataset = self._get_base_image_dataset(base_dataset)
        self.np = np
        # 获取原始数据集的 samples 列表
        self.samples = self._get_base_samples(base_dataset)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.bucket_for_index = []
        self.cache_batch_size = max(1, int(cache_batch_size or 1))
        # cache 是否需要双份 latent —— 取决于底层 ImageDataset.flip_augment
        self.flip_augment = bool(
            getattr(self.base_image_dataset, "flip_augment", False)
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

    def _expected_bucket_size(self, img_path):
        base = self.base_image_dataset
        if base is None:
            return None
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                if getattr(base, "bucket_mgr", None):
                    return base.bucket_mgr.get_bucket(img.width, img.height)
                resolution = int(getattr(base, "resolution"))
                return (resolution, resolution)
        except Exception:
            return None

    def _get_npz_path(self, img_path):
        """获取图像对应的 npz 缓存路径"""
        img_path = Path(img_path)
        return img_path.with_suffix(".npz")

    def _is_cache_valid(self, img_path, npz_path):
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
                expected_bucket = self._expected_bucket_size(img_path)
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

        per-folder repeat（5_concept 前缀）让 ImageDataset.samples 里同一张图重复 N 次，
        但 npz 落点是 img_path.with_suffix(".npz") — 每张唯一图只对应一个 npz。
        按 npz_path 去重，每张图最多 encode 一次；否则同 npz 会被反复覆盖写 N 次
        （flip_augment 模式下再乘 2），首次构建 cache 时 80% 的 VAE encode 都是浪费。
        """
        logger.info("检查 VAE latent 缓存...")
        to_encode = []
        seen_npz = set()
        unique_total = 0
        for i, sample in enumerate(self.samples):
            img_path = sample["image"]
            npz_path = self._get_npz_path(img_path)
            if npz_path in seen_npz:
                continue
            seen_npz.add(npz_path)
            unique_total += 1
            if not self._is_cache_valid(img_path, npz_path):
                to_encode.append(i)

        if to_encode:
            logger.info(f"需要编码 {len(to_encode)}/{unique_total} 张图像...")
            self._encode_and_save(to_encode, vae, device, dtype)
        else:
            logger.info(f"所有 {unique_total} 张图像已缓存")

        self._fill_bucket_for_index()

    def _fill_bucket_for_index(self):
        """Fill bucket_for_index for all samples (needed for BucketBatchSampler).
        Uses latent spatial shape (h, w) as grouping key so batches have consistent tensor sizes."""
        self.bucket_for_index = [None] * len(self.samples)
        for i in range(len(self.samples)):
            npz_path = self._get_npz_path(self.samples[i]["image"])
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

    def _encode_and_save(self, indices, vae, device, dtype):
        """编码图像并保存为 npz。

        flip_augment=True 时对每张图编码两次（flip=False / flip=True）分别存到
        `latent` / `latent_flipped` 键；训练时 __getitem__ 随机选其一。
        flip_augment=False 时只编码一次，存 `latent`。

        按实际 bucket 尺寸分组并批量送入 VAE；不同尺寸不能 stack，分别攒批。
        """
        base_img = self.base_image_dataset
        want_flip = self.flip_augment and base_img is not None
        pending = {}
        encoded_count = 0

        def _encode_pixels(pixel_tensors):
            pixels = torch.stack(pixel_tensors, dim=0).to(device, dtype=dtype)
            with torch.inference_mode():
                latents = vae.model.encode(pixels.unsqueeze(2), vae.scale)
            return latents.detach().cpu().float()

        def _flush(bucket_key):
            nonlocal encoded_count
            batch = pending.pop(bucket_key, [])
            if not batch:
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

                npz_path = self._get_npz_path(self.samples[entry["index"]]["image"])
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
        npz_path = self._get_npz_path(sample["image"])
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
        
        return {"latent": latent, "caption": caption}


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
