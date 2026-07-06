"""数据集扫描 + /api/datasets 端点测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio.services.dataset import scan as datasets


def _touch_image(folder: Path, name: str, size: int = 8) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * size)  # 假 PNG header
    return p


def test_cached_latent_invalidates_when_resolution_bucket_changes(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from PIL import Image
    import numpy as np

    from runtime.training.dataset import BucketManager, CachedLatentDataset, ImageDataset

    img_path = tmp_path / "0001.png"
    Image.new("RGB", (1536, 1536), color=(255, 255, 255)).save(img_path)
    img_path.with_suffix(".txt").write_text("1girl", encoding="utf-8")
    npz_path = img_path.with_suffix(".npz")
    np.savez(npz_path, latent=np.zeros((16, 1, 192, 192), dtype=np.float32), bucket_w=1536, bucket_h=1536)

    bucket_mgr = BucketManager(1440)
    expected_bucket = bucket_mgr.get_bucket(1536, 1536)
    dataset = ImageDataset(tmp_path, 1440, bucket_mgr)
    cached = object.__new__(CachedLatentDataset)
    cached.base_dataset = dataset
    cached.base_image_dataset = dataset
    cached.np = np
    cached.samples = dataset.samples
    cached.cache_dir = None
    cached.bucket_for_index = []

    assert cached._is_cache_valid(img_path, npz_path) is False
    np.savez(
        npz_path,
        latent=np.zeros((16, 1, expected_bucket[1] // 8, expected_bucket[0] // 8), dtype=np.float32),
        bucket_w=expected_bucket[0],
        bucket_h=expected_bucket[1],
    )
    assert cached._is_cache_valid(img_path, npz_path) is True


def test_cached_latent_invalidates_when_flip_augment_added(tmp_path: Path) -> None:
    """老 cache（只有 latent，无 latent_flipped）+ flip_augment=True → 失效重 encode。

    旧版本静默把 cache 阶段那次随机翻转 baked 进 npz，导致 50% 数据被永久镜像
    污染；新版要求 flip_augment=True 时 npz 必须同时有 latent + latent_flipped。
    """
    pytest.importorskip("torch")
    from PIL import Image
    import numpy as np

    from runtime.training.dataset import BucketManager, CachedLatentDataset, ImageDataset

    img_path = tmp_path / "0001.png"
    Image.new("RGB", (1024, 1024), color=(127, 127, 127)).save(img_path)
    img_path.with_suffix(".txt").write_text("1girl", encoding="utf-8")
    npz_path = img_path.with_suffix(".npz")
    # 老格式 cache：只有 latent，无 latent_flipped
    np.savez(
        npz_path,
        latent=np.zeros((16, 1, 128, 128), dtype=np.float32),
        bucket_w=1024,
        bucket_h=1024,
    )

    bucket_mgr = BucketManager(1024)
    dataset = ImageDataset(tmp_path, 1024, bucket_mgr, flip_augment=True)
    cached = object.__new__(CachedLatentDataset)
    cached.base_dataset = dataset
    cached.base_image_dataset = dataset
    cached.np = np
    cached.samples = dataset.samples
    cached.cache_dir = None
    cached.bucket_for_index = []
    cached.flip_augment = True

    # flip_augment=True 但 npz 缺 latent_flipped → 失效（强制重 encode）
    assert cached._is_cache_valid(img_path, npz_path) is False

    # 补全 latent_flipped → 有效
    np.savez(
        npz_path,
        latent=np.zeros((16, 1, 128, 128), dtype=np.float32),
        latent_flipped=np.zeros((16, 1, 128, 128), dtype=np.float32),
        bucket_w=1024,
        bucket_h=1024,
    )
    assert cached._is_cache_valid(img_path, npz_path) is True


def test_cached_latent_accepts_double_cache_with_flip_off(tmp_path: Path) -> None:
    """双份 cache + flip_augment=False → 仍有效（不强制再 encode；只读 latent）。

    避免用户切 flip 开关时反复重 encode；双份是单份的超集。
    """
    pytest.importorskip("torch")
    from PIL import Image
    import numpy as np

    from runtime.training.dataset import BucketManager, CachedLatentDataset, ImageDataset

    img_path = tmp_path / "0001.png"
    Image.new("RGB", (1024, 1024), color=(127, 127, 127)).save(img_path)
    img_path.with_suffix(".txt").write_text("1girl", encoding="utf-8")
    npz_path = img_path.with_suffix(".npz")
    np.savez(
        npz_path,
        latent=np.zeros((16, 1, 128, 128), dtype=np.float32),
        latent_flipped=np.zeros((16, 1, 128, 128), dtype=np.float32),
        bucket_w=1024,
        bucket_h=1024,
    )

    bucket_mgr = BucketManager(1024)
    dataset = ImageDataset(tmp_path, 1024, bucket_mgr, flip_augment=False)
    cached = object.__new__(CachedLatentDataset)
    cached.base_dataset = dataset
    cached.base_image_dataset = dataset
    cached.np = np
    cached.samples = dataset.samples
    cached.cache_dir = None
    cached.bucket_for_index = []
    cached.flip_augment = False

    assert cached._is_cache_valid(img_path, npz_path) is True


class _FakeVAEModel:
    """Mock VAE：encode 把 pixel mean / first-pixel signature 写进 latent，
    让测试能区分『原图 latent』vs『flipped latent』。"""
    def encode(self, pixels_5d, scale):
        import torch
        # pixels_5d: [B, C, T=1, H, W]
        b, c, t, h, w = pixels_5d.shape
        # 签名：取第一行的第一个像素值 + 最后一个像素值，写进 latent 头两个 channel
        # flip 后这俩会对调 → 区分有无 flip
        first = pixels_5d[:, 0:1, :, :, 0:1].mean(dim=(2, 3, 4), keepdim=True)
        last = pixels_5d[:, 0:1, :, :, -1:].mean(dim=(2, 3, 4), keepdim=True)
        latent = torch.zeros(b, 16, 1, h // 8, w // 8, dtype=pixels_5d.dtype)
        latent[:, 0, 0, 0, 0] = first.squeeze()
        latent[:, 1, 0, 0, 0] = last.squeeze()
        return latent


class _FakeVAE:
    def __init__(self):
        self.model = _FakeVAEModel()
        self.scale = 1.0

    def encode(self, pixels):
        # 镜像 VAEWrapper.encode 的非分块路径（CPU 小图不触发分块）
        return self.model.encode(pixels, self.scale)


def test_cached_latent_encodes_both_flipped_and_unflipped_when_flip_aug(tmp_path: Path) -> None:
    """flip_augment=True → cache 阶段对每张图 encode 两次，npz 同时有 latent + latent_flipped。

    用 mock VAE 把图像第一列/最后一列 mean 编进 latent 签名，验证两份 latent 是
    精确镜像对（latent[0,0,0,0,0] / latent[0,1,0,0,0] 在 flipped 版本里对调）。
    """
    pytest.importorskip("torch")
    import torch
    import numpy as np
    from PIL import Image

    from runtime.training.dataset import BucketManager, CachedLatentDataset, ImageDataset

    img_path = tmp_path / "asym.png"
    # 第一列纯红 (255,0,0)，最后一列纯蓝 (0,0,255)，区分明显
    img = Image.new("RGB", (256, 256), color=(127, 127, 127))
    for y in range(256):
        img.putpixel((0, y), (255, 0, 0))
        img.putpixel((255, y), (0, 0, 255))
    img.save(img_path)
    img_path.with_suffix(".txt").write_text("test", encoding="utf-8")

    bucket_mgr = BucketManager(256, min_reso=256, max_reso=256, step=64)
    dataset = ImageDataset(tmp_path, 256, bucket_mgr, flip_augment=True)
    cached = CachedLatentDataset(dataset, _FakeVAE(), device="cpu", dtype=torch.float32)

    npz_path = img_path.with_suffix(".npz")
    with np.load(npz_path) as data:
        assert "latent" in data.files
        assert "latent_flipped" in data.files
        # 原图：第一列红（pixel→[-1, 1] 范围内 R 通道高），最后一列蓝（B 通道高）
        # mock encode 用 channel-0 mean 当签名；R=1.0/G=B=-1.0 → mean = -1/3
        # 翻转后第一列变蓝、最后一列变红，签名对调
        sig_orig_first = float(data["latent"][0, 0, 0, 0])
        sig_orig_last = float(data["latent"][1, 0, 0, 0])
        sig_flip_first = float(data["latent_flipped"][0, 0, 0, 0])
        sig_flip_last = float(data["latent_flipped"][1, 0, 0, 0])
        # flipped 的 first 应当等于 orig 的 last（左右调换），反之亦然
        assert abs(sig_flip_first - sig_orig_last) < 1e-5
        assert abs(sig_flip_last - sig_orig_first) < 1e-5
        # 且 first ≠ last（确保签名真的有区分性，不是 mock 永远写 0）
        assert abs(sig_orig_first - sig_orig_last) > 1e-3


def test_cached_latent_encodes_single_when_flip_aug_off(tmp_path: Path) -> None:
    """flip_augment=False → npz 只有 latent，不浪费时间编 flipped 版本。"""
    pytest.importorskip("torch")
    import torch
    import numpy as np
    from PIL import Image

    from runtime.training.dataset import BucketManager, CachedLatentDataset, ImageDataset

    img_path = tmp_path / "x.png"
    Image.new("RGB", (256, 256), color=(127, 127, 127)).save(img_path)
    img_path.with_suffix(".txt").write_text("t", encoding="utf-8")

    bucket_mgr = BucketManager(256, min_reso=256, max_reso=256, step=64)
    dataset = ImageDataset(tmp_path, 256, bucket_mgr, flip_augment=False)
    cached = CachedLatentDataset(dataset, _FakeVAE(), device="cpu", dtype=torch.float32)

    npz_path = img_path.with_suffix(".npz")
    with np.load(npz_path) as data:
        assert "latent" in data.files
        assert "latent_flipped" not in data.files


class _CountingVAEModel:
    """Mock VAE：每次 encode 把调用次数 +1，让测试能数 VAE 实际被调用了几次。"""
    def __init__(self):
        self.encode_calls = 0
        self.batch_sizes = []

    def encode(self, pixels_5d, scale):
        import torch
        self.encode_calls += 1
        b, _, _, h, w = pixels_5d.shape
        self.batch_sizes.append(b)
        return torch.zeros(b, 16, 1, h // 8, w // 8, dtype=pixels_5d.dtype)


class _CountingVAE:
    def __init__(self):
        self.model = _CountingVAEModel()
        self.scale = 1.0

    def encode(self, pixels):
        # 镜像 VAEWrapper.encode 的非分块路径（CPU 小图不触发分块）
        return self.model.encode(pixels, self.scale)


def test_cached_latent_dedupes_repeats_in_encode_pass(tmp_path: Path) -> None:
    """per-folder repeat (5_concept) 让 samples 列表里同一张图重复 N 次；
    cache 阶段必须按 npz_path 去重 — 每张唯一图只 encode 一次，
    而不是按 repeat 倍数反复 VAE encode 同一张图、反复覆盖同一 npz。
    """
    pytest.importorskip("torch")
    import torch
    from PIL import Image

    from runtime.training.dataset import BucketManager, CachedLatentDataset, ImageDataset

    folder = tmp_path / "5_concept"
    folder.mkdir()
    for i in range(2):
        img_path = folder / f"img{i}.png"
        Image.new("RGB", (256, 256), color=(127 + i, 127, 127)).save(img_path)
        img_path.with_suffix(".txt").write_text("tag", encoding="utf-8")

    bucket_mgr = BucketManager(256, min_reso=256, max_reso=256, step=64)
    dataset = ImageDataset(tmp_path, 256, bucket_mgr)
    # repeat 展开：2 张图 × 5 = 10 个 samples
    assert len(dataset.samples) == 10

    vae = _CountingVAE()
    CachedLatentDataset(dataset, vae, device="cpu", dtype=torch.float32)

    # 唯一图 2 张 × flip_augment=False，默认 cache_batch_size=1 逐张 → 2 次（不是 10 次）
    assert vae.model.encode_calls == 2, (
        f"期望 2 次 encode（唯一图数）,实际 {vae.model.encode_calls} 次 — "
        "_build_cache 没按 npz_path 去重，对同一张图按 repeat 倍数重复编码"
    )
    assert vae.model.batch_sizes == [1, 1]
    # 唯一 npz 文件 = 2
    assert len(list(folder.glob("*.npz"))) == 2


def test_cached_latent_dedupes_repeats_with_flip_aug(tmp_path: Path) -> None:
    """repeat + flip_augment：唯一图 × 2（flip/不 flip 各一次），不是 repeat × 2。"""
    pytest.importorskip("torch")
    import torch
    from PIL import Image

    from runtime.training.dataset import BucketManager, CachedLatentDataset, ImageDataset

    folder = tmp_path / "3_concept"
    folder.mkdir()
    for i in range(2):
        img_path = folder / f"img{i}.png"
        Image.new("RGB", (256, 256), color=(127 + i, 127, 127)).save(img_path)
        img_path.with_suffix(".txt").write_text("tag", encoding="utf-8")

    bucket_mgr = BucketManager(256, min_reso=256, max_reso=256, step=64)
    dataset = ImageDataset(tmp_path, 256, bucket_mgr, flip_augment=True)
    assert len(dataset.samples) == 6  # 2 × 3

    vae = _CountingVAE()
    CachedLatentDataset(dataset, vae, device="cpu", dtype=torch.float32)

    # 2 张唯一图 × 2 (flip/不 flip) = 4 次，不是 6 × 2 = 12 次
    assert vae.model.encode_calls == 4, (
        f"期望 4 次 encode（2 唯一 × flip/不 flip）,实际 {vae.model.encode_calls} 次"
    )
    assert vae.model.batch_sizes == [1, 1, 1, 1]


def test_cached_latent_respects_cache_batch_size(tmp_path: Path) -> None:
    """vae_cache_batch_size 控制缓存阶段每次送入 VAE 的同尺寸图片数量。"""
    pytest.importorskip("torch")
    import torch
    from PIL import Image

    from runtime.training.dataset import BucketManager, CachedLatentDataset, ImageDataset

    for i in range(5):
        img_path = tmp_path / f"img{i}.png"
        Image.new("RGB", (256, 256), color=(127 + i, 127, 127)).save(img_path)
        img_path.with_suffix(".txt").write_text("tag", encoding="utf-8")

    bucket_mgr = BucketManager(256, min_reso=256, max_reso=256, step=64)
    dataset = ImageDataset(tmp_path, 256, bucket_mgr, flip_augment=False)
    vae = _CountingVAE()
    CachedLatentDataset(dataset, vae, device="cpu", dtype=torch.float32, cache_batch_size=2)

    assert vae.model.batch_sizes == [2, 2, 1]
    assert len(list(tmp_path.glob("*.npz"))) == 5


def test_cached_latent_getitem_picks_flipped_per_random(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """flip_augment=True 时 __getitem__ 按 random 50% 取 latent_flipped；
    flip_augment=False 时永远取 latent（即使 npz 有 flipped 也忽略）。
    """
    pytest.importorskip("torch")
    import torch
    import numpy as np
    from PIL import Image

    from runtime.training.dataset import BucketManager, CachedLatentDataset, ImageDataset
    from runtime.training import dataset as dataset_mod

    img_path = tmp_path / "y.png"
    Image.new("RGB", (256, 256), color=(127, 127, 127)).save(img_path)
    img_path.with_suffix(".txt").write_text("t", encoding="utf-8")

    bucket_mgr = BucketManager(256, min_reso=256, max_reso=256, step=64)
    dataset = ImageDataset(tmp_path, 256, bucket_mgr, flip_augment=True)
    cached = CachedLatentDataset(dataset, _FakeVAE(), device="cpu", dtype=torch.float32)

    # 注入显式的 latent / latent_flipped 值便于分辨
    npz_path = img_path.with_suffix(".npz")
    latent_orig = np.full((16, 1, 32, 32), 1.0, dtype=np.float32)
    latent_flip = np.full((16, 1, 32, 32), 2.0, dtype=np.float32)
    np.savez(npz_path, latent=latent_orig, latent_flipped=latent_flip, bucket_w=256, bucket_h=256)

    # random.random() > 0.5 控制选 flipped；patch 成 0.9 (>0.5) → flipped
    monkeypatch.setattr(dataset_mod.random, "random", lambda: 0.9)
    item = cached[0]
    assert float(item["latent"][0, 0, 0, 0]) == 2.0  # flipped

    # patch 成 0.1 (<0.5) → 原图
    monkeypatch.setattr(dataset_mod.random, "random", lambda: 0.1)
    item = cached[0]
    assert float(item["latent"][0, 0, 0, 0]) == 1.0  # 原图

    # flip_augment=False 时永远取原图（即使 npz 有 flipped 和 random=0.9）
    cached.flip_augment = False
    monkeypatch.setattr(dataset_mod.random, "random", lambda: 0.9)
    item = cached[0]
    assert float(item["latent"][0, 0, 0, 0]) == 1.0


def test_image_dataset_get_with_flip_independent_of_random_state(tmp_path: Path) -> None:
    """get_with_flip 不读 self.flip_augment，也不掷骰子 —— 用于 cache 双份编码。

    flip=False / flip=True 必须得到精确镜像对，否则 cache 会写入随机性，污染数据。
    """
    pytest.importorskip("torch")
    import random as _random
    from PIL import Image

    from runtime.training.dataset import ImageDataset

    img_path = tmp_path / "asymmetric.png"
    # 非对称图：左半红 右半蓝，flip 后左蓝右红
    img = Image.new("RGB", (256, 256), color=(255, 0, 0))
    for x in range(128, 256):
        for y in range(256):
            img.putpixel((x, y), (0, 0, 255))
    img.save(img_path)
    img_path.with_suffix(".txt").write_text("test", encoding="utf-8")

    dataset = ImageDataset(tmp_path, 256, flip_augment=True)

    _random.seed(0)
    item_no_flip = dataset.get_with_flip(0, flip=False)
    _random.seed(99)  # 不同 seed
    item_no_flip_2 = dataset.get_with_flip(0, flip=False)
    # flip=False 在任何 random 状态下结果一致
    assert (item_no_flip["pixel_values"] == item_no_flip_2["pixel_values"]).all()

    item_flipped = dataset.get_with_flip(0, flip=True)
    # flip=True 与 flip=False 应当左右镜像（取一行验证 pixel 顺序反过来）
    row_no_flip = item_no_flip["pixel_values"][:, 100, :]  # CxHxW，取 H=100 行
    row_flipped = item_flipped["pixel_values"][:, 100, :]
    # flipped 的最后一列等于原图的第一列（左右翻）
    assert (row_no_flip[:, 0] == row_flipped[:, -1]).all()
    assert (row_no_flip[:, -1] == row_flipped[:, 0]).all()


def test_image_dataset_loads_caption_utils_when_prefer_json(tmp_path: Path) -> None:
    """Regression: dataset.py 算 caption_utils.py 路径时少回溯一层 parent 会让
    JSON caption 模式静默 fallback 到 TXT（utils/ 在仓库根，不在 runtime/utils/）。"""
    pytest.importorskip("torch")
    from runtime.training.dataset import ImageDataset

    dataset = ImageDataset(tmp_path, prefer_json=True)
    assert dataset.caption_utils is not None, (
        "prefer_json=True 应启用 JSON caption 模式 — None 说明 caption_utils.py 路径解析失败"
    )
    for key in ("load_and_build", "load_json", "normalize", "build"):
        assert key in dataset.caption_utils


def test_json_caption_list_shape_does_not_crash_issue_345(tmp_path: Path) -> None:
    """#345: Studio 打标写出的简化 JSON（tags 为扁平 list、非分类 dict）以前被
    误判为标准格式直接喂给 build，触发 'list' object has no attribute 'get'。
    现在应正常构建 caption：trigger 在首位、tags 全部保留、trigger 去重。"""
    pytest.importorskip("torch")
    import json

    from runtime.training.dataset import ImageDataset

    payload = {
        "tags": ["mika_pikazo", "1girl", "solo", "blue hair"],
        "meta": {"trigger": "mika_pikazo"},
    }
    jp = tmp_path / "10032281.json"
    jp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    dataset = ImageDataset(tmp_path, prefer_json=True)
    caption = dataset._process_caption_json(jp)

    assert caption is not None, "list 形式 caption 不应崩溃 / 静默返回 None"
    assert caption.startswith("mika_pikazo"), "trigger 应在首位（keep_tokens 保护）"
    assert "1girl" in caption and "blue hair" in caption
    assert caption.count("mika_pikazo") == 1, "trigger 与 tags 内重复项应去重"


def test_json_caption_preflight_rejects_broken_json(tmp_path: Path) -> None:
    """#345 follow-up: JSON 样本没有 txt 兜底（_make_sample 置 txt_path=None），
    caption 解析失败会静默以空 caption 训练。预检应在开训前直接报错拒绝。"""
    pytest.importorskip("torch")
    from runtime.training.dataset import ImageDataset

    img = _touch_image(tmp_path, "a.png")
    img.with_suffix(".json").write_text("{ not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="拒绝开训"):
        ImageDataset(tmp_path, prefer_json=True)


def test_json_caption_preflight_passes_healthy_flat_json(tmp_path: Path) -> None:
    """健康的扁平 list caption（#345 场景修复后）应通过预检正常建数据集。"""
    pytest.importorskip("torch")
    import json

    from runtime.training.dataset import ImageDataset

    img = _touch_image(tmp_path, "a.png")
    img.with_suffix(".json").write_text(
        json.dumps({"tags": ["mika_pikazo", "1girl"], "meta": {"trigger": "mika_pikazo"}}),
        encoding="utf-8",
    )

    dataset = ImageDataset(tmp_path, prefer_json=True)
    assert len(dataset.samples) == 1


def test_parse_repeat_kohya_prefix() -> None:
    assert datasets.parse_repeat("5_concept") == (5, "concept")
    assert datasets.parse_repeat("12_a_long_name") == (12, "a_long_name")
    assert datasets.parse_repeat("noprefix") == (1, "noprefix")
    assert datasets.parse_repeat("0_zero") == (0, "zero")


def test_txt_caption_tag_dropout_kohya_semantics(tmp_path: Path) -> None:
    """TXT 路径 tag_dropout（kohya 语义）：keep_tokens 前缀免 shuffle 免 dropout，
    其余 tag 逐个独立 dropout、无保底。dropout 丢触发词是生态已知行为，保护靠
    用户显式配 keep_tokens。"""
    pytest.importorskip("torch")
    from runtime.training.dataset import ImageDataset

    # dropout=1.0 → 非保护区全部丢弃（确定性）；keep_tokens 前缀原序保留
    ds = ImageDataset(tmp_path, shuffle_caption=True, keep_tokens=2, tag_dropout=1.0)
    assert ds._process_caption_txt("trigger, quality, a, b, c") == "trigger, quality"

    # keep_tokens=0 + dropout=1.0 → 全丢、无保底（kohya 同款）
    ds_all = ImageDataset(tmp_path, shuffle_caption=False, keep_tokens=0, tag_dropout=1.0)
    assert ds_all._process_caption_txt("a, b, c") == ""

    # dropout=0 → 原样（不 shuffle 时顺序不变）
    ds_off = ImageDataset(tmp_path, shuffle_caption=False, keep_tokens=0, tag_dropout=0.0)
    assert ds_off._process_caption_txt("a, b, c") == "a, b, c"


def test_caption_kind_priority(tmp_path: Path) -> None:
    img = _touch_image(tmp_path, "a.png")
    assert datasets.caption_kind(img) == "none"
    img.with_suffix(".txt").write_text("tag1, tag2", encoding="utf-8")
    assert datasets.caption_kind(img) == "txt"
    img.with_suffix(".json").write_text("{}", encoding="utf-8")
    assert datasets.caption_kind(img) == "json"  # json 优先于 txt


def test_scan_folder_counts_and_samples(tmp_path: Path) -> None:
    folder = tmp_path / "5_concept"
    for i in range(3):
        img = _touch_image(folder, f"{i:02d}.png")
        if i == 0:
            img.with_suffix(".json").write_text("{}", encoding="utf-8")
        elif i == 1:
            img.with_suffix(".txt").write_text("tag", encoding="utf-8")
        # 第 3 张没 caption
    # 一个非图片文件不应被计数
    (folder / "notes.md").write_text("ignore me", encoding="utf-8")

    result = datasets.scan_folder(folder)
    assert result["repeat"] == 5
    assert result["label"] == "concept"
    assert result["image_count"] == 3
    assert result["caption_types"] == {"json": 1, "txt": 1, "none": 1}
    assert len(result["samples"]) == 3


def test_scan_folder_sample_limit(tmp_path: Path) -> None:
    folder = tmp_path / "1_x"
    for i in range(10):
        _touch_image(folder, f"{i:02d}.png")
    result = datasets.scan_folder(folder, sample_limit=4)
    assert len(result["samples"]) == 4


def test_scan_root_with_subfolders(tmp_path: Path) -> None:
    _touch_image(tmp_path / "1_old", "a.png")
    _touch_image(tmp_path / "5_new", "b.png")
    _touch_image(tmp_path / "5_new", "c.png")
    result = datasets.scan_dataset_root(tmp_path)
    assert result["exists"] is True
    assert result["total_images"] == 3
    # 1_old × 1 + 5_new × 2 × 5 = 11
    assert result["weighted_steps_per_epoch"] == 11
    names = {f["name"] for f in result["folders"]}
    assert names == {"1_old", "5_new"}


def test_scan_root_includes_loose_root_images(tmp_path: Path) -> None:
    """根目录直接放的图也算一个 repeat=1 的虚拟项。"""
    _touch_image(tmp_path, "loose1.png")
    _touch_image(tmp_path / "5_x", "real.png")
    result = datasets.scan_dataset_root(tmp_path)
    assert result["total_images"] == 2
    folders = result["folders"]
    # 第一项应该是根散图
    assert folders[0]["name"] == "(根目录)"
    assert folders[0]["repeat"] == 1


def test_scan_missing_root(tmp_path: Path) -> None:
    result = datasets.scan_dataset_root(tmp_path / "nonexistent")
    assert result["exists"] is False
    assert result["folders"] == []


# ---------------------------------------------------------------------------
# /api/datasets HTTP
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """为端点测试构造一个临时 dataset，并把 server 的 REPO_ROOT 指过去。

    PR-5：/api/datasets/* + /api/browse 已搬到 studio.api.routers.browse，
    handler 内引的是 `browse.REPO_ROOT` 不是 `server.REPO_ROOT`。两边一起 patch
    防 thumbnail 403 outside-repo。
    """
    from fastapi.testclient import TestClient
    from studio import server
    from studio.api.routers import browse as _browse_router

    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    ds = fake_root / "dataset"
    _touch_image(ds / "5_concept", "a.png")
    _touch_image(ds / "5_concept", "b.png")

    monkeypatch.setattr(server, "REPO_ROOT", fake_root)
    monkeypatch.setattr(_browse_router, "REPO_ROOT", fake_root)
    return TestClient(server.app), fake_root


def test_api_datasets_default_path(client_with_dataset) -> None:
    client, _ = client_with_dataset
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert data["exists"] is True
    assert data["total_images"] == 2
    assert data["folders"][0]["name"] == "5_concept"
    assert data["folders"][0]["repeat"] == 5


def test_api_datasets_custom_relative_path(client_with_dataset) -> None:
    client, _ = client_with_dataset
    resp = client.get("/api/datasets?path=dataset")
    assert resp.status_code == 200


def test_api_datasets_missing_path_returns_exists_false(client_with_dataset) -> None:
    client, _ = client_with_dataset
    resp = client.get("/api/datasets?path=nonexistent")
    assert resp.status_code == 200
    assert resp.json()["exists"] is False


def test_thumbnail_serves_image(client_with_dataset) -> None:
    client, root = client_with_dataset
    folder = root / "dataset" / "5_concept"
    resp = client.get(
        "/api/datasets/thumbnail",
        params={"folder": str(folder), "name": "a.png"},
    )
    assert resp.status_code == 200


def test_thumbnail_blocks_traversal(client_with_dataset) -> None:
    client, _ = client_with_dataset
    resp = client.get(
        "/api/datasets/thumbnail",
        params={"folder": "../etc", "name": "passwd"},
    )
    assert resp.status_code in (400, 403, 404)


def test_thumbnail_blocks_outside_repo(client_with_dataset, tmp_path: Path) -> None:
    """文件实际存在但不在 REPO_ROOT 下时拒绝。"""
    client, _ = client_with_dataset
    outside_dir = tmp_path / "outside"
    outside_img = _touch_image(outside_dir, "x.png")
    resp = client.get(
        "/api/datasets/thumbnail",
        params={"folder": str(outside_dir), "name": outside_img.name},
    )
    assert resp.status_code == 403
