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


def test_parse_repeat_kohya_prefix() -> None:
    assert datasets.parse_repeat("5_concept") == (5, "concept")
    assert datasets.parse_repeat("12_a_long_name") == (12, "a_long_name")
    assert datasets.parse_repeat("noprefix") == (1, "noprefix")
    assert datasets.parse_repeat("0_zero") == (0, "zero")


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
