"""PP5 — reg_builder 库化版本：mock booru API + 假 train 目录。

只测核心算法和主流程，避免触网。逻辑必须与源脚本一致。
"""
from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from studio.services import reg_builder


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _make_image(path: Path, size: tuple[int, int] = (512, 512)) -> None:
    img = Image.new("RGB", size, (255, 0, 0))
    img.save(path, "PNG")


def _make_train(
    base: Path,
    layout: dict[str, list[tuple[str, list[str]]]],
    *,
    sizes: dict[str, tuple[int, int]] | None = None,
) -> Path:
    """layout = {folder_name: [(image_stem, [tags...]), ...]}.

    folder_name == "" 表示根目录。所有图都建 .png + .txt。
    """
    base.mkdir(parents=True, exist_ok=True)
    sizes = sizes or {}
    for folder, items in layout.items():
        d = base / folder if folder else base
        d.mkdir(parents=True, exist_ok=True)
        for stem, tags in items:
            img_path = d / f"{stem}.png"
            _make_image(img_path, sizes.get(stem, (512, 512)))
            (d / f"{stem}.txt").write_text(", ".join(tags), encoding="utf-8")
    return base


def _png_bytes() -> bytes:
    img = Image.new("RGB", (8, 8), (0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class FakeResp:
    def __init__(self, *, json_data=None, content: bytes = b"", status: int = 200):
        self._json = json_data
        self.content = content
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


class FakeBooru:
    """监听 booru_api.search_posts / download_image 调用，返回预定义脚本。"""

    def __init__(
        self,
        search_results: list[list[dict[str, Any]]] | dict[tuple[str, ...], list[dict[str, Any]]],
    ):
        # 支持两种：list（按调用次序回） / dict（按搜索 tag 元组回）
        self._search_results = search_results
        self._search_calls: list[tuple[str, dict[str, Any]]] = []
        self._download_calls: list[str] = []

    def search_posts(
        self, api_source, tags_query, *, page=1, limit=100,
        user_id="", api_key="", username="", base_url=None, timeout=30, session=None,
    ):
        self._search_calls.append((tags_query, {
            "page": page, "limit": limit, "api_source": api_source,
        }))
        if isinstance(self._search_results, list):
            if not self._search_results:
                return []
            return self._search_results.pop(0)
        # dict 模式：tag set 匹配
        key = tuple(sorted(tags_query.split()))
        return list(self._search_results.get(key, []))

    def download_image(
        self, url, save_path, *, convert_to_png, remove_alpha_channel,
        timeout=60.0, referer=None, session=None,
    ):
        self._download_calls.append(url)
        # 真写一张小图（不 mock 文件系统）
        save_path = Path(save_path)
        if convert_to_png and save_path.suffix.lower() != ".png":
            save_path = save_path.with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (256, 256), (0, 255, 0))
        img.save(save_path, "PNG")
        return save_path


@pytest.fixture
def fake_booru(monkeypatch):
    """monkeypatch booru_api 默认无 search 结果；测试自行 set。"""
    fake = FakeBooru([])
    monkeypatch.setattr(reg_builder.booru_api, "search_posts", fake.search_posts)
    monkeypatch.setattr(reg_builder.booru_api, "download_image", fake.download_image)
    return fake


def _opts(train: Path, out: Path, **overrides) -> reg_builder.RegBuildOptions:
    base = dict(
        train_dir=train,
        output_dir=out,
        api_source="gelbooru",
        user_id="u",
        api_key="k",
        target_count=None,
        batch_size=2,
        max_search_tags=10,
        skip_similar=False,  # 测试要预测候选数；关掉跳偶数
        excluded_tags=[],
        blacklist_tags=[],
        auto_tag=False,
        based_on_version="",
    )
    base.update(overrides)
    return reg_builder.RegBuildOptions(**base)


# ---------------------------------------------------------------------------
# pure-fn tests
# ---------------------------------------------------------------------------


def test_normalize_tags_dedup_lowers_and_replaces_space() -> None:
    out = reg_builder._normalize_tags(["1girl", "Solo", "long Hair", "1girl"])
    assert out == ["1girl", "solo", "long_hair"]


def test_analyze_tags_in_file_reads_txt(tmp_path: Path) -> None:
    img = tmp_path / "x.png"
    _make_image(img)
    (tmp_path / "x.txt").write_text("1girl, Solo, long hair", encoding="utf-8")
    assert reg_builder.analyze_tags_in_file(img) == ["1girl", "solo", "long_hair"]


def test_analyze_tags_in_file_reads_json(tmp_path: Path) -> None:
    img = tmp_path / "x.png"
    _make_image(img)
    (tmp_path / "x.json").write_text(
        json.dumps({"tags": ["1girl", "Solo"]}), encoding="utf-8"
    )
    assert reg_builder.analyze_tags_in_file(img) == ["1girl", "solo"]


def test_analyze_dataset_structure_collects_subfolders(tmp_path: Path) -> None:
    train = _make_train(tmp_path / "train", {
        "5_concept": [
            ("a", ["1girl", "solo", "blue_hair"]),
            ("b", ["1girl", "long_hair"]),
        ],
        "1_general": [
            ("c", ["1girl", "outdoor"]),
        ],
    })
    s = reg_builder.analyze_dataset_structure(train, on_progress=lambda _: None)
    assert s["total_images"] == 3
    assert "5_concept" in s["subfolders"]
    assert "1_general" in s["subfolders"]
    # 1girl 出现 3 次
    assert s["global_tag_freq"]["1girl"] == 3
    # 全局权重
    assert s["global_tag_weights"]["1girl"] == pytest.approx(1.0)
    assert s["global_tag_weights"]["blue_hair"] == pytest.approx(1 / 3)
    # 中位数分辨率
    assert s["median_resolution"] == (512, 512)


def test_collect_source_image_ids_returns_stems(tmp_path: Path) -> None:
    train = _make_train(tmp_path / "train", {
        "5_concept": [("123", ["x"]), ("456", ["y"])],
    })
    ids = reg_builder.collect_source_image_ids(train)
    assert ids == {"123", "456"}


def test_calculate_missing_tags_filters_blacklist_and_failed() -> None:
    target = {"a": 1.0, "b": 0.5, "c": 0.3}
    current = {"a": 0.5}
    missing = reg_builder.calculate_missing_tags(
        target, current, blacklist_tags={"c"}, failed_tags={"b"}
    )
    # 只剩 a（差 0.5）
    assert missing == [("a", pytest.approx(0.5))]


def test_calculate_tag_similarity_prefers_matching_tags() -> None:
    target = {"a": 1.0, "b": 1.0}
    current = {}
    score_match = reg_builder.calculate_tag_similarity(target, ["a", "b"], current, 2)
    score_miss = reg_builder.calculate_tag_similarity(target, ["x", "y"], current, 2)
    assert score_match > score_miss


def test_check_aspect_ratio_disabled_passes_anything() -> None:
    assert reg_builder.check_aspect_ratio(
        100, 1000, enabled=False, min_ar=0.5, max_ar=2.0
    ) is True


def test_check_aspect_ratio_enabled_filters() -> None:
    assert reg_builder.check_aspect_ratio(
        100, 100, enabled=True, min_ar=0.5, max_ar=2.0
    ) is True
    assert reg_builder.check_aspect_ratio(
        300, 100, enabled=True, min_ar=0.5, max_ar=2.0
    ) is False  # ar=3.0 > 2.0


def test_find_best_match_skips_source_ids(tmp_path: Path) -> None:
    posts = [
        {"@attributes": {"id": "10", "file_url": "u", "tags": "1girl solo", "width": 512, "height": 512}},
        {"@attributes": {"id": "20", "file_url": "u", "tags": "1girl solo", "width": 512, "height": 512}},
    ]
    target = {"1girl": 1.0, "solo": 1.0}
    best, _ = reg_builder.find_best_match(
        posts, target, {}, target_count=1,
        api_source="gelbooru", skip_similar=False,
        source_image_ids={"10"},
    )
    assert best is not None
    assert best["@attributes"]["id"] == "20"


# ---------------------------------------------------------------------------
# main flow
# ---------------------------------------------------------------------------


def _post(pid: int, tags: str, *, w: int = 512, h: int = 512) -> dict[str, Any]:
    return {"@attributes": {
        "id": str(pid),
        "file_url": f"http://x/{pid}.png",
        "file_ext": "png",
        "tags": tags,
        "width": w,
        "height": h,
    }}


def test_build_writes_meta_and_images(tmp_path: Path, fake_booru) -> None:
    train = _make_train(tmp_path / "train", {
        "5_concept": [
            ("100", ["1girl", "solo", "blue_hair"]),
            ("101", ["1girl", "blue_hair"]),
        ],
    })
    out = tmp_path / "reg" / "1_general"

    # search 返回 2 张候选（id 不在 train 里）
    fake_booru._search_results = [[_post(2001, "1girl solo blue_hair"), _post(2002, "1girl long_hair")]]

    opts = _opts(train, out, target_count=1, batch_size=1)
    meta = reg_builder.build(opts, on_progress=lambda _: None)

    # meta 写盘
    assert (out / "meta.json").exists()
    saved = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert saved["actual_count"] >= 1
    assert saved["target_count"] == 1
    # 至少下了一张
    assert any(p.suffix == ".png" for p in out.rglob("*.png") if p.parent == out / "5_concept")


def test_build_skips_source_ids(tmp_path: Path, fake_booru) -> None:
    """train 里有 id=2001 → reg 不应再下 2001。"""
    train = _make_train(tmp_path / "train", {
        "5_concept": [
            ("2001", ["1girl"]),  # source 已有
        ],
    })
    out = tmp_path / "reg" / "1_general"

    fake_booru._search_results = [
        [_post(2001, "1girl"), _post(3001, "1girl")],  # 2001 在 source 里，应跳
    ]
    opts = _opts(train, out, target_count=1, batch_size=1)
    reg_builder.build(opts, on_progress=lambda _: None)
    files = list((out / "5_concept").glob("*.png"))
    names = {p.stem for p in files}
    assert "2001" not in names
    assert "3001" in names


def test_build_failed_tags_accumulate(tmp_path: Path, fake_booru) -> None:
    """单 tag 搜索失败 → 加入 failed_tags，meta 中体现。"""
    train = _make_train(tmp_path / "train", {
        "5_concept": [
            ("1", ["unique_tag"]),
        ],
    })
    out = tmp_path / "reg" / "1_general"

    # 任何搜索都返回空 → 触发 failed_tags 累积 → 5 次连续失败退出
    fake_booru._search_results = [[]] * 50
    opts = _opts(train, out, target_count=1, batch_size=1, max_search_tags=1)
    meta = reg_builder.build(opts, on_progress=lambda _: None)
    assert meta.actual_count == 0
    assert "unique_tag" in meta.failed_tags


def test_build_respects_blacklist(tmp_path: Path, fake_booru) -> None:
    train = _make_train(tmp_path / "train", {
        "5_concept": [
            ("1", ["1girl", "censored"]),
        ],
    })
    out = tmp_path / "reg" / "1_general"

    # 只有一个候选；它带黑名单 tag → 应被本地过滤掉
    fake_booru._search_results = [[_post(5001, "1girl censored")]]
    opts = _opts(
        train, out, target_count=1, batch_size=1,
        blacklist_tags=["censored"],
    )
    reg_builder.build(opts, on_progress=lambda _: None)
    files = list((out / "5_concept").glob("*.png"))
    assert files == []


def test_build_auto_blacklists_version_name(tmp_path: Path, fake_booru) -> None:
    train = _make_train(tmp_path / "train", {
        "5_concept": [
            ("1", ["1girl"]),
        ],
    })
    out = tmp_path / "reg" / "1_general"

    fake_booru._search_results = [[_post(5001, "1girl baseline_artist")]]
    opts = _opts(
        train, out, target_count=1, batch_size=1,
        based_on_version="baseline_artist",
    )
    reg_builder.build(opts, on_progress=lambda _: None)
    # 版本名（lower）应进 blacklist，候选被过滤
    files = list((out / "5_concept").glob("*.png"))
    assert files == []


def test_build_cancel_event_exits_promptly(tmp_path: Path, fake_booru) -> None:
    train = _make_train(tmp_path / "train", {
        "5_concept": [
            (str(i), ["1girl"]) for i in range(5)
        ],
    })
    out = tmp_path / "reg" / "1_general"
    fake_booru._search_results = [[_post(9000 + i, "1girl") for i in range(20)]] * 50

    cancel = threading.Event()
    cancel.set()  # 立刻取消
    opts = _opts(train, out, target_count=5, batch_size=1)
    meta = reg_builder.build(opts, on_progress=lambda _: None, cancel_event=cancel)
    assert meta.actual_count == 0


def test_preview_train_tag_distribution_returns_top_n(tmp_path: Path) -> None:
    train = _make_train(tmp_path / "train", {
        "5_concept": [
            ("1", ["a", "b", "c"]),
            ("2", ["a", "b"]),
            ("3", ["a"]),
        ],
    })
    top = reg_builder.preview_train_tag_distribution(train, top=2)
    assert top == [("a", 3), ("b", 2)]


def test_meta_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "reg"
    out.mkdir()
    m = reg_builder.RegMeta(
        generated_at=1.0, based_on_version="x", api_source="gelbooru",
        target_count=10, actual_count=8, source_tags=["a", "b"],
        excluded_tags=["c"], blacklist_tags=["d"], failed_tags=[],
        train_tag_distribution={"a": 5}, auto_tagged=False,
    )
    reg_builder.write_meta(out, m)
    m2 = reg_builder.read_meta(out)
    assert m2 is not None
    assert m2.actual_count == 8
    assert m2.train_tag_distribution == {"a": 5}

    reg_builder.update_meta_auto_tagged(out, True)
    m3 = reg_builder.read_meta(out)
    assert m3.auto_tagged is True
