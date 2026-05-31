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

from studio.services.reg import builder as reg_builder


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
        timeout=60.0, referer=None, session=None, username="",
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
        # B1（PR-2）：schema 默认改 flat 是 user-facing 决策；老 builder test 套件
        # 仍假设 mirror（镜像 train 子文件夹结构），所以 helper 这里默认 mirror，
        # flat 测试单独传 build_mode="flat"。
        build_mode="mirror",
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


def test_find_best_match_new_formula_prefers_aspect_bucket_match() -> None:
    """PR-3 Smell C 修法：res_score 在新公式下真能影响排序，对齐用户
    「reg 集要跟 train 落同一个 ARB 桶」的诉求。

    对比两个候选：
    - A：tag 完美匹配 + aspect 差（4.0 vs target 1.0）
    - B：tag 缺 1 个 + aspect 完美（1.0 = target 1.0）

    旧公式 `tag + 0.1*res`：A 胜（res 加成不足以抵 tag 差距）。
    新公式 `0.7*tag_norm + 0.3*res`：B 胜（aspect 完美抵过 tag 略差）。
    """
    target = {f"t{i}": 1.0 for i in range(5)}
    posts = [
        # A：5 tag 完美 + aspect 4.0（差）
        {"@attributes": {
            "id": "1", "file_url": "u", "tags": "t0 t1 t2 t3 t4",
            "width": 1024, "height": 256,
        }},
        # B：缺 1 tag + aspect 1.0（完美匹配 ARB 桶）
        {"@attributes": {
            "id": "2", "file_url": "u", "tags": "t0 t1 t2 t3",
            "width": 800, "height": 800,
        }},
    ]
    best, _ = reg_builder.find_best_match(
        posts, target, {}, target_count=5,
        api_source="gelbooru", skip_similar=False,
        target_resolution=(800, 800),
        target_aspect_ratio=1.0,
    )
    assert best is not None
    assert best["@attributes"]["id"] == "2"


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
    out = tmp_path / "reg"

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


def test_build_save_tags_caption_is_space_form(tmp_path: Path, fake_booru) -> None:
    """save_tags 路径写 reg caption 时把 booru 原生下划线 tag 转成空格 —— caption
    一律空格形式（tag-form 约定：用户可见 / caption = 空格，下划线只在 booru 边界）。"""
    train = _make_train(tmp_path / "train", {
        "5_concept": [("100", ["1girl", "blue_hair"]), ("101", ["1girl", "blue_hair"])],
    })
    out = tmp_path / "reg"
    fake_booru._search_results = [[_post(2002, "1girl long_hair blue_hair")]]
    opts = _opts(train, out, target_count=1, batch_size=1, save_tags=True)
    reg_builder.build(opts, on_progress=lambda _: None)

    txts = list(out.rglob("*.txt"))
    assert txts, "save_tags 应当写出 reg caption .txt"
    content = "\n".join(p.read_text(encoding="utf-8") for p in txts)
    assert "_" not in content, f"reg caption 不应残留下划线，实际：{content!r}"
    assert "long hair" in content or "blue hair" in content  # 下划线已转空格


def test_build_flat_mode_outputs_to_1_data_with_target_count(
    tmp_path: Path, fake_booru
) -> None:
    """B1（PR-2）— flat 模式：所有图进 1_data/ 单桶，total_target 来自 opts.target_count，
    跳过 train 子文件夹镜像。"""
    train = _make_train(tmp_path / "train", {
        "5_concept": [
            ("100", ["1girl", "solo"]),
            ("101", ["1girl"]),
        ],
        "1_general": [
            ("200", ["solo", "indoors"]),
        ],
    })
    out = tmp_path / "reg"
    fake_booru._search_results = [
        [_post(3001, "1girl solo"), _post(3002, "indoors 1girl")],
    ]
    opts = _opts(
        train, out, target_count=2, batch_size=2, build_mode="flat",
    )
    meta = reg_builder.build(opts, on_progress=lambda _: None)
    # 单桶 1_data/；不出 5_concept / 1_general
    assert (out / "1_data").is_dir()
    assert not (out / "5_concept").exists()
    assert not (out / "1_general").exists()
    # 总数 = 用户给的 target_count
    assert meta.target_count == 2
    assert meta.build_mode == "flat"


def test_build_mirror_mode_keeps_subfolder_split(
    tmp_path: Path, fake_booru
) -> None:
    """B1（PR-2）— mirror 模式：保持源脚本行为，按 train 子文件夹镜像。"""
    train = _make_train(tmp_path / "train", {
        "5_concept": [("100", ["1girl"])],
        "1_general": [("200", ["solo"])],
    })
    out = tmp_path / "reg"
    fake_booru._search_results = [
        [_post(4001, "1girl")],
        [_post(4002, "solo")],
    ]
    opts = _opts(train, out, batch_size=1, build_mode="mirror")
    meta = reg_builder.build(opts, on_progress=lambda _: None)
    assert (out / "5_concept").is_dir()
    assert (out / "1_general").is_dir()
    assert not (out / "1_data").exists()
    assert meta.build_mode == "mirror"


def test_build_mirrors_train_subfolder_repeat_prefix(
    tmp_path: Path, fake_booru
) -> None:
    """train 有 2_concept / 4_general → reg 镜像出同名同 repeat 子文件夹。"""
    train = _make_train(tmp_path / "train", {
        "2_concept": [
            ("100", ["1girl", "solo"]),
            ("101", ["1girl"]),
        ],
        "4_general": [
            ("200", ["outdoor"]),
        ],
    })
    out = tmp_path / "reg"

    fake_booru._search_results = [
        # 第一批：2_concept 的搜索
        [_post(9001, "1girl solo"), _post(9002, "1girl long_hair")],
        # 第二批：4_general 的搜索
        [_post(9101, "outdoor sky")],
    ] * 5
    opts = _opts(train, out)
    reg_builder.build(opts, on_progress=lambda _: None)

    # 验证 repeat 前缀完全保留
    assert (out / "2_concept").is_dir()
    assert (out / "4_general").is_dir()
    # 不存在「1_general」之类自创的目录
    assert not (out / "1_general").exists()


def test_build_skips_source_ids(tmp_path: Path, fake_booru) -> None:
    """train 里有 id=2001 → reg 不应再下 2001。"""
    train = _make_train(tmp_path / "train", {
        "5_concept": [
            ("2001", ["1girl"]),  # source 已有
        ],
    })
    out = tmp_path / "reg"

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
    out = tmp_path / "reg"

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
    out = tmp_path / "reg"

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
    out = tmp_path / "reg"

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
    out = tmp_path / "reg"
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


def test_collect_existing_reg_per_subfolder(tmp_path: Path) -> None:
    out = tmp_path / "reg"
    (out / "5_concept").mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(out / "5_concept" / "100.png", "PNG")
    (out / "5_concept" / "100.txt").write_text("a, b", encoding="utf-8")
    Image.new("RGB", (16, 16)).save(out / "5_concept" / "200.png", "PNG")
    Image.new("RGB", (16, 16)).save(out / "300.png", "PNG")  # 根目录
    pre = reg_builder.collect_existing_reg_per_subfolder(out)
    assert pre["5_concept"]["count"] == 2
    assert pre["5_concept"]["ids"] == {"100", "200"}
    # 100 有 caption；200 没有 → tags 列表第一项 ['a','b']，第二项 []
    tag_lists = pre["5_concept"]["tags"]
    assert sorted(tag_lists, key=len, reverse=True)[0] == ["a", "b"]
    assert pre[""]["count"] == 1
    assert pre[""]["ids"] == {"300"}


def test_build_incremental_keeps_existing(tmp_path: Path, fake_booru) -> None:
    """PP5.1：已有 reg 图被计入起点，仅补差额。"""
    train = _make_train(tmp_path / "train", {
        "5_concept": [
            ("100", ["1girl", "solo"]),
            ("101", ["1girl"]),
        ],
    })
    out = tmp_path / "reg"
    # 预先放一张已有 reg 图（id=8000）
    (out / "5_concept").mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(out / "5_concept" / "8000.png", "PNG")
    (out / "5_concept" / "8000.txt").write_text("1girl, solo", encoding="utf-8")
    # 写一份旧 meta 用来累积 incremental_runs
    reg_builder.write_meta(out, reg_builder.RegMeta(
        generated_at=1.0, based_on_version="x", api_source="gelbooru",
        target_count=2, actual_count=1, source_tags=["solo"],
        excluded_tags=[], blacklist_tags=[], failed_tags=["unique"],
        train_tag_distribution={}, auto_tagged=False, incremental_runs=0,
    ))

    fake_booru._search_results = [[_post(9001, "1girl long_hair")]]
    opts = _opts(train, out, target_count=2, batch_size=1)
    meta = reg_builder.build(opts, on_progress=lambda _: None, incremental=True)

    # 8000 仍在 + 9001 新下 = 2
    assert (out / "5_concept" / "8000.png").exists()
    assert (out / "5_concept" / "9001.png").exists()
    # incremental_runs 累加
    assert meta.incremental_runs == 1
    # failed_tags 与旧 meta 合并
    assert "unique" in meta.failed_tags


def test_build_incremental_no_op_when_target_already_met(
    tmp_path: Path, fake_booru
) -> None:
    """已有图 >= target → 直接返回，不调 booru。"""
    train = _make_train(tmp_path / "train", {
        "5_concept": [("1", ["x"])],
    })
    out = tmp_path / "reg"
    (out / "5_concept").mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(out / "5_concept" / "5000.png", "PNG")
    (out / "5_concept" / "5000.txt").write_text("x", encoding="utf-8")

    fake_booru._search_results = []  # 无候选；如果走到 booru 会失败
    opts = _opts(train, out, target_count=1, batch_size=1)
    meta = reg_builder.build(opts, on_progress=lambda _: None, incremental=True)
    assert meta.actual_count == 1
    # 没新增图
    assert {p.name for p in (out / "5_concept").glob("*.png")} == {"5000.png"}


# ---------------------------------------------------------------------------
# A2 — deleted_ids.json 排除
# ---------------------------------------------------------------------------


def test_deleted_ids_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "reg"
    out.mkdir()
    assert reg_builder.read_deleted_ids(out) == set()
    reg_builder.append_deleted_ids(out, ["42", "99"])
    assert reg_builder.read_deleted_ids(out) == {"42", "99"}
    # 追加 + 去重
    reg_builder.append_deleted_ids(out, ["99", "100"])
    assert reg_builder.read_deleted_ids(out) == {"42", "99", "100"}


def test_deleted_ids_corrupt_file_returns_empty(tmp_path: Path) -> None:
    out = tmp_path / "reg"
    out.mkdir()
    (out / ".deleted_ids.json").write_text("not json", encoding="utf-8")
    assert reg_builder.read_deleted_ids(out) == set()


def test_build_excludes_previously_deleted_ids(
    tmp_path: Path, fake_booru,
) -> None:
    """A2：incremental 时已删 ID 进 downloaded_ids，search 自动 exclude；
    候选只剩下未删的，不会被重新拉回。"""
    train = _make_train(tmp_path / "train", {
        "5_concept": [
            ("100", ["1girl", "solo"]),
            ("101", ["1girl"]),
        ],
    })
    out = tmp_path / "reg"
    out.mkdir(parents=True)
    # 用户之前删过 booru ID=7777，写到 deleted_ids
    reg_builder.append_deleted_ids(out, ["7777"])
    # search 第一波同时返回 7777（被删过）和 8888（新的）
    fake_booru._search_results = [
        [_post(7777, "1girl solo"), _post(8888, "1girl solo")],
    ]
    opts = _opts(train, out, target_count=1, batch_size=1)
    reg_builder.build(opts, on_progress=lambda _: None, incremental=False)
    # 7777 不该被下载（exclude）；只能选 8888
    downloaded = {p.stem for p in out.rglob("*.png")}
    assert "7777" not in downloaded
    assert "8888" in downloaded


# ---------------------------------------------------------------------------
# A3 — auto_tag_kind 字段
# ---------------------------------------------------------------------------


def test_reg_build_options_default_auto_tag_kind_is_wd14() -> None:
    opts = reg_builder.RegBuildOptions(
        train_dir=Path("/tmp"), output_dir=Path("/tmp/out"),
    )
    assert opts.auto_tag_kind == "wd14"


def test_update_meta_auto_tagged_writes_kind(tmp_path: Path) -> None:
    out = tmp_path / "reg"
    out.mkdir()
    m = reg_builder.RegMeta(
        generated_at=1.0, based_on_version="x", api_source="gelbooru",
        target_count=2, actual_count=2, source_tags=[],
        excluded_tags=[], blacklist_tags=[], failed_tags=[],
        train_tag_distribution={}, auto_tagged=False,
    )
    reg_builder.write_meta(out, m)
    reg_builder.update_meta_auto_tagged(out, True, kind="cltagger")
    m2 = reg_builder.read_meta(out)
    assert m2.auto_tagged is True
    assert m2.auto_tag_kind == "cltagger"


def test_update_meta_auto_tagged_clears_kind_on_failure(tmp_path: Path) -> None:
    """auto_tagged=False 时把 kind 重置为 None（未成功打过的 tagger 不应残留）。"""
    out = tmp_path / "reg"
    out.mkdir()
    m = reg_builder.RegMeta(
        generated_at=1.0, based_on_version="x", api_source="gelbooru",
        target_count=2, actual_count=2, source_tags=[],
        excluded_tags=[], blacklist_tags=[], failed_tags=[],
        train_tag_distribution={}, auto_tagged=False,
        auto_tag_kind="wd14",  # 上次成功过
    )
    reg_builder.write_meta(out, m)
    reg_builder.update_meta_auto_tagged(out, False, kind="cltagger")
    m2 = reg_builder.read_meta(out)
    assert m2.auto_tagged is False
    assert m2.auto_tag_kind is None


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
    # 默认 generation_method = "scrape"（兼容旧 meta）
    assert m2.generation_method == "scrape"

    reg_builder.update_meta_auto_tagged(out, True)
    m3 = reg_builder.read_meta(out)
    assert m3.auto_tagged is True


def test_meta_legacy_without_generation_method(tmp_path: Path) -> None:
    """旧 meta.json（无 generation_method 字段）反序列化必须不崩，落到默认 "scrape"。"""
    import json
    out = tmp_path / "reg"
    out.mkdir()
    legacy = {
        "generated_at": 1.0, "based_on_version": "x", "api_source": "gelbooru",
        "target_count": 5, "actual_count": 5,
        "source_tags": [], "excluded_tags": [], "blacklist_tags": [],
        "failed_tags": [], "train_tag_distribution": {}, "auto_tagged": False,
        "incremental_runs": 0,
        "postprocessed_at": None, "postprocess_clusters": None,
        "postprocess_method": None, "postprocess_max_crop_ratio": None,
        # 故意不带 generation_method
    }
    (out / "meta.json").write_text(json.dumps(legacy), encoding="utf-8")

    m = reg_builder.read_meta(out)
    assert m is not None
    assert m.generation_method == "scrape"
    assert m.api_source == "gelbooru"


def test_meta_ai_base_roundtrip(tmp_path: Path) -> None:
    """先验生成写出的 meta：generation_method='ai_base' + api_source 留空。"""
    out = tmp_path / "reg"
    out.mkdir()
    m = reg_builder.RegMeta(
        generated_at=2.0, based_on_version="", api_source="",
        target_count=10, actual_count=10, source_tags=[],
        excluded_tags=["face_focus"], blacklist_tags=[], failed_tags=[],
        train_tag_distribution={"1girl": 5}, auto_tagged=False,
        generation_method="ai_base",
    )
    reg_builder.write_meta(out, m)
    m2 = reg_builder.read_meta(out)
    assert m2 is not None
    assert m2.generation_method == "ai_base"
    assert m2.api_source == ""
