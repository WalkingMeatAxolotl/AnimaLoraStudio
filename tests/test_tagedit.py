"""PP4 — tagedit: stats / add / remove / replace / dedupe + format 自适应。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio.services.dataset import tagedit
from studio.services.tagging.caption_format import caption_json_to_text


@pytest.fixture
def train_dir(tmp_path: Path) -> Path:
    d = tmp_path / "train"
    d.mkdir()
    return d


def _img(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_bytes(b"x")  # 假图，仅为 caption_path 做存在判定
    return p


def _txt(image: Path, content: str) -> Path:
    p = image.with_suffix(".txt")
    p.write_text(content, encoding="utf-8")
    return p


def _json(image: Path, tags: list[str]) -> Path:
    p = image.with_suffix(".json")
    p.write_text(json.dumps({"tags": tags}, ensure_ascii=False), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# read / write
# ---------------------------------------------------------------------------


def test_read_and_write_txt(train_dir: Path) -> None:
    f = _img(train_dir / "5_a", "1.png")
    _txt(f, "a, b, c")
    assert tagedit.read_tags(f) == ["a", "b", "c"]
    out = tagedit.write_tags(f, ["x", "y"])
    assert out.suffix == ".txt"
    assert out.read_text(encoding="utf-8") == "x, y"


def test_json_takes_precedence(train_dir: Path) -> None:
    f = _img(train_dir / "5_a", "1.png")
    _txt(f, "from txt")
    _json(f, ["from", "json"])
    # 两个都在时，json 优先
    assert tagedit.read_tags(f) == ["from", "json"]
    # 写入也走 json
    out = tagedit.write_tags(f, ["new"])
    assert out.suffix == ".json"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["tags"] == ["new"]


def test_read_documented_json_caption(train_dir: Path) -> None:
    f = _img(train_dir / "5_a", "1.png")
    f.with_suffix(".json").write_text(
        json.dumps(
            {
                "fixed": {"quality": "", "series": "", "artist": ""},
                "character": {"name": "", "variant": "", "full": ""},
                "from_path": {},
                "ai_output": {
                    "count": "1girl",
                    "appearance": ["long hair"],
                    "tags": ["watercolor"],
                    "environment": ["blue background"],
                    "nl": "Soft style.",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert tagedit.read_tags(f) == [
        "1girl",
        "long hair",
        "watercolor",
        "blue background",
    ]


def test_write_documented_json_makes_editor_tags_authoritative(train_dir: Path) -> None:
    f = _img(train_dir / "5_a", "1.png")
    p = f.with_suffix(".json")
    p.write_text(
        json.dumps(
            {
                "fixed": {"quality": "best", "series": "", "artist": ""},
                "character": {"name": "", "variant": "", "full": ""},
                "from_path": {},
                "ai_output": {
                    "count": "1girl",
                    "appearance": ["old appearance"],
                    "tags": ["old tag"],
                    "environment": [],
                    "nl": "old prose",
                },
                "meta": {"trigger": "ohwx"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tagedit.write_tags(f, ["ohwx", "new tag"])

    written = json.loads(p.read_text(encoding="utf-8"))
    assert written["tags"] == ["ohwx", "new tag"]
    assert written["ai_output"]["tags"] == ["old tag"]
    assert written["meta"] == {"trigger": "ohwx"}
    assert tagedit.read_tags(f) == ["ohwx", "new tag"]
    assert caption_json_to_text(written) == "ohwx, new tag. old prose"


def test_read_missing_returns_empty(train_dir: Path) -> None:
    f = _img(train_dir / "5_a", "noaption.png")
    assert tagedit.read_tags(f) == []
    assert tagedit.has_effective_caption(f) is False


def test_effective_caption_uses_rendered_json_content(train_dir: Path) -> None:
    f = _img(train_dir / "5_a", "1.png")
    p = f.with_suffix(".json")
    p.write_text(json.dumps({"tags": [], "nl": "A quiet scene."}), encoding="utf-8")
    assert tagedit.has_effective_caption(f) is True

    p.write_text(json.dumps({"tags": [], "meta": {"trigger": "ohwx"}}), encoding="utf-8")
    assert tagedit.has_effective_caption(f) is True

    p.write_text("{broken", encoding="utf-8")
    assert tagedit.has_effective_caption(f) is False


@pytest.mark.parametrize("tags", [[], ["new"]])
def test_write_json_tags_retains_sidecars_and_non_editor_fields(
    train_dir: Path, tags: list[str],
) -> None:
    f = _img(train_dir / "5_a", "1.png")
    txt = _txt(f, "old txt")
    js = f.with_suffix(".json")
    original = {
        "tags": ["old"], "nl": "A quiet scene.", "meta": {"trigger": "ohwx"},
    }
    js.write_text(json.dumps(original), encoding="utf-8")

    assert tagedit.write_tags(f, tags) == js
    assert txt.read_text(encoding="utf-8") == "old txt"
    assert json.loads(js.read_text(encoding="utf-8")) == {**original, "tags": tags}
    assert tagedit.has_effective_caption(f) is True


def test_write_empty_tags_keeps_empty_txt(train_dir: Path) -> None:
    f = _img(train_dir / "5_a", "1.png")
    txt = _txt(f, "old")
    assert tagedit.write_tags(f, []) == txt
    assert txt.read_text(encoding="utf-8") == ""
    assert tagedit.caption_path(f) == txt
    assert tagedit.has_effective_caption(f) is False


def test_clear_flat_json_does_not_fall_back_to_txt(train_dir: Path) -> None:
    f = _img(train_dir / "5_a", "1.png")
    txt = _txt(f, "old txt")
    js = _json(f, ["old json"])
    assert tagedit.write_tags(f, []) == js
    assert json.loads(js.read_text(encoding="utf-8")) == {"tags": []}
    assert txt.exists()
    assert tagedit.read_tags(f) == []
    assert tagedit.has_effective_caption(f) is False


def test_batch_remove_last_tag_preserves_structured_non_editor_text(
    train_dir: Path,
) -> None:
    f = _img(train_dir / "5_a", "1.png")
    f.with_suffix(".json").write_text(
        json.dumps({"ai_output": {"tags": ["old"], "nl": "A quiet scene."}}),
        encoding="utf-8",
    )

    assert tagedit.remove_tags({"kind": "all"}, train_dir, ["old"]) == 1
    assert f.with_suffix(".json").exists()
    assert tagedit.has_effective_caption(f) is True


# ---------------------------------------------------------------------------
# scope ops
# ---------------------------------------------------------------------------


def _setup_scope(train_dir: Path) -> None:
    f1 = _img(train_dir / "5_a", "1.png")
    _txt(f1, "x, y")
    f2 = _img(train_dir / "5_a", "2.png")
    _txt(f2, "x, z")
    f3 = _img(train_dir / "1_data", "g.png")
    _txt(f3, "x, only_data")


def test_stats_counts_across_all(train_dir: Path) -> None:
    _setup_scope(train_dir)
    s = dict(tagedit.stats({"kind": "all"}, train_dir))
    assert s["x"] == 3
    assert s["y"] == 1
    assert s["only_data"] == 1


def test_stats_scoped_to_folder(train_dir: Path) -> None:
    _setup_scope(train_dir)
    s = dict(tagedit.stats({"kind": "folder", "name": "5_a"}, train_dir))
    assert s["x"] == 2
    assert "only_data" not in s


def test_stats_scoped_to_files(train_dir: Path) -> None:
    _setup_scope(train_dir)
    s = dict(
        tagedit.stats(
            {"kind": "files", "folder": "5_a", "names": ["1.png"]},
            train_dir,
        )
    )
    assert s == {"x": 1, "y": 1}


def test_add_back_skips_dups(train_dir: Path) -> None:
    _setup_scope(train_dir)
    n = tagedit.add_tags({"kind": "all"}, train_dir, ["x", "new1"])
    # 三张图都应该被改（都新增了 new1，x 已有不重复）
    assert n == 3
    assert tagedit.read_tags(train_dir / "5_a" / "1.png") == ["x", "y", "new1"]


def test_add_front(train_dir: Path) -> None:
    _setup_scope(train_dir)
    tagedit.add_tags(
        {"kind": "folder", "name": "5_a"}, train_dir, ["zz"], position="front"
    )
    assert tagedit.read_tags(train_dir / "5_a" / "1.png")[0] == "zz"


def test_remove(train_dir: Path) -> None:
    _setup_scope(train_dir)
    n = tagedit.remove_tags({"kind": "all"}, train_dir, ["x"])
    assert n == 3
    assert tagedit.read_tags(train_dir / "5_a" / "1.png") == ["y"]


def test_replace(train_dir: Path) -> None:
    _setup_scope(train_dir)
    n = tagedit.replace_tag({"kind": "all"}, train_dir, "x", "X2")
    assert n == 3
    assert tagedit.read_tags(train_dir / "5_a" / "1.png")[0] == "X2"


def test_replace_into_existing_dedupes(train_dir: Path) -> None:
    f = _img(train_dir / "5_a", "1.png")
    _txt(f, "a, b, c")
    n = tagedit.replace_tag({"kind": "all"}, train_dir, "a", "b")
    assert n == 1
    # b 已存在 → 把 a 删掉，b 保留一次
    assert tagedit.read_tags(f) == ["b", "c"]


def test_dedupe(train_dir: Path) -> None:
    f = _img(train_dir / "5_a", "1.png")
    _txt(f, "a, b, a, c, b")
    n = tagedit.dedupe({"kind": "all"}, train_dir)
    assert n == 1
    assert tagedit.read_tags(f) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# single-image helpers
# ---------------------------------------------------------------------------


def test_list_captions_in_folder(train_dir: Path) -> None:
    _setup_scope(train_dir)
    items = tagedit.list_captions_in_folder(train_dir, "5_a")
    names = sorted(i["name"] for i in items)
    assert names == ["1.png", "2.png"]
    by_name = {i["name"]: i for i in items}
    assert by_name["1.png"]["tag_count"] == 2
    assert by_name["1.png"]["has_caption"] is True
    assert by_name["1.png"]["has_effective_caption"] is True


def test_list_marks_empty_caption_as_ineffective(train_dir: Path) -> None:
    f = _img(train_dir / "5_a", "1.png")
    _txt(f, "  ,  ")

    item = tagedit.list_captions_in_folder(train_dir, "5_a", full=True)[0]
    assert item["has_caption"] is True
    assert item["has_effective_caption"] is False
    assert item["format"] == "txt"


def test_read_one_and_write_one(train_dir: Path) -> None:
    f = _img(train_dir / "5_a", "1.png")
    _txt(f, "a, b")
    r = tagedit.read_one(train_dir, "5_a", "1.png")
    assert r["tags"] == ["a", "b"]
    assert r["format"] == "txt"
    updated = tagedit.write_one(train_dir, "5_a", "1.png", ["x"])
    assert updated["tags"] == ["x"]


def test_read_one_404(train_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        tagedit.read_one(train_dir, "ghost", "ghost.png")
