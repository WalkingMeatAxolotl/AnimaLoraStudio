"""PP4 — tagedit: stats / add / remove / replace / dedupe + format 自适应。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio.services.dataset import tagedit


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


def test_read_missing_returns_empty(train_dir: Path) -> None:
    f = _img(train_dir / "5_a", "noaption.png")
    assert tagedit.read_tags(f) == []


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
