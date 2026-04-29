"""PP4 — WD14 tagger：mock onnx + filesystem 验证模型解析、preprocess、postprocess。"""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from studio import secrets
from studio.services import wd14_tagger


@pytest.fixture
def isolated_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sf = tmp_path / "secrets.json"
    monkeypatch.setattr(secrets, "SECRETS_FILE", sf)
    return tmp_path


def _make_local_model(model_dir: Path, tags: list[tuple[str, int]]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.onnx").write_bytes(b"fake-onnx")
    with open(model_dir / "selected_tags.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag_id", "name", "category"])
        for i, (n, c) in enumerate(tags):
            w.writerow([i, n, c])


def test_resolve_local_dir(isolated_secrets: Path) -> None:
    model_dir = isolated_secrets / "models" / "x"
    _make_local_model(model_dir, [("a", 0)])
    secrets.update({"wd14": {"local_dir": str(model_dir)}})
    t = wd14_tagger.WD14Tagger()
    resolved = t._resolve_model_dir()
    assert resolved == model_dir


def test_resolve_local_dir_missing_files_raises(isolated_secrets: Path) -> None:
    bad = isolated_secrets / "bad"
    bad.mkdir()
    secrets.update({"wd14": {"local_dir": str(bad)}})
    t = wd14_tagger.WD14Tagger()
    with pytest.raises(FileNotFoundError, match="缺少"):
        t._resolve_model_dir()


def test_postprocess_filters_by_threshold(isolated_secrets: Path) -> None:
    secrets.update({
        "wd14": {
            "threshold_general": 0.5,
            "threshold_character": 0.85,
            "blacklist_tags": ["banned"],
        }
    })
    t = wd14_tagger.WD14Tagger()
    t._tags = ["1girl", "solo", "banned", "char_a", "rating"]
    t._tag_categories = [0, 0, 0, 4, 9]  # 9 = rating, 4 = character
    scores = np.array([0.9, 0.4, 0.99, 0.7, 0.95])
    tags, raw = t._postprocess_one(scores)
    # 1girl (0.9 > 0.5 ✓), solo (0.4 < 0.5 ✗), banned (blacklist),
    # char_a (0.7 < 0.85 ✗), rating (cat=9 → drop)
    assert tags == ["1girl"]
    assert raw == {"1girl": pytest.approx(0.9)}


def test_postprocess_sorts_by_score_desc(isolated_secrets: Path) -> None:
    secrets.update({"wd14": {"threshold_general": 0.1, "threshold_character": 0.1}})
    t = wd14_tagger.WD14Tagger()
    t._tags = ["a", "b", "c"]
    t._tag_categories = [0, 0, 0]
    scores = np.array([0.3, 0.9, 0.5])
    tags, _ = t._postprocess_one(scores)
    assert tags == ["b", "c", "a"]


def test_preprocess_pads_to_square(isolated_secrets: Path) -> None:
    t = wd14_tagger.WD14Tagger()
    t._input_size = 16  # 小尺寸方便看
    img = Image.new("RGB", (10, 4), (255, 0, 0))
    arr = t._preprocess(img)
    # PP8: _preprocess 改成单图 [H, W, 3]，调用方负责 stack 成 batch
    assert arr.shape == (16, 16, 3)
    # BGR：第三通道是 R（值 255）
    assert arr[8, 8, 2] == pytest.approx(255.0, abs=1.0)


def test_overrides_replace_thresholds(isolated_secrets: Path) -> None:
    """overrides 应在内存里盖过 secrets 的 threshold，不写盘。"""
    secrets.update({
        "wd14": {
            "threshold_general": 0.5,
            "threshold_character": 0.85,
            "blacklist_tags": [],
        }
    })
    t = wd14_tagger.WD14Tagger(
        overrides={"threshold_general": 0.2, "blacklist_tags": ["solo"]}
    )
    t._tags = ["1girl", "solo", "rare"]
    t._tag_categories = [0, 0, 0]
    scores = np.array([0.3, 0.9, 0.25])
    tags, _ = t._postprocess_one(scores)
    # threshold_general 被压到 0.2 → 1girl(0.3) / rare(0.25) 都过；solo 被新 blacklist 屏蔽
    assert sorted(tags) == ["1girl", "rare"]
    # 全局没被改写
    assert secrets.load().wd14.threshold_general == 0.5
    assert secrets.load().wd14.blacklist_tags == []


def test_overrides_none_falls_back_to_global(isolated_secrets: Path) -> None:
    """overrides 中字段为 None / 缺失 → 沿用全局 settings。"""
    secrets.update({"wd14": {"threshold_general": 0.7}})
    t = wd14_tagger.WD14Tagger(overrides={"threshold_general": None})
    cfg = t._cfg()
    assert cfg.threshold_general == 0.7


def test_overrides_redirect_local_dir(isolated_secrets: Path) -> None:
    """override 的 local_dir 改路径解析，不改全局。"""
    a = isolated_secrets / "a"
    b = isolated_secrets / "b"
    _make_local_model(a, [("x", 0)])
    _make_local_model(b, [("y", 0)])
    secrets.update({"wd14": {"local_dir": str(a)}})

    # 不带 overrides → 走 a
    assert wd14_tagger.WD14Tagger()._resolve_model_dir() == a
    # 带 override → 走 b，全局仍是 a
    assert (
        wd14_tagger.WD14Tagger(overrides={"local_dir": str(b)})._resolve_model_dir()
        == b
    )
    assert secrets.load().wd14.local_dir == str(a)


def test_tag_iterator_handles_io_error(
    isolated_secrets: Path, tmp_path: Path
) -> None:
    """传入不存在的文件 → yield error，不抛错。"""
    t = wd14_tagger.WD14Tagger()
    # mock prepare 不真做；CPU EP 让 batch 强制 1
    t._session = MagicMock()
    t._session.get_providers.return_value = ["CPUExecutionProvider"]
    t._session.run.return_value = (np.array([[0.9]]),)
    t._tags = ["x"]
    t._tag_categories = [0]
    t._input_name = "input"
    t._input_size = 4

    secrets.update({"wd14": {"threshold_general": 0.1, "threshold_character": 0.1}})

    # 一张存在，一张不存在
    good = tmp_path / "good.png"
    Image.new("RGB", (8, 8)).save(good)
    bad = tmp_path / "ghost.png"

    results = list(t.tag([good, bad]))
    assert len(results) == 2
    assert results[0]["tags"] == ["x"]
    assert "error" in results[1]


# ---------------------------------------------------------------------------
# PP8 — batch 推理
# ---------------------------------------------------------------------------


def test_tag_batch_processes_chunks(
    isolated_secrets: Path, tmp_path: Path
) -> None:
    """batch_size=4 + 5 张图 → session.run 被调 2 次（4 + 1）。"""
    secrets.update({
        "wd14": {
            "threshold_general": 0.1,
            "threshold_character": 0.1,
            "batch_size": 4,
        }
    })
    t = wd14_tagger.WD14Tagger()
    t._session = MagicMock()
    t._session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def _fake_run(_outputs, feeds):
        # feeds["input"] shape: (N, 4, 4, 3) → 返回 (N, 1)，第 i 张图分数 = 0.5 + i*0.1
        n = feeds["input"].shape[0]
        return (np.array([[0.5 + i * 0.1] for i in range(n)]),)

    t._session.run.side_effect = _fake_run
    t._tags = ["x"]
    t._tag_categories = [0]
    t._input_name = "input"
    t._input_size = 4

    paths = []
    for i in range(5):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8), (i * 30, 0, 0)).save(p)
        paths.append(p)

    results = list(t.tag(paths))
    assert len(results) == 5
    assert all(r["tags"] == ["x"] for r in results)
    # 第一批 4 张同时塞 → 5 张拆 batch=4 → 2 次 run
    assert t._session.run.call_count == 2
    first_call_batch = t._session.run.call_args_list[0][0][1]["input"]
    assert first_call_batch.shape == (4, 4, 4, 3)
    second_call_batch = t._session.run.call_args_list[1][0][1]["input"]
    assert second_call_batch.shape == (1, 4, 4, 3)


def test_tag_batch_falls_back_to_one_on_cpu(
    isolated_secrets: Path, tmp_path: Path
) -> None:
    """CPU EP → batch_size 强制 1（即使 secrets 配 8），每张图独立 run。"""
    secrets.update({"wd14": {"threshold_general": 0.1, "batch_size": 8}})
    t = wd14_tagger.WD14Tagger()
    t._session = MagicMock()
    t._session.get_providers.return_value = ["CPUExecutionProvider"]
    t._session.run.return_value = (np.array([[0.9]]),)
    t._tags = ["x"]
    t._tag_categories = [0]
    t._input_name = "input"
    t._input_size = 4

    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8)).save(p)
        paths.append(p)

    list(t.tag(paths))
    assert t._session.run.call_count == 3  # 每张独立 run
    for call in t._session.run.call_args_list:
        assert call[0][1]["input"].shape == (1, 4, 4, 3)


def test_tag_batch_keeps_decode_errors(
    isolated_secrets: Path, tmp_path: Path
) -> None:
    """batch 中混了一张坏图 → 该图 yield error，其余正常推理。"""
    secrets.update({"wd14": {"threshold_general": 0.1, "batch_size": 4}})
    t = wd14_tagger.WD14Tagger()
    t._session = MagicMock()
    t._session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    t._session.run.return_value = (np.array([[0.9], [0.95]]),)
    t._tags = ["x"]
    t._tag_categories = [0]
    t._input_name = "input"
    t._input_size = 4

    good1 = tmp_path / "g1.png"
    good2 = tmp_path / "g2.png"
    bad = tmp_path / "missing.png"
    Image.new("RGB", (8, 8)).save(good1)
    Image.new("RGB", (8, 8)).save(good2)

    results = list(t.tag([good1, bad, good2]))
    assert len(results) == 3
    assert results[0]["tags"] == ["x"]
    assert "error" in results[1]
    assert results[2]["tags"] == ["x"]
    # 推理只塞了 2 张 → batch shape (2, ...)
    assert t._session.run.call_count == 1
    fed = t._session.run.call_args[0][1]["input"]
    assert fed.shape == (2, 4, 4, 3)


# ---------------------------------------------------------------------------
# PP10 — preprocess 并发
# ---------------------------------------------------------------------------


def test_preprocess_runs_concurrently_on_gpu_ep(
    isolated_secrets: Path, tmp_path: Path
) -> None:
    """batch > 1 + GPU EP → preprocess 用 wd14-prep 线程池跑，多线程同时活跃。

    用 Barrier(N) 强制 N 个 preprocess 必须并发才能解锁；如果是串行会 deadlock 卡 timeout。
    """
    import threading

    secrets.update({"wd14": {"threshold_general": 0.1, "batch_size": 4}})
    t = wd14_tagger.WD14Tagger()
    t._session = MagicMock()
    t._session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    t._session.run.return_value = (np.array([[0.9], [0.9], [0.9], [0.9]]),)
    t._tags = ["x"]
    t._tag_categories = [0]
    t._input_name = "input"
    t._input_size = 4

    barrier = threading.Barrier(parties=4, timeout=2.0)
    seen_threads: set[str] = set()
    seen_threads_lock = threading.Lock()

    real_preprocess = t._preprocess

    def gated_preprocess(img):
        with seen_threads_lock:
            seen_threads.add(threading.current_thread().name)
        # 必须 4 个 worker 同时到达才放行 —— 串行会 timeout
        barrier.wait()
        return real_preprocess(img)

    t._preprocess = gated_preprocess  # type: ignore[method-assign]

    paths = []
    for i in range(4):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8)).save(p)
        paths.append(p)

    results = list(t.tag(paths))
    assert len(results) == 4
    assert all(r["tags"] == ["x"] for r in results)
    # 至少 2 个 wd14-prep 线程参与（实际应该 4 个，但 OS 调度允许 race）
    prep_threads = {n for n in seen_threads if n.startswith("wd14-prep")}
    assert len(prep_threads) >= 2, f"expected concurrent preprocess threads, saw {seen_threads}"


def test_cpu_ep_path_does_not_spawn_pool(
    isolated_secrets: Path, tmp_path: Path
) -> None:
    """CPU EP（batch_size 强制 1） → 不开 ThreadPool，行为与 PP8 完全一致。"""
    import threading

    secrets.update({"wd14": {"threshold_general": 0.1, "batch_size": 8}})
    t = wd14_tagger.WD14Tagger()
    t._session = MagicMock()
    t._session.get_providers.return_value = ["CPUExecutionProvider"]
    t._session.run.return_value = (np.array([[0.9]]),)
    t._tags = ["x"]
    t._tag_categories = [0]
    t._input_name = "input"
    t._input_size = 4

    seen_threads: set[str] = set()
    real_preprocess = t._preprocess

    def tracking_preprocess(img):
        seen_threads.add(threading.current_thread().name)
        return real_preprocess(img)

    t._preprocess = tracking_preprocess  # type: ignore[method-assign]

    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8)).save(p)
        paths.append(p)

    list(t.tag(paths))
    # 全部应该在主线程跑 —— 没 wd14-prep 线程
    prep_threads = {n for n in seen_threads if n.startswith("wd14-prep")}
    assert prep_threads == set(), f"CPU 路径不应开 pool，saw {seen_threads}"


def test_preprocess_concurrent_preserves_chunk_order(
    isolated_secrets: Path, tmp_path: Path
) -> None:
    """并发 preprocess 后 yield 顺序仍按 chunk 输入顺序。"""
    secrets.update({"wd14": {"threshold_general": 0.1, "batch_size": 4}})
    t = wd14_tagger.WD14Tagger()
    t._session = MagicMock()
    t._session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # 4 张图，session.run 的输出按 batch 顺序 (chunk 顺序)
    t._session.run.return_value = (
        np.array([[0.5], [0.6], [0.7], [0.8]]),
    )
    t._tags = ["x"]
    t._tag_categories = [0]
    t._input_name = "input"
    t._input_size = 4

    paths = []
    for i in range(4):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8), (i * 60, 0, 0)).save(p)
        paths.append(p)

    results = list(t.tag(paths))
    # 顺序：results[k]['image'] == paths[k]
    for k, r in enumerate(results):
        assert r["image"] == paths[k]
