"""指标模型的阶段级复用（`eval_model_pool` + 各 runner 的 `shared_scorer`）。

`_stage_metric` 的形状本来就是「一个指标跑完所有候选再换下一个」，但 run_*_job 里
的 `_default_scorer` 是旧模型（每候选一个子进程）留下的：模型写在函数体里加载，于是
200 个 checkpoint 就加载 200 次。这里锁住修好之后的不变量。
"""
from __future__ import annotations

import pathlib
from typing import Any

import pytest

from studio.services import eval_ccip, eval_clip, eval_dino, eval_model_pool, eval_tag


def test_pool_reuses_on_hit_and_reloads_on_key_change() -> None:
    pool = eval_model_pool.ModelPool("probe")
    loads: list[str] = []

    def _load(name: str):
        loads.append(name)
        return {"name": name}

    assert pool.get("a", lambda: _load("a"))["name"] == "a"
    assert pool.get("a", lambda: _load("a"))["name"] == "a"
    assert loads == ["a"], "同 key 必须复用，不能重新加载"

    # 换模型名 → 先释放旧的再加载新的
    assert pool.get("b", lambda: _load("b"))["name"] == "b"
    assert loads == ["a", "b"]


def test_pool_release_is_idempotent_and_forgets_the_handle() -> None:
    pool = eval_model_pool.ModelPool("probe")
    pool.get("a", lambda: object())
    assert pool.loaded

    pool.release()
    assert not pool.loaded
    pool.release()  # 再调一次是 no-op，不该抛

    loads: list[int] = []
    pool.get("a", lambda: loads.append(1) or object())
    assert loads == [1], "release 之后同 key 也要重新加载"


def test_release_reports_through_progress() -> None:
    pool = eval_model_pool.ModelPool("clip")
    pool.get("a", lambda: object())
    lines: list[str] = []
    pool.release(lines.append)
    # 显存编排内部动作 → DEBUG 英文排障行（不进 msg_id 字典）
    assert any("model released" in line for line in lines)
    # 没加载过时不该刷无意义的日志
    lines.clear()
    eval_model_pool.ModelPool("clip").release(lines.append)
    assert lines == []


@pytest.mark.parametrize(
    "module, label",
    [
        (eval_clip, "clip"),
        (eval_dino, "dino"),
        (eval_ccip, "ccip"),
        (eval_tag, "tag"),
    ],
)
def test_shared_scorer_loads_once_across_candidates(
    module: Any, label: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一个阶段内跑 N 个候选，模型只 load 一次；阶段结束释放。"""
    loads: list[str] = []
    loader_name = {
        "clip": "_load_clip", "dino": "_load_dino",
        "ccip": "_load_ccip", "tag": "_load_tagger",
    }[label]

    def _fake_load(*_a, **_kw):
        loads.append(label)
        return ("handle",) * 5  # 各 runner 解包元数不同，给足

    monkeypatch.setattr(module, loader_name, _fake_load)

    with module.shared_scorer() as scorer:
        # scorer 是绑好 pool 的 _default_scorer；直接驱动它的取模型那一步
        pool = scorer.keywords["pool"]
        for _ in range(5):
            pool.get("m", lambda: _fake_load())
        assert loads == [label], "阶段内应只加载一次"
        assert pool.loaded

    assert not pool.loaded, "阶段结束必须释放，否则跑下一个指标时它还占着卡"


def test_default_scorer_actually_uses_the_injected_pool(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """接缝本身：`_default_scorer(pool=...)` 必须从传进来的池子取模型。

    上面那些只验了池子的行为；这条验的是 run_*_job → _default_scorer 这一段真的把
    模型取自阶段级池子 —— 断了的话「加载一次」就退化回「每候选一次」，而且没有任何
    测试会红。

    取模型之前的几步（读 run / 找图 / 找参考图）与本主张无关，一并打掉。
    """
    loads: list[int] = []
    monkeypatch.setattr(eval_clip, "_run_eval_root", lambda _run: None)
    monkeypatch.setattr(eval_clip, "_done_image_items", lambda *_a, **_kw: [])
    monkeypatch.setattr(eval_clip, "_reference_paths", lambda *_a, **_kw: {})
    monkeypatch.setattr(
        eval_clip, "_load_clip",
        lambda *_a, **_kw: loads.append(1) or (object(), object(), "cpu"),
    )
    # 取模型**之后**的打分与落盘同样与本主张无关
    import torch

    empty = torch.zeros((0, 4))
    monkeypatch.setattr(eval_clip, "_encode_images", lambda *_a, **_kw: empty)
    monkeypatch.setattr(eval_clip, "_encode_texts", lambda *_a, **_kw: empty)
    monkeypatch.setattr(eval_clip, "_write_cache_metadata", lambda *_a, **_kw: None)

    pool = eval_model_pool.ModelPool("clip")
    # 同一个池子连跑两次 = 两个候选
    for _ in range(2):
        eval_clip._default_scorer(
            {"run_id": "r1"}, tmp_path, "m", lambda _l: None, pool=pool,
        )

    assert loads == [1], "第二个候选必须复用池子里的模型"
    assert pool.loaded, "模型该留在池子里等下一个候选，由阶段结束时统一释放"


@pytest.mark.parametrize(
    "module", [eval_clip, eval_dino, eval_ccip, eval_tag],
)
def test_standalone_job_still_gets_a_fresh_pool(module: Any) -> None:
    """`run_*_job(scorer=None)` 独立调用时行为不变：自己的一次性池，用完即散。

    没有模块级全局池 —— 那会跨测试泄漏（成组跑时上一个用例的模型被下一个复用），
    也会让模型一直留到进程退出、跑下一个指标时仍占着显存。
    """
    assert not hasattr(module, "_POOL"), "不该有模块级全局池"
