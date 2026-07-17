"""daemon 任务级 prompt 预编码（_precache_prompts_and_offload）。

XY / 多 prompt generate 的 prompt 集合任务开始前封闭：krea2 先全部编进
在线 LRU 再卸 TE（训练两段式的推理版）；anima 文本栈无此 API 安全跳过；
performance 档不卸；预编码异常由逐格惰性路径兜底不阻塞任务。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "runtime"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import anima_daemon  # noqa: E402


class _Stack:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.precached: list[list[str]] = []
        self.offloaded = 0

    def precache_online_prompts(self, prompts):
        if self.fail:
            raise RuntimeError("boom")
        self.precached.append(list(prompts))
        return len(prompts)

    def offload_model(self):
        self.offloaded += 1


def _with_stack(monkeypatch, stack):
    monkeypatch.setattr(anima_daemon.CACHE, "text_stack", stack, raising=False)
    return stack


def test_precache_encodes_and_offloads(monkeypatch):
    stack = _with_stack(monkeypatch, _Stack())
    phases: list[str] = []

    anima_daemon._precache_prompts_and_offload(
        ["p", "neg"], "auto", phases.append,
    )

    assert stack.precached == [["p", "neg"]]
    assert stack.offloaded == 1
    assert phases == ["clip"]


def test_precache_performance_policy_keeps_te_resident(monkeypatch):
    stack = _with_stack(monkeypatch, _Stack())

    anima_daemon._precache_prompts_and_offload(["p"], "performance")

    assert stack.precached == [["p"]]
    assert stack.offloaded == 0


def test_precache_failure_falls_back_without_offload(monkeypatch):
    stack = _with_stack(monkeypatch, _Stack(fail=True))

    anima_daemon._precache_prompts_and_offload(["p"], "auto")  # 不抛

    assert stack.offloaded == 0


def test_precache_skips_stacks_without_api(monkeypatch):
    """anima 文本栈是 (qwen, tok, t5_tok) tuple——无 API 直接跳过。"""
    monkeypatch.setattr(
        anima_daemon.CACHE, "text_stack",
        (SimpleNamespace(), SimpleNamespace(), SimpleNamespace()),
        raising=False,
    )
    anima_daemon._precache_prompts_and_offload(["p"], "auto")  # 不抛
