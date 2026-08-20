"""`utils.log_throttle` 三个节流器的语义（verdicts-b-subprocess.md §4.2）。

这三个类被 reg_ai / caption_utils / 4 个 worker / Automagic2 共用，语义错了
会在故障路径上把 run.log 冲成噪音（或反过来把首条告警吞掉），单独锁一遍。
"""
from __future__ import annotations

from utils.log_throttle import BackoffThrottle, ProgressThrottle, RepeatThrottle


class _Recorder:
    """只实现 TaskLogLike 里节流器用到的两个级别方法。"""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []
        self.exc_infos: list[bool] = []

    def debug(self, msg: str, *args: object) -> None:
        self.lines.append(("DEBUG", msg % args if args else msg))

    def warning(self, msg: str, *args: object, exc_info: bool = False) -> None:
        self.lines.append(("WARNING", msg % args if args else msg))
        self.exc_infos.append(exc_info)


# ── 方案 A：RepeatThrottle ──────────────────────────────────────────────


def test_repeat_first_full_then_debug_then_summary() -> None:
    """首条全文 WARNING、2..N 条降 DEBUG、drain 补一条计数汇总。"""
    log = _Recorder()
    throttle = RepeatThrottle(log)
    for name in ("a.png", "b.png", "c.png"):
        throttle.hit(
            "gone",
            "%d images skipped: the source file was gone (first: %s)",
            "Image skipped: %s no longer exists",
            name,
            first=name,
        )
    throttle.drain()

    assert [lv for lv, _ in log.lines] == ["WARNING", "DEBUG", "DEBUG", "WARNING"]
    assert log.lines[0][1] == "Image skipped: a.png no longer exists"
    # 汇总带总数与**首个**样例（不是最后一个）
    assert log.lines[-1][1] == (
        "3 images skipped: the source file was gone (first: a.png)"
    )


def test_repeat_single_hit_has_no_summary() -> None:
    """只触发一次时首条就是全部信息，drain 不再重复一条汇总。"""
    log = _Recorder()
    throttle = RepeatThrottle(log)
    throttle.hit("gone", "%d skipped (first: %s)", "Image skipped: %s", "solo.png",
                 first="solo.png")
    throttle.drain()

    assert [lv for lv, _ in log.lines] == ["WARNING"]


def test_repeat_summary_without_sample_takes_only_count() -> None:
    """不带 first 的汇总串只有一个 %d（reg_ai 的批预编码两条就是这形态）。"""
    log = _Recorder()
    throttle = RepeatThrottle(log)
    for _ in range(3):
        throttle.hit(
            "vram",
            "Batch pre-encoding was skipped %d times",
            "Skipping batch pre-encoding: %.1f GB VRAM free",
            3.5,
        )
    throttle.drain()

    assert log.lines[-1] == ("WARNING", "Batch pre-encoding was skipped 3 times")


def test_repeat_groups_are_independent_and_exc_info_only_on_first() -> None:
    """不同 key 各自计数；traceback 只挂首条（C6：每条都带会淹掉 run.log）。"""
    log = _Recorder()
    throttle = RepeatThrottle(log)
    throttle.hit("a", "%d a (first: %s)", "a failed: %s", "x", first="x", exc_info=True)
    throttle.hit("a", "%d a (first: %s)", "a failed: %s", "y", first="y", exc_info=True)
    throttle.hit("b", "%d b (first: %s)", "b failed: %s", "z", first="z")

    assert throttle.count("a") == 2
    assert throttle.count("b") == 1
    # 首条 WARNING 带 traceback；b 的首条不带
    assert log.exc_infos == [True, False]

    throttle.drain()
    summaries = [ln for lv, ln in log.lines if lv == "WARNING"]
    assert "2 a (first: x)" in summaries
    assert not [s for s in summaries if s.startswith("1 b")]


def test_repeat_drain_resets_state() -> None:
    """drain 后计数清零——同一进程里跑第二个任务不会继承上个任务的数。"""
    log = _Recorder()
    throttle = RepeatThrottle(log)
    for _ in range(2):
        throttle.hit("k", "%d hit", "hit")
    throttle.drain()
    assert throttle.count("k") == 0

    log.lines.clear()
    throttle.hit("k", "%d hit", "hit")
    assert [lv for lv, _ in log.lines] == ["WARNING"]  # 又是首条，不是 DEBUG


# ── 方案 B：ProgressThrottle ────────────────────────────────────────────


def test_progress_collapses_1000_items_to_about_100_lines() -> None:
    """1000 张图的进度行从 1000 条收到 ~100 条，且首末必发。"""
    throttle = ProgressThrottle(1000, min_interval=10_000)  # 关掉时间维度
    emitted = [i for i in range(1, 1001) if throttle.should_emit(i)]

    assert emitted[0] == 1
    assert emitted[-1] == 1000
    assert 90 <= len(emitted) <= 110


def test_progress_small_task_emits_every_item() -> None:
    """总数 < 100 时 step 退化成 1，每条都发（小任务不该只看到首末两行）。"""
    throttle = ProgressThrottle(3, min_interval=10_000)
    assert [i for i in range(1, 4) if throttle.should_emit(i)] == [1, 2, 3]


def test_progress_time_window_emits_between_step_boundaries() -> None:
    """步长没到但时间窗到了也要发——慢任务不能长时间零输出。"""
    throttle = ProgressThrottle(1000, min_interval=0.0)
    assert throttle.should_emit(7) is True


def test_progress_tolerates_zero_total() -> None:
    """total=0（空任务）不能除零；此时每次调用都放行，由调用方决定发不发。"""
    throttle = ProgressThrottle(0)
    assert throttle.should_emit(1) is True


# ── 方案 C-2：BackoffThrottle ───────────────────────────────────────────


def test_backoff_reports_at_first_then_powers_of_ten() -> None:
    """十万量级的 NaN 风暴收成个位数条告警：第 1、10、100、1000… 次。"""
    throttle = BackoffThrottle()
    marks = []
    for _ in range(1200):
        count, kind = throttle.tick()
        if kind:
            marks.append((count, kind))

    assert marks == [
        (1, "first"), (10, "milestone"), (100, "milestone"), (1000, "milestone"),
    ]
    assert throttle.count == 1200


def test_backoff_silent_ticks_still_count() -> None:
    """静默区间照常累加，milestone 报的是累计数不是区间数。"""
    throttle = BackoffThrottle()
    for _ in range(9):
        throttle.tick()
    count, kind = throttle.tick()
    assert (count, kind) == (10, "milestone")
