"""EventBus 连接首/末钩子（commit 11，generate cache 断连清依赖）。"""
from __future__ import annotations

import asyncio

from studio.infrastructure.event_bus import EventBus


def test_first_subscribe_callback_only_on_zero_to_one() -> None:
    async def _run() -> None:
        bus = EventBus()
        bus.attach_loop(asyncio.get_running_loop())
        first_calls = 0
        last_calls = 0

        def on_first() -> None:
            nonlocal first_calls
            first_calls += 1

        def on_last() -> None:
            nonlocal last_calls
            last_calls += 1

        bus.set_connection_callbacks(on_first, on_last)

        q1 = await bus.subscribe()
        assert first_calls == 1 and last_calls == 0

        # 第二个连接不应再次触发 first
        q2 = await bus.subscribe()
        assert first_calls == 1

        # 解一个还有连接 → 不触发 last
        bus.unsubscribe(q1)
        assert last_calls == 0

        # 解最后一个 → 触发 last
        bus.unsubscribe(q2)
        assert last_calls == 1

        # 重新订阅应再次触发 first
        q3 = await bus.subscribe()
        assert first_calls == 2
        bus.unsubscribe(q3)
        assert last_calls == 2

    asyncio.run(_run())


def test_connection_count() -> None:
    async def _run() -> None:
        bus = EventBus()
        bus.attach_loop(asyncio.get_running_loop())
        assert bus.connection_count() == 0
        q1 = await bus.subscribe()
        q2 = await bus.subscribe()
        assert bus.connection_count() == 2
        bus.unsubscribe(q1)
        assert bus.connection_count() == 1
        bus.unsubscribe(q2)
        assert bus.connection_count() == 0

    asyncio.run(_run())


def test_no_callbacks_set_does_not_crash() -> None:
    """未设钩子应该 noop，不该 NPE。"""
    async def _run() -> None:
        bus = EventBus()
        bus.attach_loop(asyncio.get_running_loop())
        q = await bus.subscribe()
        bus.unsubscribe(q)

    asyncio.run(_run())
