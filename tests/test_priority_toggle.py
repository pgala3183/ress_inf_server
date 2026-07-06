"""Tests for FIFO priority toggle used in SLA comparison benchmarks."""

from __future__ import annotations

import asyncio

import pytest

from server.request_queue import RequestQueue


@pytest.mark.asyncio
async def test_fifo_mode_drains_by_enqueue_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIORITY_SCHEDULING", "0")
    # Re-import config flag used by RequestQueue
    import server.config as config
    import server.request_queue as rq

    monkeypatch.setattr(config, "PRIORITY_SCHEDULING", False)
    monkeypatch.setattr(rq, "PRIORITY_SCHEDULING", False)

    queue = RequestQueue()
    loop = asyncio.get_running_loop()

    async def enqueue(text: str, priority: str) -> None:
        future = loop.create_future()
        await queue.enqueue(text, future, priority)  # type: ignore[arg-type]

    await enqueue("first", "batch")
    await enqueue("second", "interactive")
    batch = await queue.try_collect_batch(2)
    assert [item.text for item in batch] == ["first", "second"]
