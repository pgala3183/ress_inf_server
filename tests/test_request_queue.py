"""Tests for priority-aware request queue behavior."""

from __future__ import annotations

import asyncio
import time

import pytest

from server import request_queue as rq
from server.request_queue import RequestQueue


@pytest.mark.asyncio
async def test_get_next_batch_prefers_interactive_over_batch() -> None:
    queue = RequestQueue()
    loop = asyncio.get_running_loop()

    await queue.enqueue("batch-1", loop.create_future(), "batch")
    await queue.enqueue("interactive-1", loop.create_future(), "interactive")
    await queue.enqueue("interactive-2", loop.create_future(), "interactive")

    batch = await queue.get_next_batch(10)
    assert [item.text for item in batch] == ["interactive-1", "interactive-2", "batch-1"]


@pytest.mark.asyncio
async def test_promoted_batch_preempts_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rq, "BATCH_SLA_MS", 100)
    monkeypatch.setattr(rq, "PROMOTION_FRACTION", 0.80)

    queue = RequestQueue()
    loop = asyncio.get_running_loop()

    await queue.enqueue("batch-starved", loop.create_future(), "batch")
    await asyncio.sleep(0.085)
    await queue.enqueue("interactive-new", loop.create_future(), "interactive")

    batch = await queue.get_next_batch(10)
    assert [item.text for item in batch] == ["batch-starved", "interactive-new"]
