"""Async request queue for dynamic batching."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class QueuedRequest:
    text: str
    future: asyncio.Future[dict[str, Any]]


class RequestQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[QueuedRequest] = asyncio.Queue()

    async def enqueue(self, text: str, future: asyncio.Future[dict[str, Any]]) -> None:
        await self._queue.put(QueuedRequest(text=text, future=future))

    async def pull_batch(self, max_size: int, wait_ms: float) -> list[QueuedRequest]:
        """Return up to max_size items, waiting at most wait_ms after the first item."""
        first = await self._queue.get()
        batch = [first]
        deadline = time.monotonic() + (wait_ms / 1000.0)

        while len(batch) < max_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            batch.append(item)

        return batch

    @property
    def depth(self) -> int:
        return self._queue.qsize()
