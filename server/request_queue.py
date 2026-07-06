"""Async priority-aware request queue for dynamic batching."""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

Priority = Literal["interactive", "batch"]

INTERACTIVE_SLA_MS = 200
BATCH_SLA_MS = 5000
PROMOTION_FRACTION = float(os.environ.get("SLA_PROMOTION_FRACTION", "0.80"))


@dataclass
class QueuedRequest:
    text: str
    future: asyncio.Future[dict[str, Any]]
    priority: Priority
    max_tokens: int = 50
    enqueue_time: float = field(default_factory=time.monotonic)


class RequestQueue:
    def __init__(self) -> None:
        self._interactive: deque[QueuedRequest] = deque()
        self._batch: deque[QueuedRequest] = deque()
        self._condition = asyncio.Condition()

    async def enqueue(
        self,
        text: str,
        future: asyncio.Future[dict[str, Any]],
        priority: Priority = "interactive",
        max_tokens: int = 50,
    ) -> None:
        item = QueuedRequest(text=text, future=future, priority=priority, max_tokens=max_tokens)
        async with self._condition:
            if priority == "interactive":
                self._interactive.append(item)
            else:
                self._batch.append(item)
            self._condition.notify_all()

    def _promotion_threshold_ms(self) -> float:
        return BATCH_SLA_MS * PROMOTION_FRACTION

    def _waiting_ms(self, item: QueuedRequest) -> float:
        return (time.monotonic() - item.enqueue_time) * 1000.0

    def _is_promotion_eligible(self, item: QueuedRequest) -> bool:
        return item.priority == "batch" and self._waiting_ms(item) >= self._promotion_threshold_ms()

    def _has_ready(self) -> bool:
        if self._interactive:
            return True
        if self._batch:
            return True
        return False

    def _collect_batch(self, max_size: int) -> list[QueuedRequest]:
        if max_size <= 0:
            return []

        batch: list[QueuedRequest] = []

        promoted: list[QueuedRequest] = []
        remaining: deque[QueuedRequest] = deque()
        for item in self._batch:
            if self._is_promotion_eligible(item) and len(batch) + len(promoted) < max_size:
                promoted.append(item)
            else:
                remaining.append(item)
        self._batch = remaining
        batch.extend(promoted)

        while self._interactive and len(batch) < max_size:
            batch.append(self._interactive.popleft())

        while self._batch and len(batch) < max_size:
            batch.append(self._batch.popleft())

        return batch

    async def try_collect_batch(self, max_size: int) -> list[QueuedRequest]:
        """Non-blocking priority-aware drain (returns [] when queue is empty)."""
        async with self._condition:
            if not self._has_ready():
                return []
            return self._collect_batch(max_size)

    async def get_next_batch(self, max_size: int) -> list[QueuedRequest]:
        """Block until work is available, then drain by priority rules."""
        async with self._condition:
            await self._condition.wait_for(self._has_ready)
            return self._collect_batch(max_size)

    async def pull_batch(self, max_size: int, wait_ms: float) -> list[QueuedRequest]:
        """Collect a batch, waiting up to wait_ms after the first items for more."""
        batch = await self.get_next_batch(max_size)
        if len(batch) >= max_size:
            return batch

        deadline = time.monotonic() + (wait_ms / 1000.0)
        while len(batch) < max_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            async with self._condition:
                extra = self._collect_batch(max_size - len(batch))
                if extra:
                    batch.extend(extra)
                    continue

                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    break

                batch.extend(self._collect_batch(max_size - len(batch)))

        return batch

    @property
    def depth(self) -> int:
        return len(self._interactive) + len(self._batch)
