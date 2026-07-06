"""Shared drain lifecycle state for Spot preemption and graceful shutdown."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from enum import Enum

logger = logging.getLogger("resilient.drain")


class DrainState(str, Enum):
    RUNNING = "RUNNING"
    DRAINING = "DRAINING"
    DRAINED = "DRAINED"


class DrainManager:
    """Coordinates graceful drain across the API, scheduler, and preemption listener."""

    def __init__(self) -> None:
        self._state = DrainState.RUNNING
        self._lock = asyncio.Lock()
        self._drain_started_at: float | None = None
        self._drained_event = asyncio.Event()
        self._drain_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> DrainState:
        return self._state

    def accepts_new_requests(self) -> bool:
        return self._state == DrainState.RUNNING

    def log_transition(self, new_state: DrainState, reason: str, **extra: object) -> None:
        payload = {
            "event": "drain_state_transition",
            "from_state": self._state.value,
            "to_state": new_state.value,
            "reason": reason,
            **extra,
        }
        logger.info(json.dumps(payload))

    async def begin_drain(self, reason: str) -> bool:
        """Idempotent entry into DRAINING. Returns True if this call started drain."""
        async with self._lock:
            if self._state != DrainState.RUNNING:
                logger.info(
                    json.dumps(
                        {
                            "event": "drain_already_in_progress",
                            "state": self._state.value,
                            "reason": reason,
                        }
                    )
                )
                return False
            self._state = DrainState.DRAINING
            self._drain_started_at = time.perf_counter()
            self.log_transition(DrainState.DRAINING, reason)
            return True

    async def mark_drained(self, reason: str) -> None:
        async with self._lock:
            if self._state == DrainState.DRAINED:
                return
            elapsed = None
            if self._drain_started_at is not None:
                elapsed = time.perf_counter() - self._drain_started_at
            self._state = DrainState.DRAINED
            self._drained_event.set()
            self.log_transition(
                DrainState.DRAINED,
                reason,
                drain_elapsed_seconds=elapsed,
            )

    async def wait_drained(self, timeout: float | None = None) -> bool:
        try:
            if timeout is None:
                await self._drained_event.wait()
                return True
            await asyncio.wait_for(self._drained_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    @property
    def drain_elapsed_seconds(self) -> float | None:
        if self._drain_started_at is None:
            return None
        return time.perf_counter() - self._drain_started_at
