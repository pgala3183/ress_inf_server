"""Background scheduler: static batching (classification) or continuous slots (generative).

Continuous batching loop (Phase 4)
----------------------------------
Production systems like vLLM keep a fixed pool of GPU "slots". Each slot holds one
in-flight sequence. When a sequence finishes decoding, its slot is immediately
reassigned to a waiting request — the remaining sequences do NOT wait for the
slowest member of the original batch.

What we implement here is that *scheduling* pattern only:
  1. Resolve finished sequences → complete their Futures → free slots
  2. Fill empty slots from RequestQueue (Phase 3 priority rules)
  3. Call model_worker.step() once for all active sequences
  4. Repeat

What we deliberately do NOT implement (see docs/architecture.md):
  - PagedAttention / custom KV-cache block tables
  - Prefix caching, speculative decoding, pipeline parallelism
  - CUDA graphs or fused attention kernels

Our model_worker re-pads and re-runs the full sequence each step — wasteful for
compute, but sufficient to demonstrate asynchronous slot reuse in tests and
benchmarks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from server.config import MAX_BATCH_SIZE, MAX_SLOTS, MAX_WAIT_MS, STEP_INTERVAL_S
from server.metrics import (
    active_slots_used,
    batch_size,
    queue_depth,
    time_to_first_token_seconds,
)
from server.model_worker import ModelWorker, SequenceState
from server.request_queue import QueuedRequest, RequestQueue

logger = logging.getLogger(__name__)


@dataclass
class _ActiveSlot:
    """One occupied decode slot in the continuous-batching pool."""

    request: QueuedRequest
    state: SequenceState
    started_at: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None


class BatchingScheduler:
    def __init__(
        self,
        queue: RequestQueue,
        worker: ModelWorker,
        *,
        generative: bool = False,
        max_batch_size: int | None = None,
        max_wait_ms: float | None = None,
        max_slots: int | None = None,
    ) -> None:
        self._queue = queue
        self._worker = worker
        self._generative = generative
        self._max_batch_size = max_batch_size or MAX_BATCH_SIZE
        self._max_wait_ms = max_wait_ms or MAX_WAIT_MS
        self._max_slots = max_slots or MAX_SLOTS
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        name = "continuous-batching-scheduler" if self._generative else "static-batching-scheduler"
        target = self._run_continuous if self._generative else self._run_static
        self._task = asyncio.create_task(target(), name=name)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    # ------------------------------------------------------------------
    # Phase 2–3: static batching (classification)
    # ------------------------------------------------------------------

    async def _run_static(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._queue.pull_batch(self._max_batch_size, self._max_wait_ms)
            if not batch:
                continue

            batch_size.observe(len(batch))
            queue_depth.set(self._queue.depth)

            texts = [item.text for item in batch]
            try:
                results = await loop.run_in_executor(None, self._worker.run_batch, texts)
                for item, result in zip(batch, results, strict=True):
                    if not item.future.done():
                        item.future.set_result(result)
            except Exception as exc:
                logger.exception("Static batch inference failed")
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)

    # ------------------------------------------------------------------
    # Phase 4: continuous batching (generative, slot pool)
    # ------------------------------------------------------------------

    async def _run_continuous(self) -> None:
        """Main scheduling loop for token-by-token generation with slot reuse."""
        loop = asyncio.get_running_loop()
        # Fixed-size slot table; None = slot is free and can accept a new request.
        slots: list[_ActiveSlot | None] = [None] * self._max_slots

        while True:
            # --- Step 1: finish sequences, resolve Futures, free slots ---------
            for index, slot in enumerate(slots):
                if slot is None or not slot.state.finished:
                    continue
                self._resolve_finished_slot(slot)
                slots[index] = None

            free_count = sum(1 for slot in slots if slot is None)
            queue_depth.set(self._queue.depth)

            # --- Step 2: fill every free slot from the priority queue ------------
            if free_count > 0:
                await self._fill_free_slots(slots)

            active = [slot for slot in slots if slot is not None and not slot.state.finished]

            if not active:
                # Entire pool idle — block until at least one request arrives.
                await self._assign_blocked_request(slots)
                continue

            active_slots_used.set(len(active))

            # --- Step 3: one batched decode step for all in-flight sequences -----
            active_states = [slot.state for slot in active]
            try:
                await loop.run_in_executor(None, self._worker.step, active_states)
            except Exception as exc:
                logger.exception("Continuous batch step failed")
                for index, slot in enumerate(slots):
                    if slot is None or slot not in active:
                        continue
                    if not slot.request.future.done():
                        slot.request.future.set_exception(exc)
                    slots[index] = None
                continue

            # Record time-to-first-token after the first generated token appears.
            now = time.perf_counter()
            for slot in active:
                if slot.first_token_at is None and slot.state.generated_count > 0:
                    slot.first_token_at = now
                    time_to_first_token_seconds.observe(now - slot.started_at)

            if STEP_INTERVAL_S > 0:
                await asyncio.sleep(STEP_INTERVAL_S)
            else:
                # Yield so HTTP handlers and queue producers can progress.
                await asyncio.sleep(0)

    async def _fill_free_slots(self, slots: list[_ActiveSlot | None]) -> None:
        """Pull as many queued requests as we have free slots (non-blocking)."""
        free_count = sum(1 for slot in slots if slot is None)
        if free_count == 0:
            return

        incoming = await self._queue.try_collect_batch(free_count)
        for request in incoming:
            slot_index = next(i for i, slot in enumerate(slots) if slot is None)
            state = self._worker.start_sequence(request.text, max_tokens=request.max_tokens)
            slots[slot_index] = _ActiveSlot(request=request, state=state)

    async def _assign_blocked_request(self, slots: list[_ActiveSlot | None]) -> None:
        """Block until work exists, then occupy exactly one newly freed slot."""
        batch = await self._queue.get_next_batch(1)
        for request in batch:
            slot_index = next(i for i, slot in enumerate(slots) if slot is None)
            state = self._worker.start_sequence(request.text, max_tokens=request.max_tokens)
            slots[slot_index] = _ActiveSlot(request=request, state=state)

    def _resolve_finished_slot(self, slot: _ActiveSlot) -> None:
        """Complete the caller's Future with the generated text."""
        if slot.request.future.done():
            return

        text = self._worker.decode_sequence(slot.state)
        result: dict[str, Any] = {
            "text": text,
            "tokens_generated": slot.state.generated_count,
            "finish_reason": slot.state.finish_reason,
        }
        slot.request.future.set_result(result)

        if slot.first_token_at is not None:
            time_to_first_token_seconds.observe(slot.first_token_at - slot.started_at)
