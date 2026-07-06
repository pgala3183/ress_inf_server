"""Background scheduler that dynamically batches queued inference requests."""

from __future__ import annotations

import asyncio
import logging
import os

from server.metrics import batch_size, queue_depth
from server.model_worker import ModelWorker
from server.request_queue import RequestQueue

logger = logging.getLogger(__name__)


class BatchingScheduler:
    def __init__(
        self,
        queue: RequestQueue,
        worker: ModelWorker,
        max_batch_size: int | None = None,
        max_wait_ms: float | None = None,
    ) -> None:
        self._queue = queue
        self._worker = worker
        self._max_batch_size = max_batch_size or int(os.environ.get("MAX_BATCH_SIZE", "8"))
        self._max_wait_ms = max_wait_ms or float(os.environ.get("MAX_WAIT_MS", "10"))
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="batching-scheduler")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
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
                logger.exception("Batch inference failed")
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
