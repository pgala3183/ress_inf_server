"""Spot preemption detection and graceful drain orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import TYPE_CHECKING, Callable

import httpx

from server.drain_state import DrainManager, DrainState
from server.metrics import drain_duration_seconds, requests_migrated_total
from server.peer_forwarder import PeerForwarder

if TYPE_CHECKING:
    from server.batching_scheduler import BatchingScheduler
    from server.request_queue import RequestQueue

logger = logging.getLogger("resilient.preemption")

GCE_PREEMPTED_URL = "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
METADATA_POLL_INTERVAL_S = float(os.environ.get("PREEMPTION_POLL_INTERVAL_S", "1"))
SIMULATE_PREEMPTION_AFTER_SECONDS = os.environ.get("SIMULATE_PREEMPTION_AFTER_SECONDS")
RETRY_AFTER_SECONDS = os.environ.get("DRAIN_RETRY_AFTER_SECONDS", "30")


class PreemptionListener:
    """Polls GCE metadata (or a local timer) and orchestrates graceful drain."""

    def __init__(
        self,
        drain_manager: DrainManager,
        queue: RequestQueue,
        scheduler: BatchingScheduler,
        peer_forwarder: PeerForwarder,
        *,
        on_shutdown: Callable[[], None] | None = None,
    ) -> None:
        self._drain_manager = drain_manager
        self._queue = queue
        self._scheduler = scheduler
        self._peer_forwarder = peer_forwarder
        self._on_shutdown = on_shutdown or (lambda: None)
        self._poll_task: asyncio.Task[None] | None = None
        self._simulate_task: asyncio.Task[None] | None = None
        self._drain_orchestration_task: asyncio.Task[None] | None = None
        self._sigterm_registered = False

    async def start(self) -> None:
        self._poll_task = asyncio.create_task(self._poll_gce_metadata(), name="gce-preemption-poll")
        if SIMULATE_PREEMPTION_AFTER_SECONDS:
            delay = float(SIMULATE_PREEMPTION_AFTER_SECONDS)
            self._simulate_task = asyncio.create_task(
                self._simulate_preemption(delay),
                name="simulate-preemption",
            )
            logger.info(
                json.dumps(
                    {
                        "event": "preemption_simulation_scheduled",
                        "delay_seconds": delay,
                    }
                )
            )
        self._register_sigterm()

    async def stop(self) -> None:
        for task in (self._poll_task, self._simulate_task, self._drain_orchestration_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def _register_sigterm(self) -> None:
        if self._sigterm_registered:
            return
        loop = asyncio.get_running_loop()

        def _handle_sigterm() -> None:
            logger.info(json.dumps({"event": "sigterm_received"}))
            asyncio.create_task(self.trigger_drain("sigterm"))

        try:
            loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)
            self._sigterm_registered = True
        except NotImplementedError:
            # Windows does not support add_signal_handler for SIGTERM in all contexts.
            signal.signal(signal.SIGTERM, lambda _signum, _frame: _handle_sigterm())

    async def trigger_drain(self, reason: str) -> None:
        """Public entry point used by metadata poll, simulation, SIGTERM, and /internal/drain."""
        if not await self._drain_manager.begin_drain(reason):
            return
        if self._drain_orchestration_task is None or self._drain_orchestration_task.done():
            self._drain_orchestration_task = asyncio.create_task(
                self._orchestrate_drain(reason),
                name="drain-orchestration",
            )

    async def wait_for_drain_complete(self, timeout: float | None = None) -> bool:
        return await self._drain_manager.wait_drained(timeout=timeout)

    async def _poll_gce_metadata(self) -> None:
        while True:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(
                        GCE_PREEMPTED_URL,
                        headers={"Metadata-Flavor": "Google"},
                    )
                if response.status_code == 200 and response.text.strip().upper() == "TRUE":
                    logger.warning(json.dumps({"event": "gce_preemption_detected"}))
                    await self.trigger_drain("gce_metadata_preempted")
                    return
            except (httpx.HTTPError, OSError):
                # Expected outside GCE — metadata server is unreachable locally.
                pass
            await asyncio.sleep(METADATA_POLL_INTERVAL_S)

    async def _simulate_preemption(self, delay: float) -> None:
        await asyncio.sleep(delay)
        logger.warning(
            json.dumps(
                {
                    "event": "simulated_preemption_triggered",
                    "delay_seconds": delay,
                }
            )
        )
        await self.trigger_drain("simulated_preemption")

    async def _orchestrate_drain(self, reason: str) -> None:
        started = time.perf_counter()
        logger.info(json.dumps({"event": "drain_orchestration_started", "reason": reason}))

        # Step 1: stop admitting new scheduler work; in-flight batches/slots continue.
        await self._scheduler.enter_drain_mode()

        # Step 2: migrate queue items that have not yet been picked up by the scheduler.
        migrated = await self._migrate_pending_queue()
        requests_migrated_total.inc(migrated)
        logger.info(json.dumps({"event": "pending_queue_migrated", "count": migrated}))

        # Step 3: wait for admitted / in-flight work to finish normally.
        await self._scheduler.wait_idle()
        logger.info(json.dumps({"event": "scheduler_idle"}))

        await self._drain_manager.mark_drained(reason)
        elapsed = time.perf_counter() - started
        drain_duration_seconds.observe(elapsed)
        logger.info(json.dumps({"event": "drain_orchestration_complete", "elapsed_seconds": elapsed}))

        self._on_shutdown()

    async def _migrate_pending_queue(self) -> int:
        if not self._peer_forwarder.has_peers:
            pending = await self._queue.drain_all_pending()
            if pending:
                logger.error(
                    json.dumps(
                        {
                            "event": "migration_impossible_no_peers",
                            "pending_count": len(pending),
                        }
                    )
                )
            return 0

        pending = await self._queue.drain_all_pending()
        if not pending:
            return 0

        migrated = 0
        async with httpx.AsyncClient(timeout=60.0) as client:
            for item in pending:
                try:
                    result = await self._peer_forwarder.forward_predict(
                        item.text,
                        item.priority,
                        client=client,
                    )
                    if not item.future.done():
                        item.future.set_result(result)
                    migrated += 1
                except Exception as exc:
                    logger.exception(
                        json.dumps(
                            {
                                "event": "peer_forward_failed",
                                "error": str(exc),
                            }
                        )
                    )
                    if not item.future.done():
                        item.future.set_exception(exc)
        return migrated


def configure_drain_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    for name in ("resilient.drain", "resilient.preemption"):
        log = logging.getLogger(name)
        log.handlers = [handler]
        log.setLevel(logging.INFO)
        log.propagate = False
