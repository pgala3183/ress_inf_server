"""Tests for Spot preemption graceful drain and peer migration."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from server.api import create_app
from server.drain_state import DrainManager
from server.model_worker import ModelWorker
from server.peer_forwarder import PeerForwarder
from server.preemption_listener import PreemptionListener
from server.request_queue import RequestQueue


def _metric_value(metrics_text: str, metric: str) -> float:
    match = re.search(rf"^{metric}\s+(\S+)", metrics_text, re.MULTILINE)
    assert match, f"Metric {metric} not found"
    return float(match.group(1))


@pytest.fixture
async def drain_setup(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DRAIN_EXIT_ON_COMPLETE", "0")
    monkeypatch.setenv("MAX_BATCH_SIZE", "1")
    monkeypatch.setenv("MAX_WAIT_MS", "0")
    monkeypatch.setenv("PEER_URLS", "http://peer.example")

    original_run_batch = ModelWorker.run_batch

    def slow_run_batch(self: ModelWorker, inputs: list[str]) -> list[dict[str, Any]]:
        time.sleep(0.1)
        return original_run_batch(self, inputs)

    monkeypatch.setattr(ModelWorker, "run_batch", slow_run_batch)

    async def fake_forward(
        self: PeerForwarder,
        text: str,
        priority: str = "interactive",
        *,
        client: Any = None,
    ) -> dict[str, Any]:
        return {"label": "POSITIVE", "score": 0.95}

    monkeypatch.setattr(PeerForwarder, "forward_predict", fake_forward)

    app = create_app(generative=False)
    transport = ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield app, client


@pytest.mark.asyncio
async def test_migrate_pending_queue_forwards_to_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Queued-but-not-admitted requests are forwarded to a peer without being dropped."""
    monkeypatch.setenv("PEER_URLS", "http://peer.example")

    async def fake_forward(
        self: PeerForwarder,
        text: str,
        priority: str = "interactive",
        *,
        client: Any = None,
    ) -> dict[str, Any]:
        return {"label": "POSITIVE", "score": 0.99}

    monkeypatch.setattr(PeerForwarder, "forward_predict", fake_forward)

    queue = RequestQueue()
    scheduler = AsyncMock()
    scheduler.enter_drain_mode = AsyncMock()
    scheduler.wait_idle = AsyncMock()
    listener = PreemptionListener(DrainManager(), queue, scheduler, PeerForwarder())

    loop = asyncio.get_running_loop()
    futures = [loop.create_future() for _ in range(3)]
    for index, future in enumerate(futures):
        await queue.enqueue(f"migrate me {index}", future, "interactive")

    migrated = await listener._migrate_pending_queue()
    assert migrated == 3

    results = await asyncio.gather(*futures)
    for result in results:
        assert result["label"] == "POSITIVE"
        assert result["score"] == 0.99


@pytest.mark.asyncio
async def test_preemption_drain_migrates_queued_requests(drain_setup) -> None:
    """End-to-end: one in-flight request completes locally; queued peers are migrated."""
    app, client = drain_setup
    listener = app.state.preemption_listener
    queue = app.state.request_queue
    loop = asyncio.get_running_loop()

    migrated_before = _metric_value(
        (await client.get("/metrics")).text,
        "requests_migrated_total",
    )

    in_flight = loop.create_future()
    await queue.enqueue("in-flight request", in_flight, "interactive")

    pending_futures = [loop.create_future() for _ in range(3)]
    for index, future in enumerate(pending_futures):
        await queue.enqueue(f"queued request {index}", future, "interactive")

    await asyncio.sleep(0.03)
    await listener.trigger_drain("test_preemption")
    assert await listener.wait_for_drain_complete(timeout=10.0)

    in_flight_result = await in_flight
    pending_results = await asyncio.gather(*pending_futures)

    assert in_flight_result["label"] in {"POSITIVE", "NEGATIVE"}
    for result in pending_results:
        assert result == {"label": "POSITIVE", "score": 0.95}

    metrics_text = (await client.get("/metrics")).text
    assert _metric_value(metrics_text, "requests_dropped_total") == 0.0
    assert _metric_value(metrics_text, "requests_migrated_total") >= migrated_before + 3


@pytest.mark.asyncio
async def test_new_requests_rejected_with_retry_after_while_draining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DRAIN_EXIT_ON_COMPLETE", "0")
    monkeypatch.setenv("PEER_URLS", "http://peer.example")

    async def fake_forward(
        self: PeerForwarder,
        text: str,
        priority: str = "interactive",
        *,
        client: Any = None,
    ) -> dict[str, Any]:
        return {"label": "POSITIVE", "score": 0.9}

    monkeypatch.setattr(PeerForwarder, "forward_predict", fake_forward)

    app = create_app(generative=False)
    transport = ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            listener = app.state.preemption_listener
            await listener.trigger_drain("test_reject")
            await asyncio.sleep(0.02)

            rejected = await client.post("/predict", json={"text": "should be rejected"})
            assert rejected.status_code == 503
            assert rejected.headers.get("retry-after") == "30"

            await listener.wait_for_drain_complete(timeout=10.0)
