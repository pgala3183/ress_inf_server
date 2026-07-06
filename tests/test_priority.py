"""Tests for SLA-aware priority scheduling under load."""

from __future__ import annotations

import asyncio
import time

import pytest
from httpx import AsyncClient

from server.request_queue import BATCH_SLA_MS


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * pct) - 1))
    return ordered[index]


@pytest.mark.asyncio
async def test_mixed_priority_latencies(client: AsyncClient) -> None:
    async def predict(priority: str, index: int) -> tuple[str, float]:
        started = time.perf_counter()
        response = await client.post(
            "/predict",
            json={"text": f"Priority test {priority} sample {index}", "priority": priority},
        )
        assert response.status_code == 200
        return priority, time.perf_counter() - started

    tasks = [predict("interactive", index) for index in range(40)]
    tasks.extend(predict("batch", index) for index in range(12))
    results = await asyncio.gather(*tasks)

    interactive_latencies = [latency for priority, latency in results if priority == "interactive"]
    batch_latencies = [latency for priority, latency in results if priority == "batch"]

    interactive_p99 = _percentile(interactive_latencies, 0.99)
    batch_p99 = _percentile(batch_latencies, 0.99)

    assert interactive_p99 < batch_p99
    assert interactive_p99 < batch_p99 * 0.85
    assert max(batch_latencies) < (BATCH_SLA_MS / 1000.0) * 2.0
