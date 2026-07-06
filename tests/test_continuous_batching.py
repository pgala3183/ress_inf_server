"""Tests for continuous slot-based batching (generative mode)."""

from __future__ import annotations

import asyncio
import time

import pytest
from httpx import ASGITransport, AsyncClient

from server.api import create_app


@pytest.fixture
async def generative_client(monkeypatch: pytest.MonkeyPatch) -> AsyncClient:
    monkeypatch.setenv("MAX_SLOTS", "3")
    gen_app = create_app(generative=True)
    transport = ASGITransport(app=gen_app)
    async with gen_app.router.lifespan_context(gen_app):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.mark.asyncio
async def test_short_sequences_finish_before_long_ones(generative_client: AsyncClient) -> None:
    """Prove slots are reused: short decodes complete while long sequences run.

    Scenario (MAX_SLOTS=3):
      1. Two long sequences occupy two slots.
      2. Four short sequences arrive; one immediately takes the third slot.
      3. Short sequences finish in ~4 decode steps and free slots for the rest.
      4. All short work completes before the slowest long sequence returns.
    """
    completion_times: list[tuple[str, float]] = []

    async def generate(label: str, max_tokens: int) -> None:
        started = time.perf_counter()
        response = await generative_client.post(
            "/generate",
            json={"prompt": f"Once upon a time {label}", "max_tokens": max_tokens},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["tokens_generated"] > 0
        completion_times.append((label, time.perf_counter() - started))

    # Occupy two slots with long-running sequences first.
    long_tasks = [
        asyncio.create_task(generate("long-a", 80)),
        asyncio.create_task(generate("long-b", 80)),
    ]
    await asyncio.sleep(0.15)

    # Flood short requests — they must reuse slots freed by early finishes.
    short_tasks = [asyncio.create_task(generate(f"short-{index}", 4)) for index in range(4)]
    await asyncio.gather(*long_tasks, *short_tasks)

    short_times = [elapsed for label, elapsed in completion_times if label.startswith("short")]
    long_times = [elapsed for label, elapsed in completion_times if label.startswith("long")]

    assert len(short_times) == 4
    assert len(long_times) == 2
    assert max(short_times) < max(long_times)
