"""Unit tests for generative step-based model_worker."""

from __future__ import annotations

import pytest

from server.model_worker import ModelWorker


@pytest.fixture(scope="module")
def generative_worker() -> ModelWorker:
    worker = ModelWorker(generative=True)
    import asyncio

    asyncio.run(worker.load())
    return worker


def test_step_respects_max_tokens(generative_worker: ModelWorker) -> None:
    state = generative_worker.start_sequence("Hello", max_tokens=3)
    for _ in range(5):
        generative_worker.step([state])
        if state.finished:
            break

    assert state.finished
    assert state.finish_reason == "max_tokens"
    assert state.generated_count == 3


def test_short_sequence_finishes_before_long_in_same_batch(generative_worker: ModelWorker) -> None:
    short = generative_worker.start_sequence("Hi", max_tokens=2)
    long = generative_worker.start_sequence("Once upon a time", max_tokens=30)

    short_done_at: int | None = None
    for step_index in range(35):
        generative_worker.step([short, long])
        if short.finished and short_done_at is None:
            short_done_at = step_index
        if long.finished:
            break

    assert short.finished
    assert short_done_at is not None
    assert long.finished
    assert short_done_at < 34
