"""Tests for dynamic request batching."""

from __future__ import annotations

import asyncio
import re

import pytest
from httpx import AsyncClient


def _metric_value(metrics_text: str, metric: str) -> float:
    match = re.search(rf"^{metric}\s+(\S+)", metrics_text, re.MULTILINE)
    assert match, f"Metric {metric} not found"
    return float(match.group(1))


@pytest.mark.asyncio
async def test_concurrent_predict_batches_requests(client: AsyncClient) -> None:
    async def predict(index: int):
        return await client.post(
            "/predict",
            json={"text": f"I love this product number {index}"},
        )

    responses = await asyncio.gather(*(predict(i) for i in range(20)))

    for response in responses:
        assert response.status_code == 200
        body = response.json()
        assert body["label"] in {"POSITIVE", "NEGATIVE"}
        assert 0.0 <= body["score"] <= 1.0

    metrics_response = await client.get("/metrics")
    metrics_text = metrics_response.text
    batch_count = _metric_value(metrics_text, "batch_size_count")
    batches_of_one = _metric_value(metrics_text, 'batch_size_bucket{le="1.0"}')

    assert batch_count > 0
    assert batches_of_one < batch_count, "Expected at least one batch with size > 1"
