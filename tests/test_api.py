"""API tests for the inference server."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_healthz(client: AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mode"] == "classification"


@pytest.mark.asyncio
async def test_predict_positive_sentiment(client: AsyncClient) -> None:
    response = await client.post("/predict", json={"text": "I love this product"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "POSITIVE"
    assert body["score"] > 0.5


@pytest.mark.asyncio
async def test_predict_rejects_empty_text(client: AsyncClient) -> None:
    response = await client.post("/predict", json={"text": ""})
    assert response.status_code == 422
