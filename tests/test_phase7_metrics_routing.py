"""Tests for GPU metrics and queue-depth-aware routing."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from server.gpu_metrics import read_gpu_utilization_percent
from server.load_router import LoadRouter, _parse_queue_depth
from server.router_app import create_router_app


def test_read_gpu_utilization_percent_cpu_stub() -> None:
    with patch("server.gpu_metrics.shutil.which", return_value=None):
        assert read_gpu_utilization_percent() == 0.0


def test_read_gpu_utilization_percent_with_gpu() -> None:
    with patch("server.gpu_metrics.shutil.which", return_value="/usr/bin/nvidia-smi"):
        with patch(
            "server.gpu_metrics.subprocess.run",
            return_value=type("R", (), {"stdout": "42\n50\n"})(),
        ):
            assert read_gpu_utilization_percent() == 46.0


def test_parse_queue_depth_from_prometheus_text() -> None:
    body = "# HELP queue_depth depth\nqueue_depth 7\n"
    assert _parse_queue_depth(body) == 7.0


def test_router_prefers_spot_when_idle() -> None:
    router = LoadRouter(spot_url="http://spot", ondemand_url="http://od", pressure_threshold=5.0)
    spot = next(p for p in router._pools if p.name == "spot")
    ondemand = next(p for p in router._pools if p.name == "ondemand")
    spot.queue_depth = 1.0
    ondemand.queue_depth = 2.0
    spot.healthy = ondemand.healthy = True
    assert router._weights() == {"spot": 9, "ondemand": 1}


def test_router_prefers_ondemand_under_pressure() -> None:
    router = LoadRouter(spot_url="http://spot", ondemand_url="http://od", pressure_threshold=5.0)
    spot = next(p for p in router._pools if p.name == "spot")
    ondemand = next(p for p in router._pools if p.name == "ondemand")
    spot.queue_depth = 6.0
    ondemand.queue_depth = 1.0
    spot.healthy = ondemand.healthy = True
    assert router._weights() == {"spot": 1, "ondemand": 9}


@pytest.mark.asyncio
async def test_metrics_exposes_gpu_utilization() -> None:
    from server.api import create_app

    app = create_app(generative=False)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics")
            assert response.status_code == 200
            assert "gpu_utilization_percent" in response.text
            assert "queue_depth" in response.text


@pytest.mark.asyncio
async def test_router_routing_snapshot() -> None:
    app = create_router_app(spot_url="http://spot", ondemand_url="http://od")
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/routing")
            assert response.status_code == 200
            body = response.json()
            assert body["pressure_threshold"] == 5.0
            assert set(body["weights"].keys()) == {"spot", "ondemand"}
