"""Queue-depth-aware weighted routing between Spot and on-demand backends."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from server.config import QUEUE_PRESSURE_THRESHOLD

logger = logging.getLogger(__name__)

PoolName = Literal["spot", "ondemand"]


@dataclass
class BackendPool:
    name: PoolName
    base_url: str
    queue_depth: float = 0.0
    healthy: bool = True
    last_updated: float = 0.0


class LoadRouter:
    """Prefer Spot when queues are idle; shift weight to on-demand under pressure."""

    def __init__(
        self,
        spot_url: str | None = None,
        ondemand_url: str | None = None,
        *,
        pressure_threshold: float | None = None,
    ) -> None:
        self._pressure_threshold = pressure_threshold or QUEUE_PRESSURE_THRESHOLD
        self._pools = [
            BackendPool("spot", (spot_url or os.environ.get("SPOT_SERVICE_URL", "")).rstrip("/")),
            BackendPool(
                "ondemand",
                (ondemand_url or os.environ.get("ONDEMAND_SERVICE_URL", "")).rstrip("/"),
            ),
        ]
        self._refresh_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._refresh_task is not None and not self._refresh_task.done():
            return
        self._refresh_task = asyncio.create_task(self._refresh_loop(), name="router-metrics-refresh")

    async def stop(self) -> None:
        if self._refresh_task is None:
            return
        self._refresh_task.cancel()
        try:
            await self._refresh_task
        except asyncio.CancelledError:
            pass
        self._refresh_task = None

    async def _refresh_loop(self) -> None:
        async with httpx.AsyncClient(timeout=3.0) as client:
            while True:
                await self.refresh_metrics(client)
                await asyncio.sleep(2.0)

    async def refresh_metrics(self, client: httpx.AsyncClient | None = None) -> None:
        owns_client = client is None
        http = client or httpx.AsyncClient(timeout=3.0)
        try:
            for pool in self._pools:
                if not pool.base_url:
                    pool.healthy = False
                    pool.queue_depth = 0.0
                    continue
                try:
                    metrics_response = await http.get(f"{pool.base_url}/metrics")
                    metrics_response.raise_for_status()
                    pool.queue_depth = _parse_queue_depth(metrics_response.text)
                    health_response = await http.get(f"{pool.base_url}/healthz")
                    pool.healthy = health_response.status_code == 200
                    pool.last_updated = time.monotonic()
                except httpx.HTTPError:
                    pool.healthy = False
                    pool.queue_depth = float("inf")
        finally:
            if owns_client:
                await http.aclose()

    def _weights(self) -> dict[PoolName, int]:
        """Return integer weights for weighted random backend selection."""
        spot = next(pool for pool in self._pools if pool.name == "spot")
        ondemand = next(pool for pool in self._pools if pool.name == "ondemand")

        max_depth = max(spot.queue_depth, ondemand.queue_depth)
        if max_depth >= self._pressure_threshold:
            # Pressure — favor guaranteed on-demand capacity.
            return {"spot": 1, "ondemand": 9}

        # Idle / low pressure — favor cheaper Spot pool.
        return {"spot": 9, "ondemand": 1}

    def choose_pool(self) -> BackendPool:
        weights = self._weights()
        healthy = [pool for pool in self._pools if pool.healthy and pool.base_url]
        if not healthy:
            raise RuntimeError("No healthy inference backend pools configured")

        population = [pool.name for pool in healthy for _ in range(max(1, weights[pool.name]))]
        chosen_name = random.choice(population)
        return next(pool for pool in healthy if pool.name == chosen_name)

    async def forward_predict(
        self,
        payload: dict[str, Any],
        *,
        client: httpx.AsyncClient | None = None,
    ) -> tuple[dict[str, Any], PoolName]:
        pool = self.choose_pool()
        owns_client = client is None
        http = client or httpx.AsyncClient(timeout=60.0)
        try:
            response = await http.post(f"{pool.base_url}/predict", json=payload)
            response.raise_for_status()
            return response.json(), pool.name
        finally:
            if owns_client:
                await http.aclose()

    def routing_snapshot(self) -> dict[str, Any]:
        weights = self._weights()
        return {
            "pressure_threshold": self._pressure_threshold,
            "weights": weights,
            "pools": [
                {
                    "name": pool.name,
                    "base_url": pool.base_url,
                    "queue_depth": pool.queue_depth,
                    "healthy": pool.healthy,
                }
                for pool in self._pools
            ],
        }


def _parse_queue_depth(metrics_text: str) -> float:
    match = re.search(r"^queue_depth\s+(\S+)", metrics_text, re.MULTILINE)
    if not match:
        return 0.0
    return float(match.group(1))
