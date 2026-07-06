"""Mixed interactive/batch load test for Phase 3 benchmark baseline."""

from __future__ import annotations

import asyncio
import statistics
import sys
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"
REQUEST_COUNT = 50
INTERACTIVE_RATIO = 0.7


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * pct) - 1))
    return ordered[index]


async def main() -> None:
    interactive_count = int(REQUEST_COUNT * INTERACTIVE_RATIO)
    batch_count = REQUEST_COUNT - interactive_count

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as client:
        await client.get("/healthz")

        async def one_request(index: int, priority: str) -> tuple[str, float]:
            started = time.perf_counter()
            response = await client.post(
                "/predict",
                json={"text": f"Benchmark sample {index}", "priority": priority},
            )
            response.raise_for_status()
            return priority, time.perf_counter() - started

        tasks: list[asyncio.Task[tuple[str, float]]] = []
        for index in range(interactive_count):
            tasks.append(asyncio.create_task(one_request(index, "interactive")))
        for index in range(batch_count):
            tasks.append(asyncio.create_task(one_request(index, "batch")))

        results = await asyncio.gather(*tasks)

    interactive = [latency for priority, latency in results if priority == "interactive"]
    batch = [latency for priority, latency in results if priority == "batch"]

    print(f"Requests: {REQUEST_COUNT} ({interactive_count} interactive, {batch_count} batch)")
    print(f"Interactive p50: {_percentile(interactive, 0.50) * 1000:.1f} ms")
    print(f"Interactive p99: {_percentile(interactive, 0.99) * 1000:.1f} ms")
    print(f"Batch p50: {_percentile(batch, 0.50) * 1000:.1f} ms")
    print(f"Batch p99: {_percentile(batch, 0.99) * 1000:.1f} ms")
    print(f"Interactive mean: {statistics.mean(interactive) * 1000:.1f} ms")
    print(f"Batch mean: {statistics.mean(batch) * 1000:.1f} ms")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except httpx.ConnectError:
        print(f"Could not connect to {BASE_URL}. Start the server first.", file=sys.stderr)
        sys.exit(1)
