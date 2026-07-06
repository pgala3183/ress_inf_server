"""Quick async load test for the static batching baseline."""

from __future__ import annotations

import asyncio
import statistics
import sys
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"
REQUEST_COUNT = 50


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        health = await client.get("/healthz")
        health.raise_for_status()

        async def one_request(index: int) -> float:
            started = time.perf_counter()
            response = await client.post(
                "/predict",
                json={"text": f"I love this product number {index}"},
            )
            response.raise_for_status()
            return time.perf_counter() - started

        latencies = await asyncio.gather(*(one_request(i) for i in range(REQUEST_COUNT)))

    latencies.sort()
    p50 = statistics.median(latencies)
    p99_index = max(0, int(len(latencies) * 0.99) - 1)
    p99 = latencies[p99_index]

    print(f"Requests: {REQUEST_COUNT}")
    print(f"p50 latency: {p50 * 1000:.1f} ms")
    print(f"p99 latency: {p99 * 1000:.1f} ms")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except httpx.ConnectError:
        print(f"Could not connect to {BASE_URL}. Start the server first.", file=sys.stderr)
        sys.exit(1)
