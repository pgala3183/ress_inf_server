"""Synthetic load spike to exercise queue_depth metrics and HPA dry-run."""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx


async def _predict(client: httpx.AsyncClient, url: str, idx: int) -> tuple[int, float, int]:
    start = time.perf_counter()
    response = await client.post(f"{url}/predict", json={"text": f"load spike sample {idx}", "priority": "batch"})
    elapsed = time.perf_counter() - start
    return response.status_code, elapsed, idx


async def run_load(url: str, concurrency: int, requests: int) -> None:
    url = url.rstrip("/")
    sem = asyncio.Semaphore(concurrency)
    results: list[tuple[int, float]] = []

    async def worker(idx: int) -> None:
        async with sem:
            async with httpx.AsyncClient(timeout=120.0) as client:
                status, elapsed, _ = await _predict(client, url, idx)
                results.append((status, elapsed))

    started = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in range(requests)))
    duration = time.perf_counter() - started

    ok = sum(1 for status, _ in results if status == 200)
    print(f"Completed {len(results)} requests in {duration:.1f}s ({ok} OK, concurrency={concurrency})")

    async with httpx.AsyncClient(timeout=10.0) as client:
        metrics = await client.get(f"{url}/metrics")
        for line in metrics.text.splitlines():
            if line.startswith("queue_depth ") or line.startswith("gpu_utilization_percent "):
                print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--requests", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(run_load(args.url, args.concurrency, args.requests))


if __name__ == "__main__":
    main()
