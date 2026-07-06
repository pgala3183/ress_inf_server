"""Async load generators and Locust subprocess wrapper."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import httpx

from benchmarks.common import MetricsSample, RequestSample, fetch_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCUSTFILE = REPO_ROOT / "load_test" / "locustfile.py"


async def _one_predict(
    client: httpx.AsyncClient,
    base_url: str,
    index: int,
    priority: str,
) -> RequestSample:
    started = time.time()
    t0 = time.perf_counter()
    status = 0
    ok = False
    try:
        response = await client.post(
            f"{base_url.rstrip('/')}/predict",
            json={"text": f"benchmark sample {index}", "priority": priority},
        )
        status = response.status_code
        ok = response.status_code == 200
    except httpx.HTTPError:
        status = 0
        ok = False
    latency = time.perf_counter() - t0
    return RequestSample(timestamp=started, priority=priority, latency_s=latency, status_code=status, ok=ok)


async def poll_metrics_loop(
    base_url: str,
    interval_s: float,
    stop: asyncio.Event,
    sink: list[MetricsSample],
) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        while not stop.is_set():
            try:
                sink.append(await fetch_metrics(client, base_url))
            except httpx.HTTPError:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_s)
            except asyncio.TimeoutError:
                continue


async def run_steady_load(
    base_url: str,
    *,
    total_requests: int = 60,
    concurrency: int = 20,
    interactive_ratio: float = 0.7,
    metrics_interval_s: float = 0.25,
) -> tuple[list[RequestSample], list[MetricsSample], float]:
    sem = asyncio.Semaphore(concurrency)
    samples: list[RequestSample] = []
    metrics_series: list[MetricsSample] = []
    stop = asyncio.Event()

    async def worker(idx: int) -> None:
        priority = "interactive" if (idx % 100) / 100.0 < interactive_ratio else "batch"
        async with sem:
            async with httpx.AsyncClient(timeout=120.0) as client:
                samples.append(await _one_predict(client, base_url, idx, priority))

    poller = asyncio.create_task(poll_metrics_loop(base_url, metrics_interval_s, stop, metrics_series))
    started = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in range(total_requests)))
    duration = time.perf_counter() - started
    stop.set()
    await poller
    return samples, metrics_series, duration


async def run_bursty_load(
    base_url: str,
    *,
    phases: list[tuple[float, int, float]] | None = None,
    interactive_ratio: float = 0.7,
    metrics_interval_s: float = 0.25,
) -> tuple[list[RequestSample], list[MetricsSample], float]:
    """Phases: (duration_s, concurrency, target_rps)."""
    if phases is None:
        phases = [
            (5.0, 5, 3.0),
            (10.0, 40, 25.0),
            (8.0, 60, 40.0),
            (7.0, 10, 8.0),
        ]

    samples: list[RequestSample] = []
    metrics_series: list[MetricsSample] = []
    stop = asyncio.Event()
    poller = asyncio.create_task(poll_metrics_loop(base_url, metrics_interval_s, stop, metrics_series))
    started = time.perf_counter()
    request_idx = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        for duration_s, concurrency, target_rps in phases:
            phase_end = time.perf_counter() + duration_s
            interval = 1.0 / max(target_rps, 0.1)
            in_flight: set[asyncio.Task[RequestSample]] = set()

            while time.perf_counter() < phase_end:
                while len(in_flight) >= concurrency:
                    done, pending = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        samples.append(task.result())
                    in_flight = pending

                priority = "interactive" if (request_idx % 100) / 100.0 < interactive_ratio else "batch"
                in_flight.add(
                    asyncio.create_task(_one_predict(client, base_url, request_idx, priority))
                )
                request_idx += 1
                await asyncio.sleep(interval)

            if in_flight:
                done, _ = await asyncio.wait(in_flight)
                for task in done:
                    samples.append(task.result())

    duration = time.perf_counter() - started
    stop.set()
    await poller
    return samples, metrics_series, duration


async def run_preemption_load(
    base_url: str,
    *,
    duration_s: float = 25.0,
    concurrency: int = 30,
    interactive_ratio: float = 0.7,
    metrics_interval_s: float = 0.5,
) -> tuple[list[RequestSample], list[MetricsSample], float]:
    """Sustained load through a simulated preemption/drain window."""
    sem = asyncio.Semaphore(concurrency)
    samples: list[RequestSample] = []
    metrics_series: list[MetricsSample] = []
    stop = asyncio.Event()
    request_idx = 0
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal request_idx
        async with sem:
            async with httpx.AsyncClient(timeout=120.0) as client:
                while not stop.is_set():
                    async with lock:
                        idx = request_idx
                        request_idx += 1
                    priority = "interactive" if (idx % 100) / 100.0 < interactive_ratio else "batch"
                    samples.append(await _one_predict(client, base_url, idx, priority))
                    await asyncio.sleep(0.05)

    poller = asyncio.create_task(poll_metrics_loop(base_url, metrics_interval_s, stop, metrics_series))
    workers = [asyncio.create_task(worker()) for _ in range(min(concurrency, 12))]
    started = time.perf_counter()
    await asyncio.sleep(duration_s)
    stop.set()
    for task in workers:
        task.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    await poller
    return samples, metrics_series, time.perf_counter() - started


def run_locust_headless(
    base_url: str,
    *,
    shape: str = "bursty",
    csv_prefix: Path,
    run_time: str = "45s",
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LOCUST_SHAPE"] = shape
    env["LOCUST_USE_SHAPE"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(LOCUSTFILE),
        "--headless",
        "--host",
        base_url,
        "--run-time",
        run_time,
        "--csv",
        str(csv_prefix),
    ]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(REPO_ROOT))


async def wait_for_server(base_url: str, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                response = await client.get(f"{base_url.rstrip('/')}/healthz")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
    raise TimeoutError(f"Server at {base_url} did not become ready within {timeout_s}s")
