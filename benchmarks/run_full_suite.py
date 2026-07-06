"""Orchestrate batch-size sweeps, traffic patterns, plots, and CSV export."""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from benchmarks.common import (
    RunSummary,
    ensure_output_dir,
    implied_replicas,
    summarize_run,
    write_csv,
    write_summary_csv,
)
from benchmarks.load_runner import (
    run_bursty_load,
    run_locust_headless,
    run_preemption_load,
    run_steady_load,
    wait_for_server,
)
from benchmarks.plot_results import generate_all_plots
from benchmarks.pricing import cost_per_1000_inferences

REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZES = [1, 4, 8, 16, 32]
DEFAULT_PORT = int(os.environ.get("BENCHMARK_PORT", "8020"))
PEER_PORT = DEFAULT_PORT + 1


class ServerProcess:
    def __init__(self, port: int, env: dict[str, str] | None = None) -> None:
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self._env = os.environ.copy()
        if env:
            self._env.update(env)
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "server.main",
            "--port",
            str(self.port),
        ]
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=self._env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


async def run_batch_size_sweep(
    summaries: list[RunSummary],
    output: Path,
    batch_sizes: list[int],
) -> list[dict]:
    utilization_rows: list[dict] = []
    detail_rows: list[dict] = []

    for batch_size in batch_sizes:
        server = ServerProcess(
            DEFAULT_PORT,
            {
                "MAX_BATCH_SIZE": str(batch_size),
                "MAX_WAIT_MS": "10",
                "PRIORITY_SCHEDULING": "1",
            },
        )
        print(f"[batch_size={batch_size}] starting server...")
        server.start()
        try:
            await wait_for_server(server.base_url)

            for pattern in ("steady", "bursty"):
                print(f"  running {pattern} load...")
                if pattern == "steady":
                    samples, metrics_series, duration = await run_steady_load(
                        server.base_url,
                        total_requests=40,
                    )
                else:
                    samples, metrics_series, duration = await run_bursty_load(server.base_url)
                cost = cost_per_1000_inferences(
                    len(samples) / duration if duration else 0,
                    use_gpu=False,
                )
                summary = summarize_run(
                    scenario=f"batch_sweep_{pattern}",
                    batch_size=batch_size,
                    traffic_pattern=pattern,
                    priority_scheduling=True,
                    samples=samples,
                    metrics_series=metrics_series,
                    duration_s=duration,
                    cost_per_1000_usd=cost,
                )
                summaries.append(summary)

                for sample in samples:
                    detail_rows.append(
                        {
                            "batch_size": batch_size,
                            "traffic_pattern": pattern,
                            "timestamp": sample.timestamp,
                            "priority": sample.priority,
                            "latency_ms": sample.latency_s * 1000,
                            "status_code": sample.status_code,
                            "ok": sample.ok,
                        }
                    )
                for metric in metrics_series:
                    row = {
                        "batch_size": batch_size,
                        "traffic_pattern": pattern,
                        "timestamp": metric.timestamp,
                        "queue_depth": metric.queue_depth,
                        "gpu_utilization_percent": metric.gpu_utilization_percent,
                        "cpu_percent": metric.cpu_percent,
                        "requests_dropped_total": metric.requests_dropped_total,
                    }
                    utilization_rows.append(row)

            locust_csv = output / f"locust_bursty_batch{batch_size}"
            print(f"  running locust bursty ({locust_csv.name})...")
            run_locust_headless(server.base_url, shape="bursty", csv_prefix=locust_csv, run_time="30s")
        finally:
            server.stop()
            await asyncio.sleep(2)

    write_csv(
        output / "request_samples.csv",
        detail_rows,
        ["batch_size", "traffic_pattern", "timestamp", "priority", "latency_ms", "status_code", "ok"],
    )
    write_csv(
        output / "utilization_timeseries.csv",
        utilization_rows,
        [
            "batch_size",
            "traffic_pattern",
            "timestamp",
            "queue_depth",
            "gpu_utilization_percent",
            "cpu_percent",
            "requests_dropped_total",
        ],
    )
    return utilization_rows


async def run_sla_comparison(summaries: list[RunSummary]) -> None:
    for enabled in (True, False):
        server = ServerProcess(
            DEFAULT_PORT,
            {
                "MAX_BATCH_SIZE": "8",
                "MAX_WAIT_MS": "10",
                "PRIORITY_SCHEDULING": "1" if enabled else "0",
            },
        )
        label = "on" if enabled else "off"
        print(f"[sla priority={label}] starting server...")
        server.start()
        try:
            await wait_for_server(server.base_url)
            samples, metrics_series, duration = await run_bursty_load(server.base_url)
            cost = cost_per_1000_inferences(len(samples) / duration if duration else 0)
            summaries.append(
                summarize_run(
                    scenario=f"sla_priority_{label}",
                    batch_size=8,
                    traffic_pattern="bursty",
                    priority_scheduling=enabled,
                    samples=samples,
                    metrics_series=metrics_series,
                    duration_s=duration,
                    cost_per_1000_usd=cost,
                )
            )
        finally:
            server.stop()
            await asyncio.sleep(2)


async def run_autoscaling_scenario(output: Path) -> list[dict]:
    server = ServerProcess(
        DEFAULT_PORT,
        {"MAX_BATCH_SIZE": "8", "MAX_WAIT_MS": "10", "PRIORITY_SCHEDULING": "1"},
    )
    print("[autoscaling] starting server...")
    server.start()
    rows: list[dict] = []
    try:
        await wait_for_server(server.base_url)
        samples, metrics_series, duration = await run_bursty_load(
            server.base_url,
            phases=[(3.0, 20, 10.0), (15.0, 100, 80.0), (12.0, 20, 15.0)],
        )
        t0 = metrics_series[0].timestamp if metrics_series else time.time()
        for metric in metrics_series:
            elapsed = metric.timestamp - t0
            rows.append(
                {
                    "elapsed_s": elapsed,
                    "queue_depth": metric.queue_depth,
                    "implied_replicas": implied_replicas(metric.queue_depth),
                    "gpu_utilization_percent": metric.gpu_utilization_percent,
                    "cpu_percent": metric.cpu_percent,
                }
            )
        write_csv(
            output / "autoscaling_timeseries.csv",
            rows,
            ["elapsed_s", "queue_depth", "implied_replicas", "gpu_utilization_percent", "cpu_percent"],
        )
        print(f"[autoscaling] peak queue_depth={max(r['queue_depth'] for r in rows):.0f}")
    finally:
        server.stop()
        await asyncio.sleep(2)
    return rows


async def run_preemption_scenario(output: Path) -> list[dict]:
    peer = ServerProcess(PEER_PORT, {"MAX_BATCH_SIZE": "8", "DRAIN_EXIT_ON_COMPLETE": "0"})
    primary = ServerProcess(
        DEFAULT_PORT,
        {
            "MAX_BATCH_SIZE": "4",
            "MAX_WAIT_MS": "0",
            "PEER_URLS": f"http://127.0.0.1:{PEER_PORT}",
            "SIMULATE_PREEMPTION_AFTER_SECONDS": "8",
            "DRAIN_EXIT_ON_COMPLETE": "0",
        },
    )
    print("[preemption] starting peer + primary...")
    peer.start()
    primary.start()
    rows: list[dict] = []
    try:
        await wait_for_server(peer.base_url)
        await wait_for_server(primary.base_url)
        started = time.time()
        samples, metrics_series, duration = await run_preemption_load(primary.base_url, duration_s=22.0)
        preemption_at = 8.0
        bucket_s = 1.0
        t0 = started
        max_t = started + duration
        bucket = t0
        while bucket < max_t:
            bucket_end = bucket + bucket_s
            bucket_samples = [s for s in samples if bucket <= s.timestamp < bucket_end]
            ok = sum(1 for s in bucket_samples if s.ok)
            total = len(bucket_samples)
            rows.append(
                {
                    "elapsed_s": bucket - t0,
                    "success_rate_pct": 100.0 * ok / total if total else 100.0,
                    "requests_in_bucket": total,
                    "preemption_at_s": preemption_at,
                }
            )
            bucket = bucket_end

        dropped = metrics_series[-1].requests_dropped_total if metrics_series else 0.0
        write_csv(
            output / "preemption_timeseries.csv",
            rows,
            ["elapsed_s", "success_rate_pct", "requests_in_bucket", "preemption_at_s"],
        )
        write_csv(
            output / "preemption_summary.csv",
            [{"requests_dropped_total": dropped, "total_requests": len(samples), "duration_s": duration}],
            ["requests_dropped_total", "total_requests", "duration_s"],
        )
        print(f"[preemption] requests_dropped_total={dropped}")
    finally:
        primary.stop()
        peer.stop()
        await asyncio.sleep(2)
    return rows


async def main(fast: bool = False, plots_only: bool = False) -> None:
    output = ensure_output_dir()

    if plots_only:
        _generate_plots_from_disk(output)
        return

    summaries: list[RunSummary] = []
    batch_sizes = [1, 8, 32] if fast else BATCH_SIZES

    utilization_rows = await run_batch_size_sweep(summaries, output, batch_sizes)
    await run_sla_comparison(summaries)
    autoscaling_rows = await run_autoscaling_scenario(output)
    preemption_rows = await run_preemption_scenario(output)

    write_summary_csv(output / "run_summaries.csv", summaries)
    write_csv(
        output / "cost_estimates.csv",
        [
            {
                "scenario": s.scenario,
                "batch_size": s.batch_size,
                "traffic_pattern": s.traffic_pattern,
                "throughput_rps": round(s.throughput_rps, 2),
                "cost_per_1000_usd": round(s.cost_per_1000_usd, 6),
            }
            for s in summaries
        ],
        ["scenario", "batch_size", "traffic_pattern", "throughput_rps", "cost_per_1000_usd"],
    )

    bursty_util = [
        r
        for r in utilization_rows
        if r.get("traffic_pattern") == "bursty" and int(r.get("batch_size", 0)) == 8
    ]
    if not bursty_util:
        bursty_util = [r for r in utilization_rows if r.get("traffic_pattern") == "bursty"]
    paths = generate_all_plots(summaries, autoscaling_rows, preemption_rows, bursty_util)
    print("\nGenerated plots:")
    for path in paths:
        print(f"  {path}")
    print(f"\nCSV outputs in {output}")


def _generate_plots_from_disk(output: Path) -> None:
    import csv

    summary_path = output / "run_summaries.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}; run the suite first.")

    summaries: list[RunSummary] = []
    with summary_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            summaries.append(
                RunSummary(
                    scenario=row["scenario"],
                    batch_size=int(float(row["batch_size"])),
                    traffic_pattern=row["traffic_pattern"],
                    priority_scheduling=row["priority_scheduling"] in ("True", "1", "true"),
                    request_count=int(float(row["request_count"])),
                    duration_s=float(row["duration_s"]),
                    throughput_rps=float(row["throughput_rps"]),
                    interactive_p50_ms=float(row["interactive_p50_ms"]),
                    interactive_p95_ms=float(row["interactive_p95_ms"]),
                    interactive_p99_ms=float(row["interactive_p99_ms"]),
                    batch_p50_ms=float(row["batch_p50_ms"]),
                    batch_p95_ms=float(row["batch_p95_ms"]),
                    batch_p99_ms=float(row["batch_p99_ms"]),
                    interactive_sla_pct=float(row["interactive_sla_pct"]),
                    batch_sla_pct=float(row["batch_sla_pct"]),
                    success_rate_pct=float(row["success_rate_pct"]),
                    requests_dropped_total=float(row["requests_dropped_total"]),
                    avg_gpu_utilization=float(row["avg_gpu_utilization"]),
                    avg_cpu_percent=float(row["avg_cpu_percent"]),
                    peak_queue_depth=float(row["peak_queue_depth"]),
                    cost_per_1000_usd=float(row["cost_per_1000_usd"]),
                )
            )

    def _read(name: str) -> list[dict]:
        path = output / name
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    autoscaling_rows = []
    for row in _read("autoscaling_timeseries.csv"):
        autoscaling_rows.append({k: float(v) for k, v in row.items()})

    preemption_rows = []
    for row in _read("preemption_timeseries.csv"):
        preemption_rows.append({k: float(v) for k, v in row.items()})

    util_rows = []
    for row in _read("utilization_timeseries.csv"):
        util_rows.append({k: v for k, v in row.items()})
    if not util_rows:
        util_rows = autoscaling_rows

    bursty_util = [
        r for r in util_rows if r.get("traffic_pattern") == "bursty" and str(r.get("batch_size")) == "8"
    ] or [r for r in util_rows if r.get("traffic_pattern") == "bursty"]

    paths = generate_all_plots(summaries, autoscaling_rows, preemption_rows, bursty_util)
    print("Generated plots:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run reduced batch-size sweep for CI")
    parser.add_argument("--plots-only", action="store_true", help="Regenerate PNGs from CSV outputs")
    args = parser.parse_args()
    asyncio.run(main(fast=args.fast, plots_only=args.plots_only))
