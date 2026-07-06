"""Shared benchmark utilities: metrics parsing, stats, CSV I/O."""

from __future__ import annotations

import csv
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from server.request_queue import BATCH_SLA_MS, INTERACTIVE_SLA_MS

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass
class RequestSample:
    timestamp: float
    priority: str
    latency_s: float
    status_code: int
    ok: bool


@dataclass
class MetricsSample:
    timestamp: float
    queue_depth: float
    gpu_utilization_percent: float
    cpu_percent: float
    requests_dropped_total: float = 0.0


@dataclass
class RunSummary:
    scenario: str
    batch_size: int
    traffic_pattern: str
    priority_scheduling: bool
    request_count: int
    duration_s: float
    throughput_rps: float
    interactive_p50_ms: float = 0.0
    interactive_p95_ms: float = 0.0
    interactive_p99_ms: float = 0.0
    batch_p50_ms: float = 0.0
    batch_p95_ms: float = 0.0
    batch_p99_ms: float = 0.0
    interactive_sla_pct: float = 0.0
    batch_sla_pct: float = 0.0
    success_rate_pct: float = 0.0
    requests_dropped_total: float = 0.0
    avg_gpu_utilization: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_queue_depth: float = 0.0
    cost_per_1000_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * pct) - 1))
    return ordered[index]


def parse_metric(metrics_text: str, name: str) -> float:
    match = re.search(rf"^{re.escape(name)}\s+(\S+)", metrics_text, re.MULTILINE)
    return float(match.group(1)) if match else 0.0


async def fetch_metrics(client: httpx.AsyncClient, base_url: str) -> MetricsSample:
    response = await client.get(f"{base_url.rstrip('/')}/metrics")
    response.raise_for_status()
    text = response.text
    cpu = 0.0
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=None)
    except Exception:
        pass
    return MetricsSample(
        timestamp=time.time(),
        queue_depth=parse_metric(text, "queue_depth"),
        gpu_utilization_percent=parse_metric(text, "gpu_utilization_percent"),
        cpu_percent=cpu,
        requests_dropped_total=parse_metric(text, "requests_dropped_total"),
    )


def sla_compliance(samples: list[RequestSample]) -> tuple[float, float]:
    interactive = [s.latency_s * 1000 for s in samples if s.priority == "interactive" and s.ok]
    batch = [s.latency_s * 1000 for s in samples if s.priority == "batch" and s.ok]
    interactive_pct = (
        100.0 * sum(1 for ms in interactive if ms <= INTERACTIVE_SLA_MS) / len(interactive)
        if interactive
        else 100.0
    )
    batch_pct = (
        100.0 * sum(1 for ms in batch if ms <= BATCH_SLA_MS) / len(batch) if batch else 100.0
    )
    return interactive_pct, batch_pct


def summarize_run(
    *,
    scenario: str,
    batch_size: int,
    traffic_pattern: str,
    priority_scheduling: bool,
    samples: list[RequestSample],
    metrics_series: list[MetricsSample],
    duration_s: float,
    cost_per_1000_usd: float,
    extra: dict[str, Any] | None = None,
) -> RunSummary:
    interactive = [s.latency_s for s in samples if s.priority == "interactive" and s.ok]
    batch = [s.latency_s for s in samples if s.priority == "batch" and s.ok]
    ok_count = sum(1 for s in samples if s.ok)
    interactive_sla, batch_sla = sla_compliance(samples)

    return RunSummary(
        scenario=scenario,
        batch_size=batch_size,
        traffic_pattern=traffic_pattern,
        priority_scheduling=priority_scheduling,
        request_count=len(samples),
        duration_s=duration_s,
        throughput_rps=len(samples) / duration_s if duration_s > 0 else 0.0,
        interactive_p50_ms=percentile(interactive, 0.50) * 1000,
        interactive_p95_ms=percentile(interactive, 0.95) * 1000,
        interactive_p99_ms=percentile(interactive, 0.99) * 1000,
        batch_p50_ms=percentile(batch, 0.50) * 1000,
        batch_p95_ms=percentile(batch, 0.95) * 1000,
        batch_p99_ms=percentile(batch, 0.99) * 1000,
        interactive_sla_pct=interactive_sla,
        batch_sla_pct=batch_sla,
        success_rate_pct=100.0 * ok_count / len(samples) if samples else 0.0,
        requests_dropped_total=metrics_series[-1].requests_dropped_total if metrics_series else 0.0,
        avg_gpu_utilization=statistics.mean(m.gpu_utilization_percent for m in metrics_series)
        if metrics_series
        else 0.0,
        avg_cpu_percent=statistics.mean(m.cpu_percent for m in metrics_series) if metrics_series else 0.0,
        peak_queue_depth=max((m.queue_depth for m in metrics_series), default=0.0),
        cost_per_1000_usd=cost_per_1000_usd,
        extra=extra or {},
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(path: Path, summaries: list[RunSummary]) -> None:
    if not summaries:
        return
    rows = []
    for summary in summaries:
        row = asdict(summary)
        row["extra"] = str(row.get("extra", {}))
        rows.append(row)
    fieldnames = list(rows[0].keys())
    write_csv(path, rows, fieldnames)


def implied_replicas(
    queue_depth: float,
    current_replicas: int = 1,
    target: float = 5.0,
    max_replicas: int = 5,
) -> int:
    if current_replicas <= 0:
        current_replicas = 1
    average = queue_depth / current_replicas
    if average <= target:
        return current_replicas
    return min(max_replicas, max(current_replicas, math.ceil(queue_depth / target)))
