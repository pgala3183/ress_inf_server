"""Prometheus metrics for the inference server."""

from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest

request_latency_seconds = Histogram(
    "request_latency_seconds",
    "End-to-end request latency in seconds",
    labelnames=["priority"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0),
)

batch_size = Histogram(
    "batch_size",
    "Number of requests processed per batch",
    buckets=(1, 2, 3, 4, 5, 8, 10, 16, 32, 64),
)

queue_depth = Gauge(
    "queue_depth",
    "Current number of requests waiting in the queue",
)

# Phase 4 — continuous batching observability
active_slots_used = Gauge(
    "active_slots_used",
    "Number of slot-pool entries currently running a generative sequence",
)

time_to_first_token_seconds = Histogram(
    "time_to_first_token_seconds",
    "Wall time from slot assignment to first generated token",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
