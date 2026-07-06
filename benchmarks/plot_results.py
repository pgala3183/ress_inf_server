"""Generate matplotlib PNG charts from benchmark CSV outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from benchmarks.common import RunSummary, ensure_output_dir


def _save(fig: plt.Figure, name: str) -> Path:
    out = ensure_output_dir() / name
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_latency_vs_batch_size(summaries: list[RunSummary]) -> Path:
    steady = [s for s in summaries if s.traffic_pattern == "steady"]
    steady.sort(key=lambda s: s.batch_size)
    sizes = [s.batch_size for s in steady]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, [s.interactive_p50_ms for s in steady], marker="o", label="interactive p50")
    ax.plot(sizes, [s.interactive_p99_ms for s in steady], marker="o", label="interactive p99")
    ax.plot(sizes, [s.batch_p50_ms for s in steady], marker="s", label="batch p50")
    ax.plot(sizes, [s.batch_p99_ms for s in steady], marker="s", label="batch p99")
    ax.set_xlabel("MAX_BATCH_SIZE")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs batch size (steady traffic)")
    ax.set_xticks(sizes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, "latency_vs_batch_size.png")


def plot_throughput_vs_batch_size(summaries: list[RunSummary]) -> Path:
    rows = sorted(
        [s for s in summaries if s.traffic_pattern in ("steady", "bursty")],
        key=lambda s: (s.batch_size, s.traffic_pattern),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for pattern, marker in (("steady", "o"), ("bursty", "s")):
        subset = [s for s in rows if s.traffic_pattern == pattern]
        if not subset:
            continue
        ax.plot(
            [s.batch_size for s in subset],
            [s.throughput_rps for s in subset],
            marker=marker,
            label=pattern,
        )
    ax.set_xlabel("MAX_BATCH_SIZE")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput vs batch size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, "throughput_vs_batch_size.png")


def plot_sla_compliance(summaries: list[RunSummary]) -> Path:
    rows = [s for s in summaries if s.scenario.startswith("sla_")]
    if len(rows) < 2:
        return ensure_output_dir() / "sla_compliance.png"

    labels = ["interactive SLA", "batch SLA"]
    on = next(s for s in rows if s.priority_scheduling)
    off = next(s for s in rows if not s.priority_scheduling)
    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], [on.interactive_sla_pct, on.batch_sla_pct], width, label="priority on")
    ax.bar([i + width / 2 for i in x], [off.interactive_sla_pct, off.batch_sla_pct], width, label="priority off (FIFO)")
    ax.set_ylabel("SLA compliance (%)")
    ax.set_title("SLA compliance: priority scheduling on vs off")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, "sla_compliance.png")


def plot_autoscaling_responsiveness(timeseries_rows: list[dict]) -> Path:
    if not timeseries_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No autoscaling timeseries data", ha="center")
        return _save(fig, "autoscaling_queue_replicas.png")

    t0 = timeseries_rows[0]["elapsed_s"]
    elapsed = [r["elapsed_s"] - t0 for r in timeseries_rows]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(elapsed, [r["queue_depth"] for r in timeseries_rows], color="tab:blue", label="queue_depth")
    ax1.set_xlabel("Elapsed (s)")
    ax1.set_ylabel("Queue depth", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(elapsed, [r["implied_replicas"] for r in timeseries_rows], color="tab:orange", label="implied replicas")
    ax2.set_ylabel("Implied replica count", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.set_title("Queue depth + implied replica count during load spike")
    ax1.grid(True, alpha=0.3)
    return _save(fig, "autoscaling_queue_replicas.png")


def plot_preemption_success_rate(timeseries_rows: list[dict]) -> Path:
    if not timeseries_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No preemption timeseries data", ha="center")
        return _save(fig, "preemption_success_rate.png")

    elapsed = [r["elapsed_s"] for r in timeseries_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(elapsed, [r["success_rate_pct"] for r in timeseries_rows], marker="o", markersize=3)
    ax.axvline(timeseries_rows[0].get("preemption_at_s", 0), color="red", linestyle="--", label="preemption")
    ax.set_xlabel("Elapsed (s)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Request success rate during simulated preemption")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, "preemption_success_rate.png")


def plot_utilization_over_time(timeseries_rows: list[dict], filename: str, title: str) -> Path:
    if not timeseries_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No utilization data", ha="center")
        return _save(fig, filename)

    if "elapsed_s" in timeseries_rows[0]:
        t0 = timeseries_rows[0]["elapsed_s"]
        elapsed = [r["elapsed_s"] - t0 for r in timeseries_rows]
    else:
        t0 = timeseries_rows[0]["timestamp"]
        elapsed = [r["timestamp"] - t0 for r in timeseries_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(elapsed, [r.get("cpu_percent", 0) for r in timeseries_rows], label="CPU %")
    ax.plot(elapsed, [r.get("gpu_utilization_percent", 0) for r in timeseries_rows], label="GPU %")
    ax.set_xlabel("Elapsed (s)")
    ax.set_ylabel("Utilization (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, filename)


def generate_all_plots(
    summaries: list[RunSummary],
    autoscaling_rows: list[dict],
    preemption_rows: list[dict],
    utilization_rows: list[dict],
) -> list[Path]:
    paths = [
        plot_latency_vs_batch_size(summaries),
        plot_throughput_vs_batch_size(summaries),
        plot_sla_compliance(summaries),
        plot_autoscaling_responsiveness(autoscaling_rows),
        plot_preemption_success_rate(preemption_rows),
        plot_utilization_over_time(
            utilization_rows,
            "utilization_over_time.png",
            "CPU/GPU utilization during bursty load",
        ),
    ]
    return paths
