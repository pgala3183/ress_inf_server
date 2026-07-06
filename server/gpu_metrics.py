"""GPU utilization sampling for Prometheus metrics."""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess

from server.metrics import gpu_utilization_percent

logger = logging.getLogger(__name__)


def read_gpu_utilization_percent() -> float:
    """Return average GPU utilization % via nvidia-smi, or 0.0 when no GPU is present."""
    if shutil.which("nvidia-smi") is None:
        return 0.0

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("nvidia-smi unavailable: %s", exc)
        return 0.0

    values: list[float] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError:
            continue

    if not values:
        return 0.0
    return sum(values) / len(values)


async def gpu_metrics_loop(interval_seconds: float = 5.0) -> None:
    """Background task that keeps gpu_utilization_percent up to date."""
    while True:
        utilization = await asyncio.get_running_loop().run_in_executor(None, read_gpu_utilization_percent)
        gpu_utilization_percent.set(utilization)
        await asyncio.sleep(interval_seconds)
