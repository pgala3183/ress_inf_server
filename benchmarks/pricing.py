"""GCE pricing constants for cost-per-1000-inferences estimates.

Update these when region or machine type changes. Values are illustrative
multipliers based on published Google Cloud Compute pricing patterns:

- On-demand: full hourly rate
- Spot (preemptible): typically ~60–91% discount vs on-demand (we use 70%)

Sources (verify before production use):
  https://cloud.google.com/compute/docs/instances/spot
  https://cloud.google.com/compute/all-pricing
"""

from __future__ import annotations

# n1-standard-2, us-central1 (example CPU inference node), USD/hour
ONDEMAND_CPU_HOURLY_USD: float = 0.0475

# Spot pays roughly this fraction of on-demand (70% discount => 0.30x)
SPOT_ONDEMAND_MULTIPLIER: float = 0.30

# Optional GPU add-on (NVIDIA T4 attach), USD/hour on-demand
GPU_T4_ONDEMAND_HOURLY_USD: float = 0.35
GPU_SPOT_MULTIPLIER: float = 0.30

# Fraction of traffic served from Spot pool under low queue pressure (Phase 7 router)
DEFAULT_SPOT_TRAFFIC_FRACTION: float = 0.85


def effective_hourly_usd(*, use_gpu: bool = False, spot_fraction: float = DEFAULT_SPOT_TRAFFIC_FRACTION) -> float:
    """Blended hourly cost for one inference replica."""
    cpu = ONDEMAND_CPU_HOURLY_USD
    gpu = GPU_T4_ONDEMAND_HOURLY_USD if use_gpu else 0.0
    ondemand_rate = cpu + gpu
    spot_rate = (cpu * SPOT_ONDEMAND_MULTIPLIER) + (gpu * GPU_SPOT_MULTIPLIER)
    spot_fraction = max(0.0, min(1.0, spot_fraction))
    return spot_rate * spot_fraction + ondemand_rate * (1.0 - spot_fraction)


def cost_per_1000_inferences(
    throughput_rps: float,
    *,
    use_gpu: bool = False,
    spot_fraction: float = DEFAULT_SPOT_TRAFFIC_FRACTION,
    replicas: int = 1,
) -> float:
    """Estimate USD to serve 1000 successful inferences at observed throughput."""
    if throughput_rps <= 0:
        return 0.0
    seconds_for_1000 = 1000.0 / throughput_rps
    compute_hours = (seconds_for_1000 * replicas) / 3600.0
    return compute_hours * effective_hourly_usd(use_gpu=use_gpu, spot_fraction=spot_fraction)
