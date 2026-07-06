"""Dry-run HPA scaling math from live /metrics (local or port-forwarded GKE pods)."""

from __future__ import annotations

import argparse
import math
import re
import sys
import urllib.error
import urllib.request


def fetch_queue_depth(url: str) -> float:
    metrics_url = url.rstrip("/") + "/metrics"
    try:
        with urllib.request.urlopen(metrics_url, timeout=5) as response:
            body = response.read().decode()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"Failed to fetch {metrics_url}: {exc}") from exc

    match = re.search(r"^queue_depth\s+(\S+)", body, re.MULTILINE)
    return float(match.group(1)) if match else 0.0


def implied_replicas(total_queue_depth: float, current_replicas: int, target: float, max_replicas: int) -> int:
    if current_replicas <= 0:
        current_replicas = 1
    average = total_queue_depth / current_replicas
    if average <= target:
        return current_replicas
    needed = math.ceil(total_queue_depth / target)
    return min(max_replicas, max(current_replicas, needed))


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate queue_depth HPA decisions")
    parser.add_argument("--url", action="append", default=[], help="Inference base URL (repeat for each pod)")
    parser.add_argument("--namespace", default="default", help="Namespace label for display only")
    parser.add_argument("--current-replicas", type=int, default=1)
    parser.add_argument("--target", type=float, default=5.0, help="HPA averageValue target")
    parser.add_argument("--max-replicas", type=int, default=5)
    args = parser.parse_args()

    urls = args.url or ["http://127.0.0.1:8000"]
    depths = [(url, fetch_queue_depth(url)) for url in urls]
    total = sum(depth for _, depth in depths)
    average = total / len(depths)
    desired = implied_replicas(total, args.current_replicas, args.target, args.max_replicas)

    print(f"namespace={args.namespace} pods={len(depths)} total_queue_depth={total:.1f} avg={average:.2f}")
    for url, depth in depths:
        print(f"  {url}: queue_depth={depth:.1f}")
    print(f"HPA target averageValue={args.target} current_replicas={args.current_replicas} -> desired={desired}")

    if average > args.target:
        print("ACTION: scale OUT (average queue depth exceeds target)")
    elif average < args.target / 2 and args.current_replicas > 1:
        print("ACTION: scale IN candidate (queue well below target)")
    else:
        print("ACTION: hold steady")


if __name__ == "__main__":
    main()
