"""Validate Kubernetes manifest YAML syntax and required fields."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = [
    ROOT / "deployment-ondemand.yaml",
    ROOT / "deployment-spot.yaml",
    ROOT / "service.yaml",
    ROOT / "hpa.yaml",
]


def load_documents(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(yaml.safe_load_all(handle))


def main() -> int:
    errors: list[str] = []

    for manifest in MANIFESTS:
        if not manifest.exists():
            errors.append(f"missing file: {manifest}")
            continue
        docs = load_documents(manifest)
        if not docs:
            errors.append(f"empty manifest: {manifest.name}")

    spot = load_documents(ROOT / "deployment-spot.yaml")[0]
    spec = spot["spec"]["template"]["spec"]
    if spec.get("terminationGracePeriodSeconds") != 25:
        errors.append("spot deployment must set terminationGracePeriodSeconds: 25")
    if "preStop" not in spec["containers"][0].get("lifecycle", {}):
        errors.append("spot deployment missing preStop lifecycle hook")

    ondemand = load_documents(ROOT / "deployment-ondemand.yaml")[0]
    if ondemand["spec"]["template"]["spec"].get("nodeSelector", {}).get("node-pool") != "ondemand":
        errors.append("ondemand deployment missing node-pool selector")

    service = load_documents(ROOT / "service.yaml")[0]
    if service["spec"]["selector"].get("app") != "resilient-inference-server":
        errors.append("service selector must target app=resilient-inference-server")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: validated {len(MANIFESTS)} manifest files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
