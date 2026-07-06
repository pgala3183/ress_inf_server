"""Runtime configuration for the inference server.

Generative / continuous-batching mode can be enabled via:
  - Environment: GENERATIVE=1
  - CLI: python -m server.main --generative
"""

from __future__ import annotations

import os

# When True, use autoregressive generation with slot-based continuous batching.
GENERATIVE: bool = os.environ.get("GENERATIVE", "").lower() in ("1", "true", "yes")

CLASSIFICATION_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
GENERATIVE_MODEL_ID = os.environ.get("GENERATIVE_MODEL_ID", "distilgpt2")

# Static (Phase 2) batching knobs — used only in classification mode.
MAX_BATCH_SIZE: int = int(os.environ.get("MAX_BATCH_SIZE", "8"))
MAX_WAIT_MS: float = float(os.environ.get("MAX_WAIT_MS", "10"))

# Continuous (Phase 4) batching knobs — used only in generative mode.
MAX_SLOTS: int = int(os.environ.get("MAX_SLOTS", "16"))
DEFAULT_MAX_TOKENS: int = int(os.environ.get("DEFAULT_MAX_TOKENS", "50"))

# Step delay between decode iterations (seconds). Zero in production; tests may
# increase slightly to make slot-reuse timing assertions more reliable.
STEP_INTERVAL_S: float = float(os.environ.get("STEP_INTERVAL_S", "0"))

# Phase 6 — graceful drain / Spot preemption
DRAIN_RETRY_AFTER_SECONDS: int = int(os.environ.get("DRAIN_RETRY_AFTER_SECONDS", "30"))

# Phase 7 — autoscaling / routing
QUEUE_PRESSURE_THRESHOLD: float = float(os.environ.get("QUEUE_PRESSURE_THRESHOLD", "5"))
HPA_QUEUE_DEPTH_TARGET: float = float(os.environ.get("HPA_QUEUE_DEPTH_TARGET", "5"))
GPU_METRICS_INTERVAL_S: float = float(os.environ.get("GPU_METRICS_INTERVAL_S", "5"))

# Phase 3 — SLA-aware priority scheduling (disable for FIFO baseline benchmarks)
PRIORITY_SCHEDULING: bool = os.environ.get("PRIORITY_SCHEDULING", "1").lower() in ("1", "true", "yes")


def set_generative(enabled: bool) -> None:
    """Override generative mode at runtime (used by tests)."""
    global GENERATIVE
    GENERATIVE = enabled
