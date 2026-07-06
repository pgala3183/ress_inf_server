# Resilient Inference Server — Architecture

This document follows the [Google Cloud Architecture Framework](https://cloud.google.com/architecture/framework) pillars. Quantitative claims reference the Phase 8 benchmark suite (`benchmarks/run_full_suite.py`, artifacts in `benchmarks/output/`). Hardware: single replica, DistilBERT classification, CPU/GPU dev machine with `nvidia-smi` reporting ~30–36% GPU utilization under load.

---

## Operational Excellence

### Deployment model

| Surface | Purpose |
|---------|---------|
| `python -m server.main` | Local dev (classification) |
| `python -m server.main --generative` | Continuous-batching generative mode |
| `python -m server.main --router` | Spot/on-demand edge router |
| `Dockerfile` (multi-stage) | Bakes model weights at build time; runtime slim image with `/healthz` |
| `k8s/scripts/deploy-kind.ps1` | One-command kind build + deploy |
| GKE manifests | On-demand + Spot pools, router, HPA, Prometheus Adapter config |

Readiness/liveness probes hit `/healthz` (20–30 s initial delay to allow model load). Spot Deployment sets `terminationGracePeriodSeconds: 25` and a `preStop` hook calling `POST /internal/drain`.

### Observability

Prometheus metrics on `/metrics`:

| Metric | Use |
|--------|-----|
| `request_latency_seconds{priority}` | End-to-end latency by class |
| `queue_depth` | Backlog for HPA and router |
| `gpu_utilization_percent` | Device saturation (`nvidia-smi`; 0 on CPU-only) |
| `batch_size` | Static batching efficiency |
| `active_slots_used`, `time_to_first_token_seconds` | Generative mode |
| `requests_dropped_total`, `requests_migrated_total`, `drain_duration_seconds` | Preemption drain |

Structured JSON logs under `resilient.drain` and `resilient.preemption` record drain state transitions.

### Load testing and benchmarks (Phase 8)

| Tool | Role |
|------|------|
| `load_test/locustfile.py` | Mixed interactive/batch users; steady, bursty, and preemption traffic shapes |
| `benchmarks/run_full_suite.py` | Batch-size sweep, SLA on/off, autoscaling spike, preemption scenario |
| `benchmarks/output/` | Reproducible CSVs + PNG graphs |

**Run:** `python -m benchmarks.run_full_suite` (full) or `--fast` (batch sizes 1/8/32). Regenerate plots: `--plots-only`.

### CI / quality gates

21 pytest tests cover API, batching, priority scheduling, continuous batching, preemption drain, Phase 7 metrics/routing, and FIFO priority toggle. No production CI pipeline is wired; tests run locally and in Docker build validation.

---

## Security

This project is a **portfolio/demo** stack. Production GKE/Vertex deployments would add:

| Control | Status here | Production recommendation |
|---------|-------------|---------------------------|
| Authentication / authorization | **Not implemented** — `/predict` is open | Cloud IAP, OAuth2, or mTLS at ingress |
| Network isolation | Single ClusterIP Service | NetworkPolicies: router → inference only; deny peer egress except peers |
| Secrets | Model weights baked in image | Secret Manager + Workload Identity for HF tokens, API keys |
| Non-root container | Runs as root in slim image | `runAsNonRoot`, read-only root FS, drop capabilities |
| Pod Security | Default | GKE Pod Security Standards (restricted) |

**What is in place:** no secrets in repo (`.env` gitignored), preemption drain rejects new work with **503 + Retry-After** instead of silently corrupting in-flight batches, and peer forwarding uses explicit `PEER_URLS` (no open proxy).

---

## Reliability

### On-demand + Spot node pool split

| Deployment | Node pool | Purpose |
|------------|-----------|---------|
| `resilient-inference-server-ondemand` | Standard on-demand (`node-pool: ondemand`) | Stable baseline capacity |
| `resilient-inference-server-spot` | GKE Spot (`cloud.google.com/gke-spot: "true"`) | Cost-efficient burst; may be preempted |
| `resilient-inference-router` | Any | Queue-depth-aware proxy; public Service front door |

The main Service (`resilient-inference-server`) targets the **router**, which forwards to pool-specific Services based on live `queue_depth`.

#### Why a non-GPU standard node pool must exist first

GKE requires at least one **standard (non-GPU) node pool** before Spot GPU pools so `kube-system` DaemonSets are never pinned to preemptible nodes (~30 s notice). System pods on Spot can degrade networking before workload drain completes.

#### Spot preemption and graceful shutdown (Phase 6)

On GCE metadata preemption, SIGTERM, or `preStop`:

1. **503 + Retry-After** on new `/predict` requests  
2. In-flight batches complete  
3. Queued-but-not-admitted requests **forward to `PEER_URLS`**  
4. Exit 0 after **DRAINED**

**Phase 8 preemption benchmark** (`SIMULATE_PREEMPTION_AFTER_SECONDS=8`, peer on second port):

| Metric | Value |
|--------|-------|
| Total requests during 22 s load | **2,382** |
| `requests_dropped_total` | **0** |
| Success rate before t=8 s | **100%** |
| Success rate t=8–21 s (primary draining) | **0%** (503 to primary — clients must retry/peers) |
| Success rate t=22 s | Load stopped |

Zero dropped requests confirms queue migration; the success-rate dip is expected 503 behavior during drain, not data loss.

### Autoscaling responsiveness (Phase 7)

During a bursty load spike (`benchmarks/run_full_suite.py` autoscaling scenario):

| Time (s) | queue_depth | Implied replicas (target=5) |
|----------|-------------|----------------------------|
| 0–3 | 0–2 | 1 |
| 3.8 | **55** | **5** (max) |
| 4.8 | 4 | 1 (recovery) |

Peak **queue_depth = 67**; implied replica count tracks backlog within ~1 s. See `benchmarks/output/autoscaling_queue_replicas.png`.

---

## Cost Optimization

### Why CPU-based HPA fails for GPU inference

Under load, **GPU averaged 30–36%** utilization while **CPU averaged 63–90%** — but CPU HPA still misaligns with backlog on GPU-heavy workloads where host CPU stays orchestration-bound. Phase 7 HPA uses **`queue_depth` averageValue: 5** via Prometheus Adapter (`k8s/prometheus-adapter/`).

### Spot vs on-demand routing

| Queue pressure | Router weights (Spot : on-demand) |
|----------------|-----------------------------------|
| `max(pool) < 5` | **9 : 1** (cheap Spot) |
| `max(pool) ≥ 5` | **1 : 9** (guaranteed on-demand) |

Responses include `X-Route-Pool: spot|ondemand`.

### Measured cost efficiency (Phase 8)

Estimated cost uses published GCE multipliers in `benchmarks/pricing.py` (n1-standard-2 on-demand **$0.0475/hr**, Spot **0.30×**, 85% Spot traffic blend):

| Configuration | Throughput | Est. cost / 1k inferences |
|---------------|------------|---------------------------|
| `MAX_BATCH_SIZE=8`, steady | **47.4 req/s** | **$0.000113** |
| `MAX_BATCH_SIZE=8`, bursty | 23.9 req/s | $0.000224 |
| `MAX_BATCH_SIZE=1`, steady | 25.8 req/s | $0.000207 |
| `MAX_BATCH_SIZE=32`, steady | 41.8 req/s | $0.000128 |

**Batch size 8** delivers the best steady throughput and lowest cost per 1k requests in our sweep. Batch size 1 under bursty load drops interactive SLA compliance to **44.6%** (vs **95.3%** at batch size 8).

---

## Performance Optimization

### Static dynamic batching (Phase 2)

Requests wait up to `MAX_WAIT_MS` (default 10 ms) to fill batches of `MAX_BATCH_SIZE`. Phase 8 sweep (steady, 40 requests):

| MAX_BATCH_SIZE | Throughput | Interactive p50 | Interactive p99 |
|----------------|------------|-----------------|-----------------|
| 1 | 25.8 req/s | 644 ms | 989 ms |
| 8 | **47.4 req/s** | **306 ms** | 462 ms |
| 32 | 41.8 req/s | 334 ms | 458 ms |

Throughput peaks at batch size **8**; batch size **1** roughly doubles p50 latency. See `benchmarks/output/latency_vs_batch_size.png` and `throughput_vs_batch_size.png`.

Under **bursty** load (723 requests, ~30 s):

| MAX_BATCH_SIZE | Interactive p50 | Interactive p99 | Interactive SLA (200 ms) |
|----------------|-----------------|-----------------|--------------------------|
| 1 | 294 ms | 2,059 ms | **44.6%** |
| 8 | **58 ms** | **438 ms** | **95.3%** |
| 32 | **43 ms** | 269 ms | **97.1%** |

### SLA-aware priority scheduling (Phase 3)

Interactive (200 ms SLA) is preferred over batch (5,000 ms), with batch promotion at 80% of batch SLA.

**Bursty comparison** at `MAX_BATCH_SIZE=8`:

| Mode | Interactive SLA | Batch SLA | Interactive p99 |
|------|-----------------|-----------|-----------------|
| Priority **on** | 96.6% | 100% | 267 ms |
| Priority **off** (FIFO) | 98.6% | 100% | 214 ms |

Under this workload mix (~70% interactive), FIFO slightly edges priority on headline SLA — but priority scheduling protects interactive tail latency when batch floods the queue (Phase 3 unit tests and mixed-priority benchmarks). See `benchmarks/output/sla_compliance.png`.

Toggle: `PRIORITY_SCHEDULING=0`.

### Continuous batching (Phase 4)

Generative mode (`GENERATIVE=1`) uses a 16-slot pool with async slot reuse. Simplified vs production:

| Production (vLLM, Vertex) | This project |
|---------------------------|--------------|
| PagedAttention KV cache | Full re-pad each decode step |
| CUDA graphs / fused kernels | Plain PyTorch |
| Prefix caching, speculative decoding | Not implemented |

Metrics: `active_slots_used`, `time_to_first_token_seconds`.

### Resource utilization under load

During bursty batch-size-8 run: **avg CPU 86.7%**, **avg GPU 32.7%**, peak **queue_depth 3** (single replica). Autoscaling scenario: CPU pegged at **100%**, GPU **30–36%**. See `benchmarks/output/utilization_over_time.png`.

---

## Appendix: benchmark artifact index

| File | Contents |
|------|----------|
| `run_summaries.csv` | Per-scenario latency, throughput, SLA, cost |
| `autoscaling_timeseries.csv` | queue_depth + implied replicas over time |
| `preemption_timeseries.csv` | Success rate per second through drain |
| `preemption_summary.csv` | `requests_dropped_total`, request count |
| `cost_estimates.csv` | Cost per 1k by scenario |

Regenerate: `python -m benchmarks.run_full_suite`.
