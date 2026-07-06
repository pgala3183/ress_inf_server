# Resilient Inference Server

A mini GPU/CPU inference serving system demonstrating production-grade patterns used by Google Cloud's Vertex AI Prediction and GKE Inference stack: continuous batching, SLA-aware priority scheduling, and graceful handling of Spot/preemptible VM preemption.

## Tech Stack

- Python 3.11, FastAPI, gRPC
- PyTorch (CPU for local dev)
- asyncio batching scheduler
- Prometheus metrics
- Docker + Kubernetes (GKE / kind / minikube)
- Locust load testing

## Quick Start

### Local (virtual environment)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn server.api:app --reload
```

### Generative mode (continuous batching)

```bash
# Option A: environment variable
GENERATIVE=1 uvicorn server.api:app --reload

# Option B: CLI flag
python -m server.main --generative

curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 32}'
```

### Docker

```bash
docker build -t resilient-inference-server:latest .
docker run --rm -p 8000:8000 resilient-inference-server:latest
```

Visit `http://localhost:8000/healthz` to verify the server is running.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product"}'
```

## Kubernetes Deployment

Manifests live in `k8s/`. Two paths — pick one:

### Path A — Local kind cluster (development / demo, no GCP billing)

Requires [kind](https://kind.sigs.k8s.io/) and kubectl.

```bash
# Build and load the image into kind
docker build -t resilient-inference-server:latest .
kind create cluster --name resilient-inf --config k8s/kind/cluster-config.yaml
kind load docker-image resilient-inference-server:latest --name resilient-inf

# Apply manifests (Spot taints are mocked on the second worker node)
kubectl apply -f k8s/deployment-ondemand.yaml \
               -f k8s/deployment-spot.yaml \
               -f k8s/service.yaml \
               -f k8s/hpa.yaml

kubectl get pods -o wide          # one pod per pool
kubectl port-forward svc/resilient-inference-server 8000:80
curl http://localhost:8000/healthz
```

Teardown: `kind delete cluster --name resilient-inf`

See also `k8s/kind/README.md`. minikube works similarly — build locally, set
`imagePullPolicy: Never`, and label/taint nodes to mimic Spot.

### Path B — GKE with on-demand + Spot GPU node pools (production-shaped)

Requires a GCP project with billing enabled.

```bash
# 1. Create the cluster (single zone for demo; expand for prod)
gcloud container clusters create resilient-inf \
  --zone us-central1-a \
  --num-nodes 1 \
  --machine-type e2-standard-4 \
  --node-labels node-pool=ondemand

# 2. Add a standard on-demand GPU pool (stable baseline — create BEFORE Spot GPU)
gcloud container node-pools create gpu-ondemand \
  --cluster resilient-inf \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --num-nodes 1 \
  --node-labels node-pool=ondemand \
  --node-taints nvidia.com/gpu=present:NoSchedule

# 3. Add a Spot GPU pool for cost-efficient burst capacity
gcloud container node-pools create gpu-spot \
  --cluster resilient-inf \
  --zone us-central1-a \
  --spot \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --num-nodes 1 \
  --node-labels cloud.google.com/gke-spot=true,node-pool=spot \
  --node-taints cloud.google.com/gke-spot=true:NoSchedule,nvidia.com/gpu=present:NoSchedule

# 4. Push image to Artifact Registry, then deploy
#    (adjust PROJECT_ID and uncomment nvidia.com/gpu limits in k8s/*.yaml)
docker tag resilient-inference-server:latest \
  us-central1-docker.pkg.dev/PROJECT_ID/resilient-inf/server:latest
docker push us-central1-docker.pkg.dev/PROJECT_ID/resilient-inf/server:latest

kubectl apply -f k8s/deployment-ondemand.yaml \
               -f k8s/deployment-spot.yaml \
               -f k8s/service.yaml \
               -f k8s/hpa.yaml
```

Why the standard pool comes first: GKE needs non-preemptible nodes for system
DaemonSets before Spot GPU nodes are added. See `docs/architecture.md` → Reliability.

### Demo: simulated Spot preemption (Phase 6)

```bash
# Terminal 1 — peer instance (stable on-demand target for migration)
PEER_URLS= uvicorn server.api:app --port 8001

# Terminal 2 — Spot instance with simulation + peer list
PEER_URLS=http://127.0.0.1:8001 \
SIMULATE_PREEMPTION_AFTER_SECONDS=30 \
DRAIN_EXIT_ON_COMPLETE=0 \
uvicorn server.api:app --port 8000

# Terminal 3 — load while watching structured JSON drain logs
python benchmarks/static_batching_latency.py
# Or: kubectl delete pod <spot-pod>   (same drain path via preStop + SIGTERM)
```

Watch for log events: `drain_state_transition`, `pending_queue_migrated`,
`scheduler_idle`, `drain_orchestration_complete`. Metrics at `/metrics`:
`requests_dropped_total` (should stay 0), `requests_migrated_total`, `drain_duration_seconds`.

## Project Layout

```
server/           Core serving modules
k8s/              Kubernetes manifests
load_test/        Locust scenarios
benchmarks/       Benchmark scripts and output graphs
docs/             Architecture documentation
tests/            Unit and integration tests
```

## Development Phases

Built incrementally — each phase is fully working and tested before moving to the next. See `docs/architecture.md` for the evolving design.
