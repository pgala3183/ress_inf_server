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
docker build -t resilient-inference-server .
docker run --rm -p 8000:8000 resilient-inference-server
```

Visit `http://localhost:8000/healthz` to verify the server is running.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product"}'
```

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
