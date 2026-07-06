# Resilient Inference Server — Architecture

This document follows the [Google Cloud Architecture Framework](https://cloud.google.com/architecture/framework) pillars. Sections will be filled in as each build phase is completed.

## Operational Excellence

<!-- Phase TBD: deployment, observability, load testing, CI/CD -->

## Security

<!-- Phase TBD: authentication, network policies, secrets management -->

## Reliability

<!-- Phase TBD: preemption handling, graceful shutdown, request durability -->

## Cost Optimization

<!-- Phase TBD: Spot/preemptible VMs, autoscaling, resource sizing -->

## Performance Optimization

### Continuous batching (Phase 4)

The generative scheduler maintains a **fixed-size slot pool** (default 16). Each slot
holds one in-flight autoregressive sequence. When a sequence hits EOS or `max_tokens`,
its slot is freed immediately and can accept a new request from the priority queue —
remaining sequences do **not** wait for the slowest member of a batch.

This mirrors the scheduling pattern used by production engines (vLLM, TensorRT-LLM,
Google Cloud's inference stack), but with deliberate simplifications:

| Production (e.g. vLLM) | This project (Phase 4) |
|------------------------|-------------------------|
| PagedAttention / block-sparse KV cache | Full re-pad + forward pass each decode step |
| Custom CUDA kernels, CUDA graphs | Plain PyTorch `AutoModelForCausalLM` |
| Prefix caching, speculative decoding | Not implemented |
| Preemption-safe checkpointing | Phase TBD |

We implement the **asynchronous slot-reuse scheduling loop** only; compute efficiency
is secondary to demonstrating the pattern. Key metrics: `active_slots_used`,
`time_to_first_token_seconds`, and per-priority `request_latency_seconds`.

Enable generative mode: `GENERATIVE=1 uvicorn server.api:app` or
`python -m server.main --generative`.

### SLA-aware priority scheduling (Phase 3)

Interactive requests (200 ms SLA target) are drained before batch-class requests
(5000 ms SLA target), with deadline-based promotion at 80% of batch SLA to prevent
starvation.

### Static dynamic batching (Phase 2)

Classification mode batches requests within a `MAX_WAIT_MS` window up to
`MAX_BATCH_SIZE` before a single forward pass.
