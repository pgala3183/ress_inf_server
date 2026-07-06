"""HTTP API entrypoint with static or continuous batching."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.responses import Response

from server.batching_scheduler import BatchingScheduler
from server.config import DEFAULT_MAX_TOKENS, DRAIN_RETRY_AFTER_SECONDS, GENERATIVE
from server.drain_state import DrainManager, DrainState
from server.metrics import queue_depth, render_metrics, request_latency_seconds
from server.model_worker import ModelWorker
from server.peer_forwarder import PeerForwarder
from server.preemption_listener import PreemptionListener, configure_drain_logging
from server.request_queue import Priority, RequestQueue

RETRY_AFTER_HEADER = str(DRAIN_RETRY_AFTER_SECONDS)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    priority: Literal["interactive", "batch"] = "interactive"


class PredictResponse(BaseModel):
    label: str
    score: float


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    priority: Literal["interactive", "batch"] = "interactive"
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=512)


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    finish_reason: str | None = None


def create_app(generative: bool | None = None) -> FastAPI:
    """Build a FastAPI app; generative mode selects continuous slot batching."""
    use_generative = GENERATIVE if generative is None else generative
    model_worker = ModelWorker(generative=use_generative)
    request_queue: RequestQueue | None = None
    batching_scheduler: BatchingScheduler | None = None
    drain_manager = DrainManager()
    peer_forwarder = PeerForwarder()
    preemption_listener: PreemptionListener | None = None

    def _shutdown_after_drain() -> None:
        """Exit cleanly once drain completes (Spot preStop / simulation demo path)."""
        if os.environ.get("DRAIN_EXIT_ON_COMPLETE", "1") == "1":
            sys.exit(0)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal request_queue, batching_scheduler, preemption_listener
        configure_drain_logging()
        request_queue = RequestQueue()
        batching_scheduler = BatchingScheduler(
            request_queue,
            model_worker,
            generative=use_generative,
        )
        preemption_listener = PreemptionListener(
            drain_manager,
            request_queue,
            batching_scheduler,
            peer_forwarder,
            on_shutdown=_shutdown_after_drain,
        )
        app.state.drain_manager = drain_manager
        app.state.preemption_listener = preemption_listener
        app.state.request_queue = request_queue
        app.state.batching_scheduler = batching_scheduler
        await model_worker.load()
        await batching_scheduler.start()
        await preemption_listener.start()
        try:
            yield
        finally:
            if preemption_listener is not None:
                await preemption_listener.stop()
            if batching_scheduler is not None:
                await batching_scheduler.stop()
            request_queue = None
            batching_scheduler = None
            preemption_listener = None

    app = FastAPI(title="Resilient Inference Server", lifespan=lifespan)

    def _get_queue() -> RequestQueue:
        if request_queue is None:
            raise RuntimeError("Request queue is not initialized")
        return request_queue

    def _reject_if_draining() -> None:
        if not drain_manager.accepts_new_requests():
            raise HTTPException(
                status_code=503,
                detail="Server is draining; retry another instance",
                headers={"Retry-After": RETRY_AFTER_HEADER},
            )

    if not use_generative:

        @app.post("/predict", response_model=PredictResponse)
        async def predict(body: PredictRequest) -> PredictResponse:
            _reject_if_draining()
            queue = _get_queue()
            started = time.perf_counter()
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            priority: Priority = body.priority
            await queue.enqueue(body.text, future, priority)
            queue_depth.set(queue.depth)

            try:
                result = await future
                return PredictResponse(**result)
            finally:
                request_latency_seconds.labels(priority=priority).observe(time.perf_counter() - started)
                queue_depth.set(queue.depth)

    else:

        @app.post("/generate", response_model=GenerateResponse)
        async def generate(body: GenerateRequest) -> GenerateResponse:
            _reject_if_draining()
            queue = _get_queue()
            started = time.perf_counter()
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            priority: Priority = body.priority
            await queue.enqueue(
                body.prompt,
                future,
                priority,
                max_tokens=body.max_tokens,
            )
            queue_depth.set(queue.depth)

            try:
                result = await future
                return GenerateResponse(**result)
            finally:
                request_latency_seconds.labels(priority=priority).observe(time.perf_counter() - started)
                queue_depth.set(queue.depth)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        mode = "generative" if use_generative else "classification"
        if drain_manager.state == DrainState.DRAINING:
            return JSONResponse(
                {"status": "draining", "mode": mode, "drain_state": drain_manager.state.value},
                status_code=503,
            )
        if drain_manager.state == DrainState.DRAINED:
            return JSONResponse(
                {"status": "drained", "mode": mode, "drain_state": drain_manager.state.value},
                status_code=503,
            )
        return JSONResponse({"status": "ok", "mode": mode, "drain_state": drain_manager.state.value})

    @app.get("/metrics")
    async def metrics() -> Response:
        payload, content_type = render_metrics()
        return Response(content=payload, media_type=content_type)

    @app.post("/internal/drain")
    async def internal_drain() -> dict[str, str]:
        """Trigger graceful drain (preStop hook, kubectl delete, manual demo)."""
        if preemption_listener is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        await preemption_listener.trigger_drain("internal_drain_api")
        completed = await preemption_listener.wait_for_drain_complete(timeout=24.0)
        return {
            "drain_state": drain_manager.state.value,
            "completed": str(completed),
        }

    return app


app = create_app()
