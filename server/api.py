"""HTTP API entrypoint with static or continuous batching."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.responses import Response

from server.batching_scheduler import BatchingScheduler
from server.config import DEFAULT_MAX_TOKENS, GENERATIVE
from server.metrics import queue_depth, render_metrics, request_latency_seconds
from server.model_worker import ModelWorker
from server.request_queue import Priority, RequestQueue


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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal request_queue, batching_scheduler
        request_queue = RequestQueue()
        batching_scheduler = BatchingScheduler(
            request_queue,
            model_worker,
            generative=use_generative,
        )
        await model_worker.load()
        await batching_scheduler.start()
        try:
            yield
        finally:
            if batching_scheduler is not None:
                await batching_scheduler.stop()
            request_queue = None
            batching_scheduler = None

    app = FastAPI(title="Resilient Inference Server", lifespan=lifespan)

    def _get_queue() -> RequestQueue:
        if request_queue is None:
            raise RuntimeError("Request queue is not initialized")
        return request_queue

    if not use_generative:

        @app.post("/predict", response_model=PredictResponse)
        async def predict(body: PredictRequest) -> PredictResponse:
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
    async def healthz() -> dict[str, str]:
        mode = "generative" if use_generative else "classification"
        return {"status": "ok", "mode": mode}

    @app.get("/metrics")
    async def metrics() -> Response:
        payload, content_type = render_metrics()
        return Response(content=payload, media_type=content_type)

    return app


# Default application used by uvicorn (respects GENERATIVE env var).
app = create_app()
