"""HTTP API entrypoint with dynamic request batching."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.responses import Response

from server.batching_scheduler import BatchingScheduler
from server.metrics import queue_depth, render_metrics, request_latency_seconds
from server.model_worker import ModelWorker
from server.request_queue import Priority, RequestQueue

model_worker = ModelWorker()
request_queue: RequestQueue | None = None
batching_scheduler: BatchingScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global request_queue, batching_scheduler
    request_queue = RequestQueue()
    batching_scheduler = BatchingScheduler(request_queue, model_worker)
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


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    priority: Literal["interactive", "batch"] = "interactive"


class PredictResponse(BaseModel):
    label: str
    score: float


def _get_queue() -> RequestQueue:
    if request_queue is None:
        raise RuntimeError("Request queue is not initialized")
    return request_queue


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    queue = _get_queue()
    started = time.perf_counter()
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    priority: Priority = request.priority
    await queue.enqueue(request.text, future, priority)
    queue_depth.set(queue.depth)

    try:
        result = await future
        return PredictResponse(**result)
    finally:
        request_latency_seconds.labels(priority=priority).observe(time.perf_counter() - started)
        queue_depth.set(queue.depth)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)
