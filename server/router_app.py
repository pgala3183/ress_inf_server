"""Edge router API — weighted routing between Spot and on-demand inference pools."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.responses import Response

from server.load_router import LoadRouter
from server.metrics import render_metrics


class RouterPredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    priority: str = "interactive"


def create_router_app(
    spot_url: str | None = None,
    ondemand_url: str | None = None,
) -> FastAPI:
    router = LoadRouter(spot_url=spot_url, ondemand_url=ondemand_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await router.start()
        try:
            yield
        finally:
            await router.stop()

    app = FastAPI(title="Resilient Inference Router", lifespan=lifespan)

    @app.post("/predict")
    async def predict(body: RouterPredictRequest, response: Response) -> dict:
        result, pool = await router.forward_predict(body.model_dump())
        response.headers["X-Route-Pool"] = pool
        return result

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "component": "router"}

    @app.get("/routing")
    async def routing_status() -> dict:
        return router.routing_snapshot()

    @app.get("/metrics")
    async def metrics() -> Response:
        payload, content_type = render_metrics()
        return Response(content=payload, media_type=content_type)

    return app


router_app = create_router_app()
