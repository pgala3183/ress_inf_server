"""Forward queued requests to healthy peer instances during drain."""

from __future__ import annotations

import json
import logging
import os
from itertools import cycle
from typing import Any

import httpx

from server.request_queue import Priority

logger = logging.getLogger("resilient.drain")


class PeerForwarder:
    """Round-robin HTTP forwarding to peer inference servers."""

    def __init__(self, peer_urls: list[str] | None = None) -> None:
        if peer_urls is None:
            peer_urls = [
                url.strip().rstrip("/")
                for url in os.environ.get("PEER_URLS", "").split(",")
                if url.strip()
            ]
        self._peer_urls = peer_urls
        self._cycle = cycle(self._peer_urls) if self._peer_urls else None

    @property
    def has_peers(self) -> bool:
        return bool(self._peer_urls)

    def next_peer(self) -> str:
        if not self._cycle:
            raise RuntimeError("No peer URLs configured (set PEER_URLS)")
        return next(self._cycle)

    async def forward_predict(
        self,
        text: str,
        priority: Priority = "interactive",
        *,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any]:
        peer = self.next_peer()
        url = f"{peer}/predict"
        payload = {"text": text, "priority": priority}
        logger.info(
            json.dumps(
                {
                    "event": "peer_forward_started",
                    "peer_url": peer,
                    "priority": priority,
                }
            )
        )

        owns_client = client is None
        http = client or httpx.AsyncClient(timeout=60.0)
        try:
            response = await http.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            logger.info(
                json.dumps(
                    {
                        "event": "peer_forward_completed",
                        "peer_url": peer,
                        "status_code": response.status_code,
                    }
                )
            )
            return result
        finally:
            if owns_client:
                await http.aclose()
