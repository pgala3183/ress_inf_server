"""CLI entrypoint with --generative flag for uvicorn."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Resilient Inference Server")
    parser.add_argument(
        "--generative",
        action="store_true",
        help="Enable generative mode with continuous slot batching (distilgpt2)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.generative:
        os.environ["GENERATIVE"] = "1"

    import uvicorn

    uvicorn.run(
        "server.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
