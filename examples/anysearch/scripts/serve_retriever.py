#!/usr/bin/env python3
"""Launch the AnySearch E5-base-v2 + FAISS Flat retrieval service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

FORK_ROOT = Path(__file__).resolve().parents[3]
if str(FORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FORK_ROOT))


def main() -> None:
    import uvicorn

    from examples.anysearch.services.retriever.app import create_app
    from examples.anysearch.services.retriever.backend import E5FaissRetriever

    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--model", default="intfloat/e5-base-v2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-top-k", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--faiss-gpu", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    backend = E5FaissRetriever(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        model_name_or_path=args.model,
        use_fp16=not args.no_fp16,
        faiss_gpu=args.faiss_gpu,
        batch_size=args.batch_size,
    )
    app = create_app(
        backend,
        default_top_k=args.top_k,
        max_top_k=args.max_top_k,
        concurrency=args.concurrency,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
