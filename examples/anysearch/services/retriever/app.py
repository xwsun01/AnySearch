"""FastAPI application factory for the local AnySearch retriever."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class SearchBackend(Protocol):
    def batch_search(self, queries: list[str], top_k: int) -> list[list[dict[str, Any]]]: ...

    def info(self) -> dict[str, Any]: ...


class QueryRequest(BaseModel):
    queries: list[str] = Field(min_length=1)
    topk: int | None = None
    return_scores: bool = False


def create_app(
    backend: SearchBackend, *, default_top_k: int = 3, max_top_k: int = 20, concurrency: int = 1
) -> FastAPI:
    if default_top_k < 1 or max_top_k < default_top_k:
        raise ValueError("invalid top-k limits")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    app = FastAPI(title="AnySearch E5 Retriever", version="1.0")
    semaphore = asyncio.Semaphore(concurrency)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", **backend.info()}

    @app.post("/retrieve")
    async def retrieve(request: QueryRequest) -> dict[str, Any]:
        top_k = default_top_k if request.topk is None else request.topk
        if not 1 <= top_k <= max_top_k:
            raise HTTPException(status_code=422, detail=f"topk must be between 1 and {max_top_k}")
        queries = [query.strip() for query in request.queries]
        if any(not query for query in queries):
            raise HTTPException(status_code=422, detail="queries must not be blank")
        async with semaphore:
            results = await asyncio.to_thread(backend.batch_search, queries, top_k)
        if not request.return_scores:
            results = [
                [item.get("document", item) if isinstance(item, dict) else item for item in query_results]
                for query_results in results
            ]
        return {"result": results}

    return app
