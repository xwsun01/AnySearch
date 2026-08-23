"""Reliable async client for the local AnySearch E5 retrieval service.

The service is deliberately kept behind a small, typed boundary.  Rollout
workers share one :class:`aiohttp.ClientSession` per event loop, while each
client retains its own request limits and retry policy.
"""

from __future__ import annotations

import asyncio
import logging
import math
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class RetrievalClientError(RuntimeError):
    """Base class for errors returned by :class:`RetrievalClient`."""


class RetrievalRequestError(RetrievalClientError):
    """The retriever did not produce a successful response after retries."""


class RetrievalSchemaError(RetrievalClientError):
    """The retriever returned JSON that violates the documented contract."""


@dataclass(frozen=True, slots=True)
class RetrievalDocument:
    """One ranked document returned by the local dense retriever."""

    contents: str
    score: float | None = None
    document_id: str | int | None = None

    @property
    def title(self) -> str:
        return self.contents.split("\n", 1)[0].strip()

    @property
    def text(self) -> str:
        parts = self.contents.split("\n", 1)
        return parts[1].strip() if len(parts) == 2 else ""


def format_documents(documents: Sequence[RetrievalDocument]) -> str:
    """Format retrieved passages exactly once before environment injection."""

    lines = []
    for rank, document in enumerate(documents, start=1):
        body = document.text
        suffix = f" {body}" if body else ""
        lines.append(f"Doc {rank}(Title: {document.title}){suffix}")
    return "\n".join(lines)


_sessions: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, aiohttp.ClientSession] = weakref.WeakKeyDictionary()
_session_lock: asyncio.Lock | None = None
_session_lock_loop: asyncio.AbstractEventLoop | None = None


def _lock_for_running_loop() -> asyncio.Lock:
    global _session_lock, _session_lock_loop
    loop = asyncio.get_running_loop()
    if _session_lock is None or _session_lock_loop is not loop:
        _session_lock = asyncio.Lock()
        _session_lock_loop = loop
    return _session_lock


async def _shared_session(*, connector_limit: int) -> aiohttp.ClientSession:
    loop = asyncio.get_running_loop()
    session = _sessions.get(loop)
    if session is not None and not session.closed:
        return session
    async with _lock_for_running_loop():
        session = _sessions.get(loop)
        if session is None or session.closed:
            connector = aiohttp.TCPConnector(limit=connector_limit)
            # Request-specific timeouts are supplied by RetrievalClient.
            session = aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=None))
            _sessions[loop] = session
        return session


async def close_shared_sessions() -> None:
    """Close all live shared sessions (primarily useful for worker shutdown/tests)."""

    sessions = list(_sessions.values())
    _sessions.clear()
    if sessions:
        await asyncio.gather(*(session.close() for session in sessions if not session.closed))


_Sleep = Callable[[float], Awaitable[None]]
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class RetrievalClient:
    """Client for ``POST /retrieve`` with bounded retry and strict validation."""

    def __init__(
        self,
        base_url: str,
        *,
        top_k: int = 3,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        backoff_seconds: float = 0.25,
        max_backoff_seconds: float = 4.0,
        connector_limit: int = 256,
        concurrency: int = 256,
        sleep: _Sleep = asyncio.sleep,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if backoff_seconds < 0 or max_backoff_seconds < 0:
            raise ValueError("backoff durations cannot be negative")
        if connector_limit < 1 or concurrency < 1:
            raise ValueError("connection limits must be positive")

        normalized = base_url.rstrip("/")
        self.url = normalized if normalized.endswith("/retrieve") else f"{normalized}/retrieve"
        self.top_k = top_k
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.connector_limit = connector_limit
        self._semaphore = asyncio.Semaphore(concurrency)
        self._sleep = sleep

    async def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievalDocument]:
        """Retrieve ranked passages for one non-empty query."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        requested_top_k = self.top_k if top_k is None else top_k
        if requested_top_k < 1:
            raise ValueError("top_k must be positive")

        payload = {"queries": [normalized_query], "topk": requested_top_k, "return_scores": True}
        attempts = self.max_retries + 1
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                async with self._semaphore:
                    session = await _shared_session(connector_limit=self.connector_limit)
                    async with session.post(self.url, json=payload, timeout=self.timeout) as response:
                        body = await response.text()
                        if response.status >= 400:
                            message = f"retriever returned HTTP {response.status}: {body[:500]}"
                            if response.status not in _RETRYABLE_STATUS:
                                raise RetrievalRequestError(message)
                            raise _RetryableRetrievalError(message)
                        try:
                            decoded = await response.json()
                        except (aiohttp.ContentTypeError, ValueError) as exc:
                            raise RetrievalSchemaError("retriever response is not valid JSON") from exc
                return _parse_single_query_response(decoded, requested_top_k)
            except (RetrievalSchemaError, RetrievalRequestError):
                raise
            except (_RetryableRetrievalError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                delay = min(self.backoff_seconds * (2**attempt), self.max_backoff_seconds)
                logger.warning(
                    "retrieval attempt %d/%d failed (%s); retrying in %.2fs",
                    attempt + 1,
                    attempts,
                    exc,
                    delay,
                )
                await self._sleep(delay)

        raise RetrievalRequestError(
            f"retrieval failed after {attempts} attempt(s) for {self.url}: {last_error}"
        ) from last_error


class _RetryableRetrievalError(RuntimeError):
    pass


def _parse_single_query_response(payload: Any, requested_top_k: int) -> list[RetrievalDocument]:
    if not isinstance(payload, Mapping):
        raise RetrievalSchemaError("retriever response must be a JSON object")
    result = payload.get("result")
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], list):
        raise RetrievalSchemaError("retriever response.result must contain exactly one result list")
    if len(result[0]) > requested_top_k:
        raise RetrievalSchemaError("retriever returned more documents than requested")
    return [_parse_document(item, rank) for rank, item in enumerate(result[0], start=1)]


def _parse_document(item: Any, rank: int) -> RetrievalDocument:
    if not isinstance(item, Mapping):
        raise RetrievalSchemaError(f"retriever document at rank {rank} must be an object")
    raw_document = item.get("document", item)
    if not isinstance(raw_document, Mapping):
        raise RetrievalSchemaError(f"retriever document at rank {rank} has an invalid document field")
    contents = raw_document.get("contents")
    if not isinstance(contents, str) or not contents.strip():
        raise RetrievalSchemaError(f"retriever document at rank {rank} requires non-empty contents")

    raw_score = item.get("score")
    if raw_score is None:
        score = None
    elif isinstance(raw_score, (int, float)) and math.isfinite(float(raw_score)):
        score = float(raw_score)
    else:
        raise RetrievalSchemaError(f"retriever document at rank {rank} has an invalid score")

    document_id = raw_document.get("id", raw_document.get("document_id"))
    if document_id is not None and not isinstance(document_id, (str, int)):
        raise RetrievalSchemaError(f"retriever document at rank {rank} has an invalid id")
    return RetrievalDocument(contents=contents.strip(), score=score, document_id=document_id)
