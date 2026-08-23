"""Fail-fast validation for the canonical AnySearch retrieval service.

This module intentionally uses only the Python standard library so launchers can
run the preflight before importing the training or retrieval dependency stacks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol
from urllib import request
from urllib.parse import SplitResult, urlsplit, urlunsplit

EXPECTED_BACKEND = "e5-faiss-flat"
EXPECTED_MODEL = "intfloat/e5-base-v2"
EXPECTED_METRIC = "inner_product"
DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_RESPONSE_BYTES = 64 * 1024


class RetrieverHealthError(RuntimeError):
    """The retrieval endpoint or its health response is not canonical."""


class _ReadableResponse(Protocol):
    status: int

    def read(self, amount: int = -1) -> bytes: ...

    def __enter__(self) -> _ReadableResponse: ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...


UrlOpener = Callable[..., _ReadableResponse]


def derive_health_url(retrieval_url: str) -> str:
    """Return the root ``/health`` URL for a safe HTTP(S) retrieval URL."""

    if not isinstance(retrieval_url, str) or not retrieval_url:
        raise RetrieverHealthError("ANYSEARCH_RETRIEVAL_URL must be a non-empty URL")
    if retrieval_url != retrieval_url.strip() or any(character.isspace() for character in retrieval_url):
        raise RetrieverHealthError("ANYSEARCH_RETRIEVAL_URL must not contain whitespace")
    if "?" in retrieval_url:
        raise RetrieverHealthError("ANYSEARCH_RETRIEVAL_URL must not contain a query")
    if "#" in retrieval_url:
        raise RetrieverHealthError("ANYSEARCH_RETRIEVAL_URL must not contain a fragment")

    try:
        parsed = urlsplit(retrieval_url)
        hostname = parsed.hostname
        # Accessing ``port`` also validates malformed and out-of-range ports.
        _ = parsed.port
    except ValueError as exc:
        raise RetrieverHealthError("ANYSEARCH_RETRIEVAL_URL is malformed") from exc

    if parsed.scheme.lower() not in {"http", "https"}:
        raise RetrieverHealthError("ANYSEARCH_RETRIEVAL_URL must use http or https")
    if not parsed.netloc or hostname is None:
        raise RetrieverHealthError("ANYSEARCH_RETRIEVAL_URL must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise RetrieverHealthError("ANYSEARCH_RETRIEVAL_URL must not contain user information")
    health = SplitResult(parsed.scheme.lower(), parsed.netloc, "/health", "", "")
    return urlunsplit(health)


def validate_health_payload(payload: object) -> dict[str, Any]:
    """Validate and return one canonical retriever health payload."""

    if not isinstance(payload, Mapping):
        raise RetrieverHealthError("retriever /health response must be a JSON object")

    expected_fields = {
        "status": "ok",
        "backend": EXPECTED_BACKEND,
        "model": EXPECTED_MODEL,
        "index_metric": EXPECTED_METRIC,
    }
    for field, expected in expected_fields.items():
        observed = payload.get(field)
        if observed != expected:
            raise RetrieverHealthError(
                f"retriever /health field {field!r} must be {expected!r}; received {observed!r}"
            )

    index_type = payload.get("index_type")
    if not isinstance(index_type, str) or not index_type.startswith("IndexFlat"):
        raise RetrieverHealthError("retriever /health field 'index_type' must identify an IndexFlat family index")

    index_size = payload.get("index_size")
    corpus_size = payload.get("corpus_size")
    if (
        not isinstance(index_size, int)
        or isinstance(index_size, bool)
        or not isinstance(corpus_size, int)
        or isinstance(corpus_size, bool)
        or index_size <= 0
        or corpus_size <= 0
        or index_size != corpus_size
    ):
        raise RetrieverHealthError("retriever /health requires equal positive integer index_size and corpus_size")

    return dict(payload)


def fetch_health(
    health_url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    opener: UrlOpener = request.urlopen,
) -> dict[str, Any]:
    """Fetch, decode, and validate one retriever health response."""

    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise RetrieverHealthError("retriever health timeout must be a positive finite number")
    timeout = float(timeout)
    if timeout <= 0 or not math.isfinite(timeout):
        raise RetrieverHealthError("retriever health timeout must be a positive finite number")

    health_request = request.Request(health_url, headers={"Accept": "application/json"}, method="GET")
    try:
        with opener(health_request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            if status != 200:
                raise RetrieverHealthError(f"retriever /health returned HTTP {status!r}, expected 200")
            raw_payload = response.read(MAX_RESPONSE_BYTES + 1)
    except RetrieverHealthError:
        raise
    except Exception as exc:
        raise RetrieverHealthError(f"retriever /health request failed: {exc}") from exc

    if len(raw_payload) > MAX_RESPONSE_BYTES:
        raise RetrieverHealthError("retriever /health response exceeds 64 KiB")
    try:
        payload = json.loads(raw_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetrieverHealthError("retriever /health response is not valid UTF-8 JSON") from exc
    return validate_health_payload(payload)


def check_retriever_health(
    retrieval_url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    opener: UrlOpener = request.urlopen,
) -> dict[str, Any]:
    """Derive, fetch, and validate the canonical retriever health endpoint."""

    return fetch_health(derive_health_url(retrieval_url), timeout=timeout, opener=opener)


def _positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if timeout <= 0 or not math.isfinite(timeout):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return timeout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the canonical AnySearch retriever before launch.")
    parser.add_argument(
        "--url",
        default=os.environ.get("ANYSEARCH_RETRIEVAL_URL"),
        help="retrieval endpoint (default: ANYSEARCH_RETRIEVAL_URL)",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_timeout,
        default=os.environ.get("ANYSEARCH_RETRIEVAL_HEALTH_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)),
        help="request timeout in seconds (default: ANYSEARCH_RETRIEVAL_HEALTH_TIMEOUT or 10)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("--url or ANYSEARCH_RETRIEVAL_URL is required")
    try:
        payload = check_retriever_health(args.url, timeout=args.timeout)
    except RetrieverHealthError as exc:
        print(f"retriever preflight failed: {exc}", file=sys.stderr)
        return 1

    print(
        "retriever preflight passed: "
        f"{payload['backend']} {payload['model']} {payload['index_type']} "
        f"({payload['index_size']} documents)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI entry point
    raise SystemExit(main())
