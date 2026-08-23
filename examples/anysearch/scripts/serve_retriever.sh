#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
REPOSITORY_ROOT="$(cd -- "${PROJECT_ROOT}/../.." >/dev/null 2>&1 && pwd)"

: "${RETRIEVER_INDEX:?set RETRIEVER_INDEX to the FAISS Flat index}"
: "${RETRIEVER_CORPUS:?set RETRIEVER_CORPUS to the Wikipedia 2018 JSONL corpus}"

[[ -f "${RETRIEVER_INDEX}" ]] || { printf 'error: index not found: %s\n' "${RETRIEVER_INDEX}" >&2; exit 1; }
[[ -f "${RETRIEVER_CORPUS}" ]] || { printf 'error: corpus not found: %s\n' "${RETRIEVER_CORPUS}" >&2; exit 1; }

export PYTHONPATH="${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
RETRIEVER_ARGS=(
  --index-path "${RETRIEVER_INDEX}"
  --corpus-path "${RETRIEVER_CORPUS}"
  --model "${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
  --host "${RETRIEVER_HOST:-127.0.0.1}"
  --port "${RETRIEVER_PORT:-8000}"
  --top-k 3
  --concurrency "${RETRIEVER_CONCURRENCY:-1}"
)
if [[ "${RETRIEVER_FAISS_GPU:-1}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="${RETRIEVER_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
  RETRIEVER_ARGS+=(--faiss-gpu)
fi
exec python3 "${SCRIPT_DIR}/serve_retriever.py" "${RETRIEVER_ARGS[@]}"
