#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export ANYSEARCH_BACKBONE="${ANYSEARCH_BACKBONE:-qwen3-4b}"
exec bash "${SCRIPT_DIR}/run_train.sh" "$@"

