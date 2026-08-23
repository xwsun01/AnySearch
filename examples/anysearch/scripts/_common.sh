#!/usr/bin/env bash
# Shared, side-effect-free helpers for AnySearch launchers.

set -Eeuo pipefail

EXPECTED_SLIME_COMMIT="52fc971bfe4ad7a1e857ac158d626d4b6373474d"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
BUNDLED_SLIME_ROOT="$(cd -- "${PROJECT_ROOT}/../.." >/dev/null 2>&1 && pwd)"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_file() {
  [[ -f "$1" ]] || die "required file not found: $1"
}

absolute_file() {
  local path="$1"
  local directory
  local filename
  require_file "${path}"
  directory="$(cd -- "$(dirname -- "${path}")" >/dev/null 2>&1 && pwd)"
  filename="$(basename -- "${path}")"
  printf '%s/%s\n' "${directory}" "${filename}"
}

absolute_dir() {
  local path="$1"
  require_dir "${path}"
  (cd -- "${path}" >/dev/null 2>&1 && pwd)
}

absolute_path() {
  local path="$1"
  if [[ "${path}" == /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${PWD}" "${path}"
  fi
}

require_dir() {
  [[ -d "$1" ]] || die "required directory not found: $1"
}

require_value() {
  local variable_name="$1"
  [[ -n "${!variable_name:-}" ]] || die "set ${variable_name} (see .env.example)"
}

initialize_paths() {
  ANYSEARCH_ROOT="${ANYSEARCH_ROOT:-${PROJECT_ROOT}}"
  # Always use the training framework bundled with this repository.
  SLIME_ROOT="${BUNDLED_SLIME_ROOT}"
  MEGATRON_ROOT="${MEGATRON_ROOT:-}"
  ANYSEARCH_CONFIG="${ANYSEARCH_CONFIG:-${ANYSEARCH_ROOT}/configs/anysearch.yaml}"
  ANYSEARCH_RETRIEVAL_URL="${ANYSEARCH_RETRIEVAL_URL:-${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}}"
  ANYSEARCH_RETRIEVAL_HEALTH_TIMEOUT="${ANYSEARCH_RETRIEVAL_HEALTH_TIMEOUT:-10}"
  RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  START_RAY="${START_RAY:-0}"
  export ANYSEARCH_ROOT SLIME_ROOT MEGATRON_ROOT ANYSEARCH_CONFIG ANYSEARCH_RETRIEVAL_URL
  export ANYSEARCH_RETRIEVAL_HEALTH_TIMEOUT
}

verify_retriever_health() {
  python3 "${ANYSEARCH_ROOT}/retrieval/health.py" \
    --url "${ANYSEARCH_RETRIEVAL_URL}" \
    --timeout "${ANYSEARCH_RETRIEVAL_HEALTH_TIMEOUT}"
}

verify_framework() {
  require_dir "${SLIME_ROOT}"
  require_file "${SLIME_ROOT}/train_async.py"
  require_file "${SLIME_ROOT}/slime/utils/arguments.py"
  require_file "${SLIME_ROOT}/UPSTREAM_SLIME_COMMIT"
  local upstream_commit
  IFS= read -r upstream_commit <"${SLIME_ROOT}/UPSTREAM_SLIME_COMMIT"
  if [[ "${upstream_commit}" != "${EXPECTED_SLIME_COMMIT}" ]]; then
    die "bundled Slime base ${upstream_commit} does not match the reviewed base ${EXPECTED_SLIME_COMMIT}"
  fi
}

select_model_args() {
  local backbone="$1"
  case "${backbone}" in
    qwen2.5-7b-instruct)
      MODEL_ARGS_FILE="${SLIME_ROOT}/scripts/models/qwen2.5-7B.sh"
      ;;
    llama3.1-8b-instruct)
      MODEL_ARGS_FILE="${SLIME_ROOT}/scripts/models/llama3.1-8B-Instruct.sh"
      ;;
    qwen3-4b)
      MODEL_ARGS_FILE="${SLIME_ROOT}/scripts/models/qwen3-4B.sh"
      ;;
    *)
      die "unsupported backbone '${backbone}'; expected qwen2.5-7b-instruct, llama3.1-8b-instruct, or qwen3-4b"
      ;;
  esac
  require_file "${MODEL_ARGS_FILE}"
  # shellcheck source=/dev/null
  source "${MODEL_ARGS_FILE}"
}

ray_dashboard_is_ready() {
  python3 -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1] + "/api/version", timeout=2)' \
    "${RAY_DASHBOARD_ADDRESS}" >/dev/null 2>&1
}

ensure_ray_cluster() {
  if ray_dashboard_is_ready; then
    return
  fi
  [[ "${START_RAY}" == "1" ]] || die "Ray dashboard is unavailable at ${RAY_DASHBOARD_ADDRESS}; start Ray or set START_RAY=1"
  require_command ray
  ray start \
    --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus 8 \
    --disable-usage-stats \
    --dashboard-host 127.0.0.1 \
    --dashboard-port 8265
}

build_runtime_env_json() {
  local runtime_pythonpath
  runtime_pythonpath="${SLIME_ROOT}:${MEGATRON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  ANYSEARCH_RUNTIME_PYTHONPATH="${runtime_pythonpath}" python3 -c '
import json
import os

names = (
    "ANYSEARCH_CONFIG",
    "ANYSEARCH_EVAL_OUTPUT_DIR",
    "ANYSEARCH_RETRIEVAL_URL",
    "EVAL_DATA_DIR",
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HUGGING_FACE_HUB_TOKEN",
    "WANDB_API_KEY",
    "WANDB_PROJECT",
)
env = {name: os.environ[name] for name in names if os.environ.get(name)}
env.update(
    {
        "PYTHONPATH": os.environ["ANYSEARCH_RUNTIME_PYTHONPATH"],
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
)
print(json.dumps({"env_vars": env}, separators=(",", ":")))
'
}

print_command() {
  printf 'command:'
  printf ' %q' "$@"
  printf '\n'
}
