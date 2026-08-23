#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
REPOSITORY_ROOT="$(cd -- "${PROJECT_ROOT}/../.." >/dev/null 2>&1 && pwd)"

: "${RESULTS_DIR:?set RESULTS_DIR to the directory containing evaluation JSON files}"
SUMMARY_PATH="${SUMMARY_PATH:-${RESULTS_DIR}/summary.csv}"
PLOT_PATH="${PLOT_PATH:-${RESULTS_DIR}/budget_curves.png}"
export PYTHONPATH="${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python3 "${SCRIPT_DIR}/summarize_results.py" --results-dir "${RESULTS_DIR}" --output "${SUMMARY_PATH}"
python3 "${SCRIPT_DIR}/plot_results.py" --summary "${SUMMARY_PATH}" --output "${PLOT_PATH}"
