#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
REPOSITORY_ROOT="$(cd -- "${PROJECT_ROOT}/../.." >/dev/null 2>&1 && pwd)"

: "${DATA_DIR:?set DATA_DIR to a local output directory}"
mkdir -p -- "${DATA_DIR}/train" "${DATA_DIR}/eval"
export PYTHONPATH="${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python3 "${SCRIPT_DIR}/prepare_data.py" --flashrag --dataset nq --split train --output "${DATA_DIR}/train/nq.parquet"
python3 "${SCRIPT_DIR}/prepare_data.py" --flashrag --dataset hotpotqa --split train --output "${DATA_DIR}/train/hotpotqa.parquet"

python3 -m examples.anysearch.cli merge-parquet \
  --sources "${DATA_DIR}/train/nq.parquet" "${DATA_DIR}/train/hotpotqa.parquet" \
  --output "${DATA_DIR}/train/nq_hotpotqa_train.parquet" \
  --seed "${DATA_SEED:-42}"

for dataset in nq triviaqa popqa hotpotqa 2wikimultihopqa musique bamboogle; do
  python3 "${SCRIPT_DIR}/prepare_data.py" \
    --flashrag \
    --dataset "${dataset}" \
    --split eval \
    --output "${DATA_DIR}/eval/${dataset}.parquet"
done

printf '%s\n' \
  "Prepared NQ and HotpotQA as a union shuffled with DATA_SEED=${DATA_SEED:-42}."
