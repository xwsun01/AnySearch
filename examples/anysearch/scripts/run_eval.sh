#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=scripts/_common.sh
source "${SCRIPT_DIR}/_common.sh"

BACKBONE="${ANYSEARCH_BACKBONE:-qwen2.5-7b-instruct}"
EVAL_CONFIG=""
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/run_eval.sh [--model NAME] [--eval-config PATH] [--dry-run] [-- EVAL_ARGS...]

Required environment: HF_CHECKPOINT, LOAD_CHECKPOINT, EVAL_DATA_DIR,
ANYSEARCH_EVAL_OUTPUT_DIR, MEGATRON_ROOT. Evaluation covers all seven datasets
at B=1..8 when the default config is used.
EOF
}

while (($#)); do
  case "$1" in
    --model)
      (($# >= 2)) || die "--model requires a value"
      BACKBONE="$2"
      shift 2
      ;;
    --eval-config)
      (($# >= 2)) || die "--eval-config requires a value"
      EVAL_CONFIG="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      die "unknown launcher argument '$1'; pass additional evaluation arguments after --"
      ;;
  esac
done

initialize_paths
EVAL_CONFIG="${EVAL_CONFIG:-${ANYSEARCH_ROOT}/configs/eval/budget_b1_b8.yaml}"
require_command python3
require_value MEGATRON_ROOT
require_value HF_CHECKPOINT
require_value LOAD_CHECKPOINT
require_value EVAL_DATA_DIR
require_value ANYSEARCH_EVAL_OUTPUT_DIR
require_dir "${MEGATRON_ROOT}"
require_dir "${HF_CHECKPOINT}"
require_dir "${LOAD_CHECKPOINT}"
require_dir "${EVAL_DATA_DIR}"
for dataset in nq triviaqa popqa hotpotqa 2wikimultihopqa musique bamboogle; do
  require_file "${EVAL_DATA_DIR}/${dataset}.parquet"
done
require_file "${EVAL_CONFIG}"
require_file "${ANYSEARCH_CONFIG}"
MEGATRON_ROOT="$(absolute_dir "${MEGATRON_ROOT}")"
HF_CHECKPOINT="$(absolute_dir "${HF_CHECKPOINT}")"
LOAD_CHECKPOINT="$(absolute_dir "${LOAD_CHECKPOINT}")"
EVAL_DATA_DIR="$(absolute_dir "${EVAL_DATA_DIR}")"
ANYSEARCH_EVAL_OUTPUT_DIR="$(absolute_path "${ANYSEARCH_EVAL_OUTPUT_DIR}")"
EVAL_CONFIG="$(absolute_file "${EVAL_CONFIG}")"
ANYSEARCH_CONFIG="$(absolute_file "${ANYSEARCH_CONFIG}")"
verify_framework
require_file "${SLIME_ROOT}/train.py"
select_model_args "${BACKBONE}"
export ANYSEARCH_CONFIG EVAL_DATA_DIR ANYSEARCH_EVAL_OUTPUT_DIR MEGATRON_ROOT

EVAL_COMMAND=(
  python3 "${SLIME_ROOT}/train.py"
  --seed "${EVAL_SEED:-42}"
  --rollout-seed "${EVAL_SEED:-42}"
  --actor-num-nodes 1
  --actor-num-gpus-per-node 4
  --rollout-num-gpus 4
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${HF_CHECKPOINT}"
  --load "${LOAD_CHECKPOINT}"
  --no-load-optim
  --no-load-rng
  --prompt-data "${EVAL_DATA_DIR}/nq.parquet"
  --input-key question
  --label-key label
  --metadata-key metadata
  --num-rollout 0
  --rollout-batch-size 512
  --n-samples-per-prompt 1
  --global-batch-size 512
  --seq-length "${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
  --rollout-max-response-len 4096
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --rollout-top-k -1
  --sglang-enable-deterministic-inference
  --eval-interval 1
  --eval-config "${EVAL_CONFIG}"
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-9216}"
  --rollout-num-gpus-per-engine 2
  --sglang-mem-fraction-static 0.6
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --lr-decay-iters 1
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.98
  --rollout-function-path examples.anysearch.slime_ext.rollout.generate_rollout
  --eval-function-path examples.anysearch.slime_ext.rollout.generate_eval_rollout
  --custom-rm-path examples.anysearch.slime_ext.rewards.eval_reward
  --custom-eval-rollout-log-function-path examples.anysearch.metrics.log_eval_rollout_data
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)
if ((${#EXTRA_ARGS[@]})); then
  EVAL_COMMAND+=("${EXTRA_ARGS[@]}")
fi

if ((DRY_RUN)); then
  printf 'runtime environment: [values redacted]\n'
  printf 'AnySearch config: %q\n' "${ANYSEARCH_CONFIG}"
  print_command "${EVAL_COMMAND[@]}"
  exit 0
fi

verify_retriever_health
mkdir -p -- "${ANYSEARCH_EVAL_OUTPUT_DIR}"
ensure_ray_cluster
require_command ray
RUNTIME_ENV_JSON="$(build_runtime_env_json)"
ray job submit \
  --address "${RAY_DASHBOARD_ADDRESS}" \
  --runtime-env-json "${RUNTIME_ENV_JSON}" \
  -- "${EVAL_COMMAND[@]}"
