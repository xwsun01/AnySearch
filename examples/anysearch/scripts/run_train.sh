#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=scripts/_common.sh
source "${SCRIPT_DIR}/_common.sh"

BACKBONE="${ANYSEARCH_BACKBONE:-qwen2.5-7b-instruct}"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/run_train.sh [--model NAME] [--config PATH] [--dry-run] [-- TRAINER_ARGS...]

Required environment: HF_CHECKPOINT, REF_LOAD, TRAIN_DATA, MEGATRON_ROOT.
Optional environment: SAVE_CHECKPOINT, LOAD_CHECKPOINT, START_RAY, USE_WANDB,
SAVE_INTERVAL, MAX_TOKENS_PER_GPU, ANYSEARCH_RETRIEVAL_URL,
ANYSEARCH_RETRIEVAL_HEALTH_TIMEOUT.
EOF
}

while (($#)); do
  case "$1" in
    --model)
      (($# >= 2)) || die "--model requires a value"
      BACKBONE="$2"
      shift 2
      ;;
    --config)
      (($# >= 2)) || die "--config requires a value"
      ANYSEARCH_CONFIG="$2"
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
      die "unknown launcher argument '$1'; pass additional trainer arguments after --"
      ;;
  esac
done

initialize_paths
require_command python3
require_value MEGATRON_ROOT
require_value HF_CHECKPOINT
require_value REF_LOAD
require_value TRAIN_DATA
require_dir "${MEGATRON_ROOT}"
require_dir "${HF_CHECKPOINT}"
require_dir "${REF_LOAD}"
require_file "${TRAIN_DATA}"
require_file "${ANYSEARCH_CONFIG}"
MEGATRON_ROOT="$(absolute_dir "${MEGATRON_ROOT}")"
HF_CHECKPOINT="$(absolute_dir "${HF_CHECKPOINT}")"
REF_LOAD="$(absolute_dir "${REF_LOAD}")"
TRAIN_DATA="$(absolute_file "${TRAIN_DATA}")"
ANYSEARCH_CONFIG="$(absolute_file "${ANYSEARCH_CONFIG}")"
export ANYSEARCH_CONFIG MEGATRON_ROOT
verify_framework
select_model_args "${BACKBONE}"

CKPT_ARGS=(--hf-checkpoint "${HF_CHECKPOINT}" --ref-load "${REF_LOAD}")
if [[ -n "${LOAD_CHECKPOINT:-}" ]]; then
  require_dir "${LOAD_CHECKPOINT}"
  LOAD_CHECKPOINT="$(absolute_dir "${LOAD_CHECKPOINT}")"
  CKPT_ARGS+=(--load "${LOAD_CHECKPOINT}")
fi
if [[ -n "${SAVE_CHECKPOINT:-}" ]]; then
  SAVE_CHECKPOINT="$(absolute_path "${SAVE_CHECKPOINT}")"
  CKPT_ARGS+=(--save "${SAVE_CHECKPOINT}" --save-interval "${SAVE_INTERVAL:-10}")
fi

TRACKING_ARGS=()
if [[ "${USE_WANDB:-0}" == "1" ]]; then
  require_value WANDB_PROJECT
  TRACKING_ARGS+=(--use-wandb --wandb-project "${WANDB_PROJECT}" --wandb-group "${WANDB_GROUP:-${BACKBONE}}")
fi

ROLLOUT_ARGS=(
  --prompt-data "${TRAIN_DATA}"
  --input-key question
  --label-key label
  --metadata-key metadata
  --rollout-shuffle
  --num-rollout 100
  --rollout-batch-size 512
  --n-samples-per-prompt 5
  --global-batch-size 512
  --num-steps-per-rollout 5
  --seq-length "${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
  --rollout-max-response-len 4096
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --rollout-top-k -1
  --balance-data
)

PERFORMANCE_ARGS=(
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
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.001
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28
  --use-rollout-logprobs
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.98
)

EXTENSION_ARGS=(
  --data-source-path examples.anysearch.slime_ext.data_source.AnySearchDataSource
  --rollout-function-path examples.anysearch.slime_ext.rollout.generate_rollout
  --eval-function-path examples.anysearch.slime_ext.rollout.generate_eval_rollout
  --custom-generate-function-path examples.anysearch.slime_ext.rollout.generate
  --group-rm
  --custom-rm-path examples.anysearch.slime_ext.rewards.group_reward
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

TRAIN_COMMAND=(
  python3 "${SLIME_ROOT}/train_async.py"
  --seed "${TRAIN_SEED:-42}"
  --rollout-seed "${TRAIN_SEED:-42}"
  --actor-num-nodes 1
  --actor-num-gpus-per-node 4
  --rollout-num-gpus 4
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${GRPO_ARGS[@]}"
  "${PERFORMANCE_ARGS[@]}"
  "${EXTENSION_ARGS[@]}"
  "${MISC_ARGS[@]}"
)
if ((${#TRACKING_ARGS[@]})); then
  TRAIN_COMMAND+=("${TRACKING_ARGS[@]}")
fi
if ((${#EXTRA_ARGS[@]})); then
  TRAIN_COMMAND+=("${EXTRA_ARGS[@]}")
fi

if ((DRY_RUN)); then
  printf 'runtime environment: [values redacted]\n'
  printf 'AnySearch config: %q\n' "${ANYSEARCH_CONFIG}"
  print_command "${TRAIN_COMMAND[@]}"
  exit 0
fi

verify_retriever_health
ensure_ray_cluster
require_command ray
RUNTIME_ENV_JSON="$(build_runtime_env_json)"
ray job submit \
  --address "${RAY_DASHBOARD_ADDRESS}" \
  --runtime-env-json "${RUNTIME_ENV_JSON}" \
  -- "${TRAIN_COMMAND[@]}"
