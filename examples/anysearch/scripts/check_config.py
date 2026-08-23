#!/usr/bin/env python3
"""Validate the canonical AnySearch geometry and evaluation matrix."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

FORK_ROOT = Path(__file__).resolve().parents[3]
if str(FORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FORK_ROOT))

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SLIME_COMMIT = "52fc971bfe4ad7a1e857ac158d626d4b6373474d"
DATASETS = {"nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"}
BUDGETS = set(range(1, 9))
EXPECTED_EXTENSION_PATHS = {
    "data_source_path": "examples.anysearch.slime_ext.data_source.AnySearchDataSource",
    "rollout_function_path": "examples.anysearch.slime_ext.rollout.generate_rollout",
    "eval_function_path": "examples.anysearch.slime_ext.rollout.generate_eval_rollout",
    "per_sample_generate_path": "examples.anysearch.slime_ext.rollout.generate",
    "training_group_reward_path": "examples.anysearch.slime_ext.rewards.group_reward",
    "evaluation_reward_path": "examples.anysearch.slime_ext.rewards.eval_reward",
    "eval_log_path": "examples.anysearch.metrics.log_eval_rollout_data",
}


def load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} must contain a mapping")
    return payload


def check_main_config() -> None:
    config = load_yaml(ROOT / "configs/anysearch.yaml")
    framework = config["framework"]
    training = config["training"]
    curriculum = config["curriculum"]
    assert framework["commit"] == EXPECTED_SLIME_COMMIT
    assert framework["entrypoint"] == "train_async.py"
    assert framework["trainer_gpus"] == framework["rollout_gpus"] == 4
    assert framework["colocate"] is False
    assert framework["trainer_tensor_parallel_size"] == 2
    assert framework["rollout_tensor_parallel_size"] == 2
    assert framework["sglang_mem_fraction_static"] == 0.6

    prompts = training["rollout_batch_size_prompts"]
    group_size = training["group_size"]
    global_batch = training["global_batch_size_trajectories"]
    steps_per_rollout = training["optimizer_steps_per_rollout"]
    assert prompts == 512 and group_size == 5 and global_batch == 512
    assert prompts * group_size == training["trajectories_per_rollout"] == 2560
    assert prompts * group_size == global_batch * steps_per_rollout
    assert training["num_rollouts"] * steps_per_rollout == training["total_optimizer_steps"] == 500

    phase_i = curriculum["phase_i"]
    phase_ii = curriculum["phase_ii"]
    assert (phase_i["num_rollouts"], phase_i["optimizer_steps"]) == (20, 100)
    assert phase_i["schedule"]["budgets"] == [5, 4, 3, 2, 1]
    assert phase_i["schedule"]["rollouts_per_budget"] == 4
    assert (phase_ii["num_rollouts"], phase_ii["optimizer_steps"]) == (80, 400)
    assert phase_i["num_rollouts"] + phase_ii["num_rollouts"] == 100
    assert curriculum["sliding_window_size"] == 20
    assert curriculum["uniform_smoothing_lambda"] == 0.6
    assert config["slime_extensions"] == EXPECTED_EXTENSION_PATHS


def check_eval_matrix() -> None:
    config = load_yaml(ROOT / "configs/eval/budget_b1_b8.yaml")
    entries = config["eval"]["datasets"]
    assert len(entries) == len(DATASETS) * len(BUDGETS) == 56
    observed = set()
    for entry in entries:
        metadata = entry["metadata_overrides"]
        key = (metadata["dataset"], metadata["budget"])
        assert key not in observed
        assert metadata["evaluation"] is True
        assert entry["custom_generate_function_path"] == EXPECTED_EXTENSION_PATHS["per_sample_generate_path"]
        assert entry["path"].endswith(f"/{metadata['dataset']}.parquet")
        observed.add(key)
    assert observed == {(dataset, budget) for dataset in DATASETS for budget in BUDGETS}


def check_launchers() -> None:
    common = (ROOT / "scripts/_common.sh").read_text(encoding="utf-8")
    assert 'BUNDLED_SLIME_ROOT="$(cd -- "${PROJECT_ROOT}/../.."' in common
    assert "${PROJECT_ROOT}/../slime" not in common
    assert "/src/anysearch" not in common
    assert "absolute_file()" in common
    assert "absolute_dir()" in common
    assert "absolute_path()" in common

    train = (ROOT / "scripts/run_train.sh").read_text(encoding="utf-8")
    assert "train_async.py" in train
    assert "--actor-num-gpus-per-node 4" in train
    assert "--rollout-num-gpus 4" in train
    assert "--rollout-batch-size 512" in train
    assert "--n-samples-per-prompt 5" in train
    assert "--global-batch-size 512" in train
    assert "--num-steps-per-rollout 5" in train
    assert '--seq-length "${ROLLOUT_MAX_CONTEXT_LEN:-32768}"' in train
    assert "--use-rollout-logprobs" in train
    assert "--custom-generate-function-path examples.anysearch.slime_ext.rollout.generate" in train
    assert "--rollout-function-path examples.anysearch.slime_ext.rollout.generate_rollout" in train
    assert "--group-rm" in train
    assert "--custom-rm-path examples.anysearch.slime_ext.rewards.group_reward" in train
    assert '--seed "${TRAIN_SEED:-42}"' in train
    assert '--rollout-seed "${TRAIN_SEED:-42}"' in train
    assert "--colocate" not in train
    assert "pkill" not in train
    assert "/root" not in train
    assert 'ANYSEARCH_CONFIG="$(absolute_file "${ANYSEARCH_CONFIG}")"' in train
    assert 'TRAIN_DATA="$(absolute_file "${TRAIN_DATA}")"' in train
    assert 'HF_CHECKPOINT="$(absolute_dir "${HF_CHECKPOINT}")"' in train

    evaluation = (ROOT / "scripts/run_eval.sh").read_text(encoding="utf-8")
    assert '--seed "${EVAL_SEED:-42}"' in evaluation
    assert '--rollout-seed "${EVAL_SEED:-42}"' in evaluation
    assert '--prompt-data "${EVAL_DATA_DIR}/nq.parquet"' in evaluation
    assert '--seq-length "${ROLLOUT_MAX_CONTEXT_LEN:-32768}"' in evaluation
    assert "--sglang-enable-deterministic-inference" in evaluation
    assert "--lr-decay-iters 1" in evaluation
    assert "--no-load-optim" in evaluation
    assert "--no-load-rng" in evaluation
    assert "--custom-config-path" not in evaluation
    assert 'EVAL_CONFIG="$(absolute_file "${EVAL_CONFIG}")"' in evaluation
    assert 'ANYSEARCH_CONFIG="$(absolute_file "${ANYSEARCH_CONFIG}")"' in evaluation
    assert 'EVAL_DATA_DIR="$(absolute_dir "${EVAL_DATA_DIR}")"' in evaluation
    assert "for dataset in nq triviaqa popqa hotpotqa 2wikimultihopqa musique bamboogle" in evaluation


def check_materialized_ablations() -> None:
    from examples.anysearch.config import ExperimentConfig

    matrix = load_yaml(ROOT / "configs/ablations.yaml")
    launcher = ROOT / "scripts/materialize_ablation.sh"
    with tempfile.TemporaryDirectory() as temporary_directory:
        for name in sorted(matrix["ablations"]):
            output = Path(temporary_directory) / f"{name}.yaml"
            subprocess.run(["bash", str(launcher), name, str(output)], check=True, capture_output=True, text=True)
            ExperimentConfig.from_yaml(output)
            materialized = load_yaml(output)
            source = matrix["ablations"][name]
            for field in ("checkpoint",):
                if field in source:
                    assert materialized["ablation"][field] == source[field]


if __name__ == "__main__":
    check_main_config()
    check_eval_matrix()
    check_launchers()
    check_materialized_ablations()
    print("AnySearch configuration is internally consistent.")
