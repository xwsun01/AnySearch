"""Reward callbacks for group training and single-sample evaluation."""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Mapping, Sequence
from typing import Any

from examples.anysearch.config import ExperimentConfig, RewardConfig
from examples.anysearch.curriculum import Phase
from examples.anysearch.rewards import RewardBreakdown, RewardSample, score_group
from slime.utils.types import Sample


def _experiment_config(args: Namespace) -> ExperimentConfig:
    config = getattr(args, "anysearch_experiment_config", None)
    if config is None:
        path = getattr(args, "anysearch_config", None) or os.environ.get("ANYSEARCH_CONFIG")
        config = ExperimentConfig.from_yaml(path) if path else ExperimentConfig()
        args.anysearch_experiment_config = config
    if not isinstance(config, ExperimentConfig):
        raise TypeError("args.anysearch_experiment_config must be an ExperimentConfig")
    return config


def _targets(label: Any) -> str | Sequence[str]:
    if isinstance(label, str) and label.strip():
        return label
    if isinstance(label, Mapping):
        for key in ("target", "answers", "answer"):
            value = label.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes))
                and value
                and all(isinstance(item, str) and item.strip() for item in value)
            ):
                return list(value)
        ground_truth = label.get("ground_truth")
        if isinstance(ground_truth, Mapping):
            return _targets(ground_truth)
    raise ValueError("sample label must contain a non-empty target/answers field")


def _generated_tokens(sample: Sample, metadata: Mapping[str, Any]) -> int:
    if "generated_tokens" in metadata:
        count = metadata["generated_tokens"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("sample.metadata.generated_tokens must be a non-negative integer")
        return count
    if sample.loss_mask is None:
        raise ValueError("AnySearch reward requires generated_tokens metadata or a loss_mask")
    return sum(int(bool(mask)) for mask in sample.loss_mask)


def _reward_sample(sample: Sample) -> RewardSample:
    metadata = sample.metadata if isinstance(sample.metadata, Mapping) else {}
    budget = metadata.get("budget_total", metadata.get("budget"))
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("sample metadata must contain a non-negative integer budget_total")
    used = metadata.get("search_count", metadata.get("budget_used"))
    if used is not None and (isinstance(used, bool) or not isinstance(used, int)):
        raise ValueError("sample metadata search_count must be an integer")
    scaffold = metadata.get("scaffold")
    if scaffold is not None and not isinstance(scaffold, bool):
        raise ValueError("sample metadata scaffold must be a boolean")
    return RewardSample(
        trajectory=sample.response,
        gold_answers=_targets(sample.label),
        budget_total=budget,
        generated_token_count=_generated_tokens(sample, metadata),
        budget_used=used,
        phase=metadata.get("phase", Phase.PHASE_II),
        scaffold=scaffold,
    )


def _write_metadata(sample: Sample, breakdown: RewardBreakdown) -> None:
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    components = breakdown.to_dict()
    sample.metadata.update(
        {
            "answer": breakdown.answer,
            "answer_correct": bool(breakdown.accuracy),
            "accuracy": breakdown.accuracy,
            "reward_total": breakdown.total,
            "reward_components": components,
        }
    )


def _score(samples: Sequence[Sample], config: RewardConfig) -> list[RewardBreakdown]:
    if not samples:
        raise ValueError("reward model cannot score an empty sample group")
    reward_samples = [_reward_sample(sample) for sample in samples]
    breakdowns = score_group(reward_samples, config)
    for sample, breakdown in zip(samples, breakdowns, strict=True):
        _write_metadata(sample, breakdown)
    return breakdowns


async def group_reward(
    args: Namespace,
    samples: Sample | Sequence[Sample],
    **_kwargs: Any,
) -> float | list[float]:
    """Compute group rewards, with single-sample dispatch for evaluation."""

    if isinstance(samples, Sample):
        return await eval_reward(args, samples)
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("group_reward expects a Sample or sequence of Sample objects")
    if not all(isinstance(sample, Sample) for sample in samples):
        raise TypeError("group_reward sequence must contain only Sample objects")
    experiment_config = _experiment_config(args)
    expected_size = experiment_config.generation.group_size
    if len(samples) != expected_size:
        raise ValueError(f"group_reward received {len(samples)} samples; expected group_size={expected_size}")
    group_index = samples[0].group_index
    if group_index is None:
        raise ValueError("group_reward requires every training sample to have a group_index")
    mismatched_indexes = [sample.group_index for sample in samples if sample.group_index != group_index]
    if mismatched_indexes:
        raise ValueError("group_reward samples must all have the same group_index")
    breakdowns = _score(samples, experiment_config.rewards)
    return [float(breakdown.total) for breakdown in breakdowns]


async def eval_reward(args: Namespace, sample: Sample, **_kwargs: Any) -> float:
    """Score one eval trajectory and return EM while retaining all components."""

    if not isinstance(sample, Sample):
        raise TypeError("eval_reward expects one Sample")
    breakdown = _score([sample], _experiment_config(args).rewards)[0]
    # Evaluation uses normalized exact match rather than the composite training reward.
    return float(breakdown.accuracy)
