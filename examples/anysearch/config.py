"""Typed, strictly validated configuration for the AnySearch experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_OPTIONAL = object()


class _OptionalSchema:
    def __init__(self, schema: Mapping[str, object]) -> None:
        self.schema = schema


_CONFIG_SCHEMA: dict[str, object] = {
    "schema_version": None,
    "experiment_name": None,
    "seed": None,
    "framework": {
        "name": None,
        "repository": None,
        "commit": None,
        "entrypoint": None,
        "asynchronous": None,
        "colocate": None,
        "trainer_gpus": None,
        "rollout_gpus": None,
        "trainer_tensor_parallel_size": None,
        "rollout_tensor_parallel_size": None,
        "sglang_mem_fraction_static": None,
    },
    "training": {
        "algorithm": None,
        "num_rollouts": None,
        "rollout_batch_size_prompts": None,
        "group_size": None,
        "trajectories_per_rollout": None,
        "global_batch_size_trajectories": None,
        "optimizer_steps_per_rollout": None,
        "total_optimizer_steps": None,
        "max_response_tokens": None,
        "max_context_tokens": None,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "shuffle": None,
        "balance_data": None,
        "learning_rate": None,
        "optimizer": None,
        "adam_beta1": None,
        "adam_beta2": None,
        "weight_decay": None,
        "lr_schedule": None,
        "kl_loss": None,
        "kl_coefficient": None,
        "kl_type": None,
        "clip_ratio_low": None,
        "clip_ratio_high": None,
        "advantage_epsilon": None,
        "entropy_coefficient": None,
        "attention_dropout": None,
        "hidden_dropout": None,
        "gradient_checkpointing": None,
        "sequence_parallel": None,
        "dynamic_batching": None,
        "max_tokens_per_gpu": None,
    },
    "curriculum": {
        "budget_min": None,
        "budget_max": None,
        "sliding_window_size": None,
        "uniform_smoothing_lambda": None,
        "smoothing_epsilon": None,
        "window_unit": None,
        "budget_assignment_unit": None,
        "phase_i": {
            "start_rollout": None,
            "num_rollouts": None,
            "optimizer_steps": None,
            "scaffold": None,
            "schedule": {
                "type": None,
                "budgets": None,
                "rollouts_per_budget": None,
                "optimizer_steps_per_budget": None,
            },
        },
        "phase_ii": {
            "start_rollout": None,
            "num_rollouts": None,
            "optimizer_steps": None,
            "scaffold": None,
            "schedule": {"type": None, "cold_start": None, "budgets": None},
        },
    },
    "reward": {
        "formula": None,
        "accuracy": {"enabled": None, "weight": None, "metric": None, "accepted_answers_key": None},
        "format": {
            "enabled": None,
            "weight": None,
            "require_paired_tags": None,
            "require_valid_order": None,
            "forbid_text_outside_tags": None,
            "phase_i_tags": None,
            "phase_ii_tags": None,
        },
        "length": {
            "enabled": None,
            "weight": None,
            "measured_tokens": None,
            "limit": None,
            "tolerance": None,
            "minimum_reward": None,
        },
        "tool": {
            "enabled": None,
            "gamma_max": None,
            "stability_xi": None,
            "correctness_gated": None,
            "absolute_component": None,
            "relative_component": None,
            "zero_correct_group_reward": None,
            "equal_correct_search_counts_relative_reward": None,
            "gamma_mode": None,
            "gamma_fixed": _OPTIONAL,
        },
    },
    "interaction": {
        "search_tag": None,
        "information_tag": None,
        "think_tag": None,
        "answer_tag": None,
        "budget_tag": None,
        "reject_search_when_exhausted": None,
        "environment_tokens_loss_mask": None,
        "model_tokens_loss_mask": None,
        "phase_i_inject_budget_each_turn": None,
        "phase_ii_total_budget_in_initial_prompt_only": None,
        "phase_ii_inject_budget_each_turn": None,
        "inference_scaffold": None,
        "invalid_action_max_retries": _OPTIONAL,
    },
    "retrieval": {
        "backend": None,
        "model": None,
        "corpus": None,
        "approximate_search": None,
        "top_k": None,
        "faiss_gpu": None,
        "endpoint_env": None,
        "default_endpoint": None,
    },
    "data": {
        "training_datasets": None,
        "evaluation_datasets": None,
        "input_key": None,
        "label_key": None,
        "metadata_key": None,
        "evaluation_budgets": None,
        "evaluation_samples_per_prompt": None,
        "primary_metric": None,
        "efficiency_metric": None,
    },
    "slime_extensions": {
        "data_source_path": None,
        "rollout_function_path": None,
        "eval_function_path": None,
        "per_sample_generate_path": None,
        "training_group_reward_path": None,
        "evaluation_reward_path": None,
        "eval_log_path": None,
    },
    "ablation": _OptionalSchema(
        {
            "name": None,
            "description": None,
            "evaluation_only": None,
            "checkpoint": _OPTIONAL,
        }
    ),
}


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _section_values(
    value: object,
    *,
    name: str,
    allowed: frozenset[str],
) -> dict[str, Any]:
    mapping = _require_mapping(value, name=name)
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} option(s): {', '.join(unknown)}")
    return dict(mapping)


def _validate_config_schema(value: object, schema: Mapping[str, object], *, path: str) -> Mapping[str, Any]:
    mapping = _require_mapping(value, name=path)
    unknown = sorted(set(mapping) - set(schema))
    if unknown:
        raise ValueError(f"unknown {path} option(s): {', '.join(unknown)}")
    missing = sorted(
        key
        for key, nested_schema in schema.items()
        if nested_schema is not _OPTIONAL and not isinstance(nested_schema, _OptionalSchema) and key not in mapping
    )
    if missing:
        raise ValueError(f"missing {path} option(s): {', '.join(missing)}")
    for key, nested_schema in schema.items():
        if isinstance(nested_schema, Mapping):
            _validate_config_schema(mapping[key], nested_schema, path=f"{path}.{key}")
        elif isinstance(nested_schema, _OptionalSchema) and key in mapping:
            _validate_config_schema(mapping[key], nested_schema.schema, path=f"{path}.{key}")
    return mapping


def _nested_mapping(value: Mapping[str, Any], key: str, *, path: str) -> Mapping[str, Any]:
    return _require_mapping(value[key], name=f"{path}.{key}")


def _require_equal(actual: object, expected: object, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}, got {actual!r}")


def _require_int(value: object, *, name: str, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _require_finite_number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        below_minimum = numeric < minimum if minimum_inclusive else numeric <= minimum
        if below_minimum:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{name} must be {operator} {minimum}")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    """Optimizer and GRPO settings used by AnySearch."""

    learning_rate: float = 1.0e-6
    beta1: float = 0.9
    beta2: float = 0.98
    weight_decay: float = 0.01
    kl_penalty: float = 0.001
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    advantage_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        _require_finite_number(
            self.learning_rate, name="optimization.learning_rate", minimum=0.0, minimum_inclusive=False
        )
        _require_finite_number(self.beta1, name="optimization.beta1", minimum=0.0, maximum=1.0)
        _require_finite_number(self.beta2, name="optimization.beta2", minimum=0.0, maximum=1.0)
        _require_finite_number(self.weight_decay, name="optimization.weight_decay", minimum=0.0)
        _require_finite_number(self.kl_penalty, name="optimization.kl_penalty", minimum=0.0)
        _require_finite_number(self.clip_ratio_low, name="optimization.clip_ratio_low", minimum=0.0)
        _require_finite_number(self.clip_ratio_high, name="optimization.clip_ratio_high", minimum=0.0)
        _require_finite_number(
            self.advantage_epsilon,
            name="optimization.advantage_epsilon",
            minimum=0.0,
            minimum_inclusive=False,
        )
        if float(self.clip_ratio_low) > float(self.clip_ratio_high):
            raise ValueError("optimization.clip_ratio_low must not exceed clip_ratio_high")

    @classmethod
    def from_mapping(cls, value: object) -> OptimizationConfig:
        values = _section_values(
            value,
            name="optimization",
            allowed=frozenset(
                {
                    "learning_rate",
                    "beta1",
                    "beta2",
                    "weight_decay",
                    "kl_penalty",
                    "clip_ratio_low",
                    "clip_ratio_high",
                    "advantage_epsilon",
                }
            ),
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Generation and batch geometry for one asynchronous rollout."""

    group_size: int = 5
    global_batch_size: int = 512
    rollout_query_batch_size: int = 512
    max_response_length: int = 4096
    max_context_length: int = 32768
    temperature: float = 1.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        _require_int(self.group_size, name="generation.group_size", minimum=1)
        _require_int(self.global_batch_size, name="generation.global_batch_size", minimum=1)
        _require_int(self.rollout_query_batch_size, name="generation.rollout_query_batch_size", minimum=1)
        _require_int(self.max_response_length, name="generation.max_response_length", minimum=1)
        _require_int(self.max_context_length, name="generation.max_context_length", minimum=1)
        _require_finite_number(self.temperature, name="generation.temperature", minimum=0.0, minimum_inclusive=False)
        _require_finite_number(self.top_p, name="generation.top_p", minimum=0.0, maximum=1.0, minimum_inclusive=False)
        if self.trajectories_per_rollout % self.global_batch_size:
            raise ValueError("generation.rollout_query_batch_size * group_size must be divisible by global_batch_size")
        if self.max_context_length < self.max_response_length:
            raise ValueError("generation.max_context_length must be >= max_response_length")

    @property
    def trajectories_per_rollout(self) -> int:
        """Number of sampled trajectories produced by one rollout."""

        return self.rollout_query_batch_size * self.group_size

    @property
    def optimizer_steps_per_rollout(self) -> int:
        """Optimizer steps needed to consume one rollout without dropping data."""

        return self.trajectories_per_rollout // self.global_batch_size

    @classmethod
    def from_mapping(cls, value: object) -> GenerationConfig:
        values = _section_values(
            value,
            name="generation",
            allowed=frozenset(
                {
                    "group_size",
                    "global_batch_size",
                    "rollout_query_batch_size",
                    "max_response_length",
                    "max_context_length",
                    "temperature",
                    "top_p",
                }
            ),
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Composite reward coefficients and DAPO length-penalty thresholds."""

    accuracy_weight: float = 0.5
    format_weight: float = 0.15
    length_weight: float = 0.05
    max_tool_weight: float = 0.3
    relative_epsilon: float = 1.0e-6
    length_limit: int = 2048
    length_tolerance: int = 1024
    accuracy_enabled: bool = True
    format_enabled: bool = True
    length_enabled: bool = True
    tool_enabled: bool = True
    absolute_component: bool = True
    relative_component: bool = True
    gamma_mode: str = "adaptive"
    gamma_fixed: float = 0.3

    def __post_init__(self) -> None:
        for field_name in ("accuracy_weight", "format_weight", "length_weight", "max_tool_weight"):
            _require_finite_number(getattr(self, field_name), name=f"rewards.{field_name}", minimum=0.0)
        _require_finite_number(
            self.relative_epsilon,
            name="rewards.relative_epsilon",
            minimum=0.0,
            minimum_inclusive=False,
        )
        _require_int(self.length_limit, name="rewards.length_limit", minimum=0)
        _require_int(self.length_tolerance, name="rewards.length_tolerance", minimum=1)
        for field_name in ("accuracy_enabled", "format_enabled", "length_enabled", "tool_enabled"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"rewards.{field_name} must be a boolean")
        for field_name in ("absolute_component", "relative_component"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"rewards.{field_name} must be a boolean")
        if self.gamma_mode not in {"adaptive", "fixed"}:
            raise ValueError("rewards.gamma_mode must be 'adaptive' or 'fixed'")
        _require_finite_number(self.gamma_fixed, name="rewards.gamma_fixed", minimum=0.0)

    @classmethod
    def from_mapping(cls, value: object) -> RewardConfig:
        values = _section_values(
            value,
            name="rewards",
            allowed=frozenset(
                {
                    "accuracy_weight",
                    "format_weight",
                    "length_weight",
                    "max_tool_weight",
                    "relative_epsilon",
                    "length_limit",
                    "length_tolerance",
                    "accuracy_enabled",
                    "format_enabled",
                    "length_enabled",
                    "tool_enabled",
                    "absolute_component",
                    "relative_component",
                    "gamma_mode",
                    "gamma_fixed",
                }
            ),
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    """Two-phase budget curriculum and runtime step mapping."""

    phase_one_optimizer_steps: int = 100
    phase_two_optimizer_steps: int = 400
    optimizer_steps_per_rollout: int = 5
    max_budget: int = 5
    sliding_window_size: int = 20
    uniform_mixing: float = 0.6
    sampling_epsilon: float = 1.0e-6
    phase_one_scaffold: bool = True
    phase_two_scaffold: bool = False

    def __post_init__(self) -> None:
        _require_int(self.phase_one_optimizer_steps, name="curriculum.phase_one_optimizer_steps", minimum=0)
        _require_int(self.phase_two_optimizer_steps, name="curriculum.phase_two_optimizer_steps", minimum=0)
        _require_int(self.optimizer_steps_per_rollout, name="curriculum.optimizer_steps_per_rollout", minimum=1)
        _require_int(self.max_budget, name="curriculum.max_budget", minimum=1)
        _require_int(self.sliding_window_size, name="curriculum.sliding_window_size", minimum=1)
        _require_finite_number(self.uniform_mixing, name="curriculum.uniform_mixing", minimum=0.0, maximum=1.0)
        _require_finite_number(
            self.sampling_epsilon,
            name="curriculum.sampling_epsilon",
            minimum=0.0,
            minimum_inclusive=False,
        )
        if self.phase_one_optimizer_steps + self.phase_two_optimizer_steps == 0:
            raise ValueError("at least one curriculum phase must contain optimizer steps")
        if not isinstance(self.phase_one_scaffold, bool):
            raise TypeError("curriculum.phase_one_scaffold must be a boolean")
        if not isinstance(self.phase_two_scaffold, bool):
            raise TypeError("curriculum.phase_two_scaffold must be a boolean")
        if self.phase_one_optimizer_steps % self.max_budget:
            raise ValueError("curriculum.phase_one_optimizer_steps must divide evenly across max_budget levels")
        if self.optimizer_steps_per_budget % self.optimizer_steps_per_rollout:
            raise ValueError("each Phase I budget level must contain a whole number of rollouts")
        if self.phase_two_optimizer_steps % self.optimizer_steps_per_rollout:
            raise ValueError("curriculum.phase_two_optimizer_steps must contain a whole number of rollouts")

    @property
    def optimizer_steps_per_budget(self) -> int:
        """Phase I optimizer steps assigned to each budget level."""

        return self.phase_one_optimizer_steps // self.max_budget

    @property
    def phase_one_rollouts(self) -> int:
        return self.phase_one_optimizer_steps // self.optimizer_steps_per_rollout

    @property
    def phase_two_rollouts(self) -> int:
        return self.phase_two_optimizer_steps // self.optimizer_steps_per_rollout

    @property
    def total_optimizer_steps(self) -> int:
        return self.phase_one_optimizer_steps + self.phase_two_optimizer_steps

    @property
    def total_rollouts(self) -> int:
        return self.phase_one_rollouts + self.phase_two_rollouts

    @classmethod
    def from_mapping(cls, value: object) -> CurriculumConfig:
        values = _section_values(
            value,
            name="curriculum",
            allowed=frozenset(
                {
                    "phase_one_optimizer_steps",
                    "phase_two_optimizer_steps",
                    "optimizer_steps_per_rollout",
                    "max_budget",
                    "sliding_window_size",
                    "uniform_mixing",
                    "sampling_epsilon",
                    "phase_one_scaffold",
                    "phase_two_scaffold",
                }
            ),
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete AnySearch configuration with strict nested YAML loading."""

    seed: int = 42
    inference_scaffold: bool = False
    invalid_action_max_retries: int = 2
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    def __post_init__(self) -> None:
        _require_int(self.seed, name="seed", minimum=0)
        if not isinstance(self.inference_scaffold, bool):
            raise TypeError("inference_scaffold must be a boolean")
        _require_int(self.invalid_action_max_retries, name="invalid_action_max_retries", minimum=0)
        if not isinstance(self.optimization, OptimizationConfig):
            raise TypeError("optimization must be an OptimizationConfig")
        if not isinstance(self.generation, GenerationConfig):
            raise TypeError("generation must be a GenerationConfig")
        if not isinstance(self.rewards, RewardConfig):
            raise TypeError("rewards must be a RewardConfig")
        if not isinstance(self.curriculum, CurriculumConfig):
            raise TypeError("curriculum must be a CurriculumConfig")
        if self.generation.optimizer_steps_per_rollout != self.curriculum.optimizer_steps_per_rollout:
            raise ValueError(
                "generation batch geometry must match curriculum.optimizer_steps_per_rollout "
                f"({self.generation.optimizer_steps_per_rollout} != "
                f"{self.curriculum.optimizer_steps_per_rollout})"
            )

    @classmethod
    def from_mapping(cls, value: object) -> ExperimentConfig:
        mapping = _require_mapping(value, name="configuration")
        if "schema_version" in mapping:
            return cls._from_canonical_mapping(mapping)
        unknown = sorted(
            set(mapping)
            - {
                "seed",
                "inference_scaffold",
                "invalid_action_max_retries",
                "optimization",
                "generation",
                "rewards",
                "curriculum",
            }
        )
        if unknown:
            raise ValueError(f"unknown top-level option(s): {', '.join(unknown)}")
        return cls(
            seed=mapping.get("seed", 42),
            inference_scaffold=mapping.get("inference_scaffold", False),
            invalid_action_max_retries=mapping.get("invalid_action_max_retries", 2),
            optimization=OptimizationConfig.from_mapping(mapping.get("optimization", {})),
            generation=GenerationConfig.from_mapping(mapping.get("generation", {})),
            rewards=RewardConfig.from_mapping(mapping.get("rewards", {})),
            curriculum=CurriculumConfig.from_mapping(mapping.get("curriculum", {})),
        )

    @classmethod
    def _from_canonical_mapping(cls, value: object) -> ExperimentConfig:
        """Map the repository's canonical YAML to core types."""

        root = _validate_config_schema(value, _CONFIG_SCHEMA, path="configuration")
        _require_equal(root["schema_version"], 1, name="schema_version")
        training = _nested_mapping(root, "training", path="configuration")
        curriculum_section = _nested_mapping(root, "curriculum", path="configuration")
        phase_one = _nested_mapping(curriculum_section, "phase_i", path="configuration.curriculum")
        phase_two = _nested_mapping(curriculum_section, "phase_ii", path="configuration.curriculum")
        phase_one_schedule = _nested_mapping(phase_one, "schedule", path="configuration.curriculum.phase_i")
        phase_two_schedule = _nested_mapping(phase_two, "schedule", path="configuration.curriculum.phase_ii")
        reward = _nested_mapping(root, "reward", path="configuration")
        accuracy_reward = _nested_mapping(reward, "accuracy", path="configuration.reward")
        format_reward = _nested_mapping(reward, "format", path="configuration.reward")
        length_reward_config = _nested_mapping(reward, "length", path="configuration.reward")
        tool_reward = _nested_mapping(reward, "tool", path="configuration.reward")
        interaction = _nested_mapping(root, "interaction", path="configuration")
        retrieval = _nested_mapping(root, "retrieval", path="configuration")
        framework = _nested_mapping(root, "framework", path="configuration")

        optimization = OptimizationConfig(
            learning_rate=training["learning_rate"],
            beta1=training["adam_beta1"],
            beta2=training["adam_beta2"],
            weight_decay=training["weight_decay"],
            kl_penalty=training["kl_coefficient"],
            clip_ratio_low=training["clip_ratio_low"],
            clip_ratio_high=training["clip_ratio_high"],
            advantage_epsilon=training["advantage_epsilon"],
        )
        generation = GenerationConfig(
            group_size=training["group_size"],
            global_batch_size=training["global_batch_size_trajectories"],
            rollout_query_batch_size=training["rollout_batch_size_prompts"],
            max_response_length=training["max_response_tokens"],
            max_context_length=training["max_context_tokens"],
            temperature=training["temperature"],
            top_p=training["top_p"],
        )
        rewards = RewardConfig(
            accuracy_weight=accuracy_reward["weight"],
            format_weight=format_reward["weight"],
            length_weight=length_reward_config["weight"],
            max_tool_weight=tool_reward["gamma_max"],
            relative_epsilon=tool_reward["stability_xi"],
            length_limit=length_reward_config["limit"],
            length_tolerance=length_reward_config["tolerance"],
            accuracy_enabled=accuracy_reward["enabled"],
            format_enabled=format_reward["enabled"],
            length_enabled=length_reward_config["enabled"],
            tool_enabled=tool_reward["enabled"],
            absolute_component=tool_reward["absolute_component"],
            relative_component=tool_reward["relative_component"],
            gamma_mode=(
                "adaptive" if tool_reward["gamma_mode"] == "adaptive_group_accuracy" else tool_reward["gamma_mode"]
            ),
            gamma_fixed=tool_reward.get("gamma_fixed", tool_reward["gamma_max"]),
        )
        curriculum = CurriculumConfig(
            phase_one_optimizer_steps=phase_one["optimizer_steps"],
            phase_two_optimizer_steps=phase_two["optimizer_steps"],
            optimizer_steps_per_rollout=training["optimizer_steps_per_rollout"],
            max_budget=curriculum_section["budget_max"],
            sliding_window_size=curriculum_section["sliding_window_size"],
            uniform_mixing=curriculum_section["uniform_smoothing_lambda"],
            sampling_epsilon=curriculum_section["smoothing_epsilon"],
            phase_one_scaffold=phase_one["scaffold"],
            phase_two_scaffold=phase_two["scaffold"],
        )
        parsed = cls(
            seed=root["seed"],
            inference_scaffold=interaction["inference_scaffold"],
            invalid_action_max_retries=interaction.get("invalid_action_max_retries", 2),
            optimization=optimization,
            generation=generation,
            rewards=rewards,
            curriculum=curriculum,
        )

        # Validate redundant documentation fields so configuration drift fails
        # before launching an expensive distributed job.
        _require_equal(training["algorithm"], "grpo", name="training.algorithm")
        _require_equal(
            training["trajectories_per_rollout"],
            generation.trajectories_per_rollout,
            name="training.trajectories_per_rollout",
        )
        _require_equal(
            training["total_optimizer_steps"],
            curriculum.total_optimizer_steps,
            name="training.total_optimizer_steps",
        )
        _require_equal(training["num_rollouts"], curriculum.total_rollouts, name="training.num_rollouts")
        _require_equal(curriculum_section["budget_min"], 1, name="curriculum.budget_min")
        _require_equal(curriculum_section["window_unit"], "trajectory", name="curriculum.window_unit")
        _require_equal(
            curriculum_section["budget_assignment_unit"],
            "prompt_group",
            name="curriculum.budget_assignment_unit",
        )
        _require_equal(phase_one["start_rollout"], 0, name="curriculum.phase_i.start_rollout")
        _require_equal(
            phase_one["num_rollouts"], curriculum.phase_one_rollouts, name="curriculum.phase_i.num_rollouts"
        )
        _require_equal(
            phase_one_schedule["type"],
            "descending_blocks",
            name="curriculum.phase_i.schedule.type",
        )
        _require_equal(
            phase_one_schedule["budgets"],
            list(range(curriculum.max_budget, 0, -1)),
            name="curriculum.phase_i.schedule.budgets",
        )
        if curriculum.phase_one_optimizer_steps:
            _require_equal(
                phase_one_schedule["optimizer_steps_per_budget"],
                curriculum.optimizer_steps_per_budget,
                name="curriculum.phase_i.schedule.optimizer_steps_per_budget",
            )
            _require_equal(
                phase_one_schedule["rollouts_per_budget"],
                curriculum.optimizer_steps_per_budget // curriculum.optimizer_steps_per_rollout,
                name="curriculum.phase_i.schedule.rollouts_per_budget",
            )
        if curriculum.phase_two_optimizer_steps:
            _require_equal(
                phase_two["start_rollout"],
                curriculum.phase_one_rollouts,
                name="curriculum.phase_ii.start_rollout",
            )
        _require_equal(
            phase_two["num_rollouts"], curriculum.phase_two_rollouts, name="curriculum.phase_ii.num_rollouts"
        )
        _require_equal(
            phase_two_schedule["type"],
            "adaptive_accuracy_gap",
            name="curriculum.phase_ii.schedule.type",
        )
        _require_equal(phase_two_schedule["cold_start"], "uniform", name="curriculum.phase_ii.schedule.cold_start")
        _require_equal(
            phase_two_schedule["budgets"],
            list(range(1, curriculum.max_budget + 1)),
            name="curriculum.phase_ii.schedule.budgets",
        )
        _require_equal(
            length_reward_config["measured_tokens"],
            "model_generated_only",
            name="reward.length.measured_tokens",
        )
        _require_equal(tool_reward["zero_correct_group_reward"], 0.0, name="reward.tool.zero_correct_group_reward")
        _require_equal(
            tool_reward["equal_correct_search_counts_relative_reward"],
            1.0,
            name="reward.tool.equal_correct_search_counts_relative_reward",
        )
        _require_equal(
            interaction["environment_tokens_loss_mask"],
            0,
            name="interaction.environment_tokens_loss_mask",
        )
        _require_equal(interaction["model_tokens_loss_mask"], 1, name="interaction.model_tokens_loss_mask")
        if not isinstance(retrieval["faiss_gpu"], bool):
            raise TypeError("retrieval.faiss_gpu must be a boolean")
        if retrieval["faiss_gpu"]:
            _require_int(framework["rollout_gpus"], name="framework.rollout_gpus", minimum=1)
        return parsed

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load a YAML file, rejecting unknown sections and options."""

        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - declared package dependency
            raise RuntimeError("YAML configuration requires PyYAML") from exc

        config_path = Path(path)
        with config_path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if payload is None:
            payload = {}
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable representation."""

        return asdict(self)
