"""Curriculum-aware rollout data source with checkpointable state."""

from __future__ import annotations

import copy
import logging
import os
import tempfile
from argparse import Namespace
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from examples.anysearch.config import ExperimentConfig
from examples.anysearch.curriculum import AnySearchCurriculum, Phase
from examples.anysearch.prompts import build_phase_one_prompt, build_phase_two_prompt
from slime.rollout.data_source import RolloutDataSourceWithBuffer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _load_experiment_config(args: Namespace) -> ExperimentConfig:
    injected = getattr(args, "anysearch_experiment_config", None)
    if injected is not None:
        if not isinstance(injected, ExperimentConfig):
            raise TypeError("args.anysearch_experiment_config must be an ExperimentConfig")
        return injected
    path = getattr(args, "anysearch_config", None) or os.environ.get("ANYSEARCH_CONFIG")
    return ExperimentConfig.from_yaml(path) if path else ExperimentConfig()


def _question_from_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        question = prompt.strip()
    elif isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)):
        question = ""
        for message in reversed(prompt):
            if isinstance(message, Mapping) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    question = content.strip()
                    break
    else:
        question = ""
    if not question:
        raise ValueError("rollout sample does not contain a non-empty question")
    return question


class AnySearchDataSource(RolloutDataSourceWithBuffer):
    """Assign one curriculum budget per query-group and persist its state."""

    def __init__(self, args: Namespace) -> None:
        if not getattr(args, "rollout_global_dataset", False):
            raise ValueError("AnySearchDataSource requires --rollout-global-dataset")
        if getattr(args, "partial_rollout", False):
            raise ValueError("AnySearch requires complete trajectories; partial_rollout is unsupported")
        if getattr(args, "apply_chat_template", False):
            raise ValueError(
                "omit --apply-chat-template for AnySearch: the full budget prompt is built after data loading "
                "and custom generation applies the model chat template"
            )
        super().__init__(args)
        self.experiment_config = _load_experiment_config(args)
        # Evaluation reuses the configuration validated during worker setup.
        self.args.anysearch_experiment_config = self.experiment_config
        curriculum_seed = getattr(args, "rollout_seed", None)
        if curriculum_seed is None:
            curriculum_seed = self.experiment_config.seed
        if isinstance(curriculum_seed, bool) or not isinstance(curriculum_seed, int) or curriculum_seed < 0:
            raise ValueError("args.rollout_seed must be a non-negative integer")
        self.curriculum = AnySearchCurriculum(
            self.experiment_config.curriculum,
            seed=curriculum_seed,
        )
        self.current_rollout_id: int | None = None
        self.current_optimizer_step: int | None = None
        self.current_phase: Phase | None = None
        self._last_recorded_rollout_id: int | None = None
        self._rollout_snapshots: dict[int, dict[str, Any]] = {}
        self._validate_training_geometry()

    def _validate_training_geometry(self) -> None:
        expected = self.experiment_config.generation
        observed = {
            "n_samples_per_prompt": (getattr(self.args, "n_samples_per_prompt", None), expected.group_size),
            "global_batch_size": (getattr(self.args, "global_batch_size", None), expected.global_batch_size),
            "rollout_batch_size": (
                getattr(self.args, "rollout_batch_size", None),
                expected.rollout_query_batch_size,
            ),
            "rollout_max_response_len": (
                getattr(self.args, "rollout_max_response_len", None),
                expected.max_response_length,
            ),
            "rollout_max_context_len": (
                getattr(self.args, "rollout_max_context_len", None),
                expected.max_context_length,
            ),
            "seq_length": (getattr(self.args, "seq_length", None), expected.max_context_length),
            "num_steps_per_rollout": (
                getattr(self.args, "num_steps_per_rollout", None),
                expected.optimizer_steps_per_rollout,
            ),
        }
        mismatches = [
            f"{name}={actual} (expected {wanted})" for name, (actual, wanted) in observed.items() if actual != wanted
        ]
        if mismatches:
            raise ValueError("training arguments do not match AnySearch config: " + ", ".join(mismatches))

    def set_rollout(self, rollout_id: int, *, evaluation: bool = False) -> None:
        """Bind the full-rollout id to the corresponding optimizer step/phase."""

        if isinstance(rollout_id, bool) or not isinstance(rollout_id, int) or rollout_id < 0:
            raise ValueError("rollout_id must be a non-negative integer")
        if evaluation:
            # Evaluation has explicit per-sample budgets and never advances the
            # training curriculum.
            return
        optimizer_step = rollout_id * self.experiment_config.curriculum.optimizer_steps_per_rollout
        if optimizer_step >= self.experiment_config.curriculum.total_optimizer_steps:
            raise RuntimeError(
                f"rollout {rollout_id} starts at optimizer step {optimizer_step}, beyond the configured curriculum"
            )
        if self.curriculum.optimizer_step != optimizer_step:
            raise RuntimeError(
                "curriculum/checkpoint mismatch: "
                f"state is at optimizer step {self.curriculum.optimizer_step}, "
                f"but rollout_id {rollout_id} requires step {optimizer_step}"
            )
        self.current_rollout_id = rollout_id
        self.current_optimizer_step = optimizer_step
        self.current_phase = self.curriculum.phase_for_optimizer_step(optimizer_step)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        if self.current_optimizer_step is None or self.current_phase is None or self.current_rollout_id is None:
            raise RuntimeError("generate_rollout must call data_source.set_rollout before requesting samples")
        groups = super().get_samples(num_samples)
        for group in groups:
            self._prepare_group(group)
        return groups

    def _prepare_group(self, group: list[Sample]) -> None:
        if len(group) != self.experiment_config.generation.group_size:
            raise RuntimeError(
                f"query-group has {len(group)} trajectories; expected {self.experiment_config.generation.group_size}"
            )
        assert self.current_optimizer_step is not None
        assert self.current_phase is not None
        assert self.current_rollout_id is not None
        scaffold = self.curriculum.scaffold_for_optimizer_step(self.current_optimizer_step)
        already_prepared = [
            bool(isinstance(sample.metadata, Mapping) and sample.metadata.get("anysearch_prompt_built"))
            for sample in group
        ]
        if any(already_prepared):
            if not all(already_prepared):
                raise RuntimeError("a query-group cannot mix fresh and buffered AnySearch trajectories")
            budgets = {int(sample.metadata.get("budget_total", -1)) for sample in group}
            if len(budgets) != 1:
                raise RuntimeError("buffered trajectories in a query-group must retain one shared budget")
            return
        budget = self.curriculum.sample_group_budget(self.current_optimizer_step)
        for sample in group:
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            question = _question_from_prompt(sample.prompt)
            if scaffold:
                sample.prompt = build_phase_one_prompt(question, budget)
            elif self.current_phase in {Phase.PHASE_I, Phase.PHASE_II}:
                sample.prompt = build_phase_two_prompt(question, budget)
            else:  # pragma: no cover - guarded in set_rollout
                raise RuntimeError("cannot draw samples after the curriculum is complete")
            metadata.update(
                {
                    "question": question,
                    "phase": self.current_phase.value,
                    "rollout_id": self.current_rollout_id,
                    "optimizer_step": self.current_optimizer_step,
                    "scaffold": scaffold,
                    "budget": budget,
                    "budget_total": budget,
                    "budget_used": 0,
                    "search_count": 0,
                    "invalid_action_max_retries": self.experiment_config.invalid_action_max_retries,
                    "anysearch_prompt_built": True,
                }
            )
            sample.metadata = metadata

    def record_rollout(self, groups: Sequence[Sequence[Sample]]) -> None:
        """Update W=20 per-budget windows once for a completed rollout."""

        if self.current_rollout_id is None:
            raise RuntimeError("cannot record a rollout before set_rollout")
        if self._last_recorded_rollout_id == self.current_rollout_id:
            raise RuntimeError(f"rollout {self.current_rollout_id} was already recorded")
        for group in groups:
            if not group:
                continue
            budgets = {int(sample.metadata.get("budget_total", -1)) for sample in group}
            if len(budgets) != 1:
                raise RuntimeError("all trajectories in a query-group must share one budget")
            budget = budgets.pop()
            accuracies = [self._sample_accuracy(sample) for sample in group]
            self.curriculum.record_trajectories(budget, accuracies)
            for sample in group:
                sample.metadata["curriculum_recorded"] = True
        self.curriculum.advance_rollout()
        self._last_recorded_rollout_id = self.current_rollout_id
        # Freeze the exact post-rollout state before asynchronous generation can
        # advance the live curriculum.
        self._rollout_snapshots[self.current_rollout_id] = self._capture_state()
        if self._is_model_checkpoint_rollout(self.current_rollout_id):
            # Store curriculum state before the model checkpoint becomes
            # discoverable, keeping resume state crash-consistent.
            self._write_snapshot(self.current_rollout_id)

    def _is_model_checkpoint_rollout(self, rollout_id: int) -> bool:
        if getattr(self.args, "save", None) is None:
            return False
        interval = getattr(self.args, "save_interval", None)
        if interval is None:
            return False
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("args.save_interval must be a positive integer")
        num_rollout = getattr(self.args, "num_rollout", None)
        is_final = isinstance(num_rollout, int) and not isinstance(num_rollout, bool) and rollout_id == num_rollout - 1
        return (rollout_id + 1) % interval == 0 or is_final

    def _snapshot_destination(self, rollout_id: int) -> Path:
        save_root = getattr(self.args, "save", None)
        if not isinstance(save_root, (str, os.PathLike)):
            raise RuntimeError("AnySearch data-source checkpointing requires args.save")
        return Path(save_root) / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"

    def _write_snapshot(self, rollout_id: int) -> None:
        if rollout_id not in self._rollout_snapshots:
            raise RuntimeError(
                f"no frozen AnySearch snapshot for rollout {rollout_id}; record_rollout must complete before save"
            )
        destination = self._snapshot_destination(rollout_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = self._rollout_snapshots[rollout_id]
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name, delete=False) as handle:
                temporary_path = handle.name
            torch.save(state, temporary_path)
            os.replace(temporary_path, destination)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @staticmethod
    def _sample_accuracy(sample: Sample) -> int:
        metadata = sample.metadata if isinstance(sample.metadata, Mapping) else {}
        if "answer_correct" in metadata:
            return int(bool(metadata["answer_correct"]))
        reward = sample.reward
        if isinstance(reward, Mapping):
            for key in ("accuracy", "r_acc", "answer_correct"):
                if key in reward:
                    return int(bool(reward[key]))
        raise RuntimeError("reward model must write answer_correct metadata before curriculum recording")

    def save(self, rollout_id: int) -> None:
        if not self.args.rollout_global_dataset:
            return
        self._write_snapshot(rollout_id)
        # Checkpoints are monotonic; snapshots older than the committed one can
        # no longer be useful and may otherwise accumulate.
        for snapshot_id in [key for key in self._rollout_snapshots if key <= rollout_id]:
            del self._rollout_snapshots[snapshot_id]

    def _capture_state(self) -> dict[str, Any]:
        return {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": copy.deepcopy(self.metadata),
            "buffer": copy.deepcopy(self.buffer),
            "anysearch_curriculum": self.curriculum.state_dict(),
            "anysearch_last_recorded_rollout_id": self._last_recorded_rollout_id,
        }

    def load(self, rollout_id: int | None = None) -> None:
        if not self.args.rollout_global_dataset or self.args.load is None:
            return
        source = Path(self.args.load) / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
        if not source.exists():
            if isinstance(rollout_id, int) and rollout_id >= 0:
                raise RuntimeError(
                    f"incomplete checkpoint: model state resumes after rollout {rollout_id}, "
                    f"but AnySearch data-source state is missing at {source}; "
                    "restore the matching file or roll back to the previous complete checkpoint"
                )
            logger.info("Checkpoint %s does not exist.", source)
            return
        logger.info("Loading AnySearch data-source state from %s", source)
        state = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(state, Mapping):
            raise RuntimeError(f"invalid data-source checkpoint at {source}")
        if "anysearch_curriculum" not in state:
            raise RuntimeError(f"checkpoint {source} predates AnySearch curriculum persistence")
        self.sample_offset = int(state.get("sample_offset", 0))
        self.epoch_id = int(state.get("epoch_id", 0))
        self.sample_group_index = int(state.get("sample_group_index", 0))
        self.sample_index = int(state.get("sample_index", 0))
        self.metadata = dict(state.get("metadata", {}))
        self.buffer = list(state.get("buffer", []))
        self.curriculum.load_state_dict(state["anysearch_curriculum"])
        recorded = state.get("anysearch_last_recorded_rollout_id")
        self._last_recorded_rollout_id = int(recorded) if recorded is not None else None
        self.current_rollout_id = None
        self.current_optimizer_step = None
        self.current_phase = None
        self._rollout_snapshots = {}
        if self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)
