"""Two-phase budget curriculum from *One Policy, Any Budget*."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from examples.anysearch.config import CurriculumConfig


class Phase(str, Enum):
    """Training phase at an optimizer step."""

    PHASE_I = "phase_i"
    PHASE_II = "phase_ii"
    COMPLETE = "complete"


def _jsonify_random_state(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonify_random_state(item) for item in value]
    return value


def _tupleify_random_state(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tupleify_random_state(item) for item in value)
    return value


def _validate_optimizer_step(step: object) -> int:
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("optimizer_step must be an integer")
    if step < 0:
        raise ValueError("optimizer_step must be non-negative")
    return step


@dataclass(frozen=True, slots=True)
class CurriculumState:
    """Portable state required for an exact curriculum resume."""

    optimizer_step: int
    accuracy_windows: dict[int, tuple[int, ...]]
    random_state: object

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable state dictionary."""

        return {
            "version": 1,
            "optimizer_step": self.optimizer_step,
            "accuracy_windows": {
                str(budget): list(observations) for budget, observations in sorted(self.accuracy_windows.items())
            },
            "random_state": _jsonify_random_state(self.random_state),
        }

    @classmethod
    def from_dict(cls, value: object) -> CurriculumState:
        """Parse a state dictionary without depending on a live curriculum."""

        if not isinstance(value, Mapping):
            raise TypeError("curriculum state must be a mapping")
        if not all(isinstance(key, str) for key in value):
            raise TypeError("curriculum state keys must be strings")
        unknown = sorted(set(value) - {"version", "optimizer_step", "accuracy_windows", "random_state"})
        if unknown:
            raise ValueError(f"unknown curriculum state field(s): {', '.join(map(str, unknown))}")
        if value.get("version") != 1:
            raise ValueError("unsupported curriculum state version")
        optimizer_step = _validate_optimizer_step(value.get("optimizer_step"))
        raw_windows = value.get("accuracy_windows")
        if not isinstance(raw_windows, Mapping):
            raise TypeError("accuracy_windows must be a mapping")
        windows: dict[int, tuple[int, ...]] = {}
        for raw_budget, raw_observations in raw_windows.items():
            if isinstance(raw_budget, bool):
                raise TypeError("accuracy window keys must be integer budgets")
            if isinstance(raw_budget, int):
                budget = raw_budget
            elif isinstance(raw_budget, str) and raw_budget.isdecimal():
                budget = int(raw_budget)
            else:
                raise TypeError("accuracy window keys must be integer budgets")
            if not isinstance(raw_observations, Sequence) or isinstance(raw_observations, (str, bytes)):
                raise TypeError(f"accuracy window for budget {budget} must be a sequence")
            observations: list[int] = []
            for observation in raw_observations:
                if isinstance(observation, bool):
                    observations.append(int(observation))
                elif isinstance(observation, int) and observation in (0, 1):
                    observations.append(observation)
                else:
                    raise ValueError("accuracy observations must be binary")
            if budget in windows:
                raise ValueError(f"duplicate accuracy window for budget {budget}")
            windows[budget] = tuple(observations)
        if "random_state" not in value:
            raise ValueError("curriculum state is missing random_state")
        return cls(
            optimizer_step=optimizer_step,
            accuracy_windows=windows,
            random_state=_tupleify_random_state(value["random_state"]),
        )


class AnySearchCurriculum:
    """Stateful Phase I annealing and Phase II adaptive sampling.

    A call to :meth:`sample_group_budget` returns exactly one budget for one
    query-group.  The caller must reuse that value for all trajectories in the
    GRPO group.  Accuracy windows are updated through
    :meth:`record_trajectories`, which stores each trajectory independently.
    """

    def __init__(
        self,
        config: CurriculumConfig | None = None,
        *,
        seed: int = 42,
        state: CurriculumState | Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config or CurriculumConfig()
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self._random = random.Random(seed)
        self._optimizer_step = 0
        self._windows: dict[int, deque[int]] = {
            budget: deque(maxlen=self.config.sliding_window_size) for budget in range(1, self.config.max_budget + 1)
        }
        if state is not None:
            parsed_state = state if isinstance(state, CurriculumState) else CurriculumState.from_dict(state)
            self.load_state(parsed_state)

    @property
    def optimizer_step(self) -> int:
        return self._optimizer_step

    @property
    def phase(self) -> Phase:
        return self.phase_for_optimizer_step(self._optimizer_step)

    @staticmethod
    def _phase_for_step(step: int, config: CurriculumConfig) -> Phase:
        if step < config.phase_one_optimizer_steps:
            return Phase.PHASE_I
        if step < config.total_optimizer_steps:
            return Phase.PHASE_II
        return Phase.COMPLETE

    def phase_for_optimizer_step(self, optimizer_step: int) -> Phase:
        """Resolve a zero-based optimizer step to its training phase."""

        step = _validate_optimizer_step(optimizer_step)
        return self._phase_for_step(step, self.config)

    def budget_for_optimizer_step(self, optimizer_step: int) -> int:
        """Return the deterministic Phase I budget for an optimizer step."""

        step = _validate_optimizer_step(optimizer_step)
        if self.phase_for_optimizer_step(step) is not Phase.PHASE_I:
            raise ValueError("a deterministic annealed budget exists only in Phase I")
        level_index = step // self.config.optimizer_steps_per_budget
        return self.config.max_budget - level_index

    def scaffold_for_optimizer_step(self, optimizer_step: int) -> bool:
        """Return whether the configured phase uses the reasoning scaffold."""

        phase = self.phase_for_optimizer_step(optimizer_step)
        if phase is Phase.PHASE_I:
            return self.config.phase_one_scaffold
        if phase is Phase.PHASE_II:
            return self.config.phase_two_scaffold
        raise ValueError("the curriculum is complete")

    def accuracy(self, budget: int) -> float:
        """Return the current per-budget trajectory accuracy.

        An unseen budget has accuracy zero. Under normal AnySearch execution,
        Phase I populates every window before Phase II begins.
        """

        self._validate_budget(budget)
        window = self._windows[budget]
        return sum(window) / len(window) if window else 0.0

    def sampling_probabilities(self, optimizer_step: int | None = None) -> dict[int, float]:
        """Compute Equation 5 for a Phase II rollout.

        The first Phase II rollout is forced to uniform sampling, matching the
        reported step-100 distribution.  Subsequent rollouts use trajectory-
        level sliding-window accuracies, including observations from Phase I.
        """

        step = self._optimizer_step if optimizer_step is None else _validate_optimizer_step(optimizer_step)
        if self.phase_for_optimizer_step(step) is not Phase.PHASE_II:
            raise ValueError("adaptive probabilities are defined only in Phase II")
        budgets = tuple(range(1, self.config.max_budget + 1))
        uniform_probability = 1.0 / self.config.max_budget
        first_phase_two_rollout_ends = self.config.phase_one_optimizer_steps + self.config.optimizer_steps_per_rollout
        if step < first_phase_two_rollout_ends:
            return {budget: uniform_probability for budget in budgets}

        accuracies = {budget: self.accuracy(budget) for budget in budgets}
        maximum_accuracy = max(accuracies.values())
        difficulty = {
            budget: maximum_accuracy - accuracies[budget] + float(self.config.sampling_epsilon) for budget in budgets
        }
        difficulty_total = sum(difficulty.values())
        adaptive_weight = 1.0 - float(self.config.uniform_mixing)
        uniform_weight = float(self.config.uniform_mixing)
        probabilities = {
            budget: adaptive_weight * difficulty[budget] / difficulty_total + uniform_weight * uniform_probability
            for budget in budgets
        }
        # Remove harmless floating-point drift while preserving Equation 5.
        residual = 1.0 - sum(probabilities.values())
        probabilities[budgets[-1]] += residual
        return probabilities

    def sample_group_budget(self, optimizer_step: int | None = None) -> int:
        """Sample one shared budget for one query-group."""

        step = self._optimizer_step if optimizer_step is None else _validate_optimizer_step(optimizer_step)
        phase = self.phase_for_optimizer_step(step)
        if phase is Phase.PHASE_I:
            return self.budget_for_optimizer_step(step)
        if phase is Phase.COMPLETE:
            raise RuntimeError("the configured curriculum is complete")
        probabilities = self.sampling_probabilities(step)
        budgets = list(probabilities)
        return self._random.choices(budgets, weights=[probabilities[budget] for budget in budgets], k=1)[0]

    def record_trajectories(self, budget: int, accuracies: Sequence[int | bool]) -> None:
        """Append individual trajectory outcomes to a budget's size-W window."""

        self._validate_budget(budget)
        if isinstance(accuracies, (str, bytes)) or not isinstance(accuracies, Sequence):
            raise TypeError("accuracies must be a sequence")
        observations: list[int] = []
        for accuracy in accuracies:
            if isinstance(accuracy, bool):
                observations.append(int(accuracy))
            elif isinstance(accuracy, int) and accuracy in (0, 1):
                observations.append(accuracy)
            else:
                raise ValueError("accuracies must contain only binary outcomes")
        self._windows[budget].extend(observations)

    def advance_rollout(self) -> int:
        """Advance by the configured five optimizer steps and return the new step."""

        if self.phase is Phase.COMPLETE:
            raise RuntimeError("the configured curriculum is already complete")
        next_step = self._optimizer_step + self.config.optimizer_steps_per_rollout
        self._optimizer_step = min(next_step, self.config.total_optimizer_steps)
        return self._optimizer_step

    def state(self) -> CurriculumState:
        """Capture windows, progress, and RNG state for deterministic resume."""

        return CurriculumState(
            optimizer_step=self._optimizer_step,
            accuracy_windows={budget: tuple(window) for budget, window in self._windows.items()},
            random_state=self._random.getstate(),
        )

    def state_dict(self) -> dict[str, Any]:
        return self.state().to_dict()

    def load_state(self, state: CurriculumState) -> None:
        """Restore a state after validating it against this configuration."""

        if state.optimizer_step > self.config.total_optimizer_steps:
            raise ValueError("state optimizer_step exceeds configured training length")
        if state.optimizer_step % self.config.optimizer_steps_per_rollout:
            raise ValueError("state optimizer_step must lie on a rollout boundary")
        expected_budgets = set(range(1, self.config.max_budget + 1))
        if set(state.accuracy_windows) != expected_budgets:
            raise ValueError("state accuracy windows do not match configured budget levels")
        restored_windows: dict[int, deque[int]] = {}
        for budget, observations in state.accuracy_windows.items():
            if len(observations) > self.config.sliding_window_size:
                raise ValueError(f"state window for budget {budget} exceeds configured window size")
            if any(observation not in (0, 1) for observation in observations):
                raise ValueError("state accuracy observations must be binary")
            restored_windows[budget] = deque(observations, maxlen=self.config.sliding_window_size)
        try:
            self._random.setstate(state.random_state)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid random_state in curriculum state") from exc
        self._windows = restored_windows
        self._optimizer_step = state.optimizer_step

    def load_state_dict(self, value: object) -> None:
        self.load_state(CurriculumState.from_dict(value))

    @classmethod
    def from_state_dict(
        cls,
        value: object,
        config: CurriculumConfig | None = None,
        *,
        seed: int = 42,
    ) -> AnySearchCurriculum:
        """Construct a curriculum directly from serialized state."""

        return cls(config, seed=seed, state=CurriculumState.from_dict(value))

    def _validate_budget(self, budget: object) -> None:
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError("budget must be an integer")
        if budget not in self._windows:
            raise ValueError(f"budget must be in [1, {self.config.max_budget}]")
