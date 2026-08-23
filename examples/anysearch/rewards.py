"""Group reward computation for AnySearch."""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from examples.anysearch.config import RewardConfig
from examples.anysearch.curriculum import Phase
from examples.anysearch.protocol import count_search_calls, extract_final_answer, validate_trajectory

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCTUATION_TRANSLATION = str.maketrans("", "", string.punctuation)


def normalize_answer(answer: str) -> str:
    """Apply the standard open-domain QA exact-match normalization."""

    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    lowered = answer.lower()
    without_punctuation = lowered.translate(_PUNCTUATION_TRANSLATION)
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match(prediction: str, gold_answers: str | Sequence[str]) -> int:
    """Return one when a prediction equals any normalized gold alias."""

    if not isinstance(prediction, str):
        raise TypeError("prediction must be a string")
    if isinstance(gold_answers, str):
        answers = (gold_answers,)
    elif isinstance(gold_answers, Sequence) and not isinstance(gold_answers, (str, bytes)):
        answers = tuple(gold_answers)
    else:
        raise TypeError("gold_answers must be a string or sequence of strings")
    if not answers or not all(isinstance(answer, str) and answer.strip() for answer in answers):
        raise ValueError("gold_answers must contain at least one non-empty string")
    normalized_prediction = normalize_answer(prediction)
    return int(any(normalized_prediction == normalize_answer(answer) for answer in answers))


def length_reward(generated_token_count: int, *, limit: int = 2048, tolerance: int = 1024) -> float:
    """Compute the DAPO piecewise penalty from generated tokens only.

    Retrieved documents and environment-injected budget/information tokens must
    not be included in ``generated_token_count``.
    """

    for value, name, minimum in (
        (generated_token_count, "generated_token_count", 0),
        (limit, "limit", 0),
        (tolerance, "tolerance", 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
    if generated_token_count <= limit:
        return 0.0
    if generated_token_count <= limit + tolerance:
        return (limit - generated_token_count) / tolerance
    return -1.0


@dataclass(frozen=True, slots=True)
class RewardSample:
    """Inputs needed to score one trajectory inside a GRPO query-group."""

    trajectory: str
    gold_answers: str | Sequence[str]
    budget_total: int
    generated_token_count: int
    budget_used: int | None = None
    phase: Phase | str = Phase.PHASE_II
    scaffold: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, str):
            raise TypeError("trajectory must be a string")
        if isinstance(self.budget_total, bool) or not isinstance(self.budget_total, int):
            raise TypeError("budget_total must be an integer")
        if self.budget_total < 0:
            raise ValueError("budget_total must be non-negative")
        if isinstance(self.generated_token_count, bool) or not isinstance(self.generated_token_count, int):
            raise TypeError("generated_token_count must be an integer")
        if self.generated_token_count < 0:
            raise ValueError("generated_token_count must be non-negative")
        if self.budget_used is not None:
            if isinstance(self.budget_used, bool) or not isinstance(self.budget_used, int):
                raise TypeError("budget_used must be an integer")
            if not 0 <= self.budget_used <= self.budget_total:
                raise ValueError("budget_used must be between zero and budget_total")
        if self.scaffold is not None and not isinstance(self.scaffold, bool):
            raise TypeError("scaffold must be a boolean or None")
        # Validate aliases now so a bad label cannot silently turn into a zero.
        exact_match("", self.gold_answers)

    @property
    def resolved_budget_used(self) -> int:
        used = count_search_calls(self.trajectory) if self.budget_used is None else self.budget_used
        if used > self.budget_total:
            raise ValueError("trajectory uses more searches than its assigned budget")
        return used


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Composite reward and every unweighted component for one trajectory."""

    answer: str | None
    accuracy: int
    format: int
    length: float
    tool_absolute: float
    tool_relative: float
    tool: float
    tool_weight: float
    total: float

    @property
    def total_reward(self) -> float:
        """Verbose alias convenient for logging integrations."""

        return self.total

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tool_components(
    accuracies: Sequence[int],
    used_budgets: Sequence[int],
    *,
    total_budget: int,
    epsilon: float,
) -> tuple[list[float], list[float], list[float]]:
    correct_used = [used for accuracy, used in zip(accuracies, used_budgets, strict=True) if accuracy]
    if not correct_used:
        zeros = [0.0] * len(accuracies)
        return zeros.copy(), zeros.copy(), zeros

    minimum_correct_used = min(correct_used)
    maximum_correct_used = max(correct_used)
    absolute: list[float] = []
    relative: list[float] = []
    tool: list[float] = []
    for accuracy, used in zip(accuracies, used_budgets, strict=True):
        if not accuracy:
            absolute_value = 0.0
            relative_value = 0.0
        else:
            # B=0 has no resource that can be saved, so its efficiency reward
            # is defined as zero rather than dividing by zero.
            absolute_value = (total_budget - used) / total_budget if total_budget else 0.0
            if maximum_correct_used == minimum_correct_used:
                relative_value = 1.0
            else:
                relative_value = 1.0 - (
                    (used - minimum_correct_used) / (maximum_correct_used - minimum_correct_used + epsilon)
                )
        absolute.append(absolute_value)
        relative.append(relative_value)
        tool.append(absolute_value * relative_value)
    return absolute, relative, tool


def score_group(
    samples: Sequence[RewardSample],
    config: RewardConfig | None = None,
) -> list[RewardBreakdown]:
    """Score a query-group using the adaptive efficiency weight.

    All trajectories in a GRPO group must share the budget sampled for their
    query.  The adaptive tool coefficient is ``gamma_max * group_accuracy``.
    """

    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence of RewardSample objects")
    if not samples:
        raise ValueError("cannot score an empty query-group")
    if not all(isinstance(sample, RewardSample) for sample in samples):
        raise TypeError("samples must contain only RewardSample objects")
    reward_config = config or RewardConfig()
    if not isinstance(reward_config, RewardConfig):
        raise TypeError("config must be a RewardConfig")

    total_budgets = {sample.budget_total for sample in samples}
    if len(total_budgets) != 1:
        raise ValueError("all trajectories in a query-group must share one total budget")
    total_budget = next(iter(total_budgets))
    used_budgets = [sample.resolved_budget_used for sample in samples]
    answers = [extract_final_answer(sample.trajectory) for sample in samples]
    accuracies = [
        exact_match(answer, sample.gold_answers) if answer is not None else 0
        for answer, sample in zip(answers, samples, strict=True)
    ]
    format_scores: list[int] = []
    for sample, used in zip(samples, used_budgets, strict=True):
        format_phase: Phase | str
        if sample.scaffold is None:
            format_phase = sample.phase
        else:
            format_phase = Phase.PHASE_I if sample.scaffold else Phase.PHASE_II
        validation = validate_trajectory(
            sample.trajectory,
            format_phase,
            expected_total_budget=sample.budget_total,
        )
        format_scores.append(int(validation.valid and validation.search_count == used))
    length_scores = [
        length_reward(
            sample.generated_token_count,
            limit=reward_config.length_limit,
            tolerance=reward_config.length_tolerance,
        )
        for sample in samples
    ]
    absolute_scores, relative_scores, full_tool_scores = _tool_components(
        accuracies,
        used_budgets,
        total_budget=total_budget,
        epsilon=float(reward_config.relative_epsilon),
    )
    if reward_config.absolute_component and reward_config.relative_component:
        tool_scores = full_tool_scores
    elif reward_config.absolute_component:
        tool_scores = absolute_scores
    elif reward_config.relative_component:
        tool_scores = relative_scores
    else:
        tool_scores = [0.0] * len(samples)
    group_accuracy = sum(accuracies) / len(accuracies)
    if not reward_config.tool_enabled:
        tool_weight = 0.0
    elif reward_config.gamma_mode == "adaptive":
        tool_weight = float(reward_config.max_tool_weight) * group_accuracy
    else:
        tool_weight = float(reward_config.gamma_fixed)

    breakdowns: list[RewardBreakdown] = []
    for answer, accuracy, format_score, length_score, absolute, relative, tool in zip(
        answers,
        accuracies,
        format_scores,
        length_scores,
        absolute_scores,
        relative_scores,
        tool_scores,
        strict=True,
    ):
        total = (
            float(reward_config.accuracy_weight) * accuracy * int(reward_config.accuracy_enabled)
            + float(reward_config.format_weight) * format_score * int(reward_config.format_enabled)
            + float(reward_config.length_weight) * length_score * int(reward_config.length_enabled)
            + tool_weight * tool
        )
        breakdowns.append(
            RewardBreakdown(
                answer=answer,
                accuracy=accuracy,
                format=format_score,
                length=length_score,
                tool_absolute=absolute,
                tool_relative=relative,
                tool=tool,
                tool_weight=tool_weight,
                total=total,
            )
        )
    return breakdowns
