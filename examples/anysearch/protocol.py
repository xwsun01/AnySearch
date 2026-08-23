"""Strict parser and state-machine validation for AnySearch trajectories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from examples.anysearch.curriculum import Phase


class Tag(str, Enum):
    BUDGET = "budget"
    THINK = "think"
    SEARCH = "search"
    INFORMATION = "information"
    ANSWER = "answer"


_TAG_PATTERN = re.compile(r"<(?P<closing>/)?(?P<name>budget|think|search|information|answer)>")
_ANSWER_PATTERN = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.DOTALL)
_SEARCH_PATTERN = re.compile(r"<search>.*?</search>", re.DOTALL)
_BUDGET_PATTERN = re.compile(
    r"remaining\s*=\s*(?P<remaining>\d+)\s*;\s*" r"used\s*=\s*(?P<used>\d+)\s*;\s*" r"total\s*=\s*(?P<total>\d+)"
)


@dataclass(frozen=True, slots=True)
class TagBlock:
    """One balanced, non-nested protocol block."""

    tag: Tag
    content: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class BudgetState:
    remaining: int
    used: int
    total: int


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Structured format validation result used by the reward function."""

    valid: bool
    reason: str
    blocks: tuple[TagBlock, ...] = ()
    search_count: int = 0
    answer: str | None = None


class ProtocolError(ValueError):
    """Raised when tag blocks cannot be parsed."""


def parse_tag_blocks(text: str) -> tuple[TagBlock, ...]:
    """Parse balanced special tags and reject output outside them.

    The parser recognizes only AnySearch's five protocol tags.  Recognized
    tags may not nest.  Whitespace between blocks is allowed; all non-whitespace
    text must occur inside a valid block.
    """

    if not isinstance(text, str):
        raise TypeError("trajectory must be a string")
    blocks: list[TagBlock] = []
    open_tag: Tag | None = None
    open_start = 0
    content_start = 0
    cursor = 0
    for match in _TAG_PATTERN.finditer(text):
        between = text[cursor : match.start()]
        closing = match.group("closing") is not None
        tag = Tag(match.group("name"))
        if open_tag is None:
            if between.strip():
                raise ProtocolError("non-whitespace content appears outside protocol tags")
            if closing:
                raise ProtocolError(f"closing </{tag.value}> has no matching opening tag")
            open_tag = tag
            open_start = match.start()
            content_start = match.end()
        else:
            if not closing:
                raise ProtocolError(f"nested <{tag.value}> inside <{open_tag.value}> is not allowed")
            if tag is not open_tag:
                raise ProtocolError(f"expected </{open_tag.value}> but found </{tag.value}>")
            blocks.append(
                TagBlock(
                    tag=open_tag,
                    content=text[content_start : match.start()],
                    start=open_start,
                    end=match.end(),
                )
            )
            open_tag = None
        cursor = match.end()
    if open_tag is not None:
        raise ProtocolError(f"unclosed <{open_tag.value}> tag")
    if text[cursor:].strip():
        raise ProtocolError("non-whitespace content appears outside protocol tags")
    return tuple(blocks)


def parse_budget_state(content: str) -> BudgetState:
    """Parse and validate ``remaining + used == total``."""

    if not isinstance(content, str):
        raise TypeError("budget content must be a string")
    match = _BUDGET_PATTERN.fullmatch(content.strip())
    if match is None:
        raise ProtocolError("budget tag must use remaining=R; used=U; total=T")
    state = BudgetState(
        remaining=int(match.group("remaining")),
        used=int(match.group("used")),
        total=int(match.group("total")),
    )
    if state.remaining + state.used != state.total:
        raise ProtocolError("budget state violates remaining + used == total")
    return state


def _coerce_phase(phase: Phase | str) -> Phase:
    if isinstance(phase, Phase):
        parsed = phase
    elif isinstance(phase, str):
        aliases = {
            "phase_i": Phase.PHASE_I,
            "phase_one": Phase.PHASE_I,
            "phase1": Phase.PHASE_I,
            "phase_ii": Phase.PHASE_II,
            "phase_two": Phase.PHASE_II,
            "phase2": Phase.PHASE_II,
        }
        try:
            parsed = aliases[phase.lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported protocol phase: {phase}") from exc
    else:
        raise TypeError("phase must be a Phase or string")
    if parsed is Phase.COMPLETE:
        raise ValueError("complete is not a trajectory protocol phase")
    return parsed


def _validate_expected_budget(expected_total_budget: int | None) -> None:
    if expected_total_budget is None:
        return
    if isinstance(expected_total_budget, bool) or not isinstance(expected_total_budget, int):
        raise TypeError("expected_total_budget must be an integer")
    if expected_total_budget < 0:
        raise ValueError("expected_total_budget must be non-negative")


def _invalid(reason: str, blocks: tuple[TagBlock, ...] = ()) -> ValidationResult:
    return ValidationResult(valid=False, reason=reason, blocks=blocks)


def _validate_phase_one(
    blocks: tuple[TagBlock, ...],
    *,
    expected_total_budget: int | None,
) -> ValidationResult:
    index = 0
    searches_seen = 0
    total_budget = expected_total_budget
    while True:
        if index >= len(blocks) or blocks[index].tag is not Tag.BUDGET:
            return _invalid("each Phase I round must begin with <budget>", blocks)
        try:
            budget_state = parse_budget_state(blocks[index].content)
        except ProtocolError as exc:
            return _invalid(str(exc), blocks)
        if total_budget is None:
            total_budget = budget_state.total
        if budget_state.total != total_budget:
            return _invalid("budget total changed within the trajectory", blocks)
        if budget_state.used != searches_seen or budget_state.remaining != total_budget - searches_seen:
            return _invalid("budget state does not match the number of completed searches", blocks)
        index += 1

        if index >= len(blocks) or blocks[index].tag is not Tag.THINK:
            return _invalid("<budget> must be followed by <think> in Phase I", blocks)
        index += 1
        if index >= len(blocks):
            return _invalid("trajectory must end with <answer>", blocks)

        action = blocks[index]
        if action.tag is Tag.ANSWER:
            if index != len(blocks) - 1:
                return _invalid("<answer> must be the final block", blocks)
            return ValidationResult(
                valid=True,
                reason="valid Phase I trajectory",
                blocks=blocks,
                search_count=searches_seen,
                answer=action.content.strip(),
            )
        if action.tag is not Tag.SEARCH:
            return _invalid("<think> must be followed by <search> or <answer>", blocks)
        if total_budget is not None and searches_seen >= total_budget:
            return _invalid("search attempted after the budget was exhausted", blocks)
        index += 1
        if index >= len(blocks) or blocks[index].tag is not Tag.INFORMATION:
            return _invalid("<search> must be followed by <information>", blocks)
        searches_seen += 1
        index += 1


def _validate_phase_two(
    blocks: tuple[TagBlock, ...],
    *,
    expected_total_budget: int | None,
) -> ValidationResult:
    index = 0
    searches_seen = 0
    while True:
        if index >= len(blocks) or blocks[index].tag is not Tag.THINK:
            return _invalid("each Phase II round must begin with <think>", blocks)
        index += 1
        if index >= len(blocks):
            return _invalid("trajectory must end with <answer>", blocks)

        action = blocks[index]
        if action.tag is Tag.ANSWER:
            if index != len(blocks) - 1:
                return _invalid("<answer> must be the final block", blocks)
            return ValidationResult(
                valid=True,
                reason="valid Phase II trajectory",
                blocks=blocks,
                search_count=searches_seen,
                answer=action.content.strip(),
            )
        if action.tag is not Tag.SEARCH:
            return _invalid("<think> must be followed by <search> or <answer>", blocks)
        if expected_total_budget is not None and searches_seen >= expected_total_budget:
            return _invalid("search attempted after the budget was exhausted", blocks)
        index += 1
        if index >= len(blocks) or blocks[index].tag is not Tag.INFORMATION:
            return _invalid("<search> must be followed by <information>", blocks)
        searches_seen += 1
        index += 1


def validate_trajectory(
    text: str,
    phase: Phase | str,
    *,
    expected_total_budget: int | None = None,
) -> ValidationResult:
    """Validate pairing, ordering, and absence of text outside valid tags."""

    parsed_phase = _coerce_phase(phase)
    _validate_expected_budget(expected_total_budget)
    try:
        blocks = parse_tag_blocks(text)
    except ProtocolError as exc:
        return _invalid(str(exc))
    if parsed_phase is Phase.PHASE_I:
        return _validate_phase_one(blocks, expected_total_budget=expected_total_budget)
    return _validate_phase_two(blocks, expected_total_budget=expected_total_budget)


def extract_final_answer(text: str) -> str | None:
    """Extract the last complete answer block, even from a malformed format."""

    if not isinstance(text, str):
        raise TypeError("trajectory must be a string")
    matches = list(_ANSWER_PATTERN.finditer(text))
    return matches[-1].group("answer").strip() if matches else None


def count_search_calls(text: str) -> int:
    """Count complete search blocks in a trajectory."""

    if not isinstance(text, str):
        raise TypeError("trajectory must be a string")
    return len(_SEARCH_PATTERN.findall(text))
