"""Evaluation metrics and result logging for AnySearch."""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any


@dataclass(frozen=True)
class EvaluationMetrics:
    dataset: str
    budget: int
    count: int
    correct: int
    exact_match: float
    searches: int
    tool_productivity: float | None
    average_searches: float
    average_generated_tokens: float
    average_retrieval_tokens: float
    average_total_tokens: float
    truncated: int
    failed: int


def _metadata(sample: Any) -> Mapping[str, Any]:
    value = getattr(sample, "metadata", None)
    return value if isinstance(value, Mapping) else {}


def _reward_accuracy(sample: Any) -> int:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, Mapping):
        for key in ("accuracy", "r_acc", "answer_correct"):
            if key in reward:
                return int(bool(reward[key]))
    metadata = _metadata(sample)
    for key in ("answer_correct", "accuracy"):
        if key in metadata:
            return int(bool(metadata[key]))
    return 0


def sample_to_result(sample: Any, *, dataset: str | None = None, budget: int | None = None) -> dict[str, Any]:
    metadata = _metadata(sample)
    loss_mask = getattr(sample, "loss_mask", None)
    generated_tokens = int(metadata.get("generated_tokens", sum(loss_mask) if loss_mask else 0))
    retrieval_tokens = int(metadata.get("retrieval_tokens", 0))
    assigned_budget = int(metadata.get("budget_total", metadata.get("budget", budget or 0)))
    status = getattr(sample, "status", None)
    status_value = getattr(status, "value", str(status or "unknown"))
    return {
        "dataset": str(metadata.get("dataset", dataset or "unknown")),
        "budget": assigned_budget,
        "correct": _reward_accuracy(sample),
        "searches": int(metadata.get("search_count", metadata.get("budget_used", 0))),
        "generated_tokens": generated_tokens,
        "retrieval_tokens": retrieval_tokens,
        "total_tokens": generated_tokens + retrieval_tokens,
        "status": status_value,
        "answer": metadata.get("answer"),
        "index": getattr(sample, "index", None),
    }


def aggregate_results(results: Iterable[Mapping[str, Any]], *, dataset: str, budget: int) -> EvaluationMetrics:
    rows = list(results)
    if not rows:
        raise ValueError("cannot aggregate an empty evaluation")
    correct = sum(int(row["correct"]) for row in rows)
    searches = sum(int(row["searches"]) for row in rows)
    tool_productivity = correct / searches if searches else None
    return EvaluationMetrics(
        dataset=dataset,
        budget=budget,
        count=len(rows),
        correct=correct,
        exact_match=correct / len(rows),
        searches=searches,
        tool_productivity=tool_productivity,
        average_searches=mean(int(row["searches"]) for row in rows),
        average_generated_tokens=mean(int(row["generated_tokens"]) for row in rows),
        average_retrieval_tokens=mean(int(row["retrieval_tokens"]) for row in rows),
        average_total_tokens=mean(int(row["total_tokens"]) for row in rows),
        truncated=sum(row.get("status") == "truncated" for row in rows),
        failed=sum(row.get("status") in {"failed", "aborted"} for row in rows),
    )


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def log_eval_rollout_data(rollout_id: int, args: Any, data: Mapping[str, Any], extra_metrics: Any) -> bool:
    """Persist per-sample predictions and aggregate metrics as JSON.

    Returning ``False`` keeps the framework's default metric logging enabled.
    Set ``ANYSEARCH_EVAL_OUTPUT_DIR`` to enable file output.
    """

    output_dir = os.environ.get("ANYSEARCH_EVAL_OUTPUT_DIR")
    if not output_dir:
        return False
    seed = int(getattr(args, "rollout_seed", 0))
    for configured_name, info in data.items():
        samples = info.get("samples", []) if isinstance(info, Mapping) else []
        if not samples:
            continue
        first_metadata = _metadata(samples[0])
        dataset = str(first_metadata.get("dataset", configured_name.rsplit("_b", 1)[0]))
        budget = int(first_metadata.get("budget", first_metadata.get("budget_total", 0)))
        results = [sample_to_result(sample, dataset=dataset, budget=budget) for sample in samples]
        summary = asdict(aggregate_results(results, dataset=dataset, budget=budget))
        payload = {
            "schema_version": 1,
            "rollout_id": rollout_id,
            "seed": seed,
            "summary": summary,
            "samples": results,
            "framework_extra_metrics": extra_metrics,
        }
        filename = f"{dataset}_budget-{budget}_seed-{seed}.json"
        _atomic_json(payload, Path(output_dir) / filename)
    return False


def summarize_runs(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Aggregate independent evaluation JSON files into mean/std rows."""

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        summary = payload["summary"]
        grouped[(summary["dataset"], int(summary["budget"]))].append(summary)

    metric_names = (
        "exact_match",
        "tool_productivity",
        "average_searches",
        "average_generated_tokens",
        "average_retrieval_tokens",
        "average_total_tokens",
    )
    output: list[dict[str, Any]] = []
    for (dataset, budget), summaries in sorted(grouped.items()):
        row: dict[str, Any] = {"dataset": dataset, "budget": budget, "runs": len(summaries)}
        for metric_name in metric_names:
            values = [float(item[metric_name]) for item in summaries if item.get(metric_name) is not None]
            row[f"{metric_name}_mean"] = mean(values) if values else None
            row[f"{metric_name}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else None
        output.append(row)
    return output
