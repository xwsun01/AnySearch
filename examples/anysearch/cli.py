"""Command-line utilities for data preparation and evaluation setup."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from examples.anysearch.data import convert_file, merge_parquet_files, prepare_flashrag_dataset
from examples.anysearch.metrics import summarize_runs

DEFAULT_DATASETS = ("nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle")


def build_eval_config(
    *,
    data_dir: Path,
    budgets: list[int],
    datasets: list[str],
    max_response_len: int = 4096,
) -> dict[str, Any]:
    if any(budget < 1 for budget in budgets):
        raise ValueError("evaluation budgets must be positive")
    entries = []
    for dataset in datasets:
        path = data_dir / f"{dataset}.parquet"
        for budget in budgets:
            entries.append(
                {
                    "name": f"{dataset}_b{budget}",
                    "path": str(path.resolve()),
                    "input_key": "question",
                    "label_key": "label",
                    "metadata_key": "metadata",
                    "n_samples_per_eval_prompt": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_response_len": max_response_len,
                    "custom_generate_function_path": "examples.anysearch.slime_ext.rollout.generate",
                    "metadata_overrides": {"budget": budget, "dataset": dataset, "evaluation": True},
                }
            )
    return {"eval": {"defaults": {}, "datasets": entries}}


def _write_yaml(payload: MappingLike, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


MappingLike = dict[str, Any]


def _prepare(args: argparse.Namespace) -> None:
    if args.flashrag:
        count = prepare_flashrag_dataset(args.dataset, args.split, args.output)
    elif args.source:
        count = convert_file(args.source, args.output, dataset=args.dataset, split=args.split)
    else:  # pragma: no cover - protected by argparse validation in main
        raise ValueError("one data source is required")
    print(json.dumps({"output": str(Path(args.output).resolve()), "records": count}))


def _eval_config(args: argparse.Namespace) -> None:
    payload = build_eval_config(
        data_dir=Path(args.data_dir),
        budgets=args.budgets,
        datasets=args.datasets,
        max_response_len=args.max_response_len,
    )
    _write_yaml(payload, Path(args.output))


def _merge_parquet(args: argparse.Namespace) -> None:
    count = merge_parquet_files(args.sources, args.output, seed=args.seed)
    print(json.dumps({"output": str(Path(args.output).resolve()), "records": count}))


def _summarize(args: argparse.Namespace) -> None:
    paths = sorted(Path(args.results_dir).glob("*_budget-*_seed-*.json"))
    rows = summarize_runs(paths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".csv":
        fieldnames = list(rows[0]) if rows else ["dataset", "budget", "runs"]
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anysearch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="convert public QA data to AnySearch Parquet")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--source")
    source.add_argument("--flashrag", action="store_true")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--split", required=True)
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(func=_prepare)

    merge = subparsers.add_parser("merge-parquet", help="merge and deterministically shuffle prompt data")
    merge.add_argument("--sources", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--seed", type=int, default=42)
    merge.set_defaults(func=_merge_parquet)

    eval_config = subparsers.add_parser("build-eval-config", help="build the seven-dataset, B=1..8 matrix")
    eval_config.add_argument("--data-dir", required=True)
    eval_config.add_argument("--output", required=True)
    eval_config.add_argument("--budgets", nargs="+", type=int, default=list(range(1, 9)))
    eval_config.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    eval_config.add_argument("--max-response-len", type=int, default=4096)
    eval_config.set_defaults(func=_eval_config)

    summarize = subparsers.add_parser("summarize", help="combine independent evaluation runs")
    summarize.add_argument("--results-dir", required=True)
    summarize.add_argument("--output", required=True)
    summarize.set_defaults(func=_summarize)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
