#!/usr/bin/env python3
"""Plot budget curves from the aggregated CSV output."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="CSV produced by summarize_results.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", default="exact_match_mean")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("plotting requires `pip install matplotlib>=3.8,<4`") from exc

    curves: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with Path(args.summary).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get(args.metric)
            if value:
                curves[row["dataset"]].append((int(row["budget"]), float(value)))

    figure, axis = plt.subplots(figsize=(8, 5))
    for dataset, points in sorted(curves.items()):
        points.sort()
        axis.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=dataset)
    axis.set_xlabel("Search budget")
    axis.set_ylabel(args.metric.replace("_mean", "").replace("_", " ").title())
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize="small")
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)


if __name__ == "__main__":
    main()
