"""Canonical Parquet schema and converters for the AnySearch experiments."""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_QUESTION_FROM_SEARCH_R1 = re.compile(r"(?:^|\s)Question:\s*(?P<question>.+?)\s*$", re.DOTALL)


@dataclass(frozen=True)
class QARecord:
    """Canonical question-answer record used for training and evaluation."""

    question: str
    label: dict[str, list[str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        question = self.question.strip()
        if not question:
            raise ValueError("question must not be empty")
        targets = self.label.get("target")
        if not isinstance(targets, list) or not targets or not all(isinstance(item, str) and item for item in targets):
            raise ValueError("label.target must be a non-empty list of strings")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _last_user_message(prompt: Sequence[Any]) -> str | None:
    for message in reversed(prompt):
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return None


def extract_question(record: Mapping[str, Any]) -> str:
    """Extract a raw question from supported QA record layouts."""

    raw_question = record.get("question")
    if isinstance(raw_question, str) and raw_question.strip():
        question = raw_question.strip()
    else:
        prompt = record.get("prompt")
        if isinstance(prompt, str):
            prompt_text = prompt
        elif isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)):
            prompt_text = _last_user_message(prompt) or ""
        else:
            prompt_text = ""
        match = _QUESTION_FROM_SEARCH_R1.search(prompt_text)
        question = match.group("question").strip() if match else prompt_text.strip()

    if not question:
        raise ValueError("record does not contain a usable question")
    if question[-1] not in "?!.":
        question += "?"
    return question


def _coerce_targets(value: Any) -> list[str]:
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        raise ValueError("answers must be a string or sequence of strings")

    targets: list[str] = []
    for item in values:
        if isinstance(item, Mapping):
            item = item.get("text") or item.get("answer")
        if isinstance(item, str) and (target := item.strip()) and target not in targets:
            targets.append(target)
    if not targets:
        raise ValueError("record does not contain a non-empty answer")
    return targets


def extract_targets(record: Mapping[str, Any]) -> list[str]:
    """Extract all accepted aliases without applying EM normalization."""

    candidates: list[Any] = [record.get("golden_answers"), record.get("answers"), record.get("answer")]
    label = record.get("label")
    if isinstance(label, Mapping):
        candidates.extend([label.get("target"), label.get("answers"), label.get("answer")])
    else:
        candidates.append(label)

    reward_model = record.get("reward_model")
    if isinstance(reward_model, Mapping):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, Mapping):
            candidates.extend([ground_truth.get("target"), ground_truth.get("answers")])

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return _coerce_targets(candidate)
        except ValueError:
            continue
    raise ValueError("record does not contain a supported answer field")


def canonicalize_record(
    record: Mapping[str, Any],
    *,
    dataset: str,
    split: str,
    index: int,
) -> QARecord:
    """Convert one source row into the canonical AnySearch schema."""

    return QARecord(
        question=extract_question(record),
        label={"target": extract_targets(record)},
        metadata={"dataset": dataset, "split": split, "index": index},
    )


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read JSON, JSONL, or Parquet records.

    Parquet support imports ``pyarrow`` lazily so importing the core package
    remains lightweight.
    """

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        with source.open(encoding="utf-8") as handle:
            if suffix == ".json":
                payload = json.load(handle)
                rows = payload if isinstance(payload, list) else payload.get("data", [])
                if not isinstance(rows, list):
                    raise ValueError(f"unsupported JSON structure in {source}")
                yield from rows
            else:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"expected an object at {source}:{line_number}")
                    yield row
        return
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Parquet conversion requires `pip install -r examples/anysearch/requirements-data.txt`"
            ) from exc
        parquet_file = pq.ParquetFile(source)
        for batch in parquet_file.iter_batches():
            yield from batch.to_pylist()
        return
    raise ValueError(f"unsupported input format: {source.suffix}")


def write_parquet(records: Iterable[QARecord], destination: str | Path) -> int:
    """Atomically write canonical records as compressed Parquet."""

    output = Path(destination)
    if output.suffix.lower() != ".parquet":
        raise ValueError(f"canonical AnySearch output must use .parquet, got {output}")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Parquet output requires `pip install -r examples/anysearch/requirements-data.txt`"
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.to_dict() for record in records]
    table = pa.Table.from_pylist(rows)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".parquet", delete=False) as handle:
            temporary_name = handle.name
        pq.write_table(table, temporary_name, compression="zstd")
        os.replace(temporary_name, output)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return table.num_rows


def merge_parquet_files(
    sources: Sequence[str | Path],
    destination: str | Path,
    *,
    seed: int,
) -> int:
    """Merge canonical prompt files, shuffle rows deterministically, and write atomically."""

    if not sources:
        raise ValueError("at least one Parquet source is required")
    output = Path(destination)
    if output.suffix.lower() != ".parquet":
        raise ValueError(f"canonical AnySearch output must use .parquet, got {output}")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Parquet merging requires `pip install -r examples/anysearch/requirements-data.txt`"
        ) from exc

    tables = [pq.read_table(Path(source)) for source in sources]
    expected_schema = tables[0].schema
    for source, table in zip(sources[1:], tables[1:], strict=True):
        if table.schema != expected_schema:
            raise ValueError(f"Parquet schema mismatch in {source}")
    mixed = pa.concat_tables(tables)
    order = list(range(mixed.num_rows))
    random.Random(seed).shuffle(order)
    mixed = mixed.take(pa.array(order, type=pa.int64()))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".parquet", delete=False) as handle:
            temporary_name = handle.name
        pq.write_table(mixed, temporary_name, compression="zstd")
        os.replace(temporary_name, output)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return mixed.num_rows


def convert_file(
    source: str | Path,
    destination: str | Path,
    *,
    dataset: str,
    split: str,
) -> int:
    rows = (
        canonicalize_record(row, dataset=dataset, split=split, index=index)
        for index, row in enumerate(iter_records(source))
    )
    return write_parquet(rows, destination)


def select_evaluation_split(available_splits: Iterable[str]) -> str:
    """Select the first labelled evaluation split available.

    Several benchmark datasets do not publish a labelled test
    split (HotpotQA, 2WikiMultiHopQA, and MuSiQue). The deterministic priority
    is ``test``, ``dev``, ``validation``, then ``train``.
    """

    available = set(available_splits)
    for candidate in ("test", "dev", "validation", "train"):
        if candidate in available:
            return candidate
    raise ValueError("FlashRAG dataset does not expose a supported evaluation split")


def load_flashrag(dataset_name: str, split: str) -> tuple[Iterable[Mapping[str, Any]], str]:
    """Load the public dataset source used by AnySearch.

    ``split='eval'`` resolves the first available labelled split and returns
    its name alongside the rows. Set
    ``FLASHRAG_REVISION`` to pin a Hugging Face dataset revision; the data
    manifest records this value in every output row.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "FlashRAG download requires `pip install -r examples/anysearch/requirements-data.txt`"
        ) from exc
    revision = os.environ.get("FLASHRAG_REVISION")
    kwargs = {"revision": revision} if revision else {}
    dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", dataset_name, **kwargs)
    resolved_split = select_evaluation_split(dataset.keys()) if split == "eval" else split
    if resolved_split not in dataset:
        available = ", ".join(sorted(dataset.keys()))
        raise ValueError(f"FlashRAG dataset {dataset_name!r} has no split {resolved_split!r}; available: {available}")
    return dataset[resolved_split], resolved_split


def prepare_flashrag_dataset(dataset_name: str, split: str, destination: str | Path) -> int:
    dataset, resolved_split = load_flashrag(dataset_name, split)
    revision = os.environ.get("FLASHRAG_REVISION", "main")

    def canonicalize(row: Mapping[str, Any], index: int) -> QARecord:
        record = canonicalize_record(row, dataset=dataset_name, split=resolved_split, index=index)
        record.metadata["flashrag_revision"] = revision
        return record

    rows = (canonicalize(row, index) for index, row in enumerate(dataset))
    return write_parquet(rows, destination)
