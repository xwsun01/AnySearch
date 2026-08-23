"""Exact dense retrieval backend used by AnySearch."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _validate_flat_inner_product_index(faiss: Any, index: Any) -> tuple[str, str]:
    """Reject an index that would silently change the configured retrieval setup."""

    index_flat = getattr(faiss, "IndexFlat", None)
    if not isinstance(index_flat, type) or not isinstance(index, index_flat):
        raise ValueError(
            f"the canonical retriever requires an exact FAISS IndexFlat index; loaded {type(index).__name__}"
        )
    expected_metric = getattr(faiss, "METRIC_INNER_PRODUCT", None)
    observed_metric = getattr(index, "metric_type", None)
    if expected_metric is None or observed_metric != expected_metric:
        raise ValueError(
            f"the canonical E5 index must use inner-product similarity; loaded metric_type={observed_metric!r}"
        )
    return type(index).__name__, "inner_product"


class E5FaissRetriever:
    """E5-base-v2 encoder over a prebuilt FAISS Flat index."""

    def __init__(
        self,
        *,
        index_path: str | Path,
        corpus_path: str | Path,
        model_name_or_path: str = "intfloat/e5-base-v2",
        use_fp16: bool = True,
        faiss_gpu: bool = False,
        query_max_length: int = 256,
        batch_size: int = 128,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import datasets
            import faiss
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional GPU stack
            raise RuntimeError(
                "retriever dependencies are missing; install examples/anysearch/requirements-retriever.txt"
            ) from exc

        self._faiss = faiss
        self._torch = torch
        self.query_max_length = query_max_length
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self.corpus = datasets.load_dataset("json", data_files=str(corpus_path), split="train", num_proc=4)
        self.index = faiss.read_index(str(index_path))
        self.index_type, self.index_metric = _validate_flat_inner_product_index(faiss, self.index)
        if int(self.index.ntotal) != len(self.corpus):
            raise ValueError(
                "FAISS index/corpus mismatch: "
                f"index contains {int(self.index.ntotal)} vectors but corpus contains {len(self.corpus)} rows"
            )
        if faiss_gpu:
            if not hasattr(faiss, "GpuMultipleClonerOptions") or not hasattr(faiss, "index_cpu_to_all_gpus"):
                raise RuntimeError(
                    "--faiss-gpu requires a CUDA-enabled FAISS build; install the FAISS GPU package "
                    "matching this machine's CUDA stack, or set RETRIEVER_FAISS_GPU=0 for a documented CPU run"
                )
            options = faiss.GpuMultipleClonerOptions()
            options.useFloat16 = True
            options.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=options)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        self.model.eval().to(self.device)
        if use_fp16 and self.device.type == "cuda":
            self.model.half()
        self.model_name_or_path = model_name_or_path

    def _encode(self, queries: Sequence[str]) -> Any:
        torch = self._torch
        prefixed = [f"query: {query}" for query in queries]
        inputs = self.tokenizer(
            prefixed,
            max_length=self.query_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model(**inputs, return_dict=True)
            mask = inputs["attention_mask"].unsqueeze(-1).bool()
            hidden = output.last_hidden_state.masked_fill(~mask, 0.0)
            embeddings = hidden.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.float().cpu().numpy()

    def batch_search(self, queries: Sequence[str], top_k: int) -> list[list[dict[str, Any]]]:
        if not queries:
            return []
        output: list[list[dict[str, Any]]] = []
        with self._lock:
            for start in range(0, len(queries), self.batch_size):
                query_batch = queries[start : start + self.batch_size]
                scores, indices = self.index.search(self._encode(query_batch), top_k)
                for row_indices, row_scores in zip(indices, scores, strict=True):
                    documents = []
                    for document_index, score in zip(row_indices, row_scores, strict=True):
                        if int(document_index) < 0:
                            continue
                        source = dict(self.corpus[int(document_index)])
                        documents.append({"document": source, "score": float(score)})
                    output.append(documents)
        return output

    def info(self) -> dict[str, Any]:
        return {
            "backend": "e5-faiss-flat",
            "model": self.model_name_or_path,
            "corpus_size": len(self.corpus),
            "index_size": int(self.index.ntotal),
            "index_type": self.index_type,
            "index_metric": self.index_metric,
            "device": str(self.device),
        }
