"""Budget-aware multi-turn rollout implementation for AnySearch.

Model token ids and log probabilities are preserved directly from generation.
Retrieved passages and other environment text receive a zero loss mask and a
dummy log probability.
"""

from __future__ import annotations

import logging
import os
import re
from argparse import Namespace
from collections.abc import Mapping, Sequence
from typing import Any

from examples.anysearch.config import ExperimentConfig
from examples.anysearch.prompts import budget_tag, build_phase_one_prompt, build_phase_two_prompt
from examples.anysearch.retrieval import RetrievalClient, RetrievalClientError, close_shared_sessions, format_documents
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_ACTION_PATTERN = re.compile(r"<(search|answer)>\s*(.*?)\s*</\1>", re.DOTALL | re.IGNORECASE)
_CLOSING_STOPS = ["</search>", "</answer>"]
_DEFAULT_GENERATED_TOKEN_LIMIT = 4096
_retrieval_clients: dict[tuple[Any, ...], RetrievalClient] = {}


def _generate_state(args: Namespace) -> Any:
    # Load the generation backend only when a rollout starts so data and reward
    # utilities remain lightweight.
    from slime.rollout.sglang_rollout import GenerateState

    return GenerateState(args)


async def _post(url: str, payload: Mapping[str, Any], *, headers: Mapping[str, str] | None = None) -> Any:
    from slime.utils.http_utils import post

    return await post(url, payload, headers=headers)


def _run_base_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    *,
    evaluation: bool,
) -> Any:
    from slime.rollout.sglang_rollout import generate_rollout as base_generate_rollout

    return base_generate_rollout(args, rollout_id, data_source, evaluation=evaluation)


def _build_eval_dataset(args: Namespace, dataset_config: Any) -> Any:
    from slime.utils.data import Dataset
    from slime.utils.processing_utils import load_processor, load_tokenizer

    return Dataset(
        path=dataset_config.path,
        tokenizer=load_tokenizer(args.hf_checkpoint, trust_remote_code=True),
        processor=load_processor(args.hf_checkpoint, trust_remote_code=True),
        max_length=args.eval_max_prompt_len,
        prompt_key=dataset_config.input_key,
        label_key=dataset_config.label_key,
        multimodal_keys=args.multimodal_keys,
        metadata_key=dataset_config.metadata_key,
        tool_key=dataset_config.tool_key,
        apply_chat_template=args.apply_chat_template,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
    )


async def _generate_eval_sample(
    args: Namespace, sample: Sample, sampling_params: dict[str, Any]
) -> Sample | list[Sample]:
    from slime.rollout.sglang_rollout import generate_and_rm

    return await generate_and_rm(args, sample, sampling_params=sampling_params, evaluation=True)


async def _eval_single_dataset(args: Namespace, rollout_id: int, dataset_config: Any) -> Mapping[str, Any]:
    """Evaluate one dataset with a bounded number of live tasks."""

    import asyncio
    import copy

    del rollout_id  # Deterministic seeds are defined per sample below.
    cache = _eval_dataset_cache()
    cache_key = dataset_config.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in cache:
        cache[cache_key] = _build_eval_dataset(args, dataset_config)
    dataset = cache[cache_key]

    base_sampling_params = {
        "temperature": dataset_config.temperature,
        "top_p": dataset_config.top_p,
        "top_k": dataset_config.top_k,
        "max_new_tokens": dataset_config.max_response_len,
        "stop": args.rollout_stop,
        "stop_token_ids": args.rollout_stop_token_ids,
        "skip_special_tokens": args.rollout_skip_special_tokens,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }
    concurrency = _runtime_int(
        args,
        "anysearch_eval_concurrency",
        "ANYSEARCH_EVAL_CONCURRENCY",
        64,
    )
    if concurrency < 1:
        raise ValueError("AnySearch eval concurrency must be positive")
    if dataset_config.n_samples_per_eval_prompt < 1:
        raise ValueError("n_samples_per_eval_prompt must be positive")

    pending: set[asyncio.Task[Sample | list[Sample]]] = set()
    data: list[Sample] = []
    sample_index = 0
    logged_example = False

    async def collect_finished(*, all_tasks: bool) -> None:
        nonlocal pending, logged_example
        if not pending:
            return
        done, pending = await asyncio.wait(
            pending,
            return_when=asyncio.ALL_COMPLETED if all_tasks else asyncio.FIRST_COMPLETED,
        )
        generated_results = await asyncio.gather(*done)
        for generated in generated_results:
            samples = generated if isinstance(generated, list) else [generated]
            data.extend(samples)
            if not logged_example and samples:
                example = samples[0]
                logger.info(
                    "AnySearch eval example for %s: %s reward=%s",
                    dataset_config.name,
                    str(example.prompt) + example.response,
                    example.reward,
                )
                logged_example = True

    try:
        for prompt_sample in dataset.samples:
            for sample_number in range(dataset_config.n_samples_per_eval_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.index = sample_index
                sample_index += 1
                sample.metadata = dataset_config.inject_metadata(getattr(sample, "metadata", None))
                sample.generate_function_path = getattr(dataset_config, "custom_generate_function_path", None)
                sampling_params = base_sampling_params
                if getattr(args, "sglang_enable_deterministic_inference", False):
                    sampling_params = base_sampling_params.copy()
                    sampling_params["sampling_seed"] = args.rollout_seed + sample_number
                pending.add(asyncio.create_task(_generate_eval_sample(args, sample, sampling_params)))
                if len(pending) >= concurrency:
                    await collect_finished(all_tasks=False)
        await collect_finished(all_tasks=True)
    finally:
        if pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    data.sort(key=lambda sample: sample.index)
    if not data:
        raise RuntimeError(f"evaluation dataset {dataset_config.name!r} is empty")

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_config.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


def _eval_dataset_cache() -> dict[tuple[Any, ...], Any]:
    from slime.rollout.sglang_rollout import EVAL_PROMPT_DATASET

    return EVAL_PROMPT_DATASET


def _eval_logger(args: Namespace) -> Any:
    path = getattr(args, "custom_eval_rollout_log_function_path", None)
    if path:
        from slime.utils.misc import load_function

        return load_function(path)
    from examples.anysearch.metrics import log_eval_rollout_data

    return log_eval_rollout_data


async def _eval_rollout_sequential(args: Namespace, rollout_id: int) -> dict[str, dict[str, Any]]:
    """Evaluate matrix entries sequentially with bounded memory.

    Detailed samples are written after each entry; only rewards and truncation
    flags remain in memory for the final aggregate report.
    """

    cache = _eval_dataset_cache()
    dataset_aliases: dict[tuple[Any, ...], Any] = {}
    log_entry = _eval_logger(args)
    compact: dict[str, dict[str, Any]] = {}
    checkpoint = getattr(args, "hf_checkpoint", None)
    apply_chat_template = getattr(args, "apply_chat_template", False)

    try:
        for dataset_config in getattr(args, "eval_datasets", []) or []:
            cache_key = dataset_config.cache_key + (checkpoint, apply_chat_template)
            # Ignore the entry name when sharing an immutable prompt dataset
            # across B=1..8 evaluations of the same file and schema.
            alias_key = cache_key[1:]
            if alias_key in dataset_aliases:
                cache[cache_key] = dataset_aliases[alias_key]
            try:
                entry = await _eval_single_dataset(args, rollout_id, dataset_config)
                if cache_key in cache:
                    dataset_aliases.setdefault(alias_key, cache[cache_key])
                if not isinstance(entry, Mapping) or len(entry) != 1:
                    raise RuntimeError("each AnySearch eval entry must return exactly one dataset result")
                # Persist per-sample metrics before releasing the current entry.
                # The compact result intentionally omits ``samples`` to prevent
                # duplicate output and empty-array reductions.
                log_entry(rollout_id, args, entry, {"anysearch_streamed_eval": True})
                for name, info in entry.items():
                    if not isinstance(info, Mapping):
                        raise RuntimeError(f"eval result {name!r} must be a mapping")
                    rewards = list(info.get("rewards", []))
                    truncated = list(info.get("truncated", []))
                    samples = info.get("samples", [])
                    if len(rewards) != len(truncated):
                        raise RuntimeError(f"eval result {name!r} has misaligned rewards/truncated arrays")
                    if samples and len(samples) != len(rewards):
                        raise RuntimeError(f"eval result {name!r} has misaligned samples/rewards arrays")
                    compact[name] = {
                        "rewards": rewards,
                        "truncated": truncated,
                        "count": len(rewards),
                        "anysearch_streamed": True,
                    }
            finally:
                # Keep at most one cache key per active entry; dataset_aliases
                # holds one object per unique path/schema until this call ends.
                cache.pop(cache_key, None)
    finally:
        # All retrieval requests are complete before the shared session closes.
        await close_shared_sessions()

    return compact


def _run_sequential_eval(args: Namespace, rollout_id: int) -> Any:
    from slime.rollout.base_types import RolloutFnEvalOutput
    from slime.utils.async_utils import run

    if getattr(args, "save_debug_rollout_data", None) is not None:
        raise ValueError(
            "bounded AnySearch evaluation is incompatible with --save-debug-rollout-data; "
            "use ANYSEARCH_EVAL_OUTPUT_DIR for streamed per-sample JSON instead"
        )
    data = run(_eval_rollout_sequential(args, rollout_id))
    return RolloutFnEvalOutput(
        data=data,
        metrics={"anysearch/eval_entries": len(data), "anysearch/eval_streamed": 1},
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _config_path(args: Namespace) -> str | os.PathLike[str] | None:
    return (
        getattr(args, "anysearch_config", None)
        or getattr(args, "custom_config_path", None)
        or os.environ.get("ANYSEARCH_CONFIG")
    )


def _raw_config(args: Namespace) -> Mapping[str, Any]:
    """Return canonical YAML values not represented by the core typed config.

    Prefer sections already injected into ``args`` and otherwise load the file
    selected by ``ANYSEARCH_CONFIG``.
    """

    cached = getattr(args, "anysearch_raw_config", None)
    if cached is not None:
        if not isinstance(cached, Mapping):
            raise TypeError("args.anysearch_raw_config must be a mapping")
        return cached

    path = _config_path(args)
    if path is None:
        loaded: Mapping[str, Any] = {}
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise RuntimeError("YAML configuration requires PyYAML") from exc
        with open(path, encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
        if not isinstance(value, Mapping):
            raise TypeError(f"AnySearch configuration at {path!s} must be a mapping")
        loaded = value
    args.anysearch_raw_config = loaded
    return loaded


def _config_option(args: Namespace, section: str, key: str, default: Any) -> Any:
    # Prefer an already-injected section before reading the configuration file.
    section_value = getattr(args, section, None)
    if section_value is None:
        section_value = _raw_config(args).get(section)
    if section_value is None:
        return default
    if not isinstance(section_value, Mapping):
        raise TypeError(f"AnySearch config section {section!r} must be a mapping")
    return section_value.get(key, default)


def _runtime_int(args: Namespace, arg_name: str, env_name: str, default: int) -> int:
    value = getattr(args, arg_name, None)
    if value is None:
        raw = os.environ.get(env_name)
        value = default if raw is None else raw
    if isinstance(value, bool):
        raise TypeError(f"{arg_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{arg_name}/{env_name} must be an integer, got {value!r}") from exc


def _runtime_float(args: Namespace, arg_name: str, env_name: str, default: float) -> float:
    value = getattr(args, arg_name, None)
    if value is None:
        raw = os.environ.get(env_name)
        value = default if raw is None else raw
    if isinstance(value, bool):
        raise TypeError(f"{arg_name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{arg_name}/{env_name} must be a number, got {value!r}") from exc


def _retriever(args: Namespace) -> RetrievalClient:
    endpoint_env = str(_config_option(args, "retrieval", "endpoint_env", "ANYSEARCH_RETRIEVAL_URL"))
    default_url = str(_config_option(args, "retrieval", "default_endpoint", "http://127.0.0.1:8000/retrieve"))
    url = getattr(args, "anysearch_retrieval_url", None) or os.environ.get(endpoint_env) or default_url
    configured_top_k = int(_config_option(args, "retrieval", "top_k", 3))
    top_k = _runtime_int(args, "anysearch_retrieval_top_k", "ANYSEARCH_RETRIEVAL_TOP_K", configured_top_k)
    timeout = _runtime_float(args, "anysearch_retrieval_timeout", "ANYSEARCH_RETRIEVAL_TIMEOUT", 30.0)
    retries = _runtime_int(args, "anysearch_retrieval_retries", "ANYSEARCH_RETRIEVAL_RETRIES", 3)
    concurrency = _runtime_int(
        args,
        "anysearch_retrieval_concurrency",
        "ANYSEARCH_RETRIEVAL_CONCURRENCY",
        256,
    )
    key = (url, top_k, timeout, retries, concurrency)
    if key not in _retrieval_clients:
        _retrieval_clients[key] = RetrievalClient(
            url,
            top_k=top_k,
            timeout_seconds=timeout,
            max_retries=retries,
            concurrency=concurrency,
            connector_limit=concurrency,
        )
    return _retrieval_clients[key]


def _metadata(sample: Sample) -> dict[str, Any]:
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    return sample.metadata


def _is_phase1(value: Any) -> bool:
    normalized = str(getattr(value, "value", value)).lower().replace("-", "_").replace(" ", "_")
    return normalized in {"1", "phase1", "phase_i", "warmup", "annealing"}


def _experiment_config(args: Namespace) -> ExperimentConfig:
    config = getattr(args, "anysearch_experiment_config", None)
    if config is None:
        config_path = _config_path(args)
        config = ExperimentConfig.from_yaml(config_path) if config_path else ExperimentConfig()
        args.anysearch_experiment_config = config
    if not isinstance(config, ExperimentConfig):
        raise TypeError("args.anysearch_experiment_config must be an ExperimentConfig")
    return config


def _scaffold_for_sample(args: Namespace, metadata: Mapping[str, Any], *, evaluation: bool) -> bool:
    if "scaffold" in metadata:
        return bool(metadata["scaffold"])
    if not evaluation:
        return _is_phase1(metadata.get("phase"))
    return _experiment_config(args).inference_scaffold


def _extract_question(prompt: str | Sequence[Mapping[str, Any]]) -> str:
    if isinstance(prompt, str):
        question = prompt.strip()
    else:
        question = ""
        for message in reversed(prompt):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                question = str(message["content"]).strip()
                break
    if not question:
        raise ValueError("AnySearch requires a non-empty question prompt")
    return question


def _render_episode_prompt(args: Namespace, state: Any, sample: Sample, *, evaluation: bool) -> None:
    metadata = _metadata(sample)
    if not metadata.get("anysearch_prompt_built"):
        budget = int(metadata.get("budget_total", metadata.get("budget", 0)))
        if budget < 0:
            raise ValueError(f"search budget cannot be negative: {budget}")
        question = _extract_question(sample.prompt)
        scaffold = _scaffold_for_sample(args, metadata, evaluation=evaluation)
        sample.prompt = (
            build_phase_one_prompt(question, budget) if scaffold else build_phase_two_prompt(question, budget)
        )
        metadata["anysearch_prompt_built"] = True
        metadata["question"] = question
        metadata["scaffold"] = scaffold

    apply_chat_template = os.environ.get("ANYSEARCH_APPLY_CHAT_TEMPLATE", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if apply_chat_template and not metadata.get("anysearch_chat_templated"):
        apply_template = getattr(state.tokenizer, "apply_chat_template", None)
        if apply_template is None:
            raise RuntimeError("tokenizer does not provide apply_chat_template; set ANYSEARCH_APPLY_CHAT_TEMPLATE=0")
        sample.prompt = apply_template(
            [{"role": "user", "content": str(sample.prompt)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        metadata["anysearch_chat_templated"] = True


def _encode(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def _append_environment(
    sample: Sample,
    tokenizer: Any,
    text: str,
    *,
    kind: str,
    context_limit: int | None,
) -> bool:
    token_ids = _encode(tokenizer, text)
    if context_limit is not None and len(sample.tokens) + len(token_ids) > context_limit:
        return False
    sample.response += text
    sample.tokens.extend(token_ids)
    sample.response_length += len(token_ids)
    if sample.loss_mask is None:
        sample.loss_mask = []
    sample.loss_mask.extend([0] * len(token_ids))
    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs.extend([0.0] * len(token_ids))

    metadata = _metadata(sample)
    metadata["environment_tokens"] = int(metadata.get("environment_tokens", 0)) + len(token_ids)
    if kind == "information":
        metadata["retrieval_tokens"] = int(metadata.get("retrieval_tokens", 0)) + len(token_ids)
    _assert_alignment(sample)
    return True


def _append_generation(sample: Sample, text: str, token_ids: list[int], log_probs: list[float]) -> None:
    if len(token_ids) != len(log_probs):
        raise RuntimeError(f"SGLang token/logprob mismatch: {len(token_ids)} tokens != {len(log_probs)} logprobs")
    sample.response += text
    sample.tokens.extend(token_ids)
    sample.response_length += len(token_ids)
    if sample.loss_mask is None:
        sample.loss_mask = []
    sample.loss_mask.extend([1] * len(token_ids))
    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs.extend(log_probs)
    metadata = _metadata(sample)
    metadata["generated_tokens"] = int(metadata.get("generated_tokens", 0)) + len(token_ids)
    _assert_alignment(sample)


def _assert_alignment(sample: Sample) -> None:
    if sample.loss_mask is None or sample.rollout_log_probs is None:
        raise RuntimeError("loss_mask and rollout_log_probs must be initialized")
    if len(sample.loss_mask) != sample.response_length:
        raise RuntimeError(
            f"response/loss-mask mismatch: {sample.response_length} tokens != {len(sample.loss_mask)} masks"
        )
    if len(sample.rollout_log_probs) != sample.response_length:
        raise RuntimeError(
            f"response/logprob mismatch: {sample.response_length} tokens != {len(sample.rollout_log_probs)} logprobs"
        )
    prompt_length = len(sample.tokens) - sample.response_length
    if prompt_length < 0 or len(sample.tokens) != prompt_length + sample.response_length:
        raise RuntimeError("sample token accounting is inconsistent")


def _finish_reason(output: Mapping[str, Any]) -> str:
    meta_info = output.get("meta_info")
    if not isinstance(meta_info, Mapping):
        raise RuntimeError("SGLang response is missing meta_info")
    finish_reason = meta_info.get("finish_reason")
    if not isinstance(finish_reason, Mapping) or not isinstance(finish_reason.get("type"), str):
        raise RuntimeError("SGLang response has an invalid finish_reason")
    return str(finish_reason["type"])


def _tokens_and_logprobs(output: Mapping[str, Any]) -> tuple[list[int], list[float]]:
    meta_info = output.get("meta_info")
    if not isinstance(meta_info, Mapping):
        raise RuntimeError("SGLang response is missing meta_info")
    entries = meta_info.get("output_token_logprobs")
    if not isinstance(entries, list):
        raise RuntimeError("SGLang did not return output_token_logprobs despite return_logprob=true")
    token_ids: list[int] = []
    log_probs: list[float] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise RuntimeError(f"invalid output_token_logprobs entry at index {index}")
        log_prob, token_id = entry[0], entry[1]
        if not isinstance(token_id, int) or not isinstance(log_prob, (int, float)):
            raise RuntimeError(f"invalid token id/logprob at index {index}")
        token_ids.append(token_id)
        log_probs.append(float(log_prob))
    return token_ids, log_probs


def _record_sglang_observability(args: Namespace, sample: Sample, output: Mapping[str, Any]) -> None:
    meta_info = output.get("meta_info")
    if not isinstance(meta_info, Mapping):  # already rejected by _finish_reason
        return
    if getattr(args, "sglang_speculative_algorithm", None):
        sample.spec_info.add(meta_info=dict(meta_info))
    sample.prefix_cache_info.add(meta_info=dict(meta_info))
    weight_version = meta_info.get("weight_version")
    if weight_version is not None:
        sample.weight_versions.append(weight_version)


def _parse_action(text: str) -> tuple[str | None, str]:
    match = _ACTION_PATTERN.search(text)
    if match is None:
        return None, ""
    return match.group(1).lower(), match.group(2).strip()


def _sampling_params_for_turn(
    args: Namespace,
    sampling_params: Mapping[str, Any],
    *,
    generated_tokens: int,
    context_tokens: int,
    evaluation: bool,
) -> dict[str, Any]:
    configured_value = getattr(args, "rollout_max_response_len", None)
    configured_limit = int(
        configured_value if configured_value is not None else _experiment_config(args).generation.max_response_length
    )
    hard_limit = _env_int("ANYSEARCH_MAX_GENERATED_TOKENS", _DEFAULT_GENERATED_TOKEN_LIMIT)
    total_limit = min(configured_limit, hard_limit)
    remaining = total_limit - generated_tokens
    params = dict(sampling_params)
    per_turn_limit = min(int(params.get("max_new_tokens", total_limit)), max(remaining, 0))
    context_limit = _context_limit(args, evaluation=evaluation)
    if context_limit is not None:
        per_turn_limit = min(per_turn_limit, max(context_limit - context_tokens, 0))
    params["max_new_tokens"] = per_turn_limit
    params["stop"] = list(_CLOSING_STOPS)
    params["no_stop_trim"] = True
    params["spaces_between_special_tokens"] = False
    return params


def _context_limit(args: Namespace, *, evaluation: bool) -> int:
    arg_name = "eval_max_context_len" if evaluation else "rollout_max_context_len"
    # Fall back to the rollout limit when no separate evaluation limit is set.
    if evaluation and getattr(args, arg_name, None) is None:
        arg_name = "rollout_max_context_len"
    limit = _runtime_int(
        args,
        arg_name,
        "ANYSEARCH_MAX_CONTEXT_TOKENS",
        _experiment_config(args).generation.max_context_length,
    )
    if limit < 1:
        raise ValueError("rollout_max_context_len must be positive")
    return limit


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: Mapping[str, Any],
    *,
    evaluation: bool = False,
) -> Sample:
    """Generate one complete AnySearch trajectory through the SGLang router."""

    if getattr(args, "partial_rollout", False):
        raise ValueError("AnySearch uses complete trajectories and does not support partial_rollout")
    if sample.response or sample.response_length:
        raise ValueError("AnySearch custom generation requires a fresh sample")

    state = _generate_state(args)
    _render_episode_prompt(args, state, sample, evaluation=evaluation)
    prompt_text = str(sample.prompt)
    prompt_ids = _encode(state.tokenizer, prompt_text)
    sample.tokens = prompt_ids
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = []

    metadata = _metadata(sample)
    context_limit = _context_limit(args, evaluation=evaluation)
    if context_limit is not None and len(prompt_ids) >= context_limit:
        sample.status = Sample.Status.TRUNCATED
        metadata["finish_reason"] = "prompt_context_limit"
        metadata["generated_tokens"] = 0
        metadata["retrieval_tokens"] = 0
        metadata["environment_tokens"] = 0
        return sample
    budget = int(metadata.get("budget_total", metadata.get("budget", 0)))
    if budget < 0:
        raise ValueError(f"search budget cannot be negative: {budget}")
    metadata.update(
        {
            "budget": budget,
            "budget_total": budget,
            "budget_used": 0,
            "search_count": 0,
            "generated_tokens": 0,
            "retrieval_tokens": 0,
            "environment_tokens": 0,
            "evaluation": bool(evaluation),
        }
    )
    scaffold = _scaffold_for_sample(args, metadata, evaluation=evaluation)
    metadata["scaffold"] = scaffold
    if scaffold and not _append_environment(
        sample,
        state.tokenizer,
        budget_tag(total=budget, used=0),
        kind="budget",
        context_limit=context_limit,
    ):
        sample.status = Sample.Status.TRUNCATED
        metadata["finish_reason"] = "budget_context_limit"
        return sample

    session_id = sample.session_id or f"anysearch-{sample.group_index}-{sample.index}"
    sample.session_id = session_id
    headers = None
    if getattr(args, "sglang_router_policy", None) == "consistent_hashing":
        headers = {"X-SMG-Routing-Key": session_id}

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    search_count = 0
    invalid_retries = 0
    configured_retries = int(
        metadata.get("invalid_action_max_retries", _experiment_config(args).invalid_action_max_retries)
    )
    max_invalid_retries = _runtime_int(
        args,
        "anysearch_invalid_action_retries",
        "ANYSEARCH_INVALID_ACTION_RETRIES",
        configured_retries,
    )
    if max_invalid_retries < 0:
        raise ValueError("ANYSEARCH_INVALID_ACTION_RETRIES cannot be negative")
    while True:
        turn_params = _sampling_params_for_turn(
            args,
            sampling_params,
            generated_tokens=int(metadata["generated_tokens"]),
            context_tokens=len(sample.tokens),
            evaluation=evaluation,
        )
        if turn_params["max_new_tokens"] == 0:
            sample.status = Sample.Status.TRUNCATED
            configured_limit = min(
                int(
                    getattr(args, "rollout_max_response_len", None)
                    or _experiment_config(args).generation.max_response_length
                ),
                _env_int("ANYSEARCH_MAX_GENERATED_TOKENS", _DEFAULT_GENERATED_TOKEN_LIMIT),
            )
            metadata["finish_reason"] = (
                "generated_token_limit" if int(metadata["generated_tokens"]) >= configured_limit else "context_limit"
            )
            break

        payload = {
            "input_ids": list(sample.tokens),
            "sampling_params": turn_params,
            "return_logprob": True,
        }
        output = await _post(url, payload, headers=headers)
        if not isinstance(output, Mapping):
            raise RuntimeError("SGLang response must be a JSON object")
        finish_reason = _finish_reason(output)
        metadata["sglang_finish_reason"] = finish_reason
        _record_sglang_observability(args, sample, output)
        if finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            metadata["finish_reason"] = "abort"
            return sample

        text = output.get("text")
        if not isinstance(text, str):
            raise RuntimeError("SGLang response.text must be a string")
        token_ids, log_probs = _tokens_and_logprobs(output)
        if len(token_ids) > int(turn_params["max_new_tokens"]):
            raise RuntimeError(
                f"SGLang returned more tokens than requested: {len(token_ids)} > {turn_params['max_new_tokens']}"
            )
        _append_generation(sample, text, token_ids, log_probs)

        if finish_reason == "length":
            sample.status = Sample.Status.TRUNCATED
            metadata["finish_reason"] = "length"
            break

        action, content = _parse_action(text)
        if action == "answer":
            sample.status = Sample.Status.COMPLETED
            metadata["answer"] = content
            metadata["finish_reason"] = "answer"
            break
        invalid_action = action != "search" or not content
        budget_exceeded = action == "search" and bool(content) and search_count >= budget
        if invalid_action or budget_exceeded:
            if budget_exceeded:
                error_reason = "budget_exceeded"
                error_message = (
                    "\n\nThe search budget is exhausted. Do not issue another <search> action. "
                    "Use the available information and provide a final <answer> instead.\n\n"
                )
                metadata["budget_violation"] = True
            else:
                error_reason = "invalid_action"
                error_message = (
                    "\n\nYour previous action was invalid. Produce a reasoning block followed by exactly one "
                    "non-empty <search>query</search> or <answer>answer</answer> action.\n\n"
                )
            metadata["invalid_action_count"] = int(metadata.get("invalid_action_count", 0)) + 1
            if invalid_retries >= max_invalid_retries:
                sample.status = Sample.Status.FAILED
                metadata["finish_reason"] = error_reason
                metadata["rollout_error"] = f"{error_reason} after {max_invalid_retries} retry turn(s)"
                break
            invalid_retries += 1
            metadata["invalid_action_retries"] = invalid_retries
            if not _append_environment(
                sample,
                state.tokenizer,
                error_message,
                kind="error",
                context_limit=context_limit,
            ):
                sample.status = Sample.Status.TRUNCATED
                metadata["finish_reason"] = "error_context_limit"
                break
            if scaffold and not _append_environment(
                sample,
                state.tokenizer,
                budget_tag(total=budget, used=search_count),
                kind="budget",
                context_limit=context_limit,
            ):
                sample.status = Sample.Status.TRUNCATED
                metadata["finish_reason"] = "budget_context_limit"
                break
            continue

        try:
            documents = await _retriever(args).retrieve(content)
        except RetrievalClientError as exc:
            sample.status = Sample.Status.FAILED
            metadata["finish_reason"] = "retrieval_error"
            metadata["rollout_error"] = f"{type(exc).__name__}: {exc}"
            logger.error("retrieval failed for sample %s: %s", sample.index, exc)
            break

        search_count += 1
        metadata["budget_used"] = search_count
        metadata["search_count"] = search_count
        information = f"\n\n<information>{format_documents(documents)}</information>\n\n"
        if not _append_environment(
            sample,
            state.tokenizer,
            information,
            kind="information",
            context_limit=context_limit,
        ):
            sample.status = Sample.Status.TRUNCATED
            metadata["finish_reason"] = "information_context_limit"
            break
        if scaffold:
            if not _append_environment(
                sample,
                state.tokenizer,
                budget_tag(total=budget, used=search_count),
                kind="budget",
                context_limit=context_limit,
            ):
                sample.status = Sample.Status.TRUNCATED
                metadata["finish_reason"] = "budget_context_limit"
                break
        # Deliberately continue after search_count == budget: this is the
        # answer turn that is lost in max_turns=B implementations.

    _assert_alignment(sample)
    return sample


def _set_rollout_context(data_source: Any, rollout_id: int, *, evaluation: bool) -> None:
    setter = getattr(data_source, "set_rollout", None)
    if callable(setter):
        setter(rollout_id, evaluation=evaluation)


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> Any:
    """Full rollout wrapper that makes ``rollout_id`` visible to curriculum state."""

    if evaluation:
        return generate_eval_rollout(args, rollout_id, data_source, evaluation=True)
    _set_rollout_context(data_source, rollout_id, evaluation=False)
    output = _run_base_rollout(args, rollout_id, data_source, evaluation=False)
    recorder = getattr(data_source, "record_rollout", None)
    if callable(recorder):
        recorder(output.samples)
    return output


def generate_eval_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = True,
) -> Any:
    """Run evaluation with single-sample rewards while training uses group rewards."""

    _set_rollout_context(data_source, rollout_id, evaluation=True)
    original_group_rm = bool(getattr(args, "group_rm", False))
    original_rm_path = getattr(args, "custom_rm_path", None)
    args.group_rm = False
    args.custom_rm_path = "examples.anysearch.slime_ext.rewards.eval_reward"
    try:
        return _run_sequential_eval(args, rollout_id)
    finally:
        args.group_rm = original_group_rm
        args.custom_rm_path = original_rm_path
