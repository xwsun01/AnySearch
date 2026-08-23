# Third-party notices

AnySearch and the bundled Slime fork are licensed under Apache-2.0. That license does not
relicense dependencies, models, datasets, corpora, indexes, or generated
checkpoints. No third-party weights or data are included here.

## Framework and reference implementation

- [THUDM/slime](https://github.com/THUDM/slime), used as the fork base at
  `52fc971bfe4ad7a1e857ac158d626d4b6373474d`, is Apache-2.0. Its source is
  retained in this repository; AnySearch uses its public data-source, rollout,
  reward, Ray, Megatron, and SGLang extension APIs.
- Slime's `examples/search-r1` informed the multi-turn search integration. It is
  part of the same Apache-2.0 repository.
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) is an upstream
  research reference under Apache-2.0. It is not vendored and no baseline is
  implemented in this repository.

Slime brings additional dependencies including Megatron-LM, Ray, SGLang,
PyTorch, Transformers, and their transitive packages. Consult the exact pinned
Slime environment and each installed distribution's metadata; this file does
not replace their license texts or notices.

## Models and data

The model backbones and retriever are distributed separately: Qwen2.5-7B-Instruct
and Qwen3-4B use Apache-2.0, E5-base-v2 uses MIT, and Meta Llama 3.1 uses the
Llama 3.1 Community License. The seven QA benchmarks are obtained through the
[FlashRAG dataset collection](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets)
and retain both the collection terms and their original benchmark terms. The
Wikipedia 2018 corpus remains subject to Wikimedia and source-distributor terms.

If you redistribute a checkpoint trained from a third-party backbone, determine
and satisfy the backbone license, dataset terms, corpus attribution/share-alike
requirements, and any acceptable-use obligations. The fact that training code
is Apache-2.0 does not determine the checkpoint's redistribution terms.
