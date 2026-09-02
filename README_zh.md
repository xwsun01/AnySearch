# AnySearch：One Policy, Any Budget

论文 **《One Policy, Any Budget: Internalizing Budget-Aware Search via
Reinforcement Learning》** 官方代码。

**EMNLP 2026 Findings**

<a href="https://arxiv.org/pdf/2609.00813"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white"></a>


AnySearch 通过两阶段预算课程学习以及答案质量和搜索效率的组级奖励，训练一个能够在
不同推理搜索预算下工作的统一搜索策略。

## 方法概览

<p align="center">
  <img src="examples/anysearch/assets/budget-behavior.png" width="95%" alt="固定预算搜索策略与 AnySearch 预算感知内化的对比">
</p>

<p align="center"><em>
固定预算策略可能在预算变化时过度搜索或未充分利用预算；AnySearch 将预算内化为策略状态，
用一个策略适配已见和未见预算。
</em></p>

AnySearch 方法包含四个主要组件：

1. **预算感知课程学习**：Phase I 将搜索预算从 `B=5` 逐步退火到 `B=1`；
   Phase II 根据各预算最近的正确率自适应采样。
2. **受预算约束的工具交互**：策略在推理、搜索、检索信息和最终答案之间交替，
   并严格遵守分配的搜索预算。
3. **组级搜索效率奖励**：同一个 GRPO group 共用一个预算，并综合答案正确率、
   输出格式、回答长度以及绝对/相对搜索效率计算奖励。
4. **跨预算评测**：使用同一个训练 checkpoint，在七个问答数据集上评测
   `B=1..8` 的表现。

### 整体框架

<p align="center">
  <img src="examples/anysearch/assets/method-overview.png" width="98%" alt="AnySearch 两阶段课程学习、自适应预算采样和组级工具奖励">
</p>

<p align="center"><em>
Phase I 使用 scaffold 和递减预算建立预算感知推理模式；Phase II 移除 scaffold，
自适应采样预算，并使用搜索效率感知的组级奖励。
</em></p>

以上两张图展示了方法动机和训练框架。

## 仓库结构

```text
.
├── train.py / train_async.py     # 训练入口
├── slime/                        # 分布式训练框架
└── examples/anysearch/
    ├── configs/                  # 训练、评测和消融配置
    ├── assets/                   # README 使用的方法示意图
    ├── retrieval/                # 异步检索客户端和健康检查
    ├── services/retriever/       # E5 + FAISS 检索服务
    ├── slime_ext/                # 数据源、rollout 和奖励集成
    ├── scripts/                  # 数据、训练、评测和分析脚本
    ├── curriculum.py             # 两阶段预算课程学习
    ├── prompts.py / protocol.py  # 提示词和工具交互协议
    └── rewards.py                # 复合组奖励
```

本仓库基于
[Slime commit `52fc971b`](https://github.com/THUDM/slime/commit/52fc971bfe4ad7a1e857ac158d626d4b6373474d)
构建，AnySearch 的专用实现位于 [`examples/anysearch`](examples/anysearch)。

## 安装

首先按 Slime 的要求准备 Megatron-LM、Ray、SGLang 以及匹配的 CUDA/PyTorch
环境，然后在仓库根目录安装：

```bash
python -m pip install -e .
python -m pip install -r examples/anysearch/requirements.txt
```

不同任务的附加依赖单独维护：

```bash
# 数据准备
python -m pip install -r examples/anysearch/requirements-data.txt

# CPU 检索服务；GPU 检索需要安装与 CUDA 匹配的 FAISS
python -m pip install -r examples/anysearch/requirements-retriever-cpu.txt
```

Megatron-LM 和 SGLang 环境配置见 Slime 的
[快速开始](docs/zh/get_started/quick_start.md)，Hugging Face 权重可通过
[`tools/convert_hf_to_torch_dist.py`](tools/convert_hf_to_torch_dist.py) 转换。
AnySearch 使用的环境变量统一列在
[`examples/anysearch/.env.example`](examples/anysearch/.env.example) 中。

## 数据准备

所有机器相关目录均通过环境变量传入。请使用绝对路径；仓库不包含模型、数据、
索引或 checkpoint。

```bash
export DATA_DIR="<absolute-path>/anysearch-data"
bash examples/anysearch/scripts/prepare_all_data.sh
```

脚本从公开的
[FlashRAG 数据集](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets)
准备 NQ 和 HotpotQA 训练数据，以及 NQ、TriviaQA、PopQA、HotpotQA、
2WikiMultiHopQA、MuSiQue 和 Bamboogle 评测数据。Prompt 数据采用以下
Parquet 结构：

```text
question: string
label:    {target: list[string]}
metadata: {dataset: string, split: string, index: int}
```

## 检索服务

AnySearch 使用 E5-base-v2 表示和 Wikipedia 2018 语料上的 FAISS 精确内积检索。

```bash
export RETRIEVER_INDEX="<absolute-path>/e5_Flat.index"
export RETRIEVER_CORPUS="<absolute-path>/wiki-18.jsonl"
export RETRIEVER_MODEL="intfloat/e5-base-v2"

# 仅在已安装兼容的 FAISS GPU 版本时设为 1
export RETRIEVER_FAISS_GPU=0
bash examples/anysearch/scripts/serve_retriever.sh
```

训练和评测启动器会在提交 Ray job 前检查检索服务的健康状态。

## 训练

训练需要提供 SGLang 使用的 Hugging Face checkpoint、对应的 Megatron
torch-distributed checkpoint、准备好的训练 Parquet 和输出目录：

```bash
export MEGATRON_ROOT="<absolute-path>/Megatron-LM"
export HF_CHECKPOINT="<absolute-path>/Qwen2.5-7B-Instruct"
export REF_LOAD="<absolute-path>/Qwen2.5-7B-Instruct_torch_dist"
export TRAIN_DATA="<absolute-path>/anysearch-data/train/nq_hotpotqa_train.parquet"
export SAVE_CHECKPOINT="<absolute-path>/checkpoints/qwen2.5-anysearch"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash examples/anysearch/scripts/run_qwen2.5_7b.sh
```

提供三个模型启动器：

```bash
bash examples/anysearch/scripts/run_qwen2.5_7b.sh
bash examples/anysearch/scripts/run_llama3.1_8b.sh
bash examples/anysearch/scripts/run_qwen3_4b.sh
```

添加 `--dry-run` 可以在不启动 Ray、不连接检索器的情况下检查最终 Slime 命令；
设置 `LOAD_CHECKPOINT` 可以继续已有训练。

## 评测

默认评测覆盖全部七个数据集和搜索预算 `B=1..8`：

```bash
export LOAD_CHECKPOINT="<absolute-path>/checkpoints/qwen2.5-anysearch"
export EVAL_DATA_DIR="<absolute-path>/anysearch-data/eval"
export ANYSEARCH_EVAL_OUTPUT_DIR="<absolute-path>/results/qwen2.5/seed-42"
export EVAL_SEED=42

bash examples/anysearch/scripts/run_eval.sh --model qwen2.5-7b-instruct
```

汇总独立评测并绘制预算曲线：

```bash
RESULTS_DIR="$ANYSEARCH_EVAL_OUTPUT_DIR" \
  bash examples/anysearch/scripts/summarize.sh
```

评测记录 Exact Match、Tool Productivity、平均搜索次数、生成/检索 token 数、
截断数和失败 episode。仓库不附带任何生成的 benchmark 结果。最终指标通过三次
独立评测计算；启动脚本使用 `42`、`43` 和 `44` 三个 seed。

## 配置

标准实验配置位于
[`examples/anysearch/configs/anysearch.yaml`](examples/anysearch/configs/anysearch.yaml)，
消融实验定义位于
[`configs/ablations.yaml`](examples/anysearch/configs/ablations.yaml)。

```bash
# 检查实验 batch 关系和全部 56 项评测配置
python examples/anysearch/scripts/check_config.py

# 生成并运行一个消融配置
bash examples/anysearch/scripts/materialize_ablation.sh \
  no_tool_reward examples/anysearch/outputs/configs/no_tool_reward.yaml
bash examples/anysearch/scripts/run_qwen3_4b.sh \
  --config examples/anysearch/outputs/configs/no_tool_reward.yaml
```

## 引用

如果 AnySearch 对您的研究有所帮助，请引用我们的论文：

```bibtex
@misc{sun2026anysearch,
  title         = {One Policy, Any Budget: Internalizing Budget-Aware Search via Reinforcement Learning},
  author        = {Sun, Xiaowei and Li, Jin and Hong, Yili and Fu, Yikun and Xiao, Yanghua},
  year          = {2026},
  eprint        = {2609.00813},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.00813}
}
```

## 致谢

本项目基于 [Slime](https://github.com/THUDM/slime) 和
[Search-R1](https://github.com/PeterGriffinJin/Search-R1) 构建。感谢这些项目的
作者和贡献者。

## 许可证

仓库代码采用 Apache License 2.0。模型、数据集、Wikipedia passages、索引和生成的
checkpoint 保留各自许可。详见 [NOTICE](NOTICE) 和
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
