# Anonymous Submission Repository

This repository contains code and experimental pipelines for an anonymous research submission. Identifying metadata has been removed from this README, while setup and execution details needed for reproducibility are preserved.

## Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) as the Python package manager. Install dependencies and set up the virtual environment:

```bash
uv sync --frozen
```

## Run N-gram Pipeline

```bash
uv run experiment --config-path=configs/ngram.yml
```

## Run PCFG Pipeline

```bash
uv run experiment --config-path=configs/pcfg.yml
```

## Run Toy Hierarchy Experiments

The controlled toy experiments from Section 3 are in `toy_model_hierarchy`.
They train matched hierarchical and flat DGP models and measure Hydra, induction,
function-vector, and assumption-probe metrics.

```bash
uv run python toy_model_hierarchy/experiment.py
uv run python toy_model_hierarchy/significance_test.py
```

## Run Checkpoint Evaluation

To reproduce the checkpoint evaluation results, run:

### Step 1: Download Prerequisites

```bash
WANDB_API_KEY=your_api_key uv run python olmo_evaluation/prerequisites/download_wandb_log.py
uv run python olmo_evaluation/prerequisites/download_checkpoints.py
uv run python olmo_evaluation/prerequisites/download_paloma.py
```

### Step 2: Run Evaluation

```bash
uv run olmo_evaluation/multi_gpu_entry.py
```

## Disclaimer

> This repository contains experimental software released to support anonymous evaluation and reproducibility.
