# SPAR Word Selection Pipeline

This repository builds structured word-selection outputs for the **question** side of `spar_234k.json`.

The pipeline extracts three parallel fields for each question:
- `selected_tokens`: token-level or short-phrase signals
- `selected_mentions`: full object mentions / referring expressions
- `spatial_terms`: spatial, directional, viewpoint, distance, or measurement terms

The primary lookup key is **not** `id`, because the source JSON may contain duplicate IDs. Use:
- `source_file`
- `source_index`
- `turn_index`

## Recommended model

Start with:
- `Qwen/Qwen2.5-14B-Instruct`

Why:
- better extraction quality than 7B
- usually practical on a single A100
- appropriate for structured extraction rather than free-form generation

If you have an A100 80GB and want a stronger baseline, try:
- `Qwen/Qwen2.5-32B-Instruct`

## Environment setup

Create the base conda environment:

```bash
conda env create -f environment.yml
conda activate spatial-word-selection
```

The base environment installs:
- `openai`
- `huggingface_hub`
- `tqdm`

## Does conda also install vLLM?

No.

`vLLM` is intentionally **not** included in `environment.yml` because:
- local development here may happen on Windows
- `vLLM` is mainly for Linux + NVIDIA GPU inference
- putting it into the base environment would make the setup less portable

Install `vllm` separately only on the Linux/HPC environment where you will actually serve the model:

```bash
conda activate spatial-word-selection
pip install vllm
```

## Inspect the data first

Sample cleaned questions:

```bash
python scripts/sample_questions.py --inputs spar_234k.json --samples 5
```

Review predicted reference styles for manual inspection:

```bash
python scripts/review_reference_styles.py \
  --input spar_234k.json \
  --limit-questions 200 \
  --format csv \
  --output artifacts/reference_style_review_200.csv
```

This produces:
- [`artifacts/reference_style_review_200.csv`](./artifacts/reference_style_review_200.csv)

## Quick local preview without LLM

Use heuristic extraction first to validate the output structure:

```bash
python scripts/select_spatial_words.py \
  --inputs spar_234k.json \
  --output selected_words_spar_preview50.json \
  --max-records-per-file 50 \
  --heuristic-only
```

## Manual vLLM serving

On Linux / HPC:

```bash
vllm serve "$FAST/hf_models/Qwen2.5-14B-Instruct" \
  --served-model-name qwen14b \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 1
```

Then run:

```bash
python scripts/select_spatial_words.py \
  --inputs spar_234k.json \
  --output selected_words_spar_full_llm.json \
  --model qwen14b \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY
```

## Offline HPC workflow

The compute node is offline, so the workflow has to be split:

### 1. Download the model on an internet-connected login node

```bash
bash hpc/download_hf_model_login.sh \
  Qwen/Qwen2.5-14B-Instruct \
  "$FAST/hf_models/Qwen2.5-14B-Instruct"
```

### 2. Submit a GPU preview job

```bash
sbatch hpc/sbatch_spar_preview_vllm.sh
```

### 3. Submit the full GPU run

```bash
sbatch hpc/sbatch_spar_full_vllm.sh
```

These jobs:
- extract cleaned question text
- start a local `vLLM` server on the allocated GPU node
- run `scripts/select_spatial_words.py`

## Output structure

Each sample keeps:
- `source_file`
- `source_index`
- `id`

Each question keeps:
- `turn_index`
- `question`
- `question_type`
- `reference_style`
- `selected_tokens`
- `selected_mentions`
- `spatial_terms`
- `method`

## Main files

Core scripts:
- [`scripts/select_spatial_words.py`](./scripts/select_spatial_words.py)
- [`scripts/question_utils.py`](./scripts/question_utils.py)
- [`scripts/review_reference_styles.py`](./scripts/review_reference_styles.py)

HPC scripts:
- [`hpc/download_hf_model_login.sh`](./hpc/download_hf_model_login.sh)
- [`hpc/sbatch_spar_preview_vllm.sh`](./hpc/sbatch_spar_preview_vllm.sh)
- [`hpc/sbatch_spar_full_vllm.sh`](./hpc/sbatch_spar_full_vllm.sh)
