# SPAR Word Selection

Question-side word selection for `spar_234k.json`.

## Output

Each question produces:
- `selected_tokens`
- `selected_mentions`
- `spatial_terms`

Use `source_file + source_index + turn_index` as the primary key. Do not rely on `id`.

## Runtime

Default runtime:
- container: `/leonardo_work/EUHPC_D32_006/vllm_model/containers/vllm-openai_latest_sandbox`
- model: `/leonardo_work/EUHPC_D32_006/vllm_model/Qwen3-30B-A3B-Instruct-2507`

Inference mode:
- direct vLLM Python API
- offline batch inference
- no local HTTP server

## Inspect

Sample cleaned questions:

```bash
python scripts/sample_questions.py --inputs spar_234k.json --samples 5
```

Review predicted reference styles:

```bash
python scripts/review_reference_styles.py \
  --input spar_234k.json \
  --limit-questions 200 \
  --format csv \
  --output artifacts/reference_style_review_200.csv
```

## Quick Preview

Inside the sandbox:

```bash
python scripts/run_spar_preview_50.py \
  --model-path /leonardo_work/EUHPC_D32_006/vllm_model/Qwen3-30B-A3B-Instruct-2507
```

## HPC

Submit preview:

```bash
sbatch hpc/sbatch_spar_preview_vllm.sh
```

Submit full run:

```bash
sbatch hpc/sbatch_spar_full_vllm.sh
```

## Files

Core:
- [`scripts/select_spatial_words.py`](./scripts/select_spatial_words.py)
- [`scripts/run_spar_preview_50.py`](./scripts/run_spar_preview_50.py)
- [`scripts/question_utils.py`](./scripts/question_utils.py)

HPC:
- [`hpc/run_vllm_word_selection.sh`](./hpc/run_vllm_word_selection.sh)
- [`hpc/sbatch_spar_preview_vllm.sh`](./hpc/sbatch_spar_preview_vllm.sh)
- [`hpc/sbatch_spar_full_vllm.sh`](./hpc/sbatch_spar_full_vllm.sh)
