# VLM-3R Word Selection on HPC

This is the dedicated word-selection pipeline for `VLM-3R-DATA`.
It does not depend on `spar`, but its HPC launch style follows the old SPAR singularity-based workflow.

## What it does

The pipeline reads `vsibench_train` and/or `vstibench_train`, normalizes the records, exports question-only artifacts, and runs LLM-based word selection through an OpenAI-compatible endpoint such as `vLLM`.

Outputs are written to:

- `artifacts/vlm3r_word_selection/dataset_manifest.json`
- `artifacts/vlm3r_word_selection/normalized_train.jsonl`
- `artifacts/vlm3r_word_selection/normalized_summary.json`
- `artifacts/vlm3r_word_selection/questions_only.jsonl`
- `artifacts/vlm3r_word_selection/preview_50.json`
- `artifacts/vlm3r_word_selection/selected_words.jsonl`
- `artifacts/vlm3r_word_selection/selection_errors.jsonl`

Each row in `selected_words.jsonl` contains:

- `visible_grounded_words`: words that should correspond to directly observable entities or properties in the image/video
- `reasoning_words`: words that are useful for reasoning but are not themselves directly visible entities, such as relations, motion, order, counting, or route constraints
- `selected_words`: backward-compatible union of the two buckets

## Local run

```bash
python scripts/vlm3r_word_selection_pipeline.py \
  --dataset-root VLM-3R-DATA \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --api-base http://127.0.0.1:8000/v1
```

To prepare only:

```bash
python scripts/vlm3r_word_selection_pipeline.py \
  --dataset-root VLM-3R-DATA \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --prepare-only
```

## HPC run

If the cluster job should start `vLLM` itself:

```bash
export SIF_IMAGE=/path/to/your_vllm.sif
sbatch hpc/sbatch_vlm3r_word_selection.sh
```

This full-run job uses:

- `partition=boost_usr_prod`
- `qos=normal`
- `time=10:00:00`
- `gpus-per-node=4`
- `cpus-per-task=32`
- `exclusive`

For a short single/debug run:

```bash
sbatch hpc/sbatch_vlm3r_word_selection_single.sh
```

This single-run job uses:

- `partition=boost_usr_prod`
- `qos=boost_qos_dbg`
- `time=00:30:00`
- `gpus-per-node=4`
- `cpus-per-task=32`
- default `LIMIT=50`

If you already have an endpoint running:

```bash
SIF_IMAGE=/path/to/your_vllm.sif \
PYTHON_BIN=python \
DATASET_ROOT=/path/to/VLM-3R-DATA \
OUTPUT_DIR=/path/to/output \
MODEL_NAME=Qwen/Qwen3-30B-A3B-Instruct-2507 \
API_BASE=http://127.0.0.1:8000/v1 \
START_VLLM=0 \
bash hpc/run_vlm3r_word_selection.sh
```

## Notes

- `vsibench_train` and `vstibench_train` are both supported.
- The selector reads from the normalized question text and returns `visible_grounded_words`, `reasoning_words`, `selected_words`, plus a short justification.
- If you want to rerun after a partial job, add `--resume` in the Python command.
- HPC scripts assume a singularity image is used for both `vLLM` startup and Python execution inside the same container.
