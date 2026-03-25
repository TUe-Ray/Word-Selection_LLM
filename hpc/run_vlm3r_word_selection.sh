#!/bin/bash
set -euo pipefail

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
module load hpcx-mpi/2.19

export CONTAINER="${CONTAINER:-/leonardo_work/EUHPC_D32_006/vllm_model/containers/vllm-openai_latest_sandbox}"
export MODEL_PATH="${MODEL_PATH:-/leonardo_work/EUHPC_D32_006/vllm_model/Qwen3-30B-A3B-Instruct-2507}"
export HF_HOME="${HF_HOME:-/leonardo_work/EUHPC_D32_006/vllm_model/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export VLLM_ENABLE_CUDA_COMPATIBILITY=1

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
DATASET_ROOT="${DATASET_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/VLM-3R-DATA}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/artifacts/vlm3r_word_selection_single}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs/word_selection}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LIMIT="${LIMIT:-50}"
SUBSETS="${SUBSETS:-vsibench_train vstibench_train}"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

echo "[runner] start_time=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[runner] model_path=$MODEL_PATH tp_size=$TENSOR_PARALLEL_SIZE dataset_root=$DATASET_ROOT output_dir=$OUTPUT_DIR"

nvidia-smi

if ! singularity exec --nv \
  --bind /leonardo_work/EUHPC_D32_006:/leonardo_work/EUHPC_D32_006 \
  --bind /leonardo_scratch/fast/EUHPC_D32_006:/leonardo_scratch/fast/EUHPC_D32_006 \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  "$CONTAINER" \
  bash -lc "command -v ${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found in container: ${PYTHON_BIN}" >&2
  exit 1
fi

singularity exec --nv \
  --bind /leonardo_work/EUHPC_D32_006:/leonardo_work/EUHPC_D32_006 \
  --bind /leonardo_scratch/fast/EUHPC_D32_006:/leonardo_scratch/fast/EUHPC_D32_006 \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --bind /dev/shm:/dev/shm \
  "$CONTAINER" \
  env HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      VLLM_LOGGING_LEVEL="$VLLM_LOGGING_LEVEL" NCCL_DEBUG="$NCCL_DEBUG" \
      VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/vlm3r_word_selection_pipeline.py" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL_PATH" \
    --limit "$LIMIT" \
    --subsets $SUBSETS

echo "Saved output to: $OUTPUT_DIR"
