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
INPUT_JSON="${INPUT_JSON:-$PROJECT_DIR/spar_234k.json}"
RUN_MODE="${RUN_MODE:-preview}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$RUN_MODE" == "preview" ]]; then
  OUTPUT_JSON="${OUTPUT_JSON:-$PROJECT_DIR/selected_grounded_schema_spar_preview50.json}"
  EXTRA_ARGS=(--max-records-per-file 50)
elif [[ "$RUN_MODE" == "full" ]]; then
  OUTPUT_JSON="${OUTPUT_JSON:-$PROJECT_DIR/selected_grounded_schema_spar_full_llm.json}"
  EXTRA_ARGS=()
else
  echo "Unsupported RUN_MODE: $RUN_MODE" >&2
  exit 1
fi

echo "[runner] start_time=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[runner] mode=$RUN_MODE model_path=$MODEL_PATH tp_size=$TENSOR_PARALLEL_SIZE batch_size=$BATCH_SIZE input=$INPUT_JSON output=$OUTPUT_JSON"

nvidia-smi

if ! singularity exec --nv \
  --bind /leonardo_work/EUHPC_D32_006:/leonardo_work/EUHPC_D32_006 \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  "$CONTAINER" \
  bash -lc "command -v ${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found in container: ${PYTHON_BIN}" >&2
  echo "Try setting PYTHON_BIN=python or PYTHON_BIN=/usr/bin/python3 in sbatch env." >&2
  exit 1
fi

singularity exec --nv \
  --bind /leonardo_work/EUHPC_D32_006:/leonardo_work/EUHPC_D32_006 \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --bind /dev/shm:/dev/shm \
  "$CONTAINER" \
  env HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      VLLM_LOGGING_LEVEL="$VLLM_LOGGING_LEVEL" NCCL_DEBUG="$NCCL_DEBUG" \
      VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/select_spatial_grounded_schema_llm_only.py" \
    --inputs "$INPUT_JSON" \
    --output "$OUTPUT_JSON" \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --batch-size "$BATCH_SIZE" \
    --hf-home "$HF_HOME" \
    --enforce-eager \
    "${EXTRA_ARGS[@]}"

echo "Saved output to: $OUTPUT_JSON"
