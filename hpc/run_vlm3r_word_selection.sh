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
PREVIEW_SIZE="${PREVIEW_SIZE:-50}"
SUBSETS="${SUBSETS:-vsibench_train vstibench_train}"
API_BASE="${API_BASE:-http://127.0.0.1:8000/v1}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-900}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-$LOG_DIR/vlm3r_vllm.log}"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

echo "[runner] start_time=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[runner] project_dir=$PROJECT_DIR"
echo "[runner] dataset_root=$DATASET_ROOT"
echo "[runner] output_dir=$OUTPUT_DIR"
echo "[runner] model_path=$MODEL_PATH"
echo "[runner] tp_size=$TENSOR_PARALLEL_SIZE"
echo "[runner] api_base=$API_BASE"

nvidia-smi

if ! singularity exec --nv \
  --bind /leonardo_work/EUHPC_D32_006:/leonardo_work/EUHPC_D32_006 \
  --bind /leonardo_scratch/fast/EUHPC_D32_006:/leonardo_scratch/fast/EUHPC_D32_006 \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  "$CONTAINER" \
  bash -lc "command -v ${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[runner] Python interpreter not found in container: ${PYTHON_BIN}" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[runner] starting local vLLM api server"
singularity exec --nv \
  --bind /leonardo_work/EUHPC_D32_006:/leonardo_work/EUHPC_D32_006 \
  --bind /leonardo_scratch/fast/EUHPC_D32_006:/leonardo_scratch/fast/EUHPC_D32_006 \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --bind /dev/shm:/dev/shm \
  "$CONTAINER" \
  env HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      VLLM_LOGGING_LEVEL="$VLLM_LOGGING_LEVEL" NCCL_DEBUG="$NCCL_DEBUG" \
      VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
  "$PYTHON_BIN" -u -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" \
      --host "$VLLM_HOST" \
      --port "$VLLM_PORT" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      > "$VLLM_LOG_FILE" 2>&1 &
VLLM_PID=$!

echo "[runner] vLLM pid=$VLLM_PID"
echo "[runner] waiting up to ${SERVER_WAIT_SECONDS}s for server readiness"

START_TS=$(date +%s)
while true; do
  if curl -fsS "$API_BASE/models" >/dev/null 2>&1; then
    echo "[runner] vLLM server is ready"
    break
  fi

  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "[runner] vLLM process exited before readiness" >&2
    echo "[runner] last lines from $VLLM_LOG_FILE:" >&2
    tail -n 80 "$VLLM_LOG_FILE" >&2 || true
    exit 1
  fi

  NOW_TS=$(date +%s)
  ELAPSED=$((NOW_TS - START_TS))
  if [[ "$ELAPSED" -ge "$SERVER_WAIT_SECONDS" ]]; then
    echo "[runner] timeout waiting for vLLM server after ${ELAPSED}s" >&2
    echo "[runner] last lines from $VLLM_LOG_FILE:" >&2
    tail -n 80 "$VLLM_LOG_FILE" >&2 || true
    exit 1
  fi

  sleep 5
done

echo "[runner] launching VLM-3R word-selection pipeline"
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
    --api-base "$API_BASE" \
    --limit "$LIMIT" \
    --preview-size "$PREVIEW_SIZE" \
    --subsets $SUBSETS

echo "[runner] saved output to: $OUTPUT_DIR"
