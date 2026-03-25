#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SINGULARITY_BIN="${SINGULARITY_BIN:-singularity}"
SIF_IMAGE="${SIF_IMAGE:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/VLM-3R-DATA}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/artifacts/vlm3r_word_selection}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/word_selection}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
API_BASE="${API_BASE:-http://127.0.0.1:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
SUBSETS="${SUBSETS:-vsibench_train vstibench_train}"
PREVIEW_SIZE="${PREVIEW_SIZE:-50}"
LIMIT="${LIMIT:-0}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-180}"
START_VLLM="${START_VLLM:-0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

if [[ -z "$SIF_IMAGE" ]]; then
  echo "SIF_IMAGE is not set. Please point it to the singularity image used for vLLM." >&2
  exit 1
fi

SINGULARITY_COMMON_ARGS=(
  exec
  --nv
  --bind "$ROOT_DIR:$ROOT_DIR"
  --pwd "$ROOT_DIR"
)

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "$START_VLLM" == "1" ]]; then
  echo "Starting vLLM server for $MODEL_NAME"
  "$SINGULARITY_BIN" "${SINGULARITY_COMMON_ARGS[@]}" "$SIF_IMAGE" \
    python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    > "$LOG_DIR/vlm3r_vllm.log" 2>&1 &
  VLLM_PID=$!

  echo "Waiting for vLLM server..."
  until curl -fsS "http://$VLLM_HOST:$VLLM_PORT/v1/models" >/dev/null; do
    sleep 5
  done
fi

"$SINGULARITY_BIN" "${SINGULARITY_COMMON_ARGS[@]}" "$SIF_IMAGE" \
  "$PYTHON_BIN" "$ROOT_DIR/scripts/vlm3r_word_selection_pipeline.py" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL_NAME" \
  --api-base "$API_BASE" \
  --api-key "$API_KEY" \
  --preview-size "$PREVIEW_SIZE" \
  --limit "$LIMIT" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --timeout "$TIMEOUT_SECONDS" \
  --subsets $SUBSETS
