#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/VLM-3R-DATA"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/artifacts/vlm3r_word_selection}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
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

mkdir -p "$ROOT_DIR/logs/word_selection"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "$START_VLLM" == "1" ]]; then
  echo "Starting vLLM server for $MODEL_NAME"
  vllm serve "$MODEL_NAME" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    > "$ROOT_DIR/logs/word_selection/vlm3r_vllm.log" 2>&1 &
  VLLM_PID=$!

  echo "Waiting for vLLM server..."
  until curl -fsS "http://$VLLM_HOST:$VLLM_PORT/v1/models" >/dev/null; do
    sleep 5
  done
fi

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
