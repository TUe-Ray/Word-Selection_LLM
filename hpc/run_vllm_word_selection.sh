#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_MODE="${RUN_MODE:-preview}"
RUN_LIMIT="${RUN_LIMIT:-}"
INPUT_JSON="${INPUT_JSON:-$PROJECT_ROOT/spar_234k.json}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-spatial-word-selection}"
CONDA_ROOT="${CONDA_ROOT:-${WORK:-$HOME}/miniconda3}"
MODEL_DIR="${MODEL_DIR:-${FAST:-$HOME}/hf_models/Qwen2.5-14B-Instruct}"
SERVE_MODEL_NAME="${SERVE_MODEL_NAME:-qwen14b}"
VLLM_PORT="${VLLM_PORT:-$((8000 + (${SLURM_JOB_ID:-1} % 1000)))}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/word_selection}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$PROJECT_ROOT/artifacts}"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

if [ ! -f "$INPUT_JSON" ]; then
  echo "[ERROR] Input JSON not found: $INPUT_JSON"
  exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "[ERROR] Model directory not found: $MODEL_DIR"
  echo "Download it first on an internet-connected node using hpc/download_hf_model_login.sh"
  exit 1
fi

if command -v module >/dev/null 2>&1; then
  module load cuda/12.6 || true
  module load cudnn || true
  module load profile/deeplrn || true
fi

export PATH="$CONDA_ROOT/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

RUN_TAG="spar_${RUN_MODE}"
OUTPUT_JSON="${OUTPUT_JSON:-$PROJECT_ROOT/selected_words_${RUN_TAG}_llm.json}"
QUESTIONS_JSONL="${QUESTIONS_JSONL:-$ARTIFACT_DIR/${RUN_TAG}_questions.jsonl}"
VLLM_LOG="$LOG_DIR/vllm_${RUN_TAG}_${SLURM_JOB_ID:-local}.log"

if [ -n "$RUN_LIMIT" ]; then
  MAX_RECORD_ARGS=(--max-records "$RUN_LIMIT")
  SELECT_LIMIT_ARGS=(--max-records-per-file "$RUN_LIMIT")
else
  MAX_RECORD_ARGS=()
  SELECT_LIMIT_ARGS=()
fi

echo "=== Word Selection Job ==="
echo "RUN_MODE: $RUN_MODE"
echo "INPUT_JSON: $INPUT_JSON"
echo "OUTPUT_JSON: $OUTPUT_JSON"
echo "QUESTIONS_JSONL: $QUESTIONS_JSONL"
echo "MODEL_DIR: $MODEL_DIR"
echo "SERVE_MODEL_NAME: $SERVE_MODEL_NAME"
echo "VLLM_PORT: $VLLM_PORT"
echo "HOSTNAME: $(hostname)"

nvidia-smi -L || true

python "$PROJECT_ROOT/scripts/extract_questions_only.py" \
  --input "$INPUT_JSON" \
  --output "$QUESTIONS_JSONL" \
  "${MAX_RECORD_ARGS[@]}"

cleanup() {
  if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    kill "$VLLM_PID" >/dev/null 2>&1 || true
    wait "$VLLM_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVE_MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --dtype bfloat16 \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM to become ready..."
python - <<PY
import json
import time
import urllib.request
import sys

url = "http://127.0.0.1:${VLLM_PORT}/v1/models"
deadline = time.time() + 600
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        print("vLLM ready:", payload.get("data", []))
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        last_error = str(exc)
        time.sleep(5)

print("vLLM did not become ready in time:", last_error)
sys.exit(1)
PY

python "$PROJECT_ROOT/scripts/select_spatial_words.py" \
  --inputs "$INPUT_JSON" \
  --output "$OUTPUT_JSON" \
  --model "$SERVE_MODEL_NAME" \
  --base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
  --api-key EMPTY \
  --report-every 100 \
  "${SELECT_LIMIT_ARGS[@]}"

echo "Finished. Output written to $OUTPUT_JSON"
