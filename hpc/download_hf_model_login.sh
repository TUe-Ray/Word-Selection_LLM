#!/bin/bash
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
MODEL_NAME="$(basename "$MODEL_ID")"
TARGET_DIR="${2:-/leonardo_work/EUHPC_D32_006/vllm_model/$MODEL_NAME}"

mkdir -p "$TARGET_DIR"

python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${MODEL_ID}",
    local_dir="${TARGET_DIR}",
    local_dir_use_symlinks=False,
)
PY

echo "Model downloaded to: ${TARGET_DIR}"
