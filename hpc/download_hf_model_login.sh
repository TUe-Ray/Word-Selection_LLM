#!/bin/bash
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen2.5-14B-Instruct}"
MODEL_NAME="$(basename "$MODEL_ID")"
TARGET_DIR="${2:-${FAST:-$HOME}/hf_models/$MODEL_NAME}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-spatial-word-selection}"
CONDA_ROOT="${CONDA_ROOT:-${WORK:-$HOME}/miniconda3}"

export PATH="$CONDA_ROOT/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

mkdir -p "$TARGET_DIR"

echo "Downloading $MODEL_ID -> $TARGET_DIR"
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=${MODEL_ID@Q},
    local_dir=${TARGET_DIR@Q},
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY

echo "Model is ready at $TARGET_DIR"
