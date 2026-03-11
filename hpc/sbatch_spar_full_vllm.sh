#!/bin/bash
#SBATCH --job-name=ws_spar_full
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_bprod
#SBATCH --output=logs/word_selection/%x_%j.out
#SBATCH --error=logs/word_selection/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

export RUN_MODE=full
unset RUN_LIMIT
export MODEL_DIR="${MODEL_DIR:-$FAST/hf_models/Qwen2.5-14B-Instruct}"
export SERVE_MODEL_NAME="${SERVE_MODEL_NAME:-qwen14b}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_vllm_word_selection.sh"
