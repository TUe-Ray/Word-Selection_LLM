#!/bin/bash
# Example:
#   sbatch hpc/sbatch_vlm3r_word_selection_single.sh

#SBATCH --job-name=vlm3r-single
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=/leonardo/home/userexternal/shuang00/Word-Selection_LLM/logs/word_selection/%x_%j.out
#SBATCH --error=/leonardo/home/userexternal/shuang00/Word-Selection_LLM/logs/word_selection/%x_%j.err
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080,lrdn0868,lrdn0808,lrdn0182,lrdn0680,lrdn0831,lrdn0084,lrdn0088,lrdn0186
#SBATCH --exclusive

set -euo pipefail

ROOT_DIR="/leonardo/home/userexternal/shuang00/Word-Selection_LLM"
cd "$ROOT_DIR"

export LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/word_selection}"
mkdir -p "$LOG_DIR"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate word_sel
fi

export PYTHON_BIN="${PYTHON_BIN:-python}"
export DATASET_ROOT="/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/VLM-3R-DATA"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/artifacts/vlm3r_word_selection_single}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
export SUBSETS="${SUBSETS:-vsibench_train vstibench_train}"
export START_VLLM="${START_VLLM:-1}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export API_BASE="${API_BASE:-http://127.0.0.1:8000/v1}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
export LIMIT="${LIMIT:-50}"

bash "$ROOT_DIR/hpc/run_vlm3r_word_selection.sh"
