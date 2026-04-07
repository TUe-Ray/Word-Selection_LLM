#!/bin/bash
# Full VLM-3R word-selection run on Leonardo.

#SBATCH --job-name=vlm3r-full
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=/leonardo/home/userexternal/shuang00/Word-Selection_LLM/logs/word_selection/%x_%j.out
#SBATCH --error=/leonardo/home/userexternal/shuang00/Word-Selection_LLM/logs/word_selection/%x_%j.err
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080,lrdn0868,lrdn0808,lrdn0182,lrdn0680,lrdn0831,lrdn0084,lrdn0088,lrdn0186
#SBATCH --exclusive

set -euo pipefail

ROOT_DIR="/leonardo/home/userexternal/shuang00/Word-Selection_LLM"
cd "$ROOT_DIR"

export PROJECT_DIR="$ROOT_DIR"
export LOG_DIR="$ROOT_DIR/logs/word_selection"
export OUTPUT_DIR="$ROOT_DIR/artifacts/vlm3r_word_selection"
export DATASET_ROOT="/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/VLM-3R-DATA"
export MODEL_PATH="/leonardo_work/EUHPC_D32_006/vllm_model/Qwen3-30B-A3B-Instruct-2507"
export CONTAINER="/leonardo_work/EUHPC_D32_006/vllm_model/containers/vllm-openai_latest_sandbox"
export LIMIT=0
export PREVIEW_SIZE=50
export SUBSETS="vsibench_train vstibench_train"
export TENSOR_PARALLEL_SIZE=4
export MAX_MODEL_LEN=32768
export GPU_MEMORY_UTILIZATION=0.95
export API_BASE="http://127.0.0.1:8000/v1"
export SERVER_WAIT_SECONDS=900

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

bash "$ROOT_DIR/hpc/run_vlm3r_word_selection.sh"
