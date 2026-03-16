#!/bin/bash
#SBATCH --job-name=spar-full
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/word_selection/%x_%j.out
#SBATCH --error=logs/word_selection/%x_%j.err
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080,lrdn0868,lrdn0808,lrdn0182,lrdn0680,lrdn0831,lrdn0084,lrdn0088,lrdn0186
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="$HOME/Word-Selection_LLM/hpc"
RUNNER_SCRIPT="${SCRIPT_DIR}/run_vllm_word_selection.sh"

export PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
export INPUT_JSON="${INPUT_JSON:-$PROJECT_DIR/spar_234k.json}"
export OUTPUT_JSON="${OUTPUT_JSON:-$PROJECT_DIR/selected_words_spar_full_llm.json}"
export RUN_MODE=full
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"

mkdir -p "$PROJECT_DIR/logs/word_selection"

if [[ ! -f "$RUNNER_SCRIPT" ]]; then
  echo "Runner script not found: $RUNNER_SCRIPT" >&2
  exit 1
fi

bash "$RUNNER_SCRIPT"
