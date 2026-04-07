#!/bin/bash
#SBATCH --job-name=vllm_smoketest
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --account=EUHPC_D32_006
#SBATCH --exclusive
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080,lrdn0868,lrdn0808,lrdn0182,lrdn0680,lrdn0831,lrdn0084,lrdn0088,lrdn0186

set -euo pipefail

echo "-------- Note --------"
echo "  note: vllm smoke test"

module purge || true
module load profile/deeplrn
module load cineca-ai/4.3.0
module load hpcx-mpi/2.19

module list
export CONTAINER=/leonardo_work/EUHPC_D32_006/vllm_model/containers/vllm-openai_latest_sandbox
export MODEL_PATH=/leonardo_work/EUHPC_D32_006/vllm_model/Qwen3-30B-A3B-Instruct-2507
export HF_HOME=/leonardo_work/EUHPC_D32_006/vllm_model/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_PORT=8000

echo "Node: $(hostname)"
nvidia-smi

test -d "$CONTAINER" || { echo "Missing container: $CONTAINER"; exit 1; }
test -d "$MODEL_PATH" || { echo "Missing model dir: $MODEL_PATH"; exit 1; }

singularity exec --nv \
  --bind /leonardo_work/EUHPC_D32_006:/leonardo_work/EUHPC_D32_006 \
  --bind /dev/shm:/dev/shm \
  "$CONTAINER" \
  vllm serve "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --served-model-name qwen3-30b-a3b > server.log 2>&1 &

SERVER_PID=$!

echo "Waiting for server to start..."
sleep 120

echo "=== /v1/models ==="
curl http://127.0.0.1:${VLLM_PORT}/v1/models

echo
echo "=== chat test ==="
curl http://127.0.0.1:${VLLM_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b-a3b",
    "messages": [
      {"role": "user", "content": "請用一句話介紹你自己。"}
    ],
    "temperature": 0.0,
    "max_tokens": 64
  }'

echo
echo "=== tail server.log ==="
tail -n 100 server.log || true

kill $SERVER_PID || true
wait $SERVER_PID || true