# HPC Notes

## Runtime

- container: `/leonardo_work/EUHPC_D32_006/vllm_model/containers/vllm-openai_latest_sandbox`
- model: `/leonardo_work/EUHPC_D32_006/vllm_model/Qwen3-30B-A3B-Instruct-2507`
- mode: direct vLLM Python API batch inference

## Submit

```bash
sbatch hpc/sbatch_spar_preview_vllm.sh
sbatch hpc/sbatch_spar_full_vllm.sh
```

## What the job does

1. loads Leonardo modules
2. enters the singularity sandbox
3. loads the local model with vLLM Python API
4. runs `scripts/select_spatial_words.py`
