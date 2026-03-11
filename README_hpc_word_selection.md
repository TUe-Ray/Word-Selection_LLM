# HPC Word Selection

This file is a short HPC-focused companion to [`README.md`](./README.md).

## Environment

Create the base conda environment:

```bash
conda env create -f environment.yml
conda activate spatial-word-selection
```

Install `vllm` separately on Linux / HPC:

```bash
conda activate spatial-word-selection
pip install vllm
```

## Model download

The compute node is offline, so the model must be downloaded on an internet-connected login node first:

```bash
bash hpc/download_hf_model_login.sh \
  Qwen/Qwen2.5-14B-Instruct \
  "$FAST/hf_models/Qwen2.5-14B-Instruct"
```

## Slurm jobs

Preview job:

```bash
sbatch hpc/sbatch_spar_preview_vllm.sh
```

Full job:

```bash
sbatch hpc/sbatch_spar_full_vllm.sh
```

## Output

Each question now contains:
- `selected_tokens`
- `selected_mentions`
- `spatial_terms`

Use `source_index` and `turn_index` as the primary sample key. Do not rely on `id` being unique.
