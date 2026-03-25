# SPAR 詞彙篩選說明

這個專案只處理 `spar_234k.json` 的 question，不處理 answer。

## 輸出格式

每個 question 會輸出三欄：
- `selected_tokens`
- `selected_mentions`
- `spatial_terms`

主鍵請用 `source_file + source_index + turn_index`，不要依賴 `id`。

## 現在的推理方式

目前已改成：
- direct vLLM Python API batch mode
- offline batch inference
- 不再使用 `vllm serve`
- 不再依賴 `http://127.0.0.1:8000/v1/...`

預設 runtime：
- container: `/leonardo_work/EUHPC_D32_006/vllm_model/containers/vllm-openai_latest_sandbox`
- model: `/leonardo_work/EUHPC_D32_006/vllm_model/Qwen3-30B-A3B-Instruct-2507`

## 資料檢查

抽樣看 question：

```bash
python scripts/sample_questions.py --inputs spar_234k.json --samples 5
```

檢查 `reference_style` 分類結果：

```bash
python scripts/review_reference_styles.py \
  --input spar_234k.json \
  --limit-questions 200 \
  --format csv \
  --output artifacts/reference_style_review_200.csv
```

## Preview

如果你已經在 sandbox 內：

```bash
python scripts/run_spar_preview_50.py \
  --model-path /leonardo_work/EUHPC_D32_006/vllm_model/Qwen3-30B-A3B-Instruct-2507
```

## HPC Step by Step

### 1. 提交 preview job

```bash
sbatch hpc/sbatch_spar_preview_vllm.sh
```

### 2. 提交 full job

```bash
sbatch hpc/sbatch_spar_full_vllm.sh
```

這兩個 job 會：
- 載入 Leonardo modules
- 在 singularity sandbox 內直接載入 vLLM Python API
- 用 batch 方式跑 `scripts/select_spatial_words.py`

## 重要說明

- 不再需要本地 HTTP server。
- 不再需要 `/v1/models` readiness check。
- vLLM 只做推理，不會更新模型權重。
- 每個 question 都是獨立樣本，不會保留前一題記憶。

## 主要檔案

- [`scripts/select_spatial_words.py`](./scripts/select_spatial_words.py)
- [`scripts/run_spar_preview_50.py`](./scripts/run_spar_preview_50.py)
- [`scripts/question_utils.py`](./scripts/question_utils.py)
- [`hpc/run_vllm_word_selection.sh`](./hpc/run_vllm_word_selection.sh)
- [`hpc/sbatch_spar_preview_vllm.sh`](./hpc/sbatch_spar_preview_vllm.sh)
- [`hpc/sbatch_spar_full_vllm.sh`](./hpc/sbatch_spar_full_vllm.sh)
