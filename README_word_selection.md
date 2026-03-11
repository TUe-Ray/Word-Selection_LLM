# SPAR 詞彙選擇流程說明

本專案目前只針對 `spar_234k.json` 做 question 端的詞彙選擇，不處理 answer。

目前每個 question 會輸出三欄：
- `selected_tokens`：拆開的 token 或短詞
- `selected_mentions`：完整物體指代表達
- `spatial_terms`：純空間、方向、距離、視角相關詞

主定位鍵不是 `id`，而是：
- `source_file`
- `source_index`
- `turn_index`

這是刻意設計的，因為原始 JSON 可能有重複 `id`。

## 1. 推薦模型

建議先用：
- `Qwen/Qwen2.5-14B-Instruct`

原因：
- 品質比 7B 穩
- 單張 A100 通常可跑
- 適合這種結構化抽取任務

如果你的 A100 是 80GB，而且想再往上試，可以再考慮：
- `Qwen/Qwen2.5-32B-Instruct`

不建議一開始就上更大的模型，因為你目前先要驗證 prompt 和資料格式，不是先追求最大模型。

## 2. Conda 環境

基礎環境：

```powershell
conda env create -f environment.yml
conda activate spatial-word-selection
```

`environment.yml` 目前會安裝：
- `openai`
- `huggingface_hub`
- `tqdm`

## 3. 要不要另外安裝 vLLM

要。

`vLLM` 沒有放進 `environment.yml`，這是刻意的，因為：
- 你現在本機是 Windows 環境
- `vLLM` 主要是 Linux + NVIDIA GPU 的推理環境
- 直接放進基礎環境容易讓本機建環境失敗

所以建議是：
- 本機 / 資料整理：只用 `environment.yml`
- HPC / Linux GPU 節點：額外安裝 `vllm`

在 HPC 的 Linux conda 環境裡：

```bash
conda activate spatial-word-selection
pip install vllm
```

## 4. 先檢查 question 格式

抽樣看前幾筆清洗後的 question：

```powershell
python scripts/sample_questions.py --inputs spar_234k.json --samples 5
```

檢查物體指代方式分類：

```powershell
python scripts/review_reference_styles.py --input spar_234k.json --limit-questions 200 --format csv --output artifacts/reference_style_review_200.csv
```

你可以直接打開：
- [reference_style_review_200.csv](C:/Users/User1/OneDrive%20-%20TU%20Eindhoven/Graduation%20Project/antigravity/Word%20Selection_LLM/artifacts/reference_style_review_200.csv)

## 5. 不用 LLM 先測流程

先只跑 heuristic，確認輸出格式：

```powershell
python scripts/select_spatial_words.py --inputs spar_234k.json --output selected_words_spar_preview50.json --max-records-per-file 50 --heuristic-only
```

## 6. HPC 離線流程

你的 compute node 沒有外網，所以流程要分兩步：

### 6.1 在有網路的 login node 下載模型

```bash
bash hpc/download_hf_model_login.sh Qwen/Qwen2.5-14B-Instruct "$FAST/hf_models/Qwen2.5-14B-Instruct"
```

### 6.2 提交 GPU 作業

快速測試前 50 筆：

```bash
sbatch hpc/sbatch_spar_preview_vllm.sh
```

全量跑：

```bash
sbatch hpc/sbatch_spar_full_vllm.sh
```

這兩個 `sbatch` 都會做：
- 啟動本地 vLLM server
- 呼叫 `scripts/select_spatial_words.py`
- 輸出 JSON 結果

## 7. 如果你要自己手動啟 vLLM

在 Linux / HPC 上：

```bash
vllm serve "$FAST/hf_models/Qwen2.5-14B-Instruct" \
  --served-model-name qwen14b \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 1
```

然後再跑：

```bash
python scripts/select_spatial_words.py \
  --inputs spar_234k.json \
  --output selected_words_spar_full_llm.json \
  --model qwen14b \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY
```

## 8. 現在輸出長什麼樣

每個 sample 會保留：
- `source_file`
- `source_index`
- `id`

每個 question 會保留：
- `turn_index`
- `question`
- `question_type`
- `reference_style`
- `selected_tokens`
- `selected_mentions`
- `spatial_terms`
- `method`

## 9. 相關檔案

主要腳本：
- [scripts/select_spatial_words.py](C:/Users/User1/OneDrive%20-%20TU%20Eindhoven/Graduation%20Project/antigravity/Word%20Selection_LLM/scripts/select_spatial_words.py)
- [scripts/question_utils.py](C:/Users/User1/OneDrive%20-%20TU%20Eindhoven/Graduation%20Project/antigravity/Word%20Selection_LLM/scripts/question_utils.py)
- [scripts/review_reference_styles.py](C:/Users/User1/OneDrive%20-%20TU%20Eindhoven/Graduation%20Project/antigravity/Word%20Selection_LLM/scripts/review_reference_styles.py)

HPC 腳本：
- [hpc/download_hf_model_login.sh](C:/Users/User1/OneDrive%20-%20TU%20Eindhoven/Graduation%20Project/antigravity/Word%20Selection_LLM/hpc/download_hf_model_login.sh)
- [hpc/sbatch_spar_preview_vllm.sh](C:/Users/User1/OneDrive%20-%20TU%20Eindhoven/Graduation%20Project/antigravity/Word%20Selection_LLM/hpc/sbatch_spar_preview_vllm.sh)
- [hpc/sbatch_spar_full_vllm.sh](C:/Users/User1/OneDrive%20-%20TU%20Eindhoven/Graduation%20Project/antigravity/Word%20Selection_LLM/hpc/sbatch_spar_full_vllm.sh)
