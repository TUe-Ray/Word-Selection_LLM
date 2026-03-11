# Word Selection Pipeline

This project now runs on `spar_234k.json` only.

## 1. Create the conda environment

```powershell
conda env create -f environment.yml
conda activate spatial-word-selection
```

## 2. Inspect cleaned sample questions

```powershell
python scripts/sample_questions.py --inputs spar_234k.json --samples 3
```

## 3. Survey task types and object-reference styles

```powershell
python scripts/survey_question_type` and `reference_styles.py --input spar_234k.json --limit 500 --output artifacts/spar_question_type` and `reference_style_report.json
```

## 4. Extract question-only rows

```powershell
python scripts/extract_questions_only.py --input spar_234k.json --output artifacts/spar_questions_only.jsonl
```

## 5. Quick local preview without LLM

```powershell
python scripts/run_spar_preview_50.py --heuristic-only
```

## 6. Local LLM run against a vLLM endpoint

```powershell
python scripts/run_spar_preview_50.py --model qwen14b --base-url http://127.0.0.1:8000/v1 --api-key EMPTY
```

## 7. Full local run against a vLLM endpoint

```powershell
python scripts/select_spatial_words.py --inputs spar_234k.json --output selected_words_spar_full_llm.json --model qwen14b --base-url http://127.0.0.1:8000/v1 --api-key EMPTY
```

Each output record keeps correspondence with the source sample through:
- `source_file`
- `source_index`
- `id`
- `question_type` and `reference_style`
- `question_selections`

For offline HPC usage, see `README_hpc_word_selection.md` and the scripts in `hpc/`.

