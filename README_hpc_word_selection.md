# HPC Word Selection

## 1. Survey SPAR task types and object-reference styles locally or on a login node

```bash
python scripts/survey_question_types.py --input spar_234k.json --limit 500 --output artifacts/spar_question_type_report.json
```

## 2. Extract cleaned question-only rows for inspection

```bash
python scripts/extract_questions_only.py --input spar_234k.json --output artifacts/spar_questions_only.jsonl
```

## 3. Download the model on an internet-connected login node

```bash
bash hpc/download_hf_model_login.sh Qwen/Qwen2.5-14B-Instruct "$FAST/hf_models/Qwen2.5-14B-Instruct"
```

The compute node is offline, so `MODEL_DIR` must already exist before you submit the job.

## 4. Quick GPU preview job

```bash
sbatch hpc/sbatch_spar_preview_vllm.sh
```

This job:
- extracts cleaned questions for the first 50 records
- starts local vLLM on the allocated GPU
- writes `selected_words_spar_preview_llm.json`

## 5. Full GPU run

```bash
sbatch hpc/sbatch_spar_full_vllm.sh
```

This job writes `selected_words_spar_full_llm.json`.

## Current prompt strategy

Task type and object-reference style are handled separately:
- `distance_depth`
- `bbox_localization`
- `named_color_reference`
- `object_id_reference`
- `frame_object_reference`
- `relative_position`

The selector first cleans the question text, removes frame index lists and answer-choice sections, classifies the question type, and then sends a type-specific prompt to vLLM.

