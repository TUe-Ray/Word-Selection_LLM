from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from json_stream import iter_json_array
from question_utils import classify_question_type, classify_reference_style, iter_human_questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Survey SPAR question task types and object-reference styles.")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--limit", type=int, default=500, help="Number of source records to inspect")
    parser.add_argument("--examples-per-group", type=int, default=3, help="How many examples to keep per group")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    task_type_counts = Counter()
    reference_style_counts = Counter()
    combination_counts = Counter()
    task_examples = {}
    reference_examples = {}
    combination_examples = {}
    total_questions = 0

    records_scanned = 0
    for source_index, record in enumerate(iter_json_array(Path(args.input))):
        if source_index >= args.limit:
            break
        records_scanned = source_index + 1

        for turn_index, question in iter_human_questions(record):
            question_type = classify_question_type(question)
            reference_style = classify_reference_style(question)
            combination_key = f"{question_type}__{reference_style}"

            task_type_counts[question_type] += 1
            reference_style_counts[reference_style] += 1
            combination_counts[combination_key] += 1
            total_questions += 1

            task_examples.setdefault(question_type, [])
            if len(task_examples[question_type]) < args.examples_per_group:
                task_examples[question_type].append(
                    {
                        "source_index": source_index,
                        "id": record.get("id"),
                        "turn_index": turn_index,
                        "question": question,
                    }
                )

            reference_examples.setdefault(reference_style, [])
            if len(reference_examples[reference_style]) < args.examples_per_group:
                reference_examples[reference_style].append(
                    {
                        "source_index": source_index,
                        "id": record.get("id"),
                        "turn_index": turn_index,
                        "question": question,
                    }
                )

            combination_examples.setdefault(combination_key, [])
            if len(combination_examples[combination_key]) < args.examples_per_group:
                combination_examples[combination_key].append(
                    {
                        "source_index": source_index,
                        "id": record.get("id"),
                        "turn_index": turn_index,
                        "question": question,
                    }
                )

    report = {
        "input": args.input,
        "records_scanned": records_scanned,
        "total_questions": total_questions,
        "task_type_counts": dict(task_type_counts),
        "reference_style_counts": dict(reference_style_counts),
        "combination_counts": dict(combination_counts),
        "task_examples": task_examples,
        "reference_examples": reference_examples,
        "combination_examples": combination_examples,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
