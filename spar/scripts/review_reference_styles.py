from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from json_stream import iter_json_array
from question_utils import classify_reference_style, iter_human_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review reference_style predictions for the first N questions."
    )
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=200,
        help="Number of question rows to export for manual review",
    )
    parser.add_argument(
        "--output",
        default="artifacts/reference_style_review_200.jsonl",
        help="Output JSONL file for manual review",
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "csv"),
        default="jsonl",
        help="Output format for manual review",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    style_counts: Counter[str] = Counter()
    rows = []

    for source_index, record in enumerate(iter_json_array(input_path)):
        for turn_index, question in iter_human_questions(record):
            reference_style = classify_reference_style(question)
            row = {
                "source_file": input_path.name,
                "source_index": source_index,
                "id": record.get("id"),
                "turn_index": turn_index,
                "reference_style": reference_style,
                "question": question,
            }
            rows.append(row)
            style_counts[reference_style] += 1
            rows_written += 1

            if rows_written >= args.limit_questions:
                break

        if rows_written >= args.limit_questions:
            break

    if args.format == "jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "source_file",
                    "source_index",
                    "id",
                    "turn_index",
                    "reference_style",
                    "question",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "format": args.format,
        "rows_written": rows_written,
        "reference_style_counts": dict(style_counts),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
