from __future__ import annotations

import argparse
import json
from pathlib import Path

from json_stream import iter_json_array
from question_utils import classify_question_type, classify_reference_style, iter_human_questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract cleaned question-only rows from a large JSON array.")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap for debugging")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows_written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for source_index, record in enumerate(iter_json_array(input_path)):
            if args.max_records is not None and source_index >= args.max_records:
                break

            for turn_index, question in iter_human_questions(record):
                row = {
                    "source_file": input_path.name,
                    "source_index": source_index,
                    "id": record.get("id"),
                    "turn_index": turn_index,
                    "question_type": classify_question_type(question),
                    "reference_style": classify_reference_style(question),
                    "question": question,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1

    print(f"Wrote {rows_written} question rows to {output_path}")


if __name__ == "__main__":
    main()
