from __future__ import annotations

import argparse
import json
from pathlib import Path

from json_stream import iter_json_array
from question_utils import classify_question_type, classify_reference_style, iter_human_questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample question entries from large JSON arrays.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON files")
    parser.add_argument("--samples", type=int, default=3, help="Number of records to sample per file")
    args = parser.parse_args()

    for file_path in args.inputs:
        path = Path(file_path)
        print(f"\n=== {path.name} ===")

        count = 0
        for index, record in enumerate(iter_json_array(path)):
            output = {
                "source_index": index,
                "id": record.get("id"),
                "questions": [
                    {
                        "turn_index": turn,
                        "question_type": classify_question_type(question),
                        "reference_style": classify_reference_style(question),
                        "question": question,
                    }
                    for turn, question in iter_human_questions(record)
                ],
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
            count += 1
            if count >= args.samples:
                break


if __name__ == "__main__":
    main()
