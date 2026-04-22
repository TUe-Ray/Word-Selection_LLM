from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VSIBENCH_FILES = [
    "merged_qa_route_plan_train.json",
    "merged_qa_scannet_train.json",
    "merged_qa_scannetpp_train.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split selected_words.jsonl into the three vsibench_train source files."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selected-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_records(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "items", "records", "questions"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def load_selected_rows(selected_path: Path) -> dict[str, dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    with selected_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("subset") != "vsibench_train":
                continue
            row_id = str(row["id"])
            rows_by_id[row_id] = row
    return rows_by_id


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    vsibench_root = args.dataset_root / "vsibench_train"
    rows_by_id = load_selected_rows(args.selected_path)

    for source_name in VSIBENCH_FILES:
        source_path = vsibench_root / source_name
        ordered_rows: list[dict[str, Any]] = []
        missing_ids: list[str] = []
        for record in iter_records(source_path):
            record_id = str(record.get("id"))
            row = rows_by_id.get(record_id)
            if row is None:
                missing_ids.append(record_id)
                continue
            ordered_rows.append(row)

        if missing_ids:
            raise ValueError(
                f"Missing {len(missing_ids)} selected rows for {source_name}; first missing id: {missing_ids[0]}"
            )

        output_path = args.output_dir / source_path.stem / "selected_words.jsonl"
        write_jsonl(output_path, ordered_rows)
        print(f"{source_name}\t{len(ordered_rows)}\t{output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
