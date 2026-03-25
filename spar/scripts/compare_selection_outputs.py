import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two selection output JSON files and export a markdown review."
    )
    parser.add_argument("--old-json", required=True, help="Baseline/old output JSON.")
    parser.add_argument("--new-json", required=True, help="New output JSON.")
    parser.add_argument("--source-json", required=True, help="Original source JSON.")
    parser.add_argument(
        "--out",
        default="artifacts/selection_comparison.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max number of changed rows to print.",
    )
    return parser.parse_args()


def iter_records(path: str) -> Iterator[dict]:
    text_path = Path(path)
    with text_path.open("r", encoding="utf-8") as handle:
        first_char = _read_first_non_whitespace(handle)
        handle.seek(0)
        if first_char == "[":
            yield from _iter_json_array(handle)
        else:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _read_first_non_whitespace(handle) -> str:
    while True:
        ch = handle.read(1)
        if not ch:
            return ""
        if not ch.isspace():
            return ch


def _iter_json_array(handle) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    depth = 0
    in_string = False
    escape = False

    while True:
        chunk = handle.read(1 << 20)
        if not chunk:
            break
        for ch in chunk:
            if not started:
                if ch.isspace() or ch == "," or ch == "[":
                    continue
                if ch == "]":
                    return
                started = True
                buffer = ch
                depth = 0
                in_string = False
                escape = False
            else:
                buffer += ch

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    yield decoder.decode(buffer.strip())
                    buffer = ""
                    started = False


def iter_question_entries(record: dict, source_index: int) -> Iterator[Tuple[int, str]]:
    if isinstance(record.get("conversations"), list):
        for turn_index, turn in enumerate(record["conversations"]):
            role = str(turn.get("from") or turn.get("role") or "").lower()
            text = turn.get("value") or turn.get("content") or ""
            if role in {"human", "user"} and isinstance(text, str):
                yield turn_index, text
    elif isinstance(record.get("question"), str):
        yield 0, record["question"]


def get_turn_index(row: dict) -> int:
    if "turn_index" in row:
        return int(row["turn_index"])
    if "turn_idx" in row:
        return int(row["turn_idx"])
    if "turn" in row:
        return int(row["turn"])
    return 0


def get_key(row: dict) -> Tuple[int, int]:
    return int(row["source_index"]), get_turn_index(row)


def as_list(row: dict, key: str) -> List[str]:
    value = row.get(key, [])
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def load_question_map(source_json: str, needed: set[Tuple[int, int]]) -> Dict[Tuple[int, int], str]:
    qmap: Dict[Tuple[int, int], str] = {}
    max_source = max(k[0] for k in needed) if needed else -1
    for source_index, record in enumerate(iter_records(source_json)):
        if source_index > max_source and len(qmap) == len(needed):
            break
        for turn_index, question in iter_question_entries(record, source_index):
            key = (source_index, turn_index)
            if key in needed:
                qmap[key] = question
        if len(qmap) == len(needed):
            break
    return qmap


def short_question(text: str, max_len: int = 240) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def main() -> int:
    args = parse_args()
    old_rows = json.loads(Path(args.old_json).read_text(encoding="utf-8"))
    new_rows = json.loads(Path(args.new_json).read_text(encoding="utf-8"))

    old_map = {get_key(row): row for row in old_rows}
    new_map = {get_key(row): row for row in new_rows}
    all_keys = sorted(set(old_map) | set(new_map))
    question_map = load_question_map(args.source_json, set(all_keys))

    changed: List[Tuple[Tuple[int, int], dict, dict]] = []
    for key in all_keys:
        old_row = old_map.get(key, {})
        new_row = new_map.get(key, {})
        old_pack = (
            as_list(old_row, "selected_tokens"),
            as_list(old_row, "selected_mentions"),
            as_list(old_row, "spatial_terms"),
            old_row.get("error"),
        )
        new_pack = (
            as_list(new_row, "selected_tokens"),
            as_list(new_row, "selected_mentions"),
            as_list(new_row, "spatial_terms"),
            new_row.get("error"),
        )
        if old_pack != new_pack:
            changed.append((key, old_row, new_row))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Selection Comparison")
    lines.append("")
    lines.append(f"- old: `{args.old_json}`")
    lines.append(f"- new: `{args.new_json}`")
    lines.append(f"- changed rows: `{len(changed)}`")
    lines.append(f"- total aligned rows: `{len(all_keys)}`")
    lines.append("")

    for idx, (key, old_row, new_row) in enumerate(changed[: args.limit], start=1):
        question = question_map.get(key, "[question not found]")
        lines.append(f"## {idx}. source_index={key[0]} turn_index={key[1]}")
        lines.append("")
        lines.append(f"Question: `{short_question(question)}`")
        lines.append("")
        lines.append(f"- old selected_tokens: `{as_list(old_row, 'selected_tokens')}`")
        lines.append(f"- new selected_tokens: `{as_list(new_row, 'selected_tokens')}`")
        lines.append(f"- old selected_mentions: `{as_list(old_row, 'selected_mentions')}`")
        lines.append(f"- new selected_mentions: `{as_list(new_row, 'selected_mentions')}`")
        lines.append(f"- old spatial_terms: `{as_list(old_row, 'spatial_terms')}`")
        lines.append(f"- new spatial_terms: `{as_list(new_row, 'spatial_terms')}`")
        lines.append(f"- old error: `{old_row.get('error')}`")
        lines.append(f"- new error: `{new_row.get('error')}`")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
