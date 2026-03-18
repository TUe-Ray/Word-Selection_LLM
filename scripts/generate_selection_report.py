import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from question_utils import classify_question_type, classify_reference_style


LEGACY_KEYS = ("selected_tokens", "selected_mentions", "spatial_terms")
SCHEMA_KEYS = (
    "grounded_mentions",
    "spatial_relations",
    "geometric_constraints",
    "viewpoint_constraints",
    "answer_constraints",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an automatic Markdown quality report from selection JSON."
    )
    parser.add_argument(
        "--input",
        default="selected_words_spar_preview50.json",
        help="Path to selection output JSON.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/selection_quality_review_preview50_auto.md",
        help="Path to generated markdown report.",
    )
    parser.add_argument(
        "--source-json",
        default=None,
        help="Optional original dataset JSON used to restore question text by source_index/turn_index.",
    )
    parser.add_argument(
        "--suspicious-limit",
        type=int,
        default=15,
        help="Number of suspicious rows to print.",
    )
    parser.add_argument(
        "--good-limit",
        type=int,
        default=8,
        help="Number of good rows to print.",
    )
    parser.add_argument(
        "--include-question",
        action="store_true",
        help="Include a short question snippet in suspicious rows.",
    )
    return parser.parse_args()


def iter_records(path: str):
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


def _iter_json_array(handle):
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


def iter_question_entries(record: Dict[str, Any]):
    if isinstance(record.get("conversations"), list):
        for turn_index, turn in enumerate(record["conversations"]):
            role = str(turn.get("from") or turn.get("role") or "").lower()
            text = turn.get("value") or turn.get("content") or ""
            if role in {"human", "user"} and isinstance(text, str):
                yield turn_index, text
    elif isinstance(record.get("question"), str):
        yield 0, record["question"]


def get_turn_index(row: Dict[str, Any]) -> int:
    if "turn_index" in row:
        return int(row["turn_index"])
    if "turn_idx" in row:
        return int(row["turn_idx"])
    if "turn" in row:
        return int(row["turn"])
    return 0


def attach_questions(data: List[Dict[str, Any]], source_json: str) -> None:
    needed = {
        (int(row.get("source_index", 0)), get_turn_index(row))
        for row in data
        if "source_index" in row
    }
    if not needed:
        return

    question_map = {}
    for source_index, record in enumerate(iter_records(source_json)):
        for turn_index, question in iter_question_entries(record):
            key = (source_index, turn_index)
            if key in needed:
                question_map[key] = question
        if len(question_map) == len(needed):
            break

    for row in data:
        key = (int(row.get("source_index", 0)), get_turn_index(row))
        if key in question_map and not row.get("question"):
            row["question"] = question_map[key]


def as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    return []


def has_schema_fields(row: Dict[str, Any]) -> bool:
    return any(key in row for key in SCHEMA_KEYS)


def get_question_text(row: Dict[str, Any]) -> str:
    return str(row.get("question") or row.get("cleaned_question") or "").strip()


def get_reference_style(row: Dict[str, Any]) -> str:
    value = str(row.get("reference_style") or "").strip()
    if value:
        return value
    question = get_question_text(row)
    return classify_reference_style(question) if question else "unknown"


def get_question_type(row: Dict[str, Any]) -> str:
    value = str(row.get("question_type") or "").strip()
    if value:
        return value
    question = get_question_text(row)
    return classify_question_type(question) if question else "unknown"


def get_tokens(row: Dict[str, Any]) -> List[str]:
    return as_list(row.get("selected_tokens"))


def get_mentions(row: Dict[str, Any]) -> List[str]:
    if has_schema_fields(row):
        return as_list(row.get("grounded_mentions"))
    return as_list(row.get("selected_mentions"))


def get_spatial_signal(row: Dict[str, Any]) -> List[str]:
    if has_schema_fields(row):
        values: List[str] = []
        for key in ("spatial_relations", "geometric_constraints", "viewpoint_constraints"):
            values.extend(as_list(row.get(key)))
        return values
    return as_list(row.get("spatial_terms"))


def get_answer_constraints(row: Dict[str, Any]) -> List[str]:
    return as_list(row.get("answer_constraints"))


def count_noise_tokens(tokens: List[str]) -> int:
    noise = {
        "please",
        "answer",
        "question",
        "calculate",
        "describe",
        "choose",
        "select",
        "correct",
        "option",
        "options",
        "system",
        "assistant",
        "user",
    }
    return sum(1 for t in tokens if t.lower() in noise)


def is_suspicious(row: Dict[str, Any]) -> bool:
    style = get_reference_style(row)
    qtype = get_question_type(row)
    tokens = get_tokens(row)
    mentions = get_mentions(row)
    spatial = get_spatial_signal(row)

    if style in {"frame_object_reference", "object_id_reference"} and len(mentions) == 0:
        return True
    if has_schema_fields(row):
        if len(mentions) == 0:
            return True
    elif len(tokens) == 0:
        return True
    if qtype in {"relative_position", "distance_depth", "bbox_localization"} and len(spatial) == 0:
        return True
    if tokens and count_noise_tokens(tokens) >= max(2, len(tokens) // 2):
        return True
    return False


def is_good(row: Dict[str, Any]) -> bool:
    style = get_reference_style(row)
    qtype = get_question_type(row)
    tokens = get_tokens(row)
    mentions = get_mentions(row)
    spatial = get_spatial_signal(row)

    if has_schema_fields(row):
        if len(mentions) == 0:
            return False
    elif len(tokens) == 0:
        return False
    if style in {"frame_object_reference", "object_id_reference"} and len(mentions) == 0:
        return False
    if qtype in {"relative_position", "distance_depth", "bbox_localization"} and len(spatial) == 0:
        return False
    return True


def row_desc(row: Dict[str, Any]) -> str:
    source_index = row.get("source_index")
    turn_index = get_turn_index(row)
    style = get_reference_style(row)
    qtype = get_question_type(row)
    mentions = get_mentions(row)
    spatial = get_spatial_signal(row)
    if has_schema_fields(row):
        answer = get_answer_constraints(row)
        return (
            f"- source_index={source_index}, turn_index={turn_index}, style={style}, type={qtype}, "
            f"grounded={len(mentions)}, spatial={len(spatial)}, answer={len(answer)}"
        )

    tokens = get_tokens(row)
    return (
        f"- source_index={source_index}, turn_index={turn_index}, style={style}, type={qtype}, "
        f"tokens={len(tokens)}, mentions={len(mentions)}, spatial={len(spatial)}"
    )


def question_snippet(row: Dict[str, Any], max_len: int = 180) -> str:
    raw = get_question_text(row)
    if not raw:
        return ""
    compact = " ".join(raw.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of rows.")
    if args.source_json:
        attach_questions(data, args.source_json)

    total = len(data)
    style_counter = Counter(get_reference_style(r) for r in data)
    type_counter = Counter(get_question_type(r) for r in data)

    schema_rows = [r for r in data if has_schema_fields(r)]
    legacy_rows = [r for r in data if not has_schema_fields(r)]

    missing_tokens = sum(1 for r in legacy_rows if len(get_tokens(r)) == 0)
    missing_mentions = sum(1 for r in data if len(get_mentions(r)) == 0)
    missing_spatial = sum(1 for r in data if len(get_spatial_signal(r)) == 0)
    missing_answer_constraints = sum(1 for r in schema_rows if len(get_answer_constraints(r)) == 0)
    empty_all = sum(
        1
        for r in data
        if len(get_mentions(r)) == 0
        and len(get_spatial_signal(r)) == 0
        and (len(get_tokens(r)) == 0 if not has_schema_fields(r) else len(get_answer_constraints(r)) == 0)
    )

    frame_rows = [r for r in data if get_reference_style(r) == "frame_object_reference"]
    id_rows = [r for r in data if get_reference_style(r) == "object_id_reference"]
    frame_empty_mentions = sum(1 for r in frame_rows if len(get_mentions(r)) == 0)
    id_empty_mentions = sum(1 for r in id_rows if len(get_mentions(r)) == 0)

    suspicious_rows = [r for r in data if is_suspicious(r)]
    good_rows = [r for r in data if is_good(r)]

    lines: List[str] = []
    lines.append("# Selection Quality Report (Auto)")
    lines.append("")
    lines.append(f"- Input: `{args.input}`")
    if args.source_json:
        lines.append(f"- Source dataset: `{args.source_json}`")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total rows: **{total}**")
    lines.append(f"- Rows using grounded schema: **{len(schema_rows)}**")
    lines.append(f"- Rows using legacy schema: **{len(legacy_rows)}**")
    lines.append(f"- Rows with all outputs empty: **{empty_all}**")
    if legacy_rows:
        lines.append(f"- Legacy rows missing `selected_tokens`: **{missing_tokens}**")
    lines.append(f"- Rows missing grounded mentions: **{missing_mentions}**")
    lines.append(f"- Rows missing spatial signal: **{missing_spatial}**")
    if schema_rows:
        lines.append(f"- Grounded-schema rows missing `answer_constraints`: **{missing_answer_constraints}**")
    lines.append("")
    lines.append("## Distribution")
    lines.append("")
    lines.append("### By reference_style")
    for k, v in sorted(style_counter.items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### By question_type")
    for k, v in sorted(type_counter.items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Sanity Checks")
    lines.append("")
    lines.append(
        f"- `frame_object_reference` rows: {len(frame_rows)}, empty mentions: {frame_empty_mentions}"
    )
    lines.append(
        f"- `object_id_reference` rows: {len(id_rows)}, empty mentions: {id_empty_mentions}"
    )
    lines.append("")
    lines.append("## Suspicious Rows")
    lines.append("")
    lines.append(
        f"Top {min(args.suspicious_limit, len(suspicious_rows))} rows that likely need manual check:"
    )
    for row in suspicious_rows[: args.suspicious_limit]:
        lines.append(row_desc(row))
        if args.include_question:
            snippet = question_snippet(row)
            if snippet:
                lines.append(f"  - question: {snippet}")
    if not suspicious_rows:
        lines.append("- None")
    lines.append("")
    lines.append("## Good Rows")
    lines.append("")
    lines.append(f"Top {min(args.good_limit, len(good_rows))} rows that look consistent:")
    for row in good_rows[: args.good_limit]:
        lines.append(row_desc(row))
    if not good_rows:
        lines.append("- None")
    lines.append("")
    lines.append("## Decision Guide")
    lines.append("")
    lines.append("- Pass if suspicious rows are rare and mostly borderline formatting cases.")
    lines.append("- Borderline if suspicious rows are frequent but fixable by prompt tuning.")
    lines.append("- Fail if core identity/spatial signal is often missing.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
