from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROMPT_TEMPLATE = """You are selecting the minimum set of words or short phrases that are essential for solving a VLM-3R spatial reasoning question.

Dataset context:
- `vsibench_train` questions are mostly static spatial understanding, route planning, landmark navigation, counting, size comparison, order of appearance, room size, and object distance/direction.
- `vstibench_train` questions are mostly temporal-spatial understanding from video, including camera displacement, camera movement direction, camera-object absolute/relative distance, and object-object relative position.

Your goal:
- Split the useful words into two groups.
- Group 1: words or short phrases that refer to things or properties that can be visually grounded in the image or video and are useful for reasoning.
- Group 2: words or short phrases that are not directly visible themselves, but are necessary for reasoning, such as spatial relations, motion, temporal order, counting targets, and route constraints.
- Drop filler words, generic grammar, polite framing, and wording that does not change the spatial reasoning target.

Task-specific guidance:
- For route planning questions, keep start point, destination, intermediate landmarks, direction verbs, and route constraints.
- For camera movement questions, keep the moving reference (camera/viewer), direction words, relative motion targets, and temporal cues.
- For relative position questions, keep both reference objects and the relation term such as left/right, near/far, in front of/behind, above/below.
- For distance questions, keep the anchor object(s), target object(s), and distance qualifier such as closest, farthest, near, far, or absolute distance wording.
- For counting/order questions, keep the counted object category and order cue such as first, second, last, before, after.

Bucket rules:
- `visible_grounded_words`: objects, landmarks, scene regions, appearance attributes, and directly observable entities or properties.
- `reasoning_words`: relation terms, motion terms, ordering terms, counting cues, route constraints, and comparison targets that are necessary but not directly visible as standalone entities.
- If a phrase contains both an object and a relation, place it in the bucket that best matches its main role.

Output rules:
- Prefer 2 to 10 items in `visible_grounded_words`.
- Prefer 1 to 8 items in `reasoning_words`.
- Keep phrases short and literal.
- Do not paraphrase into long explanations.
- Return strict JSON only.

Return this schema:
{{
  "visible_grounded_words": ["..."],
  "reasoning_words": ["..."],
  "why": "one short sentence"
}}

Examples:

Example 1
Subset: vsibench_train
Question type: route_plan
Question: How do I get from the sofa to the sink?
Output:
{{
  "visible_grounded_words": ["sofa", "sink"],
  "reasoning_words": ["route", "from", "to"],
  "why": "The visible landmarks define the path endpoints and the route relation is required."
}}

Example 2
Subset: vstibench_train
Question type: camera_displacement
Question: Did the camera move closer to the chair or farther away from it?
Output:
{{
  "visible_grounded_words": ["chair"],
  "reasoning_words": ["camera", "move", "closer", "farther away"],
  "why": "The chair is the visual anchor and the motion-distance terms define the reasoning target."
}}

Example 3
Subset: vstibench_train
Question type: obj_obj_relative_pos_lr
Question: Is the lamp to the left or right of the television?
Output:
{{
  "visible_grounded_words": ["lamp", "television"],
  "reasoning_words": ["left", "right"],
  "why": "The two visible objects are the anchors and the relation terms define the comparison."
}}

Subset: {subset}
Question type: {question_type}
Question:
{question}
"""


@dataclass
class SelectionResult:
    record_id: str
    selected_words: list[str]
    why: str
    raw_response: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize VLM-3R data and run word selection with an OpenAI-compatible endpoint."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["vsibench_train", "vstibench_train"],
        choices=["vsibench_train", "vstibench_train"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "vlm3r_word_selection",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--preview-size", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
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


def extract_text(record: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_question_answer(record: dict[str, Any]) -> tuple[str, str]:
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        question = ""
        answer = ""
        if len(conversations) > 0 and isinstance(conversations[0], dict):
            question = extract_text(conversations[0], ["value", "content", "text"])
        if len(conversations) > 1 and isinstance(conversations[1], dict):
            answer = extract_text(conversations[1], ["value", "content", "text"])
        if question or answer:
            return question, answer
    question = extract_text(record, ["question", "prompt", "query", "instruction"])
    answer = extract_text(record, ["answer", "response", "target", "label"])
    return question, answer


def normalize_dataset(dataset_root: Path, subsets: list[str]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for subset in subsets:
        subset_dir = dataset_root / subset
        for json_path in sorted(subset_dir.glob("*.json")):
            for idx, record in enumerate(iter_records(json_path)):
                question, answer = extract_question_answer(record)
                record_id = str(record.get("id") or f"{subset}:{json_path.stem}:{idx}")
                normalized.append(
                    {
                        "id": record_id,
                        "subset": subset,
                        "source_file": json_path.name,
                        "data_source": record.get("data_source", ""),
                        "scene_name": record.get("scene_name", ""),
                        "question_type": record.get("question_type", json_path.stem),
                        "video": record.get("video", ""),
                        "question": question,
                        "answer": answer,
                    }
                )
    return normalized


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_subset: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for row in rows:
        by_subset[row["subset"]] = by_subset.get(row["subset"], 0) + 1
        qtype = row["question_type"] or "unknown"
        by_type[qtype] = by_type.get(qtype, 0) + 1
    return {
        "total_records": len(rows),
        "by_subset": dict(sorted(by_subset.items())),
        "by_question_type": dict(sorted(by_type.items())),
    }


def build_preview(rows: list[dict[str, Any]], preview_size: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["question_type"] or "unknown", []).append(row)
    preview: list[dict[str, Any]] = []
    group_names = sorted(grouped)
    while len(preview) < preview_size and group_names:
        made_progress = False
        for group_name in group_names:
            items = grouped[group_name]
            if items:
                preview.append(items.pop(0))
                made_progress = True
                if len(preview) >= preview_size:
                    break
        if not made_progress:
            break
    return preview


def post_json(url: str, payload: dict[str, Any], api_key: str, timeout: int) -> str:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def clean_word_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        word = str(item).strip()
        if not word:
            continue
        if word in seen:
            continue
        seen.add(word)
        cleaned.append(word)
    return cleaned


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--:--"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressPrinter:
    def __init__(
        self,
        *,
        total: int,
        initial_done: int,
        report_every: int = 100,
        report_interval_seconds: float = 30.0,
    ) -> None:
        self.total = total
        self.initial_done = initial_done
        self.current_done = initial_done
        self.run_completed = 0
        self.run_errors = 0
        self.start_ts = time.time()
        self.last_report_ts = self.start_ts
        self.report_every = max(1, report_every)
        self.report_interval_seconds = report_interval_seconds

    def _build_line(self, *, final: bool = False) -> str:
        elapsed = time.time() - self.start_ts
        pct = (self.current_done / self.total * 100.0) if self.total else 100.0
        width = 24
        filled = min(width, int(width * self.current_done / self.total)) if self.total else width
        bar = "#" * filled + "-" * (width - filled)
        rate = (self.run_completed / elapsed) if elapsed > 0 else 0.0
        rate_per_hour = rate * 3600.0
        remaining = max(0, self.total - self.current_done)
        eta_seconds = (remaining / rate) if rate > 0 else None
        finish_at = (
            datetime.now().astimezone().replace(microsecond=0) + timedelta(seconds=eta_seconds)
            if eta_seconds is not None
            else None
        )
        status = "done" if final else "progress"
        finish_text = finish_at.isoformat(sep=" ") if finish_at else "unknown"
        return (
            f"[{status}] [{bar}] {self.current_done}/{self.total} ({pct:5.2f}%) | "
            f"new={self.run_completed} | errors={self.run_errors} | "
            f"rate={rate_per_hour:,.1f}/h | elapsed={format_duration(elapsed)} | "
            f"eta={format_duration(eta_seconds)} | finish={finish_text}"
        )

    def maybe_report(self) -> None:
        now = time.time()
        should_report = (
            self.run_completed == 0
            or self.run_completed % self.report_every == 0
            or (now - self.last_report_ts) >= self.report_interval_seconds
        )
        if should_report:
            print(self._build_line(), flush=True)
            self.last_report_ts = now

    def record_success(self) -> None:
        self.current_done += 1
        self.run_completed += 1
        self.maybe_report()

    def record_error(self) -> None:
        self.run_errors += 1
        self.run_completed += 1
        self.maybe_report()

    def report_initial(self, pending: int) -> None:
        print(
            f"[progress] resume_done={self.initial_done} | pending={pending} | total={self.total}",
            flush=True,
        )
        if pending == 0:
            print(self._build_line(final=True), flush=True)

    def report_final(self) -> None:
        print(self._build_line(final=True), flush=True)


def select_single_row(
    row: dict[str, Any],
    *,
    chat_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    question = row.get("question", "").strip()
    if not question:
        return "error", {"id": row["id"], "question": question, "error": "missing_question"}

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    subset=row.get("subset", ""),
                    question_type=row.get("question_type", ""),
                    question=question,
                ),
            }
        ],
    }

    try:
        raw = post_json(chat_url, payload, api_key=api_key, timeout=timeout)
        response_payload = json.loads(raw)
        content = response_payload["choices"][0]["message"]["content"]
        parsed = extract_json_object(content)
        visible_grounded_words = clean_word_list(parsed.get("visible_grounded_words", []))
        reasoning_words = clean_word_list(parsed.get("reasoning_words", []))
        if not visible_grounded_words and not reasoning_words:
            selected_words = clean_word_list(parsed.get("selected_words", []))
            if not selected_words:
                raise ValueError("no usable selected words returned")
            visible_grounded_words = selected_words
        selected_words = []
        for item in visible_grounded_words + reasoning_words:
            if item not in selected_words:
                selected_words.append(item)
        result = {
            **row,
            "visible_grounded_words": visible_grounded_words,
            "reasoning_words": reasoning_words,
            "selected_words": selected_words,
            "why": str(parsed.get("why", "")).strip(),
            "raw_response": content,
        }
        return "success", result
    except Exception as exc:
        return "error", {
            "id": row["id"],
            "question": question,
            "error": str(exc),
        }


def run_selection(
    rows: list[dict[str, Any]],
    output_dir: Path,
    model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    concurrency: int,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_path = output_dir / "selected_words.jsonl"
    errors_path = output_dir / "selection_errors.jsonl"
    existing_ids: set[str] = set()

    if resume and selected_path.exists():
        with selected_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    existing_ids.add(json.loads(line)["id"])
                except Exception:
                    continue

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    chat_url = api_base.rstrip("/") + "/chat/completions"
    pending_rows = [row for row in rows if row["id"] not in existing_ids]
    progress = ProgressPrinter(total=len(rows), initial_done=len(existing_ids))

    selected_path.parent.mkdir(parents=True, exist_ok=True)
    success_mode = "a" if resume and selected_path.exists() else "w"
    error_mode = "a" if resume and errors_path.exists() else "w"

    with selected_path.open(success_mode, encoding="utf-8") as success_handle, errors_path.open(
        error_mode, encoding="utf-8"
    ) as error_handle:
        progress.report_initial(len(pending_rows))
        max_workers = max(1, concurrency)
        if max_workers == 1:
            for row in pending_rows:
                status, payload = select_single_row(
                    row,
                    chat_url=chat_url,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key,
                    timeout=timeout,
                )
                if status == "success":
                    results.append(payload)
                    success_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    success_handle.flush()
                    progress.record_success()
                else:
                    errors.append(payload)
                    error_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    error_handle.flush()
                    progress.record_error()
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        select_single_row,
                        row,
                        chat_url=chat_url,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=api_key,
                        timeout=timeout,
                    )
                    for row in pending_rows
                ]
                for future in as_completed(futures):
                    status, payload = future.result()
                    if status == "success":
                        results.append(payload)
                        success_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        success_handle.flush()
                        progress.record_success()
                    else:
                        errors.append(payload)
                        error_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        error_handle.flush()
                        progress.record_error()
        if pending_rows:
            progress.report_final()
    return results, errors


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized = normalize_dataset(args.dataset_root, args.subsets)
    if args.limit > 0:
        normalized = normalized[: args.limit]

    questions_only = [
        {
            "id": row["id"],
            "subset": row["subset"],
            "source_file": row["source_file"],
            "data_source": row["data_source"],
            "scene_name": row["scene_name"],
            "question_type": row["question_type"],
            "video": row["video"],
            "question": row["question"],
        }
        for row in normalized
    ]
    preview = build_preview(questions_only, args.preview_size)
    summary = summarize(normalized)

    write_json(output_dir / "dataset_manifest.json", {"dataset_root": str(args.dataset_root), "subsets": args.subsets})
    write_jsonl(output_dir / "normalized_train.jsonl", normalized)
    write_json(output_dir / "normalized_summary.json", summary)
    write_jsonl(output_dir / "questions_only.jsonl", questions_only)
    write_json(output_dir / "preview_50.json", preview)

    if args.prepare_only:
        print(f"Prepared VLM-3R artifacts in {output_dir}")
        return 0

    if args.select_only:
        selection_rows = questions_only
    else:
        selection_rows = questions_only

    _, errors = run_selection(
        selection_rows,
        output_dir=output_dir,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        concurrency=args.concurrency,
        resume=args.resume,
    )

    summary["selection_error_count"] = len(errors)
    write_json(output_dir / "normalized_summary.json", summary)
    print(f"Finished VLM-3R word selection in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
