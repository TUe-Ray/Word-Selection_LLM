from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from tqdm import tqdm

from json_stream import iter_json_array
from question_utils import classify_question_type, classify_reference_style, iter_human_questions

BASE_SYSTEM_PROMPT = """You are a data labeling assistant for spatial reasoning.
Extract only the minimal set of words or short phrases from the question that are useful for solving the task.

Shared rules:
- Return JSON only, with the format {"terms": ["term1", "term2"]}.
- Keep short terms or short phrases up to 3 words.
- No duplicates.
- Ignore placeholders such as <image>, <video>, and generic narration.
- Ignore answer-choice letters and output-format instructions.
- If the question is not useful for spatial reasoning, return {"terms": []}.
- Preserve object identifiers exactly when they matter, for example Object77 or Frame-16.
"""

TYPE_SPECIFIC_PROMPTS = {
    "distance_depth": """Question task type: distance_depth.
Prioritize:
- Target objects and reference objects.
- Distance, depth, center, coordinate, BEV, meter-like units.
- Observer or viewpoint terms when they define the measurement reference.
Avoid:
- Option letters, standalone choice numbers, and generic verbs such as calculate or choose.
""",
    "bbox_localization": """Question task type: bbox_localization.
Prioritize:
- Target object names and reference object names.
- bounding box, bbox, first/second/third image, and any identifying color tags.
Avoid:
- Generic instruction verbs such as locate, find, draw, provide, annotate, or enclose.
""",
    "relative_position": """Question task type: relative_position.
Prioritize:
- Object names.
- Observer, viewpoint, perspective, position, movement, orientation.
- Direction and relation terms such as left, right, above, below, between, in front of, behind, near, far, closer, farther.
Avoid:
- Output-format instructions and answer-choice letters.
""",
    "other": """Question task type: other.
Only keep terms if they are clearly useful for spatial reasoning. Otherwise return {"terms": []}.
""",
}

REFERENCE_STYLE_PROMPTS = {
    "named_color_reference": """Object reference style: named_color_reference.
Objects are usually written as noun phrases followed by a color marker, for example wardrobe (red point) or power socket (blue point).
Keep the actual object names and useful marker phrases when they help disambiguate the objects.
""",
    "object_id_reference": """Object reference style: object_id_reference.
Objects may be referred to only by IDs such as Object77 or Object0.
Keep these object IDs exactly as written. Do not invent missing object names.
""",
    "frame_object_reference": """Object reference style: frame_object_reference.
Objects may be written in detailed video form such as tv (in Frame-16, Object0) (green bbox).
Keep the object name, frame ID, object ID, and color marker when they are needed to identify the object.
""",
    "plain_object_reference": """Object reference style: plain_object_reference.
Use the best available noun phrases that identify the objects.
""",
}

SPATIAL_TERMS = {
    "left", "right", "above", "below", "under", "over", "behind", "front", "forward", "backward",
    "between", "inside", "outside", "near", "far", "closest", "farthest", "top", "bottom", "center",
    "middle", "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest",
    "around", "across", "toward", "away", "adjacent", "next", "beside", "opposite", "distance",
    "meter", "meters", "m", "cm", "mm", "km", "foot", "feet", "inch", "inches", "degree", "degrees",
    "coordinate", "coordinates", "axis", "origin", "bev", "horizontal", "vertical", "position", "relative",
    "movement", "perspective", "direction", "location", "first", "second", "third", "depth", "bbox",
}

COMMON_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "being", "been", "to", "of", "in", "on", "at",
    "for", "from", "by", "with", "without", "and", "or", "that", "this", "these", "those", "what", "which",
    "who", "whom", "when", "where", "why", "how", "do", "does", "did", "can", "could", "would", "should",
    "may", "might", "will", "shall", "it", "its", "their", "there", "then", "than", "as", "if", "into",
    "about", "please", "according", "provided", "image", "images", "video", "videos", "main", "viewpoint",
}

GENERIC_NON_OBJECT = {
    "describe", "calculate", "judge", "output", "include", "maintain", "construct", "designate", "using",
    "rules", "format", "example", "transition", "required", "provided", "please", "show", "tell", "given",
    "judge", "based", "exact", "line", "separate", "select", "choose", "answer", "option", "options",
    "locate", "find", "provide", "annotate", "mark", "draw", "report", "determine", "predict", "estimate",
    "object", "objects", "frame", "frames",
}

SPECIAL_PHRASES = {
    "bounding box": "bounding box",
    "observer's perspective": "observer perspective",
    "observer perspective": "observer perspective",
    "world coordinate system": "world coordinate system",
    "bird's-eye view": "bev",
    "bird’s-eye view": "bev",
    "center point": "center",
}

FRAME_TOKEN_PATTERN = re.compile(r"frame-\d+", re.IGNORECASE)
OBJECT_TOKEN_PATTERN = re.compile(r"object\d+", re.IGNORECASE)
TOKEN_PATTERN = r"frame-\d+|object\d+|[a-z]+(?:'[a-z]+)?\d*|\d+(?:\.\d+)?(?:m|cm|mm|km|ft|feet|inch|inches)?"
UNIT_PATTERN = r"\d+(?:\.\d+)?(?:m|cm|mm|km|ft|feet|inch|inches)"


@dataclass
class SelectionResult:
    terms: List[str]
    method: str
    error: Optional[str] = None


def build_system_prompt(question_type: str, reference_style: str) -> str:
    return (
        BASE_SYSTEM_PROMPT
        + "\n"
        + TYPE_SPECIFIC_PROMPTS.get(question_type, TYPE_SPECIFIC_PROMPTS["other"])
        + "\n"
        + REFERENCE_STYLE_PROMPTS.get(reference_style, REFERENCE_STYLE_PROMPTS["plain_object_reference"])
    )


def build_user_prompt(question: str, question_type: str, reference_style: str) -> str:
    return (
        f"Question task type: {question_type}\n"
        f"Object reference style: {reference_style}\n"
        f"Question: {question}\n\n"
        "Return only JSON with a terms array."
    )


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def normalize_terms(raw_terms: Iterable[str], max_terms: int) -> List[str]:
    terms = []
    for term in raw_terms:
        if not isinstance(term, str):
            continue
        clean = term.strip().lower()
        clean = re.sub(r"\s+", " ", clean)
        clean = re.sub(r"^[^\w-]+|[^\w-]+$", "", clean)
        if not clean:
            continue
        if len(clean.split()) > 3:
            continue
        terms.append(clean)

    deduped = unique_preserve_order(terms)
    return deduped[:max_terms]


def parse_terms_payload(text: str, max_terms: int) -> List[str]:
    payload = text.strip()

    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?", "", payload).strip()
        payload = re.sub(r"```$", "", payload).strip()

    obj = None
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        obj_match = re.search(r"\{[\s\S]*\}", payload)
        arr_match = re.search(r"\[[\s\S]*\]", payload)
        if obj_match:
            obj = json.loads(obj_match.group(0))
        elif arr_match:
            obj = json.loads(arr_match.group(0))

    if isinstance(obj, list):
        return normalize_terms(obj, max_terms)

    if isinstance(obj, dict):
        for key in ("terms", "words", "selected_terms"):
            value = obj.get(key)
            if isinstance(value, list):
                return normalize_terms(value, max_terms)

    raise ValueError("Could not parse terms JSON from model response")


def heuristic_select_terms(question: str, max_terms: int, question_type: str, reference_style: str) -> List[str]:
    if question_type == "other":
        return []

    q = question.lower()
    tokens = re.findall(TOKEN_PATTERN, q)

    selected = []
    for raw_phrase, normalized_phrase in SPECIAL_PHRASES.items():
        if raw_phrase in q:
            selected.append(normalized_phrase)

    for token in tokens:
        if token in COMMON_STOPWORDS or token in GENERIC_NON_OBJECT:
            continue
        if len(token) <= 2 and token not in {"m", "cm", "mm", "km"}:
            continue

        is_number = re.fullmatch(r"\d+(?:\.\d+)?", token) is not None
        has_unit = re.fullmatch(UNIT_PATTERN, token) is not None
        is_frame_token = FRAME_TOKEN_PATTERN.fullmatch(token) is not None
        is_object_token = OBJECT_TOKEN_PATTERN.fullmatch(token) is not None

        if is_frame_token and reference_style != "frame_object_reference":
            continue
        if is_object_token and reference_style not in {"object_id_reference", "frame_object_reference"}:
            continue
        if is_number and question_type != "distance_depth":
            continue
        if has_unit or token in SPATIAL_TERMS:
            selected.append(token)
            continue
        if is_frame_token or is_object_token:
            selected.append(token)
            continue
        if question_type == "bbox_localization" and token in {"bbox", "bounding", "box", "first", "second", "third"}:
            selected.append(token)
            continue
        if question_type == "relative_position" and token in {"observer", "perspective", "position", "direction", "orientation"}:
            selected.append(token)
            continue
        if question_type == "distance_depth" and token in {"depth", "distance", "center", "observer", "coordinate", "coordinates"}:
            selected.append(token)
            continue
        if not is_number:
            selected.append(token)

    return normalize_terms(selected, max_terms)


class LLMSelector:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        max_terms: int,
        retries: int,
        heuristic_only: bool,
        fallback_on_error: bool,
        timeout: float,
    ) -> None:
        self.model = model
        self.max_terms = max_terms
        self.retries = retries
        self.heuristic_only = heuristic_only
        self.fallback_on_error = fallback_on_error

        self.client = None
        if not heuristic_only:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "openai package is not installed. Install dependencies with: conda env create -f environment.yml"
                ) from exc

            self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def select(self, question: str, question_type: str, reference_style: str) -> SelectionResult:
        if self.heuristic_only:
            return SelectionResult(
                terms=heuristic_select_terms(question, self.max_terms, question_type, reference_style),
                method="heuristic_only",
            )

        last_error = None
        for _ in range(self.retries + 1):
            try:
                assert self.client is not None
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": build_system_prompt(question_type, reference_style)},
                        {"role": "user", "content": build_user_prompt(question, question_type, reference_style)},
                    ],
                    max_tokens=200,
                )
                content = response.choices[0].message.content or ""
                terms = parse_terms_payload(content, self.max_terms)
                return SelectionResult(terms=terms, method="llm")
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)

        if self.fallback_on_error:
            return SelectionResult(
                terms=heuristic_select_terms(question, self.max_terms, question_type, reference_style),
                method="heuristic_fallback",
                error=last_error,
            )

        raise RuntimeError(f"LLM selection failed for question: {question}\nError: {last_error}")


class JsonArrayWriter:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.handle = None
        self.first = True

    def __enter__(self) -> "JsonArrayWriter":
        self.handle = self.output_path.open("w", encoding="utf-8")
        self.handle.write("[\n")
        return self

    def write(self, obj: dict) -> None:
        assert self.handle is not None
        if not self.first:
            self.handle.write(",\n")
        json.dump(obj, self.handle, ensure_ascii=False)
        self.first = False

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self.handle is not None:
            self.handle.write("\n]\n")
            self.handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select spatial reasoning words from question turns.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON files")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name for OpenAI-compatible endpoint")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="EMPTY", help="API key for the endpoint")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds")
    parser.add_argument("--retries", type=int, default=1, help="Retry count per question on LLM errors")
    parser.add_argument("--max-terms", type=int, default=20, help="Maximum terms to keep per question")
    parser.add_argument("--max-records-per-file", type=int, default=None, help="Optional debug cap per input file")
    parser.add_argument("--chunk-size", type=int, default=1_048_576, help="Stream chunk size in bytes")
    parser.add_argument("--report-every", type=int, default=100, help="Print progress every N records")
    parser.add_argument("--heuristic-only", action="store_true", help="Skip LLM calls and use heuristic selection")
    parser.add_argument("--no-fallback", action="store_true", help="Disable heuristic fallback on LLM errors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths = [Path(p) for p in args.inputs]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    selector = LLMSelector(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_terms=args.max_terms,
        retries=args.retries,
        heuristic_only=args.heuristic_only,
        fallback_on_error=not args.no_fallback,
        timeout=args.timeout,
    )

    total_records = 0
    total_questions = 0
    output_path = Path(args.output)

    with JsonArrayWriter(output_path) as writer:
        for input_path in input_paths:
            print(f"Processing {input_path} ...", flush=True)
            file_records = 0

            iterator = iter_json_array(input_path, chunk_size=args.chunk_size)
            for record_index, record in enumerate(tqdm(iterator, unit="record")):
                if args.max_records_per_file is not None and file_records >= args.max_records_per_file:
                    break

                file_records += 1
                total_records += 1

                question_entries = []
                for turn_index, question in iter_human_questions(record):
                    question_type = classify_question_type(question)
                    reference_style = classify_reference_style(question)
                    result = selector.select(question, question_type, reference_style)
                    question_item = {
                        "turn_index": turn_index,
                        "question": question,
                        "question_type": question_type,
                        "reference_style": reference_style,
                        "selected_terms": result.terms,
                        "method": result.method,
                    }
                    if result.error:
                        question_item["error"] = result.error
                    question_entries.append(question_item)
                    total_questions += 1

                if question_entries:
                    writer.write(
                        {
                            "source_file": input_path.name,
                            "source_index": record_index,
                            "id": record.get("id"),
                            "question_selections": question_entries,
                        }
                    )

                if args.report_every > 0 and file_records % args.report_every == 0:
                    print(
                        f"  {input_path.name}: processed {file_records} records, total questions {total_questions}",
                        flush=True,
                    )

    print(
        f"Done. Wrote {output_path} with {total_records} records processed and {total_questions} questions.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        raise
