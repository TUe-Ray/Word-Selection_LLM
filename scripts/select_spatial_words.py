from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

from json_stream import iter_json_array
from question_utils import classify_question_type, classify_reference_style, iter_human_questions

BASE_SYSTEM_PROMPT = """You are a data labeling assistant for spatial reasoning.
Extract structured signals from the question for downstream training.

Return JSON only with this schema:
{
  "selected_tokens": ["..."],
  "selected_mentions": ["..."],
  "spatial_terms": ["..."]
}

Shared rules:
- `selected_tokens`: short tokens or short phrases up to 3 words.
- `selected_mentions`: full object mentions or referring expressions from the question.
- `spatial_terms`: only spatial, directional, viewpoint, distance, or measurement terms.
- No duplicates.
- Ignore placeholders such as <image> and <video>.
- Ignore answer-choice letters and output-format instructions.
- If a field has no useful content, return an empty list for that field.
- Preserve object IDs and frame IDs exactly when they matter, for example `Object77` or `Frame-16`.
"""

TYPE_SPECIFIC_PROMPTS = {
    "distance_depth": """Question task type: distance_depth.
Prioritize:
- target objects and reference objects
- distance, depth, center, coordinate, BEV, meter-like units
- observer or viewpoint terms when they define the measurement reference
Avoid:
- option letters, standalone choice numbers, and generic verbs such as calculate or choose
""",
    "bbox_localization": """Question task type: bbox_localization.
Prioritize:
- target object names and reference object names
- bounding box, bbox, first/second/third image, and identifying color tags
Avoid:
- generic instruction verbs such as locate, find, draw, provide, annotate, or enclose
""",
    "relative_position": """Question task type: relative_position.
Prioritize:
- object names
- observer, viewpoint, perspective, position, movement, orientation
- direction and relation terms such as left, right, above, below, between, in front of, behind, near, far, closer, farther
Avoid:
- output-format instructions and answer-choice letters
""",
    "other": """Question task type: other.
Only keep terms if they are clearly useful for spatial reasoning. Otherwise return empty lists.
""",
}

REFERENCE_STYLE_PROMPTS = {
    "named_color_reference": """Object reference style: named_color_reference.
Objects are usually written as noun phrases followed by a color marker, for example `wardrobe (red point)` or `power socket (blue point)`.
In `selected_mentions`, keep the full mention.
In `selected_tokens`, keep the object name and useful marker tokens.
""",
    "object_id_reference": """Object reference style: object_id_reference.
Objects may be referred to by IDs such as `Object77` or mixed forms such as `window (Object2)`.
In `selected_mentions`, keep the full mention when available.
In `selected_tokens`, preserve object IDs exactly.
Do not invent missing object names.
""",
    "frame_object_reference": """Object reference style: frame_object_reference.
Objects may be written in detailed video form such as `tv (in Frame-16, Object0) (green bbox)`.
In `selected_mentions`, keep the full mention.
In `selected_tokens`, preserve the object name, frame ID, object ID, and color marker when they identify the object.
""",
    "plain_object_reference": """Object reference style: plain_object_reference.
Use the best available noun phrases that identify the objects.
""",
}

COLOR_PATTERN = r"(?:red|blue|green|yellow|orange|purple|pink|white|black|brown|gray|grey)"
OBJECT_NAME_PATTERN = r"[A-Za-z][A-Za-z0-9'/-]*(?:\s+[A-Za-z][A-Za-z0-9'/-]*){0,4}"

SPATIAL_TERMS = {
    "above",
    "across",
    "adjacent",
    "around",
    "away",
    "axis",
    "backward",
    "behind",
    "below",
    "beside",
    "bev",
    "between",
    "bottom",
    "center",
    "closest",
    "closer",
    "coordinate",
    "coordinates",
    "degree",
    "degrees",
    "depth",
    "direction",
    "distance",
    "east",
    "farthest",
    "farther",
    "far",
    "feet",
    "foot",
    "forward",
    "front",
    "horizontal",
    "inch",
    "inches",
    "inside",
    "km",
    "left",
    "location",
    "m",
    "meter",
    "meters",
    "middle",
    "mm",
    "movement",
    "near",
    "north",
    "northeast",
    "northwest",
    "opposite",
    "origin",
    "outside",
    "over",
    "perspective",
    "position",
    "relative",
    "right",
    "south",
    "southeast",
    "southwest",
    "top",
    "toward",
    "under",
    "vertical",
    "viewpoint",
    "west",
}

COMMON_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "of",
    "on",
    "or",
    "shall",
    "should",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "without",
    "would",
}

GENERIC_NON_OBJECT = {
    "according",
    "annotate",
    "answer",
    "based",
    "calculate",
    "choose",
    "construct",
    "describe",
    "determine",
    "draw",
    "enclose",
    "estimate",
    "exact",
    "example",
    "find",
    "format",
    "given",
    "include",
    "judge",
    "line",
    "locate",
    "maintain",
    "option",
    "options",
    "output",
    "please",
    "predict",
    "provide",
    "provided",
    "report",
    "required",
    "rules",
    "select",
    "separate",
    "show",
    "tell",
    "unit",
    "using",
}

SPATIAL_PHRASES = {
    "bird's-eye view": "bev",
    "bird’s-eye view": "bev",
    "bounding box": "bounding box",
    "center point": "center",
    "in front of": "in front of",
    "observer's perspective": "observer perspective",
    "observer's position": "observer position",
    "observer perspective": "observer perspective",
    "world coordinate system": "world coordinate system",
}

MENTION_PATTERNS = [
    re.compile(
        rf"{OBJECT_NAME_PATTERN}\s*\(\s*in\s+Frame-\d+\s*,\s*Object\d+\s*\)\s*\(\s*{COLOR_PATTERN}\s+(?:bbox|point|box|marker)\s*\)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{OBJECT_NAME_PATTERN}\s*\(\s*in\s+Frame-\d+\s*,\s*Object\d+\s*\)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{OBJECT_NAME_PATTERN}\s*\(\s*Object\d+\s*\)\s*\(\s*{COLOR_PATTERN}\s+(?:bbox|point|box|marker)\s*\)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{OBJECT_NAME_PATTERN}\s*\(\s*Object\d+\s*\)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"Object\d+\s*\(\s*{COLOR_PATTERN}\s+(?:bbox|point|box|marker)\s*\)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{OBJECT_NAME_PATTERN}\s*\(\s*in\s+Frame-\d+\s*\)\s*\(\s*{COLOR_PATTERN}\s+(?:bbox|point|box|marker)\s*\)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{OBJECT_NAME_PATTERN}\s*\(\s*in\s+Frame-\d+\s*\)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{OBJECT_NAME_PATTERN}\s*\(\s*{COLOR_PATTERN}\s+(?:bbox|point|box|marker)\s*\)",
        re.IGNORECASE,
    ),
]

FRAME_TOKEN_PATTERN = re.compile(r"frame-\d+", re.IGNORECASE)
OBJECT_TOKEN_PATTERN = re.compile(r"object\d+", re.IGNORECASE)
TOKEN_PATTERN = r"frame-\d+|object\d+|[a-z]+(?:'[a-z]+)?\d*|\d+(?:\.\d+)?(?:m|cm|mm|km|ft|feet|inch|inches)?"
UNIT_PATTERN = r"\d+(?:\.\d+)?(?:m|cm|mm|km|ft|feet|inch|inches)"
MENTION_BOUNDARY_WORDS = {
    "a",
    "an",
    "and",
    "approximate",
    "at",
    "becomes",
    "calculate",
    "center",
    "compare",
    "depth",
    "describe",
    "determine",
    "distance",
    "estimate",
    "from",
    "how",
    "in",
    "is",
    "located",
    "of",
    "on",
    "points",
    "relative",
    "shifts",
    "specifies",
    "the",
    "to",
    "toward",
    "use",
    "using",
    "what",
    "when",
    "where",
    "with",
}


@dataclass
class SelectionResult:
    selected_tokens: List[str]
    selected_mentions: List[str]
    spatial_terms: List[str]
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
        "Return only JSON with selected_tokens, selected_mentions, and spatial_terms."
    )


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def normalize_token_list(raw_terms: Iterable[str], max_terms: int) -> List[str]:
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
    return unique_preserve_order(terms)[:max_terms]


def normalize_mention_list(raw_mentions: Iterable[str], max_terms: int) -> List[str]:
    mentions = []
    for mention in raw_mentions:
        if not isinstance(mention, str):
            continue
        clean = mention.strip()
        clean = re.sub(r"\s+", " ", clean)
        clean = clean.strip(" ,.;:")
        if not clean:
            continue
        mentions.append(clean)
    return unique_preserve_order(mentions)[:max_terms]


def compact_mention_text(mention: str) -> str:
    mention = re.sub(r"\s+", " ", mention).strip(" ,.;:")
    paren_index = mention.find("(")
    if paren_index < 0:
        return mention

    name_part = mention[:paren_index].strip()
    suffix = mention[paren_index:].strip()

    tokens = name_part.split()
    start_index = 0
    for index, token in enumerate(tokens):
        cleaned = token.lower().strip(",.;:")
        if cleaned in MENTION_BOUNDARY_WORDS:
            start_index = index + 1

    core_tokens = tokens[start_index:]
    while core_tokens and core_tokens[0].lower().strip(",.;:") in {"the", "a", "an"}:
        core_tokens.pop(0)

    if core_tokens:
        name_part = " ".join(core_tokens)

    compact = f"{name_part} {suffix}".strip()
    compact = re.sub(r"\s+", " ", compact)
    return compact.strip(" ,.;:")


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def extract_mentions(question: str) -> List[str]:
    matches: List[Tuple[int, int, str]] = []
    occupied: List[Tuple[int, int]] = []

    for pattern in MENTION_PATTERNS:
        for match in pattern.finditer(question):
            span = match.span()
            if any(overlap(span, other) for other in occupied):
                continue
            occupied.append(span)
            matches.append((span[0], span[1], compact_mention_text(match.group(0))))

    matches.sort(key=lambda item: item[0])
    return normalize_mention_list((text for _, _, text in matches), max_terms=9999)


def spatial_terms_from_question(question: str, max_terms: int) -> List[str]:
    q = question.lower()
    selected = []

    for raw_phrase, normalized_phrase in SPATIAL_PHRASES.items():
        if raw_phrase in q:
            selected.append(normalized_phrase)

    tokens = re.findall(TOKEN_PATTERN, q)
    for token in tokens:
        if re.fullmatch(UNIT_PATTERN, token):
            selected.append(token)
            continue
        if token in SPATIAL_TERMS:
            selected.append(token)

    return normalize_token_list(selected, max_terms)


def tokens_from_mentions(mentions: Sequence[str], max_terms: int) -> List[str]:
    selected = []
    for mention in mentions:
        lowered = mention.lower()
        tokens = re.findall(TOKEN_PATTERN, lowered)
        for token in tokens:
            if token in COMMON_STOPWORDS:
                continue
            if FRAME_TOKEN_PATTERN.fullmatch(token):
                selected.append(token)
                continue
            if OBJECT_TOKEN_PATTERN.fullmatch(token):
                selected.append(token)
                continue
            if re.fullmatch(UNIT_PATTERN, token):
                selected.append(token)
                continue
            if token in SPATIAL_TERMS:
                selected.append(token)
                continue
            if token in GENERIC_NON_OBJECT:
                continue
            if len(token) <= 2 and token not in {"m", "cm", "mm", "km"}:
                continue
            selected.append(token)
    return normalize_token_list(selected, max_terms)


def heuristic_fallback(question: str, question_type: str, reference_style: str, max_terms: int) -> SelectionResult:
    mentions = extract_mentions(question)
    spatial_terms = spatial_terms_from_question(question, max_terms)
    mention_tokens = tokens_from_mentions(mentions, max_terms)

    fallback_tokens = []
    if not mentions:
        raw_tokens = re.findall(TOKEN_PATTERN, question.lower())
        for token in raw_tokens:
            if token in COMMON_STOPWORDS or token in GENERIC_NON_OBJECT:
                continue
            if len(token) <= 2 and token not in {"m", "cm", "mm", "km"}:
                continue
            if question_type != "distance_depth" and re.fullmatch(r"\d+(?:\.\d+)?", token):
                continue
            fallback_tokens.append(token)

    selected_tokens = normalize_token_list(
        list(mention_tokens) + list(spatial_terms) + list(fallback_tokens),
        max_terms,
    )

    return SelectionResult(
        selected_tokens=selected_tokens,
        selected_mentions=normalize_mention_list(mentions, max_terms),
        spatial_terms=spatial_terms,
        method="heuristic_only",
    )


def parse_structured_payload(text: str, max_terms: int) -> Tuple[List[str], List[str], List[str]]:
    payload = text.strip()

    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?", "", payload).strip()
        payload = re.sub(r"```$", "", payload).strip()

    obj = None
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        obj_match = re.search(r"\{[\s\S]*\}", payload)
        if obj_match:
            obj = json.loads(obj_match.group(0))

    if not isinstance(obj, dict):
        raise ValueError("Could not parse structured JSON from model response")

    if "selected_tokens" in obj or "selected_mentions" in obj or "spatial_terms" in obj:
        return (
            normalize_token_list(obj.get("selected_tokens", []), max_terms),
            normalize_mention_list(obj.get("selected_mentions", []), max_terms),
            normalize_token_list(obj.get("spatial_terms", []), max_terms),
        )

    legacy_terms = obj.get("terms") or obj.get("words") or obj.get("selected_terms") or []
    return (
        normalize_token_list(legacy_terms, max_terms),
        [],
        [],
    )


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
            return heuristic_fallback(question, question_type, reference_style, self.max_terms)

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
                    max_tokens=300,
                )
                content = response.choices[0].message.content or ""
                selected_tokens, selected_mentions, spatial_terms = parse_structured_payload(content, self.max_terms)
                return SelectionResult(
                    selected_tokens=selected_tokens,
                    selected_mentions=selected_mentions,
                    spatial_terms=spatial_terms,
                    method="llm",
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)

        if self.fallback_on_error:
            result = heuristic_fallback(question, question_type, reference_style, self.max_terms)
            result.method = "heuristic_fallback"
            result.error = last_error
            return result

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
    parser = argparse.ArgumentParser(description="Select structured spatial reasoning signals from question turns.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON files")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name for OpenAI-compatible endpoint")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="EMPTY", help="API key for the endpoint")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds")
    parser.add_argument("--retries", type=int, default=1, help="Retry count per question on LLM errors")
    parser.add_argument("--max-terms", type=int, default=20, help="Maximum items to keep per output field")
    parser.add_argument("--max-records-per-file", type=int, default=None, help="Optional debug cap per input file")
    parser.add_argument("--chunk-size", type=int, default=1_048_576, help="Stream chunk size in bytes")
    parser.add_argument("--report-every", type=int, default=100, help="Print progress every N records")
    parser.add_argument("--heuristic-only", action="store_true", help="Skip LLM calls and use heuristic extraction")
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
                        "selected_tokens": result.selected_tokens,
                        "selected_mentions": result.selected_mentions,
                        "spatial_terms": result.spatial_terms,
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
