import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError:
    AutoTokenizer = None
    LLM = None
    SamplingParams = None


COLOR_WORDS = "red|blue|green|yellow|orange|purple|white|black"
MARKER_WORDS = "point|bbox"
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:-[A-Za-z0-9]+)?|\d+(?:\.\d+)?")
PLACEHOLDER_PATTERN = re.compile(r"<(?:image|video|audio)>", re.IGNORECASE)
FRAME_MENTION_PATTERN = re.compile(
    rf"\b[\w][\w\s/-]*?\s*\(in\s+Frame-\d+[^)]*\)\s*(?:\(({COLOR_WORDS})\s+({MARKER_WORDS})\))?",
    re.IGNORECASE,
)
NAMED_COLOR_PATTERN = re.compile(
    rf"\b[\w][\w\s/-]*?\s*\(({COLOR_WORDS})\s+({MARKER_WORDS})\)",
    re.IGNORECASE,
)
OBJECT_ID_PATTERN = re.compile(r"\bObject\d+\b")
NAMED_OBJECT_ID_PATTERN = re.compile(r"\b[\w][\w\s/-]*?\s*\(Object\d+\)", re.IGNORECASE)
SPATIAL_TERM_PATTERN = re.compile(
    r"\b("
    r"left|right|above|below|top|bottom|center|middle|front|behind|between|near|far|"
    r"closest|farthest|inside|outside|over|under|distance|depth|meter|meters|cm|km|feet|"
    r"degree|degrees|north|south|east|west|coordinate|coordinates|bbox|bounding box"
    r")\b",
    re.IGNORECASE,
)

BASE_SYSTEM_PROMPT = (
    "Extract only the minimal words or short phrases needed for spatial reasoning. "
    "Return JSON only with keys selected_tokens, selected_mentions, spatial_terms."
)

QUESTION_TYPE_PROMPTS = {
    "distance_depth": "Keep distance, depth, measurement, and the referenced objects.",
    "bbox_localization": "Keep target objects, frame or bbox references, and localization terms.",
    "relative_position": "Keep objects and relation words such as left, right, above, and behind.",
    "other": "Keep only spatially useful words or short phrases.",
}

REFERENCE_STYLE_PROMPTS = {
    "named_color_reference": "Keep the object name and color marker as one grounded mention.",
    "object_id_reference": "Keep the object ID exactly as written; keep the object name too if present.",
    "frame_object_reference": "Keep the full frame-linked object reference, including frame, object ID, and marker.",
    "plain_object_reference": "Keep the main object names and spatial terms only.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select spatial words from SPAR questions.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records-per-file", type=int, default=None)
    parser.add_argument("--heuristic-only", action="store_true")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-vllm-tqdm", action="store_true")
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


def iter_question_entries(source_file: str, record: dict, source_index: int) -> Iterator[dict]:
    sample_id = record.get("id")
    if isinstance(record.get("conversations"), list):
        for turn_index, turn in enumerate(record["conversations"]):
            role = str(turn.get("from") or turn.get("role") or "").lower()
            text = turn.get("value") or turn.get("content") or ""
            if role in {"human", "user"} and isinstance(text, str):
                yield {
                    "source_file": source_file,
                    "source_index": source_index,
                    "id": sample_id,
                    "turn_index": turn_index,
                    "question": text,
                }
    elif isinstance(record.get("question"), str):
        yield {
            "source_file": source_file,
            "source_index": source_index,
            "id": sample_id,
            "turn_index": 0,
            "question": record["question"],
        }


def clean_question_text(question: str) -> str:
    cleaned = PLACEHOLDER_PATTERN.sub(" ", question)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def classify_question_type(question: str) -> str:
    lowered = question.lower()
    if re.search(r"\b(distance|depth|meter|meters|cm|km|feet|foot)\b", lowered):
        return "distance_depth"
    if re.search(r"\b(bbox|bounding box|coordinate|coordinates|top left|bottom right)\b", lowered):
        return "bbox_localization"
    if re.search(r"\b(left|right|above|below|behind|front|between|near|far|closest|farthest)\b", lowered):
        return "relative_position"
    return "other"


def classify_reference_style(question: str) -> str:
    if re.search(r"\(in\s+Frame-\d+[^)]*\)", question, re.IGNORECASE):
        return "frame_object_reference"
    if re.search(rf"\(({COLOR_WORDS})\s+({MARKER_WORDS})\)", question, re.IGNORECASE):
        return "named_color_reference"
    if OBJECT_ID_PATTERN.search(question):
        return "object_id_reference"
    return "plain_object_reference"


def normalize_phrase(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip(" ,.;:"))
    return text


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def extract_mentions(question: str) -> List[str]:
    mentions = []
    for pattern in (FRAME_MENTION_PATTERN, NAMED_OBJECT_ID_PATTERN, NAMED_COLOR_PATTERN):
        for match in pattern.finditer(question):
            mention = normalize_phrase(match.group(0))
            mentions.append(mention)

    for match in re.finditer(r"\bObject\d+\s*\([^)]*\)", question):
        mentions.append(normalize_phrase(match.group(0)))
    return unique_keep_order(mentions)


def extract_spatial_terms(question: str) -> List[str]:
    return unique_keep_order(normalize_phrase(match.group(0)) for match in SPATIAL_TERM_PATTERN.finditer(question))


def extract_tokens(question: str, mentions: List[str], spatial_terms: List[str]) -> List[str]:
    tokens = []
    for mention in mentions:
        for token in TOKEN_PATTERN.findall(mention):
            tokens.append(token.lower())
    for term in spatial_terms:
        for token in TOKEN_PATTERN.findall(term):
            tokens.append(token.lower())
    return unique_keep_order(tokens)


def heuristic_selection(question: str) -> Dict[str, List[str]]:
    mentions = extract_mentions(question)
    spatial_terms = extract_spatial_terms(question)
    tokens = extract_tokens(question, mentions, spatial_terms)
    return {
        "selected_tokens": tokens,
        "selected_mentions": mentions,
        "spatial_terms": spatial_terms,
    }


def build_prompts(question: str, question_type: str, reference_style: str) -> Tuple[str, str]:
    system_prompt = " ".join(
        [
            BASE_SYSTEM_PROMPT,
            QUESTION_TYPE_PROMPTS[question_type],
            REFERENCE_STYLE_PROMPTS[reference_style],
        ]
    )
    user_prompt = (
        f"Question type: {question_type}\n"
        f"Reference style: {reference_style}\n"
        f"Question: {question}\n\n"
        "Return JSON only."
    )
    return system_prompt, user_prompt


def init_vllm(args: argparse.Namespace):
    if args.heuristic_only:
        return None, None
    if not args.model_path:
        raise ValueError("--model-path is required unless --heuristic-only is used.")
    if LLM is None or SamplingParams is None or AutoTokenizer is None:
        raise ImportError("transformers and vllm must be available for direct batch inference.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        download_dir=args.hf_home,
        enforce_eager=args.enforce_eager,
    )
    return tokenizer, llm


def make_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_model_response(text: str, question: str) -> Dict[str, List[str]]:
    payload = text.strip()
    if not payload:
        return heuristic_selection(question)

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, re.DOTALL)
        if not match:
            return heuristic_selection(question)
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return heuristic_selection(question)

    result = {
        "selected_tokens": normalize_list(data.get("selected_tokens")),
        "selected_mentions": normalize_list(data.get("selected_mentions")),
        "spatial_terms": normalize_list(data.get("spatial_terms")),
    }
    if not any(result.values()):
        return heuristic_selection(question)
    return result


def normalize_list(value) -> List[str]:
    if not isinstance(value, list):
        return []
    return unique_keep_order(normalize_phrase(str(item)) for item in value if str(item).strip())


def iter_samples(input_path: str, limit: int | None) -> Iterator[dict]:
    for source_index, record in enumerate(iter_records(input_path)):
        if limit is not None and source_index >= limit:
            break
        for entry in iter_question_entries(Path(input_path).name, record, source_index):
            cleaned_question = clean_question_text(entry["question"])
            question_type = classify_question_type(cleaned_question)
            reference_style = classify_reference_style(cleaned_question)
            yield {
                **entry,
                "cleaned_question": cleaned_question,
                "question_type": question_type,
                "reference_style": reference_style,
            }


def run_vllm_batch(samples: List[dict], args: argparse.Namespace, tokenizer, llm) -> List[dict]:
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    outputs: List[dict] = []
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        prompts = []
        for sample in batch:
            system_prompt, user_prompt = build_prompts(
                sample["cleaned_question"],
                sample["question_type"],
                sample["reference_style"],
            )
            prompts.append(make_chat_prompt(tokenizer, system_prompt, user_prompt))

        generations = llm.generate(
            prompts,
            sampling_params,
            use_tqdm=not args.disable_vllm_tqdm,
        )
        for sample, generation in zip(batch, generations):
            text = generation.outputs[0].text if generation.outputs else ""
            result = parse_model_response(text, sample["cleaned_question"])
            outputs.append(
                {
                    "source_file": sample["source_file"],
                    "source_index": sample["source_index"],
                    "id": sample["id"],
                    "turn_index": sample["turn_index"],
                    "question_type": sample["question_type"],
                    "reference_style": sample["reference_style"],
                    "selected_tokens": result["selected_tokens"],
                    "selected_mentions": result["selected_mentions"],
                    "spatial_terms": result["spatial_terms"],
                    "method": "vllm_batch",
                }
            )
    return outputs


def run_heuristic(samples: List[dict]) -> List[dict]:
    outputs = []
    for sample in samples:
        result = heuristic_selection(sample["cleaned_question"])
        outputs.append(
            {
                "source_file": sample["source_file"],
                "source_index": sample["source_index"],
                "id": sample["id"],
                "turn_index": sample["turn_index"],
                "question_type": sample["question_type"],
                "reference_style": sample["reference_style"],
                "selected_tokens": result["selected_tokens"],
                "selected_mentions": result["selected_mentions"],
                "spatial_terms": result["spatial_terms"],
                "method": "heuristic",
            }
        )
    return outputs


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = llm = None
    if not args.heuristic_only:
        tokenizer, llm = init_vllm(args)

    first_item = True
    batch_counter = 0
    processed_questions = 0
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        for input_path in args.inputs:
            batch: List[dict] = []
            for sample in iter_samples(input_path, args.max_records_per_file):
                batch.append(sample)
                if len(batch) >= args.batch_size:
                    batch_counter += 1
                    processed_questions += len(batch)
                    print(
                        f"[progress] batch={batch_counter} processed_questions={processed_questions} input={input_path}",
                        file=sys.stderr,
                        flush=True,
                    )
                    outputs = run_heuristic(batch) if args.heuristic_only else run_vllm_batch(batch, args, tokenizer, llm)
                    first_item = write_outputs(handle, outputs, first_item)
                    batch = []
            if batch:
                batch_counter += 1
                processed_questions += len(batch)
                print(
                    f"[progress] batch={batch_counter} processed_questions={processed_questions} input={input_path}",
                    file=sys.stderr,
                    flush=True,
                )
                outputs = run_heuristic(batch) if args.heuristic_only else run_vllm_batch(batch, args, tokenizer, llm)
                first_item = write_outputs(handle, outputs, first_item)
        handle.write("\n]\n")
    return 0


def write_outputs(handle, outputs: List[dict], first_item: bool) -> bool:
    for item in outputs:
        if not first_item:
            handle.write(",\n")
        handle.write(json.dumps(item, ensure_ascii=False, indent=2))
        first_item = False
    return first_item


if __name__ == "__main__":
    raise SystemExit(main())
