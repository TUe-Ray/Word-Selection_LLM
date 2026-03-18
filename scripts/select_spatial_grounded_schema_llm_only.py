import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError:
    AutoTokenizer = None
    LLM = None
    SamplingParams = None


PLACEHOLDER_PATTERN = re.compile(r"<(?:image|video|audio)>", re.IGNORECASE)

PROMPT_VERSION = "spatial_grounding_schema_v1"
SCHEMA_KEYS = [
    "grounded_mentions",
    "spatial_relations",
    "geometric_constraints",
    "viewpoint_constraints",
    "answer_constraints",
]

BASE_SYSTEM_PROMPT = (
    "You extract the minimal COMPLETE text spans needed for visual grounding and spatial reasoning. "
    "Return JSON only with exactly these keys: grounded_mentions, spatial_relations, geometric_constraints, viewpoint_constraints, answer_constraints. "
    "Each value must be a JSON array of strings. "
    "grounded_mentions is the most important field. "
    "It should contain complete grounded spans that link language to a visible object, point, bounding box, image, frame, region, or candidate answer region."
)

EXTRACTION_RULES_PROMPT = (
    "Rules: "
    "1. Prefer the shortest COMPLETE contiguous span that still preserves the referent or constraint. "
    "2. Do not split an object from identifying cues when they refer to the same entity. "
    "3. Keep color markers, point markers, bbox markers, object IDs, frame references, image references, and coordinates attached when they ground the referent. "
    "4. Keep phrases such as 'the center of trash can (red point)' as one grounded mention. "
    "5. Keep phrases such as 'Object80 (red bbox)' as one grounded mention. "
    "6. Keep phrases such as 'the second image, [491, 448, 556, 664]' as one grounded mention. "
    "7. Put relation phrases like 'closest to the observer', 'left of', 'between', or 'in relation to' into spatial_relations. "
    "8. Put geometry phrases like 'based on its center point', '3D center points', 'bounding box', 'Euclidean distance', or measurement units into geometric_constraints. "
    "9. Put viewpoint or coordinate-frame phrases like 'from the observer\'s perspective', 'the observer\'s viewpoint is mirrored', or 'the first image as the primary perspective' into viewpoint_constraints. "
    "10. Put answer-format instructions like 'Select the right option from the choices provided' or 'Your answer can only include one of the options A, B, C, or D' into answer_constraints. "
    "11. Do not move a phrase into another field just to avoid duplication. A phrase may appear in more than one field if it genuinely serves both grounding and constraint roles. "
    "12. Exclude filler text that does not help grounding, spatial reasoning, viewpoint handling, or answer formatting."
)

FEW_SHOT_PROMPT = (
    "Examples:\n"
    "Question: Calculate the shortest Euclidean distance in meters from the center of trash can (red point) to storage shelf (blue point). Calculate or judge based on the 3D center points of these objects. Type in exactly one number as your reply.\n"
    "JSON: {\"grounded_mentions\": [\"the center of trash can (red point)\", \"storage shelf (blue point)\"], \"spatial_relations\": [\"shortest Euclidean distance\"], \"geometric_constraints\": [\"in meters\", \"3D center points\"], \"viewpoint_constraints\": [], \"answer_constraints\": [\"Type in exactly one number as your reply\"]}\n\n"
    "Question: Which object, based on its center point, appears closest to the observer? Choose an image showing the object and mark it with the bounding box. Calculate or judge based on the 3D center points of these objects. The observer’s viewpoint is mirrored by establishing the first image as the primary perspective. Select the right option from the choices provided. A. the first image, [330, 548, 387, 710] B. the third image, [619, 508, 881, 934] C. the second image, [491, 448, 556, 664] D. the first image, [36, 521, 119, 724] Your answer can only include one of the options A, B, C, or D.\n"
    "JSON: {\"grounded_mentions\": [\"the bounding box\", \"the first image, [330, 548, 387, 710]\", \"the third image, [619, 508, 881, 934]\", \"the second image, [491, 448, 556, 664]\", \"the first image, [36, 521, 119, 724]\"], \"spatial_relations\": [\"closest to the observer\"], \"geometric_constraints\": [\"based on its center point\", \"3D center points\", \"bounding box\"], \"viewpoint_constraints\": [\"the observer’s viewpoint is mirrored\", \"the first image as the primary perspective\"], \"answer_constraints\": [\"Select the right option from the choices provided\", \"Your answer can only include one of the options A, B, C, or D\"]}\n\n"
    "Question: Where is the object monitor, shown with a red bbox in the first image, located in the second image? Provide its bounding box.\n"
    "JSON: {\"grounded_mentions\": [\"the object monitor, shown with a red bbox in the first image\", \"the second image\"], \"spatial_relations\": [\"located in\"], \"geometric_constraints\": [\"bounding box\"], \"viewpoint_constraints\": [], \"answer_constraints\": [\"Provide its bounding box\"]}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract grounded mentions and spatial constraints from spatial questions using LLM only.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--error-output", default=None)
    parser.add_argument("--max-records-per-file", type=int, default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=384)
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


def normalize_phrase(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip(" ,.;:"))
    return text


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        item = normalize_phrase(str(item))
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def build_prompts(question: str) -> Tuple[str, str]:
    system_prompt = " ".join([BASE_SYSTEM_PROMPT, EXTRACTION_RULES_PROMPT, FEW_SHOT_PROMPT])
    user_prompt = (
        f"Question: {question}\n\n"
        "Return JSON only with exactly this schema: "
        "{\"grounded_mentions\": [], \"spatial_relations\": [], \"geometric_constraints\": [], \"viewpoint_constraints\": [], \"answer_constraints\": []}."
    )
    return system_prompt, user_prompt


def init_vllm(args: argparse.Namespace):
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


def _coerce_string_list(value) -> List[str]:
    if isinstance(value, list):
        items: List[str] = []
        for item in value:
            if isinstance(item, str):
                items.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("mention") or item.get("span") or item.get("value")
                if text:
                    items.append(str(text))
        return unique_keep_order(items)
    if isinstance(value, str):
        return unique_keep_order([value])
    return []


def empty_schema() -> dict:
    return {key: [] for key in SCHEMA_KEYS}


def parse_model_response(text: str) -> Tuple[Optional[dict], Optional[dict]]:
    payload = (text or "").strip()
    if not payload:
        return None, {"code": "empty_response", "message": "Model returned empty text."}

    data = None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
        if data is None:
            return None, {"code": "invalid_json", "message": "Model output is not valid JSON."}

    parsed = empty_schema()
    for key in SCHEMA_KEYS:
        parsed[key] = _coerce_string_list(data.get(key))

    if not parsed["grounded_mentions"]:
        return None, {
            "code": "grounded_mentions_empty",
            "message": "grounded_mentions is missing or empty.",
        }

    return parsed, None


def iter_samples(input_path: str, limit: Optional[int]) -> Iterator[dict]:
    for source_index, record in enumerate(iter_records(input_path)):
        if limit is not None and source_index >= limit:
            break
        for entry in iter_question_entries(Path(input_path).name, record, source_index):
            yield {
                **entry,
                "cleaned_question": clean_question_text(entry["question"]),
            }


def run_vllm_batch(samples: List[dict], args: argparse.Namespace, tokenizer, llm) -> Tuple[List[dict], List[dict]]:
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    outputs: List[dict] = []
    errors: List[dict] = []

    prompts = []
    for sample in samples:
        system_prompt, user_prompt = build_prompts(sample["cleaned_question"])
        prompts.append(make_chat_prompt(tokenizer, system_prompt, user_prompt))

    generations = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=not args.disable_vllm_tqdm,
    )

    for sample, generation in zip(samples, generations):
        raw_text = generation.outputs[0].text if generation.outputs else ""
        parsed, error = parse_model_response(raw_text)
        if parsed is None:
            parsed = empty_schema()
            outputs.append(
                {
                    "source_file": sample["source_file"],
                    "source_index": sample["source_index"],
                    "id": sample["id"],
                    "turn_index": sample["turn_index"],
                    **parsed,
                    "method": "vllm_batch",
                    "prompt_version": PROMPT_VERSION,
                    "error": {
                        "code": error["code"],
                        "message": error["message"],
                    },
                }
            )
            errors.append(
                {
                    "source_file": sample["source_file"],
                    "source_index": sample["source_index"],
                    "turn_index": sample["turn_index"],
                    "id": sample["id"],
                    "question": sample["cleaned_question"],
                    "error_code": error["code"],
                    "error_message": error["message"],
                    "raw_response": raw_text,
                }
            )
        else:
            outputs.append(
                {
                    "source_file": sample["source_file"],
                    "source_index": sample["source_index"],
                    "id": sample["id"],
                    "turn_index": sample["turn_index"],
                    **parsed,
                    "method": "vllm_batch",
                    "prompt_version": PROMPT_VERSION,
                    "error": None,
                }
            )
    return outputs, errors


def write_outputs(handle, outputs: List[dict], first_item: bool) -> bool:
    for item in outputs:
        if not first_item:
            handle.write(",\n")
        handle.write(json.dumps(item, ensure_ascii=False, indent=2))
        first_item = False
    return first_item


def resolve_error_output_path(args: argparse.Namespace, output_path: Path) -> Path:
    if args.error_output:
        return Path(args.error_output)
    return output_path.with_name(f"{output_path.stem}_errors.jsonl")


def write_error_file(path: Path, errors: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for err in errors:
            handle.write(json.dumps(err, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, llm = init_vllm(args)

    batch_counter = 0
    processed_questions = 0
    all_errors: List[dict] = []

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        first_item = True
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
                    outputs, errors = run_vllm_batch(batch, args, tokenizer, llm)
                    all_errors.extend(errors)
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
                outputs, errors = run_vllm_batch(batch, args, tokenizer, llm)
                all_errors.extend(errors)
                first_item = write_outputs(handle, outputs, first_item)
        handle.write("\n]\n")

    error_path = resolve_error_output_path(args, output_path)
    write_error_file(error_path, all_errors)
    print(f"[info] error_records={len(all_errors)} error_file={error_path}", file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
