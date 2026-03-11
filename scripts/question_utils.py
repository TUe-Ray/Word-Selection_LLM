from __future__ import annotations

import re
from typing import Iterator, Tuple


ANSWER_SECTION_MARKERS = (
    "choose the correct answer",
    "choose the right response",
    "choose the appropriate response",
    "choose the appropriate option",
    "select the correct answer",
    "select the right option",
    "select the appropriate option",
    "select the appropriate response",
    "your answer can only include",
)

COLOR_PATTERN = r"(?:red|blue|green|yellow|orange|purple|pink|white|black|brown|gray|grey)"
FRAME_OBJECT_PATTERN = re.compile(r"\(\s*in\s+frame-\d+\s*,\s*object\d+\s*\)", re.IGNORECASE)
OBJECT_ID_PATTERN = re.compile(r"\bobject\d+\b", re.IGNORECASE)
NAMED_COLOR_PATTERN = re.compile(
    rf"\b(?!object\d+\b)[a-z][a-z0-9'/-]*(?:\s+[a-z][a-z0-9'/-]*){{0,4}}\s*\(\s*{COLOR_PATTERN}\s+(?:point|bbox|box|marker)\s*\)",
    re.IGNORECASE,
)


def strip_answer_section(text: str) -> str:
    lower = text.lower()
    cut_positions = []

    for marker in ANSWER_SECTION_MARKERS:
        position = lower.find(marker)
        if position >= 0:
            cut_positions.append(position)

    option_block = re.search(r"\bA\.\s.+\bB\.\s", text, flags=re.IGNORECASE | re.DOTALL)
    if option_block:
        cut_positions.append(option_block.start())

    if cut_positions:
        text = text[: min(cut_positions)]

    return text.strip()


def normalize_question(text: str) -> str:
    if not isinstance(text, str):
        return ""

    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"<[^>]+>", line):
            continue
        lines.append(line)

    merged = " ".join(lines)
    merged = merged.replace("**", " ")
    merged = re.sub(r"<[^>]+>", " ", merged)
    merged = re.sub(r"(?:Frame-\d+:\s*)+", " ", merged, flags=re.IGNORECASE)
    merged = strip_answer_section(merged)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged


def iter_human_questions(record: dict) -> Iterator[Tuple[int, str]]:
    conversations = record.get("conversations", [])
    if not isinstance(conversations, list):
        return

    for turn_index, message in enumerate(conversations):
        if not isinstance(message, dict):
            continue

        sender = str(message.get("from", "")).lower()
        if sender not in {"human", "user"}:
            continue

        question = normalize_question(str(message.get("value", "")))
        if question:
            yield turn_index, question


def classify_question_type(question: str) -> str:
    q = question.lower()
    scores = {
        "distance_depth": 0,
        "bbox_localization": 0,
        "relative_position": 0,
    }

    distance_keywords = (
        "distance",
        "depth",
        "meter",
        "meters",
        "centimeter",
        "centimeters",
        "how far",
        "far apart",
        "euclidean",
        "separation",
        "separating",
        "coordinates",
        "coordinate",
        "bev",
    )
    bbox_keywords = (
        "bounding box",
        "bbox",
        "locate",
        "enclose",
        "mark it",
        "mark the",
        "outline",
        "visible and draw",
        "draw the bounding box",
    )
    relative_keywords = (
        "left",
        "right",
        "above",
        "below",
        "relative",
        "relation",
        "position",
        "placement",
        "direction",
        "perspective",
        "viewpoint",
        "orientation",
        "observer",
        "closer",
        "farthest",
        "closest",
        "behind",
        "front",
        "between",
        "movement",
        "moves",
        "moving",
    )

    for keyword in distance_keywords:
        if keyword in q:
            scores["distance_depth"] += 2

    for keyword in bbox_keywords:
        if keyword in q:
            scores["bbox_localization"] += 2

    for keyword in relative_keywords:
        if keyword in q:
            scores["relative_position"] += 2

    if "first image" in q or "second image" in q or "third image" in q:
        scores["bbox_localization"] += 1
        scores["relative_position"] += 1

    if "world coordinate system" in q or "observer's position" in q:
        scores["relative_position"] += 2
        scores["distance_depth"] += 1

    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        return "other"

    return best_type


def classify_reference_style(question: str) -> str:
    if FRAME_OBJECT_PATTERN.search(question):
        return "frame_object_reference"
    if NAMED_COLOR_PATTERN.search(question):
        return "named_color_reference"
    if OBJECT_ID_PATTERN.search(question):
        return "object_id_reference"
    return "plain_object_reference"
