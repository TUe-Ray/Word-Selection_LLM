from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_json_array(path: Path, chunk_size: int = 1_048_576) -> Iterator[Any]:
    """Stream items from a top-level JSON array without loading the full file."""
    decoder = json.JSONDecoder()

    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        started = False
        finished = False
        eof = False

        while not finished:
            if not eof and len(buffer) < chunk_size:
                chunk = handle.read(chunk_size)
                if chunk == "":
                    eof = True
                else:
                    buffer += chunk

            idx = 0

            if not started:
                while idx < len(buffer) and buffer[idx].isspace():
                    idx += 1

                if idx >= len(buffer):
                    if eof:
                        raise ValueError(f"Empty JSON file: {path}")
                    buffer = ""
                    continue

                if buffer[idx] != "[":
                    raise ValueError(f"Expected top-level JSON array in {path}")

                started = True
                idx += 1

            while True:
                while idx < len(buffer) and (buffer[idx].isspace() or buffer[idx] == ","):
                    idx += 1

                if idx >= len(buffer):
                    break

                if buffer[idx] == "]":
                    finished = True
                    idx += 1
                    break

                try:
                    item, end = decoder.raw_decode(buffer, idx)
                except json.JSONDecodeError:
                    break

                yield item
                idx = end

            buffer = buffer[idx:]

            if eof:
                if not finished and buffer.strip():
                    raise ValueError(f"Unexpected EOF while parsing JSON array: {path}")
                break
