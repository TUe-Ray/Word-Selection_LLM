from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run spatial term selection preview on first 50 samples from spar_234k.json."
    )
    parser.add_argument("--input", default="spar_234k.json", help="Input JSON file (default: spar_234k.json)")
    parser.add_argument(
        "--output",
        default="selected_words_spar_preview50.json",
        help="Output JSON file (default: selected_words_spar_preview50.json)",
    )
    parser.add_argument("--model", default="qwen14b", help="Model name exposed by your local vLLM server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="EMPTY", help="API key for endpoint")
    parser.add_argument("--max-terms", type=int, default=20, help="Max selected terms per question")
    parser.add_argument("--retries", type=int, default=1, help="Retry count per question")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds")
    parser.add_argument("--heuristic-only", action="store_true", help="Use heuristic-only mode (no LLM call)")
    parser.add_argument("--no-fallback", action="store_true", help="Disable heuristic fallback on LLM errors")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    selector_script = Path(__file__).with_name("select_spatial_words.py")
    command = [
        sys.executable,
        str(selector_script),
        "--inputs",
        args.input,
        "--output",
        args.output,
        "--max-records-per-file",
        "50",
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--api-key",
        args.api_key,
        "--max-terms",
        str(args.max_terms),
        "--retries",
        str(args.retries),
        "--timeout",
        str(args.timeout),
    ]

    if args.heuristic_only:
        command.append("--heuristic-only")
    if args.no_fallback:
        command.append("--no-fallback")

    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
