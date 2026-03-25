import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 50-sample SPAR preview with grounded-schema vLLM extraction."
    )
    parser.add_argument("--input", default="spar_234k.json")
    parser.add_argument("--output", default="selected_words_spar_preview50.json")
    parser.add_argument(
        "--model-path",
        default="/leonardo_work/EUHPC_D32_006/vllm_model/Qwen3-30B-A3B-Instruct-2507",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hf-home", default="/leonardo_work/EUHPC_D32_006/vllm_model/hf_cache")
    parser.add_argument("--heuristic-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.heuristic_only:
        raise ValueError("--heuristic-only is not supported by select_spatial_grounded_schema_llm_only.py.")

    cmd = [
        sys.executable,
        "scripts/select_spatial_grounded_schema_llm_only.py",
        "--inputs",
        args.input,
        "--output",
        args.output,
        "--max-records-per-file",
        "50",
    ]

    cmd.extend(
        [
            "--model-path",
            args.model_path,
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-model-len",
            str(args.max_model_len),
            "--batch-size",
            str(args.batch_size),
            "--hf-home",
            args.hf_home,
            "--enforce-eager",
        ]
    )

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
