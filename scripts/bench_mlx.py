#!/usr/bin/env python3
"""Benchmark Bonsai / Ternary-Bonsai MLX packs — steady-state generation speed.

Measures prompt tps, generation tps (over a long-enough run to be steady-state),
and peak memory. Prints a markdown row ready to paste into results/.

Usage:
    python bench_mlx.py --model <hf-repo-or-local-dir> [-n 512] [--json out.json]

Examples:
    python bench_mlx.py --model prism-ml/Ternary-Bonsai-27B-mlx-2bit
    python bench_mlx.py --model ./models/Ternary-Bonsai-27B-mlx-2bit -n 512
"""
import argparse
import json
import platform
import subprocess
import time

import mlx.core as mx
from mlx_lm import load, stream_generate

try:  # import location moved across mlx-lm versions
    from mlx_lm.sample_utils import make_sampler
except ImportError:
    from mlx_lm.generate import make_sampler

# A prompt that produces long, regular output — good for steady-state timing.
BENCH_PROMPT = "Count from 1 to 300, one number per line."


def hw_info() -> str:
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        chip = platform.processor() or "unknown"
    mem_gb = 0
    try:
        mem_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) // 2**30
    except Exception:
        pass
    return f"{chip} {mem_gb}GB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", "--max-tokens", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--json", help="also write results to this JSON file")
    args = ap.parse_args()

    print(f"loading {args.model} ...")
    t0 = time.perf_counter()
    loaded = load(args.model)  # (model, tokenizer[, config]) across mlx-lm versions
    model, tokenizer = loaded[0], loaded[1]
    load_s = time.perf_counter() - t0

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": BENCH_PROMPT}],
        add_generation_prompt=True,
        enable_thinking=False,  # benchmark generation, not reasoning length
        tokenize=False,
    )

    sampler = make_sampler(temp=args.temp, top_p=0.95)
    last = None
    t0 = time.perf_counter()
    for resp in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, sampler=sampler
    ):
        last = resp
    wall_s = time.perf_counter() - t0
    if last is None:
        raise SystemExit("generation produced no tokens")

    peak_gb = mx.get_peak_memory() / 2**30
    result = {
        "model": args.model,
        "hardware": hw_info(),
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "load_s": round(load_s, 1),
        "prompt_tokens": last.prompt_tokens,
        "prompt_tps": round(last.prompt_tps, 1),
        "gen_tokens": last.generation_tokens,
        "gen_tps": round(last.generation_tps, 1),
        "wall_s": round(wall_s, 1),
        "peak_gb": round(peak_gb, 2),
    }

    print(json.dumps(result, indent=2))
    print("\nmarkdown row:")
    print(
        f"| {result['hardware']} | MLX {result['mlx_version']} | {args.model.split('/')[-1]} "
        f"| {result['prompt_tps']} | {result['gen_tps']} | {result['peak_gb']} GB |"
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
