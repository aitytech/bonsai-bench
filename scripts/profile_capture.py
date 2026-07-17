#!/usr/bin/env python3
"""Capture a Metal GPU trace of the qmv decode kernels for Xcode analysis (E0).

Produces .gputrace bundles: open in Xcode -> Metal debugger -> Performance
counters (limiter: ALU vs Buffer Read vs occupancy) and shader disassembly.

Usage:
    python profile_capture.py --out /tmp/qmv_lmhead.gputrace --shape lm_head
    python profile_capture.py --out /tmp/decode_step.gputrace --model <mlx-model-dir>
"""
import argparse

import mlx.core as mx

SHAPES = {
    "lm_head": (248320, 5120),
    "mlp_up": (17408, 5120),
    "mlp_down": (5120, 17408),
}


def capture_qmv(out: str, n: int, k: int, bits: int = 2, iters: int = 20):
    x = mx.random.normal((1, k)).astype(mx.float16)
    w = mx.random.normal((n, k)).astype(mx.float16)
    qw, sc, bi = mx.quantize(w, group_size=128, bits=bits)
    # warm up outside the capture
    for _ in range(5):
        mx.eval(mx.quantized_matmul(x, qw, sc, bi, transpose=True, group_size=128, bits=bits))
    mx.synchronize()

    mx.metal.start_capture(out)
    for _ in range(iters):
        mx.eval(mx.quantized_matmul(x, qw, sc, bi, transpose=True, group_size=128, bits=bits))
    mx.synchronize()
    mx.metal.stop_capture()
    print(f"wrote {out}")


def capture_decode_step(out: str, model_dir: str):
    from mlx_lm import load

    loaded = load(model_dir)
    model, tokenizer = loaded[0], loaded[1]  # type: ignore[assignment]
    ids = mx.array([tokenizer.encode("The capital of Vietnam is")])
    logits = model(ids)  # type: ignore[operator]  # prefill, warm caches/compile
    mx.eval(logits)
    mx.synchronize()

    mx.metal.start_capture(out)
    logits = model(ids[:, -1:])  # type: ignore[operator]  # one decode-shaped step
    mx.eval(logits)
    mx.synchronize()
    mx.metal.stop_capture()
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shape", choices=SHAPES.keys())
    ap.add_argument("--model", help="mlx model dir for a full decode-step capture")
    ap.add_argument("--bits", type=int, default=2)
    args = ap.parse_args()

    if args.model:
        capture_decode_step(args.out, args.model)
    elif args.shape:
        n, k = SHAPES[args.shape]
        capture_qmv(args.out, n, k, args.bits)
    else:
        raise SystemExit("pass --shape or --model")
