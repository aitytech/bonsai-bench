#!/usr/bin/env python3
"""Microbenchmark MLX quantized_matmul (qmv decode path) at Bonsai 27B shapes.

Reports effective GB/s per (bits, shape) plus an fp16 gemv reference — the
device's practical bandwidth ceiling. Used to measure kernel optimization work
(see aitytech/mlx branch q2-kernel-opt).

Usage:
    python kernel_microbench.py            # all shapes, bits 2/4/8 + fp16
    python kernel_microbench.py --json out.json
"""
import argparse
import json
import time

import mlx.core as mx

# (rows N, cols K, label) — the matrices streamed every decode step of
# Ternary Bonsai 27B (hidden=5120). lm_head dominates bytes; mlp_* shapes
# repeat 64x per token and dominate op count.
SHAPES = [
    (248320, 5120, "lm_head"),
    (17408, 5120, "mlp_up"),
    (5120, 17408, "mlp_down"),
]
GROUP_SIZE = 128


def bench(fn, iters=100, warmup=15):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write results to JSON file")
    ap.add_argument("--bits", type=int, nargs="*", default=[2, 4, 8])
    args = ap.parse_args()

    print(f"mlx {getattr(mx, '__version__', '?')}  device: {mx.default_device()}")
    rows = []
    for n, k, label in SHAPES:
        x = mx.random.normal((1, k)).astype(mx.float16)
        w = mx.random.normal((n, k)).astype(mx.float16)

        dt = bench(lambda w=w, x=x: x @ w.T)
        gb = n * k * 2 / 1e9
        rows.append({"shape": label, "kind": "fp16", "ms": dt * 1e3, "gbps": gb / dt})

        for bits in args.bits:
            qw, sc, bi = mx.quantize(w, group_size=GROUP_SIZE, bits=bits)
            dt = bench(
                lambda qw=qw, sc=sc, bi=bi, x=x, bits=bits: mx.quantized_matmul(
                    x, qw, sc, bi, transpose=True, group_size=GROUP_SIZE, bits=bits
                )
            )
            gb = (qw.nbytes + sc.nbytes + bi.nbytes) / 1e9
            rows.append(
                {"shape": label, "kind": f"{bits}bit", "ms": dt * 1e3, "gbps": gb / dt}
            )
            del qw, sc, bi
        del w

    print(f"\n{'shape':10s} {'kind':6s} {'ms':>9s} {'GB/s':>8s}")
    for r in rows:
        print(f"{r['shape']:10s} {r['kind']:6s} {r['ms']:9.3f} {r['gbps']:8.1f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
