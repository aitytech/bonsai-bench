# Can the MLX 2-bit decode path be made faster on M1 Max? — No. Here's the proof.

Project: fork [aitytech/mlx](https://github.com/aitytech/mlx) (branch `q2-kernel-opt`),
attempt to close the measured gap between the 2-bit `quantized_matmul` kernel
(215-224 GB/s effective) and the fp16 gemv bandwidth ceiling (337 GB/s) on an
M1 Max, targeting Ternary Bonsai 27B decode (baseline 28.2 tok/s).

**Outcome: five experiment families, all negative or neutral — because the
stack is already at the hardware limit.** 28.2 tok/s is the M1 Max ceiling for
this model within ~5%. Full experiment log below so nobody re-spends this
effort.

## The three-line proof

1. **The kernel sits exactly at the ALU roofline.** 2-bit unpack needs
   ~2.2-2.5 instruction slots per weight (AND + FMA + loads/addressing).
   M1 Max issue budget vs 337 GB/s streaming predicts a practical ceiling of
   **57-64%** of fp16 bandwidth efficiency. Measured: 216.6/337 = **64.3%**.
   Nothing left.
2. **The graph is already ~94% efficient.** End-to-end decode streams
   7.2 GB/token at 204 GB/s vs the kernel's 216 GB/s — norms, attention,
   deltanet and dispatch cost only ~6%.
3. **Removing 240 of ~450 dispatches/token changed nothing** (fusion
   experiment, token-identical outputs, 28.05 → 27.4-28.2 tok/s): Metal
   command-buffer batching already amortizes launches.

## Experiment log (M1 Max 64GB, mlx 0.32.1.dev, self-built, baselines verified)

| # | Experiment | Result | Lesson |
|---|---|---|---|
| E0 | Limiter diagnosis (empirical, no Xcode) | 4-bit reads 2x the bytes of 2-bit but takes only 1.36x the time → **instruction-bound, not bandwidth-bound** | Bit-width scaling comparison is a free limiter probe |
| E1a | uint32 weight load + shift chain | 189 GB/s (**-13%**) | `wi >>= 8` serializes; the scalar path's 4 independent byte loads have better ILP |
| E1b | uint32 load + `uchar4` split | 199 GB/s (-8%) | Compiler already coalesces the byte loads optimally |
| E1c | Full-width masks on uint32, x pre-scaled by 4^k (exact powers of 2) | 215 GB/s (parity on lm_head), **-2.5% e2e** | Extra multiplies hurt the latency-bound small-N attention projections |
| E2 | 8 rows/simdgroup (16 rows/tg) | 205.6 GB/s, **-11% e2e** | Halving threadgroup count kills occupancy; hurts more than activation-load amortization helps. Also: host `bn` must match kernel constexpr in BOTH `qmv` and `gather_qmv`, and only when the fast variant is actually eligible — two correctness traps found by tests |
| E3 | `packs_per_thread` 1→2 | 42 test failures (block 1024 vs K%512 host gate) AND still slower (210.8) | The fork author's `bits==2 ? 1 : 2` choice is correct on M1 |
| E5 | Fuse 128 projection groups (GDN qkv+z+b+a, attn qkv, mlp gate+up) → ~240 fewer dispatches/token | Token-identical, **0% speed change** | In-graph dispatch cost ≈ 0; the ~250µs/op floor only exists for isolated `mx.eval` round-trips (a microbenching trap, not a runtime cost) |
| E4/E6 | half x_thread / software prefetch | Skipped per plan gates | Not occupancy-bound, not memory-latency-bound — ALU-bound |

## Methodology traps discovered (worth more than the experiments)

- **Isolated op microbenches lie below ~1ms**: every `mx.eval` round-trip has
  a ~250µs floor, so a 6MB projection "measures" 6 GB/s while running fine
  in-graph. Only the lm_head shape (1.6ms+) gives clean isolated signal;
  everything else must be judged end-to-end.
- **Metal compiler coalescing**: hand-vectorizing loads that the compiler
  already coalesces makes code slower by destroying ILP. Check disassembly
  (or A/B quickly) before believing a vectorization plan.
- **Host/kernel geometry sync**: `qmv_fast_impl` constants are duplicated in
  two host dispatch sites (`qmv`, `gather_qmv`); the fast gate must only
  widen `bn` when the fast kernel is truly eligible (N and K divisibility).

## What WOULD move the needle

- **Newer silicon**: M4 Max (546 GB/s) → ~38 tok/s by bandwidth alone; M5's
  GPU tensor units change the ALU roofline entirely (PrismML's own numbers:
  26.2 tok/s on M5 Pro whose bandwidth is *lower* than M1 Max — the ALU/tensor
  story, not bandwidth, explains it).
- **Upstream**: when PrismML's 1-bit MLX kernels (mlx#3161) land on a modern
  runtime AND get an ALU-lean unpack, 1-bit's 2x byte advantage could finally
  cash in (~35-40 tok/s theoretical on M1 Max at 4-bit-level kernel
  efficiency — today its kernel runs at 126 GB/s, ALU-bound worse than 2-bit).
- `scripts/fuse_qwen3_5.py` is kept as a research artifact — it's correct
  (token-identical) and may pay on hardware where dispatch cost is real.

## Verification state

- `python/tests/test_quantized.py`: 32/32 OK on final (reverted-to-stock) build.
- Greedy decode token-identity: verified for the fusion path (60-token prompt,
  identical output). Kernel experiments were all reverted, so stock numerics ship.
- `aitytech/mlx` branch `q2-kernel-opt` intentionally carries no kernel diffs —
  the working tree ended clean after reverts; this document is the deliverable.
