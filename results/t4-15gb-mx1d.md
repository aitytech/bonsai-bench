# NVIDIA Tesla T4 15GB — Ubuntu 24.04, driver 580.95 (CUDA 13.0)

Server "mx1d": 12 cores, 31GB RAM. Fork binaries `prism-b9591-62061f9` (CUDA 12.8 build).
Measured 2026-07-17 with `scripts/setup_t4.sh` (llama-bench, -fa 1).

| Model | File | pp512 (t/s) | tg128 (t/s) | Notes |
|---|---|---:|---:|---|
| Ternary Bonsai 27B | Q2_0 g128, 6.66 GiB | 308.2 ± 6.5 | 12.7 ± 0.4 | first published T4 numbers for this model (AFAIK) |
| 1-bit Bonsai 27B | Q1_0, 3.53 GiB | 335.8 ± 3.1 | 15.9 ± 0.5 | faster + leaves ~11GB VRAM free |

## Observations

- Fork CUDA build works on Turing/SM75 out of the box (dp4a decode path) —
  confirms the source-level analysis in the README.
- Prefill is 2x faster than Apple M1 Max (308-336 vs 152-160 t/s): CUDA compute wins.
- Two-point decode decomposition (same method as the M1 Max analysis):
  marginal weight-streaming bandwidth = **214 GB/s** — almost identical to
  Apple Metal fork (217) and MLX (215). The 2-bit/1-bit kernels stream equally
  well everywhere. What differs is the **fixed per-token overhead**:
  **~45 ms on T4** vs ~24 ms on M1 Max Metal vs ~0 on stock MLX.
  The 2018 card pays double the small-op tax (64-layer deltanet linear
  attention, kernel launches), which is why tg is only 12.7/15.9 t/s despite
  decent streaming.
- Consequence: on T4, going ternary -> 1-bit buys only +25% tg (overhead
  dominates). Speculative decoding (dspark, CUDA path) should help more here —
  untested so far.
- Practical serving: `BONSAI_VARIANT=1bit ./scripts/setup_t4.sh --server` is the
  sweet spot: 15.9 t/s, OpenAI-compatible endpoint, ~11GB VRAM headroom for
  long context / more slots.
