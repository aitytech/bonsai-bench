# Apple M1 Max 32-core GPU, 64GB — macOS 26 (Darwin 25.2)

Measured 2026-07-16/17. Model: Bonsai 27B family (base Qwen3.6-27B).
tg = generation tok/s, pp = prefill tok/s.

## Full backend matrix

| Rank | Config | tg (tok/s) | pp512 (tok/s) | Peak mem | Notes |
|---|---|---:|---:|---|---|
| 1 | Ternary Q2 + **stock MLX 0.32** (mlx-lm 0.31.2) | **28.2** | ~68* | 7.4 GB | best; *pp measured on short prompts only |
| 2 | 1-bit Q1_0 + mainline llama.cpp b9960 | 25.4 | 160 | ~5 GB | Q1_0 is merged upstream |
| 3 | 1-bit Q1_0 + fork llama.cpp (prism-b9591) | 24.0 | 160 | ~5 GB | |
| 4 | 1-bit + PrismML MLX fork (0.31.2-dev) | 20.5 | — | 4.2 GB | fork runtime is the bottleneck |
| 5 | Ternary + PrismML MLX fork | 17.8 | — | 7.4 GB | control test: same model, −37% vs stock |
| 6 | Ternary Q2_0 + fork llama.cpp | 17.5 | 152 | ~8 GB | |
| ✗ | dspark speculative (Metal, server path) | 9.5 | 32 | — | anti-win on Metal; CUDA-only feature |
| ✗ | ngram speculative | 16.5 | — | — | 4/76 draft acceptance |
| ✗ | CPU (NEON, 8 threads) | 4.9 | 9.9 | — | |
| ✗ | Ternary Q2_g64 + mainline b9960 Metal | crash | crash | — | upstream Metal ternary merge is newer than b9960; CPU path works (2 t/s) |

## Bottleneck analysis (why these numbers)

- Decode is **memory-bandwidth-bound**: per token, ~7.2 GB (ternary) of weights
  must stream through RAM. M1 Max practical ceiling ≈ 337 GB/s (measured via fp16 gemv).
- MLX 2-bit `quantized_matmul` achieves **215–224 GB/s** effective (~64% of fp16
  efficiency) → 28 t/s is ~95% of what this kernel can give. Kernel rewrite headroom ≈ 1.5x.
- Fork llama.cpp streams weights at ~217 GB/s BUT adds **~24 ms/token fixed
  overhead** (small ops: 64-layer deltanet linear attention, norms, launch overhead)
  — measured by solving t(bytes) from the Q2_0/Q1_0 pair. This is why 1-bit is only
  1.4x faster than ternary there instead of 1.9x.
- MLX fork's 1-bit kernel is **ALU-bound**: same wall time as the 2-bit kernel
  despite half the bytes (126 GB/s effective). PR pending upstream (mlx#3161).
- Thinking mode dominates perceived latency: a 12-line code answer generated
  1,729 tokens total. Cap with `--reasoning-budget 2048` (llama-server) or
  `thinking_budget_tokens` per request.

## Quality spot-checks (ternary, all passed)

- VN math (compound discount): correct (32%)
- merge_intervals codegen: 5/5 hidden tests
- OpenAI-style tool_calls via `llama-server --jinja`: correct call + args
- Vision (mmproj): studio photo described accurately incl. tiny "FENDI" belt text;
  weak at exact coordinate grounding (`<point>` mode degenerates)
