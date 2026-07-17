# NVIDIA Tesla T4 15GB — Ubuntu 24.04, driver 580.95 (CUDA 13.0)

Server "mx1d": 12 cores, 31GB RAM. Fork binaries `prism-b9591-62061f9` (CUDA 12.8 build).
Measured 2026-07-17 with `scripts/setup_t4.sh` (llama-bench, -fa 1).

| Model | File | pp512 (t/s) | tg128 (t/s) | Notes |
|---|---|---:|---:|---|
| Ternary Bonsai 27B | Q2_0 g128, 6.66 GiB | 308.2 ± 6.5 | 12.7 ± 0.4 | first published T4 numbers for this model (AFAIK) |
| 1-bit Bonsai 27B | Q1_0, 3.53 GiB | _running_ | _running_ | |

## Observations

- Fork CUDA build works on Turing/SM75 out of the box (dp4a decode path) —
  confirms the source-level analysis in the README.
- Prefill is 2x faster than Apple M1 Max (308 vs 152 t/s): CUDA compute wins.
- Generation is bandwidth+overhead bound: 12.7 t/s = ~91 GB/s effective vs
  320 GB/s paper bandwidth (~28% efficiency). The per-token fixed overhead
  (hybrid-attention small ops) hits this 2018 card harder than Apple Metal.
- Practical serving: `./scripts/setup_t4.sh --server` gives an OpenAI-compatible
  endpoint; VRAM headroom (15GB vs ~8GB used) leaves room for long context.
