#!/bin/sh
# One-command Bonsai-on-MLX for Apple Silicon: venv + model download + bench + chat.
#
# Usage:
#   ./scripts/run_mlx.sh                 # bench Ternary-Bonsai-27B (default)
#   ./scripts/run_mlx.sh -p "Xin chào"   # one-shot generate instead of bench
#   BONSAI_MLX_MODEL=prism-ml/Ternary-Bonsai-8B-mlx-2bit ./scripts/run_mlx.sh
#
# Requirements: Apple Silicon Mac, python3 >= 3.10.
# IMPORTANT: use STOCK mlx (>= 0.32) — the PrismML mlx fork is based on 0.31.2
# and is ~1.6x SLOWER end-to-end on ternary. Only the 1-bit packs need the fork.
set -e
cd "$(dirname "$0")/.."

MODEL="${BONSAI_MLX_MODEL:-prism-ml/Ternary-Bonsai-27B-mlx-2bit}"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "error: MLX needs an Apple Silicon Mac. For NVIDIA/T4 use scripts/setup_t4.sh" >&2
    exit 1
fi

# venv with pinned, known-good versions (validated 2026-07 on M1 Max)
if [ ! -x .venv/bin/python ]; then
    echo "creating venv ..."
    python3 -m venv .venv
    .venv/bin/pip -q install --upgrade pip
    .venv/bin/pip -q install "mlx-lm==0.31.2" "huggingface-hub>=1.5.0"
fi

# refuse to run on the slow fork runtime by accident
.venv/bin/python - <<'EOF'
import mlx.core as mx, sys
v = getattr(mx, "__version__", "0")
if "dev" in v or tuple(int(x) for x in v.split(".")[:2]) < (0, 32):
    sys.exit(f"error: mlx {v} detected — install stock mlx >= 0.32 (fork runtime is ~1.6x slower)")
print(f"mlx {v} OK")
EOF

if [ -n "$1" ] && [ "$1" = "-p" ]; then
    shift
    .venv/bin/python -m mlx_lm generate --model "$MODEL" --prompt "$1" --max-tokens 2048
else
    .venv/bin/python scripts/bench_mlx.py --model "$MODEL" -n 512
fi
