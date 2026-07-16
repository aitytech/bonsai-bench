#!/bin/bash
# Bonsai on NVIDIA GPU (Tesla T4 / any CUDA 12.x card, incl. Google Colab).
# Downloads PrismML's prebuilt fork llama.cpp binaries + a model, then benches.
#
# Usage:
#   ./scripts/setup_t4.sh                    # Ternary 27B (needs ~10GB VRAM)
#   BONSAI_VARIANT=1bit ./scripts/setup_t4.sh  # 1-bit 27B (needs ~6GB VRAM)
#   ./scripts/setup_t4.sh --server           # also start OpenAI-compatible server :8080
#
# On Colab: run in a cell:
#   !git clone https://github.com/<you>/bonsai-bench && cd bonsai-bench && bash scripts/setup_t4.sh
#
# Notes for T4 specifically (validated against fork source, prism-b9591):
# - decode kernels use dp4a -> works on SM75 (Turing). Prefill has no
#   Hopper fast-path on T4, expect slower pp than modern cards.
# - ternary Q2_0 g128 loads ONLY on these fork binaries, not mainline llama.cpp.
set -e
cd "$(dirname "$0")/.."

RELEASE_TAG="prism-b9591-62061f9"
VARIANT="${BONSAI_VARIANT:-ternary}"

# ── CUDA detection ──
CUDA_VER=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version:[[:space:]]*\([0-9]*\.[0-9]*\).*/\1/p')
[ -z "$CUDA_VER" ] && { echo "error: nvidia-smi not found — is this a GPU machine?" >&2; exit 1; }
MAJOR=${CUDA_VER%%.*}; MINOR=${CUDA_VER#*.}
if [ "$MAJOR" -ge 13 ] || { [ "$MAJOR" -eq 12 ] && [ "$MINOR" -ge 8 ]; }; then CUDA_TAG="12.8"; else CUDA_TAG="12.4"; fi
echo "CUDA $CUDA_VER -> using fork build for CUDA $CUDA_TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Binaries ──
if [ ! -x bin/cuda/llama-bench ]; then
    ASSET="llama-${RELEASE_TAG}-bin-linux-cuda-${CUDA_TAG}-x64.tar.gz"
    echo "downloading $ASSET ..."
    mkdir -p bin/cuda
    curl -L --fail --progress-bar \
      "https://github.com/PrismML-Eng/llama.cpp/releases/download/${RELEASE_TAG}/${ASSET}" \
      -o /tmp/llama-bin.tar.gz
    tar -xzf /tmp/llama-bin.tar.gz -C bin/cuda --strip-components=1 || tar -xzf /tmp/llama-bin.tar.gz -C bin/cuda
    rm -f /tmp/llama-bin.tar.gz
fi
export LD_LIBRARY_PATH="$PWD/bin/cuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Model ──
pip -q install "huggingface-hub>=1.5.0" 2>/dev/null || true
if [ "$VARIANT" = "1bit" ]; then
    REPO="prism-ml/Bonsai-27B-gguf";        FILE="Bonsai-27B-Q1_0.gguf"
else
    REPO="prism-ml/Ternary-Bonsai-27B-gguf"; FILE="Ternary-Bonsai-27B-Q2_0.gguf"
fi
if [ ! -f "models/$FILE" ]; then
    echo "downloading $REPO/$FILE ..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('$REPO', '$FILE', local_dir='models')"
fi

# ── Bench ──
echo; echo "=== llama-bench (pp512 / tg128) ==="
./bin/cuda/llama-bench -m "models/$FILE" -fa 1

# ── Optional server ──
if [ "$1" = "--server" ]; then
    echo "starting OpenAI-compatible server on :8080 (Ctrl-C to stop)"
    ./bin/cuda/llama-server -m "models/$FILE" -ngl 99 -fa on -c 16384 --jinja --host 0.0.0.0 --port 8080
fi

echo
echo "done. please contribute your numbers back: results/ (markdown row is in the bench output)"
