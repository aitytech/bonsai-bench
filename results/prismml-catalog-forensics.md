# PrismML catalog archaeology — what 33 public repos reveal about the method

Phase A of the "make-our-own-Bonsai" project. Scripts: `catalog` census inline,
`forensics_full.py` (full-model, 197 tensors, Ternary-Bonsai-1.7B + Bonsai-1.7B
vs original Qwen3-1.7B). Raw per-tensor numbers: `results/forensics_full_1.7b.json`.

## Census: three product lines nobody talks about

| Line | What it is | Why it matters |
|---|---|---|
| `*-unpacked` (every model) | Dequantized-ternary FP16 safetensors, HF format | Run Bonsai with stock HF tooling; confirmed NO latent leak (below) |
| `*-AWQ-4bit` + sglang instructions (8xH100 TP/DP) | Ternary weights re-packed in AWQ INT4 containers | **Their official datacenter serving path** — the "does vLLM support ternary" answer is "no, so they ship AWQ containers for sglang/vLLM ecosystems" |
| `bonsai-image-*-4B` | Ternary FLUX.2 Klein 4B text-to-image DiT: 1.21GB transformer (gemlite int2 CUDA / MLX 2-bit), HQQ-4bit text encoder, fp16 VAE | The compression method generalizes beyond LLMs; 4.5s/1024px on RTX 3080 |

Download counts: 1-bit 27B dominates (1.05M), ternary 27B 200K, everything else <30K.
Both early whitepapers (1-bit 8B 2026-03-31, ternary 8B 2026-04-16) disclose zero
method detail beyond "proprietary Caltech IP" — the weights are the only testimony.

## Full-model forensics (1.7B trio), headline numbers

**A. The released F16/unpacked files are pure dequantized ternary.** Grid check
{-s,0,+s}: 196/197 tensors perfectly on-grid in all sampled groups (worst tensor:
63/64 groups, consistent with fp16 rounding at large scales). No latent QAT
weights have leaked anywhere in the catalog.

**B. The smaller the model, the more QAT reorganizes it.**

| metric (blocks avg) | 27B (round-1, mlp L31) | 1.7B (full model) |
|---|---:|---:|
| corr(orig, ternary) | 0.84 | 0.65-0.72 |
| sign preservation | 97.8% | 84-91% |
| zeros | 29.7% | 38-41% |

The 1.7B had to move much further from Qwen to survive ternarization — matching
its lower benchmark retention (~85% vs ~95% at 8B/27B) and ParetoQ's finding that
low-bit hits small models hardest.

**C. Per-projection fingerprint** (1.7B, mean corr to original, most→least preserved):
attn_v 0.723 > ffn_down 0.705 > attn_output 0.701 > ffn_gate 0.645 > ffn_up 0.640
> attn_k 0.618 > attn_q 0.614. QAT reshapes the query/key/FFN-input side the most
and preserves the value/output pathway. Depth trend mild: last 4 layers moved most
(0.628 vs 0.67 elsewhere).

**D. Binary is NOT sign(ternary) — but they share a pipeline.** On ternary-nonzero
positions, the binary model's sign agrees 87-96% in transformer blocks (high, but
far from derivation) — yet **token_embd agrees 100.0% exactly**. Best explanation:
staged training from a shared checkpoint/artifact (e.g. binary run initialized
from the ternary solution or both from one QAT intermediate), after which blocks
diverged in further training while embeddings barely moved. A one-shot "derive
binary from ternary" is ruled out; two fully independent runs are also unlikely
(the exact embedding match has ~zero probability across independent runs).

## Implications for our Phase B pilot (QAT Qwen 1.7B on RTX 5080)

1. Expect to NEED substantial weight movement at 1.7B — plain calibration will not
   cut it; the distill/QAT loop is doing real work at this scale (validates the
   BitDistill-style recipe and dooms cheap-PTQ shortcuts).
2. Layer-uniform treatment is what they ship (uniform zero fractions, no
   mixed-precision escape hatches) — our pilot can stay uniform too.
3. Their sparsity lands at 38-41% for 1.7B (below the ~46% L2-optimal) — a
   learned, not thresholded, outcome; our pilot can track sparsity as a training
   health metric against these reference values.
4. Staged binary-from-ternary suggests a cost saver: if we ever want both
   variants, train ternary first, initialize binary from it.
