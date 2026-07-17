#!/usr/bin/env python3
"""Forensic comparison: Ternary Bonsai 27B weights vs original Qwen3.6-27B.

Question: how far did PrismML's QAT move the weights from the pretrained
model? We compare the shipped ternary codes against what pure post-training
ternarization (RTN) of the ORIGINAL weights would produce:

- twn:   classic TWN threshold rule (t = 0.7 * mean|w|), per group of 128
- opt:   per-group L2-optimal ternarization (threshold swept over sorted |w|)

Agreement ~99% => pure PTQ (weights barely moved).
Agreement 80-95% => QAT nudged weights moderately (still recognizably Qwen).
Agreement <70% => heavy retraining reorganized the weights.

Usage:
    python weight_forensics.py --orig-shard <model-0000X.safetensors> \
        --pack <mlx-2bit-dir> --tensor layers.31.mlp.down_proj --sample 4000
"""
import argparse
import json

import mlx.core as mx
import numpy as np


def load_pack_tensor(pack_dir, name):
    idx = json.load(open(f"{pack_dir}/model.safetensors.index.json"))
    full = f"language_model.model.{name}.weight"
    shard = idx["weight_map"][full]
    d = mx.load(f"{pack_dir}/{shard}")
    qw = np.array(d[full], copy=False)                       # (N, K/16) uint32
    sc = np.array(d[full.replace(".weight", ".scales")]).astype(np.float32)
    bi = np.array(d[full.replace(".weight", ".biases")]).astype(np.float32)
    # unpack 2-bit codes -> {0,1,2}; dequant value = code*scale + bias,
    # ternary packs use bias = -scale so codes map to {-1,0,+1}*scale
    K16 = qw.shape[1]
    codes = np.zeros((qw.shape[0], K16 * 16), dtype=np.int8)
    for j in range(16):
        codes[:, j::16] = ((qw >> (2 * j)) & 3).astype(np.int8)
    tern = codes - 1  # bias=-scale => code1 = 0
    return tern, sc, bi


def twn_ternarize(Wg):
    """Classic TWN: threshold 0.7*mean|w| per group. Wg: (G,128)"""
    a = np.abs(Wg)
    thr = 0.7 * a.mean(axis=1, keepdims=True)
    return np.sign(Wg) * (a > thr)


def opt_ternarize(Wg, n_cand=32):
    """Per-group L2-optimal ternary: sweep threshold over |w| quantiles."""
    a = np.abs(Wg)
    best_t = np.zeros_like(Wg, dtype=np.int8)
    best_err = np.full(Wg.shape[0], np.inf)
    qs = np.linspace(0.05, 0.95, n_cand)
    thr_cands = np.quantile(a, qs, axis=1)  # (n_cand, G)
    for c in range(n_cand):
        thr = thr_cands[c][:, None]
        mask = a > thr
        cnt = mask.sum(axis=1)
        cnt = np.maximum(cnt, 1)
        s = (a * mask).sum(axis=1) / cnt
        t = (np.sign(Wg) * mask).astype(np.int8)
        err = ((Wg - s[:, None] * t) ** 2).sum(axis=1)
        upd = err < best_err
        best_err[upd] = err[upd]
        best_t[upd] = t[upd]
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-shard", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tensor", required=True, help="e.g. layers.31.mlp.down_proj")
    ap.add_argument("--orig-prefix", default="model.language_model.")
    ap.add_argument("--sample", type=int, default=4000, help="groups to analyze")
    args = ap.parse_args()

    tern, sc, bi = load_pack_tensor(args.pack, args.tensor)
    orig = mx.load(args.orig_shard)
    W_mx = orig[f"{args.orig_prefix}{args.tensor}.weight"].astype(mx.float32)
    W = np.array(W_mx)  # bf16 has no numpy dtype — convert on the MLX side
    del W_mx
    assert W.shape[0] == tern.shape[0] and W.shape[1] == tern.shape[1], (
        W.shape, tern.shape)

    N, K = W.shape
    G = 128
    rng = np.random.default_rng(0)
    rows = rng.integers(0, N, args.sample)
    gcols = rng.integers(0, K // G, args.sample)

    Wg = np.stack([W[r, c * G : (c + 1) * G] for r, c in zip(rows, gcols)])
    Tg = np.stack([tern[r, c * G : (c + 1) * G] for r, c in zip(rows, gcols)])
    Sg = np.array([sc[r, c] for r, c in zip(rows, gcols)])

    twn = twn_ternarize(Wg)
    opt = opt_ternarize(Wg)

    def agree(a, b):
        return float((a == b).mean())

    nz = Tg != 0
    sign_match = float((np.sign(Wg)[nz] == Tg[nz]).mean())
    # where do the shipped zeros sit in the original magnitude ranking?
    ranks = np.argsort(np.argsort(np.abs(Wg), axis=1), axis=1) / (G - 1)
    zero_rank = float(ranks[Tg == 0].mean())
    deq = Sg[:, None] * Tg
    corr = float(np.mean([np.corrcoef(w, d)[0, 1] for w, d in zip(Wg, deq)
                          if d.std() > 0 and w.std() > 0]))

    print(f"tensor {args.tensor}  ({args.sample} groups of {G})")
    print(f"  pack sparsity (zeros): {float((Tg==0).mean()):.3f}  "
          f"twn: {float((twn==0).mean()):.3f}  opt: {float((opt==0).mean()):.3f}")
    print(f"  agreement pack vs TWN(orig):     {agree(Tg, twn):.4f}")
    print(f"  agreement pack vs L2-opt(orig):  {agree(Tg, opt):.4f}")
    print(f"  agreement TWN vs L2-opt (ref):   {agree(twn, opt):.4f}")
    print(f"  sign(orig) match on pack nonzeros: {sign_match:.4f}")
    print(f"  mean |orig| percentile of pack zeros: {zero_rank:.3f} "
          f"(0=smallest weights, 0.5=random)")
    print(f"  corr(orig, dequant pack): {corr:.4f}")
    print(f"  scale vs mean|orig,nz| ratio: "
          f"{float(np.mean(Sg / np.maximum((np.abs(Wg)*nz).sum(1)/np.maximum(nz.sum(1),1), 1e-8))):.3f}")


if __name__ == "__main__":
    main()
