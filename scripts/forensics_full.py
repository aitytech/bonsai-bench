#!/usr/bin/env python3
"""Full-model forensics across the PrismML 1.7B pair vs original Qwen3-1.7B.

Answers, with numbers, per tensor and aggregated per layer/type:
  A. Is the released "F16" GGUF exactly on the ternary grid {-s,0,+s}
     (pure dequant) or does it leak latent QAT precision?
  B. How far did QAT move each layer from the original Qwen3-1.7B?
     (sign match, correlation, sparsity — the per-layer movement map)
  C. Is the 1-bit model just sign(ternary), or an independent QAT run?

Inputs (downloaded to forensics-17/):
  Ternary-Bonsai-1.7B-F16.gguf   dequantized ternary, fp16
  Ternary-Bonsai-1.7B-Q2_0.gguf  packed ternary codes+scales (fork type 42)
  Bonsai-1.7B-Q1_0.gguf          packed binary codes+scales
  qwen3-1.7b-orig/               original Qwen/Qwen3-1.7B safetensors
"""
import json

from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent / "forensics-17"
G = 128

# ---- GGUF reading (tolerant of the fork's type ids) -------------------------
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as T
from gguf import GGUFReader


def _register(type_id, name, block_bytes):
    if type_id in T._value2member_map_:
        return T(type_id)
    fake = int.__new__(T, type_id)
    fake._name_ = name
    fake._value_ = type_id
    T._value2member_map_[type_id] = fake
    GGML_QUANT_SIZES[fake] = (G, block_bytes)
    return fake


Q2_FORK = _register(42, "FORK_Q2_0_G128", 34)   # 32B codes + fp16 scale
# Q1_0 upstream id — discover from file if unknown; common upstream id is 40
# (registered lazily in load_q1 if the reader hits an unknown type).


def tensors_2d(reader):
    return {t.name: t for t in reader.tensors if len(t.shape) == 2}


def deq_f16(t):
    return np.array(t.data, copy=False).reshape(int(t.shape[1]), int(t.shape[0]))


def unpack_q2(t):
    """fork Q2_0 g128 -> (codes int8 in {-1,0,1}, scales f32), shape (out, in)"""
    rows, cols = int(t.shape[1]), int(t.shape[0])
    raw = np.frombuffer(t.data.tobytes(), dtype=np.uint8).reshape(rows, -1)
    nblk = cols // G
    blocks = raw.reshape(rows, nblk, 34)
    scales = blocks[:, :, :2].copy().view(np.float16).astype(np.float32).reshape(rows, nblk)
    codes_b = blocks[:, :, 2:]  # (rows, nblk, 32) uint8, 4 codes/byte
    c = np.zeros((rows, nblk, G), dtype=np.int8)
    for j in range(4):
        c[:, :, j::4] = ((codes_b >> (2 * j)) & 3).astype(np.int8)
    return c.reshape(rows, cols) - 1, scales


def unpack_q1(t, block_bytes):
    """upstream Q1_0 g128: sign bits + fp16 scale; w = s*(2b-1)."""
    rows, cols = int(t.shape[1]), int(t.shape[0])
    raw = np.frombuffer(t.data.tobytes(), dtype=np.uint8).reshape(rows, -1)
    nblk = cols // G
    blocks = raw.reshape(rows, nblk, block_bytes)
    # assume scale first 2 bytes, then 16 bytes of bits (128 bits)
    scales = blocks[:, :, :2].copy().view(np.float16).astype(np.float32).reshape(rows, nblk)
    bits_b = blocks[:, :, 2:18]
    b = np.unpackbits(bits_b, axis=2, bitorder="little")[:, :, :G]
    return (2 * b.astype(np.int8) - 1).reshape(rows, cols), scales


# ---- name mapping gguf <-> hf ----------------------------------------------
def hf_name(gname):
    if gname == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gname == "output.weight":
        return "lm_head.weight"
    if not gname.startswith("blk."):
        return None
    parts = gname.split(".")
    i, kind = parts[1], parts[2]
    m = {
        "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
        "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
    }.get(kind)
    return f"model.layers.{i}.{m}.weight" if m else None


def load_orig():
    import mlx.core as mx
    d = {}
    for f in sorted((BASE / "qwen3-1.7b-orig").glob("*.safetensors")):
        for k, v in mx.load(str(f)).items():
            d[k] = v
    def get(name):
        v = d.get(name)
        if v is None and name == "lm_head.weight":  # tied embeddings
            v = d.get("model.embed_tokens.weight")
        return None if v is None else np.array(v.astype(mx.float32))
    return get


def main():
    tern_f16 = GGUFReader(str(BASE / "Ternary-Bonsai-1.7B-F16.gguf"))
    tern_q2 = GGUFReader(str(BASE / "Ternary-Bonsai-1.7B-Q2_0.gguf"))
    q1_path = BASE / "Bonsai-1.7B-Q1_0.gguf"
    try:
        bin_q1 = GGUFReader(str(q1_path))
    except (ValueError, KeyError) as e:
        tid = int(str(e).split()[-1]) if str(e).split()[-1].isdigit() else None
        if tid is None:
            raise
        _register(tid, f"Q1_0_id{tid}", 18)
        bin_q1 = GGUFReader(str(q1_path))

    f16_t = tensors_2d(tern_f16)
    q2_t = tensors_2d(tern_q2)
    q1_t = tensors_2d(bin_q1)
    get_orig = load_orig()

    rng = np.random.default_rng(0)
    print(f"{'tensor':34s} {'gridOK':>6s} {'corr':>6s} {'sign%':>6s} {'zero%':>6s} {'b=t%':>6s}")
    rows_out = []
    for name, tq2 in sorted(q2_t.items()):
        if int(tq2.tensor_type) != 42:
            continue
        codes, scales = unpack_q2(tq2)
        n, k = codes.shape
        # --- A: grid check on F16 gguf (sample 64 random groups)
        grid_ok = None
        if name in f16_t:
            W16 = deq_f16(f16_t[name])
            r_i = rng.integers(0, n, 64)
            g_i = rng.integers(0, k // G, 64)
            offgrid = 0
            for r, g in zip(r_i, g_i):
                seg = W16[r, g * G:(g + 1) * G].astype(np.float32)
                s = scales[r, g]
                ref = s * codes[r, g * G:(g + 1) * G]
                if not np.allclose(seg, ref, atol=max(1e-4, abs(float(s)) * 2e-3)):
                    offgrid += 1
            grid_ok = 1 - offgrid / 64
        # --- B: vs original
        hf = hf_name(name)
        W0 = get_orig(hf) if hf else None
        corr = sign = zero = None
        if W0 is not None and W0.shape == codes.shape:
            samp = rng.integers(0, n, 200)
            deq = scales[samp].repeat(G, axis=1) * codes[samp]
            w0 = W0[samp]
            corr = float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(w0, deq) if b.std() > 0]))
            nz = codes[samp] != 0
            sign = float((np.sign(w0)[nz] == codes[samp][nz]).mean())
            zero = float((~nz).mean())
        # --- C: binary vs ternary
        bt = None
        if name in q1_t:
            bcodes, _ = unpack_q1(q1_t[name], GGML_QUANT_SIZES[T(int(q1_t[name].tensor_type))][1])
            if bcodes.shape == codes.shape:
                samp2 = rng.integers(0, n, 200)
                tnz = codes[samp2] != 0
                bt = float((bcodes[samp2][tnz] == codes[samp2][tnz]).mean())
        fmt = lambda v, p=3: ("  --  " if v is None else f"{v:6.{p}f}")
        print(f"{name:34s} {fmt(grid_ok)} {fmt(corr)} {fmt(sign)} {fmt(zero)} {fmt(bt)}")
        rows_out.append(dict(name=name, grid_ok=grid_ok, corr=corr, sign=sign, zero=zero, bin_match=bt))

    with open(BASE / "forensics_full.json", "w") as f:
        json.dump(rows_out, f, indent=2)
    print("\nwrote", BASE / "forensics_full.json")


if __name__ == "__main__":
    main()
