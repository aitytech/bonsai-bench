"""Post-load projection fusion for Qwen3.5-family MLX models (Bonsai 27B).

Every decode step of the 27B launches ~450 tiny gemv kernels; on M1-class GPUs
the per-dispatch cost dominates the small projections (we measured the 2-bit
qmv kernel itself is already at its practical optimum). This module fuses
sibling projections that consume the same input into one quantized gemv:

- GatedDeltaNet (48 layers): in_proj_qkv + in_proj_z + in_proj_b + in_proj_a
  (the b/a projections are N=48 — absurdly small as standalone dispatches)
- Full attention (16 layers): q_proj + k_proj + v_proj
- MLP (64 layers): gate_proj + up_proj

That removes ~240 dispatches per token without touching any model logic: the
wrapped module's original __call__ still runs, its projections just return
pre-computed slices of the fused output.

Usage:
    from fuse_qwen3_5 import fuse_model
    model, tokenizer = load(path)
    fuse_model(model)
"""
import mlx.core as mx
import mlx.nn as nn


class _Slice:
    """Stands in for a projection module; returns its pre-computed slice."""

    def __init__(self):
        self.value = None

    def __call__(self, x):
        return self.value


class FusedInput(nn.Module):
    """One quantized gemv serving several sibling projections of `inner`.

    The named projections on `inner` are replaced with _Slice stand-ins; on
    each call the fused matmul runs first, the stand-ins receive their slice,
    and `inner`'s original __call__ runs unchanged. Safe only because all
    named projections consume the module's input tensor as-is (verified for
    GatedDeltaNet, Qwen3NextAttention, and Qwen3NextMLP).
    """

    def __init__(self, inner, names):
        super().__init__()
        lins = [getattr(inner, n) for n in names]
        assert len({l.group_size for l in lins}) == 1
        assert len({l.bits for l in lins}) == 1
        self.group_size = lins[0].group_size
        self.bits = lins[0].bits
        self.weight = mx.concatenate([l.weight for l in lins], axis=0)
        self.scales = mx.concatenate([l.scales for l in lins], axis=0)
        self.biases = mx.concatenate([l.biases for l in lins], axis=0)
        dims = [l.scales.shape[0] for l in lins]
        self.splits = [sum(dims[: i + 1]) for i in range(len(dims) - 1)]
        self.inner = inner
        self._providers = []
        for n in names:
            p = _Slice()
            setattr(inner, n, p)
            self._providers.append(p)

    def __call__(self, x, *args, **kwargs):
        y = mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        for p, s in zip(self._providers, mx.split(y, self.splits, axis=-1)):
            p.value = s
        return self.inner(x, *args, **kwargs)


def _all_quantized(mod, names):
    return all(isinstance(getattr(mod, n, None), nn.QuantizedLinear) for n in names)


def fuse_model(model):
    """Fuse sibling projections in-place. Returns the number of fused groups."""
    root = getattr(model, "language_model", model)
    layers = root.model.layers
    n = 0
    for layer in layers:
        if getattr(layer, "is_linear", False):
            names = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
            if _all_quantized(layer.linear_attn, names):
                layer.linear_attn = FusedInput(layer.linear_attn, names)
                n += 1
        elif hasattr(layer, "self_attn"):
            names = ["q_proj", "k_proj", "v_proj"]
            if _all_quantized(layer.self_attn, names):
                layer.self_attn = FusedInput(layer.self_attn, names)
                n += 1
        names = ["gate_proj", "up_proj"]
        if hasattr(layer, "mlp") and _all_quantized(layer.mlp, names):
            layer.mlp = FusedInput(layer.mlp, names)
            n += 1
    mx.eval(root.parameters())
    return n
