"""Fold BatchNorm into the preceding convolution, before quantization.

Why: with output_quant, each QuantConv2d quantizes its raw output -- and
without folding that happens BEFORE the BatchNorm, per tensor, while the
channels still have very different scales. Folding (standard in INT8
deployment, and what ONNX Runtime already does to the float graph) moves the
BN into the conv's weight and bias, so the quantized conv output is the
normalized one. Motivated by the gate (investigations/gate_output_quant.py):
output_quant cost MobileViT-S's percentile row 28 points of top-1 but DeiT-Tiny,
which has almost no convolutions, only 1.

Only folds where the BN is known to follow the conv DIRECTLY: timm's
ConvNormAct containers (timm.layers: `.conv` then `.bn`, a BatchNormAct2d that
also applies the activation; timm efficientvit_mit: `.conv` then `.norm`).

After folding, the BN is an exact identity (mean 0, var 1, eps 0, weight 1,
bias 0) and, by default, is KEPT in the model as that identity -- the
configuration fixed for the deployment set (findings_vit.md 7.6). ONNX
Runtime still executes each one as a separate BatchNormalization node (the
Q/DQ pair between conv and BN stops it being fused away: ~9 % of
MobileViT-S's INT8 time on the cluster CPU), but every conv keeps a single
consumer of its DQ and fuses to QLinearConv on both x86 and ARM.

remove_bn=True instead REPLACES each folded BN by exactly what it does after
normalizing -- nn.Sequential(drop, act) for timm's BatchNormAct2d (forward =
batch_norm -> drop -> act), nn.Identity() for a plain BatchNorm2d. Kept only to
reproduce the diagnostic: on the x86 cluster CPU the removal broke 29/35 conv
fusions (the SiLU's DQ then feeds two consumers), making the model 1.31x
slower and changing its accuracy (MobileViT-S percentile rep1 49.00 -> 48.40),
while the Jetson (ARM) fused everything. A BN of any other type is always
kept as the identity and counted, never guessed at. Every BatchNorm2d outside
a known container is left alone and counted.
"""

from __future__ import annotations

from typing import Tuple

FOLDABLE_CONTAINERS = {"ConvNormAct", "ConvBnAct", "ConvNormActAa"}


def fold_conv_bn(conv, bn) -> None:
    """In place: conv(x) followed by bn == folded conv(x) followed by the
    (now identity) bn, for bn in eval mode."""
    import torch

    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("cannot fold a BatchNorm without running statistics")
    with torch.no_grad():
        std = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight if bn.affine else torch.ones_like(std)
        beta = bn.bias if bn.affine else torch.zeros_like(std)
        scale = gamma / std
        conv.weight.mul_(scale.reshape(-1, *([1] * (conv.weight.dim() - 1))))
        old_bias = conv.bias if conv.bias is not None else torch.zeros_like(std)
        new_bias = (old_bias - bn.running_mean) * scale + beta
        if conv.bias is None:
            conv.bias = torch.nn.Parameter(new_bias.detach().clone())
        else:
            conv.bias.copy_(new_bias)

        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)
        bn.eps = 0.0
        if bn.affine:
            bn.weight.fill_(1.0)
            bn.bias.zero_()


def _after_normalization(bn):
    """The module equivalent to what a FOLDED (identity) BN still does, or
    None if its type is not one whose forward is known exactly.

    The activation is forced out of in-place mode: once the BN is gone it
    would otherwise overwrite the (quantized) conv's own output tensor instead
    of the copy the BN used to produce -- and removing the BN changed the
    quantized model's accuracy even though the float model was identical."""
    import torch.nn as nn

    if type(bn) is nn.BatchNorm2d:
        return nn.Identity()
    if type(bn).__name__ == "BatchNormAct2d":  # timm: batch_norm -> drop -> act
        drop, act = getattr(bn, "drop", nn.Identity()), getattr(bn, "act", nn.Identity())
        for module in (drop, act):
            if getattr(module, "inplace", False):
                module.inplace = False
        return nn.Sequential(drop, act)
    return None


def fold_batchnorm(model, remove_bn: bool = False) -> Tuple[int, int, int]:
    """Folds every foldable Conv+BN pair of `model` in place; the folded BN
    stays as an exact identity unless remove_bn. Returns (pairs folded,
    BatchNorm2d modules left unfolded, folded BNs kept in place as an
    identity -- all of them unless remove_bn, then only those of unknown
    type)."""
    import torch.nn as nn

    folded = 0
    kept = 0
    for module in list(model.modules()):
        if type(module).__name__ not in FOLDABLE_CONTAINERS:
            continue
        conv = getattr(module, "conv", None)
        attr = next((a for a in ("bn", "norm") if isinstance(getattr(module, a, None), nn.BatchNorm2d)), None)
        if not isinstance(conv, nn.Conv2d) or attr is None:
            continue
        bn = getattr(module, attr)
        fold_conv_bn(conv, bn)
        folded += 1
        replacement = _after_normalization(bn) if remove_bn else None
        if replacement is None:
            kept += 1
        else:
            setattr(module, attr, replacement)
    remaining = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    return folded, remaining - kept, kept


def fold_batchnorm_checked(model, input_shape, *, rtol: float = 1e-3, label: str = "",
                           remove_bn: bool = False) -> Tuple[int, int, int, float]:
    """fold_batchnorm + proof that the float model's output is unchanged on a
    fixed random input. Raises if the relative difference exceeds rtol."""
    import torch

    model.eval()
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(*input_shape, generator=generator)
    with torch.no_grad():
        before = model(x)
        folded, left, kept = fold_batchnorm(model, remove_bn=remove_bn)
        after = model(x)
    rel = float((after - before).abs().max() / before.abs().max().clamp_min(1e-12))
    print(f"[bn-fold] {label}: {folded} Conv+BN pair(s) folded, {folded - kept} folded BN removed, "
          f"{kept} kept as identity, {left} BatchNorm2d left unfolded, max relative output change {rel:.2e}")
    if rel > rtol:
        raise RuntimeError(f"BatchNorm folding changed the float model's output ({rel:.2e} > {rtol:g}) -- not applied safely")
    return folded, left, kept, rel
