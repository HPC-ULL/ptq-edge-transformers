"""Build a Brevitas-quantized, calibrated ViT model in memory.

Manual nn.Linear/nn.Conv2d -> QuantLinear/QuantConv2d substitution, not
brevitas.graph.quantize.quantize() -- the automatic FX-graph-based path hit
three separate, ViT-specific failures in a row during spike testing
(missing GraphModule trace; is_causal traced as an FX Proxy, breaking
timm's internal control flow; and finally an unconditional "Residual
handler could not find a solution to align scale factors across ADDs and
CATs" with no parameter to disable it -- confirmed by reading
brevitas/graph/quantize.py directly). See spike_test.py's module docstring
for the full account.

Manual substitution needs no FX tracing at all: it walks the module tree
and swaps nn.Linear/nn.Conv2d for their Brevitas Quant equivalents in
place, copying the pretrained weights over. Everything else (LayerNorm,
GELU, softmax, the residual adds themselves) stays plain float between
quantized layers -- the standard QCDQ dequantize/float-op/requantize
pattern, not a lesser approximation of the automatic path.

Two regimes, chosen EXPLICITLY per call (build_quantized_model's
`output_quant`, default False):

- output_quant=False (the default): every quantized layer quantizes its
  INPUT only. This is the calibration ablation's regime -- the 72 artifacts
  of Phases 1-3 under artifacts/vit/ and every result derived from them.
  The pipeline CLI (cli.py) never sets it, so running the ablation pipeline
  can never silently change that regime.
- output_quant=True: each layer also quantizes its OUTPUT, with the same
  activation quantizer as its input. The output quantizer is what lets ONNX
  Runtime fuse the exported QCDQ graph into real integer kernels (QGemm /
  QLinearConv); input-only graphs have no Q after the op and run as
  explicit Q/DQ + float compute, slower than FP32 (investigations/
  probe_qdq_fusion.py, probe_output_quant.py). Used only by the deployment
  set (deployment.py, configs/vit_deployment_manifest.py) and its accuracy
  gate (investigations/gate_output_quant.py).

Results of the two regimes must never be mixed.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .paths import timm_model_name
from .quantizer_config import resolve_act_quant, resolve_weight_quant


def load_pretrained_model(model_name: str):
    """timm model, eval mode, fused SDPA attention disabled.

    set_fused_attn(False) was needed for FX-tracing in the automatic
    quantize() path (F.scaled_dot_product_attention isn't traceable); it's
    kept here too since Brevitas's ONNX export also traces the model
    internally, and this is a harmless no-op for any model whose attention
    implementation doesn't consult this flag at all.
    """
    import timm
    from timm.layers import set_fused_attn

    set_fused_attn(False)
    model = timm.create_model(timm_model_name(model_name), pretrained=True)
    model.eval()
    return model


def resolve_input_shape(model) -> Tuple[int, int, int, int]:
    """(1, C, H, W) from timm's own resolved preprocessing config for this
    model -- not every model uses 224x224 (e.g. some MobileViT variants
    expect 256x256), so this can't be hardcoded."""
    from timm.data import resolve_data_config

    config = resolve_data_config({}, model=model)
    c, h, w = config["input_size"]
    return (1, c, h, w)


def _quantize_linear(child, weight_quant, act_quant, output_quant: bool = False):
    import torch
    from brevitas.nn import QuantLinear

    quant_linear = QuantLinear(
        in_features=child.in_features,
        out_features=child.out_features,
        bias=child.bias is not None,
        weight_quant=weight_quant,
        input_quant=act_quant,
        output_quant=act_quant if output_quant else None,
        return_quant_tensor=False,
    )
    quant_linear.weight.data = child.weight.data.clone()
    if child.bias is not None:
        quant_linear.bias.data = child.bias.data.clone()
    return quant_linear


def _quantize_conv2d(child, weight_quant, act_quant, output_quant: bool = False):
    from brevitas.nn import QuantConv2d

    quant_conv = QuantConv2d(
        in_channels=child.in_channels,
        out_channels=child.out_channels,
        kernel_size=child.kernel_size,
        stride=child.stride,
        padding=child.padding,
        dilation=child.dilation,
        groups=child.groups,
        bias=child.bias is not None,
        weight_quant=weight_quant,
        input_quant=act_quant,
        output_quant=act_quant if output_quant else None,
        return_quant_tensor=False,
    )
    quant_conv.weight.data = child.weight.data.clone()
    if child.bias is not None:
        quant_conv.bias.data = child.bias.data.clone()
    return quant_conv


def quantize_module_tree(module, weight_quant, act_quant, output_quant: bool = False) -> None:
    import torch.nn as nn

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, _quantize_linear(child, weight_quant, act_quant, output_quant))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, _quantize_conv2d(child, weight_quant, act_quant, output_quant))
        else:
            quantize_module_tree(child, weight_quant, act_quant, output_quant)


def build_quantized_model(entry: Dict, repeat_id: int, fold_bn: bool = False, remove_bn: bool = False,
                          output_quant: bool = False):
    """Load, quantize (manual substitution), and calibrate the model for
    one manifest row's (run_id, repeat_id) job.

    Returns (quant_model, input_shape) -- the caller (export/vit_to_onnx)
    exports it to ONNX while it's still in memory. Nothing is persisted to
    disk here: unlike the LLM branch, there is no intermediate artifact to
    resume from between quantization and export, since a live
    Brevitas-quantized model isn't something this project serializes on
    its own.

    fold_bn (default False, i.e. the pipeline as it runs today) folds
    BatchNorm into the preceding convolutions first -- see bn_fold.py; only
    investigations/gate_output_quant.py --fold-bn sets it, to decide whether
    the regenerated ablation should. The folded BNs stay as exact identities
    unless remove_bn (diagnostic only, --remove-bn; see bn_fold.py).

    output_quant (default False: the ablation's input-only regime) also
    quantizes every layer's output -- see the module docstring. Always
    passed explicitly by its only callers (deployment.py, the gate).
    """
    import torch
    from brevitas.graph.calibrate import calibration_mode

    from .calibration import load_calibration_batch

    model = load_pretrained_model(entry["model_name"])
    input_shape = resolve_input_shape(model)
    if fold_bn:
        from .bn_fold import fold_batchnorm_checked

        fold_batchnorm_checked(model, input_shape, label=entry["run_id"], remove_bn=remove_bn)

    weight_quant = resolve_weight_quant(entry["calibration_algorithm"], entry["granularity"], entry["bitwidth"])
    act_quant = resolve_act_quant(entry["calibration_algorithm"], entry["symmetry"], entry["bitwidth"])

    calib_batch = load_calibration_batch(model, entry["calib_set_size"], repeat_id)

    quantize_module_tree(model, weight_quant, act_quant, output_quant=output_quant)
    model.eval()

    with torch.no_grad(), calibration_mode(model):
        model(calib_batch)

    return model, input_shape
