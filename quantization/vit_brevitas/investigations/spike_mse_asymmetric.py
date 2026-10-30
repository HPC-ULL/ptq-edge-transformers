"""One-off spike (NOT part of the pipeline): can activations be quantized
with mse + asymmetric (deployment set, DeiT-Tiny asymmetric row)?

The stock preset ShiftedUint8ActPerTensorFloatMSE failed for real inside the
gate with `DependencyError: 'ShiftedUint8ActPerTensorFloatMSE' can not
resolve attribute 'dtype'`. This builds ONE QuantLinear exactly like
quantize.py's _quantize_linear (weights mse per-tensor, the candidate as
input_quant AND output_quant, return_quant_tensor=False), then runs what the
gate runs -- calibration pass, eval forward, QCDQ export -- for each
candidate, and prints OK or the full traceback. Seconds, CPU only.

    python quantization/vit_brevitas/investigations/spike_mse_asymmetric.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def candidates():
    import torch
    from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloatMSE

    class WithDtypeDevice(ShiftedUint8ActPerTensorFloatMSE):
        dtype = torch.float32
        device = torch.device("cpu")

    return {"stock ShiftedUint8ActPerTensorFloatMSE": ShiftedUint8ActPerTensorFloatMSE,
            "stock + dtype/device declared": WithDtypeDevice}


def run(act_quant) -> None:
    import brevitas.quant as bq
    import torch
    from brevitas.export import export_onnx_qcdq
    from brevitas.graph.calibrate import calibration_mode
    from brevitas.nn import QuantLinear

    torch.manual_seed(0)
    layer = QuantLinear(64, 32, bias=True, weight_quant=bq.Int8WeightPerTensorFloatMSE,
                        input_quant=act_quant, output_quant=act_quant, return_quant_tensor=False)
    layer.eval()
    calib = torch.randn(128, 64) * 3 + 1  # skewed, so an asymmetric range is non-trivial
    with torch.no_grad(), calibration_mode(layer):
        layer(calib)
    with torch.no_grad():
        out = layer(torch.randn(4, 64))
    assert torch.isfinite(out).all()
    with tempfile.TemporaryDirectory() as tmp:
        export_onnx_qcdq(layer, args=torch.randn(1, 64), export_path=str(Path(tmp) / "spike.onnx"), opset_version=13)


def main() -> int:
    import brevitas

    print(f"brevitas {brevitas.__version__}")
    ok = {}
    for name, act_quant in candidates().items():
        try:
            run(act_quant)
            ok[name] = True
            print(f"[OK]   {name}: calibration + eval forward + QCDQ export")
        except Exception:
            ok[name] = False
            print(f"[FAIL] {name}:")
            traceback.print_exc()
    return 0 if any(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
