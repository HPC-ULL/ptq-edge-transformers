#!/usr/bin/env bash
# Diagnostic-only reconnaissance (NOT part of the pipeline): does the QCDQ
# ONNX exporter in the installed Brevitas version support representing
# 6-bit and 4-bit tensors, or does it hard-restrict to 8-bit the same way
# it hard-restricts against groupwise (is_groupwise) quantization? Separate
# question from MX's groupwise blocker -- do not assume the answer carries
# over.
set -uo pipefail  # no -e: we want every check to run even if one fails

BREVITAS_DIR=$(python -c "import brevitas, os; print(os.path.dirname(brevitas.__file__))")
echo "Brevitas dir: $BREVITAS_DIR"
echo "Brevitas version:"; python -c "import brevitas; print(brevitas.__version__)"
echo "onnx package version:"; python -c "import onnx; print(onnx.__version__)"
echo "onnx package's own max supported opset (defn.ai_onnx_opset_version or similar):"
python -c "
import onnx
print('onnx.defs.onnx_opset_version():', onnx.defs.onnx_opset_version())
"

echo ""
echo "=== bit_width-related asserts/restrictions in the QCDQ export path ==="
grep -rn "bit_width" "$BREVITAS_DIR/export/onnx/standard/qcdq/" 2>/dev/null

echo ""
echo "=== Any assert/raise near 'bit_width' in the whole onnx export tree ==="
grep -rn -B2 -A2 "assert.*bit_width\|raise.*bit_width" "$BREVITAS_DIR/export/onnx/" 2>/dev/null

echo ""
echo "=== How does the QCDQ handler pick the ONNX storage dtype (int8 vs int4/uint4)? ==="
grep -rn -B3 -A15 "def.*dtype\|torch.int8\|torch.uint8\|TensorProto.INT8\|TensorProto.UINT8\|INT4\|UINT4" "$BREVITAS_DIR/export/onnx/standard/qcdq/handler.py" 2>/dev/null

echo ""
echo "=== Does Brevitas reference ONNX INT4/UINT4 (opset 21+) anywhere at all? ==="
grep -rln "INT4\|UINT4\|int4\|uint4" "$BREVITAS_DIR/export/" 2>/dev/null

echo ""
echo "=== QCDQ manager's declared/required opset version(s) ==="
grep -rn "opset_version\|OPSET" "$BREVITAS_DIR/export/onnx/standard/qcdq/manager.py" 2>/dev/null

echo ""
echo "=== Full QCDQ handler source (for manual inspection of dtype selection) ==="
cat "$BREVITAS_DIR/export/onnx/standard/qcdq/handler.py" 2>/dev/null

echo ""
echo "=== Does brevitas' QuantConv2d/QuantLinear even accept bit_width=6 or 4 without erroring, independent of export? ==="
python -c "
import torch
from brevitas.nn import QuantLinear
import brevitas.quant as bq

for bw in (8, 6, 4):
    try:
        class W(bq.Int8WeightPerTensorFloat):
            bit_width = bw
        layer = QuantLinear(4, 4, bias=False, weight_quant=W)
        layer.eval()
        with torch.no_grad():
            out = layer(torch.randn(1, 4))
        print(f'bit_width={bw}: forward OK, output shape={tuple(out.shape)}')
    except Exception as e:
        print(f'bit_width={bw}: FAILED: {type(e).__name__}: {e}')
"
