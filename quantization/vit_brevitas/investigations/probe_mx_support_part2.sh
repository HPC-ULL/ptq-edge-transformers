#!/usr/bin/env bash
# Follow-up to probe_mx_support.sh, which died early (set -e + a zero-match
# grep). No set -e here, so it runs to completion regardless of individual
# grep exit codes.

BREVITAS_DIR=$(python -c "import brevitas, os; print(os.path.dirname(brevitas.__file__))")

echo "=== Full source: brevitas/quant/mx_quant_ocp.py ==="
cat "$BREVITAS_DIR/quant/mx_quant_ocp.py"

echo ""
echo "=== MX-related matches in export/inference/handler.py (ONNX/inference export path) ==="
grep -n -B2 -A5 -iE "mx|microscal|groupwise" "$BREVITAS_DIR/export/inference/handler.py"

echo ""
echo "=== Does export_onnx_qcdq (our export path) mention MX/groupwise at all? ==="
grep -rn -iE "mx|microscal|groupwise" "$BREVITAS_DIR/export/onnx/" 2>/dev/null

echo ""
echo "=== Any file with 'mx' directly in its filename ==="
find "$BREVITAS_DIR" -iname "*mx*.py"

echo ""
echo "=== brevitas.quant top-level namespace: anything MX-named exported? ==="
python -c "
import brevitas.quant as bq
names = [n for n in dir(bq) if 'mx' in n.lower()]
print(names)
"

echo ""
echo "=== brevitas.quant.experimental namespace, if it exists ==="
python -c "
try:
    import brevitas.quant.experimental as bqe
    print(dir(bqe))
except ImportError as e:
    print('no brevitas.quant.experimental:', e)
"

echo ""
echo "=== MXMixin / GroupwiseX mixin definitions in brevitas.core.scaling (block-scale implementation) ==="
grep -rln -iE "mx|groupwise|microscal" "$BREVITAS_DIR/core/scaling/" 2>/dev/null
