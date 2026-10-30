#!/usr/bin/env bash
# Diagnostic-only reconnaissance (NOT part of the pipeline): does this
# installed Brevitas version support Microscaling (MX) formats
# (MXINT8/MXFP4/MXFP6/MXFP8)? Run on the cluster (ptq_calib_env active),
# save the output, and share it back -- same pattern as the earlier
# percentile/entropy calibration investigation.
set -e

echo "=== Brevitas version ==="
python -c "import brevitas; print(brevitas.__version__)"

BREVITAS_DIR=$(python -c "import brevitas, os; print(os.path.dirname(brevitas.__file__))")
echo "Brevitas install dir: $BREVITAS_DIR"

echo ""
echo "=== Files mentioning microscaling/MX concepts ==="
grep -rliE "microscal|mxint|mxfloat|mx_quant|ocp_mx|MXWeight|MXAct|block.?floating" \
  "$BREVITAS_DIR" --include="*.py" | tee mx_files_found.txt

echo ""
echo "=== class/def lines in each matching file ==="
while read -r f; do
  echo "--- $f ---"
  grep -n "^class \|^def \|experimental\|EXPERIMENTAL\|not.*tested\|proto" "$f"
done < mx_files_found.txt

echo ""
echo "=== Any file with 'mx' directly in its filename (dedicated module) ==="
find "$BREVITAS_DIR" -iname "*mx*.py"

echo ""
echo "=== brevitas.quant top-level namespace: anything MX-named exported? ==="
python -c "
import brevitas.quant as bq
names = [n for n in dir(bq) if 'mx' in n.lower() or 'MX' in n]
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
