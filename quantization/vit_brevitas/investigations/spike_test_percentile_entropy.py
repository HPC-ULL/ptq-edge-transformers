"""Technical spike: do the newly-composed "percentile" and "entropy"
calibration_algorithm activation quantizers (quantizer_config.py) work end
to end?

NOT part of the pipeline. Unlike spike_test.py (which validated the manual
substitution mechanism itself, across all 3 models, at the minmax baseline
config), this spike is scoped to the two calibration algorithms added on
top of that afterwards -- see quantizer_config.py's module docstring for
exactly how each is composed (ParamFromRuntimePercentileScaling for
percentile; KLMinimizerThreshold wired directly as scaling_stats_impl for
entropy) and why weight quantization stays at baseline (minmax) for both.

Reuses the REAL pipeline code
(quantization.vit_brevitas.quantize.build_quantized_model, which itself
calls quantizer_config.resolve_weight_quant/resolve_act_quant) against a
synthetic manifest-row-shaped dict for deit_tiny -- the fastest of the 3
models, already confirmed to work for every other calibration_algorithm --
rather than reimplementing quantization logic ad hoc. This validates the
actual code path `quantization.vit_brevitas.cli` runs, not a parallel copy
of it. Only the ONNX export step writes to a throwaway temp directory
instead of artifacts/vit/, so this never pollutes real pipeline output.

For each algorithm: quantize+calibrate (32 images) -> export to ONNX
(QCDQ, opset 13) -> load in ONNX Runtime -> run one inference, checking
the output is (1, 1000) logits with no error. One algorithm's failure does
not stop the other from being tried -- a summary prints at the end.

Every import is deliberately deferred into its own step's try/except, so a
missing dependency or a wrong Brevitas import path is attributed to the
right step instead of crashing before any diagnostic is printed.

If this passes, the recommended next step (not run by this script) is the
real pipeline + accuracy check already used for mse/per_channel/asymmetric:

    python -m quantization.vit_brevitas --run-id vit_deit_tiny_phase2_calibration_algorithm_percentile_int8
    python -m evaluation_offline.vit_accuracy --run-id vit_deit_tiny_phase2_calibration_algorithm_percentile_int8 --repeat-id 1
    python -m quantization.vit_brevitas --run-id vit_deit_tiny_phase2_calibration_algorithm_entropy_int8
    python -m evaluation_offline.vit_accuracy --run-id vit_deit_tiny_phase2_calibration_algorithm_entropy_int8 --repeat-id 1

Run directly:

    python quantization/vit_brevitas/spike_test_percentile_entropy.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

# When run as "python quantization/vit_brevitas/spike_test_....py"
# (direct path, not -m), Python only adds the script's own directory to
# sys.path, not the project root -- without this, "from quantization...."
# fails with ModuleNotFoundError even if the project is well structured.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MODEL_NAME = "deit_tiny"
NUM_CALIBRATION_IMAGES = 32
EXPECTED_OUTPUT_SHAPE = (1, 1000)
ONNX_OPSET = 13

ALGORITHMS_TO_TEST = ["percentile", "entropy"]


class SpikeStepFailure(Exception):
    """Aborts only the spike of the current algorithm, not the whole script."""

    def __init__(self, step: str, exc: BaseException):
        self.step = step
        self.exc = exc
        super().__init__(f"{step}: {type(exc).__name__}: {exc}")


def fail(step: str, exc: BaseException) -> None:
    print(f"\n[FAILED] step: {step}")
    print(f"        {type(exc).__name__}: {exc}")
    print()
    traceback.print_exc()
    raise SpikeStepFailure(step, exc) from exc


def synthetic_entry(calibration_algorithm: str) -> dict:
    """A dict shaped like a manifest row (configs/vit_manifest.py),
    equal to the baseline except for calibration_algorithm -- it does not need
    to come from the real manifest, it only needs the fields that
    quantization.vit_brevitas.quantize.build_quantized_model lee."""
    return {
        "run_id": f"spike_{MODEL_NAME}_{calibration_algorithm}",
        "model_name": MODEL_NAME,
        "calibration_algorithm": calibration_algorithm,
        "granularity": "per_tensor",
        "symmetry": "symmetric",
        "calib_set_size": NUM_CALIBRATION_IMAGES,
        "bitwidth": "int8",
    }


def run_spike_for_algorithm(calibration_algorithm: str) -> None:
    print(f"\n{'=' * 70}\ncalibration_algorithm={calibration_algorithm}\n{'=' * 70}")
    entry = synthetic_entry(calibration_algorithm)

    # ------------------------------------------------------------------
    # 1. Quantize + calibrate via the real pipeline (quantize.py), which in
    #    turn calls quantizer_config.resolve_weight_quant/resolve_act_quant
    #    -- this is what is really being validated.
    # ------------------------------------------------------------------
    step = f"[{calibration_algorithm}] 1/3 quantization + calibration (real pipeline)"
    print(f"{step}: quantizing and calibrating with {NUM_CALIBRATION_IMAGES} images...")
    try:
        from quantization.vit_brevitas.quantize import build_quantized_model

        quant_model, input_shape = build_quantized_model(entry, repeat_id=1)
    except Exception as e:
        fail(step, e)
    print(f"{step}: OK (input_shape={input_shape})")

    # ------------------------------------------------------------------
    # 2. Export to ONNX -- to a disposable temporary directory, never to
    #    artifacts/vit/ (this synthetic run_id is not a real manifest
    #    row).
    # ------------------------------------------------------------------
    step = f"[{calibration_algorithm}] 2/3 export to ONNX"
    onnx_dir = Path(tempfile.mkdtemp(prefix=f"vit_spike_calib_algo_{calibration_algorithm}_"))
    onnx_path = onnx_dir / f"{MODEL_NAME}_{calibration_algorithm}.onnx"
    print(f"{step}: exporting to {onnx_path}...")
    try:
        import torch
        from brevitas.export import export_onnx_qcdq

        dummy_input = torch.randn(*input_shape)
        export_onnx_qcdq(
            quant_model,
            args=dummy_input,
            export_path=str(onnx_path),
            opset_version=ONNX_OPSET,
            keep_initializers_as_inputs=False,
        )
    except Exception as e:
        fail(step, e)
    print(f"{step}: OK")

    # ------------------------------------------------------------------
    # 3. Load in ONNX Runtime and run a test inference
    # ------------------------------------------------------------------
    step = f"[{calibration_algorithm}] 3/3 loading in ONNX Runtime + inference"
    print(f"{step}: loading {onnx_path.name} and running a test inference...")
    try:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        dummy_np_input = np.random.randn(*input_shape).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_np_input})
        logits = outputs[0]

        if logits.shape != EXPECTED_OUTPUT_SHAPE:
            raise AssertionError(f"unexpected output shape: {logits.shape}, expected {EXPECTED_OUTPUT_SHAPE}")
    except Exception as e:
        fail(step, e)
    print(f"{step}: OK -- logits with shape {logits.shape}")

    print(f"\n[{calibration_algorithm}] SPIKE PASSED. Test ONNX artifact (disposable): {onnx_path}")


def main() -> int:
    results = {}
    for algorithm in ALGORITHMS_TO_TEST:
        try:
            run_spike_for_algorithm(algorithm)
            results[algorithm] = "OK"
        except SpikeStepFailure as e:
            results[algorithm] = f"FAILED at {e.step}: {type(e.exc).__name__}: {e.exc}"

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for algorithm, result in results.items():
        print(f"  {algorithm}: {result}")

    return 0 if all(r == "OK" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
