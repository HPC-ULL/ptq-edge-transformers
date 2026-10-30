"""--self-test: confirms output_parser recognizes the real output format of
the onnxruntime_perf_test binary installed in THIS environment, against a
tiny synthetic .onnx artifact -- meant to run once per new device/
environment, before trusting any real measurement taken there (different
devices may run different onnxruntime versions/builds, especially where
Section 6.1's RISC-V contingency applies).

Explicit, deliberately narrow scope: this validates ONLY that
output_parser's regexes still match this binary's stdout format. It does
NOT validate that the input flags themselves (-x, -e, -r, -o, -S, -z, -D,
-I, -y) are all recognized by this particular version/build -- if, say,
this binary was compiled without CUDA support and `-e cuda` is passed, the
binary itself will fail to even start, and self-test cannot anticipate
that; only a real benchmark run against that device/backend combination
can confirm it. The scope warning below is printed on every invocation,
not just failures, so it's never missed by someone skimming a passing run.

The fixture .onnx is generated on the fly via the `onnx` package's helper
API (NOT torch -- this stays a lighter, more likely-to-be-installable
dependency on a constrained edge device than the full torch/Brevitas
toolchain that produces the real artifacts, and this module never needs
torch for anything else). If `onnx` isn't installed on a given device,
pass --self-test-artifact with a tiny .onnx generated elsewhere (e.g. on
the cluster, then copied over) instead of generating one here.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

from .runner import RunFn, run_benchmark

SELF_TEST_RUN_ID = "runner_vit_self_test"
SELF_TEST_REPEATS = 5

SCOPE_WARNING = (
    "--self-test scope: this ONLY confirms output_parser recognizes this "
    "binary's stdout format. It does NOT confirm the input flags "
    "(-x/-e/-r/-o/-S/-z/-D/-I/-y) are valid for this build -- an "
    "unsupported flag (e.g. -e cuda on a CPU-only build) will fail at "
    "execution, not be caught here. Only a real benchmark run can confirm that."
)


# Pinned to the ai.onnx opset 17 / IR version 8 pairing (their official
# joint release, ONNX 1.12) -- NOT just "whatever this build's onnxruntime_
# perf_test happens to accept". helper.make_model() defaults ir_version to
# whatever the installed `onnx` pip package's own newest-known IR version is
# (confirmed for real: ir_version=13, from an unpinned `pip install onnx`),
# which onnxruntime 1.23.2 rejected outright ("Unsupported model IR version:
# 13, max supported IR version: 11" -- caught via a real cluster run, not
# assumed). Pairing ir_version with the opset we actually declare, rather
# than bumping it to just barely fit this one build, keeps the fixture
# loadable on older/more constrained onnxruntime builds too (RISC-V, Jetson
# JetPack-pinned builds -- see technical_architecture.md Section 6.1).
_FIXTURE_OPSET_VERSION = 17
_FIXTURE_IR_VERSION = 8


def generate_fixture_onnx(path: Union[str, Path]) -> Path:
    """A minimal single-node ONNX model (Identity, 1x4 float input/output)
    -- no weights, no real computation, just enough for onnxruntime_perf_test
    to load a session and run inferences against (with -I synthesizing the
    input tensor)."""
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "runner_vit_self_test_graph", [x], [y])
    model = helper.make_model(graph, producer_name="runner_vit_self_test")
    model.opset_import[0].version = _FIXTURE_OPSET_VERSION
    model.ir_version = _FIXTURE_IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return Path(path)


def self_test(
    device_id: str,
    *,
    artifact_path: Optional[Union[str, Path]] = None,
    run_fn: RunFn = subprocess.run,
) -> Dict[str, float]:
    print(SCOPE_WARNING)

    # repeat_id=0 is a placeholder, never meaningful here: write_result=False
    # means this never reaches results_io, so no result_id is ever built
    # from it.
    if artifact_path is not None:
        parsed = run_benchmark(
            SELF_TEST_RUN_ID, artifact_path, device_id, 0,
            repeats=SELF_TEST_REPEATS, dry_run=False, write_result=False, run_fn=run_fn,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="runner_vit_self_test_") as tmp_dir:
            fixture_path = generate_fixture_onnx(Path(tmp_dir) / "self_test_fixture.onnx")
            parsed = run_benchmark(
                SELF_TEST_RUN_ID, fixture_path, device_id, 0,
                repeats=SELF_TEST_REPEATS, dry_run=False, write_result=False, run_fn=run_fn,
            )

    print(f"--self-test OK: output_parser recognized this binary's output format ({device_id}).")
    return parsed
