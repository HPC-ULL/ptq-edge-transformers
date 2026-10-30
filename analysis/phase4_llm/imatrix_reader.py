"""Minimal read-only reader for the imatrix GGUF files (GGUF v3, F32 tensors
only), standard library + numpy: the local machine has no `gguf` package.
Format as written by llama.cpp's tools/imatrix/imatrix.cpp (save_imatrix):
per weight tensor `<name>.in_sum2` (sum of x^2 per input channel) and
`<name>.counts` (rows accumulated, one value per matrix)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
GGML_TYPE_F32 = 0


def read_gguf_f32(path: Path) -> Tuple[int, Dict, Dict[str, np.ndarray]]:
    """(version, key-values, {tensor name: array with numpy shape = reversed GGUF dims})."""
    buf = Path(path).read_bytes()
    if buf[:4] != b"GGUF":
        raise ValueError(f"{path}: not a GGUF file")
    off = 4

    def take(fmt):
        nonlocal off
        (v,) = struct.unpack_from(fmt, buf, off)
        off += struct.calcsize(fmt)
        return v

    def string():
        nonlocal off
        n = take("<Q")
        s = buf[off:off + n].decode("utf-8", "replace")
        off += n
        return s

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            it, n = take("<I"), take("<Q")
            return [value(it) for _ in range(n)]
        return take(_SCALAR[t])

    version, n_tensors, n_kv = take("<I"), take("<Q"), take("<Q")
    kv = {}
    for _ in range(n_kv):
        key = string()
        kv[key] = value(take("<I"))
    infos = []
    for _ in range(n_tensors):
        name = string()
        dims = [take("<Q") for _ in range(take("<I"))]
        infos.append((name, dims, take("<I"), take("<Q")))
    align = kv.get("general.alignment", 32)
    data0 = (off + align - 1) // align * align
    tensors = {}
    for name, dims, typ, offset in infos:
        if typ != GGML_TYPE_F32:
            raise ValueError(f"{path}: tensor {name} has ggml type {typ}, expected F32")
        n = int(np.prod(dims))
        tensors[name] = np.frombuffer(buf, dtype="<f4", count=n, offset=data0 + offset).reshape(list(reversed(dims)))
    return version, kv, tensors
