"""LLM Phase 4 (descriptive): activation outliers from the existing imatrix
files (Part A, local) and weight outliers from the F16 GGUFs (Part B,
cluster). Design and every detail fixed in calibration_protocol_llm.md,
Section 3.0.1; nothing here may change it.

Outputs: results/investigations/llm/phase4/{activations,weights}/.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE4_ROOT = PROJECT_ROOT / "results" / "investigations" / "llm" / "phase4"
IMATRIX_DIR = PROJECT_ROOT / "artifacts" / "llm_imatrix"
BASE_DIR = PROJECT_ROOT / "artifacts" / "llm_base"

MODELS = ("tinyllama_1_1b", "qwen2_5_1_5b", "ministral_3_3b")
LABELS = {"tinyllama_1_1b": "TinyLlama", "qwen2_5_1_5b": "Qwen2.5", "ministral_3_3b": "Ministral-3-3B"}
REPEATS = (1, 2, 3)
EPS = 1e-8

# Protocol table (read from the F16 GGUFs): blocks and expected input dims.
EXPECTED = {
    "tinyllama_1_1b": {"blocks": 22, "embedding": 2048, "attn_output": 2048, "ffn": 5632},
    "qwen2_5_1_5b": {"blocks": 28, "embedding": 1536, "attn_output": 1536, "ffn": 8960},
    "ministral_3_3b": {"blocks": 26, "embedding": 3072, "attn_output": 4096, "ffn": 9216},
}

# The 7 weight tensors per block, and the 4 input types (shared inputs counted once).
TENSORS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
TYPES = {"qkv": ("attn_q", "attn_k", "attn_v"), "attn_output": ("attn_output",),
         "gate_up": ("ffn_gate", "ffn_up"), "down": ("ffn_down",)}


def expected_input_dim(model: str, tensor: str) -> int:
    e = EXPECTED[model]
    return {"attn_output": e["attn_output"], "ffn_down": e["ffn"]}.get(tensor, e["embedding"])

# Part B: output head (listed apart, outside the criterion): untied in
# TinyLlama, tied (token_embd) in Qwen2.5 and Ministral-3-3B.
HEAD = {"tinyllama_1_1b": "output.weight", "qwen2_5_1_5b": "token_embd.weight", "ministral_3_3b": "token_embd.weight"}
