"""Programmatic generation of Phase 1-2 experiment manifests (ViT + LLM branches).

A baseline config is defined once per branch; every other entry is generated
as "baseline + a single factor override" (one-factor-at-a-time), never
hand-written as a static list. Phase 3 (optimal combination vs. aggressive
extreme) depends on Phase-2 results and lives in a separate module, added
once those results exist.
"""

from . import llm_manifest, vit_manifest
from .llm_manifest import generate as generate_llm_manifest
from .vit_manifest import generate as generate_vit_manifest

__all__ = [
    "llm_manifest",
    "vit_manifest",
    "generate_llm_manifest",
    "generate_vit_manifest",
]
