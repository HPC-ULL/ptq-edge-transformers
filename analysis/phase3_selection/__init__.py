"""Phase 3 "winner" selection (technical_architecture.md Section
9.3): combines each factor's best Phase-2 level into a single Phase-3
configuration per model, with a margin-based safeguard against selecting
noise as a winner. See selection.py for the algorithm, report.py for the
CSV/manifest orchestration around it.
"""

from .selection import (
    CandidateResult,
    FactorDecision,
    Measurement,
    select_level,
    select_phase3_config,
    summarize_repeats,
)

__all__ = [
    "CandidateResult",
    "FactorDecision",
    "Measurement",
    "select_level",
    "select_phase3_config",
    "summarize_repeats",
]
