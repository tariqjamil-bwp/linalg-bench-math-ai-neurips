# Module:    subcat_config_ibp.py
# Version:   1.0
"""subcat_config_ibp.py — IBP subcategory config for the depth-ladder run.

This experiment has no judge stage, so this module holds only what the
inference lane needs: the answer extractor, exposed through the same
SUBCAT_CONFIGS() interface the callers already use.

WHAT EXTRACTION IS FOR
──────────────────────
The coarse extraction-match tally logged while a run is in flight. It is NOT
the grade. The authoritative result is symbolic —
trigsimp(diff(F, x) - integrand) == 0 — in ibp_grade_depth_ladder.py. String
matching is not used for grading.

THE BOXED RULE
──────────────
The LAST \\boxed{} in the response is taken, because models often box an
intermediate step before the final answer. Extraction delegates to
ibp_symbolic_check.extract_boxed — the same function the grader uses — so the
inference lane's monitoring tally cannot disagree with the graded result on
the multi-box responses where it matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ibp_symbolic_check import extract_boxed


@dataclass
class SubcatConfig:
    """There is no judge in this pipeline; this config holds only the
    answer-extraction function for one subcategory."""
    subcat: str
    extract_answer_fn: Callable[[str], Any]


def extract_ibp_from_response(response: str) -> str | None:
    """Content of the LAST \\boxed{...} in the response, or None if absent."""
    return extract_boxed(response or "")


_IBP_CONFIG = SubcatConfig(
    subcat="ibp",
    extract_answer_fn=extract_ibp_from_response,
)


def SUBCAT_CONFIGS(subcat: str) -> SubcatConfig:
    """Return config for the given subcat. Only "ibp" is supported here."""
    if subcat != "ibp":
        raise KeyError(
            f"subcat_config_ibp.py only knows about 'ibp', got {subcat!r}"
        )
    import copy
    return copy.copy(_IBP_CONFIG)
