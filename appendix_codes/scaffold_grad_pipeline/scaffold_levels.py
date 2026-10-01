# Module:    scaffold_levels.py
# Version:   1.0
"""
scaffold_levels.py — the six fixed prompt levels used in the experiment.

Prompt wording is the experiment variable, so each level's text below is
fixed and must not be reflowed or edited.

Assembly rule for all six levels:

    prompt = <level's instruction text> + blank line + <matrix>

L0's text lives in L0_HEAD/L0_TAIL here rather than the problem bank, since
the two source CSVs store the problem text inconsistently. Each level pulls
only the bare matrix out of the bank and rebuilds the prompt around it, so
all 90 problems get identical treatment at every level. scaffold_pools.py
checks this by asserting the built L0 is byte-identical to the published
problem text.
"""

from __future__ import annotations

import os
import sys

# textguard.py lives in ../common/, shared with other pipelines in this repo.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from textguard import assert_clean

# ─────────────────────────────────────────────────────────────────────────
#  The six levels
# ─────────────────────────────────────────────────────────────────────────

# L0 is kept as head/tail halves because the matrix sits between them, and
# because scaffold_pools.py checks the built result against the published
# problem text.
L0_HEAD = "Find the eigenvalues of the 5×5 matrix A."
L0_TAIL = "Show all intermediate steps. Put your final numerical answer inside \\boxed{}."

L1 = """Find all eigenvalues of the following matrix. You MUST give every eigenvalue as a
decimal number rounded to four decimal places — do NOT leave answers in symbolic
or radical form. Give your final answer as a comma-separated list of decimals
inside \\boxed{}."""

L2_ND = """Find all eigenvalues of the following matrix using this procedure:
(1) form the characteristic polynomial det(A − λI) = 0,
(2) expand the determinant,
(3) find all roots numerically.
Give your final answer inside \\boxed{}."""

L2 = """Find all eigenvalues of the following matrix using this procedure:
(1) form the characteristic polynomial det(A − λI) = 0,
(2) expand the determinant,
(3) find all roots numerically.
You MUST give every eigenvalue as a decimal number rounded to four decimal places —
do NOT leave answers in symbolic or radical form. Give your final answer as a
comma-separated list of decimals inside \\boxed{}."""

L3 = """Find all eigenvalues of the following matrix. Follow these steps exactly:
Step 1: Write out (A − λI) explicitly.
Step 2: Cofactor-expand the determinant along the first row to obtain p(λ).
Step 3: State p(λ) explicitly.
Step 4: Find all roots of p(λ) numerically, rounded to 4 decimal places.
Step 5: Give your final answer as a comma-separated list of decimals inside \\boxed{}.
Do NOT leave any answer in symbolic or radical form."""

L4 = """Find all eigenvalues of the following matrix using the Faddeev–LeVerrier algorithm:
  M₁ = A,  c₁ = −tr(A)
  For k = 2, …, n:  Mₖ = A(Mₖ₋₁ + cₖ₋₁I),  cₖ = −(1/k) tr(Mₖ)
The characteristic polynomial is p(λ) = λⁿ + c₁λⁿ⁻¹ + … + cₙ.
Steps:
1. Compute all coefficients c₁ … c₅.
2. State p(λ) explicitly.
3. Find all roots numerically, rounded to 4 decimal places.
4. Give your final answer as a comma-separated list of decimals inside \\boxed{}.
Do NOT leave any answer in symbolic or radical form."""

LEVEL_TEXT: dict[str, str | None] = {
    "L0":    None,      # built from L0_HEAD / L0_TAIL around the matrix
    "L1":    L1,
    "L2-ND": L2_ND,
    "L2":    L2,
    "L3":    L3,
    "L4":    L4,
}

LEVELS = list(LEVEL_TEXT)


# ─────────────────────────────────────────────────────────────────────────
#  Level x model matrix
# ─────────────────────────────────────────────────────────────────────────
# L2-ND (strategy hint without the decimal demand) is Claude-only — the
# control that isolates the strategy effect from the format effect.

MODELS = ["Claude-4.5-Sonnet", "Gemini-3.1-Pro", "Qwen3-235B"]

LEVEL_MODELS: dict[str, list[str]] = {
    "L0":    MODELS,
    "L1":    MODELS,
    "L2-ND": ["Claude-4.5-Sonnet"],
    "L2":    MODELS,
    "L3":    MODELS,
    "L4":    MODELS,
}


def is_valid_cell(level: str, model: str) -> bool:
    if level not in LEVEL_MODELS:
        raise KeyError(f"Unknown level {level!r}. Valid: {LEVELS}")
    return model in LEVEL_MODELS[level]


# ─────────────────────────────────────────────────────────────────────────
#  Prompt assembly
# ─────────────────────────────────────────────────────────────────────────
_HEAD = L0_HEAD + "\n\n"
_TAIL = "\n\n" + L0_TAIL


def problem_statement(stored_problem_latex: str) -> str:
    """The bare matrix block: `A = \\begin{bmatrix} ... \\end{bmatrix}`.

    Only the matrix is taken from the bank — L1-L4 carry their own complete
    instructions, so the stored task line and \\boxed{} directive are
    stripped off. The head is required; the tail is optional, since the two
    source CSVs disagree on whether they store it.
    """
    t = stored_problem_latex.strip()
    if not t.startswith(_HEAD):
        raise ValueError(
            "problem_latex does not start with the expected task line, so the "
            "matrix block cannot be located without guessing.\n"
            f"  expected start: {_HEAD!r}\n"
            f"  got:            {t[:80]!r}"
        )
    body = t[len(_HEAD):]
    if body.endswith(_TAIL):
        body = body[: -len(_TAIL)]
    body = body.strip()
    if not body:
        raise ValueError("problem_latex has the wrapper but no matrix between it.")
    if "\\begin{bmatrix}" not in body or "\\end{bmatrix}" not in body:
        raise ValueError(
            "the extracted block is not a bmatrix — refusing to send it.\n"
            f"  got: {body[:120]!r}"
        )
    return body


def render(level: str, matrix_latex: str) -> str:
    """The exact prompt text sent to the model for this (level, problem).

    Same rule for all six levels: the level's instruction text, then the
    matrix. Nothing else is carried over from the bank.
    """
    if level not in LEVEL_TEXT:
        raise KeyError(f"Unknown level {level!r}. Valid: {LEVELS}")
    matrix = matrix_latex.strip()
    if level == "L0":
        out = f"{L0_HEAD}\n\n{matrix}\n\n{L0_TAIL}"
    else:
        out = f"{LEVEL_TEXT[level]}\n\n{matrix}"
    # Every prompt passes through here, so a corrupted \boxed{} is caught
    # before it reaches a provider.
    return assert_clean(out, f"the {level} prompt")


def full_prompt(system_prompt: str | None, user_prompt: str) -> str:
    """The two turns as one string — the full prompt the model received.

    Records store the turns separately (system_prompt, user_prompt) and this
    concatenation as full_prompt, so consumers read one field instead of each
    re-deriving the join. Defined beside prompt_message so both views of the
    prompt come from one place.

    Consistent with prompt_message, an absent system turn yields the user turn
    alone rather than a leading blank line.
    """
    system = (system_prompt or "").strip()
    return f"{system}\n\n{user_prompt}" if system else user_prompt


def prompt_message(system_prompt: str | None, user_prompt: str) -> list[dict]:
    """The exact message list sent to the model, for the result record.

    The system turn is omitted when there is none, rather than recorded as
    an empty string, so a record never claims a system turn was sent when
    it wasn't.
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages
