#!/usr/bin/env python3
"""
subcat_config.py — Per-subcategory configuration and answer extraction.

Central registry for subcategory-specific logic:
  • Ground truth computation (parse correct answer from answer_latex)
  • Answer extraction (pull model's answer from response text)
  • Error taxonomy (valid error tags + subtypes per subcat)

Each SubcatJudgeConfig bundles all these for one subcategory, making Stage 0-3
agnostic to subcat differences. Add new subcat by:
  1. Write compute_<subcat>_from_answer() function
  2. Write extract_<subcat>_from_response() function
  3. Register in get_config() mapping
  4. Add judge prompts to judge_prompts/ folder

SUPPORTED SUBCATEGORIES
───────────────────────
  Integer answer types:
    • rank       : Compute rank from matrix
    • nullity    : Compute nullity (dimension of null space)
    • trace      : Compute trace (sum of diagonal elements)

  Scalar/Float answer types:
    • det        : Compute determinant
    • eig        : Compute eigenvalues

  Matrix/Vector answer types:
    • matrix_vector  : Matrix-vector product
    • multiplication : Matrix-matrix product
    • matrix_power   : Matrix exponentiation (A^k)
    • transpose      : Matrix transpose

PUBLIC INTERFACE
────────────────
  from subcat_config import SUBCAT_CONFIGS

  config = SUBCAT_CONFIGS("rank")  # → SubcatJudgeConfig instance
  ground_truth = config.ground_truth_fn(answer_latex)  # parse answer
  extracted = config.extract_answer_fn(response_text)  # extract from response

ADDING A NEW SUBCAT
───────────────────
  1. Define ground truth parser:
     def compute_newsubcat_from_answer(answer_latex: str) -> <type>:
         # Parse answer_latex, return correct answer value

  2. Define answer extractor:
     def extract_newsubcat_from_response(response: str) -> <type> | None:
         # Search response for model's stated answer, return or None if not found

  3. Define error taxonomy:
     valid_tags = ["ERROR_TYPE_1", "ERROR_TYPE_2", ...]
     valid_subtypes = {"ERROR_TYPE_1": ["subtype_a", "subtype_b"], ...}

  4. Register in get_config() at bottom of this file:
     "newsubcat": SubcatJudgeConfig(
         subcat="newsubcat",
         ground_truth_fn=compute_newsubcat_from_answer,
         extract_answer_fn=extract_newsubcat_from_response,
         valid_error_tags=valid_tags,
         valid_subtypes=valid_subtypes,
     )

  5. Add judge prompts:
     judge_prompts/newsubcat.py with BUILD_SYSTEM_TEMPLATE, BUILD_USER_TEMPLATE,
     VALIDATE_SYSTEM_TEMPLATE, and build_validate_user_prompt()

TESTING
───────
  python -c "from subcat_config import SUBCAT_CONFIGS; \\
             cfg = SUBCAT_CONFIGS('rank'); \\
             print(cfg.ground_truth_fn('= 5'))"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Callable, Any


# Subcategory name mapping: CLI arg → canonical Subcat value in data files.
# Uses complete forms — avoids abbreviations like "null" which cause issues.
SUBCAT_ARG = {
    "det": "determinant",
    "eig": "eigenvalue",
    "rank": "rank",
    "nullity": "nullity",
    "mult": "multiplication",
    "pow": "matrix_power",
    "vec": "matrix_vector",
    "trans": "transpose",
    "trace": "trace",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIG DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubcatJudgeConfig:
    """Encapsulates everything that differs between subcategories."""

    subcat: str
    ground_truth_fn: Callable[[str], Any]        # answer_latex → ground truth
    extract_answer_fn: Callable[[str], Any]       # model response → extracted answer (or None)
    valid_error_tags: list[str]
    valid_subtypes: dict[str, list[str]]
    # Populated at runtime from JUDGE_PROMPTS in prompts_unified.py
    judge_system_prompt: str = field(default="")
    judge_user_template: str = field(default="")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SHARED LATEX HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_integer_from_latex(text: str) -> int:
    """
    Extract a single integer from a LaTeX string.
    Handles: "3", "= 3", "\\text{rank}(A) = 3", "$3$", "\\(3\\)"
    Raises ValueError if no integer found.
    """
    text = str(text).strip()
    text = re.sub(r'^\$+|\$+$', '', text).strip()
    text = re.sub(r'^\\\(|\\\)$', '', text).strip()

    if re.fullmatch(r'\d+', text):
        return int(text)

    m = re.search(r'=\s*(\d+)', text)
    if m:
        return int(m.group(1))

    digits = re.findall(r'\d+', text)
    if digits:
        return int(digits[-1])

    raise ValueError(f"Cannot parse integer from: {text!r}")


def _parse_bmatrix(latex: str) -> list:
    """
    Parse a LaTeX \\begin{bmatrix}...\\end{bmatrix} into a Python list.

    - Vectors  → list of ints/floats   e.g. [15, -10, -8]
    - Matrices → list of lists         e.g. [[-1, 4], [-4, 3]]

    Strips a leading label like "Ax =", "AB =", "A^2 =", "A^T =" before parsing.
    Raises ValueError if no bmatrix found or parsing fails.
    """
    # Strip optional label prefix (e.g. "Ax =", "AB =")
    text = re.sub(r'^[A-Za-z^{}0-9\s]+=\s*', '', latex.strip())

    m = re.search(r'\\begin\{[pb]?matrix\}(.*?)\\end\{[pb]?matrix\}',
                  text, re.DOTALL)
    if not m:
        raise ValueError(f"No bmatrix/pmatrix found in: {latex[:100]!r}")

    inner = m.group(1).strip()
    rows_raw = re.split(r'\\\\', inner)
    rows = []
    for row_raw in rows_raw:
        row_raw = row_raw.strip()
        if not row_raw:
            continue
        cells = [c.strip() for c in row_raw.split('&')]
        parsed_cells = []
        for c in cells:
            c = c.strip()
            try:
                parsed_cells.append(int(c))
            except ValueError:
                try:
                    parsed_cells.append(float(c))
                except ValueError:
                    parsed_cells.append(c)   # keep as string if non-numeric
        rows.append(parsed_cells)

    if not rows:
        raise ValueError(f"Empty matrix parsed from: {latex[:100]!r}")

    # If every row has exactly one element → vector (flat list)
    if all(len(r) == 1 for r in rows):
        return [r[0] for r in rows]

    return rows


def _extract_integer_from_response(response: str) -> Optional[int]:
    """
    General integer extractor for responses to integer-answer problems.
    Priority: \\boxed{N} → "= N" near keyword → last integer on last line.
    """
    if not response:
        return None
    text = str(response).strip()

    m = re.search(r'\\boxed\{(\d+)\}', text)
    if m:
        return int(m.group(1))

    m = re.search(r'(?:rank|nullity|dimension|dim)\s*(?:\([^)]*\))?\s*(?:is|=|:)\s*(\d+)',
                  text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # "is N" — last occurrence
    matches = re.findall(r'\bis\s+(\d+)\b', text, re.IGNORECASE)
    if matches:
        return int(matches[-1])

    # Last non-empty line ending in "= N"
    for line in reversed(text.splitlines()):
        if line.strip():
            m2 = re.search(r'=\s*(\d+)\s*$', line.strip())
            if m2:
                return int(m2.group(1))
            break

    return None


def _extract_bmatrix_from_response(response: str) -> Optional[list]:
    """
    Extract the LAST bmatrix/pmatrix from a model response.
    Returns a parsed list (vector or matrix), or None if not found.
    """
    if not response:
        return None
    # Find all bmatrix blocks
    blocks = re.findall(
        r'\\begin\{[pb]?matrix\}.*?\\end\{[pb]?matrix\}',
        response, re.DOTALL
    )
    if not blocks:
        return None
    # Use the last one (most likely the final answer)
    try:
        return _parse_bmatrix(blocks[-1])
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DETERMINANT
# ─────────────────────────────────────────────────────────────────────────────

def compute_determinant_from_answer(answer_latex: str) -> int:
    """Parse single integer determinant from 'answer_latex' (e.g. '= -712')."""
    text = str(answer_latex).strip()
    m = re.search(r'=\s*(-?\d+)', text)
    if m:
        return int(m.group(1))
    # fallback: just grab the integer
    m2 = re.search(r'-?\d+', text)
    if m2:
        return int(m2.group(0))
    raise ValueError(f"Cannot parse determinant from: {text!r}")


def extract_determinant_from_response(response: str) -> Optional[float]:
    """
    Extract numeric determinant from model response.
    Handles: \\boxed{N}, \\boxed{\\det(A)=N}, \\boxed{\\frac{a}{b}}, nested braces.
    """
    if not response:
        return None
    # One level of nested braces
    matches = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', response)
    if not matches:
        return None
    val = matches[-1].strip().replace(',', '').replace(' ', '')

    # \boxed{\det(A) = 712}
    m = re.search(r'\\?det[^=]*=\s*(-?\d+)', val)
    if m:
        return int(m.group(1))

    # \frac{a}{b}
    frac = re.match(r'^(-?)\\frac\{(\d+)\}\{(\d+)\}$', val)
    if frac:
        sign = -1 if frac.group(1) == '-' else 1
        result = sign * int(frac.group(2)) / int(frac.group(3))
        return int(result) if result == int(result) else round(result, 6)

    try:
        return int(val)
    except ValueError:
        try:
            return int(float(val))
        except ValueError:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# 3b.  EIGENVALUE
# ─────────────────────────────────────────────────────────────────────────────

def compute_eigenvalue_from_answer(answer_latex: str) -> list:
    """
    Parse eigenvalues from answer_latex.
    Handles comma/semicolon-separated lists and set notation: {1, -2, 3}.
    Returns a sorted list of floats.
    """
    text = str(answer_latex).strip()
    # Strip set braces  { ... }
    text = re.sub(r'^\{|\}$', '', text).strip()
    # Split on comma or semicolon
    parts = re.split(r'[,;]', text)
    values = []
    for p in parts:
        p = p.strip()
        # Remove LaTeX like \lambda_1 =
        p = re.sub(r'\\?lambda[_\d\s]*=\s*', '', p).strip()
        try:
            values.append(float(p))
        except ValueError:
            pass
    if not values:
        raise ValueError(f"Cannot parse eigenvalues from: {answer_latex!r}")
    return sorted(set(values))


def extract_eigenvalue_from_response(response: str) -> Optional[list]:
    """
    Extract eigenvalues from model response.
    Handles LaTeX formatting, symbolic math (sqrt), subscripts, and approximations.

    Supported formats:
    • \\boxed{0, 2+\\sqrt{22}, 2-\\sqrt{22}}
    • \\boxed{\\lambda_1 \\approx -9.8, \\lambda_2 \\approx -6.2, \\lambda_3 \\approx 0.46}
    • "eigenvalues are 1, 2, 3"
    """
    if not response:
        return None

    # Collect all boxed values
    matches = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', response)
    if matches:
        # Last boxed value (could be comma-separated)
        raw = matches[-1]
        values = _parse_eigenvalue_list(raw)
        if values:
            return sorted(set(values))

    # Fallback: "eigenvalues are 1, 2, 3" or "λ = 1, 2, 3"
    m = re.search(
        r'eigenvalue[s]?\s+(?:are|is|=)\s*([-\d\s,\.]+)',
        response, re.IGNORECASE
    )
    if m:
        parts = re.split(r'[,\s]+', m.group(1).strip())
        values = []
        for p in parts:
            try:
                values.append(float(p))
            except ValueError:
                pass
        if values:
            return sorted(set(values))

    return None


def _parse_eigenvalue_list(raw_text: str) -> Optional[list]:
    """
    Parse a comma-separated list of eigenvalues from raw LaTeX text.
    Handles:
    • LaTeX commands: \\lambda, \\approx, \\quad, subscripts (_i)
    • Symbolic math: 2+\\sqrt{22}, 2-\\sqrt{22}
    • Approximations: -9.8, -6.2, 0.46
    • Complex expressions with operators: +, -, *, /
    """
    if not raw_text:
        return None

    try:
        from sympy import sympify, N
    except ImportError:
        # Fallback if sympy not available
        return _parse_eigenvalue_list_fallback(raw_text)

    # Split on commas/semicolons, respecting nested braces
    parts = re.split(r'[,;](?![^{}]*})', raw_text)
    values = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Clean up LaTeX
        cleaned = _clean_latex_for_eigenvalue(part)

        # Try simple float parse first
        try:
            values.append(float(cleaned))
            continue
        except ValueError:
            pass

        # Try sympy evaluation for symbolic math
        try:
            result = float(N(sympify(cleaned)))
            values.append(result)
            continue
        except Exception:
            pass

    return values if values else None


def _clean_latex_for_eigenvalue(text: str) -> str:
    """Clean LaTeX formatting from eigenvalue expression."""
    # Remove \lambda, \lambda_i, etc.
    text = re.sub(r'\\lambda[_\d]*\s*=?\s*', '', text)
    # Remove \approx and surrounding whitespace
    text = re.sub(r'\s*\\approx\s*', '', text)
    # Remove \quad and other spacing commands
    text = re.sub(r'\\quad', '', text)
    # Remove extra spaces
    text = text.replace(' ', '')
    # Handle \sqrt{N} → sqrt(N)
    text = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', text)
    return text


def _parse_eigenvalue_list_fallback(raw_text: str) -> Optional[list]:
    """
    Fallback eigenvalue parsing when sympy is not available.
    Handles simple numeric expressions only.
    """
    parts = re.split(r'[,;]', raw_text)
    values = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Clean LaTeX
        cleaned = _clean_latex_for_eigenvalue(part)

        # Try float parse
        try:
            values.append(float(cleaned))
        except ValueError:
            # Give up on this part
            pass

    return values if values else None


# ─────────────────────────────────────────────────────────────────────────────
# 4.  RANK
# ─────────────────────────────────────────────────────────────────────────────

def compute_rank_from_answer(answer_latex: str) -> int:
    return _parse_integer_from_latex(answer_latex)


def extract_rank_from_response(response: str) -> Optional[int]:
    if not response:
        return None
    text = str(response).strip()

    m = re.search(r'\\boxed\{(\d+)\}', text)
    if m:
        return int(m.group(1))

    m = re.search(r'rank\s*(?:of\s+(?:matrix\s+)?[A-Za-z])?\s*(?:is|=|:)\s*(\d+)',
                  text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    return _extract_integer_from_response(response)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  NULLITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_nullity_from_answer(answer_latex: str) -> int:
    return _parse_integer_from_latex(answer_latex)


def extract_nullity_from_response(response: str) -> Optional[int]:
    if not response:
        return None
    text = str(response).strip()

    m = re.search(r'\\boxed\{(\d+)\}', text)
    if m:
        return int(m.group(1))

    m = re.search(r'nullity\s*(?:\([^)]*\))?\s*(?:is|=|:)\s*(\d+)',
                  text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    return _extract_integer_from_response(response)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MATRIX-VECTOR PRODUCT  (Ax = vector)
# ─────────────────────────────────────────────────────────────────────────────

def compute_matvec_from_answer(answer_latex: str) -> list:
    return _parse_bmatrix(answer_latex)


def extract_matvec_from_response(response: str) -> Optional[list]:
    return _extract_bmatrix_from_response(response)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MATRIX MULTIPLICATION  (AB = matrix)
# ─────────────────────────────────────────────────────────────────────────────

def compute_multiplication_from_answer(answer_latex: str) -> list:
    return _parse_bmatrix(answer_latex)


def extract_multiplication_from_response(response: str) -> Optional[list]:
    return _extract_bmatrix_from_response(response)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MATRIX POWER  (A^k = matrix)
# ─────────────────────────────────────────────────────────────────────────────

def compute_matrix_power_from_answer(answer_latex: str) -> list:
    return _parse_bmatrix(answer_latex)


def extract_matrix_power_from_response(response: str) -> Optional[list]:
    return _extract_bmatrix_from_response(response)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  TRANSPOSE  (A^T = matrix)
# ─────────────────────────────────────────────────────────────────────────────

def compute_transpose_from_answer(answer_latex: str) -> list:
    return _parse_bmatrix(answer_latex)


def extract_transpose_from_response(response: str) -> Optional[list]:
    return _extract_bmatrix_from_response(response)



# ─────────────────────────────────────────────────────────────────────────────
# 10.  TAXONOMY — VALID TAGS & SUBTYPES
# ─────────────────────────────────────────────────────────────────────────────

BASE_PRIMARY_TAGS = [
    "GENERATION_TRUNCATION",
    "FORMATTING_MISMATCH",
    "SIGN_ERROR",
    "ARITHMETIC",
    "HALLUCINATION",
    "INPUT_TRANSCRIPTION",
    "CARRY_DOWN_ERROR",
    "METHOD_FAIL",
    "MEMORY_LOSS",
    "OTHER_UNMAPPED",
]

EIGEN_SPECIFIC_TAGS = [
    "ALGEBRAIC_PRECEDENCE",
    "FALSE_VERIFICATION",
    "VARIABLE_ENTANGLEMENT",
    "GENERATION_LOOP",
]

ALL_PRIMARY_TAGS = BASE_PRIMARY_TAGS + EIGEN_SPECIFIC_TAGS

RANK_SIGN_SUBTYPES = [
    "Product_Sign_Error",    # correct multiplier magnitude, wrong sign applied
    "Operation_Direction",   # added scaled row where subtraction required, or vice versa
    "Rule_Interference",     # negative entry triggers wrong scaling/subtraction behaviour
    "Double_Negative_Trap",  # negative entry doubly negated during operation
    "Silent_Sign_Flip",      # wrong-sign output with zero working shown to trace it
]
NULLITY_SIGN_SUBTYPES = [
    "Product_Sign_Error",    # correct magnitude, wrong sign
    "Operation_Direction",   # wrong operation direction in row reduction
    "Rule_Interference",     # negative entry triggers sign error
    "Parametric_Sign_Flip",  # sign flips in parameterized solution
    "Silent_Sign_Flip",      # wrong-sign with no working shown
]
RANK_HALLUC_SUBTYPES = [
    "Complete_Collapse",
    "Teleological_Zeroing",  # row forced to [0…0] despite unsupporting arithmetic
    "Premature_Assertion",
    "Silent_Omission",
    "Ungrounded_Guess",
    "Spontaneous_Insertion", # fabricated entry with no origin in input
]

# Matrix subcats use same 5 SIGN_ERROR subtypes as rank (PDF v1.1 §2)
MATRIX_SIGN_SUBTYPES = RANK_SIGN_SUBTYPES

# HALLUCINATION subtypes ──────────────────────────────────────────────────────
# matrix_vector adds Dimension_Assertion (PDF v1.1 §3)
# Disambiguate from Ungrounded_Guess: Dimension_Assertion requires computation
# to be present; Ungrounded_Guess is pure assertion with zero working.
MATVEC_HALLUC_SUBTYPES = RANK_HALLUC_SUBTYPES + [
    "Dimension_Assertion",    # some computation present but output shape wrong (matrix returned instead of vector)
]

# METHOD_FAIL subtypes (PDF v1.1 §3) ─────────────────────────────────────────
METHOD_FAIL_MULT      = ["Operand_Confusion"]
METHOD_FAIL_MAT_POW   = ["Operand_Confusion", "Composition_Rule_Violation"]
METHOD_FAIL_MAT_VEC   = ["Operand_Confusion"]
METHOD_FAIL_TRANSPOSE = ["Composition_Rule_Violation"]

BASE_SUBTYPES: dict[str, list[str]] = {
    "ARITHMETIC":          [],
    "INPUT_TRANSCRIPTION": [],
    "CARRY_DOWN_ERROR":    [],
    "METHOD_FAIL":         [],
    "MEMORY_LOSS":         [],
    "OTHER_UNMAPPED":      [],
}

RANK_VALID_SUBTYPES = {
    "SIGN_ERROR":          RANK_SIGN_SUBTYPES,
    "HALLUCINATION":       RANK_HALLUC_SUBTYPES,
    "ARITHMETIC":          [],
    "INPUT_TRANSCRIPTION": [],
    "CARRY_DOWN_ERROR":    [],
    "METHOD_FAIL":         [],
    "MEMORY_LOSS":         [],
    "OTHER_UNMAPPED":      [],
}
NULLITY_VALID_SUBTYPES = {
    "SIGN_ERROR":          NULLITY_SIGN_SUBTYPES,
    "HALLUCINATION":       RANK_HALLUC_SUBTYPES,
    "ARITHMETIC":          [],
    "INPUT_TRANSCRIPTION": [],
    "CARRY_DOWN_ERROR":    [],
    "METHOD_FAIL":         [],
    "MEMORY_LOSS":         [],
    "OTHER_UNMAPPED":      [],
}
MULTIPLICATION_VALID_SUBTYPES = {
    "SIGN_ERROR":          MATRIX_SIGN_SUBTYPES,
    "HALLUCINATION":       RANK_HALLUC_SUBTYPES,
    "METHOD_FAIL":         METHOD_FAIL_MULT,
    "ARITHMETIC":          [],
    "INPUT_TRANSCRIPTION": [],
    "CARRY_DOWN_ERROR":    [],
    "MEMORY_LOSS":         [],
    "OTHER_UNMAPPED":      [],
}
MATRIX_POWER_VALID_SUBTYPES = {
    "SIGN_ERROR":          MATRIX_SIGN_SUBTYPES,
    "HALLUCINATION":       RANK_HALLUC_SUBTYPES,
    "METHOD_FAIL":         METHOD_FAIL_MAT_POW,
    "ARITHMETIC":          [],
    "INPUT_TRANSCRIPTION": [],
    "CARRY_DOWN_ERROR":    [],
    "MEMORY_LOSS":         [],
    "OTHER_UNMAPPED":      [],
}
MATRIX_VECTOR_VALID_SUBTYPES = {
    "SIGN_ERROR":          MATRIX_SIGN_SUBTYPES,
    "HALLUCINATION":       MATVEC_HALLUC_SUBTYPES,
    "METHOD_FAIL":         METHOD_FAIL_MAT_VEC,
    "ARITHMETIC":          [],
    "INPUT_TRANSCRIPTION": [],
    "CARRY_DOWN_ERROR":    [],
    "MEMORY_LOSS":         [],
    "OTHER_UNMAPPED":      [],
}
TRANSPOSE_VALID_SUBTYPES = {
    "SIGN_ERROR":          MATRIX_SIGN_SUBTYPES,
    "HALLUCINATION":       RANK_HALLUC_SUBTYPES,
    "METHOD_FAIL":         METHOD_FAIL_TRANSPOSE,
    "ARITHMETIC":          [],
    "INPUT_TRANSCRIPTION": [],
    "CARRY_DOWN_ERROR":    [],
    "MEMORY_LOSS":         [],
    "OTHER_UNMAPPED":      [],
}

# DET/EIG — 8 SIGN_ERROR subtypes (including Alternating_Drift and Cofactor_Neglect)
DET_SIGN_SUBTYPES = [
    "Product_Sign_Error",
    "Operation_Direction",
    "Rule_Interference",
    "Parity_Sign_Error",
    "Double_Negative_Trap",
    "Alternating_Drift",
    "Cofactor_Neglect",
    "Silent_Sign_Flip",
]
DET_VALID_SUBTYPES = {
    "SIGN_ERROR":          DET_SIGN_SUBTYPES,
    "HALLUCINATION":       RANK_HALLUC_SUBTYPES,
    "ARITHMETIC":          [],
    "INPUT_TRANSCRIPTION": [],
    "CARRY_DOWN_ERROR":    [],
    "METHOD_FAIL":         [],
    "MEMORY_LOSS":         [],
    "OTHER_UNMAPPED":      [],
}

# EIG — 8 SIGN_ERROR subtypes + 4 bespoke tags (14 total)
EIG_SIGN_SUBTYPES = DET_SIGN_SUBTYPES
EIG_VALID_SUBTYPES = {
    "SIGN_ERROR":           EIG_SIGN_SUBTYPES,
    "HALLUCINATION":        RANK_HALLUC_SUBTYPES,
    "ALGEBRAIC_PRECEDENCE": ["Bracket_Erosion", "PEMDAS_Violation", "Exponent_Detachment"],
    "FALSE_VERIFICATION":   ["Circular_Substitution", "Tautological_Check"],
    "VARIABLE_ENTANGLEMENT":["Variable_Substitution_Error", "Variable_Reassignment"],
    "GENERATION_LOOP":      ["Repetitive_Generation"],
    "ARITHMETIC":           [],
    "INPUT_TRANSCRIPTION":  [],
    "CARRY_DOWN_ERROR":     [],
    "METHOD_FAIL":          [],
    "MEMORY_LOSS":          [],
    "OTHER_UNMAPPED":       [],
}

# TRACE — 5 SIGN_ERROR subtypes (same as rank; simple scalar sum of diagonal)
TRACE_SIGN_SUBTYPES = RANK_SIGN_SUBTYPES
TRACE_VALID_SUBTYPES = {
    "SIGN_ERROR":          TRACE_SIGN_SUBTYPES,
    "HALLUCINATION":       RANK_HALLUC_SUBTYPES,
    "ARITHMETIC":          [],
    "INPUT_TRANSCRIPTION": [],
    "CARRY_DOWN_ERROR":    [],
    "METHOD_FAIL":         [],
    "MEMORY_LOSS":         [],
    "OTHER_UNMAPPED":      [],
}


# ─────────────────────────────────────────────────────────────────────────────
# 11.  REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, SubcatJudgeConfig] = {}


# ──────────────────────────────────────────────────────────────────────────────
# TRACE (Scalar: sum of diagonal elements)
# ──────────────────────────────────────────────────────────────────────────────

def compute_trace_from_answer(answer_latex: str) -> int:
    """Extract trace (scalar) from answer_latex.
    Handles: '= -10', '-10', '\\text{tr}(A) = 5', plain integers.
    """
    text = str(answer_latex).strip()
    m = re.search(r'=\s*(-?\d+(?:\.\d+)?)', text)
    if m:
        return int(float(m.group(1)))
    m = re.fullmatch(r'-?\d+(?:\.\d+)?', text)
    if m:
        return int(float(m.group(0)))
    raise ValueError(f"Cannot parse trace from: {text!r}")


def extract_trace_from_response(response: str) -> Optional[int]:
    """Extract trace value from model response.
    Priority:
      1. Last \\boxed{N}
      2. Last tr(A)=N or \\text{tr}(A)=N  (handles LaTeX and plain, catches Gemini style)
      3. "trace ... is/= N"
      4. "is **N**" bold markdown final answer
      5. Last non-empty line ending in "= N"
    """
    if not response:
        return None
    text = str(response).strip()

    # 1. Last \boxed{N} — handles plain number or equation inside boxed
    matches = re.findall(r'\\boxed\{(-?[^}]+)\}', text)
    if matches:
        for val in reversed(matches):
            val = val.strip()
            try:
                return int(float(val))
            except (ValueError, TypeError):
                pass
            # Equation inside boxed: \text{Trace}(A) = -7  or  tr(A) = 14
            m_eq = re.search(r'=\s*(-?\d+)\s*$', val)
            if m_eq:
                return int(m_eq.group(1))

    # 2. Last tr/trace/Trace(A)=N — handles \text{tr}, \text{trace}, \text{Trace}, plain
    tr_matches = re.findall(
        r'(?:\\text\{)?tr(?:ace)?\}?\s*(?:\([^)]*\))?\s*=\s*(-?\d+)',
        text, re.IGNORECASE
    )
    if tr_matches:
        return int(tr_matches[-1])

    # 3. "trace ... is/= N" (allows words between trace and value)
    m = re.search(r'\btrace\b[^=\n]{0,40}(?:is|=|:)\s*(-?\d+)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # 4. "is **N**" or "is N" — markdown bold final answer
    bold_matches = re.findall(r'\bis\s+\*{0,2}(-?\d+)\*{0,2}', text, re.IGNORECASE)
    if bold_matches:
        return int(bold_matches[-1])

    # 5. Last non-empty line ending in "= N"
    for line in reversed(text.splitlines()):
        if line.strip():
            m2 = re.search(r'=\s*(-?\d+)\s*$', line.strip())
            if m2:
                return int(m2.group(1))
            break

    return None


_SUBCAT_SPECS = {
    # Canonical short forms only: det, eig, rank, null, mult, pow, vec, trans, trace
    "det": (
        compute_determinant_from_answer,
        extract_determinant_from_response,
        DET_VALID_SUBTYPES,
    ),
    "eig": (
        compute_eigenvalue_from_answer,
        extract_eigenvalue_from_response,
        EIG_VALID_SUBTYPES,
    ),
    "rank": (
        compute_rank_from_answer,
        extract_rank_from_response,
        RANK_VALID_SUBTYPES,
    ),
    "null": (
        compute_nullity_from_answer,
        extract_nullity_from_response,
        NULLITY_VALID_SUBTYPES,
    ),
    "vec": (
        compute_matvec_from_answer,
        extract_matvec_from_response,
        MATRIX_VECTOR_VALID_SUBTYPES,
    ),
    "mult": (
        compute_multiplication_from_answer,
        extract_multiplication_from_response,
        MULTIPLICATION_VALID_SUBTYPES,
    ),
    "pow": (
        compute_matrix_power_from_answer,
        extract_matrix_power_from_response,
        MATRIX_POWER_VALID_SUBTYPES,
    ),
    "trans": (
        compute_transpose_from_answer,
        extract_transpose_from_response,
        TRANSPOSE_VALID_SUBTYPES,
    ),
    "trace": (
        compute_trace_from_answer,
        extract_trace_from_response,
        TRACE_VALID_SUBTYPES,
    ),
}


def _normalize_subcat_input(input_form: str) -> str:
    """Normalize any subcat input to canonical short form for lookup."""
    canonical_map = {
        # Determinant
        "det": "det",
        "determinant": "det",
        # Eigenvalue
        "eig": "eig",
        "eigen": "eig",
        "eigenvalue": "eig",
        "eigenvalues": "eig",
        # Rank
        "rank": "rank",
        # Nullity
        "null": "null",
        "nullity": "null",
        # Multiplication
        "mult": "mult",
        "multiplication": "mult",
        # Matrix power
        "pow": "pow",
        "pow2": "pow",
        "power": "pow",
        "matrix_power": "pow",
        # Matrix-vector
        "vec": "vec",
        "vector": "vec",
        "matrix_vector": "vec",
        "matvec": "vec",
        # Transpose
        "trans": "trans",
        "transpose": "trans",
        "transp": "trans",
        # Trace
        "trace": "trace",
    }
    lower_form = str(input_form).lower().strip()
    return canonical_map.get(lower_form, lower_form)


def get_config(subcat: str) -> SubcatJudgeConfig:
    """Return (and cache) the SubcatJudgeConfig for subcat."""
    # Normalize input to canonical short form
    normalized = _normalize_subcat_input(subcat)

    if normalized not in _REGISTRY:
        if normalized not in _SUBCAT_SPECS:
            raise KeyError(
                f"Unknown subcat: {subcat!r} (normalized to {normalized!r}). "
                f"Available: {list(_SUBCAT_SPECS)}"
            )
        gt_fn, ex_fn, subtypes = _SUBCAT_SPECS[normalized]
        _REGISTRY[normalized] = SubcatJudgeConfig(
            subcat=normalized,
            ground_truth_fn=gt_fn,
            extract_answer_fn=ex_fn,
            valid_error_tags=BASE_PRIMARY_TAGS + EIGEN_SPECIFIC_TAGS if normalized == "eig" else BASE_PRIMARY_TAGS,
            valid_subtypes=subtypes,
        )
    return _REGISTRY[normalized]


# Convenience alias used throughout the pipeline
SUBCAT_CONFIGS = get_config
