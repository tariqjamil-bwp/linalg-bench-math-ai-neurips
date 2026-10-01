"""
run_opcount_experiment.py
=========================
Op-Count Control Experiment — LinAlg-Bench.
Self-contained: generates all shallow-chain problems internally, grades all responses,
writes one row per API call to results/opcount_results_raw.csv, with warnings/errors
logged to results/opcount_errors.log. Aggregation into a per-cell accuracy table with
confidence intervals is a separate step — see build_opcount_aggregated.py.

Requirements: Python 3.10+, requests, numpy (stdlib otherwise).
API key: OPENROUTER_API_KEY, read from the repo root's .env (see .env.example).
"""

import os

try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(), override=True)
except ImportError:
    pass

# ===========================================================================
# CONFIG BLOCK — edit these before running
# ===========================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") + os.sep

# VERIFY each slug at https://openrouter.ai/models before running; edit here if renamed.
# Reasoning/hidden-CoT models (o1, o3, o1-mini, DeepSeek-R1) are EXCLUDED because
# hidden CoT is an uncontrolled depth aid that makes depth-vs-opcount uninterpretable.
MODELS = {
    # paper name -> OpenRouter slug
    "claude-4.5-sonnet":  "anthropic/claude-sonnet-4.5",
    "gemini-3.1-pro":     "google/gemini-3.1-pro-preview",
    "qwen3-235b":         "qwen/qwen3-235b-a22b",
    "llama-3.3-70b":      "meta-llama/llama-3.3-70b-instruct",
    "gpt-5.2":            "openai/gpt-5.2",
    "deepseek-v3":        "deepseek/deepseek-chat",            # DeepSeek-V3 (no hidden CoT)
    "gpt-4o":             "openai/gpt-4o",
    "mistral-large-2512": "mistralai/mistral-large-2512",
    "qwen-2.5-72b":       "qwen/qwen-2.5-72b-instruct",
}

TEMPERATURE    = 0
REPEATS        = 1
GEN_SEED       = 500          # seed for numpy RNG used in problem generation (per-cell seeds override this; see CELL_SEEDS)
MAX_TOKENS     = 8000
REQUEST_TIMEOUT = 120         # seconds per API call
RETRY_ATTEMPTS  = 3           # number of attempts per API call before logging error and moving on
SLEEP_BETWEEN_CALLS = 1.0     # seconds; increase if you hit rate limits
RETRY_BASE_SLEEP    = 4.0     # seconds; doubled on each retry (exponential backoff)

# API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ===========================================================================
# END CONFIG BLOCK
# ===========================================================================

import sys
import csv
import argparse
import json
import math
import time
import re
import logging
import datetime
import itertools
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("ERROR: 'requests' is not installed. Run: pip install requests")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("WARNING: numpy not found. Shallow-chain ground-truth computed via pure Python. "
          "Eigenvalue grading will be marked 'needs_offline_grading'.")

# ---------------------------------------------------------------------------
# Cell seeds (fixed by the experiment design)
# ---------------------------------------------------------------------------
CELL_SEEDS = {
    ("matvec",  6):  2027,
    ("matvec", 11):  2028,
    ("matvec", 16):  2036,  # phase2_frontier: Claude-4.5-Sonnet, GPT-5.2, Gemini-3.1-Pro
    ("matvec", 19):  2039,  # phase2_frontier: Claude-4.5-Sonnet, GPT-5.2
    ("matvec", 23):  2029,
    ("matvec", 26):  2046,  # phase2_frontier: Claude-4.5-Sonnet, GPT-5.2
}

# Actual op counts per condition — 45 ops per matvec step (5x5 . 5)
ACTUAL_OPS = {
    ("matvec",  6):  270,
    ("matvec", 11):  495,
    ("matvec", 16):  720,
    ("matvec", 19):  855,
    ("matvec", 23):  1035,
    ("matvec", 26):  1170,
}

PROBLEMS_PER_CELL = 15
ANSWER_MAGNITUDE_LIMIT = 10**12


# ===========================================================================
# Logging setup
# ===========================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("opcount")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(output_dir / "opcount_errors.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ===========================================================================
# Problem generation
# ===========================================================================

def _make_int_matrix(rng, low=-2, high=2, size=(5, 5)):
    """Return a 5x5 integer array using numpy or pure-python fallback."""
    if NUMPY_AVAILABLE:
        return rng.choice(np.array([-2, -1, 1, 2]), size=size)
    else:
        import random
        return [[random.choice([-2, -1, 1, 2]) for _ in range(size[1])] for _ in range(size[0])]


def _make_int_vector(rng, low=-3, high=3, size=5):
    if NUMPY_AVAILABLE:
        return rng.integers(low, high + 1, size=size)
    else:
        import random
        return [random.randint(low, high) for _ in range(size)]


def _count_nonzero_row(row):
    """Reserved for future enhancement — not currently called."""
    return sum(1 for x in row for x in ([x] if not hasattr(x, '__iter__') else x) if x != 0) \
        if False else sum(1 for x in (row.tolist() if NUMPY_AVAILABLE else row) if x != 0)


def _matvec_int(M, v):
    """Exact integer matrix-vector product (list-of-lists or numpy)."""
    if NUMPY_AVAILABLE:
        return (M.astype(object) @ v.astype(object)).tolist()
    n = len(M)
    result = []
    for i in range(n):
        s = 0
        for j in range(n):
            s += int(M[i][j]) * int(v[j])
        result.append(s)
    return result


def _matadd_int(A, B):
    """Exact integer matrix addition.

    Reserved for future enhancement — pairs with make_matadd_chain(), not
    currently called by run_experiment()."""
    if NUMPY_AVAILABLE:
        return (A.astype(object) + B.astype(object))
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def _mat_to_list(M):
    """Convert numpy array or list-of-lists to list-of-lists of ints."""
    if NUMPY_AVAILABLE and hasattr(M, 'tolist'):
        return [[int(x) for x in row] for row in M.tolist()]
    return [[int(x) for x in row] for row in M]


def _vec_to_list(v):
    if NUMPY_AVAILABLE and hasattr(v, 'tolist'):
        return [int(x) for x in v.tolist()]
    return [int(x) for x in v]


def make_matvec_chain(k, problem_idx, rng):
    """
    Generate k distinct 5x5 integer matrices and a starting 5-vector.
    Returns (matrices_as_list_of_lists, v0_list, answer_list).
    Matrices: entries in {-2,-1,1,2} (no zeros by construction).
    v0: entries in [-3,3], not all-zero.
    Answer: exact integer arithmetic.
    """
    matrices = []
    for _ in range(k):
        while True:
            if NUMPY_AVAILABLE:
                M = rng.choice(np.array([-2, -1, 1, 2]), size=25).reshape(5, 5)
                rows_ok = True
            else:
                import random
                M = [[random.choice([-2, -1, 1, 2]) for _ in range(5)] for _ in range(5)]
                rows_ok = True
            if rows_ok:
                matrices.append(M)
                break

    while True:
        v0 = _make_int_vector(rng, low=-3, high=3)
        v0_list = _vec_to_list(v0)
        if any(x != 0 for x in v0_list):
            break

    # Exact integer arithmetic forward pass
    v = list(v0_list)
    for M in matrices:
        v = _matvec_int(M, v if not NUMPY_AVAILABLE else (
            np.array(v, dtype=object) if NUMPY_AVAILABLE else v))
        if NUMPY_AVAILABLE:
            v = [int(x) for x in (v if isinstance(v, list) else v)]

    mats_list = [_mat_to_list(M) for M in matrices]
    return mats_list, v0_list, v  # v is final answer as list of ints


def _matvec_chain_answer(matrices_list, v0_list):
    """Recompute the exact answer from list-of-lists inputs.

    Reserved for future enhancement — not currently invoked by run_experiment()."""
    v = list(v0_list)
    for M in matrices_list:
        new_v = []
        for i in range(5):
            s = sum(int(M[i][j]) * int(v[j]) for j in range(5))
            new_v.append(s)
        v = new_v
    return v


def make_matadd_chain(k, problem_idx, rng):
    """
    Generate k distinct 5x5 integer matrices. Running sum S.
    Returns (matrices_as_list_of_lists, answer_as_list_of_lists).

    Reserved for future enhancement — the matadd condition (op_type="matadd")
    is not in generate_all_problems()'s cells dict, so this is not called by
    the current matvec-only experiment.
    """
    if NUMPY_AVAILABLE:
        matrices = [rng.choice(np.array([-2, -1, 1, 2]), size=25).reshape(5, 5) for _ in range(k)]
        S = matrices[0].astype(object).copy()
        for M in matrices[1:]:
            S = S + M.astype(object)
        mats_list = [_mat_to_list(M) for M in matrices]
        ans_list = [[int(x) for x in row] for row in S.tolist()]
    else:
        import random
        matrices = [[[random.choice([-2, -1, 1, 2]) for _ in range(5)] for _ in range(5)] for _ in range(k)]
        S = [[matrices[0][i][j] for j in range(5)] for i in range(5)]
        for M in matrices[1:]:
            S = [[S[i][j] + M[i][j] for j in range(5)] for i in range(5)]
        mats_list = matrices
        ans_list = S
    return mats_list, ans_list


def make_wide_chain(k_add, problem_idx, rng):
    """
    k_add mat-mat additions then one mat-vec.
    Returns (matrices_list, v0_list, answer_list).

    Reserved for future enhancement — the wide condition (op_type="wide") is
    not in generate_all_problems()'s cells dict, so this is not called by the
    current matvec-only experiment.
    """
    if NUMPY_AVAILABLE:
        matrices = [rng.choice(np.array([-2, -1, 1, 2]), size=25).reshape(5, 5) for _ in range(k_add)]
        while True:
            v0 = rng.integers(-3, 4, size=(5,))
            if np.any(v0 != 0):
                break
        S = matrices[0].astype(object).copy()
        for M in matrices[1:]:
            S = S + M.astype(object)
        answer = S @ v0.astype(object)
        mats_list = [_mat_to_list(M) for M in matrices]
        v0_list = _vec_to_list(v0)
        ans_list = [int(x) for x in answer.tolist()]
    else:
        import random
        matrices = [[[random.choice([-2, -1, 1, 2]) for _ in range(5)] for _ in range(5)] for _ in range(k_add)]
        while True:
            v0 = [random.randint(-3, 3) for _ in range(5)]
            if any(x != 0 for x in v0):
                break
        S = [[matrices[0][i][j] for j in range(5)] for i in range(5)]
        for M in matrices[1:]:
            S = [[S[i][j] + M[i][j] for j in range(5)] for i in range(5)]
        answer = [sum(S[i][j] * v0[j] for j in range(5)) for i in range(5)]
        mats_list = matrices
        v0_list = v0
        ans_list = answer
    return mats_list, v0_list, ans_list


def _get_rng(seed):
    if NUMPY_AVAILABLE:
        return np.random.default_rng(seed)
    else:
        import random
        random.seed(seed)
        return random  # duck-typed: only .integers used via wrappers above


def generate_all_problems():
    """
    Generate all shallow-chain problems. Returns a dict:
      { (op_type, k): [ {problem_id, matrices, v0, answer, actual_ops}, ... ] }
    30 problems per cell, deterministic from CELL_SEEDS.
    """
    cells = {
        ("matvec",  6):  ("matvec",  6),
        ("matvec", 11):  ("matvec", 11),
        ("matvec", 16):  ("matvec", 16),
        ("matvec", 19):  ("matvec", 19),
        ("matvec", 23):  ("matvec", 23),
        ("matvec", 26):  ("matvec", 26),
    }

    all_problems = {}
    for (op_type, k), _ in cells.items():
        seed = CELL_SEEDS[(op_type, k)]
        rng = _get_rng(seed)
        problems = []
        draw_idx = 0
        while len(problems) < PROBLEMS_PER_CELL:
            if op_type == "matvec":
                mats, v0, answer = make_matvec_chain(k, draw_idx, rng)
                answer_list = answer if isinstance(answer, list) else list(answer)
            elif op_type == "matadd":
                mats, answer = make_matadd_chain(k, draw_idx, rng)
                v0 = None
                answer_list = answer
            elif op_type == "wide":
                mats, v0, answer = make_wide_chain(k, draw_idx, rng)
                answer_list = answer
            else:
                raise ValueError(f"Unknown op_type: {op_type}")

            # Rejection: answer magnitude check (k=23/26 matvec chains naturally reach ~10^24-10^30)
            mag_limit = 10**32 if (op_type == "matvec" and k in (23, 26)) else ANSWER_MAGNITUDE_LIMIT
            if op_type == "matvec" or op_type == "wide":
                too_big = any(abs(x) > mag_limit for x in answer_list)
            else:  # matadd: answer is a 5x5 grid
                too_big = any(abs(x) > mag_limit
                              for row in answer_list for x in row)

            draw_idx += 1
            if too_big:
                continue  # discard, continue with same rng

            pid = f"OC_{op_type}_k{k:02d}_{len(problems):03d}"
            problems.append({
                "problem_id": pid,
                "draw_idx": draw_idx - 1,
                "matrices": mats,
                "v0": v0,
                "answer": answer_list,
                "actual_ops": ACTUAL_OPS[(op_type, k)],
            })

        all_problems[(op_type, k)] = problems
        print(f"  Generated {len(problems)} problems for ({op_type}, k={k})")

    return all_problems


def assert_problem_set_validity(all_problems):
    print("\n" + "="*70)
    print("ASSERTION CHECK & MAGNITUDE AUDIT")
    print("="*70)
    allowed_matrix_entries = {-2, -1, 1, 2}
    max_magnitude = 0
    for cell, probs in all_problems.items():
        for p in probs:
            for M in p["matrices"]:
                for row in M:
                    for val in row:
                        assert val in allowed_matrix_entries, f"Invalid matrix entry {val} in problem {p['problem_id']}"
            if p["v0"] is not None:
                for val in p["v0"]:
                    assert -3 <= val <= 3, f"Invalid v0 entry {val} in problem {p['problem_id']}"
            ans = p["answer"]
            if isinstance(ans[0], list):
                m = max(abs(x) for row in ans for x in row)
            else:
                m = max(abs(x) for x in ans)
            if m > max_magnitude:
                max_magnitude = m

    print("ASSERTION 1 PASSED: Every matrix entry across all problems is strictly in {-2, -1, 1, 2}.")
    print("ASSERTION 2 PASSED: Every v0 vector entry across all problems is strictly in range [-3, 3].")
    print(f"ACTUAL MAX MAGNITUDE across entire generated problem set: {max_magnitude}")
    print("="*70 + "\n")


# ===========================================================================
# Prompt builders
# ===========================================================================

def _format_matrix_block(matrices, label_prefix="A"):
    lines = []
    for idx, M in enumerate(matrices, 1):
        lines.append(f"{label_prefix}_{idx} =")
        for row in M:
            lines.append("[ " + "  ".join(f"{x:4d}" for x in row) + " ]")
        lines.append("")
    return "\n".join(lines)


def build_matvec_prompt(k, matrices, v0):
    matrix_block = _format_matrix_block(matrices, "A")
    v0_str = ", ".join(str(x) for x in v0)
    if k == 1:
        return (
            f"You are given 1 5×5 integer matrix A_1 and a starting\n"
            f"5-dimensional integer vector v_0. Compute the final vector v_1:\n\n"
            f"  v_1 = A_1 · v_0\n\n"
            f"Show vector v_1 in full.\n"
            f"Put your final answer v_1 as a comma-separated list of integers inside \\boxed{{}}.\n\n"
            f"Matrices:\n{matrix_block}\n"
            f"Starting vector v_0 = [{v0_str}]"
        )
    return (
        f"You are given {k} distinct 5×5 integer matrices A_1, A_2, ..., A_{k} and a starting\n"
        f"5-dimensional integer vector v_0. Compute the final vector v_{k} by applying each\n"
        f"matrix in order:\n\n"
        f"  v_1 = A_1 · v_0\n"
        f"  v_2 = A_2 · v_1\n"
        f"  ...\n"
        f"  v_{k} = A_{k} · v_{{{k-1}}}\n\n"
        f"Show ALL intermediate vectors v_1, v_2, ..., v_{k} in full.\n"
        f"Put your final answer v_{k} as a comma-separated list of integers inside \\boxed{{}}.\n\n"
        f"Matrices:\n{matrix_block}\n"
        f"Starting vector v_0 = [{v0_str}]"
    )


def build_matadd_prompt(k, matrices):
    """Reserved for future enhancement — pairs with make_matadd_chain(), not
    currently called by run_experiment()."""
    matrix_block = _format_matrix_block(matrices, "A")
    return (
        f"You are given {k} distinct 5×5 integer matrices A_1, A_2, ..., A_{k}.\n"
        f"Compute the running sum S by adding them in order:\n\n"
        f"  S_1 = A_1\n"
        f"  S_2 = S_1 + A_2\n"
        f"  ...\n"
        f"  S_{k} = S_{{{k-1}}} + A_{k}\n\n"
        f"Show ALL intermediate matrices S_1, S_2, ..., S_{k} in full (each as a 5×5 matrix).\n"
        f"Put your final answer S_{k} as a 5×5 grid of comma-separated integers inside \\boxed{{}}.\n"
        f"List rows left-to-right, top-to-bottom, separated by semicolons between rows.\n\n"
        f"Matrices:\n{matrix_block}"
    )


def build_wide_prompt(k_add, matrices, v0):
    """Reserved for future enhancement — pairs with make_wide_chain(), not
    currently called by run_experiment()."""
    matrix_block = _format_matrix_block(matrices, "B")
    v0_str = ", ".join(str(x) for x in v0)
    return (
        f"You are given {k_add} distinct 5×5 integer matrices B_1, B_2, ..., B_{k_add} and a\n"
        f"5-dimensional integer vector v_0.\n\n"
        f"Step 1 — Accumulate the matrix sum:\n"
        f"  S_1 = B_1\n"
        f"  S_i = S_{{i-1}} + B_i  for i = 2, ..., {k_add}\n\n"
        f"Step 2 — Compute the final vector:\n"
        f"  answer = S_{k_add} · v_0\n\n"
        f"Show ALL intermediate sums S_1 ... S_{k_add} and the final matrix-vector product in full.\n"
        f"Put your final answer as a comma-separated list of integers inside \\boxed{{}}.\n\n"
        f"Matrices:\n{matrix_block}\n"
        f"Starting vector v_0 = [{v0_str}]"
    )


def build_singledeep_prompt(problem_latex: str) -> str:
    """Use stored problem_latex verbatim.

    Reserved for future enhancement — pairs with load_eig_problems(), not
    currently called by run_experiment()."""
    return problem_latex.strip()


# ===========================================================================
# CSV loading for eigenvalue problems
# ===========================================================================

def load_eig_problems(csv_path: str) -> list[dict]:
    """Load C_5x5_eig_007 .. C_5x5_eig_036 from the benchmark CSV.

    Reserved for future enhancement — pairs with build_singledeep_prompt()
    and grade_eigenvalue(), not currently called by run_experiment()."""
    target_ids = {f"C_5x5_eig_{i:03d}" for i in range(7, 37)}
    problems = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["Problem_ID"].strip()
            if pid in target_ids:
                problems.append({
                    "problem_id": pid,
                    "problem_latex": row["problem_latex"],
                    "answer_latex": row.get("answer_latex", ""),
                    "actual_ops": 500,
                })
    problems.sort(key=lambda r: r["problem_id"])
    if len(problems) != 30:
        print(f"WARNING: expected 30 eigenvalue problems, found {len(problems)}")
    return problems


# ===========================================================================
# Grading
# ===========================================================================

def extract_boxed(text: str) -> str | None:
    """Extract content of the LAST \\boxed{} in text."""
    def _clean(s):
        s = re.sub(r'\\(?:,|;|!|quad|qquad|left|right|[ \t])', '', s)
        return s.replace('$', '')

    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return _clean(text[start:i - 1])
    return _clean(text[start:])  # unterminated — return remainder


def grade_matvec(boxed: str | None, ground_truth: list) -> tuple[int | None, bool]:
    """Returns (pass_val, parse_fail). pass_val=1 correct, 0 wrong, None=parse error."""
    if boxed is None:
        return None, True
    try:
        parts = [x.strip() for x in boxed.replace(";", ",").split(",")]
        parsed = [int(p) for p in parts if p]
        if len(parsed) != 5:
            return None, True
        correct = all(parsed[i] == int(ground_truth[i]) for i in range(5))
        return (1 if correct else 0), False
    except (ValueError, IndexError):
        return None, True


def grade_matadd(boxed: str | None, ground_truth: list) -> tuple[int | None, bool]:
    """ground_truth is 5x5 list of lists.

    Reserved for future enhancement — pairs with make_matadd_chain(), not
    currently called by run_experiment()."""
    if boxed is None:
        return None, True
    try:
        rows = [r.strip() for r in boxed.split(";") if r.strip()]
        if len(rows) != 5:
            return None, True
        parsed = []
        for r in rows:
            nums = [int(x.strip()) for x in r.split(",") if x.strip()]
            if len(nums) != 5:
                return None, True
            parsed.append(nums)
        correct = all(
            parsed[i][j] == int(ground_truth[i][j])
            for i in range(5) for j in range(5)
        )
        return (1 if correct else 0), False
    except (ValueError, IndexError):
        return None, True


def grade_eigenvalue(boxed: str | None, answer_latex: str) -> tuple[int | None, bool]:
    """
    Parse decimal eigenvalues from boxed content. Compare (unordered) to ground-truth
    eigenvalues parsed from answer_latex, tolerance 0.011.
    If numpy unavailable or parsing ground truth fails: return None (needs offline grading).

    Reserved for future enhancement — pairs with load_eig_problems() and
    build_singledeep_prompt(), not currently called by run_experiment().
    """
    if boxed is None:
        return None, True
    # Parse predicted values
    try:
        pred_strs = re.split(r'[,;\s]+', boxed.strip())
        pred_vals = [float(s) for s in pred_strs if s and s not in {'\\', ''}]
    except ValueError:
        return None, True

    # Parse ground truth from answer_latex
    if not answer_latex:
        return None, False  # no ground truth available

    try:
        gt_strs = re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', answer_latex)
        gt_vals = [float(s) for s in gt_strs]
    except ValueError:
        return None, False

    if len(gt_vals) != 5 or len(pred_vals) != 5:
        # Can't grade with mismatched count
        return None, False

    # Unordered set matching with tolerance 0.011
    tol = 0.011
    gt_remaining = list(gt_vals)
    matched = 0
    for pv in pred_vals:
        for i, gv in enumerate(gt_remaining):
            if abs(pv - gv) <= tol:
                matched += 1
                gt_remaining.pop(i)
                break
    return (1 if matched == 5 else 0), False


def find_first_error_step_matvec(response: str, matrices: list, v0: list) -> int | None:
    """
    Parse intermediate vectors from response. Return 1-indexed first wrong step, or None.
    Best-effort: only when response contains clearly formatted vectors.
    """
    # Look for patterns like v_1 = [...] or v_1 = [a, b, c, d, e]
    pattern = re.compile(r'v_\{?(\d+)\}?\s*=\s*\[([^\]]+)\]')
    matches = pattern.findall(response)
    if not matches:
        return None

    # Recompute ground truth step by step
    v = list(v0)
    gt_steps = {}
    for idx, M in enumerate(matrices, 1):
        v = [sum(int(M[i][j]) * int(v[j]) for j in range(5)) for i in range(5)]
        gt_steps[idx] = list(v)

    for step_str, vals_str in matches:
        step = int(step_str)
        if step not in gt_steps:
            continue
        try:
            vals = [int(x.strip()) for x in vals_str.split(",") if x.strip()]
            if len(vals) != 5:
                continue
            if vals != gt_steps[step]:
                return step
        except ValueError:
            continue
    return None


def find_first_error_component_matvec(boxed: str | None, ground_truth: list) -> str | None:
    """Return index of first wrong component or None."""
    if boxed is None:
        return None
    try:
        parts = [x.strip() for x in boxed.replace(";", ",").split(",")]
        parsed = [int(p) for p in parts if p]
        if len(parsed) != 5:
            return None
        for i in range(5):
            if parsed[i] != int(ground_truth[i]):
                return str(i)
        return None
    except (ValueError, IndexError):
        return None


# ===========================================================================
# Clopper–Pearson exact binomial CI (no scipy)
# ===========================================================================

def _beta_quantile(p, a, b, tol=1e-9, max_iter=200):
    """
    Compute quantile of Beta(a, b) distribution using bisection on the regularised
    incomplete beta function. For small n (n=30) this is fast enough.

    Not called by run_experiment() — used via clopper_pearson(), which is
    imported by build_opcount_aggregated.py to compute the per-cell accuracy
    CI in opcount_results_aggregated.csv.
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0

    def log_beta(a, b):
        import math
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    def reg_inc_beta(x, a, b):
        """Regularized incomplete beta via continued fraction (Lentz method)."""
        if x < 0 or x > 1:
            raise ValueError("x out of [0,1]")
        if x == 0:
            return 0.0
        if x == 1:
            return 1.0
        lbeta = log_beta(a, b)
        # Use symmetry relation for better convergence
        if x > (a + 1) / (a + b + 2):
            return 1.0 - reg_inc_beta(1 - x, b, a)
        front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
        # Continued fraction via Lentz
        TINY = 1e-300
        f = TINY
        C = f
        D = 0.0
        for m in range(200):
            for j in range(2):
                if j == 0:
                    if m == 0:
                        d = 1.0
                    else:
                        d = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
                else:
                    d = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
                D = 1.0 + d * D
                if abs(D) < TINY:
                    D = TINY
                D = 1.0 / D
                C = 1.0 + d / C
                if abs(C) < TINY:
                    C = TINY
                f *= C * D
                if abs(C * D - 1.0) < tol:
                    break
        return front * f

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        val = reg_inc_beta(mid, a, b)
        if abs(val - p) < tol:
            return mid
        if val < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k, n, alpha=0.05):
    """
    Return (lower, upper) Clopper-Pearson 95% CI for k successes out of n trials.
    Handles edge cases k=0 and k=n.

    Not called by run_experiment() — imported by build_opcount_aggregated.py
    to compute the ci_lower/ci_upper columns of opcount_results_aggregated.csv.
    """
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else _beta_quantile(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta_quantile(1 - alpha / 2, k + 1, n - k)
    return (lo, hi)


# ===========================================================================
# OpenRouter API call
# ===========================================================================

def call_api(model_slug: str, prompt: str, api_key: str, logger: logging.Logger) -> tuple[str | None, str, int | None]:
    """Call OpenRouter API with exponential backoff retries. Returns (response_text, finish_reason, reasoning_tokens)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/LinAlgBench/opcount-control",
        "X-Title": "LinAlgBench OpCount Control",
    }
    payload = {
        "model": model_slug,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    last_err = None
    for attempt in range(RETRY_ATTEMPTS + 1):  # 1 initial + 3 retries = 4 attempts total
        try:
            resp = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                choice = data["choices"][0]
                text = choice["message"]["content"]
                finish_reason = choice.get("finish_reason", "stop")
                usage = data.get("usage", {}) or {}
                reasoning_tok = (usage.get("completion_tokens_details", {}) or {}).get("reasoning_tokens")
                if reasoning_tok is None:
                    reasoning_tok = usage.get("reasoning_tokens")
                return text, finish_reason, reasoning_tok
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                logger.warning("API error attempt %d/%d model=%s: %s",
                               attempt + 1, RETRY_ATTEMPTS + 1, model_slug, last_err)
        except Exception as e:
            last_err = str(e)
            logger.warning("API exception attempt %d/%d model=%s: %s",
                           attempt + 1, RETRY_ATTEMPTS + 1, model_slug, last_err)
        if attempt < RETRY_ATTEMPTS:
            sleep_t = RETRY_BASE_SLEEP * (2 ** attempt)
            time.sleep(sleep_t)

    logger.error("All attempts failed for model=%s last_error=%s", model_slug, last_err)
    return None, f"error: {last_err}", None


# ===========================================================================
# Resume support: load already-completed rows
# ===========================================================================

RAW_CSV_FIELDS = [
    "phase", "model", "condition", "op_type", "k", "actual_ops", "problem_id",
    "repeat", "prompt_text", "response_text", "boxed_content",
    "pass", "parse_fail", "finish_reason", "reasoning_tokens", "first_error_step",
]


def load_completed(raw_csv_path: Path, phase: str | None = None) -> set:
    """Return set of (model, condition, problem_id, repeat) already done.
    If phase is given, only rows from that phase count — otherwise an earlier
    phase's row (e.g. the 'test' phase's failed k=6 attempt) would look like
    it already satisfies a later phase's requirement for the same problem_id,
    since problem_id is deterministic from k alone and collides across phases."""
    done = set()
    if not raw_csv_path.exists():
        return done
    with open(raw_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if phase is not None and row.get("phase") != phase:
                continue
            done.add((row["model"], row["condition"], row["problem_id"], row["repeat"]))
    return done


def append_raw_row(raw_csv_path: Path, row: dict, write_header: bool = False):
    needs_header = not raw_csv_path.exists() or raw_csv_path.stat().st_size == 0
    with open(raw_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_CSV_FIELDS, extrasaction="ignore")
        if needs_header or write_header:
            writer.writeheader()
        writer.writerow(row)


# ===========================================================================
# Abort check (internal convenience tooling, not a formal deliverable) — see
# opcount_abort_check.py --models ..., which composes run_experiment() and
# build_opcount_aggregated.py's accuracy math into a k=6 gate check over a
# user-specified list of models.
# ===========================================================================

# ===========================================================================
# Micro-Pilot Execution (ThreadPoolExecutor with 8 workers, 24 calls total)
# ===========================================================================

from concurrent.futures import ThreadPoolExecutor, as_completed

# ===========================================================================
# Test Run Execution (ThreadPoolExecutor with 8 workers, 36 calls total)
# ===========================================================================

from concurrent.futures import ThreadPoolExecutor, as_completed

def run_experiment(
    model: str,
    k: int,
    n: int,
    all_shallow: dict,
    api_key: str,
    raw_csv_path: Path,
    logger: logging.Logger,
):
    slug = MODELS[model]
    print("\n" + "="*70)
    print(f"RUN: model={model} k={k} N={n} max_tokens={MAX_TOKENS}")
    print("="*70)

    completed = load_completed(raw_csv_path, phase="phase2_frontier")
    rungs = [k]
    tasks = []
    for kk in rungs:
        prob_list = all_shallow[("matvec", kk)][:n]
        for prob in prob_list:
            key = (model, "CHAINED-SHALLOW", prob["problem_id"], "1")
            if key in completed:
                print(f"  [SKIP] {prob['problem_id']} already done (phase2_frontier)")
                continue
            tasks.append((model, slug, kk, prob))

    if not tasks:
        print("Nothing to do — every requested row is already in the CSV for phase2_frontier.")
        return

    results = []

    def execute_task(task):
        paper_name, slug, k, prob = task
        prompt = build_matvec_prompt(k, prob["matrices"], prob["v0"])
        response, finish_reason, reasoning_tok = call_api(slug, prompt, api_key, logger)
        boxed = extract_boxed(response) if response else None
        pass_val, parse_fail = grade_matvec(boxed, prob["answer"])
        first_err = find_first_error_step_matvec(response or "", prob["matrices"], prob["v0"])
        row = {
            "phase": "phase2_frontier",
            "model": paper_name,
            "condition": "CHAINED-SHALLOW",
            "op_type": "matvec",
            "k": k,
            "actual_ops": prob["actual_ops"],
            "problem_id": prob["problem_id"],
            "repeat": 1,
            "prompt_text": prompt,
            "response_text": response or "",
            "boxed_content": boxed or "",
            "pass": pass_val if pass_val is not None else "",
            "parse_fail": 1 if parse_fail else 0,
            "finish_reason": finish_reason,
            "reasoning_tokens": reasoning_tok if reasoning_tok is not None else "",
            "first_error_step": first_err if first_err is not None else "",
            "ground_truth": str(prob["answer"]),
        }
        return row

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(execute_task, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            append_raw_row(raw_csv_path, res)

    # Score this one (model, k) cell
    total = len(results)
    correct = sum(1 for r in results if str(r["pass"]) == "1")
    parse_fails = sum(1 for r in results if r["parse_fail"] == 1)
    truncations = sum(1 for r in results if r["finish_reason"] == "length")
    failing_or_truncated = [r for r in results if str(r["pass"]) != "1" or r["finish_reason"] == "length"]

    print("\n" + "="*70)
    print(f"SCORE: model={model} k={k} -> {correct}/{total}  "
          f"(parse_fails={parse_fails}, truncations={truncations})")
    print("="*70)

    if failing_or_truncated:
        print("!"*70)
        print(f"FAILING OR TRUNCATED RESPONSES IN FULL ({len(failing_or_truncated)} total)")
        print("!"*70)
        for idx, r in enumerate(failing_or_truncated, 1):
            print(f"\n--- ITEM #{idx}: Model={r['model']} k={r['k']} ProblemID={r['problem_id']} ---")
            print(f"Ground Truth:    {r['ground_truth']}")
            print(f"Boxed Extracted: {repr(r['boxed_content'])}")
            print(f"Parse Fail Flag: {r['parse_fail']}")
            print(f"Finish Reason:   {r['finish_reason']}")
            print("Full Response Text:")
            print("-" * 50)
            print(r['response_text'])
            print("-" * 50)
    else:
        print(f"\nALL {total} CALLS PASSED. ({total}/{total})")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    global MAX_TOKENS
    run_start = datetime.datetime.utcnow().isoformat() + "Z"

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()), help="run one cell for this model instead of the gate test")
    ap.add_argument("--k", type=int, nargs="+", help="one or more chain lengths k")
    ap.add_argument("--N", type=int, default=PROBLEMS_PER_CELL, help="number of problems per k (default 15)")
    ap.add_argument("--max-tokens", type=int, dest="max_tokens", help="max_tokens, applied to every k in this run")
    args = ap.parse_args()

    # --- Setup ---
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit(
            "ERROR: OPENROUTER_API_KEY environment variable is not set.\n"
            "Run: export OPENROUTER_API_KEY=your_key_here"
        )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    raw_csv_path = output_dir / "opcount_results_raw.csv"

    # --- Generate shallow-chain problems ---
    print("Generating shallow-chain problems (deterministic from cell seeds)...")
    all_shallow = generate_all_problems()
    print(f"  Total shallow cells generated: {len(all_shallow)}")

    # --- Requirement (1): Assertions & Magnitude audit ---
    assert_problem_set_validity(all_shallow)

    if not (args.model and args.k):
        sys.exit("Usage: python3 run_opcount_experiment.py --model <name> --k <k1> [<k2> ...] [--N <n>] [--max-tokens <t>]\n"
                  f"  --model choices: {list(MODELS.keys())}")

    if args.max_tokens:
        MAX_TOKENS = args.max_tokens
    for k in args.k:
        run_experiment(args.model, k, args.N, all_shallow, api_key, raw_csv_path, logger)


if __name__ == "__main__":
    main()
