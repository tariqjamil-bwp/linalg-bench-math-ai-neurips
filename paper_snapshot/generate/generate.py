
#!/usr/bin/env python3
"""
LinAlgBench: Problem Generator
Generates benchmark problems for any matrix size with user-defined counts.

Outputs:
  linalg_bench_{N}x{N}.csv         — main dataset → 1_inference.py
  linalg_bench_{N}x{N}_formats.csv — format sensitivity → 4_format_sensitivity.py

Usage:
  python generate.py --size 3 --seed 1
  python generate.py --size 4 --seed 2093
  python generate.py --size 5 --seed 5555
"""

import numpy as np
import csv
import argparse
from datetime import datetime

# =============================================================================
# UTILITIES
# =============================================================================

def to_latex_matrix(mat):
    mat = np.array(mat)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    lines = [" & ".join(str(int(x)) for x in row) for row in mat]
    return "\\begin{bmatrix} " + " \\\\ ".join(lines) + " \\end{bmatrix}"

def to_ascii_matrix(mat):
    mat = np.array(mat)
    if mat.ndim == 1:
        return "[" + ", ".join(str(int(x)) for x in mat) + "]"
    return "[" + ", ".join("[" + ", ".join(str(int(x)) for x in row) + "]" for row in mat) + "]"

def to_visual_matrix(mat):
    mat = np.array(mat)
    w = max(len(str(int(x))) for x in mat.flat)
    lines = ["| " + "  ".join(str(int(x)).rjust(w) for x in row) + " |" for row in mat]
    return "\n".join(lines)

# =============================================================================
# PROBLEM GENERATORS (size-agnostic)
# =============================================================================

def gen_trace(n, idx):
    A = np.random.randint(-9, 10, size=(n, n))
    trace = int(np.trace(A))
    return dict(
        subcategory="trace", cognitive_level="Reading",
        problem_text=f"Find the trace of the {n}×{n} matrix A.",
        matrix=A, answer_value=trace,
        answer_latex=f"\\text{{tr}}(A) = {trace}",
    )

def gen_transpose(n, idx):
    A = np.random.randint(-9, 10, size=(n, n))
    AT = A.T
    return dict(
        subcategory="transpose", cognitive_level="Reading",
        problem_text=f"Find the transpose of the {n}×{n} matrix A.",
        matrix=A, answer_matrix=AT,
        answer_latex=f"A^T = {to_latex_matrix(AT)}",
    )

def gen_matrix_vector(n, idx):
    A = np.random.randint(-5, 6, size=(n, n))
    x = np.random.randint(-5, 6, size=(n,))
    b = A @ x
    return dict(
        subcategory="matrix_vector", cognitive_level="Arithmetic",
        problem_text=f"Compute the matrix-vector product Ax.",
        matrix=A, vector=x, answer_values=b.tolist(),
        answer_latex=f"Ax = {to_latex_matrix(b)}",
    )

def gen_multiplication(n, idx):
    r = 4 if n <= 4 else 3
    A = np.random.randint(-r, r+1, size=(n, n))
    B = np.random.randint(-r, r+1, size=(n, n))
    C = A @ B
    return dict(
        subcategory="multiplication", cognitive_level="Arithmetic",
        problem_text=f"Compute the matrix product AB.",
        matrix=A, matrix_B=B, answer_matrix=C,
        answer_latex=f"AB = {to_latex_matrix(C)}",
    )

def gen_matrix_power(n, idx):
    r = 3 if n <= 4 else 2
    A = np.random.randint(-r, r+1, size=(n, n))
    A2 = A @ A
    return dict(
        subcategory="matrix_power", cognitive_level="Arithmetic",
        problem_text=f"Compute A² for the {n}×{n} matrix A.",
        matrix=A, answer_matrix=A2,
        answer_latex=f"A^2 = {to_latex_matrix(A2)}",
    )

def gen_rank(n, idx):
    choices = list(range(max(1, n-2), n+1))
    probs = np.ones(len(choices)); probs /= probs.sum()
    rank_target = np.random.choice(choices, p=probs)
    if rank_target == n:
        while True:
            A = np.random.randint(-5, 6, size=(n, n))
            if abs(np.linalg.det(A)) > 0.5:
                break
    else:
        A = np.random.randint(-4, 5, size=(rank_target, n))
        while A.shape[0] < n:
            coeffs = np.random.randint(-2, 3, size=A.shape[0])
            A = np.vstack([A, coeffs @ A])
        np.random.shuffle(A)
    actual_rank = int(np.linalg.matrix_rank(A))
    return dict(
        subcategory="rank", cognitive_level="Sequential",
        problem_text=f"Find the rank of the {n}×{n} matrix A.",
        matrix=A, answer_value=actual_rank,
        answer_latex=f"\\text{{rank}}(A) = {actual_rank}",
    )

def gen_nullity(n, idx):
    choices = list(range(max(1, n-2), n+1))
    probs = np.ones(len(choices)); probs /= probs.sum()
    rank_target = np.random.choice(choices, p=probs)
    if rank_target == n:
        while True:
            A = np.random.randint(-5, 6, size=(n, n))
            if abs(np.linalg.det(A)) > 0.5:
                break
    else:
        A = np.random.randint(-4, 5, size=(rank_target, n))
        while A.shape[0] < n:
            coeffs = np.random.randint(-2, 3, size=A.shape[0])
            A = np.vstack([A, coeffs @ A])
        np.random.shuffle(A)
    actual_rank = int(np.linalg.matrix_rank(A))
    nullity = n - actual_rank
    return dict(
        subcategory="nullity", cognitive_level="Sequential",
        problem_text=f"Find the nullity of the {n}×{n} matrix A.",
        matrix=A, answer_value=nullity,
        answer_latex=f"\\text{{nullity}}(A) = {nullity}",
    )

def gen_determinant(n, idx):
    A = np.random.randint(-5, 6, size=(n, n))
    det = int(round(np.linalg.det(A)))
    return dict(
        subcategory="determinant", cognitive_level="Recursive",
        problem_text=f"Find the determinant of the {n}×{n} matrix A.",
        matrix=A, answer_value=det,
        answer_latex=f"\\det(A) = {det}",
    )

def gen_eigenvalues(n, idx):
    A = np.random.randint(-4, 5, size=(n, n))
    A = A + A.T
    eigs = np.sort(np.real(np.linalg.eigvals(A)))
    eigs_rounded = [round(float(e), 4) for e in eigs]
    return dict(
        subcategory="eigenvalues", cognitive_level="Compositional",
        problem_text=f"Find all eigenvalues of the {n}×{n} symmetric matrix A.",
        matrix=A, answer_values=eigs_rounded,
        answer_latex=f"\\lambda = {', '.join(str(e) for e in eigs_rounded)}",
    )

def gen_ata(n, idx):
    """3×3 only — AtA multiplication."""
    A = np.random.randint(-5, 6, size=(n, n))
    result = A.T @ A
    return dict(
        subcategory="transpose_product", cognitive_level="Arithmetic",
        problem_text=f"Compute the product AᵀA (A-transpose times A).",
        matrix=A, answer_matrix=result,
        answer_latex=f"A^T A = {to_latex_matrix(result)}",
    )

# =============================================================================
# TASK SCHEDULE — exact counts from paper (same across all sizes)
# =============================================================================

def get_schedule(n, total=None):
    """
    Fixed counts matching Table 1 in the paper.
    Identical distribution across 3×3, 4×4, and 5×5.
    Total: 220 problems per size.
    """
    return [
        ("determinant",   gen_determinant,   50),
        ("eigenvalues",   gen_eigenvalues,   30),
        ("rank",          gen_rank,          20),
        ("nullity",       gen_nullity,       20),
        ("trace",         gen_trace,         20),
        ("matrix_vector", gen_matrix_vector, 20),
        ("multiplication",gen_multiplication,20),
        ("matrix_power",  gen_matrix_power,  20),
        ("transpose",     gen_transpose,     20),
    ]
    weights = np.array([t[2] for t in base_tasks])
    counts  = np.round(weights / weights.sum() * total).astype(int)

    # Fix rounding so total is exact
    diff = total - counts.sum()
    counts[np.argmax(weights)] += diff

    return list(zip(names, fns, counts.tolist()))

# =============================================================================
# BUILD PROBLEMS
# =============================================================================

def build_problem(n, size_str, subcategory, fn, idx):
    raw = fn(n, idx)
    A   = np.array(raw["matrix"])

    # Build problem_latex — handle two-matrix problems (multiplication)
    if "matrix_B" in raw:
        B = np.array(raw["matrix_B"])
        problem_latex = f"A = {to_latex_matrix(A)}, \\quad B = {to_latex_matrix(B)}"
        problem_ascii = f"A = {to_ascii_matrix(A)}, B = {to_ascii_matrix(B)}"
        problem_visual = f"A:\n{to_visual_matrix(A)}\nB:\n{to_visual_matrix(B)}"
    elif "vector" in raw:
        x = np.array(raw["vector"])
        problem_latex = f"A = {to_latex_matrix(A)}, \\quad x = {to_latex_matrix(x)}"
        problem_ascii = f"A = {to_ascii_matrix(A)}, x = {to_ascii_matrix(x)}"
        problem_visual = f"A:\n{to_visual_matrix(A)}\nx: {to_ascii_matrix(x)}"
    else:
        problem_latex = f"A = {to_latex_matrix(A)}"
        problem_ascii = f"A = {to_ascii_matrix(A)}"
        problem_visual = to_visual_matrix(A)

    # Unified answer_latex
    answer_latex = raw.get("answer_latex", "")

    return {
        "id":            f"C_{size_str}_{subcategory}_{idx:03d}",
        "subcategory":   subcategory,
        "size":          size_str,
        "problem_text":  raw["problem_text"],
        "problem_latex": problem_latex,
        "problem_ascii": problem_ascii,
        "problem_visual": problem_visual,
        "answer_latex":  answer_latex,
    }

# =============================================================================
# GENERATE & WRITE CSVs
# =============================================================================

def generate(n, seed):
    np.random.seed(seed)
    size_str = f"{n}x{n}"
    schedule = get_schedule(n)
    total = sum(c for _, _, c in schedule)

    problems = []
    print(f"\nGenerating {n}×{n} problems (seed={seed}, total={total})...")
    for subcategory, fn, count in schedule:
        for i in range(count):
            problems.append(build_problem(n, size_str, subcategory, fn, i + 1))
        print(f"  ✓ {count:>4}  {subcategory}")
    print(f"  {'─'*24}")
    print(f"  Total: {len(problems)}")
    return problems

def write_main_csv(problems, n):
    """Main CSV → 1_inference.py"""
    path = f"linalg_bench_{n}x{n}.csv"
    fields = ["id", "subcategory", "size", "problem_text", "problem_latex", "answer_latex"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(problems)
    print(f"\nMain dataset     → {path}  ({len(problems)} problems)")

def write_format_csv(problems, n):
    """Format sensitivity CSV → 4_format_sensitivity.py
    Each problem appears 3 times: latex, ascii, visual.
    """
    path = f"linalg_bench_{n}x{n}_formats.csv"
    fields = ["id", "subcategory", "size", "format",
              "problem_text", "problem_representation", "answer_latex"]
    rows = []
    for p in problems:
        for fmt, col in [("latex",  "problem_latex"),
                         ("ascii",  "problem_ascii"),
                         ("visual", "problem_visual")]:
            rows.append({
                "id":                     p["id"],
                "subcategory":            p["subcategory"],
                "size":                   p["size"],
                "format":                 fmt,
                "problem_text":           p["problem_text"],
                "problem_representation": p[col],
                "answer_latex":           p["answer_latex"],
            })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Format sensitivity → {path}  ({len(problems)} problems × 3 formats = {len(rows)} rows)")

# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LinAlgBench Problem Generator")
    parser.add_argument("--size", type=int, required=True, choices=[3, 4, 5],
                        help="Matrix size: 3, 4, or 5")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    problems = generate(args.size, args.seed)
    write_main_csv(problems, args.size)
    write_format_csv(problems, args.size)
