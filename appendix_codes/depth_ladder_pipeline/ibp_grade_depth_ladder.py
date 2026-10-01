#!/usr/bin/env python3
# Module:    ibp_grade_depth_ladder.py
# Version:   1.0
"""
IBP Depth-Ladder grader — the symbolic grading procedure this benchmark
requires, implemented literally:

    1. Extract the content of the LAST \\boxed{...}. Absent -> parse_fail.
    2. Strip cosmetic LaTeX (\\left, \\right, +C, ...).
    3. Parse to sympy. Failure -> parse_fail.
    4. diff_check = trigsimp(diff(F_model, x) - integrand)
    5. diff_check == 0  ->  pass

There is no string matching: a model answer in any algebraically different
but equivalent form passes, because only its derivative is compared.

WHY NOT ibp_symbolic_check.check_answer
───────────────────────────────────────
That function prefers an "anchored" route: when answer_latex is supplied it
compares against the stored answer and returns *without ever differentiating*
(ibp_symbolic_check.py:222-227). The bank carries answer_latex for all 340
problems, so the anchored route would always win and the differentiate-and-
check would never run. This module therefore drives the integrand route
directly. It reuses that module's parsing helpers, which are well tested.

WHY THE INTEGRAND IS REBUILT FROM COLUMNS
─────────────────────────────────────────
ibp_symbolic_check.integrand_from_id() only recognises the older
`ibp_pts_n7_b19` / `ibp_n5_a2b3` id spellings and returns None for this bank's
`IBP_n00_sin_b01_00` format. The bank gives n, trig and b as real columns, so
the integrand is built from those — no id parsing, nothing to misread.

USAGE
─────
    # Self-test: grade the bank's own answers. Must be 340/340.
    python3 ibp_grade_depth_ladder.py --self-test

    # As a library
    from ibp_grade_depth_ladder import grade_response
    passed, parse_fail, boxed = grade_response(text, n=5, trig='sin', b=3)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import sympy as sp

from ibp_symbolic_check import extract_boxed, parse_answer

x = sp.Symbol('x')


# ─────────────────────────────────────────────────────────────────────────
#  Integrand reconstruction
# ─────────────────────────────────────────────────────────────────────────
def build_integrand(n: int, trig: str, b: int):
    """x^n sin(bx)  or  x^n cos(bx) — the problem the model was asked to integrate."""
    trig = trig.strip().lower()
    if trig == 'sin':
        return x**n * sp.sin(b * x)
    if trig == 'cos':
        return x**n * sp.cos(b * x)
    raise ValueError(f"trig must be 'sin' or 'cos', got {trig!r}")


# ─────────────────────────────────────────────────────────────────────────
#  Grading
# ─────────────────────────────────────────────────────────────────────────
def grade_response(
    response_text: str,
    n: int,
    trig: str,
    b: int,
) -> tuple[bool, bool, str]:
    """Grade one response. Returns (passed, parse_fail, boxed_content).

    boxed_content is the raw extracted \\boxed{} body, or "" when absent —
    it is written to results_raw.csv either way.

    Never raises: a grader that dies mid-run on one malformed response would
    cost a re-run of the whole model, so every failure mode maps to a verdict.
    """
    boxed = extract_boxed(response_text or '')
    if not boxed:
        return False, True, ''

    # parse_answer applies clean_latex (strips \left, \right, +C, \displaystyle)
    # and rejects any expression carrying a free symbol other than x.
    expr = parse_answer(boxed)
    if expr is None:
        return False, True, boxed

    try:
        integrand = build_integrand(n, trig, b)
        if sp.trigsimp(sp.diff(expr, x) - integrand) == 0:
            return True, False, boxed
    except Exception:
        # A parse that succeeded but produced something undifferentiable is a
        # wrong answer, not a parse failure — the box was readable.
        return False, False, boxed

    return False, False, boxed


# ─────────────────────────────────────────────────────────────────────────
#  Self-test
# ─────────────────────────────────────────────────────────────────────────
def self_test(bank_path: str) -> int:
    """Grade every stored answer_latex as if a model had returned it.

    This is the gate before any API spend: the grader must accept all 340
    known-correct answers. A regression here means the grader would silently
    mark correct model output as wrong across the whole run.
    """
    with open(bank_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    n_pass = n_parse_fail = n_mismatch = 0
    slowest, slowest_id = 0.0, ''
    t0 = time.time()

    for r in rows:
        # Wrap in \boxed{} so the full extract -> clean -> parse path is exercised,
        # not just the comparison.
        fake_response = f"Therefore the antiderivative is \\boxed{{{r['answer_latex']}}}."
        t_row = time.time()
        passed, parse_fail, _ = grade_response(
            fake_response, int(r['n']), r['trig'], int(r['b'])
        )
        dt = time.time() - t_row
        if dt > slowest:
            slowest, slowest_id = dt, r['Problem_ID']

        if passed:
            n_pass += 1
        elif parse_fail:
            n_parse_fail += 1
            print(f"  PARSE_FAIL  {r['Problem_ID']}")
        else:
            n_mismatch += 1
            print(f"  MISMATCH    {r['Problem_ID']}")

    total = len(rows)
    print()
    print(f"  total       {total}")
    print(f"  pass        {n_pass}")
    print(f"  mismatch    {n_mismatch}")
    print(f"  parse_fail  {n_parse_fail}")
    print(f"  wall        {time.time() - t0:.1f}s   slowest row {slowest:.2f}s ({slowest_id})")
    print()

    # Regression: step 2 strips "+C", "+ C_1", "+ C_{2}" and similar trailing
    # constants before parsing. Each sample answer, already proven correct
    # above, must still grade correct with a constant glued on.
    variants = ['+C', '+ C_1', '+ C_{2}', '+\\text{const}']
    sample = [rows[i] for i in range(0, len(rows), max(1, len(rows) // 10))]
    n_const = checked = 0
    for r in sample:
        for v in variants:
            fake = f"\\boxed{{{r['answer_latex']}{v}}}"
            passed, parse_fail, _ = grade_response(
                fake, int(r['n']), r['trig'], int(r['b'])
            )
            checked += 1
            if passed:
                n_const += 1
            else:
                print(f"  CONST-REG R  {r['Problem_ID']}  variant {v!r}  "
                      f"parse_fail={parse_fail}")
    print(f"  const-reg   {n_const}/{checked} (+C / +C_1 / +C_{{2}} / +\\text const variants)")
    print()

    if n_pass == total and n_const == checked:
        print(f"  SELF-TEST PASSED — {n_pass}/{total}")
        return 0
    print(f"  SELF-TEST FAILED — {n_pass}/{total}")
    return 1


# There is no default bank; --bank is required, the same as every other path
# in this tree. This script touches no run output, so it needs no
# --projects-root.


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self-test', action='store_true',
                    help='grade the bank\'s own answers; must be 340/340')
    ap.add_argument('--bank', required=True,
                    help='path to ibp_depth_ladder_problems.csv, e.g. '
                         '../projects/source/ibp_depth_ladder_problems.csv')
    args = ap.parse_args()

    if args.self_test:
        print(f"\nPATHS\n  bank            {os.path.abspath(args.bank)}\n")
        return self_test(args.bank)

    ap.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
