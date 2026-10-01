#!/usr/bin/env python3
# Module:    ibp_symbolic_check.py
# Version:   1.0
"""
Deterministic (no-LLM) correctness check for IBP answers.

The idea
--------
An antiderivative is correct when its derivative is the original integrand.
That is a proof, not an opinion, so it needs no judge model:

    d/dx (model_answer)  ==  integrand      ->  correct

This also sidesteps the "+C" problem that makes a direct string or expression
comparison against the stored answer useless: two correct antiderivatives may
differ by any constant, but their derivatives are identical.

Why numeric and not symbolic
----------------------------
sympy.simplify() on this project's high-n answers (11 nested IBP applications,
22-digit denominators) can run for minutes or never finish. Evaluating both
sides at a handful of sample points is bounded work and, at 60-digit
precision, has no realistic chance of a false pass. Precision matters here:
coefficients reach ~1e19 and e^(61x) is astronomically large, so float64
would lose every significant digit.

Verdicts
--------
    MATCH       derivative equals the integrand at every sample point
    MISMATCH    parsed cleanly, but the derivative is not the integrand
    PARSE_FAIL  the boxed LaTeX could not be turned into an expression
    NO_BOXED    the response has no \\boxed{} to check

Only PARSE_FAIL and NO_BOXED need to fall through to the LLM check.
"""

from __future__ import annotations

import json
import os
import re

import sympy as sp

x = sp.Symbol('x')

# parse_latex maps a bare "e" to Symbol('e'), not Euler's number.
_E_SYM = sp.Symbol('e')

# Sample points for the numeric identity test. Deliberately irrational-ish and
# away from 0, where many wrong answers coincidentally agree with the right one.
_SAMPLES = (sp.Rational(3, 10), sp.Rational(7, 10), sp.Rational(11, 10),
            sp.Rational(19, 10), sp.Rational(23, 10))
_PRECISION = 60          # decimal digits carried through evaluation
_REL_TOL = sp.Float('1e-25')


# ─────────────────────────────────────────────────────────────────────────
#  Boxed-answer extraction
# ─────────────────────────────────────────────────────────────────────────
def extract_boxed(response: str) -> str | None:
    """Return the content of the LAST \\boxed{...}, matching braces properly.

    A regex cannot do this: the answers contain nested braces (\\frac{}{},
    x^{5}, \\sin{\\left(...\\right)}), so brace depth has to be counted.
    The last box wins because models often box an intermediate step first.
    """
    if not response:
        return None
    starts = [m.end() for m in re.finditer(r'\\boxed\s*\{', response)]
    for start in reversed(starts):
        depth = 1
        for i in range(start, len(response)):
            c = response[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    body = response[start:i].strip()
                    return body or None
    return None


# ─────────────────────────────────────────────────────────────────────────
#  LaTeX cleanup
# ─────────────────────────────────────────────────────────────────────────
# Constructs parse_latex either rejects or mis-reads, stripped before parsing.
# Order matters: spacing commands go first so they cannot glue tokens together.
_STRIP = [
    (r'\\displaystyle', ' '),
    (r'\\[,;:!]', ' '),               # \, \; \: \!  thin-space commands
    (r'\\quad|\\qquad', ' '),
    (r'\\left\s*', ''),               # \left( -> (   parse_latex copes, but
    (r'\\right\s*', ''),              # dropping them avoids edge cases
    (r'\\dfrac|\\tfrac', r'\\frac'),
    (r'\\mathrm\s*\{\s*d\s*\}', 'd'),
    (r'\\text\s*\{[^}]*\}', ' '),     # \text{...} annotations carry no math
    (r'\\!', ''),
]

# A trailing integration constant is expected and must not break parsing.
# Strips "+C", "+ C_1", "+ C_{2}", "c_1", "K_1" — i.e. any of C/c/K/constant
# with an optional numeric subscript, preceded by a +/- operator. Anchored
# at the end so "x + C" mid-expression is untouched; models write the
# constant only after the last term.
_TRAILING_C = re.compile(r'[+\-]\s*(?:C|c|K|constant)(?:_\d+|_\{\s*\d+\s*\})?\s*$')


def clean_latex(s: str) -> str:
    """Normalise model LaTeX into something parse_latex accepts."""
    s = s.strip()
    for pat, rep in _STRIP:
        s = re.sub(pat, rep, s)
    s = _TRAILING_C.sub('', s).strip()
    # \text{const} becomes a space above, leaving a dangling "+" or "-" that
    # the constant regex can no longer see; a bare trailing operator is never
    # meaningful in a valid expression, so drop it.
    s = re.sub(r'[+\-]\s*$', '', s)
    s = s.rstrip('.').rstrip('$').strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def parse_answer(latex: str):
    """LaTeX -> sympy expression, or None if it cannot be parsed."""
    from sympy.parsing.latex import parse_latex
    try:
        expr = parse_latex(clean_latex(latex))
    except Exception:
        return None
    if expr is None:
        return None
    # parse_latex yields Symbol('e'); the answers mean Euler's number.
    if _E_SYM in expr.free_symbols:
        expr = expr.subs(_E_SYM, sp.E)
    # Anything left other than x means the parse went wrong (stray letters
    # from prose, a mangled command name, an unbound constant).
    if expr.free_symbols - {x}:
        return None
    return expr


# ─────────────────────────────────────────────────────────────────────────
#  The check
# ─────────────────────────────────────────────────────────────────────────
def integrand_from_id(qid: str):
    """Rebuild the integrand from the problem id.

    Used when the bank's integrand_sympy is missing, which is the case for a
    few legacy rows. The id encodes the whole problem by construction:
        ibp_n{n}_a{a}b{b}   ->  x^n e^(ax) sin(bx)
        ibp_pts_n{n}_b{b}   ->  x^n sin(bx)
        ibp_ptc_n{n}_b{b}   ->  x^n cos(bx)
    """
    m = re.match(r'^ibp_n(\d+)_a(\d+)b(\d+)$', qid or '')
    if m:
        n, a, b = map(int, m.groups())
        return x**n * sp.exp(a * x) * sp.sin(b * x)
    m = re.match(r'^ibp_(pts|ptc)(?:_ext\d*)?_n(\d+)_b?(\d+)$', qid or '')
    if m:
        trig = sp.sin if m.group(1) == 'pts' else sp.cos
        n, b = int(m.group(2)), int(m.group(3))
        return x**n * trig(b * x)
    return None


def _is_zero_numerically(expr) -> bool:
    """True when expr evaluates to 0 at every sample point.

    Compared relatively, because the terms involved span many orders of
    magnitude; an absolute tolerance would be meaningless against e^(61x).
    """
    for pt in _SAMPLES:
        try:
            val = sp.N(expr.subs(x, pt), _PRECISION)
        except Exception:
            return False
        if not val.is_number or val.has(sp.zoo, sp.nan, sp.oo):
            return False
        scale = max(abs(sp.N(pt, _PRECISION)), sp.Integer(1))
        if abs(val) > _REL_TOL * max(scale, abs(val) + 1):
            return False
    return True


def _differs_by_constant(a, b) -> bool:
    """True when a - b is the same constant at every sample point.

    This is the anchored test: two antiderivatives of the same integrand
    differ by a constant and nothing else, so "+C" needs no special handling.
    Mirrors how the det pipeline anchors on the bank's answer_latex, but
    compared as functions rather than as a single number.
    """
    diffs = []
    for pt in _SAMPLES:
        try:
            val = sp.N((a - b).subs(x, pt), _PRECISION)
        except Exception:
            return False
        if not val.is_number or val.has(sp.zoo, sp.nan, sp.oo):
            return False
        diffs.append(val)
    spread = max(diffs) - min(diffs)
    scale = max(abs(d) for d in diffs) + 1
    return abs(spread) <= _REL_TOL * scale


def check_answer(response: str, qid: str = '', integrand_sympy: str = '',
                 answer_latex: str = '') -> tuple[str, str]:
    """Verdict for one response. Returns (verdict, reason).

    Two routes to a verdict, preferred order:
      1. ANCHORED — compare against the bank's answer_latex. Correct when the
         two differ by a constant. This is the analogue of the det pipeline's
         ground-truth anchoring, and needs no integrand.
      2. INTEGRAND — differentiate and compare against the integrand. Used
         when the bank answer is absent or will not parse.

    Never raises: a malformed response is a verdict, not an error, because
    this runs over thousands of records inside a pipeline stage.
    """
    boxed = extract_boxed(response)
    if boxed is None:
        return 'NO_BOXED', 'no \\boxed{} in response'

    expr = parse_answer(boxed)
    if expr is None:
        return 'PARSE_FAIL', f'could not parse: {boxed[:80]}'

    # Route 1: anchor on the bank's stored answer.
    if answer_latex and str(answer_latex).lower() not in ('nan', 'none'):
        ref = parse_answer(str(answer_latex))
        if ref is not None:
            if _differs_by_constant(expr, ref):
                return 'MATCH', 'differs from bank answer by a constant'
            return 'MISMATCH', 'differs from bank answer by more than a constant'

    integrand = None
    if integrand_sympy and str(integrand_sympy).lower() not in ('nan', 'none', ''):
        try:
            integrand = sp.sympify(str(integrand_sympy))
        except Exception:
            integrand = None
    if integrand is None:
        integrand = integrand_from_id(qid)
    if integrand is None:
        return 'PARSE_FAIL', f'no integrand available for id={qid}'

    try:
        residual = sp.diff(expr, x) - integrand
    except Exception as e:
        return 'PARSE_FAIL', f'differentiation failed: {type(e).__name__}'

    if _is_zero_numerically(residual):
        return 'MATCH', 'derivative equals integrand'
    return 'MISMATCH', 'derivative differs from integrand'


# ─────────────────────────────────────────────────────────────────────────
#  Pipeline stage
# ─────────────────────────────────────────────────────────────────────────
def apply_symbolic_check(results_file: str, verbose: bool = True) -> dict:
    """
    Deterministic verdict pass over EVERY record in a Stage 1 results JSONL.

    A stage in its own right, not a pre-filter inside the LLM equivalence
    loop. It covers the whole file — passes included — because a PASS the
    check contradicts is worth surfacing, and a FAIL it can prove correct is
    a recovery the LLM stage would otherwise have paid for.

    Adds two fields per record:
        symbolic_check   MATCH | MISMATCH | PARSE_FAIL | NO_BOXED | ERROR
        symbolic_reason  human-readable justification

    Only MATCH changes `correct`. MISMATCH stays advisory: it is only as
    good as the LaTeX parse behind it, and a mis-parsed expression must never
    mark a correct answer wrong. MATCH carries no such risk — a bad parse
    could not have produced the identity.

    Idempotent: records that already carry a symbolic_check are skipped.
    Returns the verdict counts.
    """
    if not os.path.exists(results_file):
        if verbose:
            print(f"  No results file: {results_file}")
        return {}

    records = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass          # a record still being appended by a live run

    todo = [r for r in records if "symbolic_check" not in r]
    if not todo:
        if verbose:
            print(f"  All {len(records)} records already have a symbolic verdict.")
        return {}

    if verbose:
        print(f"  Checking {len(todo)} of {len(records)} records (no API calls)...")

    def _flush():
        with open(results_file, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")

    counts: dict = {}
    recovered = 0
    for n, r in enumerate(todo, 1):
        qid = r.get("question_id") or r.get("Problem_ID") or ""
        try:
            verdict, why = check_answer(r.get("response", "") or "",
                                        qid,
                                        r.get("integrand_sympy", "") or "",
                                        r.get("answer_latex", "") or "")
        except Exception as e:
            verdict, why = "ERROR", f"{type(e).__name__}: {e}"

        r["symbolic_check"] = verdict
        r["symbolic_reason"] = why
        counts[verdict] = counts.get(verdict, 0) + 1

        # A proof of correctness overrides an earlier negative verdict and
        # spares the LLM stage a billed call on this record.
        if verdict == "MATCH" and r.get("correct") is not True:
            r["correct"] = True
            r["equivalence_check"] = "SYMBOLIC_CONFIRMED"
            r["equivalence_reason"] = why
            recovered += 1

        if n % 100 == 0:
            _flush()
            if verbose:
                print(f"    ... {n}/{len(todo)}")

    _flush()

    if verbose:
        total = sum(counts.values())
        decided = counts.get("MATCH", 0) + counts.get("MISMATCH", 0)
        print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        print(f"  Decided {decided}/{total} ({100 * decided / total:.1f}%); "
              f"{recovered} recovered to correct=True with no API call.")
    return counts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run the deterministic IBP check over a results JSONL.")
    ap.add_argument("results", help="Stage 1 results JSONL")
    apply_symbolic_check(ap.parse_args().results)
