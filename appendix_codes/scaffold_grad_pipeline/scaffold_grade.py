#!/usr/bin/env python3
# Module:    scaffold_grade.py
# Version:   1.0
"""
scaffold_grade.py — the locked grading rule for this experiment.

    1. Extract the content of the LAST \\boxed{}.
    2. symbolic_stop GATE: if that content is a symbolic expression
       (characteristic polynomial, radicals, free variables) AND carries no
       decimal number, then accuracy = 0 and symbolic_stop = TRUE. STOP HERE.
    3. Otherwise parse as a comma-separated decimal list and compare against
       ground truth at THREE decimal places.

Grading uses the pipeline's own 3-decimal-place tolerance
(EIG_DECIMAL_PLACES), pinned here so the ground-truth parser and the
response parser share one constant and cannot drift apart. The rule is
locked before the first API call; grading_rule_locked_at in
scaffold_run_summary.json records when.

The symbolic_stop gate runs BEFORE any numeric parse, unlike the existing
benchmark extractor, which feeds each part through sympy so a radical
answer evaluates to floats and scores correct. That's right for the
benchmark's own grading, and wrong here — symbolic_stop (deriving the right
form but never evaluating it) is itself one of the outcomes this experiment
measures.

"Symbolic" is a parse outcome, not a keyword list: each comma-separated
part is cleaned of label noise (\\lambda_1 =, \\approx, \\quad) and tried
as a plain float; a part that fails needs sympy, so it's a radical, a
power, a polynomial in lambda, or a free variable. The gate fires only when
BOTH a part fails to parse AND no decimal digit appears anywhere in the
boxed content — the integer pools' correct answers are plain integers with
no decimals, and are plainly not a symbolic stop.

LaTeX spacing/layout macros (\\;, \\quad, \\text{...}, aligned
environments) carry no mathematical content, but the pipeline's cleaner
leaves them in place, which can split a value like "-7,\\;0" into
unparseable fragments and mis-classify it as symbolic. _normalise() strips
this layout before parsing, without rescuing a genuinely symbolic answer.

Usage:
    python3 scaffold_grade.py --selftest     # fixed test cases, no API, free
    python3 scaffold_grade.py --replicate    # re-grade the published Claude
                                             # irr-sym responses; must be 0/30
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, asdict

import subcat_config as _cfg
from subcat_config import compute_eigenvalue_from_answer, _clean_latex_for_eigenvalue

# The tolerance is the pipeline's existing EIG_DECIMAL_PLACES = 3, pinned
# here in one place so the ground-truth parser and the response parser
# cannot drift apart — compute_eigenvalue_from_answer reads this same
# module global.
GRADE_DECIMAL_PLACES = 3
_cfg.EIG_DECIMAL_PLACES = GRADE_DECIMAL_PLACES
EIG_DECIMAL_PLACES = GRADE_DECIMAL_PLACES

# Grading rule frozen before the first API call, recorded in
# scaffold_run_summary.json and scaffold_abort_passed.json; require_pass()
# refuses to build if the two disagree, guaranteeing every cell in a run
# was graded under one fixed rule.
GRADING_RULE_LOCKED_AT = "2026-09-16T10:45:00"

_DECIMAL = re.compile(r"\d\.\d")

# LaTeX spacing and layout macros. These carry NO mathematical content — they
# are how a model writes a space — but the pipeline's cleaner does not remove
# them, so "-7,\;0,\;1,\;5,\;7" split into parts like "\;0", which fail to
# parse as a float and were therefore classified symbolic. symbolic_stop is
# H4's outcome measure, so a false symbolic stop is a corrupted result rather
# than a cosmetic problem.
_SPACING = re.compile(r"\\[;,:!]|\\q?quad\b|\\ |~|&")
# \\ and the aligned/array environments are list SEPARATORS, not spacing:
# Qwen returns its answer as \begin{aligned}&-10.083,\\&-2.363,...\end{aligned}.
_ENVIRONMENT = re.compile(r"\\(?:begin|end)\{(?:aligned|array|gathered|cases|matrix)\*?\}"
                          r"(?:\{[^{}]*\})?")
_ANNOTATION = re.compile(r"\\(?:text|textrm|mathrm|textbf)\s*\{[^{}]*\}")
# Spacing commands that take a length argument — a closed class in LaTeX, so
# enumerating these is safe in a way that enumerating "all spacing" is not.
_SPACING_ARG = re.compile(r"\\(?:h|v|m)?(?:space|kern|phantom)\*?\s*\{[^{}]*\}")


# ─────────────────────────────────────────────────────────────────────────
#  1. Boxed extraction — last box, brace-balanced
# ─────────────────────────────────────────────────────────────────────────
def extract_boxed(response: str) -> str | None:
    """Content of the LAST \\boxed{...}, with nesting handled properly.

    A regex cannot do this: eigenvalue answers routinely contain \\frac{a}{b}
    and \\sqrt{n}, so brace counting is required or the content is truncated at
    the first inner '}'. The LAST box wins because models often box an
    intermediate result (the characteristic polynomial, typically) before the
    final answer.
    """
    if not response:
        return None
    starts = [m.end() for m in re.finditer(r"\\boxed\s*\{", response)]
    for start in reversed(starts):
        depth, i = 1, start
        while i < len(response) and depth:
            c = response[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return response[start:i - 1].strip()
    return None


def count_boxed(response: str) -> int:
    """How many \\boxed{} the response opens.

    Recorded on every result record, matching the main pipeline's field of the
    same name. More than one is a hedging signal worth tracking: a model that
    boxes the characteristic polynomial and then boxes an answer has not given
    one answer, and extract_boxed() silently keeps only the last. The count is
    what makes that visible without re-reading the text.

    Counting only, never grading — which is why it is safe to call from the
    parse step, where no verdict is formed.
    """
    if not response:
        return 0
    # The same pattern extract_boxed() splits on, so the count and the
    # extraction can never disagree about what a box is (r"\\boxed\{" alone
    # would miss "\boxed {").
    return len(re.findall(r"\\boxed\s*\{", response))


# ─────────────────────────────────────────────────────────────────────────
#  2. The symbolic_stop gate
# ─────────────────────────────────────────────────────────────────────────
def _normalise(boxed: str) -> str:
    """Remove layout that is not content, before anything is split or parsed.

    Order matters: environments and \\\\ become commas so a newline-separated
    answer becomes a list, then spacing macros and \\text{...} annotations go.
    Nothing here can turn a symbolic answer into a numeric one — a radical is
    still a radical after its spaces are removed.
    """
    text = _ENVIRONMENT.sub(" ", boxed)
    text = text.replace("\\\\", ",")          # line break in a list -> separator
    # \{ \} are SET delimiters, not grouping braces: models answer
    # "\{-8, -3, -2, 1, 2\}". Left in place they raise the brace depth, so the
    # comma split never fires and the whole set reads as one unparseable part,
    # i.e. a false symbolic stop.
    text = text.replace(r"\{", " ").replace(r"\}", " ")
    text = _ANNOTATION.sub(" ", text)         # "(multiplicity 2)" and friends
    text = _SPACING_ARG.sub(" ", text)        # \hspace{2pt}, \kern{3pt}, ...
    text = _SPACING.sub(" ", text)
    # A tuple/list wrapper around the WHOLE answer — "(5, 5, -5, -5, -5)" —
    # is notation, not a term. Only stripped when it wraps everything, so an
    # expression that merely starts with a bracket is untouched.
    text = text.strip()
    while len(text) > 1 and text[0] in "([" and text[-1] in ")]":
        text = text[1:-1].strip()
    return text


def _split_parts(boxed: str) -> list[str]:
    """Split on commas/semicolons that are not inside braces."""
    parts, depth, cur = [], 0, []
    for ch in _normalise(boxed):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        if ch in ",;" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _clean(part: str) -> str:
    """Pipeline cleanup, plus \\frac{a}{b} -> (a)/(b).

    The pipeline's cleaner handles \\lambda labels, \\approx and \\sqrt but not
    \\frac, and a boxed \\frac{-8}{1} is an exact rational, not a symbolic
    stop."""
    # Strip a leading "name = " label. The pipeline's cleaner handles
    # "\lambda_1 =" but not the braced "\lambda_{1} =", which it leaves as
    # "{1}=-8" — unparseable, and therefore scored symbolic. Requiring a bare
    # name with an optional numeric subscript keeps this from eating a real
    # equation: "p(\lambda) = ..." has parentheses and does not match.
    text = _LABEL.sub("", part.strip(), count=1)
    text = _clean_latex_for_eigenvalue(text).strip()
    # A sentence-ending period after the last value ("..., -7.960000.") is
    # punctuation, not part of the number.
    while text.endswith((".", ",")):
        text = text[:-1].strip()
    for _ in range(3):      # allow a little nesting
        new = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", text)
        if new == text:
            break
        text = new

    # GENERIC SAFETY NET: correctness must not depend on the macro lists in
    # _SPACING / _ANNOTATION being complete. If a part STILL will not parse,
    # strip every backslash command from it and check whether what remains is
    # nothing but a signed number — a number wearing an unknown macro, not
    # mathematics.
    #
    # The test is deliberately narrow — the residue must match a bare decimal
    # end to end. \sqrt{22} leaves "{22}" (braces, no match), \pi leaves ""
    # (no match), \lambda^2 leaves "^2" (no match). So this can never launder a
    # genuinely symbolic term into a number; it only removes decoration.
    # The guard below is why this is safe. Stripping macros blindly would turn
    # "2\pi" into "2" — laundering a symbolic term into a number, the one error
    # that must never happen. So the fallback is refused outright if the part
    # carries any macro with mathematical meaning. Erring here is conservative:
    # a refusal means the part stays symbolic, the honest reading of an
    # answer nobody can parse.
    if not _looks_numeric(text) and not _MATH_MACRO.search(text):
        residue = re.sub(r"\\[a-zA-Z]+|\\[^a-zA-Z]", " ", text).strip()
        if _BARE_NUMBER.fullmatch(residue):
            return residue
    return text


_BARE_NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?")

# "\lambda_{1} =", "\lambda_3 =", "x_1 =", "\mu =" — a name, an optional
# numeric subscript, then "=". Anchored, and no parentheses allowed, so it
# cannot swallow "p(\lambda) = ...".
_LABEL = re.compile(r"^\s*\\?[a-zA-Z]+\s*(?:_\s*(?:\{\s*\d+\s*\}|\d+))?\s*=\s*")

# Macros that carry mathematical content. Their presence blocks the generic
# fallback. The list only has to be complete enough to be SAFE, not complete
# enough to be CORRECT: anything missed here still has to survive float() and
# sympy's is_Rational, and anything listed here merely stays symbolic.
_MATH_MACRO = re.compile(
    r"\\(?:pi|e(?![a-zA-Z])|exp|ln|log|sqrt|frac|d?frac|cdot|times|div|pm|mp|"
    r"infty|imath|jmath|alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|"
    r"theta|iota|kappa|lambda|mu|nu|xi|rho|sigma|tau|upsilon|phi|varphi|chi|"
    r"psi|omega|sin|cos|tan|sec|csc|cot|operatorname|left|right)")


def _looks_numeric(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _is_exact_rational(part: str) -> bool:
    """TRUE for an integer, a decimal, or a fraction of them — the forms that
    are NOT symbolic.

    Defined as a parse outcome, not a keyword list. A part is symbolic when it
    is anything else: a radical (sqrt(22) is irrational, so not Rational), a
    power of lambda, a free variable, a stray word. That catches forms nobody
    enumerated in advance."""
    cleaned = _clean(part)
    try:
        return float(cleaned) == float(cleaned)     # plain int/decimal
    except ValueError:
        pass
    try:
        from sympy import sympify
        expr = sympify(cleaned)
        return bool(expr.free_symbols == set() and expr.is_Rational)
    except Exception:
        return False


# A coefficient multiplying the variable — "5\lambda", "3 \lambda^2". The
# PIPELINE's cleaner strips \lambda as a label, which is right for
# "\lambda_3 = 5" and wrong here: it would leave a bare "5". Caught on the raw
# part, before any cleaning, because by then the evidence is gone. A label is
# always followed by "=", so this pattern cannot match one.
_COEFF_TIMES_VAR = re.compile(r"\d\s*\\(?:lambda|mu|alpha|beta|theta|pi|phi)(?!\s*_?\{?\d*\}?\s*=)")


def is_symbolic_stop(boxed: str) -> bool:
    """TRUE iff the boxed content is symbolic AND holds no decimal number."""
    if boxed is None:
        return False
    if _DECIMAL.search(boxed):
        return False
    parts = _split_parts(boxed)
    if not parts:
        return False
    return any(_COEFF_TIMES_VAR.search(p) or not _is_exact_rational(p) for p in parts)


# ─────────────────────────────────────────────────────────────────────────
#  3. Numeric comparison — the pipeline's tolerance, unchanged
# ─────────────────────────────────────────────────────────────────────────
def _parse_decimal_list(boxed: str) -> list[float] | None:
    values = []
    for part in _split_parts(boxed):
        cleaned = _clean(part)
        try:
            values.append(float(cleaned))
            continue
        except ValueError:
            pass
        # Reached only when the gate let the part through, i.e. the content
        # does carry a decimal somewhere. Evaluate the rest the same way the
        # pipeline does, so a mixed answer is still graded on its numbers.
        try:
            from sympy import N, sympify
            values.append(float(N(sympify(cleaned))))
        except Exception:
            return None
    if not values:
        return None
    if EIG_DECIMAL_PLACES is not None:
        values = [round(v, EIG_DECIMAL_PLACES) for v in values]
    return sorted(set(values))


@dataclass
class Grade:
    accuracy: int            # 1 = CORRECT, 0 = FAIL
    symbolic_stop: bool
    boxed_content: str | None
    reason: str              # CORRECT / WRONG_VALUES / SYMBOLIC_STOP /
                             # NO_BOX / UNPARSEABLE / NO_RESPONSE


def grade(response: str | None, answer_latex: str) -> Grade:
    if not response:
        return Grade(0, False, None, "NO_RESPONSE")

    boxed = extract_boxed(response)
    if boxed is None:
        return Grade(0, False, None, "NO_BOX")

    # Gate first — before any numeric parse. This is the whole point.
    if is_symbolic_stop(boxed):
        return Grade(0, True, boxed, "SYMBOLIC_STOP")

    got = _parse_decimal_list(boxed)
    if got is None:
        return Grade(0, False, boxed, "UNPARSEABLE")

    try:
        want = compute_eigenvalue_from_answer(answer_latex)
    except ValueError as e:
        raise SystemExit(f"Unparseable ground truth {answer_latex!r}: {e}")

    ok = got == want
    return Grade(int(ok), False, boxed, "CORRECT" if ok else "WRONG_VALUES")


# ─────────────────────────────────────────────────────────────────────────
#  Free verification
# ─────────────────────────────────────────────────────────────────────────
_GT_IRR = r"\lambda = -6.7782, -5.3145, 1.4187, 7.7088, 8.9652"
_GT_INT = r"\lambda = -8, 4, 8, 12, 16"

_CASES = [
    # (response, ground truth, expect accuracy, expect symbolic_stop, label)
    (r"so \boxed{-6.7782, -5.3145, 1.4187, 7.7088, 8.9652}", _GT_IRR, 1, False,
     "decimal list, correct"),
    (r"\boxed{-6.778, -5.314, 1.419, 7.709, 8.965}", _GT_IRR, 1, False,
     "3dp answer rounds to the 3dp ground truth -> CORRECT"),
    (r"\boxed{-6.77815, -5.31452, 1.41871, 7.70883, 8.96518}", _GT_IRR, 0, False,
     "5dp answer diverges in the 3rd place (-5.31452 -> -5.315 vs -5.314) -> FAIL"),
    (r"\boxed{-6.7783, -5.3145, 1.4187, 7.7088, 8.9652}", _GT_IRR, 1, False,
     "wrong in the 4th place only -> CORRECT (the 3dp tolerance absorbs it)"),
    (r"\boxed{-10.2460, -5.3145, 1.4187, 7.7088, 8.9652}",
     r"\lambda = -10.246, -5.3145, 1.4187, 7.7088, 8.9652", 1, False,
     "ground truth with a stripped trailing zero still matches a 3dp answer"),
    (r"\boxed{-6.7782, -5.3145, 1.4187, 7.7088, 9.0000}", _GT_IRR, 0, False,
     "one value wrong -> FAIL, not a symbolic stop"),
    (r"\boxed{2+\sqrt{22}, 2-\sqrt{22}, 0, 1, 3}", _GT_IRR, 0, True,
     "radicals, no decimals -> symbolic_stop (pipeline alone would float these)"),
    (r"\boxed{\lambda^5 - 3\lambda^4 + 2\lambda^3 - 7\lambda^2 + \lambda - 4}",
     _GT_IRR, 0, True, "characteristic polynomial -> symbolic_stop"),
    (_GT_INT.join([r"\boxed{", "}"]).replace(r"\lambda = ", ""), _GT_INT, 1, False,
     "integer answer has no decimals but is NOT a symbolic stop"),
    (r"\boxed{p(\lambda)=\lambda^5-1} and finally \boxed{-8, 4, 8, 12, 16}",
     _GT_INT, 1, False, "last box wins over a boxed intermediate"),
    (r"\boxed{\frac{-8}{1}, 4, 8, 12, 16}", _GT_INT, 1, False,
     "nested braces survive extraction"),
    # LaTeX layout is not content — see _SPACING / _normalise.
    (r"\boxed{-8,\;4,\;8,\;12,\;16}", _GT_INT, 1, False,
     r"\; spacing -> CORRECT, not a symbolic stop"),
    (r"\boxed{\lambda_1 = -8, \quad \lambda_2 = 4, \quad \lambda_3 = 8, "
     r"\quad \lambda_4 = 12, \quad \lambda_5 = 16}", _GT_INT, 1, False,
     r"\quad separators with lambda labels"),
    (r"\boxed{\begin{aligned}&-6.7782, \\&-5.3145, \\&1.4187, \\&7.7088, "
     r"\\&8.9652\end{aligned}}", _GT_IRR, 1, False,
     r"aligned environment, \\ as separator (Qwen's real format)"),
    (r"\boxed{\lambda_1 \approx -6.7782,\ \lambda_2 \approx -5.3145,\ "
     r"\lambda_3 \approx 1.4187,\ \lambda_4 \approx 7.7088,\ "
     r"\lambda_5 \approx 8.9652}", _GT_IRR, 1, False,
     r"\approx with \  spacing (DeepSeek's real format)"),
    (r"\boxed{-8,\; 4,\; 8,\; 12,\; 16.}", _GT_INT, 1, False,
     "trailing sentence period is punctuation, not a digit"),
    (r"\boxed{2+\sqrt{22},\; 2-\sqrt{22},\; 0,\; 1,\; 3}", _GT_IRR, 0, True,
     r"radicals WITH \; are STILL a symbolic stop — spacing must not rescue them"),
    (r"\boxed{\lambda_1 = 1, \quad \lambda_2 = 10, \quad \lambda_3 = -10, "
     r"\quad \lambda_4 = 1 + 3i\sqrt{3}, \quad \lambda_5 = 1 - 3i\sqrt{3}}",
     _GT_IRR, 0, True,
     "complex radicals among integers -> symbolic stop (real Claude response)"),
    (r"\boxed{\{-8, 4, 8, 12, 16\}}", _GT_INT, 1, False,
     r"set notation \{...\} — delimiters, not grouping braces"),
    (r"\boxed{(-8, 4, 8, 12, 16)}", _GT_INT, 1, False,
     "tuple notation around the whole answer"),
    (r"\boxed{(1+\sqrt{5})/2, (1-\sqrt{5})/2, 0, 1, 2}", _GT_INT, 0, True,
     "an expression that merely STARTS with a bracket is still symbolic"),
    (r"\boxed{-8 \text{ (multiplicity 2)}, 4, 8, 12}", _GT_INT, 0, False,
     r"\text annotation stripped, 4 values -> WRONG_VALUES, not symbolic"),
    (r"the answer is somewhere", _GT_INT, 0, False, "no box"),
    (None, _GT_INT, 0, False, "no response"),
]


def selftest() -> int:
    fails = 0
    print("Grading rule self-test (no API):")
    for resp, gt, exp_acc, exp_sym, label in _CASES:
        g = grade(resp, gt)
        ok = (g.accuracy == exp_acc and g.symbolic_stop == exp_sym)
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            print(f"        expected acc={exp_acc} sym={exp_sym}, got {asdict(g)}")
    return fails


# The published Claude responses for the 30 irr-sym problems, extracted from
# the released 5x5 model outputs and kept beside the code so the replication
# runs from this directory alone.
_PUBLISHED = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "source", "scaffold_published_baseline.csv")


def replicate() -> int:
    """Re-grade the published Claude responses on the 30 irr-sym problems.

    This must come out 0/30. That number is the abort check's baseline, and it
    is NOT the 61.8% in accuracy_5x5.csv — that figure aggregates all 220 5x5
    problems across every subcategory. Grading the wrong 30 rows would make the
    gate trip on every run.
    """
    import pandas as pd

    pools = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source", "scaffold_pools.csv")
    if not os.path.exists(pools):
        raise SystemExit("Run scaffold_pools.py first.")
    ids = set(pd.read_csv(pools).query("pool_id == 'irr-sym'").problem_id)

    out = pd.read_csv(os.path.abspath(_PUBLISHED))
    sub = out[(out.Model == "Claude-4.5-Sonnet") & out.Problem_ID.isin(ids)
              & (out.Subcat == "eigenvalue")]
    if len(sub) != 30:
        print(f"  [WARN] expected 30 published Claude irr-sym rows, found {len(sub)}")

    n_ok = n_sym = 0
    for r in sub.itertuples():
        g = grade(getattr(r, "_6"), r.answer_latex)   # 'Model Response'
        n_ok += g.accuracy
        n_sym += g.symbolic_stop

    print(f"\nPublished replication (Claude-4.5-Sonnet, irr-sym, L0):")
    print(f"  graded rows     : {len(sub)}")
    print(f"  CORRECT         : {n_ok}/{len(sub)}  = {100*n_ok/max(len(sub),1):.1f}%")
    print(f"  symbolic_stop   : {n_sym}")
    ok = (n_ok == 0)
    print(f"  [{'PASS' if ok else 'FAIL'}] baseline is 0/30 as expected")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--replicate", action="store_true")
    args = ap.parse_args()

    if not (args.selftest or args.replicate):
        ap.error("choose --selftest and/or --replicate")

    rc = 0
    if args.selftest:
        rc += selftest()
    if args.replicate:
        rc += replicate()
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
