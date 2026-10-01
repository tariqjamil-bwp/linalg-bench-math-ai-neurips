#!/usr/bin/env python3
# Module:    scaffold_kappa.py
# Version:   1.0
"""
scaffold_kappa.py — Cohen's kappa for the step-error coding reliability check.

    kappa = (p_o - p_e) / (1 - p_e)

p_o is observed agreement; p_e is the agreement expected from each coder's
own marginal rates. Plain percent agreement overstates reliability when one
code dominates the sample; kappa corrects for that.

Kappa below 0.70 is a hard stop: the run exits non-zero and prints the
disagreeing rows for the coders to resolve, rather than reporting the low
score and moving on.

If both coders use a single code for everything, p_e = 1 and kappa is 0/0 —
reported as undefined, not 0.0 or 1.0, since that means the sample carries
no information about reliability.

Usage:
    python3 scaffold_kappa.py --coder1 scaffold_stepcodes_coder1.xlsx \\
                              --coder2 scaffold_stepcodes_coder2.xlsx
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import pandas as pd

from scaffold_stepcode_worksheet import read_coding_sheet

KEY = ["model", "level", "problem_id", "repeat"]
THRESHOLD = 0.70


def load(path: str) -> pd.DataFrame:
    """One reader, shared with the worksheet builder and the deliverables
    builder — see read_coding_sheet's docstring for why the header is searched
    for rather than assumed."""
    return read_coding_sheet(path)[KEY + ["step_error_code"]]


def cohens_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    p_o = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if abs(1 - p_e) < 1e-12:
        return None            # undefined, not zero — see the module docstring
    return (p_o - p_e) / (1 - p_e)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coder1", required=True)
    ap.add_argument("--coder2", required=True)
    args = ap.parse_args()

    c1, c2 = load(args.coder1), load(args.coder2)
    merged = c1.merge(c2, on=KEY, suffixes=("_1", "_2"))

    blank = merged[(merged.step_error_code_1 == "") | (merged.step_error_code_2 == "")
                   | (merged.step_error_code_1 == "NAN") | (merged.step_error_code_2 == "NAN")]
    coded = merged.drop(blank.index)

    print(f"  coder 1 rows      : {len(c1)}")
    print(f"  coder 2 rows      : {len(c2)}   (the 20% reliability sample)")
    print(f"  overlapping rows  : {len(merged)}")
    if len(blank):
        print(f"  [WARN] {len(blank)} overlapping rows are not coded by both — excluded")
    if coded.empty:
        raise SystemExit("No rows coded by both coders.")

    k = cohens_kappa(coded.step_error_code_1.tolist(), coded.step_error_code_2.tolist())
    agree = (coded.step_error_code_1 == coded.step_error_code_2).mean()

    print(f"\n  rows compared     : {len(coded)}")
    print(f"  raw agreement     : {100*agree:.1f}%")
    if k is None:
        print("  Cohen's kappa     : undefined (both coders used a single code)")
        return 1
    print(f"  Cohen's kappa     : {k:.4f}")

    print("\n  confusion (coder1 rows x coder2 columns):")
    print(pd.crosstab(coded.step_error_code_1, coded.step_error_code_2).to_string())

    if k < THRESHOLD:
        dis = coded[coded.step_error_code_1 != coded.step_error_code_2]
        print(f"\n  ‼ kappa {k:.4f} is below the required {THRESHOLD:.2f} threshold.")
        print("    Resolve these disagreements and re-code before finalizing.")
        print("    Do not report this run's step codes as they stand.")
        print(f"\n  {len(dis)} disagreeing rows:")
        print(dis.to_string(index=False))
        return 1

    print(f"\n  [PASS] kappa >= {THRESHOLD:.2f}. Pass it to the deliverables builder:")
    print(f"    python3 scaffold_build_deliverables.py ... --kappa {k:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
